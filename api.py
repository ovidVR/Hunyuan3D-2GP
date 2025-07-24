import asyncio
import base64
import logging
import os
import shutil
import tempfile
import uuid
from io import BytesIO

import torch
import trimesh
import typer
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from mmgp import offload
from PIL import Image
from pydantic import BaseModel, Field

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVE_DIR = "gradio_cache"
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


class MeshGenerationParams(BaseModel):
    text: str = Field(None, description="Text prompt for text-to-image generation (if enabled)")
    image: str | None = Field(
        None,
        description="Base64 encoded image for background removal and mesh generation",
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")
    octree_resolution: int = Field(256, description="Octree resolution for mesh generation")
    num_inference_steps: int = Field(
        5, description="Number of inference steps for mesh generation"
    )
    guidance_scale: float = Field(5.0, description="Guidance scale for mesh generation")
    mc_algo: str = Field("dmc", description="Algorithm for mesh generation")
    num_chunks: int = Field(
        8000, description="Number of chunks for mesh generation (if applicable)"
    )
    target_face_num: int = Field(10000, description="Target number of faces for mesh generation")
    texture: bool = Field(
        False, description="Whether to generate texture for the mesh (if enabled)"
    )
    save_type: str = Field(
        "glb",
        description="File format for saving the generated mesh (e.g., 'glb', 'obj', 'ply')",
    )


def load_image_from_base64(image: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image)))


def replace_property_getter(instance, property_name, new_getter):
    # Get the original class and property
    original_class = type(instance)
    original_property = getattr(original_class, property_name)

    # Create a custom subclass for this instance
    custom_class = type(f"Custom{original_class.__name__}", (original_class,), {})

    # Create a new property with the new getter but same setter
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)

    # Change the instance's class
    instance.__class__ = custom_class

    return instance


class ModelWorker:
    def __init__(
        self,
        model_path="tencent/Hunyuan3D-2mini",
        tex_model_path="tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        text_model_path: str = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
        device: str = "cuda",
        enable_tex: bool = False,
        enable_text: bool = False,
        enable_flashvdm: bool = False,
        mc_algo: str = "dmc",
        low_vram_mode: bool = False,
        profile: int = 3,
        verbose: int = 1,
    ):
        self.device = device
        self.enable_tex = enable_tex
        self.enable_text = enable_text
        self.low_vram_mode = low_vram_mode

        self.rembg = BackgroundRemover()
        if enable_tex:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            self.texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)

        if enable_text:
            from hy3dgen.text2image import HunyuanDiTPipeline

            logger.info(f"Loading text-to-image model {text_model_path}...")
            self.t2i_worker = HunyuanDiTPipeline(text_model_path)

        self.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        if enable_flashvdm:
            mc_algo = "mc" if device in ["cpu", "mps"] else mc_algo
            self.i23d_worker.enable_flashvdm(mc_algo=mc_algo)
        # if compile:
        #     i23d_worker.compile()

        profile = int(profile)
        kwargs = {}
        replace_property_getter(self.i23d_worker, "_execution_device", lambda self: "cuda")
        pipe = offload.extract_models("i23d_worker", self.i23d_worker)
        if enable_tex:
            pipe.update(offload.extract_models("texgen_worker", self.texgen_worker))
            self.texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
        if enable_text:
            pipe.update(offload.extract_models("t2i_worker", self.t2i_worker))

        if profile < 5:
            kwargs["pinnedMemory"] = "i23d_worker/model"
        if profile != 1 and profile != 3:
            kwargs["budgets"] = {"*": 2200}
        offload.default_verboseLevel = int(verbose)
        offload.profile(pipe, profile_no=profile, verboseLevel=int(verbose), **kwargs)

    @torch.inference_mode()
    def generate(self, uid: str, params: MeshGenerationParams) -> str:

        if params.seed is None:
            params.seed = int(uuid.uuid4().int % (2**32 - 1))
            logger.info(f"Using random seed: {params.seed}")

        if params.image:
            image = load_image_from_base64(params.image)
        elif self.enable_text and params.text:
            text = params.text
            logger.info(f"Generating image from text: {text}")
            image = self.t2i_worker(text, seed=params.seed)
        else:
            raise ValueError("No input image or text provided")

        image = self.rembg(image)
        params.image = image

        seed_generator = torch.Generator(self.device).manual_seed(params.seed)

        mesh = self.i23d_worker(
            image=image,
            num_inference_steps=params.num_inference_steps,
            generator=seed_generator,
            octree_resolution=params.octree_resolution,
            guidance_scale=params.guidance_scale,
            mc_algo=params.mc_algo,
            num_chunks=params.num_chunks,
        )[0]

        if self.enable_tex and params.texture:
            from hy3dgen.shapegen import (
                DegenerateFaceRemover,
                FaceReducer,
                FloaterRemover,
            )

            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.target_face_num)
            mesh = self.texgen_worker(mesh, image)

        type = params.save_type
        save_path = os.path.join(SAVE_DIR, f"{uid}.{type}")
        with tempfile.NamedTemporaryFile(suffix=f".{type}", delete=True) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            mesh.export(save_path)

        if self.low_vram_mode:
            torch.cuda.empty_cache()
        return save_path

    @torch.inference_mode()
    def generate_image(self, uid: str, params: dict) -> str:
        if self.enable_text and "text" in params:
            text = params["text"]
            logger.info(f"Generating image from text: {text}")
            image = self.t2i_worker(text)
            image = self.rembg(image)
            # Save the image to a temporary file
            type = params.get("type", "png")
            save_path = os.path.join(SAVE_DIR, f"{uid}.{type}")
            image.save(save_path)
            return save_path

        return None


@app.post("/generate")
async def generate(params: MeshGenerationParams):
    logger.info("Processing /generate request...")
    uid = str(uuid.uuid4())
    try:
        async with model_semaphore:
            file_path = worker.generate(uid, params)

        return FileResponse(file_path)
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return JSONResponse({"text": "Invalid input", "error_code": 1}, status_code=400)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse({"text": "Server error", "error_code": 1}, status_code=500)


@app.post("/generate_image")
async def generate_image(request: Request):
    logger.info("Processing /generate request...")
    params = await request.json()
    uid = str(uuid.uuid4())
    try:
        async with model_semaphore:
            file_path = worker.generate_image(uid, params)
        if not file_path:
            return JSONResponse(
                {
                    "text": "No valid input provided (must include the `text` param, or the server was not started with `--enable_text`)",
                    "error_code": 1,
                },
                status_code=400,
            )
        return FileResponse(file_path)
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return JSONResponse({"text": "Invalid input", "error_code": 1}, status_code=400)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse({"text": "Server error", "error_code": 1}, status_code=500)


def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    model_path: str = "tencent/Hunyuan3D-2mini",
    tex_model_path: str = "tencent/Hunyuan3D-2",
    text_model_path: str = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
    subfolder: str = "hunyuan3d-dit-v2-mini-turbo",
    device: str = "cuda",
    enable_tex: bool = False,
    enable_text: bool = False,
    enable_flashvdm: bool = False,
    mc_algo: str = "dmc",
    low_vram_mode: bool = False,
    profile: int = 3,
    verbose: int = 1,
    limit_model_concurrency: int = 5,
):
    global worker, model_semaphore
    model_semaphore = asyncio.Semaphore(limit_model_concurrency)
    worker = ModelWorker(
        model_path=model_path,
        tex_model_path=tex_model_path,
        text_model_path=text_model_path,
        subfolder=subfolder,
        device=device,
        enable_tex=enable_tex,
        enable_text=enable_text,
        enable_flashvdm=enable_flashvdm,
        mc_algo=mc_algo,
        low_vram_mode=low_vram_mode,
        profile=profile,
        verbose=verbose,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    typer.run(main)
