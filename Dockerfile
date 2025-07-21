FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include
ENV LIBRARY_PATH=/usr/local/cuda/lib64

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    python3.11-venv \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install PyTorch with CUDA support
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Clone the repository
WORKDIR /workspaces

# To avoid having all files modified by the container
RUN git config --global core.autocrlf false
RUN git config --global core.filemode false


COPY requirements.txt /workspaces/Hunyuan3D-2GP/requirements.txt

# # Install requirements
WORKDIR /workspaces/Hunyuan3D-2GP

# # Sentencepiece is required for text to 3D generation
RUN pip install sentencepiece

RUN pip install -r requirements.txt

# Build custom rasterizer with explicit CUDA architecture settings
COPY hy3dgen /workspaces/Hunyuan3D-2GP/hy3dgen
WORKDIR /workspaces/Hunyuan3D-2GP/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py build_ext --inplace && python3 setup.py install

# Build differentiable renderer
WORKDIR /workspaces/Hunyuan3D-2GP/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py build_ext --inplace && python3 setup.py install

# # Install Gradio
# WORKDIR /workspaces/Hunyuan3D-2GP
RUN python3 -m pip install --no-cache-dir gradio


# EXPOSE 8080

# # Run Gradio app on custom port
# CMD ["bash", "-c", "python3 gradio_app.py --enable_t23d --host 0.0.0.0 --port 8080 || tail -f /dev/null"]

ENTRYPOINT ["/bin/bash"]