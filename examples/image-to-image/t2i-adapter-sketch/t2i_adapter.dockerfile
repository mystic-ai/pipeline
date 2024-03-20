FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# python build dependencies
RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN apt-get update

# Install python
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT=/pyenv
ENV PATH="/pyenv/shims:/pyenv/bin:$PATH"
RUN pyenv install 3.10.10
RUN pyenv global 3.10.10

# Install other python dependencies
RUN pip install setuptools wheel
RUN pip install pipeline-ai accelerate controlnet_aux diffusers
RUN pip install Pillow safetensors timm torch torchvision transformers opencv-python-headless
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install fastapi==0.105.0 uvicorn==0.25.0 python-multipart==0.0.6 loguru==0.7.2

# Copy in files
COPY ./ ./

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PYTHONUNBUFFERED=1
ENV PIPELINE_PATH=new_pipeline:my_new_pipeline
ENV PIPELINE_NAME=sketch-2-img
ENV PIPELINE_IMAGE=sketch-2-img

CMD ["uvicorn", "pipeline.container.startup:create_app", "--host", "0.0.0.0", "--port", "14300"]