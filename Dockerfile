FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ---------- 系统依赖 ----------
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- 设置 python3 默认指向 3.11 ----------
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

# ---------- pip ----------
RUN python3 -m pip install --upgrade pip

# ---------- PyTorch (CUDA 12.1) ----------
RUN pip install torch==2.5.1+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# ---------- HuggingFace 稳定组合 ----------
RUN pip install \
    transformers==4.41.2 \
    peft==0.10.0 \
    accelerate
