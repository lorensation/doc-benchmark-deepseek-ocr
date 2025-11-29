# NVIDIA CUDA base image with CUDA 12.1 + cuDNN8
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# -------------------------
# Install Python 3.11 and system deps
# -------------------------
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential \
    git wget curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip + build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

COPY requirements.txt .

# -------------------------
# Install PyTorch 2.4.0 + CUDA 12.1 wheels (GPU, Python 3.11)
# -------------------------
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# -------------------------
# Install Flash-Attention 2 (pre-built wheel for CUDA 12.1 + PyTorch 2.4 + Python 3.11)
# Use cxx11abiFALSE to match PyTorch ABI compatibility
# -------------------------
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    && rm flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# -------------------------
# Install DeepSeek & API deps (from requirements.txt, skip torch/flash-attn)
# -------------------------
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# HuggingFace cache
ENV MODEL_NAME=deepseek-ai/DeepSeek-OCR
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 9000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
