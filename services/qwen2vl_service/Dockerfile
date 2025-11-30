FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential \
    git wget curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV HF_HOME=/app/.cache/huggingface
ENV MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct

EXPOSE 9004

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9004"]
