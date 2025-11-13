FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV MODEL_NAME=deepseek-ai/DeepSeek-OCR
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 9000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
