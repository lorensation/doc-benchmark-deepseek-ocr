FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV UPLOAD_DIR=/app/data/uploads
ENV RESULTS_DIR=/app/data/results
ENV DATASETS_DIR=/app/data/datasets
ENV WORKER_PORT=9100

EXPOSE 9100

CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "9100"]
