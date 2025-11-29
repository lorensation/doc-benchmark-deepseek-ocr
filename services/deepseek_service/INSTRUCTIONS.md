You are assisting with the DeepSeek-OCR microservice for "doc-benchmark-deepseek-ocr".

Purpose:
- Provide a clean HTTP wrapper around the DeepSeek-OCR model.
- Uses HuggingFace Transformers with GPU acceleration.
- Model: deepseek-ai/DeepSeek-OCR.
- Load tokenizer + model at startup with CUDA and Flash Attention 2.
- Expose one endpoint: POST /ocr, which accepts an image file and returns extracted text.
- Use PIL for image handling.
- Use the model's .infer() method (trust_remote_code=True).
- Handle temp directories cleanly.
- Return JSON with field: { "text": "...", "confidence": 1.0 }

Requirements:
- Dockerfile must:
    • use nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 base image
    • install Python 3.12 and CUDA dependencies
    • install flash-attn for GPU acceleration
    • expose port 9000
- Code must:
    • detect and use CUDA when available, fallback to CPU
    • load model with torch.float16 on GPU for speed
    • use device_map="auto" for multi-GPU support
    • convert uploaded file into RGB image
    • call model.infer with correct parameters
    • include GPU info in health/ready endpoints

GPU Configuration:
- Requires NVIDIA GPU with 12+ GB VRAM
- Docker must have NVIDIA Container Toolkit installed
- See GPU_SETUP.md for detailed setup instructions

When I request changes, update ONLY the service in services/deepseek_service/.

# Build the DeepSeek service with GPU support
docker-compose build deepseek

# Start the service
docker-compose up -d deepseek

# Monitor GPU usage during model loading
docker logs -f docbench-deepseek
