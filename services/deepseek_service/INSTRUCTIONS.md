You are assisting with the DeepSeek-OCR microservice for “doc-benchmark-deepseek-ocr”.

Purpose:
- Provide a clean HTTP wrapper around the DeepSeek-OCR model.
- Uses HuggingFace Transformers.
- MUST run CPU-only (HF Spaces compatibility).
- Model: deepseek-ai/DeepSeek-OCR.
- Load tokenizer + model at startup.
- Expose one endpoint: POST /ocr, which accepts an image file and returns extracted text.
- Use PIL for image handling.
- Use the model’s .infer() method (trust_remote_code=True).
- Handle temp directories cleanly.
- Return JSON with field: { "text": "..." }

Requirements:
- Dockerfile must:
    • use python:3.12-slim
    • install HF dependencies
    • stay under reasonable size (no unnecessary libs)
- Code must:
    • convert uploaded file into RGB image
    • call model.infer with correct parameters
    • work gracefully with CPU only

When I request changes, update ONLY the service in services/deepseek_service/.
