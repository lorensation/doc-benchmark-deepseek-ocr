You are assisting with the FastAPI backend of the project “doc-benchmark-deepseek-ocr”.  
This service exposes the main API used by the UI and orchestrator.

Requirements:
- Endpoints:
    • POST /upload_and_benchmark: upload an image/PDF-page → send to DeepSeek + Tesseract services → collect outputs → compute simple metrics → save JSON results under data/results/.
    • GET /runs: list benchmark runs.
    • GET /runs/{run_id}: retrieve one benchmark result.
- All requests to OCR models are made via internal Docker network URLs:
    • DeepSeek: http://deepseek:9000/ocr
    • Tesseract: http://tesseract:9001/ocr
- FastAPI must:
    • accept file uploads
    • save files under data/uploads/
    • call services using requests
    • generate structured benchmark result objects
    • be stateless (results stored in JSON only)
- No GPU. CPU-only.
- Ensure clean, well-typed Pydantic models.
- Requirements: fastapi, uvicorn, requests, python-multipart, orjson.
- Dockerfile must remain minimal and Python 3.12-slim compatible.

Whenever I ask for modifications to API functionality,  
update only what lives in doc-benchmark-deepseek-ocr/services/api/.
