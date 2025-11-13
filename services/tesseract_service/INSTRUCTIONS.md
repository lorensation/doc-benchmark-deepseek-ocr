You are assisting with the Tesseract baseline OCR service for “doc-benchmark-deepseek-ocr”.

Purpose:
- Provide a simple comparison baseline for the benchmark.
- Wrap pytesseract inside a FastAPI HTTP service.
- Single endpoint: POST /ocr.
- Accept uploaded images, convert via PIL, call pytesseract.image_to_string, return text.

Requirements:
- CPU-only (Tesseract is CPU-only naturally).
- Dockerfile must:
    • install tesseract-ocr and libtesseract-dev
    • remain based on python:3.12-slim
- Output format: { "text": "..." }
- Must be simple, robust, deterministic.
- No advanced preprocessing; keep it minimal and baseline-level.

When requested, modify ONLY code under services/tesseract_service/.
