You are assisting with the benchmark worker for “doc-benchmark-deepseek-ocr”.

Purpose:
- Run OCR benchmarks asynchronously OR as a CLI-like helper depending on architecture.
- For now, keep it minimal: same logic as the FastAPI route, but structured in a worker file.
- Future responsibilities:
    • Batch processing
    • Scheduled evaluations
    • Dataset-wide benchmarks
    • Experimental metric calculations
    • Integration with async queues (Redis/Celery/Prefect)

Current Requirements:
- Provide a Python file worker.py with functions:
    • run_benchmark(filepath) → returns a structured Python dict
    • call_deepseek()
    • call_tesseract()
    • compute_metrics()
- Should save results in data/results/ with JSON.
- No GPU. CPU-only.
- Simple Dockerfile based on python:3.12-slim.

When I ask for changes, modify ONLY files inside services/benchmark_worker/.
