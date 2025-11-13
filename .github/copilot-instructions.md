You are assisting me in developing the project “doc-benchmark-deepseek-ocr”, an end-to-end OCR benchmarking environment comparing DeepSeek-OCR and Tesseract. I am building this as a portfolio-grade AI Engineering project with a Dockerized local architecture and a public Hugging Face Spaces deployment.

General rules for the project:
- Keep everything CPU-only (free tier).
- Local development uses Docker Compose with multiple services: FastAPI API, DeepSeek service, Tesseract service, and benchmark worker.
- The public demo will run on Hugging Face Spaces using a single Streamlit app that calls either:
  • my public API endpoint, or
  • a light adapted version of DeepSeek-OCR running directly inside the Space.
- Code should be production-clean, minimal, reproducible, and easy to deploy.

Whenever I ask for code or files:
- Follow the repo structure:
  doc-benchmark-deepseek-ocr/
    ├ docker-compose.yml
    ├ services/
    │  ├ api/
    │  ├ deepseek_service/
    │  ├ tesseract_service/
    │  └ benchmark_worker/
    ├ data/
    └ hf_space/
- All code for local services must be Docker-friendly.
- HF Spaces code must be lightweight and simplified (single app.py only).
- HF Space should display:
   • Upload area
   • DeepSeek output
   • Tesseract output
   • Length metrics
   • Comparison area
   • Run history from API (if API is public)

When generating README, documentation, or instructions:
- Use a professional tone
- Include architecture explanation, setup instructions, benchmark workflow, metrics, and roadmap

When I request improvements:
- Suggest realistic enhancements based on AI Engineering best practices, such as:
   • Better metrics (Levenshtein, token accuracy)
   • UI improvements
   • Additional OCR models
   • Async workers
   • Dataset creation
   • Evaluation pipelines

Your goal is to help me build a robust, credible, portfolio-grade OCR benchmarking framework that looks like something an AI Engineer Lead would architect.
