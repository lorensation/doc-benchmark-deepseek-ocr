# doc-benchmark-deepseek-ocr

An end-to-end OCR benchmarking environment for comparing state-of-the-art models.

---

## Overview

**doc-benchmark-deepseek-ocr** is a comprehensive OCR benchmarking platform that compares DeepSeek-OCR, an advanced open-weight vision-language model, against Tesseract, the classic OCR baseline. This project demonstrates production-ready AI Engineering practices, including:

- **Document ingestion pipeline** for processing diverse image formats
- **Multi-service architecture** with isolated, containerized OCR services
- **Evaluation logic** for measuring text quality, accuracy, and performance
- **Interactive benchmarking UI** for visualizing and comparing results

The platform is designed to be reproducible, extensible, and deployable on CPU-only infrastructure, making it accessible for development and portfolio demonstration without requiring expensive GPU resources.

---

## Goals

This project aims to:

1. **Build a reproducible OCR testing environment** that can be deployed consistently across different systems
2. **Compare model outputs, performance, and text quality** between DeepSeek-OCR and Tesseract OCR
3. **Provide an extensible architecture** for adding additional OCR models and evaluation metrics
4. **Serve as a portfolio-grade AI Engineer project** demonstrating real-world skills in:
   - Multi-service system design
   - API development and orchestration
   - Model integration and benchmarking
   - Data pipeline engineering
   - Containerization and deployment

---

## Features

- **Multi-model OCR**: Side-by-side comparison of DeepSeek-OCR and Tesseract
- **FastAPI backend**: RESTful API for document uploads and benchmark orchestration
- **CPU-only Docker deployment**: Free to run, no GPU required
- **Streamlit/HuggingFace Spaces UI**: Interactive web interface for visualization
- **JSON-based benchmarking storage**: Lightweight persistent results tracking
- **Extensible design**: Easy integration of additional OCR models
- **Local data volume**: Persistent storage for uploads and outputs
- **Async processing**: Non-blocking benchmark execution
- **Comprehensive metrics**: Multiple evaluation dimensions for thorough comparison

---

## Architecture

The system follows a microservices architecture with clear separation of concerns:

### Components

1. **API Service** (FastAPI)
   - Handles document uploads via REST endpoints
   - Orchestrates benchmark requests to OCR services
   - Aggregates and stores results
   - Serves benchmark data to the UI

2. **DeepSeek-OCR Service**
   - Containerized inference service for DeepSeek-OCR model
   - Processes documents using HuggingFace Transformers
   - Returns extracted text with metadata

3. **Tesseract Service**
   - Containerized Tesseract OCR engine
   - Baseline comparison model
   - Provides traditional OCR results

4. **Benchmark Worker**
   - Computes evaluation metrics
   - Compares outputs across models
   - Generates performance reports

5. **UI Layer** (Streamlit/HuggingFace Spaces)
   - Visualizes benchmark results
   - Displays side-by-side comparisons
   - Shows metrics and performance data

### Request Flow

```
User → Upload Document → API Service
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
      DeepSeek Service         Tesseract Service
              ↓                       ↓
              └───────────┬───────────┘
                          ↓
                  Benchmark Worker
                          ↓
                  Metrics Computation
                          ↓
                   Results Store (JSON)
                          ↓
                    UI (Streamlit)
```

> **Note**: Architecture diagram (`architecture.svg`) to be added in future updates.

---

## Technology Stack

- **Backend Framework**: FastAPI (Python 3.12)
- **OCR Models**:
  - HuggingFace Transformers (DeepSeek-OCR)
  - Pytesseract / Tesseract OCR
- **UI Framework**: Streamlit or HuggingFace Spaces
- **Containerization**: Docker & Docker Compose
- **Data Storage**: JSON files and SQLite (lightweight, version-controllable)
- **Image Processing**: PIL/Pillow
- **HTTP Client**: httpx/requests
- **Development**: Python 3.12, pip/uv for dependency management

---

## Repository Structure

```
doc-benchmark-deepseek-ocr/
├── api/                          # FastAPI application
│   ├── main.py                   # API endpoints and routing
│   ├── models.py                 # Data models and schemas
│   ├── benchmark.py              # Benchmark orchestration logic
│   └── requirements.txt          # API dependencies
├── services/
│   ├── deepseek/                 # DeepSeek-OCR service
│   │   ├── Dockerfile
│   │   ├── service.py            # DeepSeek inference server
│   │   └── requirements.txt
│   └── tesseract/                # Tesseract service
│       ├── Dockerfile
│       ├── service.py            # Tesseract inference server
│       └── requirements.txt
├── ui/                           # Streamlit UI
│   ├── app.py                    # Main Streamlit application
│   └── requirements.txt
├── data/                         # Persistent data volume
│   ├── uploads/                  # Uploaded documents
│   └── results/                  # Benchmark results (JSON)
├── tests/                        # Test suite
│   ├── test_api.py
│   ├── test_services.py
│   └── test_benchmarks.py
├── docker-compose.yml            # Multi-service orchestration
├── .env.example                  # Environment configuration template
├── README.md                     # This file
└── LICENSE                       # GPL-3.0 License

```

Each service is containerized and isolated, communicating via HTTP. This design enables independent scaling, development, and testing of components.

---

## Installation & Setup

### Prerequisites

- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)
- **CPU-only environment** (no GPU required)
- At least 8GB RAM recommended
- 10GB free disk space

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/lorensation/doc-benchmark-deepseek-ocr.git
   cd doc-benchmark-deepseek-ocr
   ```

2. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env to customize ports, paths, etc.
   ```

3. **Build and start services**
   ```bash
   docker-compose up --build
   ```

   This will:
   - Build all service containers
   - Download required models
   - Start API, DeepSeek, and Tesseract services
   - Initialize data volumes

4. **Verify services are running**
   ```bash
   # Check container status
   docker-compose ps
   
   # Test API health
   curl http://localhost:8000/health
   ```

5. **Access the API**
   - Interactive API docs: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

6. **Run the UI** (optional)
   ```bash
   # In a separate terminal
   cd ui
   pip install -r requirements.txt
   streamlit run app.py
   ```
   - UI will be available at: `http://localhost:8501`

---

## Running Benchmarks

### Via API

Use the `/upload_and_benchmark` endpoint to process documents:

```bash
curl -X POST "http://localhost:8000/upload_and_benchmark" \
  -F "file=@/path/to/document.png" \
  -F "benchmark_name=my_test"
```

**Response:**
```json
{
  "benchmark_id": "uuid-here",
  "status": "processing",
  "models": ["deepseek-ocr", "tesseract"],
  "created_at": "2025-11-13T20:56:35Z"
}
```

### Benchmark Workflow

1. **Upload**: The API receives the document and stores it in `data/uploads/`
2. **Dispatch**: Sends the document to each OCR service concurrently
3. **Process**: Each service extracts text using its respective model
4. **Collect**: Results are gathered from all services
5. **Evaluate**: Metrics are computed comparing the outputs
6. **Store**: Results saved to `data/results/{benchmark_id}.json`
7. **Display**: UI queries the API and visualizes the comparison

### Result Structure

Results are stored as JSON:

```json
{
  "benchmark_id": "...",
  "document": "filename.png",
  "timestamp": "...",
  "models": {
    "deepseek-ocr": {
      "text": "extracted text...",
      "confidence": 0.95,
      "processing_time_ms": 1234
    },
    "tesseract": {
      "text": "extracted text...",
      "confidence": 0.87,
      "processing_time_ms": 456
    }
  },
  "metrics": {
    "text_length_comparison": {...},
    "levenshtein_distance": 42,
    "token_accuracy": 0.92
  }
}
```

### Via UI

1. Open `http://localhost:8501`
2. Upload a document using the file uploader
3. Click "Run Benchmark"
4. View side-by-side comparison of DeepSeek vs Tesseract
5. Explore metrics and performance data

---

## Benchmark Metrics

### Currently Implemented

- **Text Length Comparison**: Character count for each model's output
- **Extracted Text (Qualitative)**: Full text output for manual review
- **Processing Time**: Latency measurement for each OCR service

### Planned Advanced Metrics

The project roadmap includes these additional evaluation dimensions:

- **Levenshtein Distance**: Edit distance between outputs and ground truth
- **Token Accuracy**: Percentage of correctly recognized tokens/words
- **Character Error Rate (CER)**: Fine-grained accuracy measurement
- **Word Error Rate (WER)**: Word-level accuracy
- **Table Structure Accuracy**: Evaluation of table extraction quality
- **Layout Preservation**: Assessment of formatting retention
- **Throughput**: Documents processed per second
- **Resource Utilization**: CPU/memory usage during inference
- **Confidence Scores**: Model certainty analysis
- **Language Detection**: Multilingual capability testing

### Extensibility

The metrics framework is designed for easy extension. To add new metrics:

1. Implement metric function in `api/benchmark.py`
2. Add metric to `compute_metrics()` pipeline
3. Update result schema in `api/models.py`
4. Visualize in UI (`ui/app.py`)

This modular design allows the benchmark suite to evolve with research and industry standards.

---

## Roadmap / Future Work

### Short Term
- [ ] Implement advanced metrics (Levenshtein, CER, WER)
- [ ] Add confidence score analysis
- [ ] Support batch document processing
- [ ] Add PDF support (multi-page documents)
- [ ] Improve error handling and logging

### Medium Term
- [ ] Integrate additional OCR models:
  - PaddleOCR
  - EasyOCR
  - Surya-OCR
  - Commercial APIs (Google Vision, AWS Textract)
- [ ] Add async worker queue (Celery/RQ) for background processing
- [ ] Implement ground truth comparison with labeled datasets
- [ ] Add GPU support for faster DeepSeek inference
- [ ] Create domain-specific test datasets (invoices, forms, receipts, etc.)

### Long Term
- [ ] Build continuous benchmarking pipeline (CI/CD integration)
- [ ] Add model fine-tuning capabilities
- [ ] Support custom model deployment
- [ ] Implement A/B testing framework
- [ ] Create public leaderboard for OCR models
- [ ] Add export to standard benchmarking formats (COCO, etc.)
- [ ] Develop specialized metrics for:
  - Handwriting recognition
  - Historical document processing
  - Mathematical notation
  - Non-Latin scripts

### Community & Documentation
- [ ] Add comprehensive API documentation
- [ ] Create video tutorials
- [ ] Write blog posts on findings
- [ ] Publish benchmark datasets
- [ ] Contribute findings back to OCR model communities

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

See the [LICENSE](LICENSE) file for full details.

### Why GPL-3.0?

The GPL-3.0 license ensures:
- **Freedom to use**: Run the software for any purpose
- **Freedom to study**: Access and modify the source code
- **Freedom to share**: Distribute copies to help others
- **Freedom to improve**: Distribute modified versions
- **Copyleft protection**: Derivative works must also be open source

This license choice supports the open-source AI research community while ensuring improvements benefit everyone.

### Third-Party Components

- **DeepSeek-OCR**: Subject to its respective license
- **Tesseract OCR**: Apache License 2.0
- **FastAPI**: MIT License
- **Streamlit**: Apache License 2.0

---

## Contributing

Contributions are welcome! This project is designed to be community-driven. Areas where contributions are especially valuable:

- Adding new OCR models
- Implementing additional metrics
- Creating test datasets
- Improving documentation
- Bug fixes and performance optimizations

Please open an issue before submitting major changes to discuss your approach.

---

## Acknowledgments

- **DeepSeek Team** for the DeepSeek-OCR model
- **Tesseract OCR** community for the foundational OCR engine
- **HuggingFace** for model hosting and Transformers library
- **FastAPI** and **Streamlit** communities for excellent frameworks

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/lorensation/doc-benchmark-deepseek-ocr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lorensation/doc-benchmark-deepseek-ocr/discussions)
- **Repository**: [github.com/lorensation/doc-benchmark-deepseek-ocr](https://github.com/lorensation/doc-benchmark-deepseek-ocr)

---

**Built with ❤️ for the AI Engineering community**
