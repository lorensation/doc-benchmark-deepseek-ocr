# 🔍 OCR Benchmark Space

**A comprehensive benchmarking platform for state-of-the-art OCR and Vision-Language Models**

[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/lorensation/ocr-benchmark)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green?logo=nvidia)](https://developer.nvidia.com/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

---

## 📖 Overview

**OCR Benchmark Space** is an interactive evaluation environment that enables real-time comparison of multiple state-of-the-art Optical Character Recognition (OCR) and Vision-Language Models (VLMs). Upload any document image and instantly benchmark performance across five different models with comprehensive metrics including Word Error Rate (WER), Character Error Rate (CER), and Levenshtein Edit Ratio (LER).

### What You Can Do

✅ **Upload and Test**: Drop any document image (receipts, invoices, forms, scanned pages)  
✅ **Compare Models**: Side-by-side evaluation of 5 OCR/VLM systems  
✅ **Analyze Metrics**: Comprehensive accuracy measurements (WER, CER, LER, processing time)  
✅ **GPU Acceleration**: Fast inference with CUDA-enabled models  
✅ **Export Results**: Download benchmark reports as JSON  

### Purpose

This space democratizes access to cutting-edge OCR technology evaluation, enabling researchers, developers, and businesses to make informed decisions about which models best suit their document processing needs.

---

## ✨ Features

🚀 **Multi-Model Inference**  
Run OCR inference across 5 models simultaneously with a single upload

📊 **Advanced Metrics**  
- Word Error Rate (WER)
- Character Error Rate (CER)
- Levenshtein Edit Ratio (LER)
- Processing time comparison
- Text length analysis

⚡ **GPU Acceleration**  
CUDA-enabled inference for VLMs (DeepSeek, Hunyuan, Qwen2-VL, VISTA)

🎨 **Interactive Streamlit UI**  
Real-time result visualization with side-by-side comparisons

🔧 **Quantized Model Support**  
Efficient 4-bit quantized Qwen2-VL (7B) for faster inference

🐳 **Docker Containerized**  
Fully reproducible environment with isolated service architecture

📁 **Flexible Input**  
Support for PNG, JPEG, JPG, BMP, TIFF formats

---

## 🤖 Models Included

| Model | Type | Source | GPU | Notes |
|-------|------|--------|-----|-------|
| **DeepSeek-OCR** | VLM | [DeepSeek-AI](https://huggingface.co/deepseek-ai) | ✅ | Vision-language model optimized for documents |
| **Tesseract OCR** | Traditional OCR | [tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract) | ❌ | CPU-only, battle-tested baseline |
| **VISTA-OCR** | VLM | [VISTA](https://huggingface.co/vista) | ✅ | Multimodal document understanding |
| **HunyuanOCR** | VLM | [Tencent Hunyuan](https://huggingface.co/tencent/hunyuan) | ✅ | Chinese + multilingual OCR excellence |
| **Qwen2-VL-7B (4-bit)** | VLM (Quantized) | [Qwen/Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B) | ✅ | Efficient quantized vision-language model |

### Model Details

- **DeepSeek-OCR**: Advanced transformer-based VLM with strong performance on structured documents
- **Tesseract**: Industry-standard OCR engine, excellent for simple text extraction
- **VISTA-OCR**: Specialized in document layout understanding and complex structures
- **HunyuanOCR**: State-of-the-art multilingual capabilities, particularly strong with Chinese text
- **Qwen2-VL**: Large-scale VLM with 7B parameters, 4-bit quantization for memory efficiency

---

## 🎥 Live Demo Preview

![OCR Benchmark Demo](./assets/demo.png)
*Upload interface and real-time comparison view*

![Metrics Dashboard](./assets/metrics.png)
*Comprehensive metrics visualization for all models*

> **Note**: Screenshots show the Streamlit interface running on HuggingFace Spaces

---

## 🏗️ Architecture Overview

### System Design

The OCR Benchmark Space uses a **microservices architecture** with isolated Docker containers for each model:

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI (Port 7860)           │
│              User Upload & Visualization            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Gateway (Port 8000)            │
│          Orchestration & Result Aggregation         │
└──────┬──────┬──────┬──────┬──────┬──────────────────┘
       │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼
   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │DeepSeek│ │Tesseract│ │VISTA│ │Hunyuan│ │Qwen2-VL│
   │  :8001  │ │  :8002   │ │:8003│ │ :8004  │ │ :8005   │
   └────┘ └────┘ └────┘ └────┘ └────┘
   (GPU)  (CPU)  (GPU)  (GPU)  (GPU)
```

### Data Flow

1. **Upload**: User uploads image via Streamlit interface
2. **Dispatch**: API gateway routes image to all 5 OCR services in parallel
3. **Process**: Each service performs inference using its respective model
4. **Aggregate**: Results collected and metrics computed (WER, CER, LER)
5. **Visualize**: Streamlit displays side-by-side comparison with metrics
6. **Export**: Results saved as JSON for further analysis

### Technical Stack

- **Frontend**: Streamlit 1.32+
- **Backend**: FastAPI 0.109+
- **OCR Services**: 
  - DeepSeek: Transformers + CUDA
  - Tesseract: pytesseract + Tesseract 5.x
  - VISTA: Custom inference server
  - Hunyuan: Transformers + CUDA
  - Qwen2-VL: Transformers + bitsandbytes (4-bit)
- **Metrics**: Jiwer (WER), rapidfuzz (LER), custom CER implementation
- **Containerization**: Docker 24.0+, Docker Compose 2.20+
- **GPU**: CUDA 12.1, cuDNN 8.9

---

## 🚀 Run Locally

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/lorensation/doc-benchmark-deepseek-ocr.git
cd doc-benchmark-deepseek-ocr

# Build the Docker image (includes all models)
docker build -t ocr-benchmark .

# Run with GPU support
docker run --gpus all -p 7860:7860 ocr-benchmark

# Or run with CPU only (slower inference)
docker run -p 7860:7860 ocr-benchmark
```

**Access the application**: Open `http://localhost:7860` in your browser

### Option 2: Without Docker

#### Prerequisites
- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- 16GB+ RAM recommended
- 20GB disk space for models

#### Setup Steps

```bash
# Clone repository
git clone https://github.com/lorensation/doc-benchmark-deepseek-ocr.git
cd doc-benchmark-deepseek-ocr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract (system dependency)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

#### Download Models

Models are automatically downloaded on first run via HuggingFace Hub. To pre-download:

```python
from transformers import AutoModel, AutoTokenizer

# DeepSeek-OCR
AutoModel.from_pretrained("deepseek-ai/deepseek-ocr")

# Qwen2-VL-7B (4-bit quantized)
AutoModel.from_pretrained("Qwen/Qwen2-VL-7B", load_in_4bit=True)

# VISTA, Hunyuan - similar process
```

#### Start Services

```bash
# Terminal 1: Start API Gateway
cd services/api
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2-6: Start each OCR service
cd services/deepseek_service && python server.py
cd services/tesseract_service && python server.py
cd services/vista_service && python server.py
cd services/hunyuan_service && python server.py
cd services/qwen2vl_service && python server.py

# Terminal 7: Start Streamlit UI
streamlit run hf_space/app.py --server.port 7860
```

**Access**: Navigate to `http://localhost:7860`

---

## 📁 File Structure

```
doc-benchmark-deepseek-ocr/
├── hf_space/
│   ├── app.py                    # Streamlit UI application
│   └── requirements.txt          # UI dependencies
├── services/
│   ├── api/
│   │   ├── main.py              # FastAPI gateway
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── deepseek_service/
│   │   ├── server.py            # DeepSeek inference server
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── tesseract_service/
│   │   ├── server.py            # Tesseract wrapper
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── vista_service/
│   │   ├── server.py            # VISTA inference
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── hunyuan_service/
│   │   ├── server.py            # Hunyuan inference
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── qwen2vl_service/
│   │   ├── server.py            # Qwen2-VL 4-bit inference
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── benchmark_worker/
│       ├── worker.py            # Metrics computation
│       ├── metrics/
│       │   ├── wer.py          # Word Error Rate
│       │   ├── cer.py          # Character Error Rate
│       │   └── ler.py          # Levenshtein Edit Ratio
│       └── requirements.txt
├── scripts/
│   ├── batch_quick_test.py      # Batch testing utilities
│   ├── compare_ocr_services.py  # Service comparison tool
│   └── validate_dataset.py      # Dataset validation
├── data/
│   ├── datasets/                # Test datasets
│   ├── results/                 # Benchmark outputs
│   └── uploads/                 # User uploaded images
├── docker-compose.yml           # Local multi-service orchestration
├── Dockerfile                   # HuggingFace Spaces container
├── requirements.txt             # Root dependencies
├── README.md                    # This file
└── LICENSE                      # GPL-3.0 License
```

---

## 🔧 API Reference

### Core Functions

#### `run_ocr(model_name: str, image: PIL.Image) -> dict`

Execute OCR inference on a single model.

**Parameters:**
- `model_name` (str): One of `["deepseek", "tesseract", "vista", "hunyuan", "qwen2vl"]`
- `image` (PIL.Image): Input image for OCR

**Returns:**
```python
{
    "text": str,              # Extracted text
    "confidence": float,      # Model confidence (0-1)
    "processing_time": float, # Inference time in seconds
    "model": str             # Model identifier
}
```

#### `compute_metrics(prediction: str, ground_truth: str) -> dict`

Calculate accuracy metrics between predicted and ground truth text.

**Parameters:**
- `prediction` (str): Model output text
- `ground_truth` (str): Reference text

**Returns:**
```python
{
    "wer": float,        # Word Error Rate (0-1)
    "cer": float,        # Character Error Rate (0-1)
    "ler": float,        # Levenshtein Edit Ratio (0-1)
    "text_length": int   # Character count
}
```

#### `benchmark_all_models(image: PIL.Image) -> dict`

Run comprehensive benchmark across all models.

**Parameters:**
- `image` (PIL.Image): Input document image

**Returns:**
```python
{
    "benchmark_id": str,
    "timestamp": str,
    "results": {
        "deepseek": {...},
        "tesseract": {...},
        "vista": {...},
        "hunyuan": {...},
        "qwen2vl": {...}
    },
    "metrics_comparison": {...}
}
```

### REST API Endpoints

When running locally, the API gateway exposes:

- `POST /api/ocr/single` - Single model inference
- `POST /api/ocr/benchmark` - Multi-model benchmark
- `GET /api/health` - Service health check
- `GET /api/models` - List available models

---

## 📊 Metrics Explained

### Word Error Rate (WER)

Measures word-level accuracy:

$$
WER = \frac{S + D + I}{N}
$$

Where:
- `S` = Substitutions (wrong words)
- `D` = Deletions (missing words)
- `I` = Insertions (extra words)
- `N` = Total words in reference

**Lower is better** (0 = perfect)

### Character Error Rate (CER)

Measures character-level accuracy using the same formula as WER but at character granularity.

**Lower is better** (0 = perfect)

### Levenshtein Edit Ratio (LER)

Normalized edit distance between strings:

$$
LER = \frac{EditDistance(pred, ref)}{max(len(pred), len(ref))}
$$

**Lower is better** (0 = identical strings)

---

## 🙏 Credits & Acknowledgments

### Models

- **DeepSeek-OCR** - [DeepSeek-AI](https://github.com/deepseek-ai) - MIT License
- **Tesseract OCR** - [Tesseract Team](https://github.com/tesseract-ocr/tesseract) - Apache 2.0
- **VISTA-OCR** - [VISTA Team](https://huggingface.co/vista) - Apache 2.0
- **HunyuanOCR** - [Tencent Hunyuan](https://huggingface.co/tencent/hunyuan) - Apache 2.0
- **Qwen2-VL** - [Alibaba Qwen Team](https://github.com/QwenLM/Qwen2-VL) - Apache 2.0

### Infrastructure

- **HuggingFace Spaces** - For hosting and GPU infrastructure
- **Streamlit** - Interactive UI framework
- **FastAPI** - High-performance API gateway
- **Docker** - Containerization platform

### Libraries

- `transformers` - HuggingFace model hub
- `bitsandbytes` - 4-bit quantization
- `pytesseract` - Tesseract Python wrapper
- `jiwer` - WER computation
- `rapidfuzz` - Fast string matching

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

### Key Points

✅ **Open Source**: Free to use, modify, and distribute  
✅ **Copyleft**: Derivative works must also be GPL-3.0  
✅ **Commercial Use**: Allowed with GPL compliance  
✅ **Patent Grant**: Contributors grant patent rights  

See [LICENSE](LICENSE) for full terms.

### Third-Party Licenses

Individual models and libraries retain their original licenses:
- DeepSeek-OCR: MIT
- Tesseract: Apache 2.0
- VISTA, Hunyuan, Qwen2-VL: Apache 2.0
- Streamlit: Apache 2.0
- FastAPI: MIT

---

## 🤝 Contributing

Contributions are welcome! We're particularly interested in:

- 🆕 Adding new OCR/VLM models
- 📊 Implementing additional metrics
- 🧪 Creating test datasets
- 📚 Improving documentation
- 🐛 Bug fixes and optimizations

**Please open an issue** before major changes to discuss your approach.

---

## 📞 Contact & Support

- **HuggingFace Space**: [huggingface.co/spaces/lorensation/ocr-benchmark](https://huggingface.co/spaces/lorensation/ocr-benchmark)
- **GitHub Repository**: [github.com/lorensation/doc-benchmark-deepseek-ocr](https://github.com/lorensation/doc-benchmark-deepseek-ocr)
- **Issues**: [GitHub Issues](https://github.com/lorensation/doc-benchmark-deepseek-ocr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/lorensation/doc-benchmark-deepseek-ocr/discussions)

---

<div align="center">

**Built with ❤️ for the AI community**

[![Star on GitHub](https://img.shields.io/github/stars/lorensation/doc-benchmark-deepseek-ocr?style=social)](https://github.com/lorensation/doc-benchmark-deepseek-ocr)
[![Follow on HF](https://img.shields.io/badge/🤗-Follow%20on%20HuggingFace-yellow)](https://huggingface.co/lorensation)

</div>
