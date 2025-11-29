# DeepSeek-OCR GPU Setup Guide

## Overview

DeepSeek-OCR is now configured to run with **GPU acceleration** using CUDA and Flash Attention 2 for optimal performance.

## Requirements

### Hardware
- **NVIDIA GPU** with CUDA Compute Capability 7.0+ (e.g., RTX 2060 or newer)
- **12+ GB VRAM** recommended for full model (8GB minimum)
- **CUDA 12.1+** compatible drivers

### Software
- **Docker** with NVIDIA Container Toolkit (nvidia-docker2)
- **NVIDIA Driver** 525.60.13+ (for CUDA 12.1)
- **Docker Compose** v2.0+

## Setup

### 1. Install NVIDIA Container Toolkit

#### Ubuntu/Debian
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Windows with WSL2
1. Install [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
3. Enable GPU support in Docker Desktop settings

### 2. Verify GPU Access

```bash
# Test NVIDIA container toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Should show your GPU information
```

### 3. Build and Run

```bash
# Build with GPU support
docker-compose build deepseek

# Start services
docker-compose up -d

# Check GPU utilization
docker exec docbench-deepseek nvidia-smi

# View logs
docker logs -f docbench-deepseek
```

## Configuration

### Docker Compose GPU Settings

The `docker-compose.yml` includes:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # Number of GPUs to use
          capabilities: [gpu]

environment:
  - CUDA_VISIBLE_DEVICES=0  # Specific GPU to use
  - NVIDIA_VISIBLE_DEVICES=all
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify which GPU(s) to use (e.g., `0`, `0,1`)
- `MODEL_NAME`: HuggingFace model identifier
- `HF_HOME`: Cache directory for model weights

## Performance

### Expected Speed
- **CPU**: 30-60s per document (not recommended)
- **GPU (RTX 4060)**: 3-8s per document
- **GPU (A100)**: 1-3s per document

### Memory Usage
- **Model Loading**: ~8GB VRAM
- **Peak Inference**: ~10-12GB VRAM
- **Batch Processing**: Scales with batch size

## Monitoring

### Check GPU Status
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Inside container
docker exec docbench-deepseek nvidia-smi

# Memory usage via API
curl http://localhost:9000/ready | jq '.gpu_memory'
```

### Health Check with GPU Info
```bash
curl http://localhost:9000/health | jq .
```

Response includes:
```json
{
  "status": "healthy",
  "device": "cuda",
  "gpu": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 4060",
    "device_count": 1,
    "cuda_version": "12.1"
  }
}
```

## Troubleshooting

### Issue: "no CUDA-capable device is detected"

**Solution**: Ensure NVIDIA Container Toolkit is installed and docker daemon restarted.

```bash
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size in API calls
2. Use `torch.cuda.empty_cache()` between requests
3. Use smaller model variant if available
4. Increase GPU VRAM (upgrade GPU)

### Issue: Flash Attention build fails

**Solution**: Ensure CUDA toolkit is available during build:

```dockerfile
# Dockerfile already includes this
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
```

### Issue: Model loads but inference is slow

**Check**:
1. Verify GPU is being used: `docker exec docbench-deepseek nvidia-smi`
2. Check model device: `curl http://localhost:9000/ready | jq '.device'`
3. Ensure Flash Attention is installed: `docker exec docbench-deepseek pip show flash-attn`

## Fallback to CPU

If GPU is unavailable, the service automatically falls back to CPU with eager attention:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
```

**Note**: CPU mode is significantly slower (10-20x) and not recommended for production.

## Optimization Tips

1. **Use FP16**: Enabled by default on GPU (`torch.float16`)
2. **Flash Attention 2**: Automatically used when available
3. **Model Caching**: First run downloads model (~15GB), subsequent runs are fast
4. **Batch Processing**: Process multiple images together when possible
5. **GPU Warmup**: First inference may be slower due to CUDA initialization

## Production Deployment

### Cloud GPU Options

| Provider | Instance Type | GPU | VRAM | Cost/hr |
|----------|--------------|-----|------|---------|
| AWS | g5.xlarge | A10G | 24GB | ~$1.00 |
| GCP | n1-highmem-4 + T4 | T4 | 16GB | ~$0.50 |
| Azure | NC6s v3 | V100 | 16GB | ~$0.90 |
| RunPod | GPU Cloud | RTX 4090 | 24GB | ~$0.40 |

### Kubernetes Deployment

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1
```

## References

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [DeepSeek-OCR HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Docker Compose GPU Support](https://docs.docker.com/compose/gpu-support/)
