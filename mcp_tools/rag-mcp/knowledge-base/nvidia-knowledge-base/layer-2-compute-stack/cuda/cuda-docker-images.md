---
layer: "2"
category: "cuda"
subcategory: "containers"
tags: ["cuda", "docker", "containers", "deployment"]
cuda_version: "13.0+"
cuda_verified: "13.0"
last_updated: 2025-11-17
difficulty: "beginner"
estimated_time: "20min"
---

# CUDA Docker Images - Complete Reference

*Official CUDA container images for development, ML/AI workloads, and production deployment*

## Overview

Nvidia provides official Docker images with CUDA Toolkit and cuDNN for easy containerized GPU applications.

**Docker Hub**: [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda)  
**Documentation**: [CUDA Container Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Prerequisites

### Install Nvidia Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

## Image Types

### Base Images

Minimal CUDA runtime (no development tools):

```bash
# Pull base image
docker pull nvidia/cuda:13.0.0-base-ubuntu22.04

# Run with GPU access
docker run --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

**Use cases:**
- Production deployment
- Inference only
- Minimal image size

### Runtime Images

CUDA runtime + cuDNN (no compiler):

```bash
# Pull runtime image
docker pull nvidia/cuda:13.0.0-runtime-ubuntu22.04

# With cuDNN
docker pull nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
```

**Use cases:**
- Running pre-compiled applications
- ML inference
- Smaller than devel images

### Devel Images

Full development environment (compiler, headers, samples):

```bash
# Pull devel image
docker pull nvidia/cuda:13.0.0-devel-ubuntu22.04

# With cuDNN
docker pull nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04
```

**Use cases:**
- Building CUDA applications
- Development and compilation
- Full toolchain needed

## Image Naming Convention

```
nvidia/cuda:<cuda_version>-<image_type>-<os>
```

**Examples:**
```bash
# CUDA 12.6, base, Ubuntu 22.04
nvidia/cuda:13.0.0-base-ubuntu22.04

# CUDA 12.6, runtime with cuDNN, Ubuntu 22.04
nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

# CUDA 12.6, devel with cuDNN, Ubuntu 22.04
nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

# CUDA 12.0, devel, RHEL 8
nvidia/cuda:12.0.0-devel-ubi8
```

## Available Versions

### CUDA Versions

- **12.6.x** (latest stable)
- **12.5.x**
- **12.4.x**
- **12.3.x**
- **12.0.x**
- **11.8.x** (legacy)

### Operating Systems

- **ubuntu22.04** (recommended)
- **ubuntu20.04**
- **ubi8** (RHEL/CentOS 8)
- **ubi9** (RHEL/CentOS 9)

## Basic Usage

### Running Interactive Container

```bash
# Start interactive session
docker run --gpus all -it nvidia/cuda:13.0.0-devel-ubuntu22.04 bash

# Inside container
nvidia-smi
nvcc --version
```

### Running with Volume Mounts

```bash
# Mount current directory
docker run --gpus all -v $(pwd):/workspace \
       -w /workspace \
       nvidia/cuda:13.0.0-devel-ubuntu22.04 \
       nvcc program.cu -o program

# Run compiled program
docker run --gpus all -v $(pwd):/workspace \
       -w /workspace \
       nvidia/cuda:13.0.0-runtime-ubuntu22.04 \
       ./program
```

### Running with Specific GPUs

```bash
# Use GPU 0 only
docker run --gpus '"device=0"' nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi

# Use GPUs 0 and 1
docker run --gpus '"device=0,1"' nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi

# Use all GPUs
docker run --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

## Building Custom Images

### Dockerfile Example (Development)

```dockerfile
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Copy application
COPY . /app

# Default command
CMD ["python3", "train.py"]
```

Build and run:

```bash
docker build -t my-cuda-app .
docker run --gpus all my-cuda-app
```

### Dockerfile Example (Production)

```dockerfile
# Multi-stage build
# Stage 1: Build
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04 AS builder

WORKDIR /app
COPY program.cu .
RUN nvcc -O3 -arch=sm_80 program.cu -o program

# Stage 2: Runtime
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

WORKDIR /app
COPY --from=builder /app/program .

CMD ["./program"]
```

### Dockerfile Example (ML Inference)

```dockerfile
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3-pip

# Install inference dependencies
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip3 install transformers accelerate

# Copy model and inference script
COPY model/ /app/model/
COPY inference.py /app/

WORKDIR /app
EXPOSE 8000

CMD ["python3", "inference.py"]
```

## Docker Compose

### docker-compose.yml

```yaml
version: '3.8'

services:
  training:
    image: nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./code:/workspace
    working_dir: /workspace
    command: python3 train.py

  inference:
    image: nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/models
    ports:
      - "8000:8000"
    command: python3 serve.py
```

Run:

```bash
docker-compose up training
# or
docker-compose up inference
```

## Common Use Cases

### Compile CUDA Program

```bash
docker run --rm -v $(pwd):/workspace -w /workspace \
       nvidia/cuda:13.0.0-devel-ubuntu22.04 \
       nvcc -O3 -arch=sm_80 program.cu -o program
```

### Run PyTorch Training

```bash
docker run --gpus all -v $(pwd):/workspace -w /workspace \
       nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04 \
       bash -c "pip install torch && python train.py"
```

### Interactive Jupyter Notebook

```bash
docker run --gpus all -p 8888:8888 \
       -v $(pwd):/workspace \
       nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04 \
       bash -c "pip install jupyter torch && \
                jupyter notebook --ip=0.0.0.0 --allow-root --no-browser"
```

## Image Size Comparison

Approximate sizes for CUDA 12.6 Ubuntu 22.04:

| Image Type | Size | Use Case |
|------------|------|----------|
| base | ~2GB | Runtime only |
| runtime | ~3GB | Pre-compiled apps |
| cudnn-runtime | ~4GB | ML inference |
| devel | ~5GB | Development |
| cudnn-devel | ~8GB | ML development |

## Best Practices

### 1. Choose Right Image Type

```bash
# Development → devel
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

# Production → runtime
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

# Inference only → runtime
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04
```

### 2. Multi-Stage Builds

```dockerfile
# Build stage with devel
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder
# ... compile code ...

# Runtime stage with smaller image
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04
COPY --from=builder /app/binary .
```

### 3. Layer Caching

```dockerfile
# Install dependencies first (cached)
RUN apt-get update && apt-get install -y python3-pip

# Copy requirements (cached if unchanged)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code last (changes frequently)
COPY . .
```

### 4. Clean Up

```dockerfile
RUN apt-get update && apt-get install -y package \
    && rm -rf /var/lib/apt/lists/*  # Clean up

RUN pip install --no-cache-dir package  # Don't cache pip packages
```

### 5. Non-Root User

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Rest of Dockerfile
```

## Troubleshooting

### Issue: "could not select device driver"

**Solution:**
```bash
# Install/update nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Issue: "no CUDA-capable device"

**Solution:**
```bash
# Check host GPU
nvidia-smi

# Verify container toolkit
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### Issue: Version mismatch

**Solution:**
```bash
# Check driver compatibility
nvidia-smi  # Check driver version

# Use compatible CUDA version
# Driver 560+ → CUDA 12.6
# Driver 555+ → CUDA 12.5
# Driver 525+ → CUDA 12.0
```

### Issue: Out of memory in container

**Solution:**
```bash
# Check GPU memory
nvidia-smi

# Limit container memory
docker run --gpus all --memory=8g --memory-swap=8g your-image
```

## External Resources

- [CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda)
- [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [CUDA Container Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)

## Related Guides

- [CUDA Installation](cuda-installation.md)
- [CUDA Programming Basics](cuda-basics.md)
- [Docker for LLM Deployment](../../layer-5-llm/01-foundations/docker-basics.md)
- [Production LLM Serving](../../layer-5-llm/02-inference/deployment/production-serving.md)

