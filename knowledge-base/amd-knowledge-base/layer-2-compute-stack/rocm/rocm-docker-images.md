Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "2"
category: "rocm"
subcategory: "containers"
tags: ["docker", "containers", "rocm", "images", "ubuntu", "deployment"]
rocm_version: "7.0+"
rocm_verified: "7.1"
therock_included: true
last_updated: 2025-11-14
difficulty: "beginner"
estimated_time: "20min"
---

# ROCm Docker Images - Complete Reference (ROCm 7.x)

Official AMD ROCm Docker images for containerized GPU development, ML/AI workloads, and production deployment.

**⚠️ CRITICAL: This guide provides examples only. Tags change frequently. ALWAYS verify actual available tags on Docker Hub before use:**
- https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
- https://hub.docker.com/r/rocm/pytorch/tags

## Official Docker Hub Repositories

AMD maintains several official Docker image repositories:

- **[rocm/dev-ubuntu-22.04](https://hub.docker.com/r/rocm/dev-ubuntu-22.04)** - Development images (Ubuntu 22.04)
- **[rocm/dev-ubuntu-24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04)** - Development images (Ubuntu 24.04)
- **[rocm/pytorch](https://hub.docker.com/r/rocm/pytorch)** - PyTorch with ROCm
- **[rocm/tensorflow](https://hub.docker.com/r/rocm/tensorflow)** - TensorFlow with ROCm
- **[rocm/rocm-terminal](https://hub.docker.com/r/rocm/rocm-terminal)** - Minimal ROCm runtime

**Search all ROCm images**: https://hub.docker.com/search?q=rocm&type=image

## ROCm 7.x Development Images

### Ubuntu 22.04 LTS (Recommended for Production)

```bash
# Latest ROCm 7.x on Ubuntu 22.04
docker pull rocm/dev-ubuntu-22.04:latest

# ⚠️ Example tags (verify these exist on Docker Hub first!):
docker pull rocm/dev-ubuntu-22.04:7.1    # ROCm 7.1 series
docker pull rocm/dev-ubuntu-22.04:7.0    # ROCm 7.0 series

# ⚠️ CRITICAL: Always verify tags exist on Docker Hub before pulling:
# https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
# The tags shown above are examples only and may not reflect current availability
```

**What's included:**
- ROCm SDK 7.x (full development stack)
- HIP, rocBLAS, rocFFT, rocRAND, rocSOLVER, rocSPARSE
- MIOpen (ML primitives library)
- RCCL (collective communications)
- Development tools (compilers, debuggers)
- Ubuntu 22.04 LTS base

### Ubuntu 24.04 LTS (Latest LTS)

```bash
# Latest ROCm 7.x on Ubuntu 24.04
docker pull rocm/dev-ubuntu-24.04:latest

# ⚠️ Example tags (verify these exist on Docker Hub first!):
docker pull rocm/dev-ubuntu-24.04:7.1    # ROCm 7.1 series
docker pull rocm/dev-ubuntu-24.04:7.0    # ROCm 7.0 series

# ⚠️ CRITICAL: Always verify tags exist on Docker Hub before pulling:
# https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags
# The tags shown above are examples only and may not reflect current availability
```

**What's included:**
- Same as Ubuntu 22.04 variant
- Ubuntu 24.04 LTS base (newer kernel, packages)
- Better support for latest hardware

## ML/AI Framework Images

### PyTorch with ROCm 7.x

```bash
# Latest PyTorch with ROCm (recommended - always gets current version)
docker pull rocm/pytorch:latest

# ⚠️ CRITICAL: For specific tags, MUST check Docker Hub first!
# https://hub.docker.com/r/rocm/pytorch/tags
# 
# Tag naming pattern: rocm<ver>_ubuntu<ver>_py<ver>_pytorch_<ver>
# 
# Example pattern only (DO NOT assume these exact tags exist):
# - rocm7.1_ubuntu22.04_py3.10_pytorch_2.4.0
# - rocm7.0_ubuntu22.04_py3.10_pytorch_2.3.0
# 
# ⚠️ Tags vary significantly - check Docker Hub for actual available tags
# Python versions: typically 3.10, 3.11, or 3.12
# PyTorch versions: vary by ROCm version
```

**What's included:**
- PyTorch with ROCm backend (torch.cuda API works)
- torchvision, torchaudio (versions vary by image)
- CUDA compatibility layer (torch.cuda.* calls work)
- Python 3.10+ (version varies by image tag)
- ROCm libraries (rocBLAS, MIOpen, RCCL, etc.)

### TensorFlow with ROCm 7.x

```bash
# Latest TensorFlow with ROCm (recommended - always gets current version)
docker pull rocm/tensorflow:latest

# ⚠️ CRITICAL: For specific tags, MUST check Docker Hub first!
# https://hub.docker.com/r/rocm/tensorflow/tags
#
# Example pattern only (DO NOT assume these exact tags exist):
# - rocm7.1-tf2.15-dev
# - rocm7.0-tf2.14-dev
# 
# ⚠️ Tags vary - always verify on Docker Hub before using
```

**What's included:**
- TensorFlow 2.x with ROCm backend
- Keras included
- ROCm libraries
- Python 3.10+

### ROCm Terminal (Minimal Runtime)

```bash
# Minimal ROCm runtime environment
docker pull rocm/rocm-terminal:latest

# ⚠️ Example tags (verify these exist on Docker Hub first!):
docker pull rocm/rocm-terminal:7.1    # ROCm 7.1 series
docker pull rocm/rocm-terminal:7.0    # ROCm 7.0 series

# ⚠️ CRITICAL: Always verify tags exist on Docker Hub before pulling:
# https://hub.docker.com/r/rocm/rocm-terminal/tags
# The tags shown above are examples only and may not reflect current availability
```

**What's included:**
- Minimal ROCm runtime
- Basic monitoring tools (rocm-smi)
- No development tools or ML frameworks
- Smallest image size (~2-3 GB)

## Image Selection Guide

| Use Case | Recommended Image | Size | Best For |
|----------|------------------|------|----------|
| **HIP Kernel Development** | `rocm/dev-ubuntu-22.04:latest` or `:7.1` | ~12 GB | Writing custom HIP/GPU code |
| **PyTorch Training** | `rocm/pytorch:latest` | ~15 GB | ML model training |
| **PyTorch Inference** | `rocm/pytorch:latest` | ~15 GB | Model serving, APIs |
| **TensorFlow** | `rocm/tensorflow:latest` | ~16 GB | TensorFlow workloads |
| **LLM Serving (vLLM)** | `rocm/pytorch:latest` + vLLM | ~15 GB | Production LLM APIs |
| **CI/CD Testing** | `rocm/dev-ubuntu-22.04:latest` | ~12 GB | Automated testing |
| **Minimal Runtime** | `rocm/rocm-terminal:latest` or `:7.1` | ~3 GB | Lightweight deployments |
| **Latest Ubuntu** | `rocm/dev-ubuntu-24.04:latest` or `:7.1` | ~12 GB | Newest OS features |

**⚠️ CRITICAL**: All tags shown are recommendations. ALWAYS verify actual tag existence on [Docker Hub](https://hub.docker.com/search?q=rocm&type=image) before using!

## Basic Usage Patterns

### Development Container

```bash
# Run interactive development container
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    -v $(pwd):/workspace \
    -w /workspace \
    rocm/dev-ubuntu-22.04:7.1 \
    bash
```

### PyTorch Training Container

```bash
# Run PyTorch container with GPU access
# Note: Use "latest" or check Docker Hub for specific ROCm 7.1 PyTorch tags
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 32G \
    -v $(pwd):/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 \
    rocm/pytorch:latest \
    bash
```

### Multi-GPU Training

```bash
# Run with specific GPU selection
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 64G \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e WORLD_SIZE=8 \
    -v $(pwd):/workspace \
    rocm/pytorch:latest \
    torchrun --nproc_per_node=8 train.py
```

## Creating Custom Dockerfiles (ROCm 7.x)

### Minimal Development Image

```dockerfile
# Dockerfile - Basic HIP development
# ⚠️ Verify this tag exists on Docker Hub first!
# https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
FROM rocm/dev-ubuntu-22.04:7.1

# Install additional development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV HIP_PLATFORM=amd

WORKDIR /workspace
CMD ["/bin/bash"]
```

### PyTorch + vLLM for LLM Serving

```dockerfile
# Dockerfile - vLLM on ROCm 7.x
# Note: Use latest or check Docker Hub for specific ROCm 7.1 PyTorch tags
FROM rocm/pytorch:latest

# Install vLLM and dependencies (verify versions for ROCm 7.1 compatibility)
RUN pip install --no-cache-dir \
    vllm>=0.5.0 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    sentencepiece \
    protobuf

# Create working directories
RUN mkdir -p /app /models /cache

# Set environment variables
ENV HIP_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache
ENV TRANSFORMERS_CACHE=/cache

WORKDIR /app

# Expose vLLM API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command - start vLLM server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--dtype", "bfloat16"]
```

### Multi-Stage Build (Optimized Size)

```dockerfile
# Stage 1: Build dependencies
# ⚠️ Verify this tag exists on Docker Hub first!
FROM rocm/dev-ubuntu-22.04:7.1 AS builder

WORKDIR /build
COPY requirements.txt .

# Install Python dependencies
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime
# ⚠️ Verify PyTorch tag on Docker Hub: https://hub.docker.com/r/rocm/pytorch/tags
FROM rocm/pytorch:latest

# Copy only the built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY . /app
WORKDIR /app

CMD ["python", "main.py"]
```

### Production-Ready with Security

```dockerfile
FROM rocm/pytorch:latest

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app /data /models /cache && \
    chown -R appuser:appuser /app /data /models /cache

# Install dependencies as root
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HIP_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python"]
CMD ["server.py"]
```

## Docker Compose Examples

### Simple Development Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  rocm-dev:
    # ⚠️ Verify tag on Docker Hub before using
    image: rocm/dev-ubuntu-22.04:7.1
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '16gb'
    volumes:
      - ./workspace:/workspace
    working_dir: /workspace
    command: bash
```

### PyTorch Training Environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  pytorch-training:
    image: rocm/pytorch:latest  # Or use specific ROCm 7.1 tag from Docker Hub
    container_name: pytorch-train
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '32gb'
    environment:
      - HIP_VISIBLE_DEVICES=0,1,2,3
      - PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9
    volumes:
      - ./code:/workspace
      - ./data:/data
      - ./checkpoints:/checkpoints
      - huggingface-cache:/root/.cache/huggingface
    working_dir: /workspace
    command: python train.py

volumes:
  huggingface-cache:
```

### vLLM Production Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm-server:
    image: rocm/pytorch:latest  # Check Docker Hub for specific ROCm 7.1 PyTorch tags
    container_name: vllm-api
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '64gb'
    ports:
      - "8000:8000"
    environment:
      - HIP_VISIBLE_DEVICES=0,1,2,3
      - MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
      - TENSOR_PARALLEL_SIZE=4
    volumes:
      - ./models:/models
      - huggingface-cache:/root/.cache/huggingface
    command: >
      sh -c "pip install vllm>=0.5.0 &&
      python -m vllm.entrypoints.openai.api_server
      --model ${MODEL_NAME}
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
      --dtype bfloat16
      --max-model-len 8192
      --host 0.0.0.0
      --port 8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

volumes:
  huggingface-cache:
```

## Version Compatibility Matrix

**⚠️ This matrix shows general patterns only. DO NOT rely on these exact versions - they are illustrative examples.**

| ROCm Series | PyTorch (examples) | TensorFlow (examples) | Ubuntu | Python |
|-------------|-------------------|----------------------|--------|---------|
| 7.1 | 2.3.x - 2.5.x | 2.14.x - 2.16.x | 22.04, 24.04 | 3.10, 3.11, 3.12 |
| 7.0 | 2.3.x | 2.14.x | 22.04 | 3.10 |

**⚠️ The versions above are general patterns, not guaranteed tags!**

**Version Selection Guide**:
- ✅ **Recommended**: Use `:latest` tags for current stable versions
- ✅ **Production**: Verify specific tag exists on Docker Hub, then pin it
- ⚠️ **Never**: Assume a version exists without checking Docker Hub first
- ❌ **ROCm < 7.0**: Not covered in this guide

**MANDATORY - Check Docker Hub before using ANY tag:**
- **Dev Images**: https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
- **PyTorch**: https://hub.docker.com/r/rocm/pytorch/tags  
- **TensorFlow**: https://hub.docker.com/r/rocm/tensorflow/tags
- **All ROCm**: https://hub.docker.com/search?q=rocm&type=image

## Best Practices

### 1. ALWAYS Verify Tags Exist on Docker Hub First

```dockerfile
# Step 1: Check Docker Hub for actual available tags
# Visit: https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags

# Step 2: Copy exact tag from Docker Hub
FROM rocm/dev-ubuntu-22.04:7.1  # ✅ GOOD - Verified on Docker Hub

# Step 3: Or use :latest for automatic updates
FROM rocm/dev-ubuntu-22.04:latest  # ⚠️ ACCEPTABLE - May change

# ❌ NEVER assume a tag exists without verification
FROM rocm/dev-ubuntu-22.04:7.1.1  # May not exist!
```

### 2. Recommended Approach for Production

```dockerfile
# For production: Pin specific verified version
# 1. Visit Docker Hub: https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
# 2. Find latest tag (e.g., "7.1")
# 3. Test with that specific tag
# 4. Pin it in your Dockerfile

FROM rocm/dev-ubuntu-22.04:7.1  # Verified tag from Docker Hub

# For PyTorch/TensorFlow: prefer :latest unless you need a specific version
FROM rocm/pytorch:latest  # Automatically gets current stable version
```

### 3. Required Docker Flags

```bash
# Always include these for GPU access
--device=/dev/kfd \
--device=/dev/dri \
--group-add video \
--ipc=host \
--shm-size 16G  # Adjust based on workload
```

### 4. Environment Variables

```dockerfile
# Essential ROCm environment variables
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV HIP_PLATFORM=amd
ENV HIP_VISIBLE_DEVICES=0  # GPU selection
```

### 5. Security Best Practices

```dockerfile
# Run as non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Use specific package versions
RUN pip install package==1.2.3

# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
```

## Troubleshooting

### GPU Not Detected in Container

```bash
# Check devices are properly passed
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    rocm/dev-ubuntu-22.04:7.2.2 \
    rocm-smi

# Verify group membership
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    rocm/dev-ubuntu-22.04:7.2.2 \
    groups
```

### Image Pull Failures

```bash
# Check Docker Hub connectivity
docker pull rocm/dev-ubuntu-22.04:7.2.2

# Try with explicit registry
docker pull docker.io/rocm/dev-ubuntu-22.04:7.2.2

# Check Docker daemon logs
journalctl -u docker.service -n 50
```

### Out of Memory Errors

```yaml
# Increase shared memory in docker-compose.yml
services:
  app:
    shm_size: '64gb'  # Increase from default 64MB
    ipc: host         # Use host IPC namespace
```

### Permission Denied Errors

```bash
# Add user to docker group (one-time setup)
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker socket permissions
ls -la /var/run/docker.sock
```

## Finding Latest Images

**Always check Docker Hub for the latest tags:**

1. **Development Images**: 
   - Ubuntu 22.04: https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
   - Ubuntu 24.04: https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags

2. **PyTorch Images**: https://hub.docker.com/r/rocm/pytorch/tags

3. **TensorFlow Images**: https://hub.docker.com/r/rocm/tensorflow/tags

4. **All ROCm Images**: https://hub.docker.com/search?q=rocm&type=image

## Quick Reference Commands

```bash
# ⚠️ ALWAYS check Docker Hub for current tags before using these commands!
# https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags

# Pull latest ROCm 7.x development image (verify tag on Hub first)
docker pull rocm/dev-ubuntu-22.04:7.1

# Pull latest PyTorch with ROCm (check specific tag on Hub)
docker pull rocm/pytorch:latest

# Run interactive development container
docker run -it --rm --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host --shm-size 16G \
    rocm/dev-ubuntu-22.04:7.1

# Test GPU access
docker run --rm --device=/dev/kfd --device=/dev/dri \
    rocm/dev-ubuntu-22.04:7.1 rocm-smi

# List local ROCm images
docker images | grep rocm

# Remove unused ROCm images
docker rmi rocm/dev-ubuntu-22.04:7.0
```

## References

### Official Docker Hub
- **Main Search**: https://hub.docker.com/search?q=rocm&type=image
- **rocm/dev-ubuntu-22.04**: https://hub.docker.com/r/rocm/dev-ubuntu-22.04
- **rocm/dev-ubuntu-24.04**: https://hub.docker.com/r/rocm/dev-ubuntu-24.04
- **rocm/pytorch**: https://hub.docker.com/r/rocm/pytorch
- **rocm/tensorflow**: https://hub.docker.com/r/rocm/tensorflow

### Documentation
- **ROCm Docker Guide**: https://rocm.docs.amd.com/en/latest/deploy/docker.html
- **ROCm Installation**: https://rocm.docs.amd.com/projects/install-on-linux
- **Docker Documentation**: https://docs.docker.com/

### Community
- **GitHub Discussions**: https://github.com/ROCm/ROCm/discussions
- **GitHub Issues**: https://github.com/ROCm/ROCm/issues


