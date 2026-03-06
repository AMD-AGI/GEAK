Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "5"
category: "llm"
subcategory: "foundations"
tags: ["docker", "containers", "deployment"]
cuda_version: "13.0+"
difficulty: "intermediate"
estimated_time: "40min"
last_updated: 2025-11-17
---

# Docker and Containers for CUDA

*Complete guide to containerizing LLM applications with Docker and CUDA*

## Prerequisites

- Docker installed
- Nvidia Container Toolkit
- See [CUDA Docker Images](../../layer-2-compute-stack/cuda/cuda-docker-images.md)

## LLM Inference Dockerfile

```dockerfile
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3-pip

# Install dependencies
RUN pip3 install vllm transformers

# Copy application
COPY serve.py /app/
WORKDIR /app

# Expose API port
EXPOSE 8000

CMD ["python3", "serve.py"]
```

Build and run:
```bash
docker build -t llm-server .
docker run --gpus all -p 8000:8000 llm-server
```

## Training Dockerfile

```dockerfile
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip git

# Install training frameworks
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip3 install transformers peft datasets accelerate

WORKDIR /workspace
COPY train.py .

CMD ["python3", "train.py"]
```

## Docker Compose for LLM Stack

```yaml
version: '3.8'

services:
  llm-server:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_NAME=meta-llama/Llama-2-7b-hf
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models

  monitoring:
    image: nvidia/dcgm-exporter:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    ports:
      - "9400:9400"
```

## Best Practices

1. **Use multi-stage builds**: Smaller production images
2. **Pin versions**: Reproducible builds
3. **Cache dependencies**: Faster builds
4. **Use .dockerignore**: Exclude unnecessary files
5. **Health checks**: Monitor container health

## Related Guides

- [CUDA Docker Images](../../layer-2-compute-stack/cuda/cuda-docker-images.md)
- [Production Serving](../02-inference/deployment/production-serving.md)

