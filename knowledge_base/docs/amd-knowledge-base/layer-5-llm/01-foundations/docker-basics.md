---
layer: "5"
category: "foundations"
subcategory: "containers"
tags: ["docker", "containers", "deployment", "rocm", "infrastructure"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "40min"
---

# Docker and Containers for ROCm

Complete guide to containerizing AMD GPU applications with Docker and ROCm.

## Prerequisites

- Docker 20.10+ installed
- ROCm 7.0+ drivers on host
- AMD GPU (MI50, MI100, MI200 series)

## Installation

### Install Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
```

### Verify GPU Access

```bash
# Check ROCm on host
rocm-smi

# Test GPU access in container
docker run --rm --device=/dev/kfd --device=/dev/dri \
    rocm/pytorch:latest \
    rocm-smi
```

## ROCm Base Images

> **📖 For complete Docker image catalog**, see: [ROCm Docker Images - Complete Reference](../../layer-2-compute-stack/rocm/rocm-docker-images.md)

### Official ROCm 7.x Images

```bash
# Development images (Ubuntu 22.04 LTS)
docker pull rocm/dev-ubuntu-22.04:7.1

# Development images (Ubuntu 24.04 LTS)
docker pull rocm/dev-ubuntu-24.04:7.1

# PyTorch with ROCm (check Docker Hub for specific ROCm 7.1 tags)
docker pull rocm/pytorch:latest

# TensorFlow with ROCm (check Docker Hub for specific ROCm 7.1 tags)
docker pull rocm/tensorflow:latest

# Minimal ROCm terminal
docker pull rocm/rocm-terminal:7.1

# ⚠️ Always check Docker Hub for the latest available tags:
# https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags
# https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags
# https://hub.docker.com/r/rocm/pytorch/tags
```

### Image Tags Explained

```
rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_2.4.0
          │      │        │         │        │      └─ PyTorch version
          │      │        │         │        └─ PyTorch indicator
          │      │        │         └─ Python version
          │      │        └─ OS version
          │      └─ ROCm version
          └─ Image type
```

**⚠️ Always use ROCm 7.1 (latest) for new projects. Check Docker Hub for exact available tags.**

## Basic Docker Usage

### Running Containers

```bash
# Basic run
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    rocm/pytorch:latest \
    bash

# With shared memory for multi-processing
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    rocm/pytorch:latest \
    bash

# With GPU selection
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -e HIP_VISIBLE_DEVICES=0,1 \
    rocm/pytorch:latest \
    bash

# With port mapping for API servers
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -p 8000:8000 \
    rocm/pytorch:latest \
    bash
```

### Volume Mounting

```bash
# Mount local directory
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -v $(pwd):/workspace \
    -w /workspace \
    rocm/pytorch:latest

# Mount multiple directories
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -v $(pwd)/code:/workspace \
    -v $(pwd)/data:/data \
    -v $(pwd)/models:/models \
    rocm/pytorch:latest

# Mount with read-only
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -v $(pwd)/data:/data:ro \
    rocm/pytorch:latest
```

## Creating Custom Images

### Simple Dockerfile

```dockerfile
# Dockerfile
FROM rocm/pytorch:latest  # Or use specific ROCm 7.1 tag from Docker Hub

# Install additional packages (verify versions for ROCm 7.1 compatibility)
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    datasets \
    accelerate \
    vllm>=0.5.0

# Set working directory
WORKDIR /workspace

# Copy application code
COPY . /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HIP_VISIBLE_DEVICES=0

# Default command
CMD ["bash"]
```

Build and run:
```bash
docker build -t my-rocm-app:latest .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    --group-add video my-rocm-app:latest
```

### Multi-stage Build

```dockerfile
# Build stage
FROM rocm/dev-ubuntu-22.04:7.1 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM rocm/pytorch:latest

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . /app
WORKDIR /app

CMD ["python", "main.py"]
```

### Production-Ready Dockerfile

```dockerfile
FROM rocm/pytorch:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /workspace /data /models && \
    chown -R appuser:appuser /workspace /data /models

USER appuser
WORKDIR /workspace

# Copy application
COPY --chown=appuser:appuser . /workspace

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
ENTRYPOINT ["python"]
CMD ["main.py"]
```

## Docker Compose

### Basic Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  llm-inference:
    image: rocm/pytorch:latest
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '16gb'
    ports:
      - "8000:8000"
    volumes:
      - ./code:/workspace
      - ./models:/models
      - ./cache:/root/.cache
    environment:
      - HIP_VISIBLE_DEVICES=0,1
      - PYTHONUNBUFFERED=1
    command: python -m vllm.entrypoints.openai.api_server --model /models/llama2-7b
```

Run:
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### Multi-Service Configuration

```yaml
version: '3.8'

services:
  # Training service
  training:
    build: ./training
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '32gb'
    volumes:
      - ./data:/data
      - ./checkpoints:/checkpoints
    environment:
      - HIP_VISIBLE_DEVICES=0,1,2,3
    command: python train.py

  # Inference service
  inference:
    build: ./inference
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '16gb'
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    environment:
      - HIP_VISIBLE_DEVICES=4,5
    depends_on:
      - training

  # Monitoring
  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

## Container Orchestration

### Kubernetes Deployment

```yaml
# llm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
  labels:
    app: vllm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: my-registry/vllm-rocm:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            amd.com/gpu: 4
        env:
        - name: HIP_VISIBLE_DEVICES
          value: "0,1,2,3"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache
        - name: models
          mountPath: /models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f llm-deployment.yaml
kubectl get pods
kubectl logs -f <pod-name>
```

### StatefulSet for Training

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: distributed-training
spec:
  serviceName: training
  replicas: 4
  selector:
    matchLabels:
      app: training
  template:
    metadata:
      labels:
        app: training
    spec:
      containers:
      - name: trainer
        image: my-registry/training-rocm:latest
        resources:
          limits:
            amd.com/gpu: 8
        env:
        - name: WORLD_SIZE
          value: "4"
        - name: RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['statefulset.kubernetes.io/pod-name']
        volumeMounts:
        - name: data
          mountPath: /data
        - name: checkpoints
          mountPath: /checkpoints
  volumeClaimTemplates:
  - metadata:
      name: checkpoints
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 500Gi
```

## Best Practices

### Security

```dockerfile
# Use specific image versions (pin when known from Docker Hub)
FROM rocm/dev-ubuntu-22.04:7.1

# Or use latest with comment for clarity
FROM rocm/pytorch:latest  # ROCm 7.1 compatible

# Don't run as root
RUN useradd -m -u 1000 appuser
USER appuser

# Scan for vulnerabilities
# docker scan my-image:latest

# Use secrets for sensitive data
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --secret HF_TOKEN \
    my-image:latest
```

### Resource Management

```yaml
# docker-compose.yml with resource limits
services:
  app:
    image: my-image
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 64G
        reservations:
          cpus: '4'
          memory: 32G
```

### Caching and Performance

```dockerfile
# Layer caching optimization
FROM rocm/pytorch:latest

# Install dependencies first (changes less frequently)
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy code last (changes frequently)
COPY . /app

# Use BuildKit for better caching
# DOCKER_BUILDKIT=1 docker build .
```

### Health Checks

```dockerfile
# Application health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
```

## Troubleshooting

### GPU Not Accessible

```bash
# Check host GPU
rocm-smi

# Check devices in container
docker run --rm --device=/dev/kfd --device=/dev/dri \
    rocm/pytorch:latest \
    ls -la /dev/kfd /dev/dri

# Verify group permissions
docker run --rm --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    rocm/pytorch:latest \
    groups
```

### Out of Memory

```yaml
# Increase shared memory
services:
  app:
    shm_size: '32gb'
    ipc: host
```

### Permission Issues

```dockerfile
# Fix permissions
RUN chown -R appuser:appuser /workspace
USER appuser
```

### Network Issues

```bash
# Debug networking
docker run --rm -it --network host \
    --device=/dev/kfd --device=/dev/dri \
    my-image bash
```

## Monitoring

### Resource Usage

```bash
# Container stats
docker stats

# Detailed inspection
docker inspect <container-id>

# GPU usage inside container
docker exec <container-id> rocm-smi
```

### Logging

```bash
# View logs
docker logs <container-id>

# Follow logs
docker logs -f <container-id>

# Tail logs
docker logs --tail 100 <container-id>
```

## References

- [ROCm Docker Hub](https://hub.docker.com/u/rocm)
- [Docker Documentation](https://docs.docker.com/)
- [ROCm Containers Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
- [Kubernetes AMD GPU Device Plugin](https://github.com/RadeonOpenCompute/k8s-device-plugin)

