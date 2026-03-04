---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "inference"
subcategory: "deployment"
tags: ["docker", "deployment", "production", "containers", "vllm"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "intermediate"
estimated_time: "45min"
---

# Docker Deployment for LLM Inference

Production-ready Docker deployment patterns for serving LLMs on AMD GPUs.

## Quick Start

### Simple vLLM Deployment

```dockerfile
# Dockerfile.vllm
FROM rocm/pytorch:latest  # Check Docker Hub for specific ROCm 7.1 PyTorch tags

# Install vLLM (verify version for ROCm 7.1 compatibility)
RUN pip install --no-cache-dir vllm>=0.5.0

# Set working directory
WORKDIR /app

# Environment variables
ENV HIP_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--model", "meta-llama/Llama-2-7b-chat-hf", \
     "--dtype", "bfloat16"]
```

> **📖 For complete Docker image reference**, see: [ROCm Docker Images Guide](../../layer-2-compute-stack/rocm/rocm-docker-images.md)

Build and run:
```bash
docker build -f Dockerfile.vllm -t vllm-server:latest .

docker run -d --name vllm-api \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host --shm-size 16G \
    -p 8000:8000 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    vllm-server:latest
```

## Production Dockerfile

### Multi-Stage Build

```dockerfile
# Stage 1: Builder
FROM rocm/dev-ubuntu-22.04:7.1 AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime
FROM rocm/pytorch:latest

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash llmuser && \
    mkdir -p /app /models /cache && \
    chown -R llmuser:llmuser /app /models /cache

USER llmuser
WORKDIR /app

# Copy application code
COPY --chown=llmuser:llmuser app/ /app/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache
ENV TRANSFORMERS_CACHE=/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Entrypoint
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16"]
```

Requirements file:
```txt
# requirements.txt for ROCm 7.1
# Verify package versions for ROCm 7.1 compatibility
vllm>=0.5.0
transformers>=4.40.0
accelerate>=0.30.0
sentencepiece>=0.1.99
protobuf>=3.20.0
fastapi>=0.104.0
uvicorn>=0.24.0
```

## Docker Compose Configurations

### Single Service

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm:
    build:
      context: .
      dockerfile: Dockerfile.vllm
    container_name: vllm-inference
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '32gb'
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
      - cache:/cache
    environment:
      - HIP_VISIBLE_DEVICES=0,1,2,3
      - MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
      - TENSOR_PARALLEL_SIZE=4
      - MAX_MODEL_LEN=4096
    command: >
      --model ${MODEL_NAME}
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
      --max-model-len ${MAX_MODEL_LEN}
      --dtype bfloat16
      --gpu-memory-utilization 0.95
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  cache:
    driver: local
```

Run:
```bash
docker-compose up -d
docker-compose logs -f vllm
docker-compose ps
docker-compose down
```

### Load Balanced Setup

```yaml
# docker-compose.loadbalanced.yml
version: '3.8'

services:
  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - vllm-1
      - vllm-2
    restart: unless-stopped

  # vLLM instance 1
  vllm-1:
    build: .
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '16gb'
    environment:
      - HIP_VISIBLE_DEVICES=0,1
      - MODEL_NAME=meta-llama/Llama-2-13b-chat-hf
      - TENSOR_PARALLEL_SIZE=2
    volumes:
      - cache:/cache
    command: >
      --model ${MODEL_NAME}
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
      --dtype bfloat16
    restart: unless-stopped

  # vLLM instance 2
  vllm-2:
    build: .
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '16gb'
    environment:
      - HIP_VISIBLE_DEVICES=2,3
      - MODEL_NAME=meta-llama/Llama-2-13b-chat-hf
      - TENSOR_PARALLEL_SIZE=2
    volumes:
      - cache:/cache
    command: >
      --model ${MODEL_NAME}
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
      --dtype bfloat16
    restart: unless-stopped

volumes:
  cache:
```

Nginx configuration:
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream vllm_backend {
        least_conn;
        server vllm-1:8000 max_fails=3 fail_timeout=30s;
        server vllm-2:8000 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://vllm_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Timeouts for long-running requests
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
        
        location /health {
            access_log off;
            proxy_pass http://vllm_backend/health;
        }
    }
}
```

### Multi-Model Serving

```yaml
# docker-compose.multi-model.yml
version: '3.8'

services:
  # Small model for fast inference
  vllm-small:
    build: .
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '8gb'
    ports:
      - "8001:8000"
    environment:
      - HIP_VISIBLE_DEVICES=0
      - MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
    volumes:
      - cache:/cache
    command: >
      --model ${MODEL_NAME}
      --dtype bfloat16
      --max-model-len 2048

  # Large model for complex tasks
  vllm-large:
    build: .
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '32gb'
    ports:
      - "8002:8000"
    environment:
      - HIP_VISIBLE_DEVICES=1,2,3,4
      - MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
      - TENSOR_PARALLEL_SIZE=4
    volumes:
      - cache:/cache
    command: >
      --model ${MODEL_NAME}
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
      --dtype bfloat16
      --max-model-len 4096

  # Router service
  router:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./router.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - vllm-small
      - vllm-large

volumes:
  cache:
```

## Advanced Deployment Patterns

### Auto-Scaling with Docker Swarm

```yaml
# docker-stack.yml
version: '3.8'

services:
  vllm:
    image: my-registry/vllm-server:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        max_attempts: 3
      placement:
        constraints:
          - node.labels.gpu==true
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=meta-llama/Llama-2-13b-chat-hf
    volumes:
      - models:/models
      - cache:/cache

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    deploy:
      replicas: 2
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro

volumes:
  models:
  cache:
```

Deploy:
```bash
docker swarm init
docker stack deploy -c docker-stack.yml llm-stack
docker stack ps llm-stack
docker stack rm llm-stack
```

### Blue-Green Deployment

```bash
#!/bin/bash
# deploy.sh

# Build new version
docker build -t vllm-server:green .

# Start green deployment
docker run -d --name vllm-green \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -p 8001:8000 \
    vllm-server:green

# Health check
sleep 30
if curl -f http://localhost:8001/health; then
    # Switch traffic (update load balancer config)
    echo "Green deployment healthy, switching traffic"
    
    # Stop blue deployment
    docker stop vllm-blue
    docker rm vllm-blue
    
    # Rename green to blue
    docker rename vllm-green vllm-blue
    
    echo "Deployment complete"
else
    echo "Green deployment failed health check"
    docker stop vllm-green
    docker rm vllm-green
    exit 1
fi
```

### Canary Deployment

```yaml
# docker-compose.canary.yml
version: '3.8'

services:
  # Stable version (90% traffic)
  vllm-stable:
    image: vllm-server:v1.0
    deploy:
      replicas: 9
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    # ... other config

  # Canary version (10% traffic)
  vllm-canary:
    image: vllm-server:v1.1
    deploy:
      replicas: 1
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    # ... other config

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./canary-nginx.conf:/etc/nginx/nginx.conf:ro
```

## Monitoring and Observability

### Prometheus Metrics

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  vllm:
    # ... vllm config
    ports:
      - "8000:8000"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus-data:
  grafana-data:
```

Prometheus config:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm:8000']
```

### Logging with ELK

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  vllm:
    # ... vllm config
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
        labels: "service"
    labels:
      service: "vllm"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - es-data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  es-data:
```

## Security Best Practices

### Secure Dockerfile

```dockerfile
FROM rocm/pytorch:latest

# Run as non-root
RUN useradd -m -u 1000 llmuser

# Install with specific versions (verify for ROCm 7.1 compatibility)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Set ownership
RUN mkdir -p /app /cache && \
    chown -R llmuser:llmuser /app /cache

USER llmuser
WORKDIR /app

# Read-only root filesystem
COPY --chown=llmuser:llmuser . /app

# Drop capabilities
USER llmuser

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", "--dtype", "bfloat16"]
```

### Using Secrets

```yaml
# docker-compose.secrets.yml
version: '3.8'

services:
  vllm:
    image: vllm-server:latest
    secrets:
      - hf_token
      - api_key
    environment:
      - HF_TOKEN_FILE=/run/secrets/hf_token
      - API_KEY_FILE=/run/secrets/api_key

secrets:
  hf_token:
    file: ./secrets/hf_token.txt
  api_key:
    file: ./secrets/api_key.txt
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs vllm-api

# Interactive debugging
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    vllm-server:latest bash

# Test GPU access
docker exec vllm-api rocm-smi
```

### Performance Issues

```bash
# Check resource usage
docker stats vllm-api

# Check GPU utilization
docker exec vllm-api watch -n 1 rocm-smi

# Profile memory
docker exec vllm-api python -c "import torch; print(torch.cuda.memory_summary())"
```

### Network Issues

```bash
# Test connectivity
docker exec vllm-api curl http://localhost:8000/health

# Check port binding
docker port vllm-api

# Network inspection
docker network inspect bridge
```

## References

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [ROCm Docker Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)

