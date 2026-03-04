---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "inference"
subcategory: "deployment"
tags: ["production", "serving", "scalability", "monitoring", "operations"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
difficulty: "advanced"
estimated_time: "60min"
---

# Production LLM Serving on AMD GPUs

Comprehensive guide to deploying and operating LLM inference at production scale.

## Architecture Overview

### Single-Server Setup

```
┌──────────────────────────────────────┐
│  Load Balancer (nginx)               │
└──────────┬───────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼────┐
│ vLLM-1 │   │ vLLM-2 │
│ GPU0-1 │   │ GPU2-3 │
└────────┘   └────────┘
```

### Multi-Server Setup

```
┌─────────────────────────────────────────┐
│    Global Load Balancer                 │
└──────────┬──────────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────────┐  ┌─▼──────────┐
│  Server 1  │  │  Server 2  │
│  8x MI250X │  │  8x MI250X │
│            │  │            │
│  vLLM Pod  │  │  vLLM Pod  │
└────────────┘  └────────────┘
```

## Infrastructure Setup

### Server Requirements

**Minimum (7B-13B models):**
- 1-2x AMD MI250X or MI300X
- 128GB+ RAM
- 1TB NVMe SSD
- 10Gbps network

**Recommended (70B+ models):**
- 4-8x AMD MI250X or MI300X
- 512GB+ RAM
- 2TB+ NVMe SSD
- 100Gbps network

### Network Configuration

```bash
# /etc/sysctl.conf
# Increase network buffers
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864

# Enable TCP BBR
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr

# Connection tuning
net.ipv4.tcp_max_syn_backlog = 8192
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000

# Apply settings
sudo sysctl -p
```

### Storage Configuration

```bash
# Mount models on fast NVMe
sudo mkdir -p /data/models
sudo mount /dev/nvme0n1 /data/models

# Configure caching
export HF_HOME=/data/models/cache
export TRANSFORMERS_CACHE=/data/models/cache

# Optimize filesystem
sudo tune2fs -O fast_commit /dev/nvme0n1
```

## Service Configuration

### vLLM Production Configuration

```python
# server_config.py
import os
from vllm import AsyncLLMEngine, AsyncEngineArgs

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.95

# Performance tuning
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 8192
BLOCK_SIZE = 16

# Engine arguments
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    dtype="bfloat16",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_num_seqs=MAX_NUM_SEQS,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    block_size=BLOCK_SIZE,
    trust_remote_code=True,
    # Enable metrics
    disable_log_stats=False,
)

# Create engine
engine = AsyncLLMEngine.from_engine_args(engine_args)
```

### API Server with FastAPI

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from vllm import AsyncLLMEngine, SamplingParams
from prometheus_client import Counter, Histogram, make_asgi_app
import time

app = FastAPI(title="LLM Inference API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter('llm_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('llm_request_latency_seconds', 'Request latency')
TOKENS_GENERATED = Counter('llm_tokens_generated_total', 'Total tokens generated')

# Request models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    tokens: int
    latency: float

# Initialize engine
from server_config import engine

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt"""
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        
        # Generate
        results = await engine.generate(
            request.prompt,
            sampling_params,
            request_id=str(time.time())
        )
        
        output = results.outputs[0]
        text = output.text
        tokens = len(output.token_ids)
        
        latency = time.time() - start_time
        
        # Update metrics
        TOKENS_GENERATED.inc(tokens)
        REQUEST_LATENCY.observe(latency)
        
        return GenerateResponse(
            text=text,
            tokens=tokens,
            latency=latency
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": engine.engine.model_config.model}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return make_asgi_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info",
        access_log=True
    )
```

### Systemd Service

```ini
# /etc/systemd/system/vllm-inference.service
[Unit]
Description=vLLM Inference Service
After=network.target

[Service]
Type=simple
User=llmuser
Group=llmuser
WorkingDirectory=/opt/vllm-server
Environment="PATH=/opt/vllm-server/venv/bin"
Environment="HIP_VISIBLE_DEVICES=0,1,2,3"
Environment="HF_HOME=/data/models/cache"
ExecStart=/opt/vllm-server/venv/bin/python api_server.py
Restart=on-failure
RestartSec=10s
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65535
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm-inference
sudo systemctl start vllm-inference
sudo systemctl status vllm-inference

# View logs
sudo journalctl -u vllm-inference -f
```

## Load Balancing

### Nginx Configuration

```nginx
# /etc/nginx/nginx.conf
user www-data;
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 10000;
    use epoll;
    multi_accept on;
}

http {
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
    
    # Upstream servers
    upstream vllm_backend {
        least_conn;
        
        server 127.0.0.1:8001 max_fails=3 fail_timeout=30s weight=1;
        server 127.0.0.1:8002 max_fails=3 fail_timeout=30s weight=1;
        server 127.0.0.1:8003 max_fails=3 fail_timeout=30s weight=1;
        server 127.0.0.1:8004 max_fails=3 fail_timeout=30s weight=1;
        
        keepalive 32;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
    
    server {
        listen 80;
        server_name api.example.com;
        
        # Rate limiting
        limit_req zone=api_limit burst=20 nodelay;
        limit_conn conn_limit 10;
        
        # API endpoints
        location /v1/ {
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts for long requests
            proxy_connect_timeout 60s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
            
            # Buffering
            proxy_buffering off;
            proxy_request_buffering off;
        }
        
        # Health check
        location /health {
            access_log off;
            proxy_pass http://vllm_backend/health;
            proxy_connect_timeout 5s;
            proxy_read_timeout 5s;
        }
        
        # Metrics (restrict access)
        location /metrics {
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://vllm_backend/metrics;
        }
    }
}
```

### HAProxy Alternative

```haproxy
# /etc/haproxy/haproxy.cfg
global
    log /dev/log local0
    maxconn 50000
    daemon

defaults
    mode http
    log global
    option httplog
    option dontlognull
    timeout connect 10s
    timeout client 600s
    timeout server 600s

frontend llm_api
    bind *:80
    default_backend vllm_servers
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny if { sc_http_req_rate(0) gt 100 }

backend vllm_servers
    balance leastconn
    option httpchk GET /health
    
    server vllm1 127.0.0.1:8001 check inter 10s fall 3 rise 2
    server vllm2 127.0.0.1:8002 check inter 10s fall 3 rise 2
    server vllm3 127.0.0.1:8003 check inter 10s fall 3 rise 2
    server vllm4 127.0.0.1:8004 check inter 10s fall 3 rise 2

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
```

## Monitoring

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Token metrics
TOKENS_GENERATED = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated'
)

TOKENS_PER_SECOND = Gauge(
    'llm_tokens_per_second',
    'Current tokens per second throughput'
)

# GPU metrics
GPU_UTILIZATION = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

GPU_MEMORY_USED = Gauge(
    'llm_gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

# Model info
MODEL_INFO = Info(
    'llm_model',
    'Model information'
)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "LLM Inference Monitoring",
    "panels": [
      {
        "title": "Requests Per Second",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Request Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(llm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(llm_request_latency_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Tokens Per Second",
        "targets": [
          {
            "expr": "llm_tokens_per_second"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "llm_gpu_utilization_percent"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
  - name: llm_alerts
    interval: 30s
    rules:
      # High latency alert
      - alert: HighRequestLatency
        expr: histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency detected"
          description: "P95 latency is {{ $value }}s"
      
      # Low throughput alert
      - alert: LowThroughput
        expr: rate(llm_requests_total[5m]) < 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low request throughput"
          description: "Request rate is {{ $value }} req/s"
      
      # GPU memory alert
      - alert: HighGPUMemory
        expr: llm_gpu_memory_used_bytes / llm_gpu_memory_total_bytes > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage high"
          description: "GPU {{ $labels.gpu_id }} memory at {{ $value }}%"
      
      # Error rate alert
      - alert: HighErrorRate
        expr: rate(llm_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}"
```

## Auto-Scaling

### Horizontal Pod Autoscaler (Kubernetes)

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: llm_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

### Custom Auto-Scaler Script

```python
# autoscaler.py
import time
import subprocess
from prometheus_api_client import PrometheusConnect

prom = PrometheusConnect(url="http://localhost:9090")

MIN_INSTANCES = 2
MAX_INSTANCES = 8
TARGET_LATENCY_P95 = 5.0  # seconds
SCALE_UP_THRESHOLD = 7.0
SCALE_DOWN_THRESHOLD = 3.0

def get_current_instances():
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=vllm", "-q"],
        capture_output=True, text=True
    )
    return len(result.stdout.strip().split('\n'))

def scale_up():
    current = get_current_instances()
    if current < MAX_INSTANCES:
        new_instance = current + 1
        subprocess.run([
            "docker", "run", "-d",
            "--name", f"vllm-{new_instance}",
            "--device=/dev/kfd", "--device=/dev/dri",
            "vllm-server:latest"
        ])
        print(f"Scaled up to {new_instance} instances")

def scale_down():
    current = get_current_instances()
    if current > MIN_INSTANCES:
        subprocess.run(["docker", "stop", f"vllm-{current}"])
        subprocess.run(["docker", "rm", f"vllm-{current}"])
        print(f"Scaled down to {current-1} instances")

def main():
    while True:
        # Get P95 latency
        query = 'histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m]))'
        result = prom.custom_query(query)
        
        if result:
            p95_latency = float(result[0]['value'][1])
            print(f"Current P95 latency: {p95_latency}s")
            
            if p95_latency > SCALE_UP_THRESHOLD:
                scale_up()
            elif p95_latency < SCALE_DOWN_THRESHOLD:
                scale_down()
        
        time.sleep(60)

if __name__ == "__main__":
    main()
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/llm-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup model cache
rsync -av /data/models/cache/ $BACKUP_DIR/cache/

# Backup configuration
cp -r /opt/vllm-server/config $BACKUP_DIR/

# Backup monitoring data
docker exec prometheus tar czf - /prometheus | cat > $BACKUP_DIR/prometheus.tar.gz

# Upload to S3
aws s3 sync $BACKUP_DIR s3://my-bucket/backups/llm-$(date +%Y%m%d)/

echo "Backup complete: $BACKUP_DIR"
```

### Recovery Procedures

```bash
#!/bin/bash
# recover.sh

BACKUP_DATE=$1
BACKUP_DIR="/backups/llm-$BACKUP_DATE"

# Download from S3
aws s3 sync s3://my-bucket/backups/llm-$BACKUP_DATE/ $BACKUP_DIR/

# Restore model cache
rsync -av $BACKUP_DIR/cache/ /data/models/cache/

# Restore configuration
cp -r $BACKUP_DIR/config /opt/vllm-server/

# Restart services
systemctl restart vllm-inference

echo "Recovery complete from $BACKUP_DATE"
```

## Performance Optimization

### Kernel Tuning

```bash
# /etc/sysctl.conf

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File descriptors
fs.file-max = 2097152
fs.nr_open = 2097152

# Network optimization
net.core.somaxconn = 65535
net.ipv4.tcp_max_tw_buckets = 1440000

# Apply
sudo sysctl -p
```

### CPU Affinity

```bash
# Pin processes to specific CPUs
taskset -c 0-15 python api_server.py
```

## References

- [Production Best Practices](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)
- [Nginx Performance Tuning](https://www.nginx.com/blog/tuning-nginx/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Kubernetes Production Patterns](https://kubernetes.io/docs/concepts/cluster-administration/)

