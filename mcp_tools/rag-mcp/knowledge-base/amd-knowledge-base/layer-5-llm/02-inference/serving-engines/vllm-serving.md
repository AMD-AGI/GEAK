---
layer: "5"
category: "inference"
subcategory: "serving-engines"
tags: ["vllm", "inference", "serving", "llm", "deployment"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
difficulty: "intermediate"
estimated_time: "30min"
---

# vLLM Deployment on AMD GPUs

vLLM is a fast and easy-to-use library for LLM inference and serving, with excellent AMD GPU support.

**This documentation targets ROCm 7.0+ only.**

**Official Repository**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)  
**Documentation**: [https://docs.vllm.ai/](https://docs.vllm.ai/)  
**Latest Release**: v0.11.0 (October 2025)

> **About vLLM**: Originally developed in the Sky Computing Lab at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry. Now a hosted project under PyTorch Foundation as of May 2025.

## Key Features

vLLM is fast with:
- **State-of-the-art serving throughput**
- **PagedAttention**: Efficient management of attention key and value memory
- **Continuous batching** of incoming requests
- **Fast model execution** with CUDA/HIP graph
- **Quantizations**: GPTQ, AWQ, AutoRound, INT4, INT8, and FP8
- **Optimized kernels**: Integration with FlashAttention and FlashInfer
- **Speculative decoding**
- **Chunked prefill**

vLLM is flexible and easy to use with:
- **Seamless integration** with popular Hugging Face models
- **High-throughput serving** with various decoding algorithms (parallel sampling, beam search)
- **Tensor, pipeline, data and expert parallelism** for distributed inference
- **Streaming outputs**
- **OpenAI-compatible API server**
- **Multi-hardware support**: AMD GPUs (ROCm), NVIDIA GPUs, Intel GPUs, TPU, and more
- **Prefix caching support**
- **Multi-LoRA support**

> **Note**: vLLM V1 alpha was announced in January 2025, featuring a major architectural upgrade with 1.7x speedup, clean code, optimized execution loop, zero-overhead prefix caching, and enhanced multimodal support.

## Installation

### Prerequisites

Before installing vLLM with ROCm support, ensure you have:
- **ROCm 7.0.0 or 7.0.2** installed and configured
- **Python 3.8-3.12** (Python 3.10 or 3.11 recommended)
- **PyTorch 2.0+** with ROCm support

### Option 1: Using pip (Recommended)

```bash
# Install vLLM with ROCm support
pip install vllm

# Verify installation
python -c "import vllm; print(vllm.__version__)"

# Test with a simple model
python -c "from vllm import LLM; llm = LLM('facebook/opt-125m'); print('vLLM is ready!')"
```

### Option 2: Building from Source

For the latest features and AMD GPU optimizations:

```bash
# Clone the official repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install build dependencies
pip install -U pip setuptools wheel
pip install -r requirements-build.txt

# Build and install with ROCm support
# The build will automatically detect ROCm
pip install -e .

# Or use the build script for more control
python setup.py build_ext --inplace
pip install -e .
```

#### Building with Specific ROCm Architecture

```bash
# Set GPU architecture for optimized builds
export PYTORCH_ROCM_ARCH="gfx90a;gfx942"  # For MI200/MI300 series

# Build with specific features
export VLLM_BUILD_WITH_TRITON=1  # Enable Triton support
pip install -e .
```

### Option 3: Using Docker (Recommended for Production)

```bash
# Pull official vLLM ROCm image
docker pull vllm/vllm-openai:latest

# For ROCm-specific image (if available)
docker pull vllm/vllm-rocm:latest

# Run container with OpenAI-compatible API
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-hf

# Run with specific GPU selection
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    -e HIP_VISIBLE_DEVICES=0,1 \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-13b-hf \
    --tensor-parallel-size 2
```

### Option 4: Building Custom Docker Image

```dockerfile
# Dockerfile for custom vLLM with ROCm
FROM rocm/pytorch:rocm7.0.2_ubuntu22.04_py3.10_pytorch_latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    git python3-pip cmake \
    && rm -rf /var/lib/apt/lists/*

# Clone and install vLLM
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV HIP_VISIBLE_DEVICES=0
ENV VLLM_USE_ROCM=1

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
```

Build and run:
```bash
docker build -t vllm-rocm:custom .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host -p 8000:8000 \
    vllm-rocm:custom --model facebook/opt-125m
```

## Basic Usage

### Simple Inference

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # Number of GPUs
    trust_remote_code=True
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# Generate
prompts = [
    "The future of AI is",
    "Once upon a time"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 80)
```

### Batch Processing

```python
# Process large batch of prompts
prompts = [f"Question {i}: What is AI?" for i in range(100)]

# vLLM automatically handles batching for optimal throughput
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Response {i}: {output.outputs[0].text[:100]}...")
```

## OpenAI-Compatible API Server

### Start Server

```bash
# Basic server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16

# With tensor parallelism (multi-GPU)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --dtype bfloat16

# With specific GPU selection
HIP_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4
```

### Client Usage

```python
from openai import OpenAI

# Connect to vLLM server
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)

# Streaming response
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

## Advanced Configuration

### Memory Optimization

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    # GPU memory utilization (0.0 - 1.0)
    gpu_memory_utilization=0.95,
    # Enable/disable KV cache swap
    swap_space=4,  # GB of CPU swap space
    # Quantization
    quantization="awq",  # or "gptq", "squeezellm"
    dtype="float16"
)
```

### PagedAttention Configuration

```python
# PagedAttention is enabled by default in vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    # Block size for PagedAttention
    block_size=16,
    # Maximum number of sequences in a batch
    max_num_seqs=256,
    # Maximum number of batched tokens
    max_num_batched_tokens=4096
)
```

### Multi-GPU Deployment

```python
# Tensor Parallelism (split model across GPUs)
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=8,  # Use 8 GPUs
    pipeline_parallel_size=1  # No pipeline parallelism
)

# Pipeline Parallelism (layers across GPUs)
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=2,
    pipeline_parallel_size=4  # 2x4 = 8 GPUs total
)
```

## Production Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-rocm:latest
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ipc: host
    shm_size: '32gb'
    ports:
      - "8000:8000"
    environment:
      - HIP_VISIBLE_DEVICES=0,1,2,3
    volumes:
      - ./models:/models
      - ./cache:/root/.cache
    command: >
      --model /models/Llama-2-70b-hf
      --tensor-parallel-size 4
      --dtype bfloat16
      --gpu-memory-utilization 0.95
      --max-model-len 4096
```

### Kubernetes Deployment

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
spec:
  replicas: 1
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
        image: vllm/vllm-rocm:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            amd.com/gpu: 4
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        - --model
        - meta-llama/Llama-2-70b-hf
        - --tensor-parallel-size
        - "4"
        - --dtype
        - bfloat16
        env:
        - name: HIP_VISIBLE_DEVICES
          value: "0,1,2,3"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
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

## Performance Tuning

### Benchmarking

```bash
# Benchmark throughput
python benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 128 \
    --output-len 512 \
    --num-prompts 1000 \
    --tensor-parallel-size 1

# Benchmark latency
python benchmarks/benchmark_latency.py \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 128 \
    --output-len 512
```

### Profiling

```python
import torch
from vllm import LLM, SamplingParams

# Enable profiling
torch.cuda.set_device(0)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=100))

prof.export_chrome_trace("vllm_trace.json")
```

### Optimization Tips

1. **Use BF16 on CDNA2+**: Better performance than FP16
   ```bash
   --dtype bfloat16
   ```

2. **Adjust GPU Memory Utilization**: Start with 0.90 and increase
   ```bash
   --gpu-memory-utilization 0.95
   ```

3. **Tune Batch Size**: Balance latency and throughput
   ```bash
   --max-num-seqs 256
   --max-num-batched-tokens 8192
   ```

4. **Enable Quantization**: For larger models
   ```bash
   --quantization awq
   ```

5. **Use Tensor Parallelism**: For multi-GPU setups
   ```bash
   --tensor-parallel-size 4
   ```

## Monitoring

### Metrics Endpoint

```python
import requests

# Get metrics from vLLM server
response = requests.get("http://localhost:8000/metrics")
print(response.text)
```

### Custom Monitoring

```python
from vllm import LLM
import time

class MonitoredLLM:
    def __init__(self, *args, **kwargs):
        self.llm = LLM(*args, **kwargs)
        self.total_requests = 0
        self.total_tokens = 0
        
    def generate(self, prompts, sampling_params):
        start = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = time.time() - start
        
        self.total_requests += len(prompts)
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        self.total_tokens += tokens
        
        print(f"Throughput: {tokens/elapsed:.2f} tokens/s")
        return outputs
```

## Troubleshooting

### Out of Memory

```bash
# Reduce memory utilization
--gpu-memory-utilization 0.85

# Enable CPU swap
--swap-space 8

# Use quantization
--quantization awq
```

### Slow Performance

```bash
# Check GPU utilization
watch -n 1 rocm-smi

# Profile with rocprof
rocprof --hip-trace python -m vllm.entrypoints.openai.api_server ...

# Adjust batch sizes
--max-num-seqs 128
```

### Model Loading Issues

```python
# Verify model download
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# If this works, vLLM should work too
```

## Supported Models

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- **Transformer-like LLMs**: Llama, Mistral, Qwen, Phi, Gemma, etc.
- **Mixture-of-Expert LLMs**: Mixtral, Deepseek-V2 and V3, Qwen-MoE
- **Embedding Models**: E5-Mistral, BGE-M3
- **Multi-modal LLMs**: LLaVA, Qwen-VL, InternVL, Pixtral

Find the full list of supported models at [vLLM Model Support](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Community and Support

vLLM is a community-driven project hosted under PyTorch Foundation with contributions from:

**Sponsors (Compute Resources)**:
- AMD, NVIDIA, Intel
- Cloud providers: AWS, Google Cloud, Alibaba Cloud, Nebius
- AI platforms: Anyscale, Databricks, Replicate, RunPod

**Communication Channels**:
- **GitHub Issues**: Technical questions and feature requests
- **vLLM Forum**: User discussions at [discuss.vllm.ai](https://discuss.vllm.ai)
- **Slack**: Developer coordination at [slack.vllm.ai](https://slack.vllm.ai)
- **Email**: Collaborations at vllm-questions@lists.berkeley.edu

## References

### Official Resources

- **[vLLM GitHub Repository](https://github.com/vllm-project/vllm)** - Official source code (61.9k+ stars)
- **[vLLM Documentation](https://docs.vllm.ai/)** - Complete user guide and API reference
- **[vLLM Blog](https://blog.vllm.ai/)** - Updates and technical deep-dives
- **[vLLM Forum](https://discuss.vllm.ai/)** - Community discussions

### Installation & Setup

- **[AMD ROCm Installation Guide](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html)** - ROCm-specific setup
- **[Quickstart Guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)** - Get started quickly
- **[Model Support List](https://docs.vllm.ai/en/latest/models/supported_models.html)** - Supported models

### Advanced Topics

- **[Distributed Inference](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)** - Multi-GPU deployment
- **[Quantization Guide](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)** - GPTQ, AWQ, FP8
- **[Performance Tuning](https://docs.vllm.ai/en/latest/serving/performance_tuning.html)** - Optimization tips

### Research & Publications

- **[PagedAttention Paper (SOSP 2023)](https://arxiv.org/abs/2309.06180)** - Original research paper
- **[vLLM V1 Blog Post](https://blog.vllm.ai/2025/01/13/v1-alpha.html)** - V1 architecture announcement

### Related Guides

- [PyTorch with ROCm](../../layer-4-frameworks/pytorch/pytorch-rocm-basics.md)
- [Production Serving](../deployment/production-serving.md)
- [Docker Deployment](../deployment/docker-deployment.md)
- [GPU Optimization Best Practices](../../../best-practices/performance/gpu-optimization.md)

