---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "5"
category: "inference"
subcategory: "serving-engines"
tags: ["sglang", "inference", "structured-generation", "llm", "deployment", "vlm"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
difficulty: "intermediate"
estimated_time: "30min"
---

# SGLang on AMD GPUs

SGLang (Structured Generation Language) is a fast serving framework for large language models and vision-language models.

**This documentation targets ROCm 7.0+ only.**

**Official Repository**: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)  
**Documentation**: [https://docs.sglang.ai/](https://docs.sglang.ai/)  
**Latest Release**: v0.5.4 (October 2025)

> **About SGLang**: SGLang is a high-performance serving framework designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters. Hosted under the non-profit open-source organization LMSYS and part of the PyTorch ecosystem as of March 2025.

## Key Features

**Fast Backend Runtime:**
- **RadixAttention** for prefix caching
- **Zero-overhead CPU scheduler**
- **Prefill-decode disaggregation**
- **Speculative decoding**
- **Continuous batching**
- **Paged attention**
- **Tensor/pipeline/expert/data parallelism**
- **Structured outputs**
- **Chunked prefill**
- **Quantization**: FP4, FP8, INT4, AWQ, GPTQ
- **Multi-LoRA batching**

**Extensive Model Support:**
- **Generative Models**: Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.
- **Embedding Models**: e5-mistral, gte, mcdse
- **Reward Models**: Skywork
- **Vision-Language Models**: LLaVA, LLaVA-OneVision, Qwen-VL, etc.
- Compatible with most Hugging Face models and OpenAI APIs

**Extensive Hardware Support:**
- **NVIDIA GPUs**: GB200, B300, H100, A100, Spark
- **AMD GPUs**: MI355, MI300
- **Intel Xeon CPUs**
- **Google TPUs** (native support via SGLang-Jax backend)
- **Ascend NPUs**

**Active Community:**
- Open-source with 19.7k+ GitHub stars
- Deployed at scale, powering over **300,000 GPUs worldwide**
- Generating **trillions of tokens daily** in production
- Trusted by xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, and many others

## Installation

### Prerequisites

Before installing SGLang with ROCm support, ensure you have:
- **ROCm 7.0.0 or 7.0.2** installed and configured
- **Python 3.8-3.12** (Python 3.10 or 3.11 recommended)
- **PyTorch 2.0+** with ROCm support

### Option 1: Using pip (Recommended)

```bash
# Install SGLang with all features
pip install "sglang[all]"

# Or install core package only
pip install sglang

# Verify installation
python -c "import sglang; print(sglang.__version__)"
```

### Option 2: Building from Source

For the latest features and AMD GPU optimizations:

```bash
# Clone the official repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install in editable mode with all dependencies
pip install -e "python[all]"

# Or install core package only
pip install -e python
```

### Option 3: Using Docker

```bash
# Pull official SGLang image (if available)
docker pull sglang/sglang:latest

# Run container with AMD GPU access
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 16G \
    -p 30000:30000 \
    sglang/sglang:latest

# For custom builds, see Dockerfile section below
```

## Basic Usage

### Simple Generation

```python
import sglang as sgl

# Initialize runtime
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    tp_size=1  # Tensor parallel size
)

# Set default runtime
sgl.set_default_backend(runtime)

# Simple generation
@sgl.function
def generate_story(s, topic):
    s += f"Tell me a story about {topic}.\n"
    s += sgl.gen("story", max_tokens=512)

# Run
state = generate_story.run(topic="a robot")
print(state["story"])

# Cleanup
runtime.shutdown()
```

### Structured Generation

```python
@sgl.function
def structured_qa(s, question):
    s += f"Question: {question}\n"
    s += "Let's think step by step.\n"
    s += sgl.gen("reasoning", max_tokens=256)
    s += "\nFinal Answer: "
    s += sgl.gen("answer", max_tokens=50, stop="\n")

state = structured_qa.run(
    question="What is the capital of France?"
)

print("Reasoning:", state["reasoning"])
print("Answer:", state["answer"])
```

### Function Calling

```python
@sgl.function
def tool_use(s, task):
    s += f"Task: {task}\n"
    s += "Available tools:\n"
    s += "1. calculator(expression)\n"
    s += "2. search(query)\n"
    s += "Which tool should be used? "
    s += sgl.gen("tool", max_tokens=20, stop="\n")
    s += "\nTool arguments: "
    s += sgl.gen("arguments", max_tokens=50, stop="\n")

state = tool_use.run(
    task="What is 25 * 37?"
)

print(f"Tool: {state['tool']}")
print(f"Arguments: {state['arguments']}")
```

## Server Mode

### Start Server

```bash
# Basic server
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --port 30000

# Multi-GPU
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-hf \
    --tp-size 4 \
    --port 30000

# With specific GPUs
HIP_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-hf \
    --tp-size 4
```

### Client Usage

```python
import requests

# Send request to server
url = "http://localhost:30000/generate"
data = {
    "text": "Once upon a time",
    "sampling_params": {
        "temperature": 0.8,
        "max_new_tokens": 512
    }
}

response = requests.post(url, json=data)
print(response.json()["text"])
```

### OpenAI-Compatible API

```python
from openai import OpenAI

# Connect to SGLang server
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:30000/v1"
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Advanced Features

### Constrained Generation

```python
@sgl.function
def json_generation(s):
    s += "Generate a person profile in JSON:\n"
    s += "{\n"
    s += '  "name": "'
    s += sgl.gen("name", max_tokens=20, stop='"')
    s += '",\n'
    s += '  "age": '
    s += sgl.gen("age", max_tokens=3, regex=r"\d+")
    s += ',\n'
    s += '  "city": "'
    s += sgl.gen("city", max_tokens=20, stop='"')
    s += '"\n}'

state = json_generation.run()
print(state.text())
```

### Batch Processing

```python
@sgl.function
def batch_qa(s, question):
    s += f"Q: {question}\nA: "
    s += sgl.gen("answer", max_tokens=100)

# Process batch
questions = [
    "What is AI?",
    "Explain machine learning",
    "What is deep learning?"
]

states = batch_qa.run_batch([{"question": q} for q in questions])

for i, state in enumerate(states):
    print(f"Q{i+1}: {questions[i]}")
    print(f"A{i+1}: {state['answer']}\n")
```

### Streaming

```python
@sgl.function
def streaming_gen(s, prompt):
    s += prompt
    s += sgl.gen("output", max_tokens=512)

# Stream tokens
for token in streaming_gen.run_stream(prompt="The future of AI is"):
    print(token, end='', flush=True)
```

## RadixAttention

SGLang uses RadixAttention for efficient prefix caching:

```python
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    # RadixAttention is enabled by default
    enable_cache=True,
    cache_size=8192  # Cache size in tokens
)
```

## Multi-GPU Configuration

```python
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-70b-hf",
    tp_size=8,  # Tensor parallelism
    # Memory fraction
    mem_fraction_static=0.8,
    # TP communication backend
    nccl_init_addr="127.0.0.1:28765"
)
```

## Production Deployment

### Docker

```dockerfile
FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_2.1.1

# Install SGLang
RUN pip install "sglang[all]"

# Expose port
EXPOSE 30000

# Start server
CMD ["python", "-m", "sglang.launch_server", \
     "--model-path", "meta-llama/Llama-2-7b-hf", \
     "--host", "0.0.0.0", \
     "--port", "30000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sglang
  template:
    metadata:
      labels:
        app: sglang
    spec:
      containers:
      - name: sglang
        image: sglang-rocm:latest
        ports:
        - containerPort: 30000
        resources:
          limits:
            amd.com/gpu: 4
        env:
        - name: HIP_VISIBLE_DEVICES
          value: "0,1,2,3"
        command:
        - python
        - -m
        - sglang.launch_server
        - --model-path
        - meta-llama/Llama-2-70b-hf
        - --tp-size
        - "4"
        - --host
        - "0.0.0.0"
```

## Performance Optimization

### Benchmarking

```python
import time
from sglang import Runtime, function, gen

runtime = Runtime(model_path="meta-llama/Llama-2-7b-hf")

@function
def bench_gen(s, prompt):
    s += prompt
    s += gen("out", max_tokens=512)

prompts = ["Hello world"] * 100

start = time.time()
states = bench_gen.run_batch([{"prompt": p} for p in prompts])
elapsed = time.time() - start

total_tokens = sum(len(s["out"].split()) for s in states)
print(f"Throughput: {total_tokens/elapsed:.2f} tokens/s")
```

### Memory Optimization

```python
runtime = Runtime(
    model_path="meta-llama/Llama-2-70b-hf",
    tp_size=4,
    # Adjust memory fractions
    mem_fraction_static=0.85,  # For weights
    # Enable CPU offloading
    load_balance_method="round_robin"
)
```

## Comparison: SGLang vs vLLM

| Feature | SGLang | vLLM |
|---------|--------|------|
| Structured Generation | ✅ Native | ⚠️ Limited |
| RadixAttention | ✅ Yes | ❌ No |
| PagedAttention | ✅ Yes | ✅ Yes |
| Prefill-Decode Disaggregation | ✅ Yes | ⚠️ Limited |
| Vision-Language Models | ✅ Native | ✅ Native |
| Batch Performance | Excellent | Excellent |
| Programming Model | Python DSL + API | Python API |
| Use Case | Complex prompts + VLM | High throughput |
| Expert Parallelism | ✅ Large-scale | ⚠️ Limited |

**Choose SGLang for:**
- Structured generation and constrained decoding
- Complex prompt engineering and control flow
- Prefix sharing across requests (RadixAttention)
- Vision-language model deployment
- Large-scale expert parallelism (MoE models)
- Prefill-decode disaggregation

**Choose vLLM for:**
- Simple completions at maximum throughput
- Wide model compatibility
- Simple deployment requirements

> **Note**: Both SGLang and vLLM are excellent choices for AMD GPUs. SGLang offers more advanced features for complex workloads, while vLLM excels at straightforward high-throughput serving.

## Recent Updates and Highlights

**2025 Updates:**
- **October 2025**: Native TPU support via SGLang-Jax backend
- **September 2025**: Day-0 support for DeepSeek-V3.2 with sparse attention
- **September 2025**: Deploying DeepSeek on GB200 NVL72: 3.8x prefill, 4.8x decode throughput
- **August 2025**: Day-0 support for OpenAI gpt-oss model
- **June 2025**: Awarded third batch of Open Source AI Grant by a16z
- **May 2025**: Deploying DeepSeek with PD disaggregation and large-scale EP on 96 H100 GPUs
- **March 2025**: Joined PyTorch Ecosystem

**2024 Updates:**
- **December 2024**: v0.4 Release - Zero-overhead batch scheduler, cache-aware load balancer, faster structured outputs
- **September 2024**: v0.3 Release - 7x faster DeepSeek MLA, 1.5x faster torch.compile, multi-image/video support
- **July 2024**: v0.2 Release - Faster Llama3 serving vs TensorRT-LLM and vLLM

## Adoption and Community

SGLang has been deployed at large scale, generating **trillions of tokens** in production each day. It is trusted by:

**Technology Companies:**
- xAI, AMD, NVIDIA, Intel
- LinkedIn, Cursor
- Oracle Cloud, Google Cloud, Microsoft Azure, AWS

**Cloud & Infrastructure Providers:**
- Atlas Cloud, Voltage Park, Nebius
- DataCrunch, Novita, InnoMatrix

**Academic Institutions:**
- MIT, Stanford, UC Berkeley, UCLA
- University of Washington, Tsinghua University

**Deployment Scale:**
- Over **300,000 GPUs** worldwide
- De facto industry standard for LLM inference
- Hosted under LMSYS non-profit organization
- Part of PyTorch ecosystem

## AMD GPU Optimizations

SGLang provides specific optimizations for AMD GPUs:

- **Native ROCm Support**: Optimized kernels for MI300/MI355 series
- **HIP Integration**: Efficient memory management with ROCm
- **Matrix Core Acceleration**: Leveraging CDNA architecture
- **Multi-GPU Scaling**: Excellent scaling on AMD multi-GPU systems

AMD has published several blog posts featuring SGLang:
- **Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X**
- **Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU**
- **DeepSeek V3 Day-One Support on NVIDIA and AMD GPUs**

## References

### Official Resources

- **[SGLang GitHub Repository](https://github.com/sgl-project/sglang)** - Official source code (19.7k+ stars)
- **[SGLang Documentation](https://docs.sglang.ai/)** - Complete user guide and API reference
- **[SGLang Blog](https://lmsys.org/blog/)** - Updates and technical deep-dives
- **[Join Slack](https://join.slack.com/t/sgl-fru7574/shared_invite/)** - Community discussions

### Installation & Getting Started

- **[Install Guide](https://docs.sglang.ai/install.html)** - Installation instructions
- **[Quick Start](https://docs.sglang.ai/start/)** - Get started quickly
- **[Backend Tutorial](https://docs.sglang.ai/backend/)** - Backend serving guide
- **[Frontend Tutorial](https://docs.sglang.ai/frontend/)** - Programming language guide

### Performance & Benchmarks

- **[v0.2 Blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)** - Faster Llama3 serving
- **[v0.3 Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)** - 7x faster DeepSeek MLA
- **[v0.4 Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)** - Zero-overhead scheduler
- **[Large-scale Expert Parallelism](https://lmsys.org/blog/2025-05-02-sglang-ep/)** - DeepSeek on 96 GPUs

### Research & Publications

- **[RadixAttention Paper](https://arxiv.org/abs/2312.07104)** - Original SGLang paper
- **[PyTorch Blog](https://pytorch.org/blog/sglang/)** - SGLang joins PyTorch ecosystem

### AMD Resources

- **[AMD DeepSeek-R1 Blog](https://community.amd.com/t5/instinct-accelerators/supercharge-deepseek-r1-inference-on-amd-instinct-mi300x/ba-p/711954)** - DeepSeek-R1 on MI300X
- **[AMD DeepSeek V3 Blog](https://community.amd.com/t5/instinct-accelerators/day-one-support-for-deepseek-v3-on-amd-instinct-mi300x-gpus/ba-p/711234)** - DeepSeek V3 optimization

### Related Guides

- [PyTorch with ROCm](../../layer-4-frameworks/pytorch/pytorch-rocm-basics.md)
- [vLLM Serving](vllm-serving.md)
- [Production Serving](../deployment/production-serving.md)
- [Docker Deployment](../deployment/docker-deployment.md)
- [GPU Optimization Best Practices](../../../best-practices/performance/gpu-optimization.md)

