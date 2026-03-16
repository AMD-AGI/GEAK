---
layer: "5"
category: "llm"
subcategory: "inference"
tags: ["vllm", "inference", "serving", "llm"]
cuda_version: "13.0+"
difficulty: "intermediate"
estimated_time: "30min"
last_updated: 2025-11-17
---

# vLLM Deployment on Nvidia GPUs

*Fast and easy-to-use library for LLM inference and serving with excellent Nvidia GPU support*

## Overview

vLLM is optimized for high-throughput LLM serving with features like PagedAttention, continuous batching, and FlashAttention support.

**GitHub**: [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Installation

```bash
pip install vllm
```

## Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize LLM
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# Generate
prompts = [
    "Hello, my name is",
    "The capital of France is",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

## OpenAI-Compatible API Server

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000

# Use with OpenAI client
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(completion.choices[0].message.content)
```

## Multi-GPU Deployment

```bash
# Tensor parallelism across GPUs
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --port 8000
```

## Quantization

```bash
# AWQ 4-bit quantization
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-AWQ \
    --quantization awq

# GPTQ quantization
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-GPTQ \
    --quantization gptq
```

## Performance Tuning

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,  # GPUs for tensor parallelism
    gpu_memory_utilization=0.9,  # GPU memory to use
    max_num_batched_tokens=8192,  # Max batch size
    max_num_seqs=256,  # Max concurrent sequences
)
```

## Docker Deployment

```bash
docker run --gpus all -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 1
```

## Benchmarking

```bash
# Install benchmark tools
pip install vllm

# Run benchmark
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf &

# Benchmark
python benchmark_serving.py \
    --model meta-llama/Llama-2-7b-hf \
    --num-prompts 1000 \
    --request-rate 10
```

## Best Practices

1. **Use tensor parallelism**: For models >30B parameters
2. **Tune GPU memory utilization**: Start with 0.9
3. **Enable continuous batching**: Automatic in vLLM
4. **Use quantization**: For memory-constrained deployments
5. **Monitor metrics**: Track throughput and latency

## Common Issues

### Out of Memory

```python
# Reduce GPU memory utilization
llm = LLM(model=model_name, gpu_memory_utilization=0.7)

# Or use quantization
llm = LLM(model=model_name, quantization="awq")
```

### Low Throughput

```python
# Increase max batch size
llm = LLM(
    model=model_name,
    max_num_batched_tokens=16384,
    max_num_seqs=512
)
```

## External Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Performance Benchmarks](https://blog.vllm.ai/2023/06/20/vllm.html)

## Related Guides

- [Quickstart Inference](../../00-quickstart/quickstart-inference.md)
- [Production Serving](../deployment/production-serving.md)
- [TensorRT-LLM](tensorrt-llm.md)

