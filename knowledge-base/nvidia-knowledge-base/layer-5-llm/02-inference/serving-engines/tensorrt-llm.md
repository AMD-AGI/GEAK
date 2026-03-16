---
layer: "5"
category: "llm"
subcategory: "inference"
tags: ["tensorrt-llm", "inference", "optimization"]
cuda_version: "13.0+"
difficulty: "advanced"
estimated_time: "60min"
last_updated: 2025-11-17
---

# TensorRT-LLM Serving

*Nvidia's optimized LLM inference engine for maximum performance*

## Overview

TensorRT-LLM provides state-of-the-art inference performance for LLMs on Nvidia GPUs with optimizations like FP8, in-flight batching, and multi-GPU/multi-node support.

**GitHub**: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

## Installation

```bash
# Using Docker (recommended)
docker pull nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3

# Or pip install
pip install tensorrt-llm
```

## Convert Model to TensorRT-LLM

```bash
# Convert LLaMA to TensorRT-LLM
python convert_checkpoint.py \
    --model_dir ./llama-2-7b-hf \
    --output_dir ./trt_engines/llama/7B \
    --dtype float16

# Build engine
trtllm-build \
    --checkpoint_dir ./trt_engines/llama/7B \
    --output_dir ./trt_engines/llama/7B/engine \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 1024 \
    --max_output_len 512
```

## Run Inference

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

# Load engine
runner = ModelRunner.from_dir('./trt_engines/llama/7B/engine')

# Generate
prompts = ["Hello, my name is"]
outputs = runner.generate(prompts, max_new_tokens=100)

for output in outputs:
    print(output)
```

## Triton Inference Server Deployment

```python
# Deploy with Triton
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/trt_engines:/engines \
    nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3 \
    tritonserver --model-repository=/engines
```

## FP8 Quantization (Hopper)

```bash
# Build with FP8 on H100
trtllm-build \
    --checkpoint_dir ./trt_engines/llama/7B \
    --output_dir ./trt_engines/llama/7B/fp8 \
    --use_fp8 \
    --max_batch_size 128
```

## Multi-GPU

```bash
# Tensor parallelism
trtllm-build \
    --checkpoint_dir ./trt_engines/llama/70B \
    --output_dir ./trt_engines/llama/70B/engine \
    --world_size 4 \
    --tp_size 4
```

## Performance Optimization

1. **Enable in-flight batching**: Continuous batching for higher throughput
2. **Use FP8 on Hopper**: 2x faster than FP16
3. **Tune batch sizes**: Maximize GPU utilization
4. **Enable KV cache optimization**: Reduced memory usage

## Best Practices

1. **Pre-build engines**: Compilation takes time
2. **Use Triton**: Production-ready serving
3. **Monitor GPU utilization**: Optimize batch sizes
4. **Use FP8 on H100**: Maximum performance
5. **Benchmark your workload**: Measure actual performance

## External Resources

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [Triton Inference Server](https://github.com/triton-inference-server)

## Related Guides

- [vLLM Serving](vllm-serving.md)
- [Production Serving](../deployment/production-serving.md)
- [Hopper Architecture](../../../layer-1-hardware/nvidia-gpu-arch/hopper-architecture.md)

