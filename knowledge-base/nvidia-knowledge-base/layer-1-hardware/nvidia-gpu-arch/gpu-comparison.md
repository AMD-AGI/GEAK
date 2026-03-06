Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "1"
category: "nvidia-gpu-arch"
tags: ["comparison", "gpu-selection", "architecture", "benchmarks"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# Nvidia GPU Architecture Comparison

*Comparison of latest Nvidia datacenter GPU architectures: Blackwell and Hopper*

## Quick Reference Table

| GPU | Architecture | Tensor Cores | Memory | Memory BW | FP8 Tensor | FP4 Tensor | NVLink | TDP |
|-----|--------------|--------------|---------|-----------|------------|------------|---------|-----|
| **B200** | Blackwell | 1000+ (5th) | 192GB HBM3e | 8 TB/s | ~10 PF | ~20 PF | 1.8 TB/s | 1000W+ |
| **B100** | Blackwell | 1000+ (5th) | 192GB HBM3e | 8 TB/s | ~7 PF | ~14 PF | 1.8 TB/s | 700W |
| **H200** | Hopper | 528 (4th) | 141GB HBM3e | 4.8 TB/s | 3,958 TF | N/A | 900 GB/s | 700W |
| **H100** | Hopper | 528 (4th) | 80/96GB HBM3 | 3.35 TB/s | 3,958 TF | N/A | 900 GB/s | 700W |

## Generation Comparison: Blackwell vs Hopper

### Training Performance (Relative to H100 = 1.0x)

| Model | H100 FP8 | B200 FP4 | Improvement |
|-------|----------|----------|-------------|
| GPT-4 1.7T params | 1.0x | 2.5x | 2.5x faster |
| LLaMA 405B | 1.0x | 3.0x | 3x faster |
| Mixtral 8x22B (MoE) | 1.0x | 4.0x | 4x faster |
| BERT Large | 1.0x | 2.5x | 2.5x faster |

### Inference Performance (Relative to H100 = 1.0x)

| Model | H100 FP8 | B200 FP4 | Improvement |
|-------|----------|----------|-------------|
| LLaMA 405B | 1.0x | 4.0x | 4x faster |
| GPT-4 class | 1.0x | 3.5x | 3.5x faster |
| Mixtral 8x22B | 1.0x | 5.0x | 5x faster |

### Key Feature Evolution

| Feature | Hopper | Blackwell |
|---------|--------|-----------|
| **Tensor Cores** | ✅ 4th gen | ✅ 5th gen |
| **FP8** | ✅ | ✅ Enhanced |
| **FP4** | ❌ | ✅ New |
| **FP6** | ❌ | ✅ New |
| **Transformer Engine** | ✅ v1 | ✅ v2 |
| **MIG** | ✅ | ✅ Enhanced |
| **Structured Sparsity** | ✅ (2:4) | ✅ Enhanced |
| **NVLink** | ✅ 4.0 | ✅ 5.0 |
| **Max Memory** | 141GB | 192GB |
| **Memory Bandwidth** | 4.8 TB/s | 8 TB/s |

## Use Case Recommendations

### LLM Training

**For 7B-13B models:**
- **Best**: B200 (192GB) - FP4, massive memory
- **Good**: H100 (80GB) - FP8, proven
- Single GPU handles easily

**For 30B-70B models:**
- **Best**: B200 single GPU - 192GB handles comfortably
- **Alternative**: 2x H100 with NVLink

**For 175B-405B models:**
- **Best**: 2-4x B200 with NVLink 5.0
- **Alternative**: 4-8x H100 with NVLink 4.0
- **GB200**: 2x B200 + Grace CPU optimal

**For 1T+ models:**
- **Required**: 8-144x B200 with NVSwitch Gen 4
- Full NVLink 5.0 fabric
- Trillion-parameter training now feasible

### LLM Inference

**For 7B-13B models:**
- **Best**: B200 with FP4 - Maximum throughput
- **Good**: H100 with FP8 - Proven performance

**For 30B-70B models:**
- **Best**: B200 (192GB) single GPU - FP4 for 4x throughput
- **Good**: H100 (80GB) with FP8

**For 175B-405B models:**
- **Best**: B200 (192GB) single GPU with FP4 - Can fit 405B!
- **Alternative**: 2x H100 with tensor parallelism
- **GB200**: Optimal for high-throughput serving

**For 1T+ models:**
- **Required**: 2-8x B200 with NVLink 5.0
- GB200 superchip for maximum efficiency

### Fine-tuning (LoRA/QLoRA)

**For 7B-13B models:**
- **Any B200/H100**: Single GPU with plenty of headroom
- FP4/FP8 enables large batch sizes

**For 30B-70B models:**
- **B200**: Single GPU LoRA comfortably
- **H100**: Single GPU with FP8

**For 175B-405B models:**
- **B200 (192GB)**: Single GPU QLoRA possible!
- **2x H100**: LoRA with FSDP
- **GB200**: Optimal for fine-tuning + serving

### Computer Vision

**Training:**
- **H100/A100**: Large batch sizes, fast iteration
- **A30**: Good for smaller models
- **V100**: Still viable for many CV workloads

**Inference:**
- **A10**: Excellent price/performance for inference
- **A30**: Multi-model serving with MIG
- **T4**: Ultra cost-effective (not in table, but good option)

### Research & Experimentation

**Best overall:**
- **H100**: Fastest iteration, all features
- **A100 (80GB)**: Large models, proven

**Budget:**
- **A30**: Good balance
- **V100**: Still useful for many research tasks

## Memory Capacity Guide

### Model Size vs GPU Memory (Inference, FP16)

| Model Size | Min GPU Memory | Recommended GPU |
|------------|----------------|-----------------|
| 7B params | 14GB | A30 (24GB), A100 (40GB) |
| 13B params | 26GB | A100 (40GB), H100 (80GB) |
| 30B params | 60GB | A100 (80GB), H100 (80GB) |
| 65-70B params | 130GB | H200 (141GB), 2x A100 (80GB) |
| 175B params | 350GB | 4-8x H100/A100 |

### Model Size vs GPU Memory (Training, FP32/TF32)

| Model Size | Min GPU Memory | Recommended Setup |
|------------|----------------|-------------------|
| 1B params | 16GB | 1x A100 (40GB) |
| 7B params | 112GB | 2x A100 (80GB) |
| 13B params | 208GB | 4x A100 (80GB) |
| 30B params | 480GB | 8x A100 (80GB) |
| 70B params | 1.1TB | 16x A100 (80GB) |

*With mixed precision, gradient checkpointing, and optimizer state offloading, requirements can be reduced significantly*

## Price/Performance Considerations

### Approximate Price Tier (Lower is cheaper)

```
Tier 1 (Most Expensive): H200, H100
Tier 2: A100 (80GB)
Tier 3: A100 (40GB)
Tier 4: A30, A10
Tier 5: V100 (legacy, but available)
```

### Best Value by Workload

**Training LLMs:**
- **Best performance**: H100 (FP8 + high memory BW)
- **Best value**: A100 (80GB) - Proven, widely available
- **Budget**: A100 (40GB) or A30

**Inference (high throughput):**
- **Best performance**: H100 with FP8
- **Best value**: A10 (excellent price/perf for inference)
- **Budget**: A30 or T4

**Fine-tuning:**
- **Best for large models**: H100/A100 (80GB)
- **Best value**: A30 (LoRA/QLoRA)
- **Budget**: A10 with QLoRA

**Multi-tenant:**
- **Best**: A100 or A30 with MIG
- **Value**: A30 (lower cost, MIG support)

## Compute Capability Reference

| GPU | Compute Capability | CUDA Version Required |
|-----|-------------------|-----------------------|
| H100/H200 | 9.0 | CUDA 13.0+ |
| A100/A30/A10 | 8.0/8.6 | CUDA 11.0+ |
| V100 | 7.0 | CUDA 9.0+ |
| P100 | 6.0 | CUDA 8.0+ |
| P40/P4 | 6.1 | CUDA 8.0+ |

### Compiler Flags

```bash
# Hopper (H100/H200)
nvcc -arch=sm_90 kernel.cu

# Ampere (A100/A30)
nvcc -arch=sm_80 kernel.cu

# Ampere (A10, RTX 30-series)
nvcc -arch=sm_86 kernel.cu

# Volta (V100)
nvcc -arch=sm_70 kernel.cu

# Pascal (P100)
nvcc -arch=sm_60 kernel.cu

# Multiple architectures
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90 kernel.cu
```

## Checking Your GPU

```bash
# Get GPU name and compute capability
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Example output:
# name, compute_cap, memory.total [MiB]
# NVIDIA H100, 9.0, 81559 MiB
# NVIDIA A100-SXM4-80GB, 8.0, 81920 MiB

# Detailed architecture info
nvidia-smi -q | grep -E "Product Name|Compute Capability|Memory"

# Check NVLink availability
nvidia-smi nvlink --status

# Check MIG support
nvidia-smi --query-gpu=mig.mode.current --format=csv
```

## Migration Guide

### From V100 to A100

**Benefits:**
- 2.5x faster training with TF32 (automatic)
- 2x memory capacity (80GB vs 32GB)
- 2x NVLink bandwidth
- MIG support for multi-tenancy

**Code changes:**
- None required for TF32 (automatic)
- Consider increasing batch sizes
- Recompile with `-arch=sm_80`

### From A100 to H100

**Benefits:**
- 2x faster training with FP8 (requires code changes)
- Up to 2.4x inference with FP8
- 1.67x memory bandwidth
- 1.5x NVLink bandwidth

**Code changes:**
```python
# Add Transformer Engine for FP8
import transformer_engine.pytorch as te

# Replace Linear layers
# Before:
layer = nn.Linear(hidden_size, hidden_size)

# After:
layer = te.Linear(hidden_size, hidden_size)

# Use FP8 autocast
with te.fp8_autocast(enabled=True):
    output = model(input)
```

### From Pascal to Ampere

**Benefits:**
- 6-8x faster training (Tensor Cores + TF32)
- Much larger memory capacity
- MIG support
- Better multi-GPU scaling

**Code changes:**
- Use mixed precision or TF32
- Increase batch sizes significantly
- Recompile with `-arch=sm_80`

## Decision Matrix

### Training New LLM from Scratch
→ **H100** (fastest iteration, FP8 support)

### Fine-tuning Pre-trained LLMs
→ **A100 (80GB)** (proven, good capacity)

### High-Throughput Inference
→ **H100** with FP8 or **A10** (cost-effective)

### Multi-Tenant Inference Serving
→ **A100** or **A30** with MIG

### Research on Modest Budgets
→ **A30** (good balance) or **V100** (legacy but usable)

### Production Inference at Scale
→ **A10** (best value) or **H100** (best performance)

## External Resources

- [Nvidia GPU Comparison Tool](https://www.nvidia.com/en-us/data-center/products/)
- [MLPerf Training Results](https://mlcommons.org/en/training-normal-21/)
- [MLPerf Inference Results](https://mlcommons.org/en/inference-datacenter-21/)
- [Nvidia GPU Architecture Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

## Related Guides

- [Hopper Architecture Guide](hopper-architecture.md)
- [Ampere Architecture Guide](ampere-architecture.md)
- [Volta Architecture Guide](volta-architecture.md)
- [Pascal Architecture Guide](pascal-architecture.md)
- [CUDA Programming Basics](../../layer-2-compute-stack/cuda/cuda-basics.md)

