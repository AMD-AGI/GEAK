Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "1"
category: "nvidia-gpu-arch"
tags: ["hopper", "architecture", "h100", "h200", "datacenter", "ai-acceleration", "tensor-cores"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# Hopper Architecture Guide

*Comprehensive guide to Nvidia Hopper architecture and H-series datacenter GPUs for AI workloads*

## Overview

Nvidia Hopper is the latest datacenter GPU architecture (2022+), designed specifically for large-scale AI training and inference. Named after computing pioneer Grace Hopper, it introduces revolutionary features like the Transformer Engine and 4th generation Tensor Cores.

## Hopper Architecture Generations

### H100 (First Generation Hopper)

The flagship Hopper GPU for AI and HPC workloads:

```
Architecture: Hopper (GH100)
Process: TSMC 4N (custom 5nm)
SM Count: 132 SMs (8 GPCs, 66 TPCs)
CUDA Cores: 16,896
Tensor Cores: 528 (4th gen)
Memory: 80GB or 96GB HBM3
Memory BW: 3.35 TB/s (HBM3)
FP64: 34 TFLOPS
FP32: 67 TFLOPS
TF32 Tensor: 989 TFLOPS
FP16 Tensor: 1,979 TFLOPS
FP8 Tensor: 3,958 TFLOPS
NVLink: 900 GB/s (18 links, NVLink 4.0)
TDP: 700W (SXM5)
Form Factor: SXM5, PCIe
```

### H200 (Enhanced Hopper)

Improved memory capacity and bandwidth:

```
Architecture: Hopper (GH100)
Process: TSMC 4N
SM Count: 132 SMs
Memory: 141GB HBM3e
Memory BW: 4.8 TB/s (HBM3e)
FP8 Tensor: 3,958 TFLOPS
NVLink: 900 GB/s
TDP: 700W
Form Factor: SXM5
```

## Key Architectural Features

### 4th Generation Tensor Cores

Major improvements for AI workloads:

- **FP8 Support**: Native 8-bit floating point for 2x throughput
- **Transformer Engine**: Automatic FP8/FP16 mixed precision
- **Sparsity Acceleration**: 2:4 structured sparsity (2x speedup)
- **Enhanced TF32**: Improved accuracy for single precision

Tensor Core performance hierarchy:
```
FP64 Tensor:    34 TFLOPS
FP32:           67 TFLOPS  
TF32 Tensor:    989 TFLOPS  (default for PyTorch)
FP16 Tensor:    1,979 TFLOPS
BF16 Tensor:    1,979 TFLOPS
FP8 Tensor:     3,958 TFLOPS (with Transformer Engine)
INT8 Tensor:    3,958 TOPS
```

### Transformer Engine

Revolutionary feature for LLM training and inference:

**What it does:**
- Dynamically selects FP8 or FP16 precision per layer
- Maintains model accuracy while using FP8
- Delivers up to 2x speedup for transformer models
- Reduces memory bandwidth requirements

**How to use:**
```python
import torch
import transformer_engine.pytorch as te

# Enable Transformer Engine
layer = te.Linear(hidden_size, hidden_size, device='cuda')

# Automatic FP8/FP16 selection during training
with te.fp8_autocast(enabled=True):
    output = model(input)
```

### HBM3/HBM3e Memory

Significant memory improvements:

**H100 (HBM3):**
- 80GB or 96GB capacity
- 3.35 TB/s bandwidth
- 50% more bandwidth than A100

**H200 (HBM3e):**
- 141GB capacity (76% more than H100)
- 4.8 TB/s bandwidth (43% faster than H100)
- Critical for large model inference

### NVLink 4.0

Enhanced multi-GPU interconnect:

- **900 GB/s** bidirectional bandwidth per GPU
- **18 NVLink connections** (vs 12 on A100)
- **NVSwitch**: 64 H100s in full NVLink fabric
- **GPU-to-GPU latency**: <2 microseconds

### Multi-Instance GPU (MIG)

Partition single GPU into isolated instances:

**H100 MIG Configurations:**
```
1x 1g.10gb   (1/7 GPU,  10GB)
1x 2g.20gb   (2/7 GPU,  20GB)
1x 3g.40gb   (3/7 GPU,  40GB)
1x 4g.40gb   (4/7 GPU,  40GB)
1x 7g.80gb   (full GPU, 80GB)

Mixed: 3x 1g.10gb + 2x 2g.20gb
```

Enable MIG mode:
```bash
# Enable MIG mode
sudo nvidia-smi -mig 1

# Create instances
sudo nvidia-smi mig -cgi 3g.40gb -C

# List instances
nvidia-smi mig -lgi
```

## Hopper vs Previous Generations

### Hopper (H100) vs Ampere (A100)

| Feature | H100 | A100 | Improvement |
|---------|------|------|-------------|
| Process | 4nm | 7nm | Better efficiency |
| CUDA Cores | 16,896 | 6,912 | 2.4x |
| Tensor Cores | 528 (4th gen) | 432 (3rd gen) | FP8 support |
| Memory | 80/96GB HBM3 | 40/80GB HBM2e | More capacity |
| Memory BW | 3.35 TB/s | 2.0 TB/s | 1.7x faster |
| FP16 Tensor | 1,979 TF | 624 TF | 3.2x |
| FP8 Tensor | 3,958 TF | N/A | New feature |
| NVLink BW | 900 GB/s | 600 GB/s | 1.5x |
| Transformer Engine | Yes | No | New feature |

### Training Performance Comparison

Typical speedups for LLM training (H100 vs A100):

- **GPT-3 175B**: 2.2x faster (with Transformer Engine)
- **BERT Large**: 2.5x faster
- **Llama 2 70B**: 2.3x faster
- **Stable Diffusion**: 2.1x faster

### Inference Performance Comparison

LLM inference improvements (H100 vs A100):

- **Llama 2 70B FP8**: 2.4x throughput
- **GPT-3 175B**: 2.8x throughput with FP8
- **Latency**: 30-40% reduction

## Hopper for LLM Workflows

### Training Large Models

H100 advantages for training:

1. **Larger Model Capacity**:
   - 96GB memory: Train 13B models on single GPU
   - H200 141GB: Train 20B models on single GPU

2. **Faster Training**:
   - FP8 with Transformer Engine: 2x speedup
   - Higher memory bandwidth: Better utilization
   - NVLink 4.0: Faster multi-GPU scaling

3. **Better Scaling**:
   - 8x H100 NVLink pod: Train 175B models efficiently
   - NVSwitch fabric: Up to 64 GPUs with full interconnect

### Inference Optimization

H100 features for inference:

1. **FP8 Inference**:
   ```python
   # Use FP8 for 2x throughput
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-2-70b-hf",
       torch_dtype=torch.float8_e4m3fn,  # FP8
       device_map="auto"
   )
   ```

2. **MIG for Multi-Tenancy**:
   - Serve multiple models on one GPU
   - Isolated resources and QoS
   - Cost-effective for diverse workloads

3. **High Memory for Large Context**:
   - H200 141GB: Serve 70B models with 32K+ context
   - Reduced need for quantization

## Programming for Hopper

### Compiler Flags

Optimize for Hopper architecture:

```bash
# For H100 (compute capability 9.0)
nvcc -arch=sm_90 kernel.cu

# Or use PTX for forward compatibility
nvcc -arch=compute_90 -code=sm_90 kernel.cu
```

### Tensor Core Usage

Leverage 4th gen Tensor Cores:

```cuda
// Include Tensor Core headers
#include <mma.h>
#include <cuda_fp8.h>

// Use WMMA for Tensor Core operations
nvcuda::wmma::fragment<...> a, b, c;
nvcuda::wmma::load_matrix_sync(a, ptr, stride);
nvcuda::wmma::mma_sync(c, a, b, c);
```

### FP8 Support

Use FP8 for maximum performance:

```python
# PyTorch with Transformer Engine
import transformer_engine.pytorch as te

# Enable FP8 training
with te.fp8_autocast(enabled=True):
    loss = model(inputs)
    loss.backward()
```

## Best Practices for H100/H200

### Memory Optimization

1. **Use FP8 when possible**: 2x memory savings + 2x compute
2. **Leverage large memory**: H200's 141GB enables larger batches
3. **Enable memory pools**: Reduce allocation overhead

### Performance Tuning

1. **Enable Tensor Cores**: Use TF32/FP16/FP8
2. **Optimize for NVLink**: Minimize cross-GPU transfers
3. **Use async operations**: Hide latency with concurrent kernels

### Multi-GPU Strategies

1. **Prefer NVLink over PCIe**: 10-15x faster interconnect
2. **Use NCCL for collectives**: Optimized for NVLink topology
3. **Consider MIG**: For multi-tenant or diverse workloads

## Checking GPU Architecture

Verify your Hopper GPU:

```bash
# Check GPU architecture
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Expected output for H100:
# H100, 9.0

# Detailed GPU info
nvidia-smi -q

# Check Tensor Core support
cuda-deviceQuery
```

## Common Issues and Solutions

### Issue: Not using Tensor Cores

**Symptom**: Performance below expectations

**Solution**:
```python
# Ensure correct data types
model = model.half()  # FP16
# Or use automatic mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

### Issue: MIG mode conflicts

**Symptom**: Cannot create compute instance

**Solution**:
```bash
# Reset MIG configuration
sudo nvidia-smi mig -dci
sudo nvidia-smi mig -dgi

# Recreate instances
sudo nvidia-smi mig -cgi 3g.40gb -C
```

### Issue: Transformer Engine not activating

**Symptom**: Not seeing 2x FP8 speedup

**Solution**:
```python
# Verify Transformer Engine is installed
import transformer_engine
print(transformer_engine.__version__)

# Check CUDA version compatibility (needs 12.0+)
import torch
print(torch.version.cuda)
```

## External Resources

- [Nvidia H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [Hopper Architecture Documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [H100 Product Page](https://www.nvidia.com/en-us/data-center/h100/)
- [CUDA Programming Guide - Hopper](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-9-x)

## Related Guides

- [CUDA Programming Basics](../../layer-2-compute-stack/cuda/cuda-basics.md)
- [Ampere Architecture Guide](ampere-architecture.md)
- [cuDNN Usage with Hopper](../../layer-3-libraries/dnn/cudnn-usage.md)
- [PyTorch CUDA Optimization](../../layer-4-frameworks/pytorch/pytorch-cuda-basics.md)
- [LLM Training on H100](../../layer-5-llm/03-training/fine-tuning/full-finetuning.md)

