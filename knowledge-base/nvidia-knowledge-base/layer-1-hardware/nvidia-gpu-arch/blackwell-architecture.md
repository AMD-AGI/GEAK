---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "1"
category: "nvidia-gpu-arch"
tags: ["blackwell", "architecture", "b100", "b200", "datacenter", "ai-acceleration"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# Blackwell Architecture Guide

*Next-generation Nvidia datacenter GPU architecture for AI workloads*

## Overview

Nvidia Blackwell (2024+) is the successor to Hopper, featuring revolutionary improvements in AI performance, memory bandwidth, and power efficiency. Named after [David Blackwell](https://en.wikipedia.org/wiki/David_Blackwell), a pioneering mathematician and statistician who made significant contributions to game theory, probability theory, and Bayesian statistics.

**Key Innovations:**
- 2.5x AI training performance vs H100
- 4x AI inference performance with FP4 precision
- 192GB HBM3e memory (2.4x H100's 80GB)
- 8 TB/s memory bandwidth (2.4x H100)
- 5th generation Tensor Cores with FP4/FP6 support
- NVLink 5.0 with 1.8 TB/s bidirectional bandwidth

Reference: [Nvidia Blackwell Architecture](https://resources.nvidia.com/en-us-blackwell-architecture)

## Blackwell Architecture Products

### GB200 Grace Blackwell Superchip (Flagship)

The ultimate AI superchip combining CPU and GPU:

```
Architecture: Blackwell + Grace CPU
Process: TSMC 4NP (custom 4nm)
GPU Dies: 2x B200 GPU dies + 1x Grace CPU
Total CUDA Cores: ~28,000 (across 2 dies)
Total Tensor Cores: 1,000+ (5th gen)
GPU Memory: 192GB HBM3e per GPU (384GB total)
Memory BW: 8 TB/s per GPU
CPU: 72-core Grace ARM CPU
CPU Memory: 480GB LPDDR5X
CPU-GPU Link: 900 GB/s NVLink-C2C
FP64: ~60 TFLOPS (double precision)
TF32 Tensor: ~2.5 PetaFLOPS
FP16 Tensor: ~5 PetaFLOPS
FP8 Tensor: ~10 PetaFLOPS
FP6 Tensor: ~13 PetaFLOPS
FP4 Tensor: ~20 PetaFLOPS (4-bit float)
NVLink: 1.8 TB/s per GPU (NVLink 5.0)
TDP: 1000W+ per superchip
Form Factor: Integrated CPU+GPU module
Target: DGX GB200 systems, cloud AI
```

**GB200 NVL72 System:**
- 36 Grace CPUs + 72 B200 GPUs
- 13.5 exaFLOPS FP4 AI performance
- NVLink Switch System interconnect
- Liquid cooling required

### B200 (Dual-Die GPU)

High-performance Blackwell GPU for AI workloads:

```
Architecture: Blackwell (GB200 component)
Process: TSMC 4NP (custom 4nm)
GPU Dies: 2x GPU dies on single package
CUDA Cores: ~14,000 per die (28,000 total)
Tensor Cores: ~500 per die (1,000+ total)
Memory: 192GB HBM3e (unified)
Memory BW: 8 TB/s
FP8 Tensor: ~10 PetaFLOPS
FP4 Tensor: ~20 PetaFLOPS
NVLink: 1.8 TB/s (NVLink 5.0)
PCIe: Gen 5.0 x16
TDP: 700-1000W
Form Factor: SXM5 module, PCIe card
Target: AI training, large model inference
```

### B100 (Single Package)

Standard Blackwell GPU for broader deployment:

```
Architecture: Blackwell (single package variant)
Process: TSMC 4NP
GPU Dies: 2x GPU dies (like B200)
Memory: 192GB HBM3e
Memory BW: 8 TB/s
FP8 Tensor: ~7-9 PetaFLOPS
FP4 Tensor: ~14-18 PetaFLOPS
NVLink: 1.8 TB/s (NVLink 5.0)
PCIe: Gen 5.0 x16
TDP: 700W
Form Factor: SXM5, PCIe card
Target: AI inference, edge deployment
```

### B40 (Cloud and Edge)

Lower-power Blackwell for cloud and edge inference:

```
Architecture: Blackwell (optimized)
Memory: 48-96GB HBM3e
Memory BW: 4 TB/s
FP8 Tensor: ~3 PetaFLOPS
FP4 Tensor: ~6 PetaFLOPS
TDP: 300-400W
Form Factor: PCIe card, cloud instances
Target: Cost-effective AI inference
```

## Key Architectural Features

### 5th Generation Tensor Cores

Massive improvements for AI:

- **FP4 Support**: 4-bit floating point for 2x FP8 throughput
- **Dynamic Range**: Better accuracy than INT4
- **FP6 Support**: Middle ground between FP4 and FP8
- **Double Precision Tensor Cores**: Enhanced FP64 AI workloads
- **Sparsity**: Enhanced structured sparsity support

Performance hierarchy:
```
FP64 Tensor:    TBD
FP32:           TBD
TF32 Tensor:    TBD
FP16 Tensor:    ~5 PetaFLOPS
FP8 Tensor:     ~10 PetaFLOPS
FP6 Tensor:     ~13 PetaFLOPS (estimated)
FP4 Tensor:     ~20 PetaFLOPS
```

### Second-Generation Transformer Engine

Enhanced for modern LLMs:

**New Features:**
- FP4 precision support
- FP6 for better accuracy/performance tradeoff
- Per-channel quantization
- Dynamic precision scaling
- Optimized for Mixture-of-Experts models

**Usage:**
```python
import transformer_engine.pytorch as te

# Enable second-gen Transformer Engine with FP4
with te.fp8_autocast(enabled=True, fp8_recipe=te.DelayedScaling(fp4_enabled=True)):
    output = model(input)
```

### Massive Memory Bandwidth

Revolutionary memory subsystem:

**B200:**
- 192GB HBM3e (unified across 2 dies)
- 8 TB/s bandwidth (2.4x H100)
- Coherent memory between GPU dies
- Enables trillion-parameter models

### NVLink 5.0

Next-generation GPU interconnect:

- **1.8 TB/s** bidirectional per GPU (2x NVLink 4.0)
- **18 NVLink connections**
- **NVSwitch Gen 4**: 144 Blackwell GPUs in fabric
- **Sub-microsecond latency**

### GB200 Superchip

Integrated CPU+GPU design:

- **2x B200 GPUs** + **Grace CPU**
- **900 GB/s** CPU-GPU bandwidth via NVLink-C2C
- Unified memory architecture
- Optimal for inference serving

## Blackwell vs Hopper

### Performance Comparison

| Feature | H100 (Hopper) | B200 (Blackwell) | Improvement |
|---------|---------------|------------------|-------------|
| Process | 4nm | 4nm+ | Better efficiency |
| GPU Dies | 1 | 2 | Double compute |
| Memory | 80GB HBM3 | 192GB HBM3e | 2.4x capacity |
| Memory BW | 3.35 TB/s | 8 TB/s | 2.4x faster |
| FP8 Tensor | 3,958 TF | ~10 PF | ~2.5x |
| FP4 Tensor | N/A | ~20 PF | New feature |
| NVLink BW | 900 GB/s | 1.8 TB/s | 2x |

### Training Performance (Relative to H100 = 1.0x)

| Model | H100 | B200 | Improvement |
|-------|------|------|-------------|
| GPT-4 scale (1.7T) | 1.0x | ~2.5x | FP8 + memory BW |
| LLaMA 405B | 1.0x | ~3x | FP4 acceleration |
| Mixture-of-Experts | 1.0x | ~4x | Optimized routing |

### Inference Performance

| Model | H100 FP8 | B200 FP4 | Improvement |
|-------|----------|----------|-------------|
| GPT-4 class | 1.0x | ~3.5x | FP4 + bandwidth |
| LLaMA 405B | 1.0x | ~4x | Fits in single GPU |
| Mixtral 8x22B | 1.0x | ~5x | MoE optimization |

## Blackwell for LLM Workflows

### Training Trillion-Parameter Models

B200 advantages:

1. **Massive Memory**:
   - 192GB per GPU: Train 100B+ models per GPU
   - GB200: 384GB unified memory
   - 1024 GPUs: 196TB combined memory

2. **FP4 Training**:
   - 2x faster than FP8
   - Maintains model quality
   - Enables larger batch sizes

3. **NVLink 5.0 Scaling**:
   - 144 GPU pods with NVSwitch Gen 4
   - Near-perfect scaling to 1000+ GPUs
   - Trillion-parameter model training feasible

### Inference Optimization

B200 features for serving:

1. **FP4 Inference**:
   ```python
   # FP4 for maximum throughput
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-3-405b",
       torch_dtype=torch.float4_e2m1,  # FP4
       device_map="auto"
   )
   ```

2. **Single-GPU Large Models**:
   - Serve 405B models on single B200
   - 192GB enables large context windows
   - Reduced need for tensor parallelism

3. **GB200 for Serving**:
   - 2x B200 + Grace CPU
   - 900 GB/s CPU-GPU bandwidth
   - Optimal for high-throughput inference

## Programming for Blackwell

### Compiler Flags

```bash
# For B100/B200 (compute capability 10.0)
nvcc -arch=sm_100 kernel.cu

# Or use PTX
nvcc -arch=compute_100 -code=sm_100 kernel.cu
```

### FP4 Support

```python
# PyTorch with Transformer Engine
import transformer_engine.pytorch as te

# FP4 recipe
fp4_recipe = te.DelayedScaling(
    fp4_enabled=True,
    fp8_enabled=True,
    amax_history_len=16
)

# Training with FP4
with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
    loss = model(inputs)
    loss.backward()
```

### NVLink 5.0 Optimization

```python
# Ensure NCCL uses NVLink 5.0
import os
os.environ['NCCL_NET_GDR_LEVEL'] = '5'
os.environ['NCCL_NVLINK_ENABLE'] = '1'

# Initialize with NCCL
import torch.distributed as dist
dist.init_process_group(backend='nccl')
```

## Best Practices for B200

### Memory Optimization

1. **Use FP4**: 4x memory savings vs FP16
2. **Leverage 192GB**: Train/serve largest models
3. **Unified memory**: Optimize for 2-die coherency

### Performance Tuning

1. **Enable FP4 Tensor Cores**: Maximum throughput
2. **Optimize for NVLink 5.0**: Minimize cross-GPU transfers
3. **Use GB200 for inference**: CPU-GPU co-design benefits

### Multi-GPU Strategies

1. **NVLink 5.0 first**: 2x faster than NVLink 4.0
2. **NVSwitch Gen 4**: Full fabric for 144 GPUs
3. **Consider GB200**: Integrated CPU+GPU for serving

## Checking GPU Architecture

```bash
# Check GPU architecture
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Expected for B200:
# B200, 10.0

# Check NVLink 5.0
nvidia-smi nvlink --status
nvidia-smi nvlink --capabilities
```

## System Configurations

### DGX GB200

Complete AI supercomputer:

- **36 GB200 Superchips** (72 B200 GPUs + 36 Grace CPUs)
- **13.5 exaFLOPS** FP4 AI performance
- **72x 192GB** HBM3e = 13.8TB GPU memory
- **NVLink Switch System** for all-to-all connectivity
- **Liquid cooling** with 2U/4U form factors

### MGX B200 Servers

Flexible configurations:

- 4-8 B200 GPUs per server
- Air or liquid cooling options
- PCIe Gen 5.0 connectivity
- Compatible with standard racks

### Cloud Instances

- **Amazon EC2**: P6 instances (expected H2 2025)
- **Microsoft Azure**: ND GB200 v5 series
- **Google Cloud**: A4 instances with B100/B200
- **Oracle Cloud Infrastructure**: BM.GPU.GB200 shapes

## Availability

- **GB200 DGX Systems**: Q2 2025 (limited availability)
- **GB200 NVL72**: Q3 2025 (volume production)
- **B200**: Cloud providers Q2-Q3 2025
- **B100**: General availability Q3 2025
- **B40**: Q4 2025

## External Resources

- [Nvidia Blackwell Architecture White Paper](https://resources.nvidia.com/en-us-blackwell-architecture) - Official technical overview
- [Blackwell Architecture Overview](https://www.nvidia.com/en-us/data-center/blackwell-architecture/) - Product page
- [GB200 Grace Blackwell Superchip](https://www.nvidia.com/en-us/data-center/grace-blackwell-superchip/) - Superchip details
- [DGX GB200 Systems](https://www.nvidia.com/en-us/data-center/dgx-gb200/) - Complete AI supercomputer
- [CUDA 13.x Documentation](https://docs.nvidia.com/cuda/) - Programming guide

## Related Guides

- [Hopper Architecture](hopper-architecture.md)
- [CUDA 13.x Programming](../../layer-2-compute-stack/cuda/cuda-basics.md)
- [Transformer Engine v2](../../layer-3-libraries/ml-frameworks/transformer-engine.md)
- [LLM Training on B200](../../layer-5-llm/03-training/fine-tuning/full-finetuning.md)

