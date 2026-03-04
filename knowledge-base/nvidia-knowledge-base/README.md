---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Nvidia CUDA Knowledge Base

Comprehensive knowledge for GPU programming, ML frameworks, and LLM deployment on Nvidia hardware.

## 📚 Overview

This knowledge base provides curated, tested content for AI engineers working with Nvidia GPUs, CUDA, and the Nvidia AI ecosystem.

**All content is verified for CUDA 13.0+ only.**

**Current focus**: CUDA 13.0 with Hopper (H100/H200) and Blackwell (B100/B200) architectures

## 🗂️ Directory Structure

```
nvidia-knowledge-base/
├── layer-1-hardware/              # Nvidia GPU Architecture
│   └── nvidia-gpu-arch/           # Pascal, Volta, Ampere, Hopper architectures
│
├── layer-2-compute-stack/         # CUDA & Compute Platform
│   ├── cuda/                      # CUDA installation, programming, tools
│   └── nsight/                    # Nsight profiling tools
│
├── layer-3-libraries/             # CUDA Libraries & Primitives
│   ├── blas/                      # Linear algebra (cuBLAS)
│   ├── dnn/                       # Deep learning (cuDNN)
│   ├── communications/            # Multi-GPU comm (NCCL)
│   └── compilers/                 # Triton, kernel compilation
│
├── layer-4-frameworks/            # ML Frameworks
│   ├── pytorch/                   # PyTorch with CUDA
│   ├── tensorflow/                # TensorFlow with CUDA
│   └── jax/                       # JAX with CUDA
│
├── layer-5-llm/                   # LLM & Advanced AI
│   ├── 00-quickstart/             # Fast-start guides (5-15 min)
│   ├── 01-foundations/            # Core LLM concepts
│   ├── 02-inference/              # Model serving & deployment
│   │   ├── serving-engines/       # vLLM, TensorRT-LLM, Triton
│   │   ├── optimization/          # Inference optimizations
│   │   └── deployment/            # Production deployment
│   ├── 03-training/               # Fine-tuning & training
│   │   ├── preparation/           # Dataset prep, environment
│   │   ├── fine-tuning/           # LoRA, QLoRA, full fine-tuning
│   │   ├── distributed/           # FSDP, DeepSpeed, multi-node
│   │   └── optimization/          # Training optimizations
│   ├── 04-models/                 # Model-specific guides
│   │   ├── llama/                 # LLaMA 2/3
│   │   └── mistral/               # Mistral, Mixtral
│   └── 05-advanced/               # Advanced topics
│       └── custom-kernels/        # CUDA kernels
│
└── best-practices/                # Best Practices & Standards
    ├── performance/               # Optimization techniques
    ├── debugging/                 # Debugging workflows
    └── testing/                   # Testing strategies
```

## 🚀 Quick Navigation

### New to Nvidia GPUs?
1. Start with [Blackwell Architecture](layer-1-hardware/nvidia-gpu-arch/blackwell-architecture.md) or [Hopper Architecture](layer-1-hardware/nvidia-gpu-arch/hopper-architecture.md)
2. Install [CUDA 13.0 Toolkit](layer-2-compute-stack/cuda/cuda-installation.md)
3. Learn [CUDA Basics](layer-2-compute-stack/cuda/cuda-basics.md)

### Want to Serve LLMs?
1. Quick: [LLM Inference in 5 min](layer-5-llm/00-quickstart/quickstart-inference.md)
2. Choose engine: [vLLM](layer-5-llm/02-inference/serving-engines/vllm-serving.md) or [TensorRT-LLM](layer-5-llm/02-inference/serving-engines/tensorrt-llm.md)
3. Deploy: [Docker deployment](layer-5-llm/02-inference/deployment/docker-deployment.md) or [Production serving](layer-5-llm/02-inference/deployment/production-serving.md)
4. Optimize: [Attention](layer-5-llm/02-inference/optimization/attention-optimization.md) and [Serving](layer-5-llm/02-inference/optimization/serving-optimization.md)

### Want to Fine-tune LLMs?
1. Setup: [Training environment](layer-5-llm/03-training/preparation/environment-setup.md)
2. Prepare: [Dataset preparation](layer-5-llm/03-training/preparation/dataset-preparation.md)
3. Fine-tune: [LoRA](layer-5-llm/03-training/fine-tuning/lora-finetuning.md), [QLoRA](layer-5-llm/03-training/fine-tuning/qlora-finetuning.md), or [Full](layer-5-llm/03-training/fine-tuning/full-finetuning.md)
4. Scale: [Distributed training (FSDP)](layer-5-llm/03-training/distributed/fsdp-training.md)
5. Optimize: [Memory optimization](layer-5-llm/03-training/optimization/memory-optimization.md)

### Optimizing Performance?
1. [Memory optimization](best-practices/performance/memory-optimization.md)
2. [Kernel optimization](best-practices/performance/kernel-optimization.md)
3. [CUDA profiling](layer-2-compute-stack/cuda/cuda-profiling.md)

### Working with CUDA Libraries?

**Math Libraries**:
- [cuBLAS](layer-3-libraries/blas/cublas-usage.md) - Linear algebra
- [cuFFT](layer-3-libraries/fft/cufft-usage.md) - Fast Fourier Transform
- [cuSPARSE](layer-3-libraries/sparse/cusparse-usage.md) - Sparse operations
- [cuRAND](layer-3-libraries/random/curand-usage.md) - Random number generation

**ML/DL Libraries**:
- [cuDNN](layer-3-libraries/dnn/cudnn-usage.md) - Deep learning primitives
- [NCCL](layer-3-libraries/communications/nccl-usage.md) - Multi-GPU communication

## 📖 Full Index

For complete navigation by use case, model, difficulty, or layer, see:
- **[INDEX.md](INDEX.md)** - Complete knowledge base index

## 🎯 Content Organization

### By Workflow
- **00-quickstart**: Get running in 5-15 minutes
- **01-foundations**: Core concepts everyone needs
- **02-inference**: Serving and deploying models
- **03-training**: Fine-tuning and training
- **04-models**: Model-specific optimizations
- **05-advanced**: Advanced techniques

### By Technology Layer
- **Layer 1**: Hardware fundamentals
- **Layer 2**: Compute platform (CUDA, Nsight)
- **Layer 3**: Libraries and primitives
- **Layer 4**: ML frameworks
- **Layer 5**: LLM applications

## ✅ Content Standards

All knowledge base content:
- ✅ **CUDA 13.0+ only** - No support for older versions
- ✅ Tested on CUDA 13.0 with H100/B200
- ✅ Includes working code examples
- ✅ Links to official documentation
- ✅ Provides troubleshooting tips
- ✅ Regular maintenance and updates

## 📊 Knowledge Statistics

- **Total Files**: 28+ focused guides
- **CUDA Version**: 13.0+ only
- **GPU Focus**: Hopper (H100/H200) and Blackwell (B100/B200)
- **Last Updated**: 2025-11-17
- **Coverage**: LLM inference, training, deployment with latest GPUs
- **Maintenance**: Regular updates focused on latest architectures

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - Contribution guidelines
- Quality standards and templates
- Testing requirements
- Submission process

## 📝 Metadata Format

Each knowledge file includes comprehensive metadata:

```yaml
---
layer: "5"
category: "inference"
subcategory: "serving-engines"
title: "Your Topic"
cuda_version: "13.0+"
cuda_verified: "13.0"
last_updated: "2025-11-17"
last_verified: "2025-11-17"
update_frequency: "monthly"
status: "stable"
difficulty: "intermediate"
estimated_time: "30min"
prerequisites: []
related: []
tags: []
---
```

This enables:
- Automated version tracking
- Prerequisite chains
- Smart search and filtering
- Maintenance scheduling

## 🔍 Using the Knowledge Base

### With AMD AI DevTool CLI

```bash
# Search across all knowledge bases
amd-ai-devtool search "CUDA kernel optimization"

# Browse documentation
amd-ai-devtool docs --vendor nvidia

# Initialize a new project
amd-ai-devtool init my-project --preset llm-engineer --vendor nvidia
```

### Manual Reading

Browse files directly or use [INDEX.md](INDEX.md) for navigation.

## 📚 External Resources

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Nvidia Developer Documentation](https://docs.nvidia.com/)
- [Nvidia GPU Architecture](https://developer.nvidia.com/gpu-architecture)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)

## 📄 License

This knowledge base is part of the AMD AI DevTool project, licensed under MIT License.

---

**Need help?** Open an issue or check the [Contributing Guide](../../../CONTRIBUTING.md).

