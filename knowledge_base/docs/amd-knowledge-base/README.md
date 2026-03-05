# AMD AI DevTool Knowledge Base

Comprehensive knowledge for GPU programming, ML frameworks, and LLM deployment on AMD hardware.

## 📚 Overview

This knowledge base provides curated, tested content for AI engineers working with AMD GPUs, ROCm, and the AMD AI ecosystem. 

**All content is verified for ROCm 7.0+ only.**

**Current focus**: ROCm 7.0.2 (latest stable release - October 2024)

## 🗂️ Directory Structure

```
knowledge-base/
├── layer-1-hardware/              # AMD GPU Architecture
│   └── amd-gpu-arch/              # CDNA, RDNA architectures
│
├── layer-2-compute-stack/         # ROCm & Compute Platform
│   ├── rocm/                      # ROCm installation, tools
│   ├── rocm-systems/              # ROCm systems monorepo (runtime, profiling)
│   ├── hip/                       # HIP programming language
│   └── therock/                   # TheRock build system
│
├── layer-3-libraries/             # ROCm Libraries & Primitives
│   ├── rocm-libraries/            # Unified monorepo documentation
│   ├── blas/                      # Linear algebra (rocBLAS, hipBLAS)
│   ├── fft/                       # Fast Fourier Transform (rocFFT)
│   ├── solver/                    # Linear solvers (rocSOLVER)
│   ├── sparse/                    # Sparse operations (rocSPARSE)
│   ├── random/                    # Random number generation (rocRAND)
│   ├── algorithms/                # Parallel algorithms (rocThrust)
│   ├── communications/            # Multi-GPU comm (RCCL)
│   ├── ml-primitives/             # ML operations (MIOpen)
│   └── compilers/                 # Triton, kernel compilation
│
├── layer-4-frameworks/            # ML Frameworks
│   ├── pytorch/                   # PyTorch on ROCm
│   ├── tensorflow/                # TensorFlow on ROCm
│   └── jax/                       # JAX on ROCm
│
├── layer-5-llm/                   # LLM & Advanced AI
│   ├── 00-quickstart/             # Fast-start guides (5-15 min)
│   ├── 01-foundations/            # Core LLM concepts
│   ├── 02-inference/              # Model serving & deployment
│   │   ├── serving-engines/       # vLLM, SGLang, TGI
│   │   ├── optimization/          # Inference optimizations
│   │   └── deployment/            # Production deployment
│   ├── 03-training/               # Fine-tuning & training
│   │   ├── preparation/           # Dataset prep, environment
│   │   ├── fine-tuning/           # LoRA, QLoRA, full fine-tuning
│   │   ├── distributed/           # FSDP, DeepSpeed, multi-node
│   │   └── optimization/          # Training optimizations
│   ├── 04-models/                 # Model-specific guides
│   │   ├── llama/                 # LLaMA 2/3
│   │   ├── mistral/               # Mistral, Mixtral
│   │   ├── gpt-models/            # GPT architectures
│   │   └── other-models/          # Falcon, MPT, etc.
│   └── 05-advanced/               # Advanced topics
│       ├── custom-kernels/        # Triton, HIP kernels
│       ├── research/              # Research techniques
│       └── operations/            # MLOps, monitoring
│
└── best-practices/                # Best Practices & Standards
    ├── performance/               # Optimization techniques
    ├── debugging/                 # Debugging workflows
    ├── testing/                   # Testing strategies
    └── ci-cd/                     # CI/CD pipelines
```

## 🚀 Quick Navigation

### New to AMD GPUs?
1. Start with [AMD GPU Architecture](layer-1-hardware/amd-gpu-arch/cdna-architecture.md)
2. Install [ROCm](layer-2-compute-stack/rocm/rocm-installation.md)
3. Learn [HIP Basics](layer-2-compute-stack/hip/hip-basics.md)

### Want to Serve LLMs?
1. Quick: [LLM Inference in 5 min](layer-5-llm/00-quickstart/quickstart-inference.md)
2. Choose engine: [vLLM](layer-5-llm/02-inference/serving-engines/vllm-serving.md) or [SGLang](layer-5-llm/02-inference/serving-engines/sglang-serving.md)
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
3. [ROCm profiling](layer-2-compute-stack/rocm/rocm-profiling.md)

### Working with ROCm Libraries?

**🆕 ROCm 7.1+ Monorepo**: All ROCm libraries are now in a [unified monorepo](layer-3-libraries/rocm-libraries/rocm-libraries-usage.md)

**Math Libraries**:
- [rocBLAS](layer-3-libraries/blas/rocblas-usage.md) / [hipBLAS](layer-3-libraries/blas/hipblas-usage.md) - Linear algebra
- [rocFFT](layer-3-libraries/fft/rocfft-usage.md) - Fast Fourier Transform
- [rocSOLVER](layer-3-libraries/solver/rocsolver-usage.md) - Linear solvers
- [rocSPARSE](layer-3-libraries/sparse/rocsparse-usage.md) - Sparse operations
- [rocRAND](layer-3-libraries/random/rocrand-usage.md) - Random number generation
- [rocThrust](layer-3-libraries/algorithms/rocthrust-usage.md) - Parallel algorithms

**ML/DL Libraries**:
- [MIOpen](layer-3-libraries/ml-primitives/miopen-usage.md) - Deep learning primitives
- [RCCL](layer-3-libraries/communications/rccl-usage.md) - Multi-GPU communication

**Naming Guide**:
- **roc\*** libraries (rocBLAS, rocFFT): AMD-native, best performance
- **hip\*** libraries (hipBLAS, hipFFT): Portable wrappers, CUDA-compatible API

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
- **Layer 2**: Compute platform (ROCm, HIP)
- **Layer 3**: Libraries and primitives
- **Layer 4**: ML frameworks
- **Layer 5**: LLM applications

## ✅ Content Standards

All knowledge base content:
- ✅ **ROCm 7.0+ only** - No support for older versions
- ✅ Tested on ROCm 7.0.2 (latest stable)
- ✅ Includes working code examples
- ✅ Links to official documentation
- ✅ Provides troubleshooting tips
- ✅ Regular maintenance and updates

## 📊 Knowledge Statistics

- **Total Files**: 53+ (continuously expanding)
- **ROCm Version**: 7.0+ (verified on 7.0.2)
- **Last Updated**: 2025-11-05
- **ROCm Support**: 7.0.0, 7.0.2, and future 7.x releases
- **Coverage**: Complete LLM inference, training, deployment, and optimization
- **Build Systems**: TheRock, rocm-libraries monorepo
- **Maintenance**: Automated checks + community contributions

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - Contribution guidelines
- Quality standards and templates
- Testing requirements
- Submission process

## 🔧 Maintenance

### For Maintainers

Run maintenance checks:

```bash
# Check for outdated versions
python -m amd_ai_devtool.maintenance.version_checker

# Test code examples
python -m amd_ai_devtool.maintenance.example_tester

# Regenerate index
python -m amd_ai_devtool.maintenance.index_generator
```

## 📝 Metadata Format

Each knowledge file includes comprehensive metadata:

```yaml
---
layer: "5"
category: "inference"
subcategory: "serving-engines"
title: "Your Topic"
rocm_version: "7.0+"
rocm_verified: "7.0.1"
last_updated: "2025-11-01"
last_verified: "2025-11-01"
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
# Build semantic search index (optional, for developers)
amd-ai-devtool maintenance build-index

# Initialize a new project
amd-ai-devtool init my-project --preset llm-engineer

# Setup IDE integration in existing project
amd-ai-devtool setup --ide cursor --layers 4,5
```

### With MCP Server

The MCP server provides AI assistants with semantic search access to this knowledge:

```python
# Query specific topics
query_amd_knowledge(query="How to optimize vLLM on MI250X?")

# Get related content
get_related_content(topic="distributed training")
```

### Manual Reading

Browse files directly or use [INDEX.md](INDEX.md) for navigation.

## 📚 External Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD Instinct GPUs](https://www.amd.com/en/products/accelerators.html)
- [ROCm GitHub](https://github.com/ROCm)
- [AMD AI Developer Resources](https://www.amd.com/en/developer/resources.html)

## 📄 License

This knowledge base is part of the AMD AI DevTool project, licensed under MIT License.

---

**Need help?** Open an issue or check the [Contributing Guide](../../../CONTRIBUTING.md).
