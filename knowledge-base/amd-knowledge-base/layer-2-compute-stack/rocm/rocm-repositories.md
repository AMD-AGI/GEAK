Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "2"
category: "rocm"
subcategory: "repositories"
tags: ["rocm", "github", "repositories", "source-code", "development", "open-source"]
rocm_version: "7.0+"
rocm_verified: "7.1"
therock_included: true
last_updated: 2025-11-14
difficulty: "beginner"
estimated_time: "15min"
---

# ROCm GitHub Repositories - Complete Reference

Comprehensive guide to all official AMD ROCm repositories on GitHub, organized by the 5-layer AMD AI stack architecture.

**⚠️ Repository URLs and organization may change. Always search https://github.com/ROCm for the most current list.**

## Overview

AMD maintains 40+ open-source repositories for the ROCm ecosystem on GitHub under the [ROCm organization](https://github.com/ROCm).

**Main ROCm Organization**: https://github.com/ROCm

### Quick Links

- **ROCm Platform**: https://github.com/ROCm/ROCm (Main repo - 5.8k+ stars)
- **HIP Programming**: https://github.com/ROCm/HIP (4.2k+ stars)
- **All Repositories**: https://github.com/orgs/ROCm/repositories

---

## Layer 1: Hardware & Architecture

**Focus**: Hardware specifications, architecture documentation, performance counters

While Layer 1 is primarily documentation, these repositories contain architecture-related code and tools:

### Core Repositories

| Repository | Description | Key Use Cases |
|-----------|-------------|---------------|
| [ROCm/rocminfo](https://github.com/ROCm/rocminfo) | GPU device information tool | Query GPU capabilities, architecture details |
| [ROCm/rocm_smi_lib](https://github.com/ROCm/rocm_smi_lib) | System Management Interface library | GPU monitoring, power management, device queries |

**Documentation**: Hardware specs are documented in [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/)

---

## Layer 2: Compute Stack & Programming

**Focus**: ROCm platform, HIP programming, compilers, runtime, profiling tools

### Core Platform

| Repository | Description | Stars | Key Features |
|-----------|-------------|-------|--------------|
| **[ROCm/ROCm](https://github.com/ROCm/ROCm)** | Main ROCm platform repository | 5.8k+ | Installation, documentation, releases |
| **[ROCm/HIP](https://github.com/ROCm/HIP)** | HIP programming language | 4.2k+ | CUDA compatibility, GPU programming |
| **[ROCm/TheRock](https://github.com/ROCm/TheRock)** | Modern build system for ROCm | 525+ | Windows support, unified builds |

### Compiler & Runtime

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/llvm-project](https://github.com/ROCm/llvm-project) | AMD's LLVM/Clang fork | HIP compiler, AMDGPU backend |
| [ROCm/HIPCC](https://github.com/ROCm/HIPCC) | HIP compiler driver | Compile HIP programs |
| [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY) | CUDA to HIP converter | 628+ stars, automatic porting |
| [ROCm/hipamd](https://github.com/ROCm/hipamd) | HIP implementation for AMD | Runtime support |
| [ROCm/ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime) | ROCr Runtime (HSA) | Low-level GPU runtime |
| [ROCm/ROCT-Thunk-Interface](https://github.com/ROCm/ROCT-Thunk-Interface) | Kernel driver interface | User-space to kernel communication |

### Development Tools & Profiling

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocprofiler](https://github.com/ROCm/rocprofiler) | GPU profiling tool | Performance analysis, tracing |
| [ROCm/roctracer](https://github.com/ROCm/roctracer) | Runtime tracing library | API call tracing, debugging |
| [ROCm/rocgdb](https://github.com/ROCm/rocgdb) | ROCm debugger (GDB) | Kernel debugging |
| [ROCm/rocr_debug_agent](https://github.com/ROCm/rocr_debug_agent) | Debug agent for ROCr | Enhanced debugging |

### Unified Monorepos

| Repository | Description | Contents |
|-----------|-------------|----------|
| **[ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)** | Systems projects monorepo | Runtime, profiler, tracer, SMI |

---

## Layer 3: Libraries & Computational Primitives

**Focus**: Math libraries, ML primitives, communication libraries

### Main Monorepo

| Repository | Description | Stars | Contents |
|-----------|-------------|-------|----------|
| **[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)** | Unified libraries monorepo | - | All math & ML libraries (migration in progress) |

### Linear Algebra Libraries

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) | Basic Linear Algebra Subprograms | BLAS level 1/2/3 operations |
| [ROCm/hipBLAS](https://github.com/ROCm/hipBLAS) | CUDA-compatible BLAS wrapper | Drop-in replacement for cuBLAS |
| [ROCm/rocSOLVER](https://github.com/ROCm/rocSOLVER) | LAPACK-compatible solver | Linear system solvers |
| [ROCm/hipSOLVER](https://github.com/ROCm/hipSOLVER) | CUDA-compatible solver wrapper | Drop-in for cuSOLVER |

### FFT & Signal Processing

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocFFT](https://github.com/ROCm/rocFFT) | Fast Fourier Transforms | 1D/2D/3D FFT |
| [ROCm/hipFFT](https://github.com/ROCm/hipFFT) | CUDA-compatible FFT wrapper | Drop-in for cuFFT |

### Sparse Operations

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocSPARSE](https://github.com/ROCm/rocSPARSE) | Sparse linear algebra | Sparse matrix operations |
| [ROCm/hipSPARSE](https://github.com/ROCm/hipSPARSE) | CUDA-compatible sparse wrapper | Drop-in for cuSPARSE |
| [ROCm/hipSPARSELt](https://github.com/ROCm/hipSPARSELt) | Structured sparsity | Drop-in for cuSPARSELt |

### Random Number Generation

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocRAND](https://github.com/ROCm/rocRAND) | Random number generation | Multiple RNG algorithms |
| [ROCm/hipRAND](https://github.com/ROCm/hipRAND) | CUDA-compatible RNG wrapper | Drop-in for cuRAND |

### Parallel Algorithms

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocPRIM](https://github.com/ROCm/rocPRIM) | GPU parallel primitives | Device-level algorithms |
| [ROCm/rocThrust](https://github.com/ROCm/rocThrust) | C++ parallel algorithms | STL-like parallel algorithms |
| [ROCm/hipCUB](https://github.com/ROCm/hipCUB) | CUDA-compatible primitives | Drop-in for CUB |

### ML Primitives & Kernels

| Repository | Description | Stars | Key Features |
|-----------|-------------|-------|---------------|
| **[ROCm/MIOpen](https://github.com/ROCm/MIOpen)** | ML primitives library | 1.2k+ | Convolutions, pooling, activation, normalization |
| [ROCm/hipDNN](https://github.com/ROCm/hipDNN) | CUDA-compatible DNN wrapper | Drop-in for cuDNN |
| **[ROCm/composable_kernel](https://github.com/ROCm/composable_kernel)** | High-performance ML kernels | 481+ | Optimized kernels for ML ops |
| [ROCm/Tensile](https://github.com/ROCm/Tensile) | Tensor contraction library | GEMM kernel generation |
| [ROCm/rocWMMA](https://github.com/ROCm/rocWMMA) | Wave Matrix Multiply-Accumulate | Matrix operations using MFMA |

### Communication Libraries

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rccl](https://github.com/ROCm/rccl) | Collective communications | Multi-GPU, multi-node communication |

### Compiler Tools

| Repository | Description | Stars | Key Features |
|-----------|-------------|-------|---------------|
| [ROCm/triton](https://github.com/ROCm/triton) | Triton compiler for AMD | 136+ | Python-based kernel development |

---

## Layer 4: ML Frameworks

**Focus**: PyTorch, TensorFlow, JAX integration with ROCm

### Framework Repositories

| Repository | Description | Stars | Key Features |
|-----------|-------------|-------|---------------|
| **[ROCm/pytorch](https://github.com/ROCm/pytorch)** | PyTorch with ROCm support | 244+ | Official ROCm PyTorch builds |
| **[ROCm/tensorflow-upstream](https://github.com/ROCm/tensorflow-upstream)** | TensorFlow ROCm port | 698+ | TensorFlow 2.x on ROCm |
| [ROCm/jax](https://github.com/ROCm/jax) | JAX fork for ROCm | - | JAX with AMD GPU support |
| [ROCm/xla](https://github.com/ROCm/xla) | XLA compiler for ROCm | - | Compiler for ML frameworks |

### Graph Optimization

| Repository | Description | Stars | Key Features |
|-----------|-------------|-------|---------------|
| **[ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX)** | Graph optimization engine | 262+ | Model optimization, ONNX support |

### Supporting Tools

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/aomp](https://github.com/ROCm/aomp) | OpenMP compiler | 230+ stars, offloading support |

---

## Layer 5: LLM & Advanced AI

**Focus**: LLM serving, inference, fine-tuning

### LLM Serving Engines

**Note**: These are ecosystem projects with AMD GPU support, not official ROCm repos

| Repository | Description | Stars | AMD Support |
|-----------|-------------|-------|-------------|
| **[vllm-project/vllm](https://github.com/vllm-project/vllm)** | High-throughput LLM inference | 61.9k+ | ✅ Official ROCm support |
| **[sgl-project/sglang](https://github.com/sgl-project/sglang)** | Structured generation for LLMs | 19.7k+ | ✅ Official ROCm support |

### Fine-tuning & Training

ROCm-compatible fine-tuning is typically done through PyTorch (Layer 4) with libraries like:
- HuggingFace Transformers (compatible via PyTorch ROCm)
- DeepSpeed (partial ROCm support)
- FSDP (native PyTorch, works on ROCm)

---

## Cross-Layer & Utility Repositories

These repositories span multiple layers or provide general utilities:

### Container & Deployment

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/ROCm-docker](https://github.com/ROCm/ROCm-docker) | Official Docker images | Dockerfiles for ROCm containers |

### Documentation & Examples

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocm-examples](https://github.com/ROCm/rocm-examples) | Example code | HIP, ML, library examples |
| [ROCm/rocm-docs-core](https://github.com/ROCm/rocm-docs-core) | Documentation source | Sphinx-based documentation |

### Benchmarking

| Repository | Description | Key Features |
|-----------|-------------|---------------|
| [ROCm/rocm-bandwidth-test](https://github.com/ROCm/rocm-bandwidth-test) | Memory bandwidth testing | PCIe, HBM bandwidth tests |
| [ROCm/hipBench](https://github.com/ROCm/hipBench) | GPU benchmarking suite | Performance benchmarks |

---

## Repository Selection by Use Case

### I want to...

**Write GPU kernels**:
- [ROCm/HIP](https://github.com/ROCm/HIP) - HIP programming language
- [ROCm/triton](https://github.com/ROCm/triton) - Python-based kernel development

**Port CUDA code**:
- [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY) - Automatic CUDA → HIP converter

**Do linear algebra**:
- [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) - Native BLAS
- [ROCm/hipBLAS](https://github.com/ROCm/hipBLAS) - CUDA-compatible wrapper

**Build ML applications**:
- [ROCm/MIOpen](https://github.com/ROCm/MIOpen) - ML primitives
- [ROCm/pytorch](https://github.com/ROCm/pytorch) - PyTorch on ROCm

**Optimize ML models**:
- [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) - Graph optimization

**Serve LLMs**:
- [vllm-project/vllm](https://github.com/vllm-project/vllm) - High-throughput inference
- [sgl-project/sglang](https://github.com/sgl-project/sglang) - Structured generation

**Profile & debug**:
- [ROCm/rocprofiler](https://github.com/ROCm/rocprofiler) - GPU profiling
- [ROCm/roctracer](https://github.com/ROCm/roctracer) - Runtime tracing
- [ROCm/rocgdb](https://github.com/ROCm/rocgdb) - GPU debugging

**Multi-GPU communication**:
- [ROCm/rccl](https://github.com/ROCm/rccl) - Collective communications

**Build ROCm from source**:
- [ROCm/TheRock](https://github.com/ROCm/TheRock) - Modern build system

---

## Repository Naming Conventions

Understanding AMD's naming patterns:

### Prefixes

- **`roc*`**: ROCm-native libraries (rocBLAS, rocFFT, rocSOLVER)
- **`hip*`**: HIP/CUDA compatibility wrappers (hipBLAS, hipFFT, hipSOLVER)
- **`MIOpen`**: ML primitives (like cuDNN)
- **`MI*`**: MI series specific (MIGraphX)

### Patterns

- **Native APIs**: `roc*` libraries are ROCm-native implementations
- **Compatibility Wrappers**: `hip*` libraries wrap `roc*` for CUDA compatibility
- **Example**: `rocBLAS` (native) ← `hipBLAS` (CUDA-compatible wrapper)

---

## Important Migration Notes

### Monorepo Transition (2024+)

AMD is consolidating repositories into monorepos for better management:

1. **rocm-systems** - Runtime, profiler, tracer, SMI (Layer 2)
2. **rocm-libraries** - All math and ML libraries (Layer 3)

**What this means**:
- Individual library repos may become read-only
- Active development moves to monorepos
- Better integration between components
- Unified build system

**Migration Status** (as of Nov 2025):
- ⏳ In progress - some libraries still in individual repos
- ⚠️ Check both monorepo and individual repo for latest code

---

## Getting Started with Repositories

### Clone a Repository

```bash
# Clone main ROCm platform
git clone https://github.com/ROCm/ROCm.git

# Clone HIP
git clone https://github.com/ROCm/HIP.git

# Clone with submodules (if needed)
git clone --recursive https://github.com/ROCm/TheRock.git
```

### Build from Source

Most repositories use CMake:

```bash
cd repository-name
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

For TheRock-based builds:
```bash
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx942
cmake --build build
```

### Find Documentation

Each repository typically has:
- `README.md` - Getting started
- `docs/` - Detailed documentation
- `examples/` - Code examples
- `CHANGELOG.md` - Version history

---

## Contributing to ROCm

### How to Contribute

1. **Read CONTRIBUTING.md** in each repository
2. **Sign Contributor License Agreement (CLA)** - Usually required
3. **Fork & create pull request** following repo guidelines
4. **Test thoroughly** on supported ROCm versions

### Reporting Issues

- **Main Platform Issues**: https://github.com/ROCm/ROCm/issues
- **Library-Specific**: Open issue in specific library repo
- **Security Issues**: See SECURITY.md in each repo

### Community

- **GitHub Discussions**: https://github.com/ROCm/ROCm/discussions
- **Discord**: Check individual repos for community links
- **AMD Developer Community**: https://community.amd.com

---

## Repository Search Tips

### Finding Repositories

```bash
# Search ROCm organization
https://github.com/ROCm?q=<search-term>&type=repositories

# Examples:
https://github.com/ROCm?q=blas&type=repositories
https://github.com/ROCm?q=ml&type=repositories
```

### GitHub Topics

Repositories are tagged with topics:
- `rocm` - ROCm platform
- `hip` - HIP programming
- `gpu` - GPU computing
- `amd` - AMD-specific
- `machine-learning` - ML libraries

---

## Official Resources

### Main Links

- **GitHub Organization**: https://github.com/ROCm
- **Official Documentation**: https://rocm.docs.amd.com
- **Docker Hub**: https://hub.docker.com/u/rocm
- **Release Notes**: https://github.com/ROCm/ROCm/releases
- **Developer Portal**: https://www.amd.com/en/developer.html

### Documentation by Layer

- **Layer 2 (Platform)**: https://rocm.docs.amd.com/projects/install-on-linux
- **Layer 3 (Libraries)**: https://rocm.docs.amd.com/projects/rocm-libraries
- **Layer 4 (Frameworks)**: https://rocm.docs.amd.com/en/latest/how-to/deep-learning-rocm.html
- **Layer 5 (LLMs)**: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization

---

## Quick Reference Table

| Layer | Key Repositories | Purpose |
|-------|-----------------|---------|
| **1: Hardware** | rocminfo, rocm_smi_lib | Device info, monitoring |
| **2: Compute** | ROCm, HIP, HIPIFY, TheRock | Platform, programming, build |
| **3: Libraries** | rocBLAS, rocFFT, MIOpen, composable_kernel | Math, ML primitives |
| **4: Frameworks** | pytorch, tensorflow-upstream, AMDMIGraphX | ML frameworks |
| **5: LLM** | vllm, sglang | LLM inference & serving |

---

**⚠️ Note**: Repository organization and URLs may change as AMD continues to evolve the ROCm ecosystem. Always check https://github.com/ROCm for the most current information.


