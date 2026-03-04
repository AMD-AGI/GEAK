---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "2"
category: "rocm"
subcategory: "overview"
tags: ["rocm", "compute-stack", "overview", "architecture"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
---

# ROCm Compute Stack Overview

ROCm (Radeon Open Compute) is AMD's complete **open-source** software stack for GPU computing, designed for high-performance computing (HPC), artificial intelligence (AI), scientific computing, and computer-aided design (CAD). ROCm is powered by AMD's **Heterogeneous-computing Interface for Portability (HIP)**, an open-source C++ GPU programming environment.

**This documentation targets ROCm 7.0+ only.**

**Latest Version**: ROCm 7.0.2 
**Official Repository**: [https://github.com/ROCm/ROCm](https://github.com/ROCm/ROCm)  
**Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)  
**Build System**: [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock)  
**Systems Monorepo**: [https://github.com/ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)  
**Libraries Monorepo**: [https://github.com/ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)

## ROCm Stack Architecture

The ROCm stack is organized in layers:

```
┌─────────────────────────────────────────────┐
│  Applications & Frameworks                   │
│  (PyTorch, TensorFlow, JAX, vLLM, etc.)     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Libraries & Primitives                      │
│  (rocBLAS, rocFFT, MIOpen, RCCL, etc.)      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Programming Models                          │
│  (HIP, OpenMP, OpenCL)                      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Runtime & Compiler                          │
│  (HIP Runtime, ROCm Compiler, LLVM)         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  System Layer                                │
│  (ROCr Runtime, ROCt Thunk, KFD Driver)     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Hardware                                    │
│  (AMD GPUs: MI300, MI250, MI210, etc.)      │
└─────────────────────────────────────────────┘
```

## Key Components

### 1. HIP (Heterogeneous-compute Interface for Portability)

AMD's primary GPU programming language:
- C++-based GPU programming interface
- Portable between AMD and NVIDIA GPUs
- CUDA-compatible API surface
- Part of ROCm core
- Now in [rocm-systems monorepo](https://github.com/ROCm/rocm-systems)

**Use cases:**
- Custom kernel development
- Performance-critical code
- GPU algorithm implementation

**Documentation**: See [HIP basics](../hip/hip-basics.md) | [rocm-systems](../rocm-systems/rocm-systems-usage.md)

### 2. ROCm Runtime

Low-level runtime environment:
- **ROCr**: Runtime API for GPU management
- **ROCt**: Thunk library for kernel interface
- **KFD**: Kernel Fusion Driver for GPU access

### 3. ROCm Compiler

Based on LLVM:
- Optimizing compiler for AMD GPUs
- Supports HIP, OpenCL, OpenMP
- Architecture-specific optimizations

### 4. ROCm Libraries

High-performance libraries:
- **rocBLAS**: BLAS (linear algebra)
- **rocFFT**: Fast Fourier Transform
- **rocRAND**: Random number generation
- **rocSOLVER**: LAPACK functionality
- **rocSPARSE**: Sparse linear algebra
- **MIOpen**: Deep learning primitives
- **RCCL**: Collective communications

### 5. ROCm Tools

Development, profiling, and debugging:
- **rocprof**: GPU profiler
- **rocgdb**: GPU debugger
- **rocm-smi**: System management interface
- **roctracer**: API tracing tool
- **rocprofiler-sdk**: Profiling SDK
- **rocprofiler-systems**: System-level profiler
- **rocminfo**: System information tool

> **Note**: Most ROCm tools are now in the [rocm-systems monorepo](../rocm-systems/rocm-systems-usage.md)

## Building ROCm from Source

### TheRock Build System

**Important**: ROCm is now built using [TheRock](https://github.com/ROCm/TheRock), a new open-source build platform featuring:
- Unified CMake build system with bundled dependencies
- Cross-platform support (Linux and Windows)
- Selective component building
- Integration with rocm-libraries monorepo

**For building ROCm from source**, use TheRock:

```bash
# Clone TheRock with submodules
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock

# Build for your GPU architecture
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942  # MI300
cmake --build build

# Or with CCache for faster rebuilds
eval "$(./build_tools/setup_ccache.py)"
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build
```

See [TheRock documentation](../../therock/therock-overview.md) for detailed build instructions.

### ROCm Repository Structure

The [ROCm repository](https://github.com/ROCm/ROCm) contains:
- **Manifest files** (`default.xml`) for ROCm releases
- **Release information** and changelogs
- **Documentation** in the `/docs` folder
- **Not source code** - use TheRock to build from source

The `default.xml` manifest file contains information for all repositories and commits used to build each ROCm release.

## Installation Options

### Option 1: Package Manager (Recommended)

Install pre-built ROCm packages:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install rocm-dev rocm-libs

# Install specific components
sudo apt install rocblas rocfft miopen-hip rccl

# For ML/AI workloads
sudo apt install rocm-ml-libraries
```

See [ROCm Installation Guide](./rocm-installation.md) for detailed instructions.

### Option 2: Build from Source

Use TheRock to build ROCm from source:

```bash
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942
cmake --build build
```

### Option 3: Docker Containers

Use official ROCm Docker images:

```bash
# PyTorch with ROCm
docker pull rocm/pytorch:latest

# TensorFlow with ROCm
docker pull rocm/tensorflow:latest

# Base ROCm development
docker pull rocm/dev-ubuntu-22.04:latest
```

## ROCm 7.0+ Versions

> **Note**: This project targets ROCm 7.0+ only. For older versions, refer to official ROCm documentation.

### ROCm 7.0.2 (Latest Stable)
- **Status**: Latest stable release - **Recommended**
- Latest bug fixes and stability improvements
- Enhanced vLLM 0.4.x+ support
- Improved PyTorch 2.4+ performance
- Better multi-GPU scaling with RCCL
- Production-ready

### ROCm 7.0.0 (Major Release)
- **Status**: Major release with significant updates
- Significant performance improvements
- Flash Attention 2 support
- Enhanced ML framework integration
- Improved Windows support via TheRock
- Better vLLM integration

### Future Releases
- Check [ROCm Releases](https://github.com/ROCm/ROCm/releases) for upcoming versions
- Follow [ROCm GitHub](https://github.com/ROCm/ROCm) for development updates

## Component Compatibility (ROCm 7.0+)

| Component | ROCm 7.0.0 | ROCm 7.0.2 | Notes |
|-----------|------------|------------|-------|
| PyTorch 2.3+ | ✓ | ✓ | Full support |
| PyTorch 2.4+ | ✓ | ✓ | Recommended |
| TensorFlow 2.15+ | ✓ | ✓ | Full support |
| TensorFlow 2.16+ | ✓ | ✓ | Latest features |
| JAX 0.4+ | ✓ | ✓ | Full support |
| vLLM 0.4+ | ✓ | ✓ | Optimized |
| vLLM 0.5+ | Limited | ✓ | Latest features |
| SGLang 0.2+ | ✓ | ✓ | Full support |
| Flash Attention 2 | ✓ | ✓ | Full support |

> For frameworks requiring older ROCm versions, please upgrade to ROCm 7.0+

## Development Workflow

### 1. Kernel Development (HIP)
```
Write HIP kernel → Compile with hipcc → Profile with rocprof → Optimize
```

### 2. Library Usage
```
Choose library → Link with hipcc → Use API → Benchmark
```

### 3. Framework Development
```
Install framework → Write Python/C++ → Train/Infer → Deploy
```

### 4. LLM Deployment
```
Setup TheRock → Configure vLLM/SGLang → Load model → Serve
```

## Environment Setup

```bash
# Set ROCm paths
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# HIP configuration
export HIP_PLATFORM=amd
export HIP_VISIBLE_DEVICES=0,1,2,3

# For specific GPU architecture
export PYTORCH_ROCM_ARCH=gfx90a  # MI250X
export PYTORCH_ROCM_ARCH=gfx942  # MI300
```

## Verification

```bash
# Check ROCm installation
rocminfo

# Check HIP
hipconfig

# Check GPU
rocm-smi

# Check libraries
ls /opt/rocm/lib/librocblas.so
ls /opt/rocm/lib/libMIOpen.so

# Test HIP
cat > test.cpp << EOF
#include <hip/hip_runtime.h>
#include <iostream>
int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Devices: " << deviceCount << std::endl;
    return 0;
}
EOF

hipcc test.cpp -o test && ./test
```

## Best Practices

### Installation & Setup

1. **Use Package Manager for Production**: Install official ROCm packages for stability
2. **Use Docker for Development**: Isolated environments, easy cleanup
3. **Build from Source for Customization**: Use TheRock when you need specific configurations

### Development

1. **Match ROCm version to GPU**: Check compatibility matrix before installation
2. **Keep components in sync**: Don't mix ROCm versions across libraries
3. **Use official containers**: Consistent, tested environments
4. **Profile before optimizing**: Use `rocprof`, `rocm-smi`, `roctracer`
5. **Set environment variables**: Ensure proper `PATH` and `LD_LIBRARY_PATH`

### Version Selection (ROCm 7.0+ Only)

- **Production workloads**: ROCm 7.0.2 (latest stable, recommended)
- **Development/Testing**: ROCm 7.0.2 or build from TheRock
- **Bleeding edge**: Build from TheRock develop branch
- **Note**: This project does not support ROCm < 7.0

### Performance

1. **Set correct GPU architecture**: Export `PYTORCH_ROCM_ARCH` or `AMDGPU_TARGETS`
2. **Enable HSA_OVERRIDE_GFX_VERSION**: If needed for unsupported GPUs
3. **Use appropriate batch sizes**: Based on GPU memory
4. **Monitor GPU utilization**: With `rocm-smi` to ensure full GPU usage

## References

### Official Resources

- **ROCm GitHub**: [https://github.com/ROCm/ROCm](https://github.com/ROCm/ROCm)
- **ROCm Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)
- **TheRock Build System**: [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock)
- **ROCm Systems Monorepo**: [https://github.com/ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)
- **ROCm Libraries Monorepo**: [https://github.com/ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)

### Documentation

- **HIP Programming Guide**: [https://rocm.docs.amd.com/projects/HIP](https://rocm.docs.amd.com/projects/HIP)
- **Installation Guide**: [https://rocm.docs.amd.com/projects/install-on-linux](https://rocm.docs.amd.com/projects/install-on-linux)
- **GPU Support Matrix**: [https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
- **Release Notes**: [https://github.com/ROCm/ROCm/releases](https://github.com/ROCm/ROCm/releases)

### Community

- **GitHub Discussions**: [https://github.com/ROCm/ROCm/discussions](https://github.com/ROCm/ROCm/discussions)
- **GitHub Issues**: [https://github.com/ROCm/ROCm/issues](https://github.com/ROCm/ROCm/issues)
- **AMD Developer Community**: [https://community.amd.com](https://community.amd.com)

