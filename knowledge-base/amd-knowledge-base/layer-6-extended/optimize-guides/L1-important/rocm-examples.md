---
tags: ["optimization", "performance", "hip", "examples", "tutorials", "libraries"]
priority: "L1-important"
source_url: "https://github.com/ROCm/rocm-examples"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# ROCm Examples Repository

## Overview

The ROCm Examples repository is a comprehensive collection of code samples designed to help users learn and work with the ROCm software stack. It caters to both newcomers seeking foundational knowledge and advanced developers exploring sophisticated applications.

## Repository Description

"This repository is a collection of examples to enable new users to start using ROCm, as well as provide more advanced examples for experienced users." The examples are systematically organized into several major categories.

## Main Categories

**HIP-Basic**: Self-contained recipes demonstrating HIP runtime functionality, including topics like device querying, memory management, kernel launching, and stream operations.

**HIP-Doc**: Example code from official HIP documentation, organized by programming concepts and organized for both reference and standalone use.

**Libraries**: Implementations using ROCm libraries including hipBLAS, hipBLASLt, hipCUB, hipRAND, rocBLAS, rocFFT, rocRAND, and rocSOLVER.

**Applications**: Practical GPU-accelerated implementations including bitonic sort, convolution, Floyd-Warshall algorithm, histogram computation, and Monte Carlo estimation.

**AI**: Instructions and examples for using ROCm in AI/ML workflows, particularly with MIGraphX quantization.

**Tutorials**: Accompanying code for HIP tutorials found in official documentation.

## Prerequisites

### Linux Requirements
- CMake (minimum version 3.21)
- GNU Make (optional alternative build system)
- ROCm (version 7.x.x or later)

### Windows Requirements
- Visual Studio 2019 or 2022 with C++ workload
- HIP SDK for Windows
- CMake and Ninja (optional for CMake builds)
- Perl (for hipify-related scripts)

## Build Instructions

### Linux with CMake
```bash
git clone https://github.com/ROCm/rocm-examples.git
cd rocm-examples
cmake -S . -B build
cmake --build build
cmake --install build --prefix install
```

For CUDA support: `cmake -S . -B build -D GPU_RUNTIME=CUDA`

### Docker Alternative
Pre-configured Docker images eliminate prerequisite installation while requiring only the host GPU driver:

```bash
docker build . -t rocm-examples -f hip-libraries-rocm-ubuntu.Dockerfile
docker run -it --device /dev/kfd --device /dev/dri rocm-examples bash
```

### Windows with Visual Studio
The repository includes solution files for Visual Studio 2017, 2019, and 2022. Projects can be built directly through the IDE or via MSBuild command-line tools.

## Build Options

CMake supports configuration through these key parameters:

- `GPU_RUNTIME`: Set to "HIP" (default) or "CUDA"
- `CMAKE_HIP_ARCHITECTURES`: Target specific AMD GPU architectures
- `CMAKE_CUDA_ARCHITECTURES`: Target specific NVIDIA compute capabilities

## Repository Structure Highlights

The collection spans diverse topics:

- **Memory management** across device, host, and unified memory spaces
- **Multi-GPU operations** including peer-to-peer communication
- **Asynchronous execution** with streams and events
- **HIP graphs** for command capture and replay
- **Algorithm implementations** with performance optimization techniques

## Contributing

The repository welcomes community contributions. Guidelines are available in the `Docs/CONTRIBUTING.md` file.

## License

The project is distributed under the MIT License, permitting broad usage and modification rights.
