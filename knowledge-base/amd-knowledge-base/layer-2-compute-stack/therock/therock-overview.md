---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "2"
category: "therock"
subcategory: "build-system"
tags: ["therock", "rocm", "build-system", "hip", "development", "docker"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-03
---

# TheRock: The HIP Environment and ROCm Kit

TheRock is a lightweight open source build system for HIP and ROCm that provides a unified way to build, develop, and package the entire ROCm software stack from source. It's designed for developers who need to build ROCm components, customize builds, or create Docker images with specific ROCm configurations.

**Repository**: [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock)

**License**: MIT

## What is TheRock?

TheRock provides:

- **Unified Build System**: Single CMake-based build system for all ROCm components
- **Component Selection**: Build only what you need with granular feature flags
- **Development Workflow**: Daily driver for developing ROCm components
- **Container Support**: Built-in Dockerfiles and container build support
- **Cross-Platform**: Supports both Linux and Windows
- **Optimization**: CCache support for fast incremental builds
- **Submodule Management**: All ROCm components as git submodules

## Quick Start

### Prerequisites

- CMake 3.20 or later
- Ninja build system (recommended)
- Git with submodule support
- Python 3.8+
- C++ compiler (GCC 11+ or Clang 14+)
- 50+ GB free disk space for full build

### Clone the Repository

```bash
# Clone with submodules
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock

# If already cloned without --recursive
git submodule update --init --recursive
```

### Discovering Your AMD GPU Architecture

Before building, identify your GPU architecture to set the correct `THEROCK_AMDGPU_FAMILIES` flag:

**Linux Tools**:

```bash
# Using amd-smi (recommended)
amd-smi static

# Using rocm-smi
rocm-smi --showproductname

# Using rocm_agent_enumerator
rocm_agent_enumerator

# Using offload-arch (cross-platform)
offload-arch
```

**Windows Tools**:

```powershell
# Using hipinfo
hipinfo

# Using offload-arch
offload-arch
```

**Without existing ROCm installation**, install the `rocm` Python package:

```bash
# Create temporary venv with rocm package
python build_tools/setup_venv.py --index-name nightly \
  --index-subdir gfx110X-all --packages rocm .tmpvenv

# Run offload-arch
.tmpvenv/bin/offload-arch  # Linux
.tmpvenv\Scripts\offload-arch  # Windows

# Clean up
rm -rf .tmpvenv
```

**Common GPU architectures**:

- `gfx90a` - MI200 series (MI210, MI250, MI250X)
- `gfx942` - MI300 series (MI300A, MI300X)
- `gfx110X-all` - RDNA3 (RX 7000 series)
- `gfx103X-all` - RDNA2 (RX 6000 series)

## Building ROCm from Source

### Basic Build (Without CCache)

For simple builds or first-time users:

```bash
# Configure build for your GPU architecture
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942

# Build everything (can take hours)
cmake --build build

# Install to system (optional)
cmake --install build --prefix /opt/rocm
```

**Example for different GPUs**:

```bash
# MI300 series
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942

# MI200 series
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx90a

# RDNA3 (RX 7000 series)
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx110X-all

# Multiple architectures
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES="gfx90a;gfx942"
```

### Optimized Build with CCache (Recommended)

For **frequent rebuilds** and development, use CCache to dramatically speed up compilation:

**Linux Setup**:

```bash
# Install ccache (>= 4.11 required)
sudo apt install ccache  # Ubuntu/Debian
sudo dnf install ccache  # Fedora/RHEL

# Setup ccache configuration with TheRock's helper script
eval "$(./build_tools/setup_ccache.py)"

# Configure build with ccache
cmake -B build -GNinja -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  .

# Build (first build still slow, subsequent builds much faster)
cmake --build build
```

**Key CCache Features**:

- Creates `.ccache` directory in repo root with optimized configuration
- Supports `--offload-compress` for AMDGPU device code compression
- Hard-linking support via `include_file_ctime`
- Safe caching with compiler bootstrapping

**Windows CCache**:

> Note: CCache support on Windows is still under investigation. Not recommended for production use yet.

### Minimal Build (Core Only)

Build only essential components:

```bash
# Reset all features and enable only core runtime
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON

cmake --build build
```

## Configuration Flags

### Group Flags (Enable/Disable Component Categories)

By default, TheRock builds **everything**. Use these flags to disable entire categories:

| Flag | Description |
|------|-------------|
| `-DTHEROCK_ENABLE_ALL=OFF` | Disables all optional components (minimal build) |
| `-DTHEROCK_ENABLE_CORE=OFF` | Disables all core components |
| `-DTHEROCK_ENABLE_COMM_LIBS=OFF` | Disables communication libraries (RCCL) |
| `-DTHEROCK_ENABLE_MATH_LIBS=OFF` | Disables math libraries (BLAS, FFT, etc.) |
| `-DTHEROCK_ENABLE_ML_LIBS=OFF` | Disables ML libraries (MIOpen, hipDNN) |
| `-DTHEROCK_ENABLE_PROFILER=OFF` | Disables profilers |
| `-DTHEROCK_RESET_FEATURES=ON` | Forces minimal build (overrides defaults) |

**Example - ML-focused build**:

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_ML_LIBS=ON \
  -DTHEROCK_ENABLE_MATH_LIBS=ON
```

### Individual Component Flags

Fine-grained control over specific components:

| Flag | Description |
|------|-------------|
| `-DTHEROCK_ENABLE_COMPILER=ON` | GPU and host compiler toolchain |
| `-DTHEROCK_ENABLE_HIPIFY=ON` | CUDA to HIP translation tool |
| `-DTHEROCK_ENABLE_CORE_RUNTIME=ON` | Core runtime and tools |
| `-DTHEROCK_ENABLE_HIP_RUNTIME=ON` | HIP runtime |
| `-DTHEROCK_ENABLE_OCL_RUNTIME=ON` | OpenCL runtime |
| `-DTHEROCK_ENABLE_ROCPROFV3=ON` | ROCm Profiler v3 |
| `-DTHEROCK_ENABLE_RCCL=ON` | ROCm Communication Collectives Library |
| `-DTHEROCK_ENABLE_PRIM=ON` | ROCm Primitives library |
| `-DTHEROCK_ENABLE_BLAS=ON` | rocBLAS (Basic Linear Algebra) |
| `-DTHEROCK_ENABLE_RAND=ON` | rocRAND (Random Number Generation) |
| `-DTHEROCK_ENABLE_SOLVER=ON` | rocSOLVER (Linear Algebra Solvers) |
| `-DTHEROCK_ENABLE_SPARSE=ON` | rocSPARSE (Sparse Linear Algebra) |
| `-DTHEROCK_ENABLE_MIOPEN=ON` | MIOpen (Deep Learning Primitives) |
| `-DTHEROCK_ENABLE_HIPDNN=ON` | hipDNN (DNN API wrapper) |

**Example - Compiler toolchain only**:

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_COMPILER=ON
```

**Example - Deep learning stack**:

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_MIOPEN=ON \
  -DTHEROCK_ENABLE_BLAS=ON \
  -DTHEROCK_ENABLE_RAND=ON
```

### External Source Configuration

Use external sources for specific components instead of submodules:

| Flag | Description |
|------|-------------|
| `-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON` | Use external composable-kernel |
| `-DTHEROCK_USE_EXTERNAL_RCCL=ON` | Use external RCCL |
| `-DTHEROCK_USE_EXTERNAL_RCCL_TESTS=ON` | Use external RCCL tests |
| `-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=<path>` | Path to composable-kernel |
| `-DTHEROCK_RCCL_SOURCE_DIR=<path>` | Path to RCCL |
| `-DTHEROCK_RCCL_TESTS_SOURCE_DIR=<path>` | Path to RCCL tests |

**Example - Use local RCCL development**:

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_USE_EXTERNAL_RCCL=ON \
  -DTHEROCK_RCCL_SOURCE_DIR=/path/to/my/rccl
```

### Additional Build Options

| Flag | Description |
|------|-------------|
| `-DTHEROCK_ENABLE_MPI=ON` | Enable MPI support in components (requires MPI installed) |
| `-DBUILD_TESTING=ON/OFF` | Enable/disable testing infrastructure |

## Running Tests

TheRock includes comprehensive testing infrastructure:

```bash
# Enable testing during configure
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DBUILD_TESTING=ON

# Build
cmake --build build

# Run all tests
ctest --test-dir build

# Run tests with verbose output
ctest --test-dir build --output-on-failure

# Run specific test
ctest --test-dir build -R <test_name>

# Run tests in parallel
ctest --test-dir build -j $(nproc)
```

**Test Categories**:

- **Build integrity tests**: Verify build system correctness (enabled by default)
- **Functional tests**: Test components on actual GPUs (requires GPU access)
- **Unit tests**: Component-specific unit tests

## Development Workflows

### Using TheRock as a Daily Driver

TheRock is designed to be used as a development environment for any ROCm component:

**1. Develop a specific component**:

```bash
# Clone with submodules
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock

# Navigate to component (e.g., rocBLAS)
cd math-libs/rocBLAS

# Make your changes
vim library/src/blas1/rocblas_axpy.cpp

# Go back to root and rebuild
cd ../..
cmake --build build --target rocblas
```

**2. Use external source for active development**:

```bash
# Clone the component you want to develop separately
git clone https://github.com/ROCm/rocBLAS.git ~/dev/rocBLAS

# Configure TheRock to use your local version
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_USE_EXTERNAL_ROCBLAS=ON \
  -DTHEROCK_ROCBLAS_SOURCE_DIR=~/dev/rocBLAS

# Now changes in ~/dev/rocBLAS are automatically picked up
cmake --build build --target rocblas
```

**3. Test your changes**:

```bash
# Run component-specific tests
ctest --test-dir build -R rocblas

# Install locally for testing
cmake --install build --prefix /tmp/rocm-test
export PATH=/tmp/rocm-test/bin:$PATH
export LD_LIBRARY_PATH=/tmp/rocm-test/lib:$LD_LIBRARY_PATH
```

### Incremental Builds

With CCache configured, incremental builds are very fast:

```bash
# Make a small change
vim core/HIP/src/hip_runtime.cpp

# Rebuild only what changed (seconds, not hours)
cmake --build build

# Rebuild specific target
cmake --build build --target hip_runtime
```

### Debugging Builds

```bash
# Build with debug symbols
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DCMAKE_BUILD_TYPE=Debug

# Verbose build output
cmake --build build --verbose

# See what flags are used
cmake --build build -- VERBOSE=1
```

## Building Docker Images with TheRock

TheRock includes Dockerfiles for creating custom ROCm containers:

### Using Provided Dockerfiles

TheRock includes Dockerfiles in the `dockerfiles/` directory:

```bash
# List available Dockerfiles
ls dockerfiles/

# Example Dockerfiles:
# - rocm-base.dockerfile      - Base ROCm installation
# - rocm-dev.dockerfile       - Development environment
# - pytorch.dockerfile        - PyTorch with ROCm
```

### Building a Base ROCm Container

```bash
# Build base ROCm image from TheRock build
docker build -f dockerfiles/rocm-base.dockerfile \
  --build-arg AMDGPU_FAMILIES=gfx942 \
  -t my-rocm:latest .

# Run the container
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  my-rocm:latest
```

### Building Development Container

Create a custom Dockerfile that builds ROCm from TheRock:

```dockerfile
# Dockerfile
FROM ubuntu:22.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone TheRock
WORKDIR /build
RUN git clone --recursive https://github.com/ROCm/TheRock.git

# Build ROCm stack
WORKDIR /build/TheRock
RUN cmake -B build -GNinja . \
    -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
    -DTHEROCK_ENABLE_ALL=OFF \
    -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
    -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
    -DTHEROCK_ENABLE_MATH_LIBS=ON \
    && cmake --build build \
    && cmake --install build --prefix /opt/rocm

# Set environment
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Clean up build files
RUN rm -rf /build

WORKDIR /workspace
CMD ["/bin/bash"]
```

```bash
# Build the image
docker build -t rocm-dev:custom .

# Run with GPU access
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --ipc=host --shm-size 16G \
  -v $(pwd):/workspace \
  rocm-dev:custom
```

### Building ML Framework Container

```dockerfile
# Dockerfile.pytorch
FROM ubuntu:22.04 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build ROCm with ML libs
WORKDIR /build/TheRock
COPY . .
RUN cmake -B build -GNinja . \
    -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
    -DTHEROCK_ENABLE_ML_LIBS=ON \
    -DTHEROCK_ENABLE_MATH_LIBS=ON \
    && cmake --build build \
    && cmake --install build --prefix /opt/rocm

# Runtime stage
FROM ubuntu:22.04

# Copy ROCm from builder
COPY --from=builder /opt/rocm /opt/rocm

# Install Python and PyTorch
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with ROCm
RUN pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0

# Install ML packages
RUN pip3 install transformers accelerate datasets

# Set environment
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV PYTORCH_ROCM_ARCH=gfx942

WORKDIR /workspace
CMD ["/bin/bash"]
```

### Multi-GPU Container Setup

```bash
# Run with all GPUs
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 32G \
  rocm-dev:custom

# Run with specific GPU
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --group-add video \
  --ipc=host \
  -e ROCM_VISIBLE_DEVICES=0 \
  rocm-dev:custom
```

## Best Practices

### Build System

**1. Use CCache for Development**:

```bash
# Always set up ccache for repeated builds
eval "$(./build_tools/setup_ccache.py)"
```

**2. Start Minimal, Add What You Need**:

```bash
# Don't build everything if you only need HIP
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON
```

**3. Check Configuration Report**:

```bash
# CMake prints enabled/disabled features report
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942
# Look for the feature summary in output
```

**4. Parallel Builds**:

```bash
# Use all available cores
cmake --build build -j $(nproc)

# Or limit cores to prevent OOM
cmake --build build -j 8
```

### Docker Builds

**1. Multi-stage Builds**:

- Separate build and runtime stages to reduce image size
- Copy only necessary artifacts to runtime stage

**2. Layer Caching**:

```dockerfile
# Install dependencies first (changes less frequently)
RUN apt-get update && apt-get install -y deps...

# Copy source and build (changes more frequently)
COPY . .
RUN cmake --build build
```

**3. Resource Requirements**:

- Minimum 16GB RAM for building
- 50GB+ disk space for full build
- Use `--shm-size` for multi-GPU containers

## Troubleshooting

### Build Failures

**1. Submodule issues**:

```bash
# Update submodules if build fails
git submodule update --init --recursive

# Force clean submodules
git submodule foreach --recursive git clean -xfd
git submodule foreach --recursive git reset --hard
```

**2. CMake configuration errors**:

```bash
# Clear CMake cache
rm -rf build/CMakeCache.txt build/CMakeFiles

# Or start fresh
rm -rf build
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=gfx942
```

**3. Compiler errors**:

```bash
# Ensure you have recent enough compiler
gcc --version  # Should be 11+
clang --version  # Should be 14+

# On Ubuntu 20.04, install newer GCC
sudo apt install gcc-11 g++-11
export CC=gcc-11 CXX=g++-11
```

**4. Out of memory during build**:

```bash
# Limit parallel jobs
cmake --build build -j 4

# Or build specific components
cmake --build build --target hip_runtime
cmake --build build --target rocblas
```

### Runtime Issues

**1. GPU not detected**:

```bash
# Check if GPU is visible
rocm-smi

# If command not found, add to PATH
export PATH=/opt/rocm/bin:$PATH

# Check HSA runtime
rocminfo
```

**2. Library loading errors**:

```bash
# Set library path
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Or use installed ROCm
export LD_LIBRARY_PATH=/path/to/TheRock/build/lib:$LD_LIBRARY_PATH
```

**3. Wrong GPU architecture**:

```bash
# Verify your GPU architecture
rocminfo | grep "Name:"

# Rebuild for correct architecture
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=<your-arch>
cmake --build build
```

### Container Issues

**1. Container doesn't see GPU**:

```bash
# Check device mounting
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    rocm-dev:custom rocm-smi

# Verify permissions
ls -l /dev/kfd /dev/dri/render*

# Add user to video group on host
sudo usermod -a -G video $USER
# Log out and back in
```

**2. Out of memory in container**:

```bash
# Increase shared memory
docker run --shm-size 32G ...

# Or use host IPC namespace
docker run --ipc=host ...
```

**3. Slow build in container**:

```bash
# Mount ccache directory from host
docker run -v ~/.ccache:/root/.ccache ...

# Or use build context with ccache
docker build --build-arg CCACHE_DIR=/ccache ...
```

## Component Architecture

TheRock organizes ROCm components into logical groups:

```text
TheRock/
├── base/              # Foundation components
├── compiler/          # LLVM/Clang toolchain
├── core/             # Core runtime (HIP, OpenCL)
├── math-libs/        # rocBLAS, rocFFT, rocSPARSE, rocSOLVER, rocRAND
├── ml-libs/          # MIOpen, hipDNN
├── comm-libs/        # RCCL (collective communications)
├── profiler/         # rocprofv3
├── build_tools/      # Build scripts and utilities
├── dockerfiles/      # Container definitions
└── examples/         # Sample applications
```

## Advanced Topics

### Building External Projects (PyTorch, etc.)

```bash
# TheRock includes external-builds/ for major frameworks
cd external-builds/pytorch

# Follow instructions to build PyTorch with your TheRock ROCm
cmake -B build -DROCM_PATH=/path/to/TheRock/build
```

### Contributing to ROCm Components

1. Fork the specific component repository
2. Clone TheRock and configure with your fork:

```bash
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock
cmake -B build -GNinja . \
  -DTHEROCK_USE_EXTERNAL_<COMPONENT>=ON \
  -DTHEROCK_<COMPONENT>_SOURCE_DIR=/path/to/your/fork
```

3. Make changes in your fork
4. Build and test with TheRock
5. Submit PR to component repository

### Git Submodule Management

```bash
# Update all submodules to latest
git submodule update --remote --recursive

# Update specific submodule
git submodule update --remote math-libs/rocBLAS

# Switch submodule to specific branch
cd math-libs/rocBLAS
git checkout develop
cd ../..
git add math-libs/rocBLAS
git commit -m "Update rocBLAS to develop branch"
```

## References and Resources

### Official Documentation

- **TheRock GitHub**: [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock)
- **Contribution Guidelines**: [https://github.com/ROCm/TheRock/blob/main/CONTRIBUTING.md](https://github.com/ROCm/TheRock/blob/main/CONTRIBUTING.md)
- **Development Guide**: Check `docs/` in the repository
- **Build System Documentation**: Detailed info in `docs/build-system.md`
- **Environment Setup**: Platform-specific setup in `docs/environment-setup.md`

### ROCm Documentation

- **ROCm Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)
- **HIP Programming Guide**: [https://rocm.docs.amd.com/projects/HIP](https://rocm.docs.amd.com/projects/HIP)
- **ROCm Libraries**: Component-specific documentation

### Community

- **GitHub Issues**: [https://github.com/ROCm/TheRock/issues](https://github.com/ROCm/TheRock/issues)
- **Discussions**: [https://github.com/ROCm/TheRock/discussions](https://github.com/ROCm/TheRock/discussions)
- **ROCm Blogs**: Latest updates and tutorials

### Release Information

- **Build Artifacts**: Nightly and release builds available
- **Releases Page**: [https://github.com/ROCm/TheRock/releases](https://github.com/ROCm/TheRock/releases)
- **GPU Support Roadmap**: Check documentation for supported architectures

## Summary

TheRock is the **essential tool** for:

- **Developers** building or modifying ROCm components
- **Researchers** needing custom ROCm builds for specific hardware
- **DevOps teams** creating optimized container images
- **Organizations** requiring reproducible ROCm builds

**Key Advantages**:

✓ **Unified build system** - One command to build entire ROCm stack  
✓ **Granular control** - Select exactly what you need  
✓ **Fast development** - CCache support for rapid iteration  
✓ **Flexible deployment** - From source to containers  
✓ **Open source** - MIT licensed, community-driven  
✓ **Cross-platform** - Linux and Windows support  

**Get Started**:

```bash
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock
eval "$(./build_tools/setup_ccache.py)"
cmake -B build -GNinja . -DTHEROCK_AMDGPU_FAMILIES=<your-gpu>
cmake --build build
```

