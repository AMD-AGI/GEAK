---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "2"
category: "rocm-systems"
subcategory: "monorepo"
tags: ["rocm-systems", "runtime", "profiling", "hip", "monitoring", "monorepo"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
---

# ROCm Systems Usage Guide

The ROCm Systems super-repo consolidates multiple ROCm systems projects into a single repository, providing unified development, CI, and integration for ROCm runtime, profiling, monitoring, and systems-level components.

**Repository**: [https://github.com/ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)

**Status**: Active consolidation (similar to rocm-libraries monorepo)

## What is rocm-systems?

The rocm-systems super-repo brings together ROCm systems projects that were previously maintained in separate GitHub repositories. This includes:

- **Runtime Components**: ROCr Runtime, HIP, CLR
- **Profiling Tools**: rocprof, rocprofiler-sdk, rocprofiler-systems, roctracer
- **Monitoring Tools**: AMD SMI ([see AMD SMI guide](amd-smi-usage.md)), ROCm SMI Library
- **Testing**: HIP tests
- **Communication**: RCCL (ROCm Communication Collectives Library)
- **Other Systems**: rocminfo, rocm-core, rocshmem

## Repository Structure

The rocm-systems monorepo is organized into project folders:

```text
rocm-systems/
├── projects/              # Systems projects released as distinct packages
│   ├── amdsmi/            # AMD System Management Interface
│   ├── aqlprofile/        # AQL (AMD Queue Language) profiler
│   ├── clr/               # Common Language Runtime
│   ├── hip/               # HIP (Heterogeneous-compute Interface for Portability)
│   ├── hipother/          # HIP utilities
│   ├── hip-tests/         # HIP test suite
│   ├── rccl/              # ROCm Communication Collectives Library
│   ├── rdc/               # ROCm Data Center Tool
│   ├── rocm-core/         # ROCm core utilities
│   ├── rocminfo/          # ROCm system information tool
│   ├── rocmsmilib/        # ROCm SMI library
│   ├── rocprofiler/       # ROCm profiler
│   ├── rocprofiler-compute/    # GPU compute profiler
│   ├── rocprofiler-register/   # Profiler registration
│   ├── rocprofiler-sdk/   # Profiler SDK
│   ├── rocprofiler-systems/    # System-level profiler
│   ├── rocrruntime/       # ROCr Runtime (HSA runtime)
│   ├── rocshmem/          # ROCm OpenSHMEM implementation
│   └── roctracer/         # ROCm tracer for API calls
│
└── shared/                # Shared dependencies
    └── amdgpu-windows-interop/  # Windows interop layer
```

### Nomenclature Standards

Project names have been standardized to match released package casing and punctuation:
- Consistent casing: `rocminfo`, `aqlprofile`
- Removed inconsistent camel-casing and underscores
- Standardized across all projects

## Migration Status

### Completed Migrations

The following components have been **fully migrated** to the rocm-systems super-repo:

**Runtime & Execution**:
- ✅ HIP (Heterogeneous-compute Interface for Portability)
- ✅ CLR (Common Language Runtime)
- ✅ ROCr Runtime (HSA Runtime)

**Profiling & Tracing**:
- ✅ rocprofiler
- ✅ rocprofiler-compute
- ✅ rocprofiler-register
- ✅ rocprofiler-sdk
- ✅ rocprofiler-systems
- ✅ roctracer
- ✅ aqlprofile

**Monitoring & Management**:
- ✅ ROCm SMI Library (rocmsmilib)
- ✅ rocminfo
- ✅ rocm-core

**Communication**:
- ✅ RCCL (ROCm Communication Collectives Library)
- ✅ rocshmem

**Testing**:
- ✅ HIP tests

### Pending Migrations

| Component | Status |
|-----------|--------|
| AMD SMI | Pending migration to monorepo (documentation available in [AMD SMI guide](amd-smi-usage.md)) |

## CI/CD Integration

The rocm-systems repo maintains comprehensive CI/CD pipelines:

### CI Systems

1. **Azure CI**: Primary CI for most components
2. **GitHub Actions**: Component-specific workflows
3. **TheRock CI**: Multi-component integration testing

### CI Status per Component

Each project has its own CI pipelines:

| Component | Azure CI | GitHub Actions | Status |
|-----------|----------|----------------|--------|
| HIP | ✓ | ✓ | Completed |
| CLR | ✓ | - | Completed |
| rocprofiler-sdk | ✓ | ✓ | Completed |
| rocprofiler-systems | ✓ | ✓ | Completed |
| roctracer | ✓ | - | Completed |
| RCCL | ✓ | - | Completed |
| ROCr Runtime | ✓ | - | Completed |

All CI status badges and build information available on the [repository README](https://github.com/ROCm/rocm-systems).

## Key Components

### HIP (Heterogeneous-compute Interface for Portability)

**Purpose**: GPU programming language and runtime

**Repository**: Within rocm-systems at `projects/hip/`

**Features**:
- C++ GPU programming interface
- Portable between AMD and NVIDIA GPUs
- CUDA-compatible API surface
- Part of ROCm core

**Documentation**: See [HIP documentation](../hip/hip-basics.md)

### ROCr Runtime

**Purpose**: HSA (Heterogeneous System Architecture) runtime

**Location**: `projects/rocrruntime/`

**Features**:
- Low-level runtime API for GPU management
- HSA-compliant implementation
- Queue management, memory operations
- Device discovery and initialization

### Profiling Tools

#### rocprofiler-sdk

**Purpose**: Comprehensive profiling SDK

**Features**:
- Performance counter collection
- Kernel tracing
- Memory bandwidth profiling
- API for custom profiling tools

#### rocprofiler-systems

**Purpose**: System-level profiling (formerly Omnitrace)

**Features**:
- CPU and GPU profiling
- MPI application support
- Python integration
- System-wide performance analysis

#### roctracer

**Purpose**: API tracing for ROCm applications

**Features**:
- HIP API tracing
- HSA API tracing
- Callback mechanisms
- Timeline generation

### Monitoring Tools

#### AMD SMI (amdsmi)

**Purpose**: Modern unified system management interface for AMD GPUs

**Repository**: [https://github.com/ROCm/amdsmi](https://github.com/ROCm/amdsmi)

**Features**:
- Multi-language support (C++, Python, Go, CLI)
- Comprehensive GPU monitoring (temperature, power, utilization, memory)
- Performance management and power capping
- Process tracking and resource management
- Successor to rocm_smi_lib with enhanced capabilities

**Documentation**: See [AMD SMI Usage Guide](amd-smi-usage.md) for detailed examples and API reference

#### ROCm SMI Library (rocmsmilib)

**Purpose**: Legacy system management interface for AMD GPUs

**Note**: Being superseded by AMD SMI for new projects

**Features**:
- GPU monitoring and management
- Temperature, power, utilization tracking
- Clock frequency control
- Memory usage monitoring

#### rocminfo

**Purpose**: Display ROCm system information

**Features**:
- GPU enumeration
- Architecture information
- Capability reporting
- ISA details

### Communication Libraries

#### RCCL (ROCm Communication Collectives Library)

**Purpose**: Multi-GPU and multi-node communication

**Location**: `projects/rccl/`

**Features**:
- Collective operations (AllReduce, Broadcast, etc.)
- Multi-GPU communication
- NCCL-compatible API
- Optimized for AMD GPUs

**Documentation**: See [RCCL documentation](../../layer-3-libraries/communications/rccl-usage.md)

## Getting Started

### Prerequisites

- CMake 3.20+
- ROCm 7.0+ installed or TheRock build environment
- Git with submodule support
- Python 3.8+
- Ninja build system (recommended)

### Clone the Repository

```bash
# Full clone (very large - all projects)
git clone --recursive https://github.com/ROCm/rocm-systems.git
cd rocm-systems

# Update submodules if needed
git submodule update --init --recursive
```

### Sparse Checkout (Recommended)

For working on specific components, use sparse checkout:

```bash
# Clone without checkout
git clone --no-checkout https://github.com/ROCm/rocm-systems.git
cd rocm-systems

# Enable sparse checkout
git sparse-checkout init --cone

# Add only what you need
git sparse-checkout set projects/hip projects/roctracer

# Checkout the files
git checkout develop
```

**Common sparse checkout patterns**:

```bash
# Work on HIP
git sparse-checkout set projects/hip projects/clr

# Work on profiling tools
git sparse-checkout set projects/rocprofiler-sdk projects/roctracer

# Work on RCCL
git sparse-checkout set projects/rccl

# Add more later
git sparse-checkout add projects/rocminfo
```

### Building Individual Components

```bash
# Configure specific component
cmake -B build/hip -S projects/hip \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build/hip

# Install
sudo cmake --install build/hip --prefix /opt/rocm
```

### Building with TheRock

The rocm-systems repo is integrated into TheRock as a submodule:

```bash
# TheRock includes rocm-systems
cd TheRock
ls rocm-systems/  # It's a submodule

# TheRock configuration automatically uses rocm-systems
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_CORE=ON
```

## Development Workflow

### Working on a Single Component

**1. Sparse checkout the component**:

```bash
git clone --no-checkout https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/rocprofiler-sdk
git checkout develop
```

**2. Create a feature branch**:

```bash
git checkout -b feature/my-profiler-improvement
```

**3. Make changes**:

```bash
cd projects/rocprofiler-sdk
# Edit source files
vim source/lib/rocprofiler-sdk/context.cpp
```

**4. Build and test**:

```bash
# Configure
cmake -B ../../build/rocprofiler-sdk -S . -GNinja

# Build
cmake --build ../../build/rocprofiler-sdk

# Test
cd ../../build/rocprofiler-sdk
ctest --output-on-failure
```

**5. Commit and push**:

```bash
git add projects/rocprofiler-sdk
git commit -m "rocprofiler-sdk: Improve context handling"
git push origin feature/my-profiler-improvement
```

### Working Across Multiple Components

When changes span multiple projects (e.g., HIP API change affecting roctracer):

```bash
# Checkout both components
git sparse-checkout set projects/hip projects/roctracer

# Make atomic changes
# ... edit hip files ...
# ... edit roctracer files ...

# Single commit with coordinated changes
git add projects/hip projects/roctracer
git commit -m "hip/roctracer: Update API and tracing support"
```

### Testing Locally

```bash
# Build with testing enabled
cmake -B build/rocprofiler-sdk -S projects/rocprofiler-sdk \
  -GNinja \
  -DBUILD_TESTING=ON

# Run tests
cd build/rocprofiler-sdk
ctest

# Run specific test
ctest -R context_test

# Run with verbose output
ctest -V -R context_test
```

## Best Practices

### For Developers

**1. Use Sparse Checkout**:
```bash
# Only checkout what you need
git sparse-checkout set projects/hip
```

**2. Keep Submodules Updated**:
```bash
# Update to latest
git submodule update --remote --recursive
```

**3. Coordinate Cross-Component Changes**:
```bash
# Make changes to multiple components atomically
git add projects/hip projects/clr
git commit -m "hip/clr: Synchronize runtime updates"
```

**4. Test Integration**:
```bash
# Test your component
ctest --test-dir build/hip

# Test dependent components
ctest --test-dir build/roctracer
```

### For CI/CD

**1. Check Component Status**:
- Visit [rocm-systems repository](https://github.com/ROCm/rocm-systems)
- Check Azure CI and GitHub Actions badges
- Monitor TheRock CI for integration status

**2. Parallel Builds**:
```bash
# Build multiple components in parallel
cmake --build build -j $(nproc)

# Or limit to prevent OOM
cmake --build build -j 8
```

## Integration with Other ROCm Repos

### Relationship with rocm-libraries

```text
TheRock (Build System)
├── rocm-systems (submodule)
│   ├── HIP, ROCr Runtime, Profilers
│   └── RCCL, Monitoring Tools
└── rocm-libraries (submodule)
    ├── Math Libraries (rocBLAS, rocFFT)
    └── ML Libraries (MIOpen)
```

**Dependencies**:
- rocm-libraries depends on HIP from rocm-systems
- RCCL in rocm-systems may use rocm-libraries components
- Both are integrated through TheRock

### Using Both Systems

**Option 1: Use TheRock (Recommended for full stack)**:
```bash
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock
cmake -B build -GNinja . -DTHEROCK_ENABLE_ALL=ON
cmake --build build
```

**Option 2: Use rocm-systems directly (For systems development)**:
```bash
git clone https://github.com/ROCm/rocm-systems.git
cd rocm-systems
# Use sparse checkout for specific component
```

## Profiling Tools Quick Reference

### rocprof (Legacy)

```bash
# Profile HIP application
rocprof ./my_app

# Profile with metrics
rocprof --stats ./my_app

# Generate trace
rocprof --sys-trace ./my_app
```

### rocprofiler-sdk

```bash
# Link your application
target_link_libraries(my_app PRIVATE rocprofiler-sdk::rocprofiler-sdk)

# Use API for custom profiling
#include <rocprofiler-sdk/rocprofiler.h>
```

### roctracer

```bash
# Trace HIP API calls
export ROCTRACER_DOMAIN="hip"
./my_app

# Trace HSA calls
export ROCTRACER_DOMAIN="hsa"
./my_app
```

### rocprofiler-systems (Omnitrace)

```bash
# Install
apt install rocprofiler-systems

# Run with profiling
rocprof-sys -- ./my_app

# View results
rocprof-sys-avail -G output_file.txt
```

## Monitoring Tools Quick Reference

### amd-smi (Recommended)

```bash
# Show GPU information
amd-smi list

# Monitor in real-time
amd-smi monitor

# Monitor with custom interval (1 second)
amd-smi monitor --watch 1000

# Show specific metrics
amd-smi metric --gpu --temperature --power --memory-usage

# Set performance level
sudo amd-smi set --perf-level high

# Set power cap (300W)
sudo amd-smi set --power-cap 300 --gpu 0

# Show processes using GPUs
amd-smi process
```

See [AMD SMI Usage Guide](amd-smi-usage.md) for comprehensive examples

### rocm-smi (Legacy)

```bash
# Show GPU information
rocm-smi

# Monitor in real-time
rocm-smi --showtemp --showuse --showmemuse -t 1000

# Set clock frequency
sudo rocm-smi --setperflevel high
```

### rocminfo

```bash
# Show all ROCm information
rocminfo

# Show agent (GPU) information
rocminfo | grep -A 20 "Agent "

# Show ISA information
rocminfo | grep "Name:" | grep gfx
```

## Troubleshooting

### Build Issues

**Problem**: Submodule initialization fails

```bash
# Solution: Update submodules manually
git submodule update --init --recursive --depth 1

# Or force clean
git submodule foreach --recursive git clean -xfd
git submodule update --init --recursive
```

**Problem**: CMake can't find dependencies

```bash
# Solution: Set ROCm path
cmake -B build -S projects/hip \
  -DCMAKE_PREFIX_PATH=/opt/rocm

# Or use TheRock build
export PATH=/path/to/TheRock/build/bin:$PATH
export LD_LIBRARY_PATH=/path/to/TheRock/build/lib:$LD_LIBRARY_PATH
```

**Problem**: Build fails with OOM

```bash
# Solution: Limit parallel jobs
cmake --build build -j 4

# Or build components sequentially
cmake --build build --target hip
cmake --build build --target roctracer
```

### Runtime Issues

**Problem**: rocm-smi not found

```bash
# Solution: Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# Or install from rocm-systems build
sudo cmake --install build/rocmsmilib --prefix /opt/rocm
```

**Problem**: Profiler can't attach

```bash
# Solution: Check permissions
ls -l /dev/kfd /dev/dri/render*

# Add user to render group
sudo usermod -a -G render $USER
# Log out and back in
```

### Profiling Issues

**Problem**: No profile data generated

```bash
# Solution: Check environment
export ROCPROFILER_METRICS_PATH=/opt/rocm/lib/rocprofiler/metrics.xml

# Enable verbose logging
export ROCTRACER_LOG=1
export ROCPROFILER_LOG=1
```

## Contributing to rocm-systems

### Contribution Guidelines

1. **Read CONTRIBUTING.md**: Component-specific requirements
2. **Follow naming conventions**: Match standardized nomenclature
3. **Sparse checkout**: Don't clone everything unless necessary
4. **Atomic commits**: Related changes across components in one commit
5. **Test thoroughly**: Run component and integration tests
6. **Documentation**: Update docs for API/behavior changes

### Pull Request Process

```bash
# 1. Fork the repository
# 2. Clone your fork with sparse checkout
git clone --no-checkout https://github.com/YOUR_USERNAME/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/hip

# 3. Create feature branch
git checkout -b feature/description

# 4. Make changes and commit
git add projects/hip
git commit -m "hip: Description of changes"

# 5. Push to your fork
git push origin feature/description

# 6. Open PR on GitHub
```

## References

### Official Resources

- **rocm-systems GitHub**: [https://github.com/ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)
- **TheRock Build System**: [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock)
- **ROCm Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)

### Component Documentation

- **HIP**: [https://rocm.docs.amd.com/projects/HIP](https://rocm.docs.amd.com/projects/HIP)
- **rocprofiler**: [https://rocm.docs.amd.com/projects/rocprofiler](https://rocm.docs.amd.com/projects/rocprofiler)
- **roctracer**: [https://rocm.docs.amd.com/projects/roctracer](https://rocm.docs.amd.com/projects/roctracer)
- **RCCL**: [https://rocm.docs.amd.com/projects/rccl](https://rocm.docs.amd.com/projects/rccl)

### Community

- **GitHub Discussions**: [https://github.com/ROCm/rocm-systems/discussions](https://github.com/ROCm/rocm-systems/discussions)
- **GitHub Issues**: [https://github.com/ROCm/rocm-systems/issues](https://github.com/ROCm/rocm-systems/issues)

## Summary

The ROCm Systems super-repo provides:

✓ **Unified Development**: Single repository for all ROCm systems components  
✓ **Better Integration**: Coordinated changes across runtime, profiling, and monitoring  
✓ **Simplified Workflow**: One clone, one build system, one CI  
✓ **Enhanced Collaboration**: Improved visibility and communication  
✓ **Modern Practices**: Monorepo approach for better dependency management  
✓ **TheRock Integration**: Seamless integration with ROCm build system  

**Quick Start**:

```bash
# Sparse checkout for specific component
git clone --no-checkout https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/hip
git checkout develop

# Build
cmake -B build/hip -S projects/hip -GNinja
cmake --build build/hip
```

**For Full ROCm Stack**: Use [TheRock](https://github.com/ROCm/TheRock) which includes rocm-systems as a submodule.

