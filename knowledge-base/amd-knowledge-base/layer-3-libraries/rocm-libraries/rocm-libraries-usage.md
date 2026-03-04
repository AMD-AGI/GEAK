---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "3"
category: "rocm-libraries"
subcategory: "monorepo"
tags: ["rocm-libraries", "monorepo", "build-system", "integration", "development", "libraries"]
rocm_version: "7.1+"
therock_included: true
last_updated: 2025-11-03
---

# ROCm Libraries Usage Guide

The ROCm Libraries monorepo is AMD's unified repository that consolidates all ROCm math and ML libraries into a single codebase. This represents a major architectural shift from individual standalone repositories to a centralized monorepo approach for better integration and development workflows.

**Repository**: [https://github.com/ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)

**Status**: Active migration (Started 2024, ongoing)

## What is rocm-libraries?

The rocm-libraries monorepo is a unified repository that brings together all ROCm libraries that were previously maintained in separate GitHub repositories. This includes:

- **Math Libraries**: rocBLAS, rocFFT, rocRAND, rocSOLVER, rocSPARSE, rocPRIM, rocThrust, rocWMMA
- **HIP Wrappers**: hipBLAS, hipBLASLt, hipFFT, hipRAND, hipSOLVER, hipSPARSE, hipSPARSELt, hipCUB
- **ML Libraries**: MIOpen, hipDNN
- **Shared Components**: Tensile, Composable Kernel, rocRoller, MXDataGenerator

## Why a Monorepo?

### Goals

1. **Unified Build and Test Workflows**: Single CI/CD pipeline across all libraries
2. **Shared Tooling**: Common build scripts, testing infrastructure, and development tools
3. **Improved Integration**: Better coordination between interdependent libraries
4. **Enhanced Collaboration**: Easier cross-library development and visibility
5. **Simplified Contributor Experience**: One repository to clone, one workflow to learn
6. **Atomic Changes**: Multi-library changes in a single PR

### Benefits

- **Consistency**: Standardized naming, structure, and processes
- **Efficiency**: Shared CI resources and build infrastructure
- **Visibility**: Easier to see dependencies and interactions
- **Testing**: Integrated testing across library boundaries
- **Versioning**: Coordinated releases across all components

## Repository Structure

The monorepo is organized into two main categories:

```text
rocm-libraries/
├── projects/              # Libraries released as distinct packages
│   ├── composablekernel/  # Composable Kernel for ML primitives
│   ├── hipblas/           # HIP wrapper for BLAS
│   ├── hipblas-common/    # Common utilities for hipBLAS
│   ├── hipblaslt/         # HIP wrapper for BLAS-like operations
│   ├── hipcub/            # HIP wrapper for CUB
│   ├── hipdnn/            # HIP wrapper for DNN
│   ├── hipfft/            # HIP wrapper for FFT
│   ├── hiprand/           # HIP wrapper for random number generation
│   ├── hipsolver/         # HIP wrapper for linear algebra solvers
│   ├── hipsparse/         # HIP wrapper for sparse linear algebra
│   ├── hipsparselt/       # HIP wrapper for sparse BLAS-like
│   ├── miopen/            # Deep learning primitives library
│   ├── rocblas/           # Basic Linear Algebra Subprograms
│   ├── rocfft/            # Fast Fourier Transform library
│   ├── rocprim/           # Parallel primitives library
│   ├── rocrand/           # Random number generation library
│   ├── rocsolver/         # Lapack-like linear algebra solvers
│   ├── rocsparse/         # Sparse linear algebra library
│   ├── rocthrust/         # Thrust parallel algorithms
│   └── rocwmma/           # Wave Matrix Multiply-Accumulate API
│
└── shared/                # Shared dependencies, not standalone packages
    ├── tensile/           # Kernel generator for rocBLAS
    ├── rocroller/         # Testing infrastructure
    └── mxdatagenerator/   # Mixed precision data generator
```

### Nomenclature Standards

Project names have been standardized to match released package casing and punctuation:

- **Consistent casing**: `hipBLAS`, `rocBLAS`, `MIOpen`
- **Removed underscores**: `composable_kernel` → `composablekernel`
- **Removed camelCase inconsistencies**: Standardized across all projects

## Migration Status

### Completed Migrations (as of ROCm 7.1.0)

The following libraries have been **fully migrated** to the monorepo:

**Math Libraries**:
- ✅ rocBLAS
- ✅ rocFFT
- ✅ rocPRIM
- ✅ rocRAND
- ✅ rocSOLVER
- ✅ rocSPARSE
- ✅ rocThrust
- ✅ rocWMMA

**HIP Wrappers**:
- ✅ hipBLAS
- ✅ hipBLAS-Common
- ✅ hipBLASLt
- ✅ hipCUB
- ✅ hipFFT
- ✅ hipRAND
- ✅ hipSOLVER
- ✅ hipSPARSE
- ✅ hipSPARSELt

**ML Libraries**:
- ✅ MIOpen
- ✅ hipDNN

**Shared Components**:
- ✅ Tensile
- ✅ rocRoller
- ✅ MXDataGenerator

### Tentative Migrations

| Component | Status |
|-----------|--------|
| Composable Kernel | In progress / TBD |

> Note: Migration schedules are subject to change. Check the [official repository](https://github.com/ROCm/rocm-libraries) for latest status.

## CI/CD Integration

The monorepo maintains comprehensive CI/CD pipelines:

### CI Systems

1. **Azure CI**: Primary CI for most components
2. **Math-CI**: Internal AMD CI for math libraries
3. **MICI**: Internal AMD CI for ML libraries (MIOpen)
4. **TheRock CI**: Multi-component integration testing

### CI Status per Component

Each library has its own CI pipelines that run on commits:

| Library | Azure CI | AMD Internal CI |
|---------|----------|-----------------|
| rocBLAS | ✓ | Math-CI |
| rocFFT | ✓ | Math-CI |
| rocSOLVER | ✓ | Math-CI |
| MIOpen | ✓ | MICI |
| hipBLAS | ✓ | Math-CI |
| Tensile | ✓ | Math-CI |

All CI status badges are available on the [repository README](https://github.com/ROCm/rocm-libraries).

## Getting Started

### Prerequisites

- CMake 3.20+
- ROCm 6.0+ or TheRock build
- Git with submodule support
- Python 3.8+
- Ninja build system (recommended)

### Clone the Repository

```bash
# Full clone (very large - all libraries)
git clone --recursive https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries

# Update submodules if needed
git submodule update --init --recursive
```

### Sparse Checkout (Recommended)

For working on specific libraries, use sparse checkout to avoid cloning everything:

```bash
# Clone without checkout
git clone --no-checkout https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries

# Enable sparse checkout
git sparse-checkout init --cone

# Add only what you need
git sparse-checkout set projects/rocblas shared/tensile

# Checkout the files
git checkout develop
```

**Common sparse checkout patterns**:

```bash
# Work on rocBLAS and its dependencies
git sparse-checkout set projects/rocblas shared/tensile cmake

# Work on hipBLAS (needs rocBLAS)
git sparse-checkout set projects/hipblas projects/rocblas \
  projects/hipblas-common shared/tensile cmake

# Work on MIOpen
git sparse-checkout set projects/miopen projects/composablekernel cmake

# Add more later
git sparse-checkout add projects/rocfft
```

### Building Individual Libraries

```bash
# Configure specific library
cmake -B build/rocblas -S projects/rocblas \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx90a;gfx942"

# Build
cmake --build build/rocblas

# Install
cmake --install build/rocblas --prefix /opt/rocm
```

### Building Multiple Libraries

```bash
# Configure from root with specific projects
cmake -B build -GNinja . \
  -DBUILD_PROJECTS="rocblas;rocfft;rocsolver" \
  -DAMDGPU_TARGETS="gfx942"

# Build all configured projects
cmake --build build

# Build specific target
cmake --build build --target rocblas
```

### Building with TheRock

The rocm-libraries monorepo is integrated into TheRock as a submodule:

```bash
# TheRock includes rocm-libraries
cd TheRock
ls rocm-libraries/  # It's a submodule

# TheRock configuration automatically uses rocm-libraries
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_MATH_LIBS=ON
```

## Development Workflow

### Working on a Single Library

**1. Sparse checkout the library**:

```bash
git clone --no-checkout https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas shared/tensile cmake
git checkout develop
```

**2. Create a feature branch**:

```bash
git checkout -b feature/my-rocblas-improvement
```

**3. Make changes**:

```bash
cd projects/rocblas
# Edit source files
vim library/src/blas1/rocblas_axpy.cpp
```

**4. Build and test**:

```bash
# Configure
cmake -B ../../build/rocblas -S . -GNinja

# Build
cmake --build ../../build/rocblas

# Test
cd ../../build/rocblas
ctest --output-on-failure
```

**5. Commit and push**:

```bash
git add projects/rocblas
git commit -m "rocblas: Improve axpy performance on gfx942"
git push origin feature/my-rocblas-improvement
```

### Working Across Multiple Libraries

When changes span multiple libraries (e.g., rocBLAS API change affecting hipBLAS):

```bash
# Checkout both libraries
git sparse-checkout set projects/rocblas projects/hipblas \
  projects/hipblas-common shared/tensile cmake

# Make atomic changes
# ... edit rocblas files ...
# ... edit hipblas files ...

# Single commit with coordinated changes
git add projects/rocblas projects/hipblas
git commit -m "Update API: Coordinate rocBLAS and hipBLAS changes"
```

### Testing Locally

```bash
# Build with testing enabled
cmake -B build/rocblas -S projects/rocblas \
  -GNinja \
  -DBUILD_TESTING=ON

# Run tests
cd build/rocblas
ctest

# Run specific test
ctest -R axpy_test

# Run with verbose output
ctest -V -R axpy_test
```

## Contributing to rocm-libraries

### Contribution Guidelines

1. **Read CONTRIBUTING.md**: Each library may have specific requirements
2. **Follow naming conventions**: Match the standardized nomenclature
3. **Sparse checkout**: Don't clone everything unless necessary
4. **Atomic commits**: Related changes across libraries in one commit
5. **Test thoroughly**: Run CI locally before pushing
6. **Documentation**: Update docs for API changes

### Pull Request Process

```bash
# 1. Fork the repository
# 2. Clone your fork with sparse checkout
git clone --no-checkout https://github.com/YOUR_USERNAME/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas cmake

# 3. Create feature branch
git checkout -b feature/description

# 4. Make changes and commit
git add projects/rocblas
git commit -m "rocblas: Description of changes"

# 5. Push to your fork
git push origin feature/description

# 6. Open PR on GitHub
```

### PR Guidelines

- **Title format**: `[library]: Brief description`
- **Description**: Explain what, why, and how
- **Testing**: Include test results
- **Breaking changes**: Clearly document
- **Cross-library**: Tag all affected libraries
- **CI**: Ensure all CI passes

## Relationship with TheRock

### Integration Architecture

```text
TheRock (Build System)
├── rocm-libraries (submodule)
│   ├── projects/
│   │   ├── rocblas/
│   │   ├── rocfft/
│   │   └── ...
│   └── shared/
│       ├── tensile/
│       └── ...
└── (other ROCm components)
```

### How They Work Together

1. **TheRock** uses rocm-libraries as a git submodule
2. **TheRock CMake** configuration selectively builds libraries from rocm-libraries
3. **Feature flags** in TheRock control which libraries are built
4. **TheRock CI** tests integration across the full ROCm stack

### Using Both Systems

**Option 1: Use TheRock (Recommended for full stack)**

```bash
# Build entire ROCm stack including math libraries
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock
cmake -B build -GNinja . -DTHEROCK_ENABLE_MATH_LIBS=ON
cmake --build build
```

**Option 2: Use rocm-libraries directly (For library development)**

```bash
# Build specific libraries only
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
# Use sparse checkout for specific library
```

**Option 3: Hybrid approach**

```bash
# Clone TheRock
git clone --recursive https://github.com/ROCm/TheRock.git
cd TheRock

# Use external rocm-libraries for active development
git clone https://github.com/ROCm/rocm-libraries.git ~/dev/rocm-libraries

# Configure TheRock to use external rocm-libraries
cmake -B build -GNinja . \
  -DTHEROCK_USE_EXTERNAL_ROCM_LIBRARIES=ON \
  -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=~/dev/rocm-libraries
```

## Best Practices

### For Library Developers

**1. Use Sparse Checkout**

```bash
# Only checkout what you need
git sparse-checkout set projects/rocblas shared/tensile cmake
```

**2. Keep Submodules Updated**

```bash
# Update to latest
git submodule update --remote --recursive

# Or for specific submodule
git submodule update --remote projects/rocblas
```

**3. Coordinate Cross-Library Changes**

```bash
# Make changes to multiple libraries atomically
git add projects/rocblas projects/hipblas
git commit -m "Coordinate API update across rocBLAS and hipBLAS"
```

**4. Test Integration**

```bash
# Test your library
ctest --test-dir build/rocblas

# Test dependent libraries
ctest --test-dir build/hipblas
```

### For CI/CD Integration

**1. Use CMake Presets**

```bash
# List available presets
cmake --list-presets

# Use preset
cmake --preset=release-gfx942
cmake --build --preset=release-gfx942
```

**2. Cache Build Artifacts**

```bash
# Use ccache
export CCACHE_DIR=/cache/ccache
cmake -B build -GNinja . \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

**3. Parallel Builds**

```bash
# Build multiple libraries in parallel
cmake --build build -j $(nproc)

# Or limit to prevent OOM
cmake --build build -j 8
```

### For Users

**1. Use Stable Releases**

```bash
# Clone specific release tag
git clone --branch rocm-7.1.0 \
  https://github.com/ROCm/rocm-libraries.git
```

**2. Use Pre-built Packages**

```bash
# Instead of building from source, use packages
sudo apt install rocblas rocfft rocsolver  # Ubuntu/Debian
sudo dnf install rocblas rocfft rocsolver  # Fedora/RHEL
```

**3. Report Issues to Correct Repository**

- **Library-specific issues**: Report to rocm-libraries with library tag
- **Build system issues**: Report to TheRock
- **Integration issues**: Report to rocm-libraries with integration tag

## Troubleshooting

### Clone Issues

**Problem**: Clone is too large/slow

```bash
# Solution: Use sparse checkout
git clone --no-checkout https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas cmake
git checkout develop
```

**Problem**: Submodule initialization fails

```bash
# Solution: Update submodules manually
git submodule update --init --recursive --depth 1

# Or force clean
git submodule foreach --recursive git clean -xfd
git submodule update --init --recursive
```

### Build Issues

**Problem**: CMake can't find dependencies

```bash
# Solution: Set ROCm path
cmake -B build -S projects/rocblas \
  -DCMAKE_PREFIX_PATH=/opt/rocm

# Or use TheRock build
export PATH=/path/to/TheRock/build/bin:$PATH
export LD_LIBRARY_PATH=/path/to/TheRock/build/lib:$LD_LIBRARY_PATH
```

**Problem**: Out of memory during build

```bash
# Solution: Limit parallel jobs
cmake --build build -j 4

# Or build libraries sequentially
cmake --build build --target rocblas
cmake --build build --target rocfft
```

### Development Issues

**Problem**: Changes not reflected after rebuild

```bash
# Solution: Clean and rebuild
cmake --build build --target clean
cmake --build build
```

**Problem**: Git operations slow with full checkout

```bash
# Solution: Switch to sparse checkout
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas cmake
```

### Testing Issues

**Problem**: Tests fail with GPU errors

```bash
# Solution: Check GPU visibility
rocm-smi
rocminfo

# Set correct GPU architecture
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
```

**Problem**: Tests can't find libraries

```bash
# Solution: Set library path
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH

# Or install locally
cmake --install build --prefix /tmp/test-rocm
export LD_LIBRARY_PATH=/tmp/test-rocm/lib:$LD_LIBRARY_PATH
```

## Migration from Standalone Repositories

### For Existing Contributors

If you previously contributed to standalone repositories:

**1. Update your workflow**:

```bash
# Old way (standalone)
git clone https://github.com/ROCm/rocBLAS.git

# New way (monorepo with sparse checkout)
git clone --no-checkout https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout set projects/rocblas shared/tensile cmake
git checkout develop
```

**2. Update your bookmarks**:

- Old: `https://github.com/ROCm/rocBLAS`
- New: `https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas`

**3. Update CI references**:

```yaml
# Old .github/workflows
- uses: actions/checkout@v3
  with:
    repository: ROCm/rocBLAS

# New .github/workflows
- uses: actions/checkout@v3
  with:
    repository: ROCm/rocm-libraries
    sparse-checkout: |
      projects/rocblas
      shared/tensile
      cmake
```

### Standalone Repository Status

**Archived**: Original standalone repositories are archived but kept for historical reference

**Issues**: New issues should be opened in rocm-libraries repository with appropriate library tags

**Branches**: Stable branches remain in standalone repos for older ROCm versions

## Reference Tables

### Library Categories

| Category | Libraries | Purpose |
|----------|-----------|---------|
| BLAS | rocBLAS, hipBLAS, hipBLASLt | Basic Linear Algebra |
| FFT | rocFFT, hipFFT | Fast Fourier Transforms |
| Random | rocRAND, hipRAND | Random Number Generation |
| Solver | rocSOLVER, hipSOLVER | Linear Algebra Solvers |
| Sparse | rocSPARSE, hipSPARSE, hipSPARSELt | Sparse Linear Algebra |
| Primitives | rocPRIM, rocThrust, hipCUB | Parallel Primitives |
| ML | MIOpen, hipDNN | Deep Learning Primitives |
| WMMA | rocWMMA | Wave Matrix Operations |

### Dependencies Between Libraries

```text
hipBLAS → rocBLAS → Tensile
hipFFT → rocFFT
hipRAND → rocRAND
hipSOLVER → rocSOLVER → rocBLAS
hipSPARSE → rocSPARSE
MIOpen → Composable Kernel, rocBLAS
```

## Resources and Links

### Official Documentation

- **Monorepo**: [https://github.com/ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)
- **Contributing Guide**: [https://github.com/ROCm/rocm-libraries/blob/develop/CONTRIBUTING.md](https://github.com/ROCm/rocm-libraries/blob/develop/CONTRIBUTING.md)
- **TheRock Integration**: [https://github.com/ROCm/TheRock](https://github.com/ROCm/TheRock)

### Library-Specific Documentation

Each library maintains its own documentation in `projects/<library>/docs/`:

- rocBLAS: API reference, programming guide
- MIOpen: User guide, kernel documentation
- Tensile: Kernel generation guide

### Community

- **Discussions**: [https://github.com/ROCm/rocm-libraries/discussions](https://github.com/ROCm/rocm-libraries/discussions)
- **Issues**: [https://github.com/ROCm/rocm-libraries/issues](https://github.com/ROCm/rocm-libraries/issues)
- **ROCm Documentation**: [https://rocm.docs.amd.com](https://rocm.docs.amd.com)

### CI Status

- **CI Dashboard**: Check repository README for live CI status badges
- **Azure Pipelines**: Individual pipeline links per library
- **TheRock CI**: Multi-component integration testing

## Summary

The ROCm Libraries monorepo represents AMD's commitment to:

✓ **Unified Development**: Single repository for all ROCm libraries  
✓ **Better Integration**: Coordinated changes across library boundaries  
✓ **Simplified Workflow**: One clone, one build system, one CI  
✓ **Enhanced Collaboration**: Improved visibility and communication  
✓ **Modern Practices**: Monorepo approach used by major tech companies  
✓ **Backward Compatibility**: Existing packages and APIs maintained  

**Quick Start**:

```bash
# Sparse checkout for specific library
git clone --no-checkout https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/rocblas shared/tensile cmake
git checkout develop

# Build
cmake -B build/rocblas -S projects/rocblas -GNinja
cmake --build build/rocblas
```

**For Full ROCm Stack**: Use [TheRock](https://github.com/ROCm/TheRock) which includes rocm-libraries as a submodule.

