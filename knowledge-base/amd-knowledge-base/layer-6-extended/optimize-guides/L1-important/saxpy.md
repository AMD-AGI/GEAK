---
tags: ["optimization", "performance", "hip", "kernel", "tutorial", "getting-started"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/saxpy.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# SAXPY - Hello, HIP

## Prerequisites

To follow this tutorial, you'll need installed drivers and a HIP compiler toolchain. For more information about installing HIP development packages, see the installation documentation.

## Heterogeneous Programming

Heterogeneous programming deals with devices of varying capabilities simultaneously. Offloading focuses on the asynchronous aspects of computation. HIP encompasses both, exposing GPGPU programming similarly to ordinary host-side CPU programming while enabling data movement across various devices.

Target devices are built for specific purposes with different performance characteristics than traditional CPUs. Even subtle code changes might adversely affect execution time.

## Your First Lines of HIP Code

SAXPY (Single-precision A times X Plus Y) is a fundamental GPGPU operation: a vector equation *a·x + y = z* where *a* is a scalar and *x, y, z* are vectors. The sequential version uses a simple for loop:

```cpp
for (int i = 0; i < N; ++i)
    z[i] = a * x[i] + y[i];
```

### Memory Management

Device memory allocation and host-to-device data transfer precedes kernel execution:

```cpp
float* d_x{};
float* d_y{};
HIP_CHECK(hipMalloc(&d_x, size_bytes));
HIP_CHECK(hipMalloc(&d_y, size_bytes));
HIP_CHECK(hipMemcpy(d_x, x.data(), size_bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_y, y.data(), size_bytes, hipMemcpyHostToDevice));
```

### Kernel Launch

Device-side functions use the `__global__` qualifier for host-callable entry points:

```cpp
__global__ void saxpy_kernel(const float a, const float* d_x,
                             float* d_y, const unsigned int size)
{
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx < size)
    {
        d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
    }
}
```

Launch syntax uses triple chevron notation specifying grid dimensions, block size, shared memory, and stream:

```cpp
saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0,
              hipStreamDefault>>>(a, d_x, d_y, size);
```

### Key Kernel Characteristics

- `__global__` instructs the compiler to generate device-executable code
- Functions don't return values; results communicate through output parameters
- Arguments use pass-by-value with TriviallyCopyable types only
- Pointer arguments reference device memory (typically VRAM)

### Result Retrieval

Device-to-host memory transfer returns computed results:

```cpp
HIP_CHECK(hipMemcpy(y.data(), d_y, size_bytes, hipMemcpyDeviceToHost));
```

## Compiling on the Command Line

### Setting Up the Command Line

**Linux with AMD:** Add ROCm to PATH:
```bash
export PATH=/opt/rocm/bin:${PATH}
```

**Linux with NVIDIA:** CUDA tools are typically available by default.

**Windows with AMD/NVIDIA:** Requires Visual Studio 2022 Build Tools with Windows SDK and MSVC toolchain. For AMD, establish developer shell context via PowerShell COM detection.

### Invoking the Compiler

**Linux AMD:**
```bash
amdclang++ ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common -lamdhip64 \
  -L /opt/rocm/lib -O2
```

**Linux NVIDIA:**
```bash
nvcc ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common \
  -I /opt/rocm/include -O2 -x cu
```

**Windows AMD:**
```bash
clang++ .\HIP-Basic\saxpy\main.hip -o saxpy.exe -I .\Common -lamdhip64 \
  -L ${env:HIP_PATH}lib -O2
```

**Windows NVIDIA:**
```bash
nvcc .\HIP-Basic\saxpy\main.hip -o saxpy.exe -I ${env:HIP_PATH}include \
  -I .\Common -O2 -x cu
```

### Device Binary Inspection

**AMD Inspection Tools:**
- Use `llvm-objdump --offloading` to list embedded binaries
- Extract disassembled code with `llvm-objdump --disassemble`
- Compile with `--save-temps` flag to preserve intermediate files

**NVIDIA Inspection Tools:**
- Use `cuobjdump --list-ptx` to display PTX ISA versions embedded
- PTX files show compute capability (e.g., `sm_52` for capability 5.2)

### Device Capability Detection

**AMD:**
```bash
/opt/rocm/bin/rocminfo | grep gfx
amdclang++ ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common \
  -lamdhip64 -L /opt/rocm/lib -O2 --offload-arch=gfx906:sramecc+:xnack-
```

**NVIDIA:**
```bash
nvcc ./HIP-Basic/device_query/main.cpp -o device_query -I ./Common \
  -I /opt/rocm/include -O2
./device_query | grep "major.minor"
nvcc ./HIP-Basic/saxpy/main.hip -o saxpy -I ./Common \
  -I /opt/rocm/include -O2 -x cu -arch=sm_70,sm_86
```
