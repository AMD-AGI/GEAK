---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "2"
category: "hip"
subcategory: "porting"
tags: ["cuda", "hip", "porting", "migration", "hipify"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
therock_included: true
last_updated: 2025-11-03
---

# CUDA to HIP Porting Guide

Comprehensive guide to migrating CUDA code to run on AMD GPUs with HIP and ROCm.

**Official HIP Repository**: [https://github.com/ROCm/HIP](https://github.com/ROCm/HIP) (within rocm-systems)  
**Hipify Tools**: [https://github.com/ROCm/HIPIFY](https://github.com/ROCm/HIPIFY)  
**Documentation**: [https://rocm.docs.amd.com/projects/HIP](https://rocm.docs.amd.com/projects/HIP)

## Overview

HIP (Heterogeneous-compute Interface for Portability) is AMD's C++ runtime API and kernel language that allows developers to create portable GPU code. HIP code can run on both AMD and NVIDIA GPUs, making it an excellent target for CUDA migration.

> **Note**: HIP is now part of the [rocm-systems monorepo](../rocm-systems/rocm-systems-usage.md).

### Why Port to HIP?

- **Portability**: Single codebase runs on AMD and NVIDIA GPUs
- **Performance**: Native performance on AMD hardware
- **Compatibility**: ~95% CUDA API coverage
- **Future-proof**: Support for latest AMD accelerators (MI350, MI300)

### Porting Strategies

| Strategy | Effort | Portability | Performance | Use Case |
|----------|--------|-------------|-------------|----------|
| **Automated hipify** | Low | High | Good | Standard CUDA code |
| **Manual porting** | Medium | High | Excellent | Complex/optimized code |
| **Hybrid approach** | Medium | High | Excellent | Large codebases (recommended) |

## Automatic Porting with hipify Tools

### hipify-perl

Fast, regex-based conversion tool for simple porting:

```bash
# Install hipify-perl
git clone https://github.com/ROCm/HIPIFY.git
cd HIPIFY
export PATH=$PWD:$PATH

# Convert single file
hipify-perl cuda_code.cu > hip_code.cpp

# Convert directory recursively
find . -name "*.cu" -o -name "*.cuh" | xargs -I {} hipify-perl {} > {}.hip

# In-place conversion
hipify-perl -inplace kernel.cu
```

### hipify-clang

AST-based conversion with better accuracy:

```bash
# Install with ROCm
sudo apt install hipify-clang

# Convert single file
hipify-clang cuda_code.cu --cuda-path=/usr/local/cuda

# Convert with include paths
hipify-clang kernel.cu \
  --cuda-path=/usr/local/cuda \
  -I/path/to/includes \
  -- -x cuda

# Process entire project
hipify-clang $(find . -name "*.cu") \
  --cuda-path=/usr/local/cuda \
  -o hip_output/
```

**Recommendation**: Use `hipify-clang` for production porting, `hipify-perl` for quick prototypes.

## API Mapping Reference

### Basic CUDA → HIP Conversions

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaMalloc` | `hipMalloc` | Identical signature |
| `cudaMemcpy` | `hipMemcpy` | Identical signature |
| `cudaFree` | `hipFree` | Identical signature |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | Identical signature |
| `cudaGetDeviceCount` | `hipGetDeviceCount` | Identical signature |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` | Identical signature |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | Identical value |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | Identical value |
| `cudaStream_t` | `hipStream_t` | Type replacement |
| `cudaError_t` | `hipError_t` | Type replacement |
| `cudaSuccess` | `hipSuccess` | Value replacement |

### Kernel Launch

```cpp
// CUDA
kernelName<<<gridDim, blockDim, sharedMem, stream>>>(args);

// HIP (identical syntax!)
kernelName<<<gridDim, blockDim, sharedMem, stream>>>(args);
```

### Thread Indexing

```cpp
// CUDA and HIP (identical)
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
int idz = blockIdx.z * blockDim.z + threadIdx.z;
```

### Device Functions

```cpp
// CUDA
__global__ void kernel() { }
__device__ float helper() { }
__host__ __device__ float both() { }

// HIP (identical)
__global__ void kernel() { }
__device__ float helper() { }
__host__ __device__ float both() { }
```

### Memory Management

```cpp
// CUDA
cudaMalloc(&d_ptr, size);
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
cudaFree(d_ptr);

// HIP (change prefix only)
hipMalloc(&d_ptr, size);
hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice);
hipFree(d_ptr);
```

## Manual Porting Examples

### Example 1: Vector Addition

**CUDA Version:**

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

**HIP Version (Manual Port):**

```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(vectorAdd, blocksPerGrid, threadsPerBlock, 0, 0, 
                       d_A, d_B, d_C, N);
    // Or use the chevron syntax (also works):
    // vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

**Changes Required:**
1. Replace `cuda_runtime.h` → `hip/hip_runtime.h`
2. Replace `cuda` prefix → `hip` prefix
3. Optionally use `hipLaunchKernelGGL` for explicit kernel launch

### Example 2: Matrix Multiplication with Shared Memory

**CUDA Version:**

```cpp
#define TILE_SIZE 16

__global__ void matmul_cuda(float* C, float* A, float* B, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && (t * TILE_SIZE + tx) < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && (t * TILE_SIZE + ty) < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

**HIP Version:**

```cpp
#define TILE_SIZE 16

__global__ void matmul_hip(float* C, float* A, float* B, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && (t * TILE_SIZE + tx) < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && (t * TILE_SIZE + ty) < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

**Note:** The kernel code is **identical**! Only the launch and runtime API calls need prefix changes.

## Build System Changes

### CMake for HIP Projects

**CUDA CMakeLists.txt:**

```cmake
cmake_minimum_required(VERSION 3.10)
project(CudaProject CUDA)

find_package(CUDA REQUIRED)

cuda_add_executable(myapp main.cu kernel.cu)
target_link_libraries(myapp ${CUDA_LIBRARIES})
```

**HIP CMakeLists.txt:**

```cmake
cmake_minimum_required(VERSION 3.16)
project(HipProject)

# Find HIP
find_package(HIP REQUIRED)

# Set HIP compiler
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_STANDARD 14)

# Add executable
add_executable(myapp main.cpp kernel.cpp)

# Link HIP libraries
target_link_libraries(myapp hip::host hip::device)

# Set HIP architecture (optional, for specific GPU)
set_target_properties(myapp PROPERTIES 
    HIP_ARCHITECTURES "gfx90a;gfx942"  # MI250X, MI300X
)
```

### Alternative: Using hip_add_executable

```cmake
cmake_minimum_required(VERSION 3.16)
project(HipProject)

find_package(HIP REQUIRED)

# Simpler approach
hip_add_executable(myapp main.cpp kernel.cpp)
```

### Makefile Example

**CUDA Makefile:**

```makefile
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_80

all: myapp

myapp: main.cu kernel.cu
	$(NVCC) $(CUDA_FLAGS) -o myapp main.cu kernel.cu

clean:
	rm -f myapp
```

**HIP Makefile:**

```makefile
HIPCC = hipcc
HIP_FLAGS = -O3 --offload-arch=gfx90a

all: myapp

myapp: main.cpp kernel.cpp
	$(HIPCC) $(HIP_FLAGS) -o myapp main.cpp kernel.cpp

clean:
	rm -f myapp
```

### Compile Single File

```bash
# CUDA
nvcc -o app kernel.cu -arch=sm_80

# HIP
hipcc -o app kernel.cpp --offload-arch=gfx90a
```

## Library Porting

### cuBLAS → hipBLAS

```cpp
// CUDA with cuBLAS
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    d_A, m,
    d_B, k,
    &beta,
    d_C, m);

cublasDestroy(handle);
```

```cpp
// HIP with hipBLAS
#include <hipblas.h>

hipblasHandle_t handle;
hipblasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;
hipblasSgemm(handle,
    HIPBLAS_OP_N, HIPBLAS_OP_N,
    m, n, k,
    &alpha,
    d_A, m,
    d_B, k,
    &beta,
    d_C, m);

hipblasDestroy(handle);
```

### cuFFT → hipFFT

```cpp
// CUDA with cuFFT
#include <cufft.h>

cufftHandle plan;
cufftPlan1d(&plan, N, CUFFT_C2C, 1);
cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
cufftDestroy(plan);
```

```cpp
// HIP with hipFFT
#include <hipfft.h>

hipfftHandle plan;
hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);
hipfftExecC2C(plan, d_data, d_data, HIPFFT_FORWARD);
hipfftDestroy(plan);
```

### cuRAND → hipRAND/rocRAND

```cpp
// CUDA with cuRAND
#include <curand.h>

curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
curandGenerateUniform(gen, d_data, n);
curandDestroyGenerator(gen);
```

```cpp
// HIP with rocRAND
#include <rocrand.h>

rocrand_generator gen;
rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
rocrand_set_seed(gen, 1234ULL);
rocrand_generate_uniform(gen, d_data, n);
rocrand_destroy_generator(gen);
```

## Important Differences to Handle

### 1. Warp vs Wavefront Size

```cpp
// CUDA: warp size is always 32
#define WARP_SIZE 32

// HIP: wavefront size varies (AMD: 64 for CDNA/RDNA)
// Query at runtime:
hipDeviceProp_t prop;
hipGetDeviceProperties(&prop, 0);
int wavefrontSize = prop.warpSize;  // 64 on AMD, 32 on NVIDIA

// Or use compile-time constant
#ifdef __HIP_PLATFORM_AMD__
    #define WAVEFRONT_SIZE 64
#else
    #define WAVEFRONT_SIZE 32
#endif
```

### 2. Warp-Level Primitives

```cpp
// CUDA warp shuffle
int value = __shfl_down_sync(0xffffffff, var, offset);

// HIP equivalent (works on both AMD and NVIDIA)
int value = __shfl_down(var, offset);  // No mask needed on AMD

// For portable code:
#ifdef __HIP_PLATFORM_AMD__
    int value = __shfl_down(var, offset);
#else
    int value = __shfl_down_sync(0xffffffff, var, offset);
#endif
```

### 3. Atomics

Most atomics are identical, but some advanced ones differ:

```cpp
// Standard atomics (identical)
atomicAdd(&counter, 1);
atomicMax(&value, new_val);
atomicCAS(&target, compare, val);

// Double precision atomics
// CUDA: atomicAdd for double (compute capability 6.0+)
atomicAdd(&d_sum, d_value);

// HIP: Same on both platforms
atomicAdd(&d_sum, d_value);
```

### 4. Memory Fence Instructions

```cpp
// CUDA
__threadfence();        // Device memory fence
__threadfence_block();  // Block memory fence
__threadfence_system(); // System memory fence

// HIP (identical)
__threadfence();
__threadfence_block();
__threadfence_system();
```

### 5. Dynamic Shared Memory

```cpp
// CUDA
extern __shared__ float shared_data[];

// HIP (identical, but can also use HIP_DYNAMIC_SHARED)
extern __shared__ float shared_data[];
// Or:
HIP_DYNAMIC_SHARED(float, shared_data);
```

### 6. Texture Memory

CUDA texture memory is not directly supported in HIP. Use global memory with caching instead:

```cpp
// CUDA texture (not portable)
texture<float, 1> tex;

// HIP alternative: Use __ldg() for cached reads
__device__ float read_cached(const float* ptr, int idx) {
#ifdef __HIP_PLATFORM_AMD__
    return ptr[idx];  // AMD GPUs have automatic caching
#else
    return __ldg(ptr + idx);  // NVIDIA: explicit cache load
#endif
}
```

## Performance Optimization After Porting

### 1. Adjust Thread Block Sizes

```cpp
// CUDA optimized for warps of 32
dim3 blockDim(32, 8);  // 256 threads

// HIP on AMD: optimize for wavefronts of 64
dim3 blockDim(64, 4);  // 256 threads, better for AMD
```

### 2. Occupancy Tuning

```cpp
// Query occupancy
int numBlocks;
int blockSize = 256;

#ifdef __HIP_PLATFORM_AMD__
    // AMD-specific tuning
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, myKernel, blockSize, 0);
#else
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, myKernel, blockSize, 0);
#endif
```

### 3. Memory Coalescing

```cpp
// Good on both platforms: coalesced access
int tid = blockIdx.x * blockDim.x + threadIdx.x;
output[tid] = input[tid];

// Bad on both: strided access
int tid = blockIdx.x * blockDim.x + threadIdx.x;
output[tid] = input[tid * stride];  // Avoid large strides
```

## Testing and Validation

### 1. Correctness Testing

```cpp
// Compare results between CUDA and HIP
void validate_results(float* cuda_result, float* hip_result, int N) {
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = fabs(cuda_result[i] - hip_result[i]);
        max_error = fmax(max_error, error);
    }
    printf("Max error: %f\n", max_error);
}
```

### 2. Error Checking

```cpp
// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// HIP error checking (almost identical)
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            printf("HIP error: %s\n", hipGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Usage
HIP_CHECK(hipMalloc(&d_ptr, size));
HIP_CHECK(hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice));
```

### 3. Profiling

```bash
# CUDA profiling
nvprof ./cuda_app
nsys profile ./cuda_app

# HIP profiling on AMD
rocprof ./hip_app
rocprof --stats ./hip_app
```

## Common Porting Patterns

### Pattern 1: Portable Header

Create a header that works for both CUDA and HIP:

```cpp
// gpu_runtime.h
#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#ifdef USE_HIP
    #include <hip/hip_runtime.h>
    #define GPU_CHECK HIP_CHECK
    #define gpuMalloc hipMalloc
    #define gpuMemcpy hipMemcpy
    #define gpuFree hipFree
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuSuccess hipSuccess
    #define gpuGetErrorString hipGetErrorString
#else
    #include <cuda_runtime.h>
    #define GPU_CHECK CUDA_CHECK
    #define gpuMalloc cudaMalloc
    #define gpuMemcpy cudaMemcpy
    #define gpuFree cudaFree
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuSuccess cudaSuccess
    #define gpuGetErrorString cudaGetErrorString
#endif

#endif // GPU_RUNTIME_H
```

### Pattern 2: Portable Build System

```cmake
option(USE_HIP "Use HIP instead of CUDA" OFF)

if(USE_HIP)
    find_package(HIP REQUIRED)
    set(GPU_LANG HIP)
    set(GPU_TARGETS hip::host hip::device)
else()
    enable_language(CUDA)
    set(GPU_LANG CUDA)
    find_package(CUDA REQUIRED)
    set(GPU_TARGETS ${CUDA_LIBRARIES})
endif()

add_executable(myapp main.cpp)
target_link_libraries(myapp ${GPU_TARGETS})
```

### Pattern 3: Runtime Detection

```cpp
void print_device_info() {
#ifdef __HIP_PLATFORM_AMD__
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("AMD GPU: %s\n", prop.name);
    printf("Compute units: %d\n", prop.multiProcessorCount);
    printf("Wavefront size: %d\n", prop.warpSize);
#else
    cudaDeviceProp_t prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("NVIDIA GPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);
#endif
}
```

## Troubleshooting Common Issues

### Issue 1: Compilation Errors

**Problem:** `hipcc` not found

```bash
# Solution: Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

**Problem:** Architecture not specified

```bash
# Solution: Specify target architecture
hipcc -o app kernel.cpp --offload-arch=gfx90a  # MI250X
hipcc -o app kernel.cpp --offload-arch=gfx942  # MI300X
```

### Issue 2: Runtime Errors

**Problem:** `hipErrorNoBinaryForGpu`

```bash
# Solution: Recompile for correct architecture
# Check your GPU architecture:
rocminfo | grep gfx

# Compile for that architecture:
hipcc --offload-arch=gfx90a kernel.cpp
```

**Problem:** Incorrect results

- Check warp/wavefront size assumptions
- Verify memory access patterns
- Test with smaller data sizes
- Compare against CUDA results

### Issue 3: Performance Issues

**Problem:** Slower than CUDA

- Tune block sizes for wavefront size of 64
- Profile with `rocprof` to find bottlenecks
- Check occupancy with `--stats` flag
- Ensure proper memory coalescing
- Use rocBLAS/MIOpen for standard operations

## Best Practices

### 1. Use hipify-clang for Initial Port

```bash
# Automated conversion is 95% accurate
hipify-clang mycode.cu --cuda-path=/usr/local/cuda
```

### 2. Write Portable Code from Start

```cpp
// Good: Portable
#include <hip/hip_runtime.h>  // Works with hipcc on AMD and NVIDIA

// Better: Completely portable
#ifdef USE_HIP
    #include <hip/hip_runtime.h>
#else
    #include <cuda_runtime.h>
#endif
```

### 3. Test on Both Platforms

```bash
# Build for AMD
mkdir build_amd && cd build_amd
cmake -DUSE_HIP=ON ..
make

# Build for NVIDIA
mkdir build_nvidia && cd build_nvidia
cmake -DUSE_HIP=OFF ..
make
```

### 4. Profile and Optimize Per Platform

```bash
# AMD
rocprof --stats ./app
rocprof --hip-trace ./app

# NVIDIA
nvprof ./app
```

### 5. Use Platform-Specific Libraries

- AMD: Use rocBLAS, MIOpen, rocFFT for best performance
- NVIDIA: Use cuBLAS, cuDNN, cuFFT
- HIP wrappers (hipBLAS, hipFFT) work on both but may have overhead

## Complete Porting Workflow

### Step 1: Preparation

```bash
# Install ROCm and HIP
sudo apt install rocm-dev hipify-clang

# Verify installation
hipcc --version
rocminfo
```

### Step 2: Initial Conversion

```bash
# Automated conversion
hipify-clang mycode.cu --cuda-path=/usr/local/cuda > mycode.cpp

# Or for entire project
find . -name "*.cu" -o -name "*.cuh" | \
    xargs hipify-clang --cuda-path=/usr/local/cuda
```

### Step 3: Manual Fixes

- Update includes: `cuda_runtime.h` → `hip/hip_runtime.h`
- Fix library calls (cuBLAS → hipBLAS, etc.)
- Handle warp size differences
- Adjust texture memory usage

### Step 4: Build System

- Update CMakeLists.txt or Makefile
- Add HIP as build target
- Specify GPU architecture

### Step 5: Testing

- Compile and run
- Validate correctness against CUDA version
- Fix any runtime errors

### Step 6: Optimization

- Profile with rocprof
- Tune block sizes for AMD GPUs
- Optimize memory access patterns
- Consider using AMD-optimized libraries

## Resources

### Documentation

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [HIP API Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIPIFY Tools](https://github.com/ROCm/HIPIFY)

### Sample Code

- [HIP Examples](https://github.com/ROCm/HIP-Examples)
- [ROCm Examples](https://github.com/amd/rocm-examples)

### Related Guides

- [HIP Basics](hip-basics.md) - HIP programming fundamentals
- [HIP Debugging](hip-debugging.md) - Debugging HIP applications
- [ROCm Installation](../rocm/rocm-installation.md) - Setting up ROCm

---

*Note: HIP maintains ~95% CUDA API compatibility. Most CUDA code can be ported with minimal changes. For complex applications, expect 80-90% automation and 10-20% manual optimization.*

