---
title: "HIP vs CUDA Programming Comparison"
category: "comparison"
tags: ["hip", "cuda", "comparison", "porting", "fundamentals"]
rocm_version: "7.0+"
cuda_version: "12.0+"
last_updated: 2025-11-20
source: "GPU Programming 101 - Modules 1-3"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# HIP vs CUDA Programming Comparison

*Side-by-side comparison of GPU programming fundamentals on AMD and Nvidia platforms*

## Overview

This guide compares HIP (AMD) and CUDA (Nvidia) programming for Modules 1-3 of GPU Programming 101: fundamentals, memory management, and thread synchronization.

**Related Documentation:**
- AMD: [HIP Fundamentals](../amd-knowledge-base/layer-2-compute-stack/hip/hip-gpu-programming-fundamentals.md)
- Nvidia: [CUDA Fundamentals](../nvidia-knowledge-base/layer-2-compute-stack/cuda/cuda-gpu-programming-fundamentals.md)

## Quick Reference Table

| Feature | AMD HIP | Nvidia CUDA | Similarity |
|---------|---------|-------------|------------|
| **Warp/Wavefront** | 64 threads | 32 threads | 95% |
| **Shared Memory** | LDS (Local Data Share) | Shared Memory | 99% |
| **Block Size** | Up to 1024 | Up to 1024 | 100% |
| **API Similarity** | `hip*` | `cuda*` | 95% |
| **Compiler** | `hipcc` | `nvcc` | Similar |
| **Syntax** | C++/HIP | C++/CUDA | 98% |

## Hello World Comparison

### HIP (AMD)

```cpp
#include <hip/hip_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));
    
    // Copy data
    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);
    
    // Launch kernel
    hipLaunchKernelGGL(vectorAdd, dim3(grid), dim3(block), 0, 0, 
                       d_A, d_B, d_C, N);
    
    // Copy result
    hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);
    
    // Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
}

// Compile
// hipcc -O3 -march=gfx90a vector_add.cpp -o vector_add
```

### CUDA (Nvidia)

```cuda
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Copy data
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
    
    // Copy result
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Compile
// nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add
```

### Key Differences

1. **Header:** `hip/hip_runtime.h` vs `cuda_runtime.h`
2. **API prefix:** `hip*` vs `cuda*`
3. **Launch syntax:** `hipLaunchKernelGGL(...)` vs `kernel<<<...>>>`
4. **Compilation:** `hipcc` vs `nvcc`
5. **Architecture flag:** `-march=gfx90a` vs `-arch=sm_80`

**Similarity:** 98% - mostly naming conventions

## API Translation

### Memory Management

| Operation | HIP | CUDA |
|-----------|-----|------|
| Allocate | `hipMalloc(&ptr, size)` | `cudaMalloc(&ptr, size)` |
| Free | `hipFree(ptr)` | `cudaFree(ptr)` |
| Copy H→D | `hipMemcpy(..., hipMemcpyHostToDevice)` | `cudaMemcpy(..., cudaMemcpyHostToDevice)` |
| Copy D→H | `hipMemcpy(..., hipMemcpyDeviceToHost)` | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` |
| Set | `hipMemset(ptr, val, size)` | `cudaMemset(ptr, val, size)` |
| Unified | `hipMallocManaged(&ptr, size)` | `cudaMallocManaged(&ptr, size)` |

### Synchronization

| Operation | HIP | CUDA |
|-----------|-----|------|
| Device sync | `hipDeviceSynchronize()` | `cudaDeviceSynchronize()` |
| Block sync | `__syncthreads()` | `__syncthreads()` |
| Thread fence | `__threadfence()` | `__threadfence()` |
| Block fence | `__threadfence_block()` | `__threadfence_block()` |

### Built-in Variables

| Variable | HIP | CUDA |
|----------|-----|------|
| Thread ID | `threadIdx.x` | `threadIdx.x` |
| Block ID | `blockIdx.x` | `blockIdx.x` |
| Block size | `blockDim.x` | `blockDim.x` |
| Grid size | `gridDim.x` | `gridDim.x` |

**Identical!**

## Warp/Wavefront Comparison

### Size Difference

| AMD HIP | Nvidia CUDA |
|---------|-------------|
| 64-thread wavefront | 32-thread warp |
| CDNA architecture | All Nvidia GPUs |

### Shuffle Operations

#### HIP (64 threads)

```cpp
__device__ float wavefrontSum(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 64);  // 64-wide
    }
    return val;
}
```

#### CUDA (32 threads)

```cuda
__device__ float warpSum(float val) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);  // 32-wide
    }
    return val;
}
```

### Key Differences

1. **Size:** 64 vs 32
2. **Sync mask:** HIP doesn't require, CUDA needs `_sync` versions
3. **Loop iterations:** Different offsets (32→1 vs 16→1)

## Shared Memory / LDS

### Terminology

| AMD | Nvidia |
|-----|--------|
| LDS (Local Data Share) | Shared Memory |
| 64 KB per CU | 48-227 KB per SM |
| `__shared__` keyword | `__shared__` keyword |

### Example Comparison

#### HIP

```cpp
__global__ void matmulHIP() {
    __shared__ float tileA[32][32];  // LDS
    __shared__ float tileB[32][32];
    
    // Load to LDS
    tileA[threadIdx.y][threadIdx.x] = ...;
    __syncthreads();
    
    // Use LDS
    float sum = 0.0f;
    for (int k = 0; k < 32; ++k) {
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
}
```

#### CUDA

```cuda
__global__ void matmulCUDA() {
    __shared__ float tileA[32][32];  // Shared memory
    __shared__ float tileB[32][32];
    
    // Load to shared memory
    tileA[threadIdx.y][threadIdx.x] = ...;
    __syncthreads();
    
    // Use shared memory
    float sum = 0.0f;
    for (int k = 0; k < 32; ++k) {
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
}
```

**Identical syntax!**

### Bank Conflicts

| AMD HIP | Nvidia CUDA |
|---------|-------------|
| 32 banks | 32 banks |
| 64-byte banks (CDNA3) | 32-bit banks |
| Same avoidance strategies | Same avoidance strategies |

## Atomic Operations

### Comparison

| Operation | HIP | CUDA | Notes |
|-----------|-----|------|-------|
| Add | `atomicAdd(ptr, val)` | `atomicAdd(ptr, val)` | Identical |
| Sub | `atomicSub(ptr, val)` | `atomicSub(ptr, val)` | Identical |
| Max | `atomicMax(ptr, val)` | `atomicMax(ptr, val)` | Identical |
| Min | `atomicMin(ptr, val)` | `atomicMin(ptr, val)` | Identical |
| CAS | `atomicCAS(ptr, cmp, val)` | `atomicCAS(ptr, cmp, val)` | Identical |
| Float Add | `atomicAdd(float*, float)` | `atomicAdd(float*, float)` | Identical |

**100% API compatible!**

## Voting Primitives

### HIP (AMD)

```cpp
__global__ void votingHIP() {
    int predicate = (threadIdx.x > 32);
    
    // 64-wide wavefront operations
    unsigned long long mask = __ballot(predicate);  // 64-bit mask
    int count = __popcll(mask);  // Population count (long long)
    int any = __any(predicate);
    int all = __all(predicate);
}
```

### CUDA (Nvidia)

```cuda
__global__ void votingCUDA() {
    int predicate = (threadIdx.x > 16);
    
    // 32-wide warp operations
    unsigned mask = 0xffffffffu;
    unsigned ballot = __ballot_sync(mask, predicate);  // 32-bit mask
    int count = __popc(ballot);  // Population count
    int any = __any_sync(mask, predicate);
    int all = __all_sync(mask, predicate);
}
```

### Differences

1. **Mask type:** `unsigned long long` (64-bit) vs `unsigned` (32-bit)
2. **Sync versions:** HIP doesn't need `_sync`, CUDA does
3. **Pop count:** `__popcll` vs `__popc`

## Performance Comparison

### Theoretical Peak (FP32)

| GPU | Vendor | FP32 TFLOPS | Memory BW | Warp/Wave Size |
|-----|--------|-------------|-----------|----------------|
| MI300X | AMD | 653 | 5.3 TB/s | 64 |
| H100 | Nvidia | 67 | 3.35 TB/s | 32 |
| MI250X | AMD | 47.9 | 3.2 TB/s | 64 |
| A100 | Nvidia | 19.5 | 1.6 TB/s | 32 |

### Memory Bandwidth Efficiency

Both achieve ~80-90% of peak with coalesced access.

## Porting Guide

### Automatic Porting (CUDA → HIP)

AMD provides `hipify` tool for automatic conversion:

```bash
# Automatic conversion
hipify-perl input.cu > output.hip
# or
hipify-clang input.cu -- -I/path/to/cuda/include

# Manual replacements
sed 's/cuda/hip/g' input.cu > output.hip
sed 's/__syncthreads/__syncthreads/g'
```

### Common Manual Changes

1. **Kernel launch syntax:**
   ```cpp
   // CUDA
   kernel<<<grid, block>>>(args);
   
   // HIP
   hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), 0, 0, args);
   ```

2. **Warp/Wavefront size:**
   ```cpp
   // CUDA: 32
   #define WARP_SIZE 32
   
   // HIP: 64
   #define WAVESIZE 64
   ```

3. **Shuffle loop iterations:** Adjust for 64 vs 32

## Compilation Comparison

### HIP

```bash
# Basic
hipcc -O3 kernel.hip -o kernel

# Target MI250X
hipcc -O3 -march=gfx90a kernel.hip -o kernel

# Target MI300X
hipcc -O3 -march=gfx942 kernel.hip -o kernel

# With debugging
hipcc -g -O0 kernel.hip -o kernel
```

### CUDA

```bash
# Basic
nvcc -O3 kernel.cu -o kernel

# Target A100
nvcc -O3 -arch=sm_80 kernel.cu -o kernel

# Target H100
nvcc -O3 -arch=sm_90 kernel.cu -o kernel

# With debugging
nvcc -g -G kernel.cu -o kernel
```

## Best Practices Comparison

| Practice | HIP (AMD) | CUDA (Nvidia) |
|----------|-----------|---------------|
| Block size | Multiples of 64 | Multiples of 32 |
| Occupancy target | 50-75% | 50-75% |
| Shared memory | Use LDS for reuse | Use shared mem for reuse |
| Coalescing | Critical | Critical |
| Atomics | Minimize contention | Minimize contention |
| Register pressure | Watch spilling | Watch spilling |

## References

### Related Documentation
- **AMD HIP:**
  - [HIP Fundamentals](../amd-knowledge-base/layer-2-compute-stack/hip/hip-gpu-programming-fundamentals.md)
  - [HIP Memory](../amd-knowledge-base/layer-2-compute-stack/hip/hip-memory-management.md)
  - [HIP Synchronization](../amd-knowledge-base/layer-2-compute-stack/hip/hip-thread-synchronization.md)

- **Nvidia CUDA:**
  - [CUDA Fundamentals](../nvidia-knowledge-base/layer-2-compute-stack/cuda/cuda-gpu-programming-fundamentals.md)
  - [CUDA Memory](../nvidia-knowledge-base/layer-2-compute-stack/cuda/cuda-memory-management.md)
  - [CUDA Synchronization](../nvidia-knowledge-base/layer-2-compute-stack/cuda/cuda-thread-synchronization.md)

### Official Resources
- **HIP Porting Guide:** https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/hip_porting_guide.html
- **CUDA to HIP API:** https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_compare.html

### GPU Programming 101
- **Repository:** https://github.com/AIComputing101/gpu-programming-101
- **Modules 1-3:** Fundamentals, Memory, Synchronization
- **License:** MIT

