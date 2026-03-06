Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "3"
category: "algorithms"
subcategory: "optimization"
tags: ["hip", "optimization", "performance", "lds", "mfma", "occupancy"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
complexity: "advanced"
last_updated: 2025-11-20
source: "GPU Programming 101 - Modules 4-6"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# HIP Performance Optimization

*Advanced optimization techniques for AMD GPUs*

## Overview

Advanced performance optimization techniques from GPU Programming 101 Modules 4-6: memory optimization, instruction-level parallelism, and occupancy tuning.

**Prerequisites:**
- [HIP Fundamentals](../../layer-2-compute-stack/hip/hip-gpu-programming-fundamentals.md)
- [HIP Memory Management](../../layer-2-compute-stack/hip/hip-memory-management.md)
- [HIP Thread Synchronization](../../layer-2-compute-stack/hip/hip-thread-synchronization.md)

**Related Examples:**
- [Optimized Matrix Multiplication](../../../../examples/amd/hip/optimized_matmul_lds.cpp)
- [Parallel Reduction](../../../../examples/amd/hip/parallel_reduction.cpp)
- [2D Convolution](../../../../examples/amd/hip/convolution_2d.cpp)

## Module 4: Memory Optimization

### LDS Optimization Strategies

**1. Tiling for Data Reuse**

```cpp
#define TILE_SIZE 32

__global__ void tiledComputation(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Load tile (coalesced)
    tileA[threadIdx.y][threadIdx.x] = A[...];
    __syncthreads();
    
    // Compute using tile (high data reuse)
    for (int k = 0; k < TILE_SIZE; ++k) {
        result += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
}
```

**2. Bank Conflict Avoidance**

```cpp
// Padding to avoid conflicts
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 padding

// Double buffering
__shared__ float buffer[2][TILE_SIZE][TILE_SIZE];
int current = 0;
```

### Memory Coalescing Patterns

```cpp
// Global memory transactions
// 128-byte transactions (MI300X)
// Best: 128 consecutive bytes by 64 threads = 2 bytes/thread (half)

// Optimal: float (4 bytes)
float val = data[tid];  // 32 threads × 4B = 128B transaction

// Good: float2 (8 bytes)  
float2 val = ((float2*)data)[tid];  // 16 threads × 8B

// Poor: Strided access
float val = data[tid * stride];  // Multiple transactions
```

## Module 5: Instruction-Level Parallelism (ILP)

### Loop Unrolling

```cpp
// Manual unrolling
__global__ void unrolled(float* data, int N) {
    int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    // Process 4 elements per thread
    float4 vals;
    vals.x = (i + 0 * blockDim.x < N) ? data[i + 0 * blockDim.x] : 0.0f;
    vals.y = (i + 1 * blockDim.x < N) ? data[i + 1 * blockDim.x] : 0.0f;
    vals.z = (i + 2 * blockDim.x < N) ? data[i + 2 * blockDim.x] : 0.0f;
    vals.w = (i + 3 * blockDim.x < N) ? data[i + 3 * blockDim.x] : 0.0f;
    
    // Process vals...
}

// Compiler-directed
#pragma unroll 4
for (int i = 0; i < 32; i++) {
    sum += a[i] * b[i];
}
```

### Vectorized Memory Access

```cpp
// Load 4 floats at once
__global__ void vectorized(float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4* in4 = (float4*)in;
    float4* out4 = (float4*)out;
    
    if (i * 4 < N) {
        float4 val = in4[i];  // One 128-bit transaction
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;
        out4[i] = val;
    }
}
```

### MFMA Intrinsics (Matrix Operations)

```cpp
// Matrix Fused Multiply-Add for BF16 (MI300X)
#include <hip/hip_bfloat16.h>

__global__ void mfmaExample() {
    // MFMA 16x16x16 BF16 operation
    // Computes C = A * B + C
    // Where A, B are 16x16 BF16 matrices
    
    __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(...);
    
    // Achieves 1307 TFLOPS on MI300X
}
```

## Module 6: Occupancy and Resource Management

### Occupancy Calculation

```cpp
// Query occupancy
int minGridSize, blockSize;
hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

// Calculate active wavefronts
// Occupancy = (Active Wavefronts per CU) / (Max Wavefronts per CU)
// MI300X: Max 32 wavefronts/CU (2048 threads / 64)

// Factors limiting occupancy:
// 1. Registers: 512 VGPRs/wavefront (MI300X)
// 2. LDS: 64 KB per CU
// 3. Wavefronts: Max 32/CU
```

### Register Pressure Management

```cpp
// Check register usage
// hipcc --resource-usage kernel.hip

// Reduce register pressure
// 1. Use LDS for temporary values
// 2. Recompute instead of store
// 3. Launch attributes

__global__ void 
__launch_bounds__(256, 4)  // 256 threads/block, min 4 blocks/CU
optimized Kernel() {
    // Limited register use
}
```

### LDS Usage Optimization

```cpp
// Dynamic vs static allocation
__global__ void dynamicLDS() {
    extern __shared__ float lds[];  // Runtime size
    // Use when size varies
}

__global__ void staticLDS() {
    __shared__ float lds[1024];  // Compile-time size
    // Better performance when size is known
}

// LDS limits (MI300X):
// - 64 KB per CU
// - Shared across all wavefronts in workgroup
// Example: 4 blocks × 16 KB = 64 KB (full utilization)
```

## Advanced Optimization Patterns

### Grid-Stride Loops

```cpp
__global__ void gridStride(float* data, int N) {
    // Each thread processes multiple elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < N; 
         i += blockDim.x * gridDim.x) {
        data[i] = process(data[i]);
    }
}

// Benefits:
// - Handle any array size
// - Better occupancy
// - Reduced kernel launch overhead
```

### Persistent Threads

```cpp
__global__ void persistentThreads(float* data, int* queue, int N) {
    // Thread stays alive and processes multiple work items
    while (true) {
        int work_id = atomicAdd(&queue[0], 1);
        if (work_id >= N) break;
        
        process(data[work_id]);
    }
}
```

### Stream Pipelining

```cpp
hipStream_t streams[4];
for (int i = 0; i < 4; i++) {
    hipStreamCreate(&streams[i]);
}

// Overlap compute and memory transfers
for (int i = 0; i < N; i++) {
    int sid = i % 4;
    hipMemcpyAsync(d_data, h_data, bytes, hipMemcpyHostToDevice, streams[sid]);
    kernel<<<grid, block, 0, streams[sid]>>>(d_data);
    hipMemcpyAsync(h_result, d_result, bytes, hipMemcpyDeviceToHost, streams[sid]);
}
```

## Performance Measurement

### rocprof Basics

```bash
# Basic profiling
rocprof --stats ./my_app

# Collect specific metrics
rocprof --hsa-trace --hip-trace ./my_app

# Performance counters
rocprof --timestamp on -i metrics.txt ./my_app

# metrics.txt example:
# pmc: SQ_INSTS_VALU
# pmc: SQ_INSTS_MFMA_MOPS_BF16
# pmc: TA_BUSY_avr
```

### Metrics to Watch

```
Key Performance Indicators (MI300X):

1. Memory Bandwidth Utilization
   - Target: >80% of 5.3 TB/s
   - Metric: Effective BW / Peak BW

2. Compute Utilization
   - FP32: Target >400 GFLOPS
   - BF16 (MFMA): Target >1000 TFLOPS
   - Metric: VALU_UTIL, MFMA_UTIL

3. Occupancy
   - Target: 50-75%
   - Metric: Active wavefronts / Max wavefronts

4. Cache Hit Rates
   - L2: Target >90%
   - L1: Target >80%
   
5. Memory Latency
   - HBM: ~300-400 cycles
   - LDS: ~20 cycles
```

## Best Practices Summary

1. **Memory:**
   - Coalesce all global memory access
   - Use LDS for data reuse (tiling)
   - Minimize bank conflicts
   - Vectorize loads when possible

2. **Compute:**
   - Maximize ILP (unroll loops)
   - Use MFMA for matrix operations
   - Keep arithmetic intensity high

3. **Resources:**
   - Target 50-75% occupancy
   - Balance registers vs LDS
   - Use __launch_bounds__ appropriately

4. **Execution:**
   - Use wavefront primitives
   - Minimize divergence
   - Aggregate atomics

## References

- **ROCm Profiling:** https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
- **GPU Programming 101:** https://github.com/AIComputing101/gpu-programming-101
- **Related:** [HIP Memory](../../layer-2-compute-stack/hip/hip-memory-management.md), [HIP Synchronization](../../layer-2-compute-stack/hip/hip-thread-synchronization.md)
- **Examples:** [Matrix Multiplication](../../../../examples/amd/hip/optimized_matmul_lds.cpp), [Reduction](../../../../examples/amd/hip/parallel_reduction.cpp)

