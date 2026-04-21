---
layer: "2"
category: "cuda"
subcategory: "fundamentals"
tags: ["cuda", "gpu-programming", "basics", "kernel", "execution-model", "nvidia"]
cuda_version: "12.0+"
cuda_verified: "13.0"
complexity: "beginner"
last_updated: 2025-11-20
source: "GPU Programming 101 - Module 1"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# CUDA GPU Programming Fundamentals

*Foundation concepts for GPU programming with CUDA on Nvidia GPUs*

## Overview

This guide covers fundamental GPU programming concepts using CUDA, Nvidia's parallel computing platform. Content is adapted from GPU Programming 101 Module 1.

**Prerequisites:**
- C/C++ programming knowledge
- Basic parallel computing concepts
- CUDA Toolkit 12.0+ installation

**Learning Objectives:**
- Understand Nvidia GPU architecture
- Write and launch CUDA kernels
- Manage GPU memory
- Understand thread hierarchy

**Related Examples:**
- [cuda-matmul-shared-optimized](/examples/nvidia/cuda/optimized_matmul_shared.cu)
- [cuda-parallel-reduction](/examples/nvidia/cuda/parallel_reduction.cu)

## Nvidia GPU Architecture

### Hopper Architecture (H100/H200)

```
┌─────────────────────────────────────────────┐
│          H100 GPU (Hopper)                   │
├─────────────────────────────────────────────┤
│  SMs (Streaming Multiprocessors): 132       │
│  └─ Each SM has 4 warp schedulers           │
│     └─ Each schedules 32-thread warps       │
│                                              │
│  Memory Hierarchy:                           │
│  ├─ HBM3: 80 GB @ 3.35 TB/s                │
│  ├─ L2 Cache: 50 MB                         │
│  ├─ L1 Cache/Shared: 256 KB per SM         │
│  └─ Registers: 65,536 per SM               │
│                                              │
│  Peak Performance:                           │
│  ├─ FP32: 67 TFLOPS                         │
│  ├─ FP16 Tensor Core: 1,979 TFLOPS         │
│  └─ FP8 Tensor Core: 3,958 TFLOPS          │
└─────────────────────────────────────────────┘
```

**Key Features:**
- **Warp Size:** 32 threads
- **SMs:** Similar to AMD CUs
- **Shared Memory:** Fast on-chip memory
- **Tensor Cores:** Specialized for matrix operations

## CUDA Programming Model

### Hello World Kernel

```cuda
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    
    // Host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }
    
    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Launch: 4 blocks × 256 threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            break;
        }
    }
    
    std::cout << (success ? "PASS" : "FAIL") << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return 0;
}
```

**Compilation:**
```bash
nvcc -O3 -arch=sm_80 vector_add.cu -o vector_add
./vector_add
```

### Function Qualifiers

```cuda
// __global__: Kernel - runs on GPU, called from host
__global__ void myKernel() {
    // GPU code
}

// __device__: Device function - runs on GPU, called from GPU
__device__ float deviceFunc(float x) {
    return x * x;
}

// __host__: Host function - runs on CPU (default)
__host__ void hostFunc() {
    // CPU code
}

// __host__ __device__: Dual compilation
__host__ __device__ float square(float x) {
    return x * x;
}
```

## Thread Hierarchy

### Grid → Block → Thread

```
Grid
├─ Block (0,0)
│  ├─ Warp 0: Threads 0-31
│  ├─ Warp 1: Threads 32-63
│  └─ ...
├─ Block (1,0)
└─ Block (2,0)
```

### Thread Indexing

```cuda
__global__ void indexing() {
    // 1D indexing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2D indexing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Warp info
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
}
```

### Launching Kernels

```cuda
// 1D launch
dim3 grid(4);
dim3 block(256);
myKernel<<<grid, block>>>();

// 2D launch
dim3 grid2d(16, 16);
dim3 block2d(16, 16);
myKernel2D<<<grid2d, block2d>>>();

// With shared memory
size_t sharedMem = 512;
myKernel<<<grid, block, sharedMem>>>();
```

## Memory Management

```cuda
// Allocate
float* d_data;
cudaMalloc(&d_data, N * sizeof(float));

// Initialize
cudaMemset(d_data, 0, N * sizeof(float));

// Transfer
cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

// Free
cudaFree(d_data);
```

## Warp Execution (32 threads)

```cuda
__global__ void warpDemo() {
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    // All 32 threads execute together (SIMT)
    if (lane == 0) {
        printf("Warp %d\n", warpId);
    }
}
```

## Error Handling

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s:%d, %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, bytes));
CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

// Check kernel errors
myKernel<<<grid, block>>>();
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## Performance Basics

### Occupancy

```cuda
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

printf("Recommended block size: %d\n", blockSize);
```

### Memory Coalescing

```cuda
// GOOD: Coalesced
__global__ void coalesced(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[i];  // Adjacent threads → adjacent memory
}

// BAD: Strided
__global__ void strided(float* data, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[i * stride];  // Non-adjacent
}
```

## Best Practices

1. **Thread block size:** Multiples of 32 (warp size)
2. **Grid size:** Enough blocks to saturate GPU (≥ 2× SM count)
3. **Compilation:** Target specific architecture (`-arch=sm_80`)

## References

- **CUDA Documentation:** https://docs.nvidia.com/cuda/
- **GPU Programming 101:** https://github.com/AIComputing101/gpu-programming-101
- **Related:** [CUDA Memory Management](./cuda-memory-management.md), [CUDA Thread Synchronization](./cuda-thread-synchronization.md)
- **Examples:** [Matrix Multiplication](/examples/nvidia/cuda/optimized_matmul_shared.cu), [Parallel Reduction](/examples/nvidia/cuda/parallel_reduction.cu)

