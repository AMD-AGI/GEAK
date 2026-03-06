---
layer: "2"
category: "hip"
subcategory: "fundamentals"
tags: ["hip", "gpu-programming", "basics", "kernel", "execution-model", "architecture"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
complexity: "beginner"
last_updated: 2025-11-20
source: "GPU Programming 101 - Module 1"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# HIP GPU Programming Fundamentals

*Foundation concepts for GPU programming with HIP on AMD GPUs*

## Overview

This guide covers fundamental GPU programming concepts using HIP (Heterogeneous-compute Interface for Portability), AMD's C++ programming language for GPU computing. Content is adapted from GPU Programming 101 Module 1: GPU Architecture & Programming Basics.

**Prerequisites:**
- C/C++ programming knowledge
- Understanding of basic parallel computing concepts
- ROCm 7.0+ installation

**Learning Objectives:**
- Understand GPU architecture and execution model
- Write and launch HIP kernels
- Manage GPU memory
- Understand thread hierarchy and execution

**Related Examples:**
- [hip-matmul-lds-optimized](/examples/amd/hip/optimized_matmul_lds.cpp)
- [hip-parallel-reduction](/examples/amd/hip/parallel_reduction.cpp)

## GPU Architecture Basics

### AMD CDNA Architecture

AMD's CDNA (Compute DNA) architecture powers the Instinct MI series GPUs, designed specifically for HPC and AI workloads.

```
┌─────────────────────────────────────────────┐
│          MI300X GPU (CDNA3)                  │
├─────────────────────────────────────────────┤
│  Compute Units (CUs): 304                    │
│  └─ Each CU has 4 SIMD units                │
│     └─ Each SIMD executes 64-wide wavefront │
│                                              │
│  Memory Hierarchy:                           │
│  ├─ HBM3: 192 GB @ 5.3 TB/s                 │
│  ├─ L2 Cache: 256 MB                        │
│  ├─ L1 Cache: 16 KB per CU                  │
│  └─ LDS (Local Data Share): 64 KB per CU   │
│                                              │
│  Peak Performance:                           │
│  ├─ FP32: 653 TFLOPS                        │
│  ├─ FP16: 1,307 TFLOPS                      │
│  └─ BF16 (MFMA): 1,307 TFLOPS               │
└─────────────────────────────────────────────┘
```

**Key Architectural Features:**
- **Wavefront Size:** 64 threads (AMD's equivalent of CUDA warps)
- **Compute Units (CUs):** Similar to CUDA Streaming Multiprocessors (SMs)
- **LDS (Local Data Share):** Fast on-chip memory, equivalent to CUDA shared memory
- **MFMA Instructions:** Matrix Fused Multiply-Add for AI acceleration

### Host-Device Model

```
┌──────────────────┐         ┌──────────────────┐
│   HOST (CPU)     │         │   DEVICE (GPU)   │
│                  │  PCIe/  │                  │
│  System Memory   │ ◄─────► │  HBM3 (VRAM)     │
│  (DDR RAM)       │ Infinity│                  │
│                  │  Fabric │  304 CUs         │
│  Launches        │         │  Execute         │
│  Kernels         │         │  Kernels         │
└──────────────────┘         └──────────────────┘
```

## HIP Programming Model

### Kernel Basics

A **kernel** is a function that runs on the GPU, executed by thousands of threads in parallel.

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

// Kernel definition - runs on GPU
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // Get global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);
    
    // Copy data to device
    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);
    
    // Launch kernel: 4 blocks × 256 threads = 1024 threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 
                       0, 0, d_A, d_B, d_C, N);
    
    // Copy result back to host
    hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            break;
        }
    }
    
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << std::endl;
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return 0;
}
```

**Compilation:**
```bash
hipcc -O3 -march=gfx90a vector_add.cpp -o vector_add
./vector_add
```

### Function Qualifiers

```cpp
// __global__: Kernel function - runs on GPU, called from host
__global__ void myKernel() {
    // Executed by many threads on GPU
}

// __device__: Device function - runs on GPU, called from GPU
__device__ float deviceFunction(float x) {
    return x * x;
}

// __host__: Host function - runs on CPU (default, usually omitted)
__host__ void hostFunction() {
    // Executed on CPU
}

// __host__ __device__: Can be compiled for both CPU and GPU
__host__ __device__ float square(float x) {
    return x * x;
}
```

## Thread Hierarchy

### Three-Level Hierarchy

HIP organizes threads in a three-level hierarchy: **Grid → Blocks → Threads**

```
Grid (entire kernel launch)
├─ Block (0, 0)
│  ├─ Thread (0, 0)
│  ├─ Thread (0, 1)
│  ├─ Thread (0, 2)
│  └─ ...
├─ Block (1, 0)
│  ├─ Thread (0, 0)
│  ├─ Thread (0, 1)
│  └─ ...
└─ Block (2, 0)
   └─ ...
```

### Thread Indexing

```cpp
__global__ void indexingExample() {
    // Built-in variables
    int blockId = blockIdx.x;     // Block index in grid
    int threadId = threadIdx.x;   // Thread index in block
    int blockSize = blockDim.x;   // Number of threads per block
    
    // Calculate global thread ID
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For 2D grids/blocks
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
}
```

### Launching Kernels

```cpp
// 1D launch: 1024 threads total (4 blocks × 256 threads)
dim3 grid(4);        // 4 blocks
dim3 block(256);     // 256 threads per block
hipLaunchKernelGGL(myKernel, grid, block, 0, 0);

// 2D launch: 256×256 threads total (16×16 blocks, each 16×16 threads)
dim3 grid2d(16, 16);
dim3 block2d(16, 16);
hipLaunchKernelGGL(myKernel2D, grid2d, block2d, 0, 0);

// With shared memory (512 bytes)
size_t sharedMemBytes = 512;
hipLaunchKernelGGL(myKernel, grid, block, sharedMemBytes, 0);
```

## Memory Management

### Memory Allocation and Transfer

```cpp
// Host-side memory management
float *h_data = new float[N];

// Device memory allocation
float *d_data;
hipMalloc(&d_data, N * sizeof(float));

// Host to Device
hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice);

// Device to Host
hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost);

// Device to Device
hipMemcpy(d_dest, d_src, bytes, hipMemcpyDeviceToDevice);

// Free device memory
hipFree(d_data);
```

### Memory Initialization

```cpp
// Set all bytes to 0
hipMemset(d_data, 0, N * sizeof(float));

// Asynchronous operations
hipStream_t stream;
hipStreamCreate(&stream);
hipMemcpyAsync(d_data, h_data, bytes, hipMemcpyHostToDevice, stream);
hipMemsetAsync(d_data, 0, bytes, stream);
```

## Wavefront Execution Model

### 64-Wide Wavefronts (CDNA)

AMD CDNA GPUs execute threads in groups of 64 called **wavefronts**.

```cpp
__global__ void wavefrontDemo() {
    // Get wavefront ID and lane ID
    int lane = threadIdx.x % 64;  // Lane within wavefront (0-63)
    int waveId = threadIdx.x / 64; // Wavefront ID within block
    
    // All 64 threads in a wavefront execute in lockstep
    if (lane == 0) {
        // Only first lane executes, but all 64 lanes wait
        printf("Wavefront %d\n", waveId);
    }
}
```

**Key Characteristics:**
- 64 threads execute together (SIMT: Single Instruction, Multiple Thread)
- Branch divergence causes serialization within a wavefront
- Wavefront-level primitives enable fast inter-thread communication

## Error Handling

### Checking HIP Calls

```cpp
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error: %s:%d, %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
HIP_CHECK(hipMalloc(&d_data, bytes));
HIP_CHECK(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
```

### Kernel Error Checking

```cpp
// Launch kernel
hipLaunchKernelGGL(myKernel, grid, block, 0, 0, args);

// Check for launch errors
HIP_CHECK(hipGetLastError());

// Wait for kernel completion and check errors
HIP_CHECK(hipDeviceSynchronize());
```

## Performance Considerations

### Occupancy

**Occupancy** = (Active wavefronts per CU) / (Maximum wavefronts per CU)

Factors affecting occupancy:
- **Threads per block:** Should be multiple of 64 (wavefront size)
- **Register usage:** High register use limits active wavefronts
- **LDS usage:** Limited to 64 KB per CU (CDNA3)

```cpp
// Query occupancy
int minGridSize, blockSize;
hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

printf("Recommended block size: %d\n", blockSize);
printf("Minimum grid size for max occupancy: %d\n", minGridSize);
```

### Memory Coalescing

For best performance, adjacent threads should access adjacent memory:

```cpp
// GOOD: Coalesced access
__global__ void coalescedAccess(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[i];  // Thread i accesses element i
}

// BAD: Strided access (poor performance)
__global__ void stridedAccess(float* data, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[i * stride];  // Threads access non-adjacent elements
}
```

## Best Practices

### Kernel Launch Configuration

1. **Thread block size:**
   - Use multiples of 64 (wavefront size)
   - Typical values: 128, 256, 512
   - Balance occupancy and resource usage

2. **Grid size:**
   - Ensure enough blocks to saturate GPU
   - Rule of thumb: blocks ≥ 2× number of CUs
   - MI300X: 304 CUs, so use ≥ 608 blocks

```cpp
// Calculate grid size
int threadsPerBlock = 256;  // Multiple of 64
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

// Ensure minimum block count for good occupancy
int minBlocks = 2 * 304;  // 2x CUs for MI300X
if (blocksPerGrid < minBlocks) {
    // Reduce threadsPerBlock to increase blocksPerGrid
    threadsPerBlock = 128;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
}
```

### Compilation Flags

```bash
# Basic optimization
hipcc -O3 my_kernel.cpp -o my_kernel

# Target specific architecture (MI250X = gfx90a)
hipcc -O3 -march=gfx90a my_kernel.cpp -o my_kernel

# MI300X (gfx942)
hipcc -O3 -march=gfx942 my_kernel.cpp -o my_kernel

# Generate assembly for inspection
hipcc -O3 -march=gfx90a -save-temps my_kernel.cpp

# Verbose compilation
hipcc -O3 -v my_kernel.cpp
```

## Common Pitfalls

### 1. Race Conditions

```cpp
// WRONG: Race condition
__global__ void raceCondition(int* counter) {
    (*counter)++;  // Multiple threads modifying same location
}

// CORRECT: Use atomics
__global__ void atomicUpdate(int* counter) {
    atomicAdd(counter, 1);  // Atomic operation
}
```

### 2. Uninitialized Memory

```cpp
// WRONG: Using uninitialized device memory
float* d_data;
hipMalloc(&d_data, N * sizeof(float));
myKernel<<<grid, block>>>(d_data);  // d_data contains garbage

// CORRECT: Initialize memory
hipMemset(d_data, 0, N * sizeof(float));
// or
hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice);
```

### 3. Synchronization Bugs

```cpp
// WRONG: No synchronization before reading result
hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice);
myKernel<<<grid, block>>>(d_data);
hipMemcpy(h_result, d_result, bytes, hipMemcpyDeviceToHost);  // May copy before kernel finishes

// CORRECT: Synchronize
myKernel<<<grid, block>>>(d_data);
hipDeviceSynchronize();  // Wait for kernel completion
hipMemcpy(h_result, d_result, bytes, hipMemcpyDeviceToHost);
```

## References

### Official Documentation
- **ROCm Documentation:** https://rocm.docs.amd.com/
- **HIP Programming Guide:** https://rocm.docs.amd.com/projects/HIP/
- **HIP API Reference:** https://rocm.docs.amd.com/projects/HIP/en/latest/reference/index.html

### Related Knowledge Base Sections
- [HIP Memory Management](./hip-memory-management.md)
- [HIP Thread Synchronization](./hip-thread-synchronization.md)
- [CDNA Architecture](../../layer-1-hardware/amd-gpu-arch/cdna-architecture.md)
- [HIP Performance Optimization](../../layer-3-libraries/algorithms/hip-performance-optimization.md)

### GPU Programming 101
- **Original Course:** https://github.com/AIComputing101/gpu-programming-101
- **Module 1:** GPU Architecture & Programming Basics
- **License:** MIT

### Example Code
- [Optimized Matrix Multiplication with LDS](/examples/amd/hip/optimized_matmul_lds.cpp)
- [Parallel Reduction](/examples/amd/hip/parallel_reduction.cpp)
- [2D Convolution](/examples/amd/hip/convolution_2d.cpp)
- [Wavefront Primitives](/examples/amd/hip/wavefront_primitives.hip)

