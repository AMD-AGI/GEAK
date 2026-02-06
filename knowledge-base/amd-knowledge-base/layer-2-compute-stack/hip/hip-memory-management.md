---
layer: "2"
category: "hip"
subcategory: "memory"
tags: ["hip", "memory", "lds", "global-memory", "coalescing", "optimization"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
complexity: "intermediate"
last_updated: 2025-11-20
source: "GPU Programming 101 - Module 2"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# HIP Memory Management

*Comprehensive guide to memory types, patterns, and optimization for AMD GPUs*

## Overview

Effective memory management is critical for GPU performance. This guide covers HIP memory types, access patterns, and optimization techniques adapted from GPU Programming 101 Module 2.

**Prerequisites:**
- [HIP GPU Programming Fundamentals](./hip-gpu-programming-fundamentals.md)
- Understanding of memory hierarchies

**Learning Objectives:**
- Master different memory types (global, LDS, constant, private)
- Understand memory coalescing for optimal bandwidth
- Use LDS for inter-thread data sharing
- Apply tiling strategies for cache optimization

**Related Examples:**
- [Optimized Matrix Multiplication with LDS](../../../examples/amd/hip/optimized_matmul_lds.cpp)
- [2D Convolution](../../../examples/amd/hip/convolution_2d.cpp)

## Memory Hierarchy

### AMD GPU Memory Organization

```
┌─────────────────────────────────────────────────────────┐
│ Registers (Private Memory) - Per Thread                 │
│ - Fastest access (~1 cycle)                              │
│ - Limited: ~256 VGPRs per thread                         │
│ - Spilling to memory if exceeded                         │
└─────────────────────────────────────────────────────────┘
          ↓ (if shared)
┌─────────────────────────────────────────────────────────┐
│ LDS (Local Data Share) - Per Workgroup                  │
│ - Fast: ~10-20 cycles                                    │
│ - 64 KB per CU (CDNA3)                                   │
│ - Shared within wavefronts/workgroup                     │
│ - Bank conflict considerations                           │
└─────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ L1 Cache - Per CU                                        │
│ - 16 KB per CU                                           │
│ - Automatic caching of global memory                     │
└─────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ L2 Cache - Shared across GPU                            │
│ - 256 MB (MI300X)                                        │
│ - Reduces HBM access                                     │
└─────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────┐
│ HBM (High Bandwidth Memory) - Global Memory             │
│ - 192 GB @ 5.3 TB/s (MI300X)                            │
│ - High latency: ~300-400 cycles                         │
│ - Coalescing critical for performance                    │
└─────────────────────────────────────────────────────────┘
```

## Global Memory

### Allocation and Transfer

```cpp
#include <hip/hip_runtime.h>

// Allocate device memory
float *d_array;
size_t bytes = N * sizeof(float);
hipMalloc(&d_array, bytes);

// Initialize to zero
hipMemset(d_array, 0, bytes);

// Transfer data
float *h_array = new float[N];
hipMemcpy(d_array, h_array, bytes, hipMemcpyHostToDevice);  // Host → Device
hipMemcpy(h_array, d_array, bytes, hipMemcpyDeviceToHost);  // Device → Host

// Device-to-device copy
float *d_copy;
hipMalloc(&d_copy, bytes);
hipMemcpy(d_copy, d_array, bytes, hipMemcpyDeviceToDevice);

// Free memory
hipFree(d_array);
delete[] h_array;
```

### Memory Coalescing

**Coalesced access:** Adjacent threads access adjacent memory addresses.

```cpp
// ✓ GOOD: Coalesced access (optimal bandwidth)
__global__ void coalescedRead(float* data, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = data[i];  // Thread 0→data[0], Thread 1→data[1], etc.
}

// ✗ BAD: Strided access (poor bandwidth)
__global__ void stridedRead(float* data, float* output, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = data[i * stride];  // Non-adjacent memory accesses
}

// ✗ BAD: Reverse access (poor bandwidth)
__global__ void reverseRead(float* data, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = data[N - i - 1];  // Backward memory access
}
```

**Performance Impact:**
- Coalesced: ~5.3 TB/s (MI300X peak)
- Strided (stride=2): ~2.6 TB/s (50% loss)
- Strided (stride=4): ~1.3 TB/s (75% loss)

### Structure of Arrays vs Array of Structures

```cpp
// AoS (Array of Structures) - Poor for GPUs
struct Particle {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
};

__global__ void updateAoS(Particle* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].x += particles[i].vx;  // Non-coalesced
    }
}

// SoA (Structure of Arrays) - Optimal for GPUs
struct ParticlesSoA {
    float *x, *y, *z;     // Positions
    float *vx, *vy, *vz;  // Velocities
};

__global__ void updateSoA(ParticlesSoA particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles.x[i] += particles.vx[i];  // Coalesced!
    }
}
```

## LDS (Local Data Share)

### Basics

LDS is fast on-chip memory shared by all threads in a workgroup (block).

```cpp
// Static allocation (compile-time size)
__global__ void ldsStat

icExample() {
    __shared__ float lds_data[256];  // 256 floats = 1 KB
    
    int tid = threadIdx.x;
    
    // Load from global memory
    extern float *g_data;  // Assume this is passed
    lds_data[tid] = g_data[blockIdx.x * blockDim.x + tid];
    
    // Synchronize: ensure all threads have written
    __syncthreads();
    
    // Access neighbor's data
    if (tid < 255) {
        float avg = (lds_data[tid] + lds_data[tid + 1]) * 0.5f;
    }
}

// Dynamic allocation (runtime size)
__global__ void ldsDynamicExample(float* g_data) {
    extern __shared__ float lds_data[];  // Size determined at launch
    
    int tid = threadIdx.x;
    lds_data[tid] = g_data[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    
    // Use lds_data...
}

// Launch with dynamic LDS
size_t ldsBytes = 256 * sizeof(float);
hipLaunchKernelGGL(ldsDynamicExample, grid, block, ldsBytes, 0, g_data);
```

### Tiled Matrix Multiplication

Classic example of LDS usage for performance optimization:

```cpp
#define TILE_SIZE 32

__global__ void matmulTiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Allocate LDS for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A into LDS (coalesced)
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            tileA[ty][tx] = A[row * K + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        // Load tile of B into LDS (coalesced)
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();  // Wait for tiles to load
        
        // Compute partial dot product from tiles
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();  // Wait before loading next tile
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Benefits:**
- Reduces global memory accesses by factor of TILE_SIZE
- 32×32 tile: 32× reduction (2 loads + 1 store instead of 64 loads + 1 store)
- MI300X: ~800 GFLOPS achieved vs ~50 GFLOPS without tiling

### Bank Conflicts

LDS is organized into **banks** (32 banks on CDNA). Simultaneous access to the same bank by different threads causes serialization.

```cpp
// ✗ BAD: Bank conflicts (stride of 32)
__shared__ float lds[256];
__syncthreads();
float val = lds[threadIdx.x * 32];  // All threads hit same bank

// ✓ GOOD: No bank conflicts (sequential access)
float val = lds[threadIdx.x];  // Each thread accesses different bank

// ✓ GOOD: Broadcasting (all threads read same address is okay)
float val = lds[0];  // All threads read same element - broadcast
```

**Padding to Avoid Conflicts:**

```cpp
// Without padding: 32×32 causes conflicts on column access
__shared__ float tile[32][32];
float val = tile[threadIdx.x][0];  // Conflict!

// With padding: 32×33 avoids conflicts
__shared__ float tile[32][33];  // Extra column for padding
float val = tile[threadIdx.x][0];  // No conflict!
```

## Constant Memory

### Usage

Constant memory is cached and broadcast-optimized for uniform reads across a wavefront.

```cpp
// Declare constant memory (64 KB limit)
__constant__ float c_coefficients[256];

// Copy to constant memory (from host)
float h_coefficients[256];
// ... initialize h_coefficients ...
hipMemcpyToSymbol(c_coefficients, h_coefficients, 256 * sizeof(float));

// Use in kernel
__global__ void applyFilter(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float result = 0.0f;
        for (int k = 0; k < 256; ++k) {
            result += data[i + k] * c_coefficients[k];  // Fast broadcast read
        }
        data[i] = result;
    }
}
```

**Best For:**
- Small read-only data (<64 KB)
- Uniform access (all threads read same element)
- Filter coefficients, lookup tables, constants

**Avoid For:**
- Large data (use global memory)
- Non-uniform access (use LDS or global memory)

## Unified Memory (Managed Memory)

### Automatic Migration

```cpp
// Allocate unified memory
float *data;
hipMallocManaged(&data, N * sizeof(float));

// Access from host
for (int i = 0; i < N; i++) {
    data[i] = i;
}

// Access from device (automatic migration)
myKernel<<<grid, block>>>(data, N);

// Wait and access from host again
hipDeviceSynchronize();
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += data[i];
}

// Free unified memory
hipFree(data);
```

**Pros:**
- Simplified memory management
- Single pointer for CPU and GPU

**Cons:**
- Page fault overhead on first access
- Less control over data placement
- May be slower than explicit transfers for large data

## Advanced Patterns

### Halo Exchange for Convolution

```cpp
#define TILE_SIZE 16
#define FILTER_SIZE 5
#define HALO_SIZE (FILTER_SIZE / 2)  // 2

__global__ void convolution2D(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height)
{
    // Tile with halo region
    __shared__ float tile[TILE_SIZE + 2*HALO_SIZE][TILE_SIZE + 2*HALO_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    // Load tile including halo (requires multiple loads per thread)
    for (int i = ty; i < TILE_SIZE + 2*HALO_SIZE; i += blockDim.y) {
        for (int j = tx; j < TILE_SIZE + 2*HALO_SIZE; j += blockDim.x) {
            int input_row = blockIdx.y * TILE_SIZE + i - HALO_SIZE;
            int input_col = blockIdx.x * TILE_SIZE + j - HALO_SIZE;
            
            // Boundary handling
            if (input_row >= 0 && input_row < height &&
                input_col >= 0 && input_col < width) {
                tile[i][j] = input[input_row * width + input_col];
            } else {
                tile[i][j] = 0.0f;  // Zero-padding
            }
        }
    }
    
    __syncthreads();
    
    // Apply filter
    if (col < width && row < height) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                sum += tile[ty + i][tx + j] * c_filter[i * FILTER_SIZE + j];
            }
        }
        output[row * width + col] = sum;
    }
}
```

### Reduction with LDS

```cpp
__global__ void reduceSum(const float* input, float* output, int N) {
    __shared__ float lds[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Load and reduce multiple elements per thread (grid-stride)
    float sum = 0.0f;
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        sum += input[idx];
    }
    lds[tid] = sum;
    __syncthreads();
    
    // Reduction in LDS (sequential addressing)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            lds[tid] += lds[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicAdd(output, lds[0]);
    }
}
```

## Performance Optimization

### Memory Bandwidth Benchmarking

```cpp
__global__ void copyKernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i];
    }
}

// Measure bandwidth
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

hipEventRecord(start);
copyKernel<<<grid, block>>>(d_input, d_output, N);
hipEventRecord(stop);
hipEventSynchronize(stop);

float ms;
hipEventElapsedTime(&ms, start, stop);

size_t bytes = 2 * N * sizeof(float);  // Read + write
double bandwidth_GB = (bytes / 1e9) / (ms / 1000.0);
printf("Bandwidth: %.2f GB/s\n", bandwidth_GB);
```

### Best Practices Summary

1. **Coalesce global memory access**
   - Align data structures to 128-byte boundaries
   - Use SoA instead of AoS
   - Avoid strided access patterns

2. **Use LDS for data reuse**
   - Tile computations to fit in LDS
   - Avoid bank conflicts with padding
   - Synchronize after LDS writes

3. **Minimize global memory traffic**
   - Compute rather than load when possible
   - Use registers for temporary values
   - Fuse kernels to avoid intermediate storage

4. **Use constant memory appropriately**
   - Small read-only data (<64 KB)
   - Uniform access patterns
   - Filter coefficients, lookup tables

## Common Pitfalls

### 1. Missing Synchronization

```cpp
// ✗ WRONG: Race condition
__shared__ float lds[256];
lds[threadIdx.x] = data[i];
float val = lds[threadIdx.x + 1];  // May read before neighbor writes!

// ✓ CORRECT: Synchronize
lds[threadIdx.x] = data[i];
__syncthreads();
float val = lds[threadIdx.x + 1];
```

### 2. LDS Overuse

```cpp
// ✗ WRONG: Too much LDS (reduces occupancy)
__shared__ float bigArray[16384];  // 64 KB - entire CU limit!

// ✓ CORRECT: Use reasonable amount
__shared__ float tile[32][32];  // 4 KB - allows multiple blocks per CU
```

### 3. Unaligned Memory Access

```cpp
// ✗ WRONG: Misaligned allocation
float* d_data = (float*)((char*)d_buffer + 1);  // Misaligned!

// ✓ CORRECT: Proper alignment
float* d_data;
hipMalloc(&d_data, N * sizeof(float));  // Automatically aligned
```

## References

### Related Documentation
- [HIP GPU Programming Fundamentals](./hip-gpu-programming-fundamentals.md)
- [HIP Thread Synchronization](./hip-thread-synchronization.md)
- [HIP Performance Optimization](../../layer-3-libraries/algorithms/hip-performance-optimization.md)
- [CDNA Architecture](../../layer-1-hardware/amd-gpu-arch/cdna-architecture.md)

### Official Resources
- **HIP Memory Guide:** https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- **AMD GPU Architecture:** https://www.amd.com/en/technologies/cdna

### GPU Programming 101
- **Repository:** https://github.com/AIComputing101/gpu-programming-101
- **Module 2:** Memory Management and Optimization
- **License:** MIT

### Example Code
- [Optimized Matrix Multiplication with LDS](../../../examples/amd/hip/optimized_matmul_lds.cpp)
- [2D Convolution](../../../examples/amd/hip/convolution_2d.cpp)
- [Parallel Reduction](../../../examples/amd/hip/parallel_reduction.cpp)

