---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "2"
category: "cuda"
subcategory: "memory"
tags: ["cuda", "memory", "shared-memory", "global-memory", "coalescing", "optimization"]
cuda_version: "12.0+"
cuda_verified: "13.0"
complexity: "intermediate"
last_updated: 2025-11-20
source: "GPU Programming 101 - Module 2"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# CUDA Memory Management

*Comprehensive guide to memory types, patterns, and optimization for Nvidia GPUs*

## Overview

Effective memory management is critical for GPU performance. Adapted from GPU Programming 101 Module 2.

**Prerequisites:**
- [CUDA GPU Programming Fundamentals](./cuda-gpu-programming-fundamentals.md)

**Related Examples:**
- [Optimized Matrix Multiplication](../../../examples/nvidia/cuda/optimized_matmul_shared.cu)
- [2D Convolution](../../../examples/nvidia/cuda/convolution_2d.cu)

## Memory Hierarchy

```
┌───────────────────────────────────────┐
│ Registers (Per Thread)                │
│ - Fastest (~1 cycle)                  │
│ - Limited: 255 registers/thread       │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Shared Memory (Per Block)             │
│ - Fast: ~20-30 cycles                 │
│ - 48-164 KB per SM (configurable)     │
│ - Bank conflict considerations        │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ L1 Cache (Per SM)                     │
│ - Automatic caching                   │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ L2 Cache (GPU-wide)                   │
│ - 40-50 MB (H100)                     │
└───────────────────────────────────────┘
          ↓
┌───────────────────────────────────────┐
│ Global Memory (HBM)                   │
│ - 80 GB @ 3.35 TB/s (H100)           │
│ - High latency: ~400-800 cycles      │
└───────────────────────────────────────┘
```

## Global Memory

### Allocation and Transfer

```cuda
// Allocate
float *d_array;
cudaMalloc(&d_array, N * sizeof(float));

// Initialize
cudaMemset(d_array, 0, N * sizeof(float));

// Transfer
cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy(d_dest, d_src, bytes, cudaMemcpyDeviceToDevice);

// Free
cudaFree(d_array);
```

### Memory Coalescing

**Coalesced:** Adjacent threads access adjacent memory.

```cuda
// ✓ GOOD: Coalesced (optimal)
__global__ void coalesced(float* data, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = data[i];
}

// ✗ BAD: Strided (poor performance)
__global__ void strided(float* data, float* output, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = data[i * stride];
}

// ✗ BAD: Reversed
__global__ void reversed(float* data, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = data[N - i - 1];
}
```

**Performance Impact:**
- Coalesced: ~3.35 TB/s (H100 peak)
- Stride-2: ~1.7 TB/s (50% loss)
- Stride-4: ~850 GB/s (75% loss)

### SoA vs AoS

```cuda
// AoS (Poor for GPUs)
struct Particle {
    float x, y, z, vx, vy, vz;
};

__global__ void updateAoS(Particle* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    particles[i].x += particles[i].vx;  // Non-coalesced
}

// SoA (Optimal)
struct ParticlesSoA {
    float *x, *y, *z, *vx, *vy, *vz;
};

__global__ void updateSoA(ParticlesSoA particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    particles.x[i] += particles.vx[i];  // Coalesced!
}
```

## Shared Memory

### Basic Usage

```cuda
// Static allocation
__global__ void sharedStatic() {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    sdata[tid] = tid;
    __syncthreads();
    
    float neighbor = sdata[(tid + 1) % 256];
}

// Dynamic allocation
__global__ void sharedDynamic(float* g_data) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    sdata[tid] = g_data[blockIdx.x * blockDim.x + tid];
    __syncthreads();
}

// Launch with dynamic shared memory
size_t sharedBytes = 256 * sizeof(float);
sharedDynamic<<<grid, block, sharedBytes>>>(g_data);
```

### Tiled Matrix Multiplication

```cuda
#define TILE_SIZE 32

__global__ void matmulTiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles
        int aCol = t * TILE_SIZE + tx;
        if (row < M && aCol < K) {
            tileA[ty][tx] = A[row * K + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        int bRow = t * TILE_SIZE + ty;
        if (bRow < K && col < N) {
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Bank Conflicts

Shared memory has 32 banks. Simultaneous access to same bank causes serialization.

```cuda
// ✗ BAD: Bank conflicts
__shared__ float s[256];
float val = s[threadIdx.x * 32];  // All threads hit same bank

// ✓ GOOD: No conflicts
float val = s[threadIdx.x];  // Sequential access

// ✓ GOOD: Broadcast (all read same address)
float val = s[0];

// Padding to avoid conflicts
__shared__ float tile[32][33];  // Extra column avoids conflicts
```

## Constant Memory

```cuda
// Declare (64 KB limit)
__constant__ float c_coefficients[256];

// Copy from host
cudaMemcpyToSymbol(c_coefficients, h_coefficients, 256 * sizeof(float));

// Use in kernel
__global__ void applyFilter(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float result = 0.0f;
        for (int k = 0; k < 256; ++k) {
            result += data[i + k] * c_coefficients[k];  // Fast broadcast
        }
        data[i] = result;
    }
}
```

**Best for:**
- Small read-only data
- Uniform access (all threads read same element)
- Filter coefficients, lookup tables

## Texture Memory

```cuda
// Declare texture object
texture<float, 1, cudaReadModeElementType> texRef;

// Bind texture
cudaBindTexture(0, texRef, d_data, N * sizeof(float));

// Use in kernel
__global__ void texExample(float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = tex1Dfetch(texRef, i);  // Cached read
    }
}

// Unbind
cudaUnbindTexture(texRef);
```

## Unified Memory

```cuda
// Allocate unified memory
float* data;
cudaMallocManaged(&data, N * sizeof(float));

// Access from host
for (int i = 0; i < N; i++) {
    data[i] = i;
}

// Access from device (automatic migration)
myKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// Access from host again
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += data[i];
}

cudaFree(data);
```

## Advanced Patterns

### Reduction with Shared Memory

```cuda
__global__ void reduce(const float* input, float* output, int N) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
```

### Halo Exchange

```cuda
#define TILE_SIZE 16
#define HALO_SIZE 2

__global__ void convolution2D(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height)
{
    __shared__ float tile[TILE_SIZE + 2*HALO_SIZE][TILE_SIZE + 2*HALO_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    // Load tile with halo
    for (int i = ty; i < TILE_SIZE + 2*HALO_SIZE; i += blockDim.y) {
        for (int j = tx; j < TILE_SIZE + 2*HALO_SIZE; j += blockDim.x) {
            int input_row = blockIdx.y * TILE_SIZE + i - HALO_SIZE;
            int input_col = blockIdx.x * TILE_SIZE + j - HALO_SIZE;
            
            if (input_row >= 0 && input_row < height &&
                input_col >= 0 && input_col < width) {
                tile[i][j] = input[input_row * width + input_col];
            } else {
                tile[i][j] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // Apply filter
    if (col < width && row < height) {
        float sum = 0.0f;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                sum += tile[ty + i][tx + j] * c_filter[i * 5 + j];
            }
        }
        output[row * width + col] = sum;
    }
}
```

## Best Practices

1. **Coalesce global memory** - Use SoA, align data, avoid strided access
2. **Use shared memory for reuse** - Tile computations
3. **Avoid bank conflicts** - Add padding when needed
4. **Minimize global traffic** - Compute vs load when possible

## Common Pitfalls

### Missing Synchronization

```cuda
// ✗ WRONG: Race condition
__shared__ float sdata[256];
sdata[threadIdx.x] = data[i];
float val = sdata[threadIdx.x + 1];  // May read before write!

// ✓ CORRECT
sdata[threadIdx.x] = data[i];
__syncthreads();
float val = sdata[threadIdx.x + 1];
```

### Shared Memory Overuse

```cuda
// ✗ WRONG: Too much (reduces occupancy)
__shared__ float bigArray[16384];  // 64 KB!

// ✓ CORRECT
__shared__ float tile[32][32];  // 4 KB
```

## References

- **CUDA Memory Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy
- **GPU Programming 101:** https://github.com/AIComputing101/gpu-programming-101
- **Related:** [CUDA Fundamentals](./cuda-gpu-programming-fundamentals.md), [CUDA Thread Synchronization](./cuda-thread-synchronization.md)
- **Examples:** [Matrix Multiplication](../../../examples/nvidia/cuda/optimized_matmul_shared.cu), [Convolution](../../../examples/nvidia/cuda/convolution_2d.cu)

