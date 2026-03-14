---
layer: "2"
category: "cuda"
subcategory: "synchronization"
tags: ["cuda", "synchronization", "warp", "atomics", "barriers", "primitives"]
cuda_version: "12.0+"
cuda_verified: "13.0"
complexity: "intermediate"
last_updated: 2025-11-20
source: "GPU Programming 101 - Module 3"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# CUDA Thread Synchronization

*Thread coordination, warp primitives, and atomic operations for Nvidia GPUs*

## Overview

Proper synchronization is essential for correct GPU programming. Adapted from GPU Programming 101 Module 3.

**Prerequisites:**
- [CUDA GPU Programming Fundamentals](./cuda-gpu-programming-fundamentals.md)
- [CUDA Memory Management](./cuda-memory-management.md)

**Related Examples:**
- [Warp Primitives](../../../examples/nvidia/cuda/warp_primitives.cu)
- [Parallel Reduction](../../../examples/nvidia/cuda/parallel_reduction.cu)

## Thread Hierarchy

```
GPU (H100: 132 SMs)
├─ SM (Streaming Multiprocessor)
│  ├─ Block (up to 2048 threads)
│  │  ├─ Warp 0 (32 threads - execute in lockstep)
│  │  ├─ Warp 1 (32 threads)
│  │  └─ Warp N
│  └─ Shared Memory: 227 KB
```

**Key:** Warp = 32 threads executing SIMT

## Block-Level Synchronization

### __syncthreads()

```cuda
__global__ void blockSyncExample() {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    
    // Write phase
    sdata[tid] = tid * 2.0f;
    
    // MUST synchronize
    __syncthreads();
    
    // Read phase (safe)
    float neighbor = sdata[(tid + 1) % 256];
}
```

### Correct vs Incorrect

```cuda
// ✗ WRONG: Conditional sync (deadlock!)
__global__ void badSync() {
    __shared__ float sdata[256];
    
    if (threadIdx.x < 128) {
        sdata[threadIdx.x] = 1.0f;
        __syncthreads();  // Only half reach here!
    }
}

// ✓ CORRECT: All threads reach barrier
__global__ void goodSync() {
    __shared__ float sdata[256];
    
    if (threadIdx.x < 128) {
        sdata[threadIdx.x] = 1.0f;
    }
    
    __syncthreads();  // All threads reach this
    
    if (threadIdx.x < 128) {
        // Use data
    }
}
```

## Warp-Level Primitives (32 threads)

### Shuffle Operations

```cuda
// Reduction using __shfl_down_sync
__device__ float warpSum(float val) {
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void reduceWithShuffle(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    float val = (tid < N) ? input[tid] : 0.0f;
    val = warpSum(val);
    
    if (lane == 0) {
        atomicAdd(output, val);
    }
}
```

### Shuffle Variants

```cuda
// __shfl_up_sync: Get from lower lane
__device__ float warpScan(float val) {
    unsigned mask = 0xffffffffu;
    int lane = threadIdx.x % 32;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float temp = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) {
            val += temp;
        }
    }
    return val;
}

// __shfl_sync: Broadcast from specific lane
__global__ void broadcast(float* data, int N) {
    unsigned mask = 0xffffffffu;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < N) ? data[tid] : 0.0f;
    float broadcast = __shfl_sync(mask, val, 0);  // From lane 0
    
    if (tid < N) {
        data[tid] = broadcast;
    }
}

// __shfl_xor_sync: XOR-based shuffle (butterfly)
__global__ void butterfly(float* data, int N) {
    unsigned mask = 0xffffffffu;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < N) ? data[tid] : 0.0f;
    float neighbor = __shfl_xor_sync(mask, val, 1);
    
    if (tid < N) {
        data[tid] = (val + neighbor) * 0.5f;
    }
}
```

### Voting Primitives

```cuda
__global__ void votingExample(const float* data, int* output, int N, float threshold) {
    unsigned mask = 0xffffffffu;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    float val = (tid < N) ? data[tid] : 0.0f;
    int predicate = (val > threshold);
    
    // Ballot: bitmask of which lanes pass
    unsigned ballot = __ballot_sync(mask, predicate);
    
    // Count
    int count = __popc(ballot);
    
    // Any/all
    int any_true = __any_sync(mask, predicate);
    int all_true = __all_sync(mask, predicate);
    
    if (lane == 0) {
        output[warpId * 3 + 0] = count;
        output[warpId * 3 + 1] = any_true;
        output[warpId * 3 + 2] = all_true;
    }
}
```

## Atomic Operations

### Basic Atomics

```cuda
__global__ void atomicExample(int* counter, float* sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        // Integer atomics
        atomicAdd(counter, 1);
        atomicSub(counter, 1);
        atomicMax(counter, tid);
        atomicMin(counter, tid);
        atomicInc(counter, N);
        atomicDec(counter, N);
        atomicExch(counter, tid);
        atomicCAS(counter, 0, 1);  // Compare-and-swap
        
        // Float atomics
        atomicAdd(sum, 1.0f);
        atomicMax(sum, float(tid));
        atomicMin(sum, float(tid));
    }
}
```

### atomicCAS (Compare-And-Swap)

```cuda
// Implement any atomic using CAS
__device__ float atomicMul(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        float new_val = __int_as_float(assumed) * val;
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    
    return __int_as_float(old);
}
```

### Reducing Atomic Contention

```cuda
// ✗ BAD: High contention
__global__ void highContention(int* counter, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(counter, 1);  // All threads contend!
    }
}

// ✓ BETTER: Block-level reduction
__global__ void lowContention(int* counter, int N) {
    __shared__ int block_counter;
    
    if (threadIdx.x == 0) {
        block_counter = 0;
    }
    __syncthreads();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(&block_counter, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(counter, block_counter);
    }
}

// ✓ BEST: Warp reduction
__global__ void minimalAtomics(int* counter, int N) {
    unsigned mask = 0xffffffffu;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    
    int val = (tid < N) ? 1 : 0;
    
    // Warp reduction (no atomics)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    
    // One atomic per warp
    if (lane == 0) {
        atomicAdd(counter, val);
    }
}
```

## Cooperative Groups

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void gridSync() {
    cg::grid_group grid = cg::this_grid();
    
    // Work...
    
    // Synchronize entire grid
    grid.sync();
    
    // Continue...
}

// Launch
void* args[] = { /* ... */ };
cudaLaunchCooperativeKernel((void*)gridSync, dim3(256), dim3(256), args);
```

## Stream Synchronization

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Async launches
kernel1<<<grid, block, 0, stream1>>>(args1);
kernel2<<<grid, block, 0, stream2>>>(args2);

// Sync specific stream
cudaStreamSynchronize(stream1);

// Sync all
cudaDeviceSynchronize();

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

## Memory Fences

```cuda
__threadfence();         // Device-wide
__threadfence_block();   // Block-wide
__threadfence_system();  // System-wide (host+device)
```

## Best Practices

1. **Minimize synchronization** - Use warp primitives when possible
2. **All threads must reach barriers** - Never in conditional code
3. **Reduce atomic contention** - Aggregate at warp/block level
4. **Appropriate scope** - Warp → Block → Device

## Common Pitfalls

```cuda
// ✗ Race condition
__shared__ int counter;
counter++;  // Multiple threads!

// ✓ Use atomic
atomicAdd(&counter, 1);

// ✗ Deadlock
if (threadIdx.x < N) {
    __syncthreads();  // Not all threads!
}
```

## References

- **CUDA Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions
- **GPU Programming 101:** https://github.com/AIComputing101/gpu-programming-101
- **Related:** [CUDA Fundamentals](./cuda-gpu-programming-fundamentals.md), [CUDA Memory](./cuda-memory-management.md)
- **Examples:** [Warp Primitives](../../../examples/nvidia/cuda/warp_primitives.cu), [Reduction](../../../examples/nvidia/cuda/parallel_reduction.cu)

