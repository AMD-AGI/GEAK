Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "2"
category: "hip"
subcategory: "synchronization"
tags: ["hip", "synchronization", "wavefront", "atomics", "barriers", "primitives"]
rocm_version: "7.0+"
rocm_verified: "7.0.2"
complexity: "intermediate"
last_updated: 2025-11-20
source: "GPU Programming 101 - Module 3"
reference: "https://github.com/AIComputing101/gpu-programming-101"
---

# HIP Thread Synchronization

*Mastering thread coordination, wavefront primitives, and atomic operations on AMD GPUs*

## Overview

Proper thread synchronization is essential for correct and efficient GPU programming. This guide covers synchronization primitives, wavefront-level operations, and atomic operations for AMD CDNA architectures, adapted from GPU Programming 101 Module 3.

**Prerequisites:**
- [HIP GPU Programming Fundamentals](./hip-gpu-programming-fundamentals.md)
- [HIP Memory Management](./hip-memory-management.md)

**Learning Objectives:**
- Understand thread hierarchy and execution model
- Use synchronization barriers correctly
- Master wavefront-level primitives
- Apply atomic operations safely
- Avoid common synchronization pitfalls

**Related Examples:**
- [Wavefront Primitives](../../../examples/amd/hip/wavefront_primitives.hip)
- [Parallel Reduction](../../../examples/amd/hip/parallel_reduction.cpp)

## Thread Hierarchy Review

### Execution Model

```
GPU
├─ Compute Unit (CU) × 304 (MI300X)
│  ├─ Workgroup (Block) - up to 1024 threads
│  │  ├─ Wavefront 0 - 64 threads (execute in lockstep)
│  │  ├─ Wavefront 1 - 64 threads
│  │  └─ Wavefront N
│  └─ LDS: 64 KB shared across workgroup
└─ ...
```

**Key Concepts:**
- **Wavefront:** 64 threads executing in SIMT (Single Instruction, Multiple Thread)
- **Workgroup:** Collection of wavefronts sharing LDS
- **Lane:** Individual thread within a wavefront (0-63)

## Block-Level Synchronization

### __syncthreads()

Synchronizes all threads within a workgroup (block).

```cpp
__global__ void blockSyncExample() {
    __shared__ float lds[256];
    
    int tid = threadIdx.x;
    
    // Phase 1: Write to LDS
    lds[tid] = tid * 2.0f;
    
    // ✓ MUST synchronize before reading
    __syncthreads();
    
    // Phase 2: Read from LDS (safe - all threads have written)
    float neighbor = lds[(tid + 1) % 256];
}
```

**Important Rules:**
1. **All threads must reach the barrier:** Cannot be in conditional code where some threads don't execute
2. **Synchronizes within workgroup only:** Does not sync across different workgroups
3. **Does not guarantee memory visibility across CUs:** Use atomics or device sync for that

### Correct vs Incorrect Usage

```cpp
// ✗ WRONG: Conditional synchronization (deadlock!)
__global__ void badSync() {
    __shared__ float lds[256];
    
    if (threadIdx.x < 128) {
        lds[threadIdx.x] = 1.0f;
        __syncthreads();  // Only half of threads reach here!
    }
}

// ✓ CORRECT: All threads reach barrier
__global__ void goodSync() {
    __shared__ float lds[256];
    
    if (threadIdx.x < 128) {
        lds[threadIdx.x] = 1.0f;
    }
    
    __syncthreads();  // All threads reach this point
    
    if (threadIdx.x < 128) {
        // Use data...
    }
}

// ✗ WRONG: Loop-dependent synchronization
__global__ void badLoopSync(int* data, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        data[i] = i;
        __syncthreads();  // Different threads execute different iterations!
    }
}

// ✓ CORRECT: Synchronize outside loop or ensure all threads iterate same count
__global__ void goodLoopSync(int* data, int N) {
    int iters = (N + blockDim.x - 1) / blockDim.x;
    for (int iter = 0; iter < iters; ++iter) {
        int i = iter * blockDim.x + threadIdx.x;
        if (i < N) {
            data[i] = i;
        }
        __syncthreads();  // All threads execute same iterations
    }
}
```

## Wavefront-Level Primitives

### Wavefront Basics (64 threads)

AMD CDNA GPUs execute 64 threads per wavefront in lockstep (SIMT).

```cpp
__global__ void wavefrontInfo() {
    int lane = threadIdx.x % 64;           // Lane ID: 0-63
    int waveId = threadIdx.x / 64;         // Wavefront ID in block
    
    // Wavefront-level operations don't need __syncthreads()
    // All 64 lanes execute simultaneously
}
```

### Shuffle Operations

Shuffle operations allow direct data exchange between lanes in a wavefront **without LDS**.

#### __shfl_down

```cpp
// Shuffle data from lane + offset
__device__ float wavefrontSum(float val) {
    // Sum reduction across 64 lanes
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 64);
        // Each lane gets value from lane (id + offset)
    }
    return val;  // Lane 0 has the sum
}

__global__ void reduceWithShuffle(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 64;
    int waveId = threadIdx.x / 64;
    
    // Load data
    float val = (tid < N) ? input[tid] : 0.0f;
    
    // Wavefront reduction
    val = wavefrontSum(val);
    
    // First lane writes result
    if (lane == 0) {
        atomicAdd(output, val);
    }
}
```

#### __shfl_up

```cpp
// Inclusive prefix sum (scan)
__device__ float wavefrontScan(float val) {
    int lane = threadIdx.x % 64;
    
    #pragma unroll
    for (int offset = 1; offset < 64; offset <<= 1) {
        float temp = __shfl_up(val, offset, 64);
        if (lane >= offset) {
            val += temp;
        }
    }
    return val;
}

__global__ void prefixSum(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < N) ? input[tid] : 0.0f;
    val = wavefrontScan(val);
    
    if (tid < N) {
        output[tid] = val;
    }
}
```

#### __shfl

```cpp
// Broadcast from specific lane
__global__ void broadcastDemo(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < N) ? data[tid] : 0.0f;
    
    // Broadcast lane 0's value to all lanes in wavefront
    float broadcast = __shfl(val, 0, 64);
    
    if (tid < N) {
        data[tid] = broadcast;
    }
}
```

#### __shfl_xor

```cpp
// XOR-based shuffle (butterfly pattern)
__global__ void butterflyDemo(float* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < N) ? data[tid] : 0.0f;
    
    // Exchange with lane XOR 1 (swap adjacent lanes)
    float neighbor = __shfl_xor(val, 1, 64);
    
    // Average with neighbor
    if (tid < N) {
        data[tid] = (val + neighbor) * 0.5f;
    }
}
```

### Voting Primitives

Vote across all lanes in a wavefront.

#### __ballot

```cpp
__global__ void ballotDemo(const float* data, int* output, int N, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 64;
    int waveId = threadIdx.x / 64;
    
    // Each thread evaluates predicate
    float val = (tid < N) ? data[tid] : 0.0f;
    int predicate = (val > threshold);
    
    // Get bitmask: bit i is set if lane i's predicate is true
    unsigned long long mask = __ballot(predicate);
    
    // Count how many lanes passed
    int count = __popcll(mask);  // Population count
    
    // First lane writes result
    if (lane == 0) {
        output[waveId] = count;
    }
}
```

#### __any and __all

```cpp
__global__ void votingDemo(const float* data, int N, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (tid < N) ? data[tid] : 0.0f;
    int predicate = (val > threshold);
    
    // Check if ANY lane in wavefront satisfies predicate
    int any_true = __any(predicate);
    
    // Check if ALL lanes in wavefront satisfy predicate
    int all_true = __all(predicate);
    
    if (threadIdx.x % 64 == 0) {
        printf("Wave %d: any=%d, all=%d\n", 
               threadIdx.x / 64, any_true, all_true);
    }
}
```

### Complete Wavefront Reduction Example

```cpp
__global__ void wavefrontReduction(const float* input, float* output, int N) {
    __shared__ float wave_results[256 / 64];  // One per wavefront
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 64;
    int waveId = threadIdx.x / 64;
    
    // Load data
    float val = (tid < N) ? input[tid] : 0.0f;
    
    // Step 1: Reduce within wavefront using shuffle
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 64);
    }
    
    // Step 2: First lane writes wavefront result to LDS
    if (lane == 0) {
        wave_results[waveId] = val;
    }
    __syncthreads();
    
    // Step 3: Final reduction across wavefronts (by first wavefront)
    if (waveId == 0) {
        val = (lane < blockDim.x / 64) ? wave_results[lane] : 0.0f;
        
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset, 64);
        }
        
        if (lane == 0) {
            atomicAdd(output, val);
        }
    }
}
```

## Atomic Operations

### Basic Atomics

Atomic operations guarantee thread-safe read-modify-write operations.

```cpp
__global__ void atomicExample(int* counter, float* sum, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        // Atomic integer operations
        atomicAdd(counter, 1);              // counter += 1
        atomicSub(counter, 1);              // counter -= 1
        atomicMax(counter, tid);            // counter = max(counter, tid)
        atomicMin(counter, tid);            // counter = min(counter, tid)
        atomicInc(counter, N);              // counter = (counter + 1) % N
        atomicDec(counter, N);              // counter = (counter - 1) % N
        atomicExch(counter, tid);           // counter = tid (return old value)
        atomicCAS(counter, 0, 1);           // Compare-and-swap
        
        // Atomic float operations
        atomicAdd(sum, 1.0f);               // sum += 1.0f
        atomicMax(sum, float(tid));         // sum = max(sum, tid)
        atomicMin(sum, float(tid));         // sum = min(sum, tid)
    }
}
```

### atomicCAS (Compare-And-Swap)

Most powerful atomic - can implement any atomic operation.

```cpp
// Atomic OR using CAS
__device__ int atomicOr(int* address, int val) {
    int old = *address;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address, assumed, assumed | val);
    } while (assumed != old);
    
    return old;
}

// Atomic float multiply using CAS
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

### Performance Considerations

```cpp
// ✗ BAD: High contention (all threads update same location)
__global__ void highContention(int* counter, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(counter, 1);  // Serializes all threads!
    }
}

// ✓ BETTER: Reduce contention with LDS
__global__ void lowContention(int* counter, int N) {
    __shared__ int lds_counter;
    
    if (threadIdx.x == 0) {
        lds_counter = 0;
    }
    __syncthreads();
    
    // Atomic within block (fewer threads contending)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(&lds_counter, 1);
    }
    __syncthreads();
    
    // Single atomic per block
    if (threadIdx.x == 0) {
        atomicAdd(counter, lds_counter);
    }
}

// ✓ BEST: Use wavefront reduction to minimize atomics
__global__ void minimalAtomics(int* counter, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 64;
    
    int val = (tid < N) ? 1 : 0;
    
    // Reduction within wavefront (no atomics)
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 64);
    }
    
    // Only one atomic per wavefront
    if (lane == 0) {
        atomicAdd(counter, val);
    }
}
```

## Device-Wide Synchronization

### Grid Synchronization

HIP supports cooperative groups for grid-wide synchronization (requires compatible hardware and launch).

```cpp
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void gridSync() {
    cg::grid_group grid = cg::this_grid();
    
    // Do work...
    
    // Synchronize ALL blocks in the grid
    grid.sync();
    
    // Continue with work that depends on all blocks completing...
}

// Launch with cooperative groups
void launchGridSync() {
    int blockSize = 256;
    int numBlocks = 256;
    
    void* args[] = { /* kernel arguments */ };
    
    hipLaunchCooperativeKernel((void*)gridSync, 
                               dim3(numBlocks), 
                               dim3(blockSize), 
                               args, 
                               0, 
                               nullptr);
}
```

### Stream Synchronization

```cpp
// Create streams
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// Launch kernels asynchronously
kernel1<<<grid, block, 0, stream1>>>(args1);
kernel2<<<grid, block, 0, stream2>>>(args2);

// Synchronize specific stream
hipStreamSynchronize(stream1);  // Wait for stream1 only

// Synchronize all streams
hipDeviceSynchronize();  // Wait for everything

// Cleanup
hipStreamDestroy(stream1);
hipStreamDestroy(stream2);
```

## Memory Fences

### Ensuring Memory Ordering

```cpp
// Ensure all writes are visible
__threadfence();         // Visible to all threads in device
__threadfence_block();   // Visible to all threads in block
__threadfence_system();  // Visible to host and device

// Example: Producer-consumer pattern
__shared__ int flag;
__shared__ float data[256];

__device__ void producer() {
    data[threadIdx.x] = compute();
    
    __threadfence_block();  // Ensure data is written
    
    if (threadIdx.x == 0) {
        flag = 1;  // Signal data is ready
    }
}

__device__ void consumer() {
    while (atomicAdd(&flag, 0) == 0);  // Spin until ready
    
    __threadfence_block();  // Ensure we see data
    
    process(data[threadIdx.x]);
}
```

## Common Patterns

### Double Buffering with Barriers

```cpp
__global__ void doubleBuffer(float* data, int iters) {
    __shared__ float buffer[2][256];
    
    int current = 0;
    
    // Initial load
    buffer[current][threadIdx.x] = data[threadIdx.x];
    __syncthreads();
    
    for (int iter = 0; iter < iters; ++iter) {
        int next = 1 - current;
        
        // Compute next values while current buffer is stable
        buffer[next][threadIdx.x] = stencil(buffer[current], threadIdx.x);
        __syncthreads();
        
        current = next;
    }
    
    data[threadIdx.x] = buffer[current][threadIdx.x];
}
```

### Wavefront-Aggregated Atomics

```cpp
__global__ void aggregatedAtomics(const int* keys, const float* values, 
                                  float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 64;
    
    if (tid >= N) return;
    
    int key = keys[tid];
    float val = values[tid];
    
    // Find which lanes have the same key
    unsigned long long mask = __ballot(1);  // Active lanes
    
    // Aggregate values with same key within wavefront
    for (int i = 0; i < 64; ++i) {
        int broadcast_key = __shfl(key, i, 64);
        if (key == broadcast_key) {
            float broadcast_val = __shfl(val, i, 64);
            val += broadcast_val;
        }
    }
    
    // Only first lane with each key does atomic
    unsigned long long same_key_mask = __ballot(key == __shfl(key, lane, 64));
    if (__popcll(same_key_mask & ((1ULL << lane) - 1)) == 0) {
        atomicAdd(&output[key], val);
    }
}
```

## Debugging Synchronization Issues

### Race Condition Detection

```cpp
// Enable race detection with sanitizer
// Compile: hipcc -g -O0 -fsanitize=thread ...

__global__ void raceExample() {
    __shared__ int counter;
    
    // Race condition: multiple threads writing
    counter = threadIdx.x;  // ✗ RACE!
    
    // Fix 1: Only one thread writes
    if (threadIdx.x == 0) {
        counter = 0;  // ✓ Safe
    }
    
    // Fix 2: Use atomic
    atomicExch(&counter, threadIdx.x);  // ✓ Safe
}
```

### Deadlock Detection

```cpp
// Common deadlock: conditional barrier
__global__ void deadlockExample(int* data, int N) {
    __shared__ float lds[256];
    
    if (threadIdx.x < N) {  // ✗ DEADLOCK if N < 256!
        lds[threadIdx.x] = data[threadIdx.x];
        __syncthreads();  // Some threads never reach here
    }
}
```

## Best Practices

1. **Minimize synchronization overhead**
   - Use wavefront primitives instead of LDS + barriers when possible
   - Aggregate atomics to reduce contention
   - Avoid synchronization in hot loops

2. **Ensure all threads reach barriers**
   - Never put `__syncthreads()` in conditional code
   - Validate with compute-sanitizer

3. **Use appropriate synchronization scope**
   - Wavefront primitives: Within 64 threads (no barrier needed)
   - `__syncthreads()`: Within workgroup only
   - Atomics: Device-wide
   - Host synchronization: For host-device coordination

4. **Optimize atomic operations**
   - Reduce to wavefront/block level first
   - Use CAS for complex operations
   - Consider alternative algorithms to avoid atomics

## References

### Related Documentation
- [HIP GPU Programming Fundamentals](./hip-gpu-programming-fundamentals.md)
- [HIP Memory Management](./hip-memory-management.md)
- [HIP Performance Optimization](../../layer-3-libraries/algorithms/hip-performance-optimization.md)

### Official Resources
- **HIP Cooperative Groups:** https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- **Wavefront Primitives:** https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html#warp-cross-lane-functions

### GPU Programming 101
- **Repository:** https://github.com/AIComputing101/gpu-programming-101
- **Module 3:** Thread Synchronization and Coordination
- **License:** MIT

### Example Code
- [Wavefront Primitives](../../../examples/amd/hip/wavefront_primitives.hip)
- [Parallel Reduction](../../../examples/amd/hip/parallel_reduction.cpp)

