---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "memory", "coalescing", "hip", "silu", "bandwidth"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# Memory Coalescing in HIP

## Overview

Memory coalescing is the single most important optimization for memory-bound kernels like element-wise operations, enabling up to 10-20x performance improvement over uncoalesced access patterns. When consecutive threads in a wavefront access consecutive memory locations, AMD GPUs can combine multiple memory requests into a single, wide transaction, dramatically reducing memory latency and maximizing bandwidth utilization. For SiLU and similar activation functions, achieving coalesced access is critical because the operation is entirely memory-bound: the arithmetic workload (sigmoid and multiply) is trivial compared to the cost of fetching data from global memory.

On AMD GPUs with CDNA architecture (MI200 series and later), a wavefront consists of 64 threads that execute in lockstep. When these threads access memory, the hardware analyzes the access pattern and attempts to coalesce requests into 128-byte (or larger) transactions. Perfect coalescing occurs when threads 0-63 access addresses `base`, `base+4`, `base+8`, ..., `base+252` for 4-byte data types like fp32 or when threads access packed bf16 data at properly aligned boundaries.

Coalesced access can achieve 70-90% of theoretical peak bandwidth (1.6 TB/s on MI250X), while uncoalesced or random access patterns may achieve only 5-15% of peak bandwidth. For a SiLU kernel processing 1M elements, this translates to a difference between 50 μs (coalesced) and 600 μs (uncoalesced) execution time.

## Technical Details

Memory coalescing depends on several factors:

1. **Access Pattern Requirements**:
   - **Consecutive Threads → Consecutive Addresses**: Thread ID N should access address `base + N * sizeof(type)`
   - **Alignment**: Starting address must be aligned to transaction size (128 bytes optimal)
   - **Contiguity**: All accesses within a wavefront should fall within a single cache line or aligned region
   - **Uniformity**: All threads in a wavefront should participate (no divergence in memory access)

2. **Transaction Sizes on AMD GPUs**:
   - **L1 Cache Line**: 64 bytes (16 × fp32 or 32 × bf16)
   - **L2 Cache Line**: 128 bytes (32 × fp32 or 64 × bf16)
   - **Optimal Transaction**: 128-256 bytes (full wavefront, 64 threads × 4 bytes = 256 bytes)

3. **Coalescing Efficiency Metrics**:
   - **Perfect Coalescing**: 1 memory transaction per wavefront = 100% efficiency
   - **Partial Coalescing**: 2-4 transactions per wavefront = 25-50% efficiency
   - **No Coalescing**: 64 transactions per wavefront = 1.5% efficiency

4. **Impact of Data Types**:
   - **FP32**: 4 bytes, 64 threads access 256 bytes (2 cache lines)
   - **BF16**: 2 bytes, 64 threads access 128 bytes (1-2 cache lines)
   - **BF16 Vectorized (bfloat162)**: Each thread loads 4 bytes, 64 threads = 256 bytes (optimal)

For SiLU kernels, the typical pattern is:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = input[idx];  // Coalesced if input is contiguous
```

This guarantees coalescing because:
- Thread 0 accesses `input[0]`, thread 1 accesses `input[1]`, etc.
- Consecutive threads access consecutive memory locations
- Memory controller combines these into wide transactions

Anti-patterns that break coalescing:
- **Stride Access**: `input[idx * stride]` where stride > 1
- **Random Access**: `input[indices[idx]]` with random indices
- **Reverse Access**: `input[N - idx - 1]` (still coalesced but inefficient)
- **Structure of Arrays with Gaps**: Accessing non-adjacent fields

## Code Examples

### Example 1: Perfect Coalescing for SiLU

```cpp
#include <hip/hip_runtime.h>

// Perfectly coalesced SiLU kernel
__launch_bounds__(256, 4)
__global__ void silu_coalesced(
    const float* __restrict__ input,   // __restrict__ hint for aliasing
    float* __restrict__ output,
    int num_elements
) {
    // Standard 1D indexing pattern
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop maintains coalescing across iterations
    for (int i = idx; i < num_elements; i += stride) {
        // Coalesced read: consecutive threads read consecutive addresses
        float x = input[i];

        // Compute SiLU
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float result = x * sigmoid_x;

        // Coalesced write: consecutive threads write consecutive addresses
        output[i] = result;
    }
}

// Verify coalescing with proper launch configuration
void launch_silu_coalesced(
    const float* d_input,
    float* d_output,
    int num_elements
) {
    const int threads = 256;  // Multiple of wavefront size (64)
    const int blocks = (num_elements + threads - 1) / threads;

    hipLaunchKernelGGL(
        silu_coalesced,
        dim3(blocks),
        dim3(threads),
        0, 0,
        d_input, d_output, num_elements
    );
}
```

### Example 2: BF16 Vectorized Coalescing

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Optimal coalescing with bf16 vectorization
__launch_bounds__(256, 4)
__global__ void silu_bf16_vectorized_coalesced(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat162* __restrict__ output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_pairs; i += stride) {
        // Coalesced 4-byte load (2 × bf16)
        // Thread 0: address 0, Thread 1: address 4, Thread 2: address 8, ...
        // Perfect alignment and contiguity
        __hip_bfloat162 x = input[i];

        // Compute sigmoid (simplified)
        float x1 = __bfloat162float(x.x);
        float x2 = __bfloat162float(x.y);
        float sig1 = 1.0f / (1.0f + expf(-x1));
        float sig2 = 1.0f / (1.0f + expf(-x2));
        __hip_bfloat162 sigmoid_x = __floats2bfloat162_rn(sig1, sig2);

        __hip_bfloat162 result = __hmul2(x, sigmoid_x);

        // Coalesced 4-byte store
        output[i] = result;
    }
}

// Memory allocation ensuring alignment
void setup_coalesced_buffers(
    __hip_bfloat16** d_input,
    __hip_bfloat16** d_output,
    int num_elements
) {
    size_t bytes = num_elements * sizeof(__hip_bfloat16);

    // hipMalloc provides 256-byte alignment automatically
    hipMalloc(d_input, bytes);
    hipMalloc(d_output, bytes);

    // Verify alignment (optional, for debugging)
    size_t alignment = 256;
    assert(((uintptr_t)*d_input & (alignment - 1)) == 0);
    assert(((uintptr_t)*d_output & (alignment - 1)) == 0);
}
```

### Example 3: Avoiding Strided Access Anti-Pattern

```cpp
#include <hip/hip_runtime.h>

// ANTI-PATTERN: Strided access (poor coalescing)
__global__ void silu_strided_bad(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    int stride  // stride > 1 breaks coalescing
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // BAD: Thread 0 reads input[0], Thread 1 reads input[stride],
        // Thread 2 reads input[2*stride], ...
        // Memory controller cannot coalesce these into single transaction
        float x = input[idx * stride];

        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float result = x * sigmoid_x;

        output[idx * stride] = result;
    }
}

// SOLUTION: Restructure data layout or access pattern
__global__ void silu_transposed_fixed(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // GOOD: Consecutive access even if logical layout is different
        // Pre-transpose data on host if necessary
        float x = input[idx];

        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float result = x * sigmoid_x;

        output[idx] = result;
    }
}

// Alternative: Use shared memory to reorder access
__global__ void silu_shared_memory_coalesce(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    int stride
) {
    __shared__ float shared_input[256];
    __shared__ float shared_output[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < num_elements) {
        // Coalesced load into shared memory
        shared_input[tid] = input[idx * stride];
    }
    __syncthreads();

    // Compute on shared memory
    if (idx < num_elements) {
        float x = shared_input[tid];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        shared_output[tid] = x * sigmoid_x;
    }
    __syncthreads();

    // Coalesced write back
    if (idx < num_elements) {
        output[idx * stride] = shared_output[tid];
    }
}
```

### Example 4: Verifying Coalescing with Profiling

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

// Helper to measure effective bandwidth
struct BandwidthResult {
    float elapsed_ms;
    float effective_bandwidth_gb_s;
    float efficiency_percent;
};

BandwidthResult measure_bandwidth(
    const float* d_input,
    float* d_output,
    int num_elements,
    int num_iterations = 100
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warmup
    hipLaunchKernelGGL(silu_coalesced, dim3(blocks), dim3(threads),
                      0, 0, d_input, d_output, num_elements);
    hipDeviceSynchronize();

    // Timed measurement
    hipEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        hipLaunchKernelGGL(silu_coalesced, dim3(blocks), dim3(threads),
                          0, 0, d_input, d_output, num_elements);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    elapsed_ms /= num_iterations;

    // Calculate metrics
    size_t bytes = num_elements * sizeof(float) * 2;  // Read + write
    float bandwidth_gb_s = (bytes / 1e9) / (elapsed_ms / 1000.0f);

    // Get peak bandwidth for comparison (MI250X: 1600 GB/s)
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    float peak_bandwidth = prop.memoryClockRate * 1000.0f *
                          (prop.memoryBusWidth / 8) * 2 / 1e9;

    float efficiency = (bandwidth_gb_s / peak_bandwidth) * 100.0f;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return {elapsed_ms, bandwidth_gb_s, efficiency};
}

// Usage example
void verify_coalescing() {
    int num_elements = 16 * 1024 * 1024;  // 16M elements
    float *d_input, *d_output;

    hipMalloc(&d_input, num_elements * sizeof(float));
    hipMalloc(&d_output, num_elements * sizeof(float));

    auto result = measure_bandwidth(d_input, d_output, num_elements);

    std::cout << "Bandwidth: " << result.effective_bandwidth_gb_s << " GB/s\n";
    std::cout << "Efficiency: " << result.efficiency_percent << "%\n";

    // Good coalescing: >70% efficiency
    if (result.efficiency_percent > 70.0f) {
        std::cout << "GOOD: Memory access is well coalesced\n";
    } else {
        std::cout << "WARNING: Poor memory coalescing detected\n";
    }

    hipFree(d_input);
    hipFree(d_output);
}
```

## Best Practices

**Use Standard 1D Indexing**: Always use `idx = blockIdx.x * blockDim.x + threadIdx.x` for element-wise operations on 1D arrays. This pattern guarantees perfect coalescing when accessing `array[idx]`.

**Choose Block Size as Multiple of Wavefront Size**: Use block sizes of 64, 128, 256, or 512 threads (multiples of 64) to ensure full wavefronts without partial groups that might reduce coalescing efficiency.

**Align Data Structures**: Ensure arrays are allocated with proper alignment using `hipMalloc` (provides 256-byte alignment) or aligned malloc on the host. Misaligned data can reduce coalescing efficiency by 20-50%.

**Avoid Strided Access**: If your algorithm requires strided access, consider restructuring the data layout (transpose on host) or using shared memory as an intermediate buffer to enable coalesced global memory access.

**Use Grid-Stride Loops**: The pattern `for (int i = idx; i < N; i += stride)` maintains coalescing across all iterations because the stride equals the total number of threads, keeping consecutive threads accessing consecutive locations.

**Profile with ROCProfiler**: Use `rocprof --stats` to measure memory throughput and efficiency. Look for metrics like `MemUnitBusy` and compare achieved bandwidth to theoretical peak. Target >70% efficiency for memory-bound kernels.

**Leverage __restrict__ Keyword**: Use `__restrict__` on pointer parameters to inform the compiler that pointers don't alias, enabling additional optimization for coalesced access.

**Common Pitfalls**:
- Using block sizes not divisible by 64 (wavefront size)
- Accessing array-of-structures (AoS) instead of structure-of-arrays (SoA)
- Random or indirect indexing: `array[indices[idx]]`
- Reverse iteration that might confuse compiler optimizations
- Forgetting to align custom-allocated buffers

## Performance Impact

Memory bandwidth comparison for SiLU on MI250X (1M fp32 elements):

**Perfect Coalescing** (consecutive access):
- Bandwidth: 1200-1400 GB/s (75-87% of peak 1600 GB/s)
- Latency: 50-60 μs
- Transactions: ~4K memory requests

**Strided Access** (stride = 2):
- Bandwidth: 600-800 GB/s (38-50% of peak)
- Latency: 100-150 μs
- Transactions: ~8K memory requests

**Random Access** (scattered indices):
- Bandwidth: 80-240 GB/s (5-15% of peak)
- Latency: 600-1000 μs
- Transactions: ~64K memory requests (one per thread)

For bf16 vectorized access with `__hip_bfloat162`:
- Bandwidth: 1300-1500 GB/s (81-93% of peak)
- Even better efficiency due to larger transactions per thread

**Key Takeaway**: Proper coalescing can provide 10-20x speedup for memory-bound kernels.

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- HIP Programming Model: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- Hardware Implementation: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- Related Topics: Memory bandwidth optimization, wavefront execution, cache hierarchy, grid-stride loops
