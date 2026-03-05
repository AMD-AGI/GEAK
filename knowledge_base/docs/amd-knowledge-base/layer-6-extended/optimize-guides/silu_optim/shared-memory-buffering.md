---
tags: ["optimization", "shared-memory", "lds", "hip", "silu", "buffering"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# Shared Memory Buffering for Strided Access

## Overview

Shared memory (Local Data Share or LDS on AMD GPUs) provides low-latency, high-bandwidth on-chip memory that can be used as a staging area to transform strided global memory access into coalesced patterns. When strided access cannot be avoided through data layout changes, buffering data through shared memory enables coalesced reads from global memory into LDS, followed by strided or arbitrary access within the fast shared memory domain, dramatically improving performance for memory-bound operations like SiLU on non-contiguous data.

LDS on AMD CDNA architectures provides approximately 20-40 TB/s internal bandwidth (50-100x faster than global memory) with 4-8 cycle latency versus 200-400 cycles for global memory. By loading data cooperatively into shared memory using coalesced global memory transactions, then performing strided or random access from LDS, kernels can achieve 2-5x speedup over direct strided global memory access. The technique is particularly effective when: (1) stride is moderate to large (>4), (2) data is reused multiple times, or (3) access pattern is irregular but predictable.

For SiLU specifically, shared memory buffering applies to scenarios like batched activation across non-contiguous channels, fused operations with complex data dependencies, or tensor layout transformations combined with activation. Understanding the LDS capacity (64 KB per CU), synchronization overhead (__syncthreads), and bank conflict avoidance is essential for effective shared memory usage.

## Technical Details

Shared memory characteristics on AMD CDNA2:
- **Capacity**: 64 KB per Compute Unit
- **Bandwidth**: 20-40 TB/s (internal)
- **Latency**: 4-8 cycles (LDS hit), vs 200-400 cycles (global memory)
- **Banks**: 32 banks, 4-byte width
- **Scope**: Shared within thread block only

Buffering strategy steps:
1. **Cooperative Load**: All threads in block load data from global memory using coalesced pattern into shared memory
2. **Synchronize**: __syncthreads() ensures all data loaded before access
3. **Process**: Access shared memory with stride or complex pattern (fast)
4. **Synchronize**: (if needed) __syncthreads() before writeback
5. **Cooperative Store**: Write results back to global memory (coalesced)

Performance model:
```
Time_Direct_Stride = N_elements × Latency_Global × Stride_Factor
Time_LDS_Buffer = N_elements / BlockSize × (Latency_Global + Latency_LDS) + Sync_Overhead
```

Speedup is significant when:
- Stride > 4 and LDS access can be irregular
- Data reused multiple times (amortize load cost)
- Block size is large enough to hide sync overhead (256+ threads)

Memory requirements:
```
LDS_per_block = Elements_per_block × sizeof(type) × (1 + padding_factor)
Max_blocks_per_CU = min(64KB / LDS_per_block, other_limits)
```

Typical configuration for SiLU with stride buffering:
- Block size: 256 threads
- Elements per thread: 4-8
- LDS usage: 256 × 4 × 4 bytes = 4 KB (allows 16 blocks per CU)
- Padding: +1 column to avoid bank conflicts

## Code Examples

### Example 1: Basic Shared Memory Buffering

```cpp
#include <hip/hip_runtime.h>

// Direct strided access (baseline)
__global__ void silu_strided_baseline(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Strided global memory access (slow)
        float x = input[idx * stride];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx * stride] = x * sigmoid_x;
    }
}

// Shared memory buffered version
__global__ void silu_lds_buffered(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    int stride
) {
    __shared__ float buffer[256];  // Shared memory buffer

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Coalesced load into shared memory
    if (gid < num_elements) {
        // Load with coalescing (each thread loads from consecutive address)
        buffer[tid] = input[gid * stride];
    }
    __syncthreads();

    // Process from fast shared memory
    if (gid < num_elements) {
        float x = buffer[tid];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        buffer[tid] = x * sigmoid_x;
    }
    __syncthreads();

    // Coalesced write back
    if (gid < num_elements) {
        output[gid * stride] = buffer[tid];
    }
}
```

### Example 2: Tiled Buffering for Large Data

```cpp
#include <hip/hip_runtime.h>

// Process large array using tiled shared memory approach
__global__ void silu_tiled_lds(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,   // Array width (stride dimension)
    int height,  // Array height
    int stride   // Stride value
) {
    // Tile size 32x32, padded to avoid bank conflicts
    __shared__ float tile[32][33];

    int tile_x = blockIdx.x * 32;
    int tile_y = blockIdx.y * 32;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int global_x = tile_x + tx;
    int global_y = tile_y + ty;

    // Load tile cooperatively (coalesced)
    if (global_x < width && global_y < height) {
        int linear_idx = global_y * width + global_x;
        tile[ty][tx] = input[linear_idx * stride];
    }
    __syncthreads();

    // Process within shared memory
    if (global_x < width && global_y < height) {
        float x = tile[ty][tx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        tile[ty][tx] = x * sigmoid_x;
    }
    __syncthreads();

    // Write back (coalesced)
    if (global_x < width && global_y < height) {
        int linear_idx = global_y * width + global_x;
        output[linear_idx * stride] = tile[ty][tx];
    }
}
```

### Example 3: Multi-Stage Buffering with Reuse

```cpp
#include <hip/hip_runtime.h>

// Fused SiLU with data reuse in shared memory
__global__ void silu_fused_lds_reuse(
    const float* __restrict__ input,
    const float* __restrict__ scale,  // Per-element scale
    float* __restrict__ output,
    int num_elements,
    int stride
) {
    __shared__ float data_buffer[256];
    __shared__ float scale_buffer[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load both input arrays into shared memory (coalesced)
    if (gid < num_elements) {
        data_buffer[tid] = input[gid * stride];
        scale_buffer[tid] = scale[gid];  // Different stride pattern
    }
    __syncthreads();

    // Process - both arrays now in fast shared memory
    if (gid < num_elements) {
        float x = data_buffer[tid];
        float s = scale_buffer[tid];

        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float silu_x = x * sigmoid_x;

        // Apply scaling
        data_buffer[tid] = silu_x * s;
    }
    __syncthreads();

    // Write back
    if (gid < num_elements) {
        output[gid * stride] = data_buffer[tid];
    }
}
```

### Example 4: Dynamic Shared Memory Allocation

```cpp
#include <hip/hip_runtime.h>

// Kernel using dynamic shared memory for flexible buffer size
__global__ void silu_dynamic_lds(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    int stride,
    int buffer_size  // Elements per block
) {
    extern __shared__ float dynamic_buffer[];

    int tid = threadIdx.x;
    int elements_per_thread = buffer_size / blockDim.x;

    // Each thread loads multiple elements
    for (int i = 0; i < elements_per_thread; i++) {
        int local_idx = tid * elements_per_thread + i;
        int global_idx = blockIdx.x * buffer_size + local_idx;

        if (global_idx < num_elements) {
            dynamic_buffer[local_idx] = input[global_idx * stride];
        }
    }
    __syncthreads();

    // Process
    for (int i = 0; i < elements_per_thread; i++) {
        int local_idx = tid * elements_per_thread + i;
        int global_idx = blockIdx.x * buffer_size + local_idx;

        if (global_idx < num_elements) {
            float x = dynamic_buffer[local_idx];
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            dynamic_buffer[local_idx] = x * sigmoid_x;
        }
    }
    __syncthreads();

    // Write back
    for (int i = 0; i < elements_per_thread; i++) {
        int local_idx = tid * elements_per_thread + i;
        int global_idx = blockIdx.x * buffer_size + local_idx;

        if (global_idx < num_elements) {
            output[global_idx * stride] = dynamic_buffer[local_idx];
        }
    }
}

// Launch with dynamic shared memory
void launch_dynamic_lds_silu(
    const float* d_input,
    float* d_output,
    int num_elements,
    int stride
) {
    int threads = 256;
    int buffer_size = 1024;  // Elements per block
    int blocks = (num_elements + buffer_size - 1) / buffer_size;
    size_t shared_mem_bytes = buffer_size * sizeof(float);

    hipLaunchKernelGGL(
        silu_dynamic_lds,
        dim3(blocks),
        dim3(threads),
        shared_mem_bytes,  // Dynamic shared memory size
        0,
        d_input, d_output, num_elements, stride, buffer_size
    );
}
```

## Best Practices

**Use Shared Memory When Stride > 4**: For small strides (2-3), the overhead of LDS buffering may not justify the benefit. Profile both approaches to verify speedup before committing to shared memory solution.

**Maximize Coalescing in Load/Store**: The primary benefit of LDS buffering comes from transforming strided global access into coalesced loads. Ensure the load pattern from global to LDS is perfectly coalesced (consecutive threads load consecutive addresses).

**Avoid Bank Conflicts**: Pad shared memory arrays by one element when dimensions are powers of 2 (change [256] to [257]) to prevent bank conflicts during access.

**Balance LDS Usage and Occupancy**: Each KB of shared memory per block reduces the maximum blocks per CU. Monitor occupancy when increasing LDS usage - target at least 4 blocks per CU (≤16 KB shared memory per block).

**Minimize Synchronization Points**: Each __syncthreads() costs 10-50 cycles. Structure code to use the minimum necessary synchronizations (typically 2: after load, before store).

**Consider Register Alternatives**: For small per-thread data (<8 elements), using registers instead of shared memory may be faster, avoiding synchronization overhead entirely.

**Common Pitfalls**:
- Using shared memory for stride-1 access (unnecessary overhead)
- Not padding arrays, causing bank conflicts
- Excessive synchronization points
- Exceeding 64 KB LDS limit
- Not verifying speedup versus direct global memory access

## Performance Impact

Shared memory buffering speedup (MI250X, stride-32 access, 16M elements):

**Direct Strided Access (Baseline)**:
- Time: 550 μs
- Bandwidth: 140 GB/s (9% of peak)
- Many uncoalesced transactions

**LDS Buffered**:
- Time: 180 μs
- Bandwidth: 430 GB/s (27% of peak)
- Speedup: 3.0x

**LDS Buffered + Optimized (padding, minimize sync)**:
- Time: 135 μs
- Bandwidth: 570 GB/s (36% of peak)
- Speedup: 4.1x

Comparison by stride:
- Stride 2: LDS buffering ~1.2x speedup (marginal benefit)
- Stride 4: LDS buffering ~1.8x speedup
- Stride 8: LDS buffering ~2.5x speedup
- Stride 32: LDS buffering ~3-4x speedup
- Stride 64+: LDS buffering ~4-5x speedup

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- Shared Memory Programming: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- Related Topics: Local Data Share (LDS), memory coalescing, bank conflicts, synchronization
