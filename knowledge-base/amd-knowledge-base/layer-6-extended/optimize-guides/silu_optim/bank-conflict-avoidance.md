Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "shared-memory", "lds", "hip", "silu", "bank-conflict"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# Bank Conflict Avoidance in Shared Memory

## Overview

Bank conflicts occur when multiple threads in a wavefront access different addresses within the same memory bank of Local Data Share (LDS, AMD's equivalent of CUDA shared memory), forcing sequential access and reducing throughput. On AMD CDNA architectures, LDS is organized into 32 banks with 4-byte width, and understanding bank conflict patterns is essential for kernels that use shared memory as a staging area. For SiLU and element-wise operations, shared memory is typically unnecessary since each thread processes independent elements, but bank conflict knowledge becomes critical when implementing optimizations like shared memory buffering for strided access patterns or data reorganization.

A bank conflict occurs when two or more threads in a wavefront access addresses that map to the same bank but are not the same address (broadcast reads from the same address are conflict-free). AMD GPUs serialize conflicting accesses, so a 2-way conflict doubles latency, a 4-way conflict quadruples it, and worst-case 64-way conflicts can increase latency by 64x. For typical SiLU implementations that process contiguous data, bank conflicts are rare, but they become significant when using shared memory for transpose operations, data reordering, or as a scratchpad for complex activation functions.

Understanding and avoiding bank conflicts involves careful indexing patterns, padding strategies, and access pattern design. While simple element-wise SiLU doesn't require shared memory, advanced optimizations for batched operations, fused kernels, or specialized data layouts may benefit from LDS usage, making bank conflict avoidance knowledge essential for performance-critical scenarios.

## Technical Details

AMD LDS bank organization (CDNA2 architecture):
- **Number of Banks**: 32 banks
- **Bank Width**: 4 bytes (32 bits)
- **Total Banks**: 64 KB LDS / 32 banks = 2 KB per bank
- **Wavefront Size**: 64 threads (2 threads per bank on average)

Address-to-bank mapping:
```
Bank Number = (Address / 4) % 32
```

For a 4-byte (fp32) access at address `addr`:
- Thread accessing `addr` uses bank `(addr/4) % 32`

Common conflict patterns:

1. **Power-of-2 Stride Conflict** (worst case):
   ```cpp
   __shared__ float data[32][32];
   float val = data[threadIdx.x][0];  // All threads access different banks (good)
   float val = data[0][threadIdx.x];  // Threads 0,32 conflict (2-way conflict)
   ```

2. **Column-Major Access of Row-Major Array**:
   ```cpp
   __shared__ float data[32][32];
   float val = data[threadIdx.x][col];  // If col is constant, 32-way conflict!
   ```

3. **Diagonal Access Pattern**:
   ```cpp
   __shared__ float data[64][64];
   float val = data[threadIdx.x][(threadIdx.x + offset) % 64];  // Depends on offset
   ```

Solutions to avoid conflicts:

1. **Padding**: Add extra column to break power-of-2 alignment
   ```cpp
   __shared__ float data[32][33];  // Extra column prevents conflicts
   ```

2. **Swizzling**: Permute indices to distribute accesses
   ```cpp
   int bank_id = (threadIdx.x) ^ (threadIdx.x / 32);
   ```

3. **Access Pattern Optimization**: Ensure consecutive threads access different banks
   ```cpp
   float val = data[threadIdx.x][col];  // Good if threadIdx.x varies
   ```

For SiLU-specific usage:
- Basic SiLU doesn't need LDS (operates directly on global memory)
- Batched SiLU with transpose might use LDS for reordering
- Fused operations (SiLU + LayerNorm) could benefit from LDS staging

## Code Examples

### Example 1: Bank Conflict in Transpose Operation

```cpp
#include <hip/hip_runtime.h>

// INEFFICIENT: Bank conflicts in transpose
__global__ void transpose_with_conflicts(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    __shared__ float tile[32][32];  // 32x32 tile

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Load from global memory (coalesced)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    // Transpose indices
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Write to global memory (conflict on read from shared memory!)
    // All threads with same threadIdx.x access same column
    // -> 32-way bank conflict
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// EFFICIENT: Padded to avoid bank conflicts
__global__ void transpose_no_conflicts(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height
) {
    // Add one extra column to break power-of-2 alignment
    __shared__ float tile[32][33];  // Padded!

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Load (coalesced global, no LDS conflict)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Write (no bank conflict due to padding)
    // Consecutive threads access consecutive addresses in different banks
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Example 2: Using LDS for Batched SiLU

```cpp
#include <hip/hip_runtime.h>

// Batched SiLU using shared memory for data reuse
__launch_bounds__(256)
__global__ void batched_silu_with_lds(
    const float* __restrict__ input,   // Shape: [batch, channels]
    float* __restrict__ output,
    int batch_size,
    int channels
) {
    // Use shared memory to cache a tile of data
    __shared__ float tile[16][257];  // Padded to avoid conflicts (256 + 1)

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int batch_idx = blockIdx.y * 16 + tid_y;
    int channel_idx = blockIdx.x * 256 + tid_x;

    // Load data into shared memory (coalesced global read)
    if (batch_idx < batch_size && channel_idx < channels) {
        tile[tid_y][tid_x] = input[batch_idx * channels + channel_idx];
    }
    __syncthreads();

    // Compute SiLU (bank-conflict-free read from LDS)
    if (batch_idx < batch_size && channel_idx < channels) {
        float x = tile[tid_y][tid_x];  // No conflict due to padding
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float result = x * sigmoid_x;

        // Write back
        output[batch_idx * channels + channel_idx] = result;
    }
}
```

### Example 3: Swizzling Pattern for Conflict Avoidance

```cpp
#include <hip/hip_runtime.h>

// Advanced: Swizzled access pattern
__device__ __forceinline__ int swizzle_index(int row, int col, int stride) {
    // XOR-based swizzling to distribute bank accesses
    int swizzled_col = col ^ (row & 0x1F);  // XOR with lower 5 bits of row
    return row * stride + swizzled_col;
}

__global__ void silu_with_swizzled_lds(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    __shared__ float buffer[64][64];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int elements_per_block = 64 * 64;

    // Load with swizzling
    for (int i = tid; i < elements_per_block; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int global_idx = bid * elements_per_block + i;

        if (global_idx < num_elements) {
            // Swizzled write to avoid conflicts
            int swizzled = swizzle_index(row, col, 64);
            buffer[swizzled / 64][swizzled % 64] = input[global_idx];
        }
    }
    __syncthreads();

    // Compute SiLU
    for (int i = tid; i < elements_per_block; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int global_idx = bid * elements_per_block + i;

        if (global_idx < num_elements) {
            int swizzled = swizzle_index(row, col, 64);
            float x = buffer[swizzled / 64][swizzled % 64];
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            buffer[swizzled / 64][swizzled % 64] = x * sigmoid_x;
        }
    }
    __syncthreads();

    // Write back
    for (int i = tid; i < elements_per_block; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        int global_idx = bid * elements_per_block + i;

        if (global_idx < num_elements) {
            int swizzled = swizzle_index(row, col, 64);
            output[global_idx] = buffer[swizzled / 64][swizzled % 64];
        }
    }
}
```

### Example 4: Measuring Bank Conflicts with Profiling

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

// Test kernel with intentional bank conflicts
__global__ void lds_bank_conflict_test(
    const float* __restrict__ input,
    float* __restrict__ output,
    int use_padding
) {
    __shared__ float buffer_conflict[256][256];    // Conflicts
    __shared__ float buffer_padded[256][257];      // No conflicts

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (use_padding == 0) {
        // Access pattern with conflicts
        buffer_conflict[tid][0] = input[idx];
        __syncthreads();
        float val = buffer_conflict[0][tid];  // 32-way conflict!
        output[idx] = val;
    } else {
        // Access pattern without conflicts
        buffer_padded[tid][0] = input[idx];
        __syncthreads();
        float val = buffer_padded[0][tid];  // No conflict (padding breaks alignment)
        output[idx] = val;
    }
}

void benchmark_bank_conflicts() {
    int num_elements = 256 * 256;
    float *d_input, *d_output;
    hipMalloc(&d_input, num_elements * sizeof(float));
    hipMalloc(&d_output, num_elements * sizeof(float));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Test with conflicts
    hipEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        hipLaunchKernelGGL(lds_bank_conflict_test, dim3(256), dim3(256),
                          0, 0, d_input, d_output, 0);  // use_padding=0
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float time_conflict;
    hipEventElapsedTime(&time_conflict, start, stop);

    // Test without conflicts
    hipEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        hipLaunchKernelGGL(lds_bank_conflict_test, dim3(256), dim3(256),
                          0, 0, d_input, d_output, 1);  // use_padding=1
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float time_padded;
    hipEventElapsedTime(&time_padded, start, stop);

    std::cout << "With bank conflicts: " << time_conflict / 1000.0f << " ms\n";
    std::cout << "Without conflicts (padded): " << time_padded / 1000.0f << " ms\n";
    std::cout << "Speedup from padding: " << (time_conflict / time_padded) << "x\n";

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_input);
    hipFree(d_output);
}
```

## Best Practices

**Pad Shared Memory Arrays**: Add one extra element to the inner dimension when using 2D arrays with power-of-2 sizes. Change `[32][32]` to `[32][33]` to avoid bank conflicts with minimal memory overhead (3% increase for 32-wide arrays).

**Access Consecutive Elements**: Ensure consecutive threads access consecutive shared memory addresses. Pattern `data[threadIdx.x][col]` with varying threadIdx.x is conflict-free; `data[row][threadIdx.x]` may have conflicts if row is constant across threads.

**Avoid Constant Index Across Wavefront**: If all threads in a wavefront access the same column/row with different rows/columns, conflicts occur. Restructure access patterns or use padding/swizzling.

**Profile with rocprof**: Use `rocprof --stats` and check for `LDSBankConflict` metrics. High conflict rates (>10% of LDS accesses) indicate optimization opportunities.

**Consider Shared Memory Alternatives**: For simple element-wise operations like SiLU, avoiding shared memory entirely (direct global memory access) is often faster due to excellent coalescing and absence of synchronization overhead.

**Use Swizzling for Complex Patterns**: When padding isn't sufficient, implement XOR-based index swizzling to distribute accesses across banks more uniformly.

**Common Pitfalls**:
- Not padding shared memory arrays with power-of-2 dimensions
- Accessing columns with constant index across threads
- Over-using shared memory when global memory would suffice
- Ignoring profiling data showing bank conflicts
- Assuming padding always helps (verify with benchmarks)

## Performance Impact

Bank conflict impact on shared memory operations (MI250X):

**No Conflicts** (optimal access pattern):
- LDS Latency: 4-8 cycles
- Throughput: 64 accesses per wavefront per cycle

**2-Way Conflict**:
- Latency: 8-16 cycles (2x slowdown)
- Throughput: 32 accesses per cycle

**4-Way Conflict**:
- Latency: 16-32 cycles (4x slowdown)
- Throughput: 16 accesses per cycle

**32-Way Conflict** (worst case for 32 banks):
- Latency: 128-256 cycles (32x slowdown)
- Throughput: 2 accesses per cycle

For transpose operations (common LDS usage):
- With conflicts: 150-200 μs for 4K×4K fp32 transpose
- Without conflicts (padding): 40-60 μs (3-4x speedup)

For typical SiLU kernels that don't use LDS, bank conflicts are not applicable, but understanding them is valuable for advanced fusion scenarios.

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- Shared Memory Programming: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- Related Topics: Local Data Share (LDS), memory banking, shared memory padding, transpose optimization
