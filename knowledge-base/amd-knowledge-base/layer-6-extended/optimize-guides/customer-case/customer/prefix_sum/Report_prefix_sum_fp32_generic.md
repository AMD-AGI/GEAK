Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: block_prefix_sum

## Variant Context
- Input semantic type: Parallel prefix sum (scan) algorithm
- Datatype(s): fp32
- Data representation: Dense array
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel computes the prefix sum (inclusive scan) within a block. Each thread processes two elements, and the algorithm uses a tree-based approach to compute partial sums. The results are then propagated across blocks using a separate kernel.

## Optimization 1: Warp-Level Shuffle Operations for Intra-Warp Scan
- Commit ID: baseline → optimized
- Optimization type: Compute / Memory
- Summary: Replaced shared memory tree operations with warp shuffle intrinsics for intra-warp prefix sum
- Detailed explanation: The baseline uses shared memory with multiple synchronization barriers to perform the tree-based scan. The optimized version uses `__shfl_up` warp shuffle intrinsics to perform the scan within a warp without shared memory access. This eliminates shared memory latency and reduces synchronization overhead for the intra-warp portion of the algorithm.

- Code excerpt (baseline):
    ```cpp
    // Build up tree
    int tree_offset = 1;
    for(int tree_size = size >> 1; tree_size > 0; tree_size >>= 1)
    {
        __syncthreads();
        if(thread_id < tree_size)
        {
            int from = tree_offset * (2 * thread_id + 1) - 1;
            int to   = tree_offset * (2 * thread_id + 2) - 1;
            block[to] += block[from];
        }
        tree_offset <<= 1;
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Wavefront-level inclusive scan over per-thread totals (a1)
    float sum = a1;
    const int lane    = thread_id & (warpSize - 1);
    const int wave_id = thread_id / warpSize;

    #pragma unroll
    for (int delta = 1; delta < warpSize; delta <<= 1) {
        float n_sh = __shfl_up(sum, delta, warpSize);
        if (lane >= delta) { sum += n_sh; }
    }

    // Offset from preceding threads in the same wavefront
    float thread_prefix = sum - a1;
    ```

- Evidence mapping:
  - "Warp shuffle" → `__shfl_up(sum, delta, warpSize)`
  - "No shared memory in inner loop" → Shuffle operates on registers directly
  - "Unrolled loop" → `#pragma unroll` for the shuffle loop

## Optimization 2: Register-Based Per-Thread Scan
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Perform 2-element per-thread scan in registers before warp-level operations
- Detailed explanation: The optimized version loads two elements into registers (a0, a1) and performs a local 2-element inclusive scan (`a1 += a0`) before the warp-level scan. This reduces the amount of data that needs to be communicated through shuffles or shared memory.

- Code excerpt (optimized):
    ```cpp
    // Load up to two values into registers; zero if out of range
    float a0 = 0.0f;
    float a1 = 0.0f;
    if (x < size) { a0 = d_data[x]; }
    if (x + offset < size) { a1 = d_data[x + offset]; }

    // Per-thread 2-item inclusive scan in registers
    a1 += a0;
    ```

- Evidence mapping:
  - "Register storage" → `float a0 = 0.0f; float a1 = 0.0f;`
  - "Local scan" → `a1 += a0;` performs 2-element prefix sum in registers

## Optimization 3: Simplified Inter-Wavefront Communication
- Commit ID: baseline → optimized
- Optimization type: Memory / Compute
- Summary: Use minimal shared memory only for inter-wavefront prefix propagation
- Detailed explanation: The optimized version only uses shared memory to store wavefront totals for inter-wavefront communication. Each wavefront's last lane writes its total to shared memory, and other wavefronts read the prefix from previous wavefronts. This is much simpler than the baseline's tree-based approach.

- Code excerpt (optimized):
    ```cpp
    // Inter-wavefront accumulation: store each wavefront's total to shared memory
    if (lane == warpSize - 1) {
        smem[wave_id] = sum;
    }
    __syncthreads();

    // Add prefix from previous wavefronts within the block
    float wave_prefix = 0.0f;
    if (wave_id > 0) {
        // For 128 threads (2 wavefronts), wave_prefix is sum of wave 0
        wave_prefix = smem[0];
    }

    out0 += wave_prefix;
    out1 += wave_prefix;
    ```

- Evidence mapping:
  - "Minimal shared memory" → Only `smem[wave_id]` used for wavefront totals
  - "Single sync" → One `__syncthreads()` instead of multiple in baseline
  - "Simple prefix addition" → `wave_prefix = smem[0];` for second wavefront

## Optimization 4: LDS Bank Conflict Mitigation with Padding
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Added padding function to reduce LDS bank conflicts
- Detailed explanation: The optimized version includes a lambda function `idx_pad` that adds padding to shared memory indices to reduce bank conflicts. The padding formula `i + (i >> 5)` adds one element of padding every 32 elements.

- Code excerpt (optimized):
    ```cpp
    // Padded indexing to mitigate LDS bank conflicts: idx_pad(i) = i + (i >> 5)
    auto idx_pad = [](int i) __device__ { return i + (i >> 5); };
    ```

- Evidence mapping:
  - "Padding function" → `auto idx_pad = [](int i) __device__ { return i + (i >> 5); };`
  - "Bank conflict reduction" → Adding 1 element per 32 elements breaks bank conflict patterns
