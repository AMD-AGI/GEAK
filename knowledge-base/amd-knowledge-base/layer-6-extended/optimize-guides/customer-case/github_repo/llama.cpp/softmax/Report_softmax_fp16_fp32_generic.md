Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Softmax

## Variant Context
- Input semantic type: Softmax normalization (attention scores, logits)
- Datatype(s): FP16, FP32
- Data representation: Dense tensors
- Target architecture: Generic (NVIDIA, AMD, Moore Threads)

## Functionality
The softmax kernel computes the softmax function along a specified dimension, typically used for attention score normalization and output probability computation. It implements the numerically stable version: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))).

Key features:
- Numerically stable computation with max subtraction
- Support for broadcasting masks
- Warp and block-level reductions
- Dynamic shared memory for large row sizes

---

## Optimization 1: Dynamic Shared Memory for Large Rows
- Commit ID: 55c2646b4
- Optimization type: Memory (shared memory management)
- Summary: Use dynamic shared memory allocation for softmax to handle variable row sizes efficiently
- Detailed explanation: Instead of statically allocating shared memory based on maximum possible row size, this optimization uses dynamic shared memory allocation. This allows the kernel to adapt to different row sizes without wasting shared memory, improving occupancy for smaller rows while still supporting large rows.

- Code excerpt:
    ```cpp
    // CUDA: add dynamic shared mem to softmax
    template<typename T, int block_size>
    __global__ void softmax_kernel(
        const T * __restrict__ x,
        T * __restrict__ dst,
        const int ncols,
        const float scale) {
        
        // Dynamic shared memory for row data
        extern __shared__ float sdata[];
        
        const int row = blockIdx.x;
        const int tid = threadIdx.x;
        
        // Load row to shared memory with coalesced access
        float max_val = -INFINITY;
        for (int col = tid; col < ncols; col += block_size) {
            float val = (float)x[row * ncols + col] * scale;
            sdata[col] = val;
            max_val = fmaxf(max_val, val);
        }
        
        // Warp reduction for max
        max_val = warp_reduce_max(max_val);
        ...
    }
    
    // Launch with dynamic shared memory
    const size_t shmem = ncols * sizeof(float);
    softmax_kernel<T, 256><<<nrows, 256, shmem, stream>>>(...);
    ```

- Evidence mapping:
  - "Dynamic shared memory" → `extern __shared__ float sdata[]`
  - "Variable row sizes" → `shmem = ncols * sizeof(float)` at launch
  - "Improved occupancy" → smaller shmem for smaller rows

---

## Optimization 2: Block Reduce Refactoring
- Commit ID: 36f013246
- Optimization type: Compute (code reuse, efficiency)
- Summary: Factor out and reuse block_reduce function across softmax, normalization, and other kernels
- Detailed explanation: This optimization extracts the two-stage warp reduction pattern into a reusable `block_reduce` function. The function handles both sum and max reductions efficiently using warp shuffles for intra-warp reduction and shared memory for inter-warp reduction.

- Code excerpt:
    ```cpp
    // CUDA: Factor out and re-use block_reduce function
    enum class block_reduce_method {
        SUM,
        MAX
    };
    
    template<block_reduce_method method, typename T, int block_size>
    __device__ __forceinline__ T block_reduce(T val, T * smem) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        
        // Intra-warp reduction using shuffles
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            if constexpr (method == block_reduce_method::SUM) {
                val += __shfl_xor_sync(0xffffffff, val, offset);
            } else {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
            }
        }
        
        // Store warp results to shared memory
        if (lane_id == 0) {
            smem[warp_id] = val;
        }
        __syncthreads();
        
        // Final reduction by first warp
        if (warp_id == 0) {
            val = lane_id < (block_size / WARP_SIZE) ? smem[lane_id] : 
                  (method == block_reduce_method::SUM ? 0.0f : -INFINITY);
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                if constexpr (method == block_reduce_method::SUM) {
                    val += __shfl_xor_sync(0xffffffff, val, offset);
                } else {
                    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
                }
            }
        }
        return val;
    }
    ```

- Evidence mapping:
  - "Reusable function" → `block_reduce` template function
  - "Two-stage reduction" → warp shuffle + shared memory
  - "Method selection" → `block_reduce_method::SUM` vs `MAX`

---

## Optimization 3: Softmax Broadcasting for Attention Masks
- Commit ID: 55a1c5a5f
- Optimization type: Compute (broadcasting)
- Summary: Add support for broadcasting attention masks in softmax computation
- Detailed explanation: Attention masks often have different shapes than the attention scores (e.g., [1, 1, seq, seq] vs [batch, heads, seq, seq]). This optimization adds efficient broadcasting support so masks can be applied without explicit expansion.

- Code excerpt:
    ```cpp
    // CUDA: add softmax broadcast
    template<typename T>
    __global__ void softmax_with_mask_broadcast(
        const T * __restrict__ x,
        const T * __restrict__ mask,
        T * __restrict__ dst,
        const int ncols,
        const int mask_stride,  // Stride for broadcasting
        const float scale) {
        
        const int row = blockIdx.x;
        const int mask_row = row % mask_stride;  // Broadcast dimension
        
        for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
            float val = (float)x[row * ncols + col] * scale;
            val += (float)mask[mask_row * ncols + col];  // Broadcasted mask
            ...
        }
    }
    ```

- Evidence mapping:
  - "Broadcasting" → `mask_row = row % mask_stride`
  - "Mask application" → `val += (float)mask[mask_row * ncols + col]`
  - "Flexible shapes" → `mask_stride` parameter for broadcast control
