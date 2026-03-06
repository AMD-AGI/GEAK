Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: causal_conv1d_simple

## Variant Context
- Input semantic type: Simple causal 1D convolution
- Datatype(s): fp16 (half precision)
- Data representation: Dense sequence tensors
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs simple causal 1D convolution with a width-4 filter. It processes input sequences in chunks, using shared memory for inter-thread communication and double buffering for prefetching.

## Optimization 1: Double Buffering for Prefetching
- Commit ID: baseline → optimized
- Optimization type: Memory / Scheduling
- Summary: Use double-buffered arrays to overlap data loading with computation
- Detailed explanation: The optimized version uses two alternating buffers (x_vals_buf0, x_vals_buf1) to prefetch the next chunk while processing the current chunk. This hides memory latency by overlapping computation with data transfer.

- Code excerpt (optimized):
    ```cpp
    // Double-buffered prefetch arrays with 16-byte alignment
    alignas(16) input_t x_vals_buf0[2 * kNElts] = {__float2half(0.0f)};
    alignas(16) input_t x_vals_buf1[2 * kNElts] = {__float2half(0.0f)};
    input_t* cur_buf = x_vals_buf0;
    input_t* next_buf = x_vals_buf1;

    // Prefetch next chunk into next_buf
    if (chunk + 1 < n_chunks) {
        // ... load into next_buf
    }
    // ... process cur_buf
    // Swap buffers
    input_t* tmp = cur_buf;
    cur_buf = next_buf;
    next_buf = tmp;
    ```

- Evidence mapping:
  - "Double buffering" → Two aligned buffers `x_vals_buf0` and `x_vals_buf1`
  - "Prefetch overlap" → Loading next chunk while processing current

## Optimization 2: Warp Shuffle for Inter-Thread Communication
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Use 64-bit packed warp shuffles to reduce instruction count for tail exchange
- Detailed explanation: The optimized version packs two 32-bit values into 64-bit and uses __shfl_up to exchange data between threads within a warp, reducing the number of shuffle instructions needed.

- Code excerpt (optimized):
    ```cpp
    // Packed 64-bit shuffles to reduce instruction count
    uint64_t cur_lo = (static_cast<uint64_t>(cur_tail_u4.y) << 32) | cur_tail_u4.x;
    uint64_t cur_hi = (static_cast<uint64_t>(cur_tail_u4.w) << 32) | cur_tail_u4.z;

    uint64_t prev_lo64 = __shfl_up(cur_lo, 1, warpSize);
    uint64_t prev_hi64 = __shfl_up(cur_hi, 1, warpSize);
    ```

- Evidence mapping:
  - "64-bit packing" → Combining two 32-bit values into uint64_t
  - "Warp shuffle" → `__shfl_up(cur_lo, 1, warpSize)`

## Optimization 3: FMA Instructions for Convolution
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Use fused multiply-add (fmaf) for convolution computation
- Detailed explanation: The optimized version uses fmaf intrinsics to combine multiplication and addition into single instructions, improving throughput and precision.

- Code excerpt (optimized):
    ```cpp
    float acc = bias_val;
    acc = fmaf(w0, f0, acc);
    acc = fmaf(w1, f1, acc);
    acc = fmaf(w2, f2, acc);
    acc = fmaf(w3, f3, acc);
    ```

- Evidence mapping:
  - "FMA intrinsic" → `fmaf(w0, f0, acc)` instead of `acc += w0 * f0`

## Optimization 4: Shared Memory Weight Caching
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Load weights into shared memory once, then cache in registers
- Detailed explanation: Weights are loaded into shared memory by a subset of threads, synchronized, then cached into per-thread registers to eliminate repeated shared memory reads in the hot loop.

- Code excerpt (optimized):
    ```cpp
    __shared__ float weight_shared[kWidth];

    if (tidx < kWidth) {
        weight_shared[tidx] = __half2float(weight[tidx * weight_width_stride]);
    }
    __syncthreads();

    // Cache weights into registers
    const float w0 = weight_shared[0];
    const float w1 = weight_shared[1];
    const float w2 = weight_shared[2];
    const float w3 = weight_shared[3];
    ```

- Evidence mapping:
  - "Shared memory for weights" → `__shared__ float weight_shared[kWidth];`
  - "Register caching" → `const float w0 = weight_shared[0];`
