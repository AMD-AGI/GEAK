Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: MoE Align Block Size Kernel

## Variant Context
- Input semantic type: Token-to-expert mapping and alignment for MoE computation
- Datatype(s): INT32 (indices), INT64 (for large token counts)
- Data representation: Sorted token IDs, expert IDs, cumulative sums
- Target architecture: Generic CUDA (SM70+), with ROCm support

## Functionality
This kernel prepares the input data for MoE (Mixture of Experts) computation by:
1. Counting tokens assigned to each expert
2. Computing cumulative sums for expert boundaries
3. Sorting token IDs by expert assignment
4. Padding to block-aligned sizes for efficient GEMM execution
5. Generating expert IDs for each block

## Optimization 1: Parallel Sorted Token ID Initialization
- Commit ID: c5947ecd8
- Optimization type: Parallelism / Memory
- Summary: Uses a separate thread block to initialize sorted_token_ids in parallel with the main computation, hiding initialization latency.

- Detailed explanation:
  Previously, the sorted_token_ids array was initialized sequentially after the main computation. This optimization launches an additional thread block (blockIdx.x == 1) dedicated to filling the sorted_token_ids array with the padding value (numel). This initialization happens in parallel with the main computation in blockIdx.x == 0, effectively hiding the initialization latency.

- Code excerpt:
    ```cpp
    __global__ void moe_align_block_size_kernel(
        // ... parameters ...
        int32_t max_num_tokens_padded) {
      // Use a separate thread block to populate sorted_token_ids
      if (blockIdx.x == 1) {
        if (pad_sorted_token_ids) {
          Vec fill_vec;
          fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
          int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
          Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
          for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
            out_ptr[i] = fill_vec;
          }
        }
        return;
      }
      // Main computation continues in blockIdx.x == 0
      // ...
    }
    ```

    Launch configuration:
    ```cpp
    // Launch with 2 blocks instead of 1
    align_kernel<<<2, threads, shared_mem_size, stream>>>(...);
    ```

- Evidence mapping:
  - "Separate thread block" → `if (blockIdx.x == 1)` handles initialization
  - "Parallel execution" → `<<<2, threads, ...>>>` launches 2 blocks
  - "Vectorized fill" → `Vec fill_vec` with 4 int32 values per store
  - "Early return" → `return` after initialization, main work in block 0

## Optimization 2: Small Batch Expert Mode with Dedicated Fill Threads
- Commit ID: c5947ecd8
- Optimization type: Parallelism / Scheduling
- Summary: For small batches, uses a portion of threads within the same block for initialization while others perform computation, with synchronization barriers.

- Detailed explanation:
  For small batch sizes (< 1024 tokens) with few experts (≤ 64), a specialized kernel path is used. This optimization dedicates a fixed number of threads (256) within the same thread block for filling sorted_token_ids, while the remaining threads perform the main computation. Three `__syncthreads()` barriers ensure proper synchronization between the fill threads and compute threads.

- Code excerpt:
    ```cpp
    template <typename scalar_t, int32_t fill_threads>
    __global__ void moe_align_block_size_small_batch_expert_kernel(
        // ... parameters ...
        int32_t max_num_tokens_padded) {
      // Use an additional group of threads to fill sorted_token_ids
      if (threadIdx.x < fill_threads) {
        // Initialize sorted_token_ids with numel
        if (pad_sorted_token_ids) {
          for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
            sorted_token_ids[it] = numel;
          }
        }
        // Three __syncthreads() corresponding to the other threads
        __syncthreads();
        __syncthreads();
        __syncthreads();
        return;
      }

      // Compute threads (threadIdx.x >= fill_threads)
      const size_t tid = threadIdx.x - fill_threads;
      const size_t stride = blockDim.x - fill_threads;
      // ... main computation ...
    }
    ```

    Launch configuration:
    ```cpp
    constexpr int32_t fill_threads = 256;
    small_batch_expert_kernel<<<1, fill_threads + threads, shared_mem_size, stream>>>(...);
    ```

- Evidence mapping:
  - "Dedicated fill threads" → `if (threadIdx.x < fill_threads)`
  - "Thread count adjustment" → `tid = threadIdx.x - fill_threads`
  - "Synchronization barriers" → Three `__syncthreads()` calls
  - "Combined launch" → `fill_threads + threads` total threads

## Optimization 3: Vectorized Memory Operations
- Commit ID: a3398d847 (earlier optimization, refined in c5947ecd8)
- Optimization type: Memory
- Summary: Uses vectorized int4 (Vec) stores for filling sorted_token_ids, achieving 4x memory bandwidth utilization.

- Detailed explanation:
  Instead of storing one int32 at a time, the kernel uses a Vec type (int4) to store 4 int32 values in a single memory transaction. This quadruples the effective memory bandwidth for the initialization operation.

- Code excerpt:
    ```cpp
    // Vec type for vectorized stores
    using Vec = int4;  // 4 x int32 = 128 bits
    constexpr int VEC_SIZE = 4;

    // Vectorized fill
    Vec fill_vec;
    fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
    int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
      out_ptr[i] = fill_vec;
    }
    ```

- Evidence mapping:
  - "Vectorized type" → `using Vec = int4`
  - "4 values per store" → `fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel`
  - "Reinterpret cast" → `reinterpret_cast<Vec*>(sorted_token_ids)`
  - "Coalesced access" → sequential `i` values map to consecutive memory

## Optimization 4: Fused Sorted Token ID Padding
- Commit ID: 57ab77691
- Optimization type: Fusion
- Summary: Fuses the padding of sorted_token_ids into the main kernel, eliminating a separate kernel launch.

- Detailed explanation:
  Previously, padding sorted_token_ids required a separate operation. This optimization integrates the padding directly into the moe_align_block_size kernel, reducing kernel launch overhead and improving overall latency.

- Code excerpt:
    ```cpp
    // Padding is now part of the main kernel
    void moe_align_block_size(
        // ...
        bool pad_sorted_token_ids,  // Flag to enable padding
        // ...
    ) {
      // Kernel handles padding internally based on flag
    }
    ```

- Evidence mapping:
  - "Fused operation" → `pad_sorted_token_ids` parameter controls in-kernel padding
  - "No separate launch" → padding happens within the same kernel execution

## Optimization 5: Reduced torch.zeros Overhead
- Commit ID: 8b5f83ed3
- Optimization type: Memory / Host-side
- Summary: Reduces overhead from torch.zeros by pre-allocating buffers and using kernel-side initialization.

- Detailed explanation:
  Calling torch.zeros for temporary buffers adds Python/PyTorch overhead. This optimization moves the zero-initialization into the CUDA kernel, avoiding the host-side allocation and initialization overhead.

- Code excerpt:
    ```cpp
    // Kernel-side initialization instead of torch.zeros
    // The sorted_token_ids is filled with numel (padding value) in the kernel
    // rather than being pre-initialized with zeros on the host
    ```

- Evidence mapping:
  - "Kernel-side init" → initialization happens in CUDA kernel, not Python
  - "Reduced overhead" → avoids torch.zeros call and associated synchronization
