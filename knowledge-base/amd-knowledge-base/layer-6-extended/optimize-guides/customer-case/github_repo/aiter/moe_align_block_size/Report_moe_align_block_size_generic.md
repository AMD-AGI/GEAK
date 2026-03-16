# Kernel: moe_align_block_size

## Variant Context
- Input semantic type: MoE (Mixture of Experts) token routing
- Datatype(s): INT32 (token IDs and indices)
- Data representation: Token-to-expert mapping with block alignment
- Target architecture: Generic (gfx942, gfx950)

## Functionality
This kernel aligns token assignments to experts with a specified block size for efficient batched GEMM operations in MoE layers. It takes topk_ids (expert assignments per token) and produces sorted_token_ids, expert_ids, and token_nums arrays that enable coalesced memory access patterns in subsequent MoE GEMM kernels.

## Optimization 1: O(E) Shared Memory Instead of O(E²)
- Commit ID: 62653348a
- Optimization type: Memory
- Summary: Reduced shared memory usage from O(num_experts²) to O(num_experts) by using atomic operations, achieving 18% average speedup.

- Detailed explanation:
  The original implementation used a 2D tensor `tokens_cnts[num_threads][num_experts]` in shared memory to track per-thread token counts for each expert. This required O(num_experts × num_threads) = O(E²) shared memory (since num_threads ≈ num_experts).
  
  The optimized version uses:
  1. `expert_token_counts[num_experts]` - global counts per expert
  2. `cumsum[num_experts + 1]` - prefix sum for block offsets  
  3. `write_positions[num_experts]` - atomic write counters
  
  Total: 3 × num_experts + 1 = O(E) shared memory
  
  This reduces shared memory pressure and enables better occupancy on AMD GPUs.

- Code excerpt:
    ```cpp
    // BEFORE: O(E²) shared memory
    int32_t* tokens_cnts = shared_mem; // 2d tensor with shape (num_experts + 1, num_experts)
    int32_t* cumsum = shared_mem + (num_experts + 1) * num_experts;
    const int32_t shared_mem = ((num_experts + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
    
    // AFTER: O(E) shared memory
    int32_t* expert_token_counts = shared_mem;
    int32_t* cumsum = shared_mem + num_experts;
    int32_t* write_positions = cumsum + (num_experts + 1);
    const int32_t shared_mem = (3 * num_experts + 1) * sizeof(int32_t);
    ```

- Evidence mapping:
  - Reduced shared memory → `(3 * num_experts + 1)` vs `((num_experts + 1) * num_experts + (num_experts + 1))`
  - O(E) layout → Three 1D arrays instead of one 2D array

## Optimization 2: Atomic Operations for Token Counting
- Commit ID: 62653348a
- Optimization type: Compute / Scheduling
- Summary: Replaced per-thread counting with atomic additions, eliminating the need for cross-thread reduction.

- Detailed explanation:
  The original algorithm required each thread to maintain separate counts and then perform a sequential reduction across threads. The optimized version uses atomic operations to directly update global expert counts, which:
  1. Eliminates the O(num_threads) reduction loop
  2. Enables parallel counting without synchronization barriers
  3. Simplifies the algorithm from 4 passes to 3 passes

- Code excerpt:
    ```cpp
    // BEFORE: Per-thread counting + reduction
    for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
    {
        ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
    }
    __syncthreads();
    // Reduction loop
    for(int i = 1; i <= blockDim.x; ++i)
    {
        tokens_cnts[index(num_experts, i, threadIdx.x)] +=
            tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
    
    // AFTER: Atomic counting
    for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
    {
        int32_t expert_id = topk_ids[i];
        atomicAdd(&expert_token_counts[expert_id], 1);
    }
    ```

- Evidence mapping:
  - Atomic counting → `atomicAdd(&expert_token_counts[expert_id], 1)`
  - Eliminated reduction → No per-thread accumulation loop

## Optimization 3: Atomic Write Position Assignment
- Commit ID: 62653348a
- Optimization type: Scheduling
- Summary: Used atomic operations for output position assignment, enabling parallel writes without conflicts.

- Detailed explanation:
  The original implementation computed write positions based on per-thread prefix sums, requiring complex index calculations. The optimized version initializes write positions from the cumsum and uses atomic increments to get unique output positions for each token.

- Code excerpt:
    ```cpp
    // BEFORE: Complex index calculation
    int32_t rank_post_pad = tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
                            cumsum[expert_id] * block_size;
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
    
    // AFTER: Atomic position assignment
    for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
    {
        int32_t expert_id = topk_ids[i];
        // Atomically get the next write position for this expert
        int32_t rank_post_pad = atomicAdd(&write_positions[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
    ```

- Evidence mapping:
  - Atomic position assignment → `atomicAdd(&write_positions[expert_id], 1)`
  - Simplified indexing → Direct use of atomic return value as write position
