# Kernel: gemm_streamk (Stream-K GEMM Kernel)

## Variant Context
- Input semantic type: Matrix multiplication with load-balanced work distribution
- Datatype(s): FP16/BF16/FP32
- Data representation: Dense tensors with Stream-K partitioning
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The Stream-K GEMM kernel implements a load-balanced work distribution strategy where the K dimension is partitioned across workgroups in a streaming fashion. Unlike traditional data-parallel GEMM where each workgroup computes a complete output tile, Stream-K allows multiple workgroups to contribute to the same output tile, with partial results accumulated via reduction. This improves load balancing for irregular problem sizes.

## Optimization 1: Tree Reduction Strategy
- Commit ID: 22b945e06
- Optimization type: compute / scheduling
- Summary: Implemented tree reduction to reduce the number of accumulation steps from O(N) to O(log N) where N is the number of workgroups contributing to a C tile.

- Detailed explanation:
  In the original Stream-K implementation, partial results from multiple workgroups were accumulated sequentially (linear reduction). The tree reduction optimization organizes the accumulation in a binary tree pattern:
  - Level 0: Workgroups 0,1 reduce → result 0; Workgroups 2,3 reduce → result 1
  - Level 1: Results 0,1 reduce → final result
  
  This reduces the critical path from O(N) to O(log N) steps, significantly improving performance when many workgroups contribute to the same tile.

- Code excerpt:
    ```cpp
    // Tree reduction reduces accumulation steps from O(N) to O(logN)
    // where N is the number of workgroups contributing to a C tile.
    
    // Determine reduction level and partner workgroup
    auto get_tree_reduction_partner = [](index_t wg_idx, index_t level) {
        // At each level, workgroups pair with their neighbor
        // Level 0: (0,1), (2,3), (4,5), ...
        // Level 1: (0,2), (4,6), ...
        // Level 2: (0,4), ...
        index_t stride = 1 << level;
        index_t partner = wg_idx ^ stride;
        return partner;
    };
    
    // Perform tree reduction
    for(index_t level = 0; level < log2(num_contributors); ++level)
    {
        index_t partner = get_tree_reduction_partner(my_wg_idx, level);
        if(my_wg_idx < partner)
        {
            // Wait for partner's partial result
            wait_for_flag(partner);
            // Accumulate partner's result
            accumulate_partial(partner_result);
        }
        else
        {
            // Signal completion and exit
            set_flag(my_wg_idx);
            return;
        }
    }
    ```

- Evidence mapping:
  - "O(N) to O(log N)" → Tree structure with `level < log2(num_contributors)`
  - "Binary tree pattern" → `stride = 1 << level` and `partner = wg_idx ^ stride`
  - "Reduction levels" → Loop over levels with partner selection

## Optimization 2: Cache-Bypassing Flag Synchronization
- Commit ID: 22b945e06
- Optimization type: memory / synchronization
- Summary: Replaced atomic operations with cache-bypassing memory operations (GLC modifier) for flag synchronization, avoiding expensive acquire/release semantics.

- Detailed explanation:
  The original implementation used atomic operations with relaxed semantics for flag synchronization, but this was insufficient to guarantee correct ordering. Stronger acquire/release semantics were too expensive. The optimization uses cache modifiers (GLC - Global Level Coherent) to bypass the cache when writing flags, ensuring visibility without atomics.

  Key insight: By writing directly to memory (bypassing L2 cache), we guarantee that other workgroups will see the updated flag value when they read from memory, without needing expensive atomic operations.

- Code excerpt:
    ```cpp
    // streamk_gemm_coherency.hpp
    
    // Cache-bypassing write for flag synchronization
    // Uses GLC (Global Level Coherent) modifier to bypass cache
    template <typename T>
    CK_TILE_DEVICE void store_glc(T* ptr, T value)
    {
        // GLC modifier ensures write goes directly to memory
        // avoiding cache coherency issues without atomics
        __builtin_amdgcn_global_store_dword(
            reinterpret_cast<int*>(ptr),
            static_cast<int>(value),
            /*glc=*/true,
            /*slc=*/false);
    }
    
    // Non-atomic flag setting using cache bypass
    auto* sk_flags_ptr = static_cast<uint32_t*>(workspace_ptr);
    store_glc(&sk_flags_ptr[my_tile_idx], READY_FLAG);
    
    // Reading with cache bypass to ensure fresh value
    template <typename T>
    CK_TILE_DEVICE T load_glc(T* ptr)
    {
        return __builtin_amdgcn_global_load_dword(
            reinterpret_cast<int*>(ptr),
            /*glc=*/true,
            /*slc=*/false);
    }
    ```

- Evidence mapping:
  - "Cache-bypassing" → `glc=true` parameter in load/store intrinsics
  - "Avoid atomics" → Direct memory operations instead of atomic_store/load
  - "GLC modifier" → `__builtin_amdgcn_global_store_dword` with GLC flag

## Optimization 3: Tile Partitioner with Extra Iterations Tracking
- Commit ID: 22b945e06
- Optimization type: scheduling
- Summary: Enhanced tile partitioner to track extra iterations before each workgroup, enabling correct partial result accumulation.

- Detailed explanation:
  The tile partitioner now tracks `extra_iters_before_me` which indicates how many extra K iterations were processed by previous workgroups for the same output tile. This information is essential for:
  1. Determining the correct K offset for each workgroup
  2. Knowing how many partial results need to be accumulated
  3. Identifying which workgroup is responsible for final reduction

- Code excerpt:
    ```cpp
    struct StreamKTilePartitioner
    {
        // ... existing members ...
        
        // Number of extra K iterations processed by workgroups before this one
        // for the same output tile. Used to determine:
        // 1. K offset for this workgroup's computation
        // 2. Number of partial results to accumulate
        // 3. Whether this workgroup performs final reduction
        index_t extra_iters_before_me;
        
        CK_TILE_DEVICE auto get_k_offset() const
        {
            return (base_k_iters + extra_iters_before_me) * KPerBlock;
        }
        
        CK_TILE_DEVICE bool is_final_reducer() const
        {
            // Last workgroup contributing to this tile performs final reduction
            return extra_iters_before_me + my_k_iters == total_k_iters_for_tile;
        }
    };
    ```

- Evidence mapping:
  - "Extra iterations tracking" → `extra_iters_before_me` member variable
  - "K offset calculation" → `get_k_offset()` using extra iterations
  - "Final reducer identification" → `is_final_reducer()` method
