# Kernel: fmha_batch_prefill (Flash Attention Batch Prefill)

## Variant Context
- Input semantic type: Attention with paged KV cache (for LLM inference)
- Datatype(s): FP16/BF16
- Data representation: Paged KV cache with vectorized or linear layout
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The batch prefill kernel handles the prefill phase of LLM inference where multiple sequences are processed together with paged KV cache. It supports different KV cache memory layouts (LINEAR_LAYOUT, VECTORIZED_LAYOUT) and computes attention over variable-length sequences with page-based memory management.

## Optimization 1: Multi-dimensional Page Index Support for VECTORIZED_LAYOUT
- Commit ID: e3556fed0
- Optimization type: memory / compute
- Summary: Optimized V tensor loading for VECTORIZED_LAYOUT KV cache by matching the tile distribution to GEMM's K-dimension decomposition pattern.

- Detailed explanation:
  The optimization addresses performance issues with VECTORIZED_LAYOUT KV cache by restructuring how V tensor tiles are distributed across threads. The key insight is that the K dimension (seqlen_k) decomposition must match the GEMM's warp-level distribution pattern for efficient memory access.

  For VECTORIZED_LAYOUT, the K dimension is decomposed into 3 sub-dimensions: K = K2 × K0 × K1
  - K2 (V_KIterOuter): Outer iteration count
  - K0 (V_KLanes): Lanes for K dimension (matches GEMM kABKLane)
  - K1 (V_KIterInner): Vector load size (matches GEMM kKPerThread)

  This decomposition enables coalesced memory access and proper alignment with the MFMA instruction's data layout requirements.

- Code excerpt:
    ```cpp
    // V tensor K-dimension decomposition for page index computation
    // The K dimension (seqlen_k) in V distribution is decomposed into multiple sub-dimensions.
    // VECTORIZED_LAYOUT (ColumnMajor, custom distribution):
    //   3D decomposition: K = K2 × K0 × K1
    //   - K2 (V_KIterOuter): Outer iteration count
    //   - K0 (V_KLanes):     Lanes for K dimension (matches GEMM kABKLane)
    //   - K1 (V_KIterInner): Vector load size (matches GEMM kKPerThread)
    
    constexpr index_t V_KIterOuter = [] {
        if constexpr(kKVMemoryLayout ==
                     BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
        {
            // VECTORIZED_LAYOUT: 3D decomposition {K2, K0, K1} when outer iteration is needed
            if constexpr(VDstrEncode::hs_lengthss_[I1].size() == 3)
                return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I0]);
            else
                return index_t{1};
        }
        else
        {
            // LINEAR_LAYOUT: No outer iteration for page lookup
            return index_t{1};
        }
    }();

    constexpr index_t V_KLanes = [] {
        if constexpr(kKVMemoryLayout ==
                     BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
        {
            // VECTORIZED_LAYOUT: K0 is the lanes dimension
            if constexpr(V_KIterOuter > 1)
                return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I1]);
            else
                return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I0]);
        }
        else
        {
            // LINEAR_LAYOUT: First dimension is K0 (lanes)
            return static_cast<index_t>(VDstrEncode::hs_lengthss_[I1][I0]);
        }
    }();
    ```

- Evidence mapping:
  - "3D K decomposition" → `V_KIterOuter`, `V_KLanes`, `V_KIterInner` constexpr calculations
  - "Match GEMM distribution" → Comments referencing "matches GEMM kABKLane" and "matches GEMM kKPerThread"
  - "Layout-specific handling" → `if constexpr(kKVMemoryLayout == VECTORIZED_LAYOUT)` branches

## Optimization 2: Multi-dimensional Y-space Page Lookup
- Commit ID: e3556fed0
- Optimization type: memory
- Summary: Added support for multi-dimensional page index computation in Y-space for efficient gather operations.

- Detailed explanation:
  The optimization introduces `VPageIndexYDims` which specifies which Y-space dimensions participate in page index computation. For VECTORIZED_LAYOUT with outer iteration, both K1 and K2 dimensions are used for 2D page lookup, enabling more efficient memory access patterns for paged KV cache.

- Code excerpt:
    ```cpp
    // VPageIndexYDims: Y-space dimension indices that participate in page index computation
    // VECTORIZED_LAYOUT with outer iteration: sequence<Y_K1, Y_K2>
    //   - Both K1 and K2 are in Y-space (thread iteration dimensions)
    //   - gather_index = y_k1 + y_k2 * len(Y_K1)  (linearized 2D -> 1D)
    
    constexpr auto VPageIndexYDims = []() {
        constexpr index_t K1Minor = VDstrEncode::hs_lengthss_[I1].size() - 1;
        constexpr index_t Y_K1    = VDstrEncode::detail::rhs_major_minor_to_ys_[2][K1Minor];

        if constexpr(kKVMemoryLayout ==
                         BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT &&
                     V_KIterOuter > 1)
        {
            // VECTORIZED_LAYOUT with outer iteration: need 2D page lookup
            constexpr index_t Y_K2 = VDstrEncode::detail::rhs_major_minor_to_ys_[2][I0];
            return sequence<Y_K1, Y_K2>{};
        }
        else
        {
            // LINEAR_LAYOUT or VECTORIZED_LAYOUT without outer iteration: 1D page lookup
            return sequence<Y_K1>{};
        }
    }();
    ```

- Evidence mapping:
  - "Multi-dimensional page lookup" → `sequence<Y_K1, Y_K2>{}` for 2D case
  - "Y-space dimension indices" → `VDstrEncode::detail::rhs_major_minor_to_ys_` mapping
  - "Linearized gather index" → Comment explaining `gather_index = y_k1 + y_k2 * len(Y_K1)`

## Optimization 3: Unified V Offset Update Function
- Commit ID: e3556fed0
- Optimization type: compute / code maintainability
- Summary: Refactored V offset computation into a unified lambda function that handles both 2D and 3D K decomposition cases.

- Detailed explanation:
  The optimization consolidates the V offset update logic into a single `update_v_offsets` lambda that handles both outer iteration (3D decomposition) and simple (2D decomposition) cases. This reduces code duplication and ensures consistent offset computation across the pipeline.

- Code excerpt:
    ```cpp
    auto update_v_offsets = [&](auto k_loop_start) {
        constexpr index_t kLoopStart = decltype(k_loop_start)::value;
        // For 3D K decomposition (K2, K0, K1), compute offsets for each K2 slice
        if constexpr(V_KIterOuter > 1)
        {
            static_for<0, V_KIterOuter, 1>{}([&](auto k2) {
                statically_indexed_array<index_t, V_KIterInner> v_offsets_k2;
                kv_offset_array_transform<statically_indexed_array<index_t, V_KIterInner>,
                                          decltype(v_coord),
                                          I1,
                                          kPageBlockSize,
                                          kLoopStart + k2.value * V_KLanes * V_KIterInner,
                                          V_KIterInner,
                                          1,
                                          kKVMemoryLayout,
                                          false,
                                          kN0,
                                          kVectorSize>(
                    page_idx, stride_v, page_stride_v, v_coord, v_offsets_k2, current_seq_k);
                static_for<0, V_KIterInner, 1>{}([&](auto k1) {
                    constexpr auto idx = number<k1.value + k2.value * V_KIterInner>{};
                    v_offsets[idx]     = v_offsets_k2[k1];
                });
            });
        }
        else
        {
            kv_offset_array_transform<...>(
                page_idx, stride_v, page_stride_v, v_coord, v_offsets, current_seq_k);
        }
    };
    ```

- Evidence mapping:
  - "Unified lambda function" → `auto update_v_offsets = [&](auto k_loop_start)` definition
  - "3D decomposition handling" → `if constexpr(V_KIterOuter > 1)` branch with nested loops
  - "Offset merging" → `v_offsets[idx] = v_offsets_k2[k1]` combining K2 and K1 offsets
