Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: fmha_bwd (Flash Multi-Head Attention Backward)

## Variant Context
- Input semantic type: Attention backward pass (gradient computation)
- Datatype(s): FP16/BF16/FP32
- Data representation: Dense tensors with optional masking
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The FMHA backward kernel computes gradients for the flash attention operation: dQ, dK, dV given the output gradient dO. It supports various optimizations including transpose loading, IGLP scheduling, and decode-specific pipelines.

## Optimization 1: Transpose Load Pipeline for GFX950
- Commit ID: 4fde1646e
- Optimization type: memory / compute
- Summary: Added transpose load (trload) pipeline optimized for GFX950, enabling efficient data layout transformation during memory access.

- Detailed explanation:
  The transpose load pipeline performs data transposition during the load operation, avoiding separate transpose passes. This is particularly beneficial for the backward pass where Q, K, V tensors need to be accessed in different layouts for dQ, dK, dV computation.

  Key features:
  - Two-stage prefetching for hiding memory latency
  - IGLP (Instruction-Level Graph Parallelism) scheduling
  - Optimized for GFX950's memory subsystem

- Code excerpt:
    ```cpp
    // block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp
    
    template <typename Problem_, typename Policy_>
    struct BlockFmhaBwdDqDkDvPipelineTrloadKrKtrVr
    {
        // Transpose load configuration
        static constexpr bool kEnableTrload = true;
        
        // Two-stage prefetching
        template <typename KDramWindow, typename VDramWindow>
        CK_TILE_DEVICE void prefetch_kv(KDramWindow& k_window, VDramWindow& v_window)
        {
            // Stage 1: Prefetch K with transpose
            async_load_tile_transpose(k_lds_window, k_window);
            
            // Stage 2: Prefetch V
            async_load_tile(v_lds_window, v_window);
            
            // Move windows for next iteration
            move_tile_window(k_window, {kK0, 0});
            move_tile_window(v_window, {kK1, 0});
        }
        
        // IGLP scheduling for instruction overlap
        CK_TILE_DEVICE void schedule_gemm()
        {
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
            __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
        }
    };
    ```

- Evidence mapping:
  - "Transpose load" → `async_load_tile_transpose` function
  - "Two-stage prefetching" → Separate K and V prefetch stages
  - "IGLP scheduling" → `sched_group_barrier` calls

## Optimization 2: Decode Pipeline for Inference
- Commit ID: 8e1eb0c1e
- Optimization type: compute / scheduling
- Summary: Added specialized decode pipeline for inference scenarios with short query sequences.

- Detailed explanation:
  The decode pipeline is optimized for the common inference pattern where seqlen_q is small (often 1) while seqlen_k can be large. This pipeline uses different tile sizes and scheduling strategies optimized for this access pattern.

- Code excerpt:
    ```cpp
    // Decode-specific tile configuration
    static constexpr index_t kM0Decode = 16;  // Smaller M for decode
    static constexpr index_t kN0Decode = 64;  // Larger N for K/V
    
    // Decode pipeline selector
    template <typename Problem>
    struct FmhaBwdPipelineSelector
    {
        using type = std::conditional_t<
            Problem::kIsDecodeMode,
            BlockFmhaBwdDecodePipeline<Problem>,
            BlockFmhaBwdPrefillPipeline<Problem>
        >;
    };
    ```

- Evidence mapping:
  - "Decode mode" → `kIsDecodeMode` template parameter
  - "Different tile sizes" → `kM0Decode`, `kN0Decode` constants
  - "Pipeline selection" → `FmhaBwdPipelineSelector` conditional

## Optimization 3: Remove Unnecessary Padding
- Commit ID: b0a97498b
- Optimization type: memory
- Summary: Removed unnecessary padding in backward pass tensors, reducing memory footprint and improving cache utilization.

- Detailed explanation:
  The optimization analyzes tensor access patterns and removes padding that was previously added for alignment but not actually needed. This reduces memory bandwidth requirements and improves cache hit rates.

- Code excerpt:
    ```cpp
    // Before: padded to power of 2
    static constexpr index_t kHDimPadded = next_power_of_2(kHDim);
    
    // After: pad only to multiple of 8 (MFMA requirement)
    static constexpr index_t kHDimPadded = (kHDim + 7) / 8 * 8;
    ```

- Evidence mapping:
  - "Reduced padding" → Changed from power-of-2 to multiple-of-8
  - "MFMA requirement" → Minimum alignment for MFMA instructions

## Optimization 4: Int32 Overflow Fix for Deterministic Mode
- Commit ID: fcc9372c0
- Optimization type: correctness / compute
- Summary: Fixed integer overflow in deterministic FMHA backward by using 64-bit arithmetic for large sequence lengths.

- Detailed explanation:
  For very long sequences, the product of indices could overflow 32-bit integers. The fix uses 64-bit arithmetic for index calculations in deterministic mode where exact reproducibility is required.

- Code excerpt:
    ```cpp
    // Use int64 for large index calculations
    CK_TILE_DEVICE auto compute_global_offset(index_t batch, index_t head, 
                                               index_t seq_q, index_t seq_k)
    {
        // Use int64_t to avoid overflow for large sequences
        int64_t offset = static_cast<int64_t>(batch) * batch_stride +
                         static_cast<int64_t>(head) * head_stride +
                         static_cast<int64_t>(seq_q) * seq_q_stride +
                         static_cast<int64_t>(seq_k);
        return offset;
    }
    ```

- Evidence mapping:
  - "64-bit arithmetic" → `int64_t` type usage
  - "Overflow prevention" → Explicit casting before multiplication
