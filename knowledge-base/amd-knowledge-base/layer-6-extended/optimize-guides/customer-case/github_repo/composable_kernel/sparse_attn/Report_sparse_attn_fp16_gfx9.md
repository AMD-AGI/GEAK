Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Sparse Attention (VSA - Variable Sparse Attention)

## Variant Context
- Input semantic type: Sparse attention patterns for efficient long-sequence processing
- Datatype(s): FP16/BF16
- Data representation: Sparse attention masks with variable patterns
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The Sparse Attention kernel implements Variable Sparse Attention (VSA) patterns for efficient processing of long sequences. Instead of computing full N×N attention, it computes attention only for specified sparse patterns (e.g., local windows, strided patterns, or learned sparse masks), reducing complexity from O(N²) to O(N×k) where k is the average number of attended positions.

## Optimization 1: Variable Sparse Attention (VSA) Support
- Commit ID: 4d2f8c111
- Optimization type: compute / memory
- Summary: Added Variable Sparse Attention support for FMHA, enabling efficient attention computation with arbitrary sparse patterns.

- Detailed explanation:
  VSA allows specifying which query-key pairs should be computed, skipping the rest. This is implemented through:
  1. Block-sparse patterns: Attention computed for specified blocks
  2. Variable-length patterns: Different sparsity per query position
  3. Efficient mask representation: Compressed sparse format

  The kernel processes only the non-zero blocks, significantly reducing computation for sparse patterns.

- Code excerpt:
    ```cpp
    // Variable Sparse Attention kernel
    template <typename Problem>
    struct SparseAttentionKernel
    {
        // Sparse block configuration
        static constexpr index_t kBlockSizeQ = Problem::kBlockSizeQ;
        static constexpr index_t kBlockSizeK = Problem::kBlockSizeK;
        
        // Sparse pattern representation
        struct SparsePattern
        {
            const index_t* block_offsets;  // Start offset for each Q block
            const index_t* block_indices;  // K block indices to attend to
            index_t num_q_blocks;
        };
        
        CK_TILE_DEVICE void operator()(
            const QType* q,
            const KType* k,
            const VType* v,
            OType* output,
            const SparsePattern& pattern)
        {
            const index_t q_block_idx = blockIdx.x;
            
            // Get K blocks to attend to for this Q block
            index_t k_start = pattern.block_offsets[q_block_idx];
            index_t k_end = pattern.block_offsets[q_block_idx + 1];
            index_t num_k_blocks = k_end - k_start;
            
            // Skip if no K blocks to attend to
            if(num_k_blocks == 0)
            {
                // Output zeros or handle appropriately
                return;
            }
            
            // Load Q block
            auto q_tile = load_tile(q + q_block_idx * kBlockSizeQ * head_dim);
            
            // Accumulate attention over sparse K blocks
            AccType m = -infinity;
            AccType l = 0;
            auto o_acc = make_zero_tile<AccType>();
            
            for(index_t i = 0; i < num_k_blocks; ++i)
            {
                index_t k_block_idx = pattern.block_indices[k_start + i];
                
                // Load K, V blocks
                auto k_tile = load_tile(k + k_block_idx * kBlockSizeK * head_dim);
                auto v_tile = load_tile(v + k_block_idx * kBlockSizeK * head_dim);
                
                // Compute attention for this block
                auto s = gemm(q_tile, transpose(k_tile));
                s = s * scale;
                
                // Online softmax update
                AccType m_new = max(m, row_max(s));
                AccType scale_old = exp(m - m_new);
                AccType scale_new = exp(row_max(s) - m_new);
                
                l = l * scale_old + row_sum(exp(s - m_new));
                o_acc = o_acc * scale_old + gemm(softmax_local(s, m_new), v_tile);
                m = m_new;
            }
            
            // Finalize output
            auto output_tile = o_acc / l;
            store_tile(output + q_block_idx * kBlockSizeQ * head_dim, output_tile);
        }
    };
    ```

- Evidence mapping:
  - "Variable sparse patterns" → `SparsePattern` struct with offsets and indices
  - "Block-sparse" → `kBlockSizeQ`, `kBlockSizeK` for block granularity
  - "Skip computation" → Loop only over `num_k_blocks` non-zero blocks
  - "Online softmax" → Incremental m, l updates for numerical stability

## Optimization 2: Compressed Sparse Pattern Representation
- Commit ID: 4d2f8c111
- Optimization type: memory
- Summary: Uses CSR-like compressed format for sparse attention patterns.

- Detailed explanation:
  The sparse pattern is stored in a compressed format similar to CSR (Compressed Sparse Row):
  - `block_offsets`: Array of size (num_q_blocks + 1), where block_offsets[i] is the start index in block_indices for Q block i
  - `block_indices`: Array of K block indices to attend to

  This representation is memory-efficient and allows O(1) lookup of which K blocks each Q block should attend to.

- Code excerpt:
    ```cpp
    // CSR-like sparse pattern
    // Example: Q blocks 0,1,2 attend to K blocks as follows:
    // Q0 -> K0, K1, K2
    // Q1 -> K1, K2, K3, K4
    // Q2 -> K3, K4
    //
    // block_offsets = [0, 3, 7, 9]
    // block_indices = [0, 1, 2, 1, 2, 3, 4, 3, 4]
    
    struct SparsePatternCSR
    {
        const index_t* block_offsets;  // Size: num_q_blocks + 1
        const index_t* block_indices;  // Size: total_nonzero_blocks
        
        CK_TILE_DEVICE index_t get_num_k_blocks(index_t q_block) const
        {
            return block_offsets[q_block + 1] - block_offsets[q_block];
        }
        
        CK_TILE_DEVICE index_t get_k_block(index_t q_block, index_t local_idx) const
        {
            return block_indices[block_offsets[q_block] + local_idx];
        }
    };
    ```

- Evidence mapping:
  - "CSR-like format" → `block_offsets` and `block_indices` arrays
  - "Memory efficient" → Only stores non-zero block indices
  - "O(1) lookup" → Direct indexing into arrays

## Optimization 3: Integration with FMHA Pipeline
- Commit ID: 4d2f8c111
- Optimization type: compute
- Summary: Integrated sparse attention with existing FMHA pipeline infrastructure.

- Detailed explanation:
  The sparse attention kernel reuses the optimized FMHA pipeline components (tile loading, GEMM, softmax) while adding sparse pattern handling. This ensures the sparse kernel benefits from all existing FMHA optimizations.

- Code excerpt:
    ```cpp
    // Sparse attention using FMHA pipeline components
    template <typename FmhaPipeline, typename SparsePattern>
    struct SparseFmhaKernel
    {
        using Pipeline = FmhaPipeline;
        
        // Reuse FMHA tile shapes
        static constexpr index_t kM0 = Pipeline::kM0;
        static constexpr index_t kN0 = Pipeline::kN0;
        static constexpr index_t kK0 = Pipeline::kK0;
        
        CK_TILE_DEVICE void operator()(...)
        {
            // Use FMHA's optimized tile operations
            auto q_tile = Pipeline::load_q_tile(q_ptr, q_offset);
            
            // Sparse iteration over K blocks
            for(auto k_block : sparse_pattern.get_k_blocks(q_block))
            {
                auto k_tile = Pipeline::load_k_tile(k_ptr, k_block * kN0);
                auto v_tile = Pipeline::load_v_tile(v_ptr, k_block * kN0);
                
                // Reuse FMHA's GEMM and softmax
                auto s = Pipeline::compute_qk(q_tile, k_tile);
                Pipeline::apply_softmax(s, m, l);
                Pipeline::accumulate_output(o_acc, s, v_tile);
            }
            
            Pipeline::store_output(output_ptr, o_acc, l);
        }
    };
    ```

- Evidence mapping:
  - "Pipeline reuse" → `using Pipeline = FmhaPipeline`
  - "Shared tile shapes" → `kM0`, `kN0`, `kK0` from Pipeline
  - "Optimized operations" → `Pipeline::compute_qk`, `Pipeline::apply_softmax`
