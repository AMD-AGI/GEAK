# Kernel: FMHA SplitKV / PagedKV / AppendKV (Inference Attention Kernels)

## Variant Context
- Input semantic type: Attention for LLM inference with KV cache
- Datatype(s): FP16/BF16/FP8
- Data representation: Split KV, Paged KV cache, Incremental decoding
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
These kernels are optimized for LLM inference scenarios:
- **SplitKV**: Splits the KV dimension across multiple workgroups for long sequences
- **PagedKV**: Supports paged memory layout for KV cache (like vLLM)
- **AppendKV**: Efficiently appends new KV entries during incremental decoding

## Optimization 1: SplitKV Decode Optimization (seqlen_q=1)
- Commit ID: 24b12d04a
- Optimization type: compute / scheduling
- Summary: Optimized SplitKV kernel for decode phase where query sequence length is 1.

- Detailed explanation:
  During LLM decode, each token generates a query of length 1 that attends to the entire KV cache. The optimization:
  1. Uses specialized tile sizes for seqlen_q=1
  2. Reduces thread block size for better occupancy
  3. Optimizes reduction across split-K workgroups

- Code excerpt:
    ```cpp
    // SplitKV optimization for decode (seqlen_q=1)
    template <typename Problem>
    struct FmhaSplitKVDecodeOptimization
    {
        // Specialized tile sizes for decode
        static constexpr index_t kM0Decode = 1;   // seqlen_q = 1
        static constexpr index_t kN0Decode = 128; // Larger K tile for throughput
        
        // Reduced block size for decode
        static constexpr index_t kBlockSizeDecode = 64;  // vs 256 for prefill
        
        CK_TILE_DEVICE void operator()(...)
        {
            // Single query row - no M-dimension parallelism needed
            const index_t q_idx = blockIdx.x;
            
            // Split K dimension across workgroups
            const index_t k_split_idx = blockIdx.y;
            const index_t k_start = k_split_idx * kKPerSplit;
            const index_t k_end = min(k_start + kKPerSplit, seqlen_k);
            
            // Compute partial attention for this K split
            auto [m_partial, l_partial, o_partial] = 
                compute_attention_partial(q, k, v, k_start, k_end);
            
            // Store partial results for reduction
            store_partial_results(workspace, k_split_idx, m_partial, l_partial, o_partial);
        }
    };
    ```

- Evidence mapping:
  - "seqlen_q=1" → `kM0Decode = 1`
  - "Reduced block size" → `kBlockSizeDecode = 64`
  - "Split-K" → `k_split_idx` and partial result storage

## Optimization 2: Paged KV Cache Support
- Commit ID: 9f4c5d737
- Optimization type: memory
- Summary: Added paged KV cache prefill support for memory-efficient inference.

- Detailed explanation:
  Paged KV cache stores KV entries in fixed-size pages rather than contiguous memory. This enables:
  1. Dynamic memory allocation for variable-length sequences
  2. Memory sharing across requests (prefix caching)
  3. Reduced memory fragmentation

- Code excerpt:
    ```cpp
    // Paged KV cache access
    template <typename Problem>
    struct PagedKVAccess
    {
        static constexpr index_t kPageSize = Problem::kPageSize;
        
        CK_TILE_DEVICE auto get_kv_ptr(
            const void* kv_cache,
            const index_t* page_table,
            index_t seq_idx,
            index_t head_idx)
        {
            // Compute page and offset within page
            index_t page_idx = seq_idx / kPageSize;
            index_t page_offset = seq_idx % kPageSize;
            
            // Look up physical page from page table
            index_t physical_page = page_table[page_idx];
            
            // Compute final address
            return reinterpret_cast<const KVType*>(kv_cache) + 
                   physical_page * kPageSize * head_dim +
                   page_offset * head_dim;
        }
    };
    ```

- Evidence mapping:
  - "Paged memory" → `page_table` lookup
  - "Fixed-size pages" → `kPageSize` constant
  - "Dynamic allocation" → Indirection through page table

## Optimization 3: Attention Sink Support
- Commit ID: f5573f56d
- Optimization type: compute
- Summary: Added attention sink support for streaming LLM inference.

- Detailed explanation:
  Attention sink keeps the first few tokens (sink tokens) always in the attention window, even when using sliding window attention. This improves quality for streaming inference with very long sequences.

- Code excerpt:
    ```cpp
    // Attention sink configuration
    template <typename Problem>
    struct AttentionSinkConfig
    {
        index_t sink_size;  // Number of sink tokens to keep
        
        CK_TILE_DEVICE bool is_sink_token(index_t k_idx) const
        {
            return k_idx < sink_size;
        }
        
        CK_TILE_DEVICE bool should_attend(
            index_t q_idx, 
            index_t k_idx,
            index_t window_size) const
        {
            // Always attend to sink tokens
            if(is_sink_token(k_idx))
                return true;
            
            // Sliding window for non-sink tokens
            return (q_idx - k_idx) < window_size;
        }
    };
    ```

- Evidence mapping:
  - "Sink tokens" → `sink_size` parameter
  - "Always attend" → `is_sink_token` check
  - "Sliding window" → Combined sink + window logic

## Optimization 4: SplitKV Combine Kernel
- Commit ID: (fmha_fwd_splitkv_combine_kernel.hpp)
- Optimization type: compute
- Summary: Efficient reduction of partial attention results from split-K workgroups.

- Detailed explanation:
  After split-K workgroups compute partial attention, a combine kernel reduces the results. The optimization uses online softmax reduction to combine partial (m, l, o) tuples efficiently.

- Code excerpt:
    ```cpp
    // SplitKV combine kernel
    template <typename Problem>
    struct FmhaSplitKVCombineKernel
    {
        CK_TILE_DEVICE void operator()(
            const AccType* partial_m,    // [num_splits, seqlen_q]
            const AccType* partial_l,    // [num_splits, seqlen_q]
            const AccType* partial_o,    // [num_splits, seqlen_q, head_dim]
            OutputType* output,
            index_t num_splits)
        {
            const index_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Online softmax reduction across splits
            AccType m_combined = -infinity;
            AccType l_combined = 0;
            AccType o_combined[kHeadDim] = {0};
            
            for(index_t s = 0; s < num_splits; ++s)
            {
                AccType m_s = partial_m[s * seqlen_q + q_idx];
                AccType l_s = partial_l[s * seqlen_q + q_idx];
                
                // Update combined max
                AccType m_new = max(m_combined, m_s);
                
                // Rescale factors
                AccType scale_old = exp(m_combined - m_new);
                AccType scale_new = exp(m_s - m_new);
                
                // Update l and o
                l_combined = l_combined * scale_old + l_s * scale_new;
                for(index_t d = 0; d < kHeadDim; ++d)
                {
                    o_combined[d] = o_combined[d] * scale_old + 
                                    partial_o[(s * seqlen_q + q_idx) * kHeadDim + d] * scale_new;
                }
                
                m_combined = m_new;
            }
            
            // Finalize output
            for(index_t d = 0; d < kHeadDim; ++d)
            {
                output[q_idx * kHeadDim + d] = o_combined[d] / l_combined;
            }
        }
    };
    ```

- Evidence mapping:
  - "Combine kernel" → Separate kernel for reduction
  - "Online softmax" → Incremental m, l, o updates
  - "Rescaling" → `scale_old`, `scale_new` factors

## Optimization 5: Reverse Block Index Assignment for Masking
- Commit ID: c42b957d6
- Optimization type: scheduling
- Summary: Assign block indices reversely when using causal mask for better load balancing.

- Detailed explanation:
  With causal masking, earlier query positions attend to fewer keys than later positions. Reversing block assignment ensures workgroups processing later queries (more work) start first, improving load balancing.

- Code excerpt:
    ```cpp
    // Reverse block index for causal mask
    CK_TILE_DEVICE index_t get_block_idx(index_t block_id, index_t num_blocks, bool use_mask)
    {
        if(use_mask)
        {
            // Reverse: later queries (more work) get lower block IDs
            return num_blocks - 1 - block_id;
        }
        else
        {
            return block_id;
        }
    }
    ```

- Evidence mapping:
  - "Reverse assignment" → `num_blocks - 1 - block_id`
  - "Causal mask" → `use_mask` condition
  - "Load balancing" → Later queries start first
