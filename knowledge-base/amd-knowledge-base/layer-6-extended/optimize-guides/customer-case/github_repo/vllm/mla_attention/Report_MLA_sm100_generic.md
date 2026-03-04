---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Multi-head Latent Attention (MLA)

## Variant Context
- Input semantic type: Attention with compressed KV representations
- Datatype(s): fp16, bf16, fp8
- Data representation: Latent KV vectors with RoPE-encoded keys
- Target architecture: CUDA sm100 (Blackwell), Triton

## Functionality
Multi-head Latent Attention (MLA) is used in DeepSeek models to reduce KV cache memory by compressing key-value pairs into lower-dimensional latent representations. The kernel handles:
1. Compressed KV cache with latent vectors
2. Separate RoPE-encoded key components
3. Efficient attention computation with decompression

## Optimization 1: CUTLASS SM100 MLA Kernel
- Commit ID: (csrc/attention/mla/sm100_cutlass_mla_kernel.cu)
- Optimization type: Compute
- Summary: Implement MLA using CUTLASS 3.x with Blackwell-specific optimizations
- Detailed explanation:
  The SM100 MLA kernel leverages Blackwell's enhanced Tensor Cores and TMA for efficient latent attention computation. It uses warp-specialized execution for overlapping memory and compute operations.
- Code excerpt:
    ```cpp
    // SM100 MLA with TMA and warp specialization
    template <typename Element, int HeadDim, int LatentDim>
    struct SM100MLAKernel {
      using TileShape = Shape<_128, _128, _64>;
      
      // Warp-specialized mainloop
      using CollectiveMainloop = cutlass::gemm::collective::
          CollectiveMma<MainloopSm100TmaGmmaWarpSpecialized<...>>;
      
      // Custom epilogue for MLA reduction
      using CollectiveEpilogue = MLAReductionEpilogue<...>;
    };
    ```
- Evidence mapping:
  - "SM100 optimization" → `MainloopSm100TmaGmmaWarpSpecialized`
  - "Latent dimension" → `LatentDim` template parameter
  - "Custom reduction" → `MLAReductionEpilogue` for attention output

## Optimization 2: Triton MLA Implementation
- Commit ID: (vllm/v1/attention/backends/mla/triton_mla.py)
- Optimization type: Compute / Memory
- Summary: Triton-based MLA for flexible deployment across GPU architectures
- Detailed explanation:
  The Triton MLA kernel provides a portable implementation that works across NVIDIA and AMD GPUs. It handles the latent decompression and attention computation in a fused manner.
- Code excerpt:
    ```python
    @triton.jit
    def mla_attention_kernel(
        q_ptr, k_latent_ptr, k_rope_ptr, v_latent_ptr,
        output_ptr,
        q_head_dim, kv_lora_rank, qk_rope_head_dim,
        ...
    ):
        # Load query
        q = tl.load(q_ptr + ...)
        
        # Load compressed KV
        k_latent = tl.load(k_latent_ptr + ...)
        k_rope = tl.load(k_rope_ptr + ...)
        v_latent = tl.load(v_latent_ptr + ...)
        
        # Decompress key: k = concat(k_rope, W_k @ k_latent)
        k = decompress_key(k_latent, k_rope, W_k)
        
        # Compute attention scores
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Decompress value and compute output
        v = decompress_value(v_latent, W_v)
        output = tl.dot(tl.softmax(scores), v)
    ```
- Evidence mapping:
  - "Latent compression" → Separate `k_latent_ptr` and `k_rope_ptr`
  - "Decompression" → `decompress_key` and `decompress_value` functions
  - "Fused computation" → Attention computed after decompression

## Optimization 3: Sparse MLA for Long Sequences
- Commit ID: (vllm/v1/attention/backends/mla/flashmla_sparse.py)
- Optimization type: Compute
- Summary: Sparse attention patterns for efficient long-sequence MLA
- Detailed explanation:
  For very long sequences, sparse attention patterns reduce computation while maintaining quality. This optimization implements block-sparse MLA.
- Code excerpt:
    ```python
    @triton.jit
    def sparse_mla_kernel(
        q_ptr, k_latent_ptr, v_latent_ptr,
        block_mask_ptr,  # Sparse attention mask
        ...
    ):
        # Check if this block should be computed
        block_idx = tl.program_id(0)
        if not tl.load(block_mask_ptr + block_idx):
            return  # Skip masked blocks
        
        # Compute attention for non-masked blocks
        # ...
    ```
- Evidence mapping:
  - "Sparse pattern" → `block_mask_ptr` for block-level sparsity
  - "Early exit" → Skip computation for masked blocks
  - "Long sequence support" → Reduced computation for distant tokens
