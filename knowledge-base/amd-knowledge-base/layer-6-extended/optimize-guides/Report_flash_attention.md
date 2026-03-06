Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: flash_attention_fwd (FMHA V3)

## Variant Context
- Input semantic type: Multi-Head Attention (forward pass)
- Datatype(s): BF16/FP16 (Q, K, V), FP8 (optional quantized KV cache)
- Data representation: Dense or paged KV cache with vectorized layout
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel implements Flash Attention forward pass for transformer models. It computes scaled dot-product attention with memory-efficient tiling to avoid materializing the full attention matrix. The implementation supports various features including causal masking, variable sequence lengths, grouped-query attention (GQA), and paged KV cache.

## Optimization 1: Vectorized KV Cache Layout
- Commit ID: 93903b141
- Optimization type: Memory
- Summary: Implemented 5D vectorized KV cache layout with x=8 swizzling for optimal memory access patterns.

- Detailed explanation:
  The optimization changes the KV cache memory layout from a simple 4D layout to a 5D vectorized layout:
  - Old: `[num_blocks, num_kv_heads, block_size, head_dim]`
  - New: `[num_blocks, num_kv_heads, head_dim/8, block_size, 8]` for K
  - New: `[num_blocks, num_kv_heads, block_size/8, head_dim, 8]` for V
  
  This layout enables:
  1. Coalesced 128-bit (8 × FP16) memory accesses
  2. Better cache line utilization
  3. Reduced bank conflicts in shared memory

- Code excerpt:
    ```cpp
    // Vectorized KV cache layout with x=8 swizzling
    // K: [num_blocks, num_kv_heads, head_size/8, block_size, 8]
    // V: [num_blocks, num_kv_heads, block_size/8, head_size, 8]
    
    // Layout enforcement in host bindings
    // Added strict checks for 5D vectorized KV layout (swizzled x=8)
    ```

- Evidence mapping:
  - 5D layout → `[num_blocks, num_kv_heads, head_size/8, block_size, 8]`
  - x=8 swizzling → Last dimension of size 8 for vectorized access
  - Commit message → "enforce vectorized KV layout"

## Optimization 2: vLLM/SGLang Block Table Support
- Commit ID: 93903b141
- Optimization type: Memory / Scheduling
- Summary: Added support for both vLLM-style 2D block tables and SGLang-style 1D page tables.

- Detailed explanation:
  The kernel now supports two different block table formats:
  1. vLLM: 2D block table `[batch_size, max_num_blocks_per_seq]`
  2. SGLang: 1D page table with separate page indices
  
  The kernel automatically selects the appropriate trait (`VLLM_BLOCK_TABLE_2D` or `SGLANG_PAGE_TABLE_1D`) based on input arguments, enabling compatibility with both inference frameworks.

- Code excerpt:
    ```cpp
    // CodeGen: Automatically select VLLM_BLOCK_TABLE_2D or SGLANG_PAGE_TABLE_1D 
    // trait based on input arguments
    
    // API: Added block_table and seqlen_k arguments to python/C++ interfaces
    ```

- Evidence mapping:
  - vLLM support → `VLLM_BLOCK_TABLE_2D` trait
  - SGLang support → `SGLANG_PAGE_TABLE_1D` trait
  - Automatic selection → CodeGen based on input arguments

## Optimization 3: Instruction Alignment for Causal Mode (HD192)
- Commit ID: a6ee2e699
- Optimization type: Compute
- Summary: Optimized instruction alignment for head dimension 192 in causal attention mode.

- Detailed explanation:
  For head dimension 192 (used in models like Llama-3), the kernel optimizes instruction scheduling in causal mode. This involves aligning MFMA (Matrix Fused Multiply-Add) instructions to avoid pipeline stalls and maximize throughput.

- Code excerpt:
    ```
    # Updated binary files for HD192 causal mode:
    # - fwd_hd192x128_bf16_causal_rtna.co
    # - fwd_hd192x128_bf16_causal_rtne.co
    # - fwd_hd192x128_bf16_causal_rtz.co
    # (and their _group variants)
    ```

- Evidence mapping:
  - HD192 optimization → File names contain `hd192x128`
  - Causal mode → File names contain `causal`
  - Multiple rounding modes → `rtna`, `rtne`, `rtz` variants

## Optimization 4: Sink Token Support (GPT-OSS)
- Commit ID: 78db92192
- Optimization type: Compute / Memory
- Summary: Added sink token support for streaming/infinite context scenarios.

- Detailed explanation:
  Sink tokens are special tokens at the beginning of the sequence that are always attended to, even in sliding window attention. This enables:
  1. Streaming inference with bounded memory
  2. Infinite context length support
  3. Better handling of important initial tokens

- Code excerpt:
    ```cpp
    // Add sink_size parameter for mha_fwd and varlen_mha_fwd api
    // Enable gptoss_sink feature
    ```

- Evidence mapping:
  - Sink token support → `sink_size` parameter
  - GPT-OSS → Commit title "Enable gptoss_sink"

## Optimization 5: FP8 KV Cache with Per-Tensor Quantization
- Commit ID: 6bdd064f0
- Optimization type: Memory / Precision
- Summary: Added FP8 support for batch prefill with per-tensor quantization for reduced memory bandwidth.

- Detailed explanation:
  The kernel supports FP8 (E4M3) quantized KV cache with per-tensor scaling factors. This reduces memory bandwidth requirements by 50% compared to FP16/BF16, enabling:
  1. Larger batch sizes
  2. Longer context lengths
  3. Better memory efficiency

- Code excerpt:
    ```cpp
    // Add FP8 support for batch_prefill with per-tensor quantization
    ```

- Evidence mapping:
  - FP8 support → Commit title mentions "FP8 support"
  - Per-tensor quantization → Single scale factor per tensor
