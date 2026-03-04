---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: ROCM MLA Decode with RoPE Kernel

## Variant Context
- Input semantic type: Multi-head Latent Attention (MLA) decode with Rotary Position Embedding
- Datatype(s): FP16, BF16, FP32 accumulation
- Data representation: Compressed KV cache with latent representations (kv_lora_rank + qk_rope_head_dim)
- Target architecture: AMD ROCm (gfx90a, gfx942 - MI200/MI300 series)

## Functionality
This kernel implements fused MLA (Multi-head Latent Attention) decode with integrated RoPE (Rotary Position Embedding) for DeepSeek V3 and similar models on AMD GPUs. MLA uses a compressed KV cache where:
- K is stored as [KV_compressed; K_PE] with dimensions (kv_lora_rank + qk_rope_head_dim)
- V is stored as [KV_compressed] with dimension kv_lora_rank
- RoPE is applied to the K_PE portion during attention computation

## Optimization 1: Fused RoPE Application in Attention
- Commit ID: 6ce9dbe82
- Optimization type: Fusion
- Summary: Fuses RoPE (Rotary Position Embedding) application directly into the attention kernel, eliminating a separate RoPE kernel launch.

- Detailed explanation:
  Instead of applying RoPE as a separate preprocessing step, this kernel applies RoPE on-the-fly during attention computation. The kernel:
  1. Loads the cos/sin cache based on token positions
  2. Applies RoPE to Q_PE and K_PE during the attention score computation
  3. Supports both NeoX-style and standard RoPE implementations
  
  This fusion eliminates memory round-trips for the RoPE-transformed tensors.

- Code excerpt:
    ```python
    @triton.jit
    def _fwd_grouped_kernel_stage1_rope(
        Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
        K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
        V_buffer,  # Holds [KV], b*s x (c)
        cos_sin_cache,  # max_seq_len x (rotary_dim * 2)
        positions,  # sequence positions
        # ...
        USE_ROPE: tl.constexpr,
        IS_NEOX_STYLE: tl.constexpr,
    ):
        # Load position and cos/sin values
        if USE_ROPE:
            pos = tl.load(positions + cur_batch * stride_positions_b)
            cos = tl.load(
                cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary,
                mask=mask_rotary,
                other=1.0,
            )
            sin = tl.load(
                cos_sin_cache + pos * stride_cos_sin_cache_s + offs_rotary + rotary_dim // 2,
                mask_rotary,
                other=0.0,
            )
            
            # Apply RoPE: x_rot = x * cos + rotate(x) * sin
            if IS_NEOX_STYLE:
                # NeoX style: split in half and rotate
                offs_qk_rot_r = kv_lora_rank + (
                    (tl.arange(0, BLOCK_R) + (rotary_dim // 2)) % rotary_dim
                )
                mask_rotate = tl.arange(0, BLOCK_R) < (rotary_dim // 2)
            else:
                # Standard style: interleaved rotation
                offs_qk_rot_r = (
                    kv_lora_rank
                    + (((tl.arange(0, BLOCK_R) + 1) % 2) * 2)
                    - 1
                    + tl.arange(0, BLOCK_R)
                )
                mask_rotate = tl.arange(0, BLOCK_R) % 2 < 1
    ```

- Evidence mapping:
  - "Fused RoPE" → RoPE applied within attention kernel, not separate
  - "cos/sin cache loading" → `tl.load(cos_sin_cache + pos * stride_cos_sin_cache_s + ...)`
  - "NeoX style support" → `IS_NEOX_STYLE` constexpr controls rotation pattern
  - "Position-based indexing" → `pos = tl.load(positions + cur_batch * stride_positions_b)`

## Optimization 2: Grouped Query Attention with Block-wise Head Processing
- Commit ID: 6ce9dbe82
- Optimization type: Parallelism
- Summary: Processes multiple query heads per thread block for efficient GQA (Grouped Query Attention) computation.

- Detailed explanation:
  For GQA where multiple query heads share the same KV head, this kernel processes BLOCK_H query heads together in a single thread block. This improves:
  1. KV cache reuse - same KV data used for multiple query heads
  2. Reduced memory bandwidth - KV loaded once, used multiple times
  3. Better occupancy - more work per thread block

- Code excerpt:
    ```python
    # Block-wise head processing for GQA
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # Load Q for multiple heads at once
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)
    ```

- Evidence mapping:
  - "Multiple heads per block" → `cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)`
  - "Head masking" → `mask_h = cur_head < q_head_num`
  - "Batched Q load" → `q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & ...)`

## Optimization 3: Separated NOPE and PE Components
- Commit ID: 6ce9dbe82
- Optimization type: Memory Access
- Summary: Separately handles the NOPE (non-positional) and PE (positional embedding) components of Q and K for efficient memory access.

- Detailed explanation:
  MLA uses a decomposed representation where Q = [Q_NOPE; Q_PE] and K = [KV_compressed; K_PE]. This kernel:
  1. Loads Q_NOPE and Q_PE with separate offset calculations
  2. Computes attention scores as: score = Q_NOPE @ KV_compressed.T + Q_PE @ K_PE.T
  3. Uses different block sizes for the compressed (BLOCK_C) and PE (BLOCK_R) dimensions

- Code excerpt:
    ```python
    # Separate offsets for NOPE and PE components
    offs_c = tl.arange(0, BLOCK_C)  # For compressed KV
    offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # For K_PE

    # Load Q components separately
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]
    off_q_pe = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]

    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)
    q_pe = tl.load(Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0)

    # Compute attention with both components
    # att = q @ kv.T + q_pe @ k_pe.T (after RoPE)
    ```

- Evidence mapping:
  - "Separate offsets" → `offs_c` for compressed, `offs_qk_r` for PE starting at `kv_lora_rank`
  - "Separate loads" → `q` and `q_pe` loaded independently
  - "Different block sizes" → `BLOCK_C` vs `BLOCK_R` constexprs

## Optimization 4: Split-KV with Online Softmax
- Commit ID: 6ce9dbe82
- Optimization type: Memory / Compute
- Summary: Uses split-KV approach with online softmax for memory-efficient long-context attention.

- Detailed explanation:
  Similar to the standard decode attention, this kernel splits the KV sequence into NUM_KV_SPLITS chunks processed in parallel. Each split computes partial attention with online softmax, and results are merged in a second stage. This enables processing of long contexts without materializing the full attention matrix.

- Code excerpt:
    ```python
    # Split-KV processing
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    # Online softmax accumulation
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        # Compute attention scores
        att_value = tl.sum(q[:, :, None] * kv[None, :, :], 1)
        att_value += tl.sum(q_pe_rope[:, :, None] * k_pe_rope[None, :, :], 1)
        att_value *= sm_scale
        
        # Online softmax update
        n_e_max = tl.maximum(tl.max(att_value, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(att_value - n_e_max[:, None])
        acc = acc * re_scale[:, None] + tl.sum(p[:, :, None] * v[None, :, :], 1)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    ```

- Evidence mapping:
  - "Split-KV" → `split_kv_start`, `split_kv_end` based on `NUM_KV_SPLITS`
  - "Online softmax" → `n_e_max`, `re_scale`, running `e_max` and `e_sum`
  - "Incremental accumulation" → `acc = acc * re_scale[:, None] + ...`
