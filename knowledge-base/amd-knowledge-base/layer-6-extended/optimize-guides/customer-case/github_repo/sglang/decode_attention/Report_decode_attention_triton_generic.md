Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Triton Decode Attention Kernel

## Variant Context
- Input semantic type: Attention computation during LLM decode phase (single token per sequence)
- Datatype(s): FP16, BF16, FP32 accumulation
- Data representation: Paged KV cache with configurable page size
- Target architecture: Generic (CUDA and ROCm), with architecture-specific tuning

## Functionality
This kernel implements memory-efficient attention for the decode phase of LLM inference. It uses a split-KV approach where the KV sequence is divided into splits that are processed in parallel, followed by a reduction phase. This is optimized for the decode scenario where each sequence generates one token at a time but may have long context lengths.

## Optimization 1: Dynamic KV Splits Based on Workload
- Commit ID: c0e9a36c5
- Optimization type: Scheduling / Load Balancing
- Summary: Dynamically adjusts the number of KV splits per batch based on sequence lengths and GPU core count, replacing the static split configuration.

- Detailed explanation:
  Previously, the number of KV splits was a static configuration parameter. This optimization introduces a Triton kernel that computes the optimal number of splits per batch element based on:
  1. The ratio between max and min sequence lengths in the batch
  2. The number of GPU cores available
  3. The batch size and number of attention heads
  
  This allows better GPU utilization for batches with varying sequence lengths, avoiding over-splitting for short sequences and under-splitting for long sequences.

- Code excerpt:
    ```python
    @triton.jit
    def get_num_kv_splits_triton(
        num_kv_splits_ptr,
        seq_lens_ptr,
        bs,
        num_head,
        num_kv_head,
        max_kv_splits,
        device_core_count,
        MAX_BS: tl.constexpr,
    ):
        offs_b = tl.arange(0, MAX_BS)
        mask_b = offs_b < bs

        seq_lens = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=0)
        max_seq_len = tl.max(seq_lens)
        seq_lens = tl.load(seq_lens_ptr + offs_b, mask=mask_b, other=max_seq_len)
        min_seq_len = tl.min(seq_lens)
        
        # Avoid over-splitting when sequences have similar lengths
        if max_seq_len * 8 < min_seq_len * 10:
            min_seq_len = max_seq_len
        max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
        kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

        # Scale splits with sequence length logarithmically
        ext_seq_len = tl.cast(tl.cdiv(max_seq_len, 256), tl.float32)
        ext_device_core_count = device_core_count * tl.maximum(
            tl.cast(tl.ceil(tl.log2(ext_seq_len)), tl.int32), 1
        )
        
        # Compute grid size for GQA
        block_h, num_kv_group = 16, num_head // num_kv_head
        if num_kv_group == 1:
            bh_grid = bs * num_head
        else:
            block_h = tl.minimum(block_h, num_kv_group)
            bh_grid = bs * tl.cdiv(num_head, block_h)
        max_kv_splits_2 = tl.minimum(tl.cdiv(ext_device_core_count, bh_grid), max_kv_splits)
        
        num_kv_splits = tl.maximum(
            tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
        )
        tl.store(num_kv_splits_ptr + offs_b, num_kv_splits, mask=mask_b)
    ```

- Evidence mapping:
  - "Dynamic splits based on sequence length ratio" → `max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)`
  - "GPU core count consideration" → `ext_device_core_count = device_core_count * tl.maximum(...)`
  - "Logarithmic scaling with sequence length" → `tl.ceil(tl.log2(ext_seq_len))`
  - "Per-batch element splits" → `tl.store(num_kv_splits_ptr + offs_b, num_kv_splits, mask=mask_b)`

## Optimization 2: Fused Flash Decoding with Online Softmax
- Commit ID: 7dc66fcb4
- Optimization type: Memory / Compute Fusion
- Summary: Replaces two-stage attention (compute scores, then softmax+reduce) with fused flash decoding that computes attention incrementally with online softmax.

- Detailed explanation:
  The original implementation had two separate kernels:
  1. Stage 1: Compute Q*K attention scores and store to global memory
  2. Stage 2: Load scores, apply softmax, multiply by V
  
  The optimized version fuses these into a single kernel using online softmax (log-sum-exp trick), eliminating the intermediate global memory write of attention scores. This significantly reduces memory bandwidth requirements.

- Code excerpt:
    ```python
    # Online softmax accumulation within the kernel
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            # Load K and compute QK
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale
            
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))
            
            # Load V
            v = tl.load(V_Buffer + offs_buf_v, ...)
            
            # Online softmax update
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)
            
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max
    ```

- Evidence mapping:
  - "Online softmax" → `n_e_max = tl.maximum(tl.max(qk, 0), e_max)` and `re_scale = tl.exp(e_max - n_e_max)`
  - "Fused V accumulation" → `acc += tl.sum(p[:, None] * v, 0)` happens in same loop as softmax
  - "No intermediate storage" → attention scores `qk` are consumed immediately, not stored to global memory
  - "Rescaling for numerical stability" → `acc *= re_scale` maintains correct softmax normalization

## Optimization 3: Separated Attention Output and LSE Storage
- Commit ID: c0e9a36c5
- Optimization type: Memory Layout
- Summary: Separates the attention output and log-sum-exp (LSE) values into different tensors instead of packing them together.

- Detailed explanation:
  Previously, the attention output and LSE were packed into a single tensor with shape `[bs, num_head, num_kv_splits, v_head_dim + 1]`. This optimization separates them into:
  - `att_out`: `[bs, num_head, num_kv_splits, v_head_dim]`
  - `att_lse`: `[bs, num_head, num_kv_splits]`
  
  This improves memory access patterns during the reduction phase and allows for more efficient vectorized loads.

- Code excerpt:
    ```python
    # Separated storage allocation
    attn_logits = [
        torch.empty(
            (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        ),
        torch.empty(
            (bs, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        ),
    ]
    ```

    In the kernel:
    ```python
    # Store output and LSE separately
    offs_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_dv
    )
    tl.store(Att_Out + offs_mid_o, acc, mask=mask_dv)
    
    offs_mid_o_1 = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + split_kv_id * stride_mid_os
    ) // Lv
    tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum))
    ```

- Evidence mapping:
  - "Separated tensors" → Two separate `torch.empty` calls with different shapes
  - "LSE stored separately" → `tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum))`
  - "Improved memory layout" → LSE tensor has shape `[bs, num_head, num_kv_splits]` without v_head_dim

## Optimization 4: Minimum Block KV Size for Split Alignment
- Commit ID: c0e9a36c5
- Optimization type: Scheduling
- Summary: Introduces a minimum block size for KV splits to ensure proper alignment and avoid inefficient small splits.

- Detailed explanation:
  When computing the KV length per split, the optimization ensures that each split processes at least `MIN_BLOCK_KV` (32) tokens. This prevents scenarios where very small splits would lead to poor GPU utilization due to thread divergence or inefficient memory access patterns.

- Code excerpt:
    ```python
    _MIN_BLOCK_KV = 32

    # In the kernel:
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    ```

- Evidence mapping:
  - "Minimum block size constant" → `_MIN_BLOCK_KV = 32`
  - "Aligned split size" → `tl.cdiv(..., MIN_BLOCK_KV) * MIN_BLOCK_KV` rounds up to multiple of 32
  - "Prevents small splits" → ensures each split has at least 32 tokens to process
