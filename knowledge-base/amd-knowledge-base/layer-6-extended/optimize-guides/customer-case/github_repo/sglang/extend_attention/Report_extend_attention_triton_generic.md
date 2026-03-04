---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Triton Extend Attention Kernel

## Variant Context
- Input semantic type: Attention computation during prefill/extend phase (multiple tokens per sequence)
- Datatype(s): FP16, BF16, FP32 accumulation
- Data representation: Paged KV cache with custom mask support
- Target architecture: Generic CUDA and ROCm with platform-specific tuning

## Functionality
This kernel implements attention computation for the prefill/extend phase of LLM inference, where multiple new tokens are processed and attend to both the existing KV cache and the newly added tokens. It supports custom attention masks for various attention patterns (causal, sliding window, etc.).

## Optimization 1: AMD ROCm Transposed Store
- Commit ID: 634a3561a
- Optimization type: Memory Access Pattern
- Summary: Uses transposed store operations on AMD GPUs to improve memory write coalescing.

- Detailed explanation:
  AMD GPUs have different memory subsystem characteristics compared to NVIDIA GPUs. This optimization transposes the output tensor during the store operation on ROCm platforms. By storing in transposed order, the kernel achieves better memory coalescing for the specific memory access patterns of AMD's memory controllers.

- Code excerpt:
    ```python
    # Conditional transposed store for AMD
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )
    ```

    Platform detection and configuration:
    ```python
    def extend_attention_fwd(...):
        if is_hip_:
            BLOCK_M, BLOCK_N = (32, 32)
            num_warps = 2
        # ...
        _fwd_kernel[grid](
            # ...
            STORE_TRANSPOSE=is_hip_,
            # ...
        )
    ```

- Evidence mapping:
  - "Transposed store" → `tl.store(O_Extend + offs_o.T, (acc / deno[:, None]).T, ...)`
  - "AMD-specific" → `STORE_TRANSPOSE=is_hip_`
  - "Transposed mask" → `mask=(mask_m[:, None] & mask_dv[None, :]).T`

## Optimization 2: AMD-Optimized Block Sizes and Warp Count
- Commit ID: 634a3561a
- Optimization type: Launch Configuration
- Summary: Uses smaller block sizes and fewer warps on AMD GPUs for better occupancy and performance.

- Detailed explanation:
  AMD GPUs have different wavefront sizes (64 threads vs NVIDIA's 32-thread warps) and different register file characteristics. This optimization reduces the block size from 64x64 to 32x32 and the warp count from 4 to 2 on ROCm platforms, which better matches AMD's hardware characteristics and improves occupancy.

- Code excerpt:
    ```python
    if is_hip_:
        BLOCK_M, BLOCK_N = (32, 32)
        num_warps = 2
    else:
        if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
            # Hopper-specific configuration
            BLOCK_M, BLOCK_N = (128, 64) if Lq <= 128 else (64, 64)
            num_stages = 2
            num_warps = 4 if Lq <= 128 else 8
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 64)
            num_stages = 1
            num_warps = 4
    ```

- Evidence mapping:
  - "Smaller blocks for AMD" → `BLOCK_M, BLOCK_N = (32, 32)` vs `(64, 64)` for CUDA
  - "Fewer warps" → `num_warps = 2` vs `num_warps = 4` for CUDA
  - "Platform detection" → `if is_hip_:`

## Optimization 3: Sliding Window Attention Skip
- Commit ID: 0475448ee
- Optimization type: Compute
- Summary: Skips attention computation for tokens outside the sliding window, reducing unnecessary computation.

- Detailed explanation:
  For models with sliding window attention (like Mistral), tokens beyond the window size don't need to attend to earlier tokens. This optimization detects when a block of queries is entirely outside the attention window for a block of keys and skips the computation entirely.

- Code excerpt:
    ```python
    # Skip computation for tokens outside sliding window
    if sliding_window > 0:
        # Check if this KV block is within the sliding window
        kv_block_end = start_n + BLOCK_N
        q_block_start = cur_seq_len - extend_seq_len + start_m
        
        # Skip if KV block is entirely before the sliding window
        if kv_block_end < q_block_start - sliding_window:
            continue
    ```

- Evidence mapping:
  - "Sliding window check" → comparison of `kv_block_end` with `q_block_start - sliding_window`
  - "Skip computation" → `continue` statement bypasses the attention computation
  - "Reduces FLOPs" → avoids computing attention scores that would be masked to zero

## Optimization 4: SM120 Shared Memory Size Handling
- Commit ID: 632c7afa8
- Optimization type: Memory / Launch Configuration
- Summary: Adds block size logic to handle the larger shared memory available on SM120 (Blackwell) GPUs.

- Detailed explanation:
  Blackwell GPUs (SM120) have significantly more shared memory per SM. This optimization adjusts the block size selection logic to take advantage of the increased shared memory, allowing for larger tile sizes that can improve performance through better data reuse.

- Code excerpt:
    ```python
    # Block size selection considering SM120 shared memory
    if is_cuda_available and CUDA_CAPABILITY[0] >= 12:
        # Larger blocks possible with more shared memory
        BLOCK_M, BLOCK_N = (128, 128) if Lq <= 128 else (64, 128)
        num_stages = 3
    ```

- Evidence mapping:
  - "SM120 detection" → `CUDA_CAPABILITY[0] >= 12`
  - "Larger block sizes" → `(128, 128)` compared to `(128, 64)` on Hopper

## Optimization 5: Deterministic Mode with Single-Stage Kernel
- Commit ID: 4fff1ec1d
- Optimization type: Compute / Precision
- Summary: Adds a deterministic mode using a single-stage Triton kernel for reproducible results.

- Detailed explanation:
  Multi-stage pipelining in Triton can introduce non-determinism due to floating-point operation reordering. This optimization provides a deterministic mode that uses `num_stages=1` to ensure reproducible results, which is important for debugging and certain production requirements.

- Code excerpt:
    ```python
    # Deterministic mode configuration
    if deterministic_mode:
        num_stages = 1  # Single stage for determinism
    ```

- Evidence mapping:
  - "Single stage" → `num_stages = 1`
  - "Deterministic execution" → eliminates pipeline-induced non-determinism
