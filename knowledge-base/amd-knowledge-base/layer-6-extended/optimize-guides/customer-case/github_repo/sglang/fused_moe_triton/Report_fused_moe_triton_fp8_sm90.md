Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Fused MoE Triton Kernel

## Variant Context
- Input semantic type: Mixture of Experts (MoE) token routing and expert computation
- Datatype(s): FP8 (float8_e4m3fn), BF16, FP16
- Data representation: Blockwise quantized FP8 with per-block scales
- Target architecture: SM90 (Hopper), with specific optimizations for H20 GPUs

## Functionality
This kernel implements fused Mixture of Experts computation combining token routing, expert selection, and matrix multiplication in a single kernel. It handles the gate projection (up projection) and down projection for MoE layers commonly used in models like DeepSeek V3, Mixtral, and other MoE-based LLMs.

## Optimization 1: SwapAB Matrix Transposition for SM90
- Commit ID: 67b61a4e8
- Optimization type: Compute / Memory Access Pattern
- Summary: Swaps A and B matrices in the GEMM computation to improve memory access patterns on SM90 GPUs, specifically optimized for H20 GPUs.

- Detailed explanation: 
  The SwapAB optimization transposes the A and B matrices before the dot product computation and transposes the result back. This changes the memory access pattern to be more favorable for the SM90 tensor core layout. The optimization is conditionally enabled based on:
  1. Running on H20 GPU (not H100/H200)
  2. SM90 architecture support
  3. Block size constraints: BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64
  
  By swapping the matrices, the kernel achieves better utilization of the tensor cores and improved memory coalescing for the specific block sizes used in MoE workloads.

- Code excerpt:
    ```python
    @functools.lru_cache(maxsize=8)
    def should_enable_swap_ab(
        BLOCK_SIZE_M: int,
        BLOCK_SIZE_N: int,
    ) -> bool:
        if not _is_cuda:
            return False

        @functools.lru_cache(maxsize=1)
        def is_h20_device_and_sm90_supported():
            device_name = get_device_name()
            is_h20_device = (
                device_name and "H20" in device_name and "H200" not in device_name
            )
            return is_h20_device and is_sm90_supported()

        return (
            is_h20_device_and_sm90_supported() and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64
        )
    ```

    Inside the kernel:
    ```python
    # Swap A and B matrices for better memory access
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # During computation:
    if swap_ab:
        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
        a_scale, b_scale = b_scale, a_scale
    
    # After accumulation:
    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))
    ```

- Evidence mapping:
  - "Swaps A and B matrices" → `a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))`
  - "Transposes accumulator shape" → `accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), ...)` vs `(BLOCK_SIZE_M, BLOCK_SIZE_N)`
  - "H20-specific optimization" → `"H20" in device_name and "H200" not in device_name`
  - "Block size constraints" → `BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64`

## Optimization 2: TMA (Tensor Memory Accelerator) for Down Projection
- Commit ID: d2b8c4123
- Optimization type: Memory / Data Loading
- Summary: Adds TMA (Tensor Memory Accelerator) support for loading A and B matrices in the down projection kernel, leveraging Hopper's hardware-accelerated memory transfers.

- Detailed explanation:
  TMA is a hardware feature on Hopper GPUs that provides asynchronous, hardware-managed tensor data movement. This optimization:
  1. Uses TensorDescriptor from Triton to describe the tensor layout
  2. Enables TMA-based loading for both A (activations) and B (weights) matrices
  3. Provides sorted output option (c_sorted) to avoid additional reordering
  4. The TMA path is particularly beneficial for the down projection where the input is the intermediate activation after SiLU

- Code excerpt:
    ```python
    from triton.tools.tensor_descriptor import TensorDescriptor

    # TMA descriptor creation
    if a_use_tma:
        a_desc = TensorDescriptor(
            A, A.shape, A.stride(), [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]]
        )
    else:
        a_desc = None
    if b_use_tma:
        b_desc = TensorDescriptor(
            B,
            B.shape,
            B.stride(),
            [1, config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]],
        )
    else:
        b_desc = None
    ```

    Inside the kernel:
    ```python
    # TMA-based loading vs pointer-based loading
    if a_desc is not None:
        a = a_desc.load([start_offs_m, k_start])
    elif even_Ks:
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
    
    if b_desc is not None:
        b = (
            b_desc.load([off_experts_i32, start_offs_n, k_start])
            .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
            .T
        )
    elif even_Ks:
        b = tl.load(b_ptrs)
    ```

- Evidence mapping:
  - "TMA descriptor creation" → `TensorDescriptor(A, A.shape, A.stride(), [config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]])`
  - "Hardware-accelerated loading" → `a_desc.load([start_offs_m, k_start])` replaces manual pointer arithmetic
  - "3D tensor descriptor for weights" → `[1, config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]]` handles expert dimension

## Optimization 3: Skip Activation for Masked Experts
- Commit ID: 061f41aff
- Optimization type: Compute
- Summary: Skips SiLU/GELU activation computation for experts that are masked out (not selected by the router).

- Detailed explanation:
  In MoE models with expert parallelism, some experts may not be present on the current rank. Previously, the kernel would still compute activations for these masked experts. This optimization adds a filter_expert flag that allows skipping the zero-write and activation computation for experts marked as -1 (not present).

- Code excerpt:
    ```python
    # Conditional expert filtering
    if filter_expert and off_experts == -1:
        # Skip computation for masked experts
        write_zeros_to_output(...)
        return
    ```

- Evidence mapping:
  - "Conditional skip" → `if filter_expert and off_experts == -1`
  - "Early return" → avoids unnecessary computation for non-local experts

## Optimization 4: Optimized Scale Broadcasting for Blockwise Quantization
- Commit ID: d2b8c4123
- Optimization type: Compute
- Summary: Optimizes the scale multiplication for blockwise FP8 quantization by reducing redundant broadcasts.

- Detailed explanation:
  When BLOCK_SIZE_N equals or is smaller than group_n (the quantization group size), the scale broadcast can be simplified. Instead of computing `a_scale[:, None] * b_scale[None, :]` which creates a 2D broadcast, the optimization uses `a_scale[:, None] * b_scale` when the scales are uniform across the block.

- Code excerpt:
    ```python
    if BLOCK_SIZE_N > group_n:
        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
    else:
        accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
    ```

- Evidence mapping:
  - "Conditional broadcast optimization" → `if BLOCK_SIZE_N > group_n` determines broadcast pattern
  - "Reduced broadcast" → `(a_scale[:, None] * b_scale)` vs `a_scale[:, None] * b_scale[None, :]`
