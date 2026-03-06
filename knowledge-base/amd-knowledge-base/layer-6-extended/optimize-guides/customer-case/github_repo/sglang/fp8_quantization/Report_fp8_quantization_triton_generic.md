Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: FP8 Quantization Kernels

## Variant Context
- Input semantic type: Activation and weight quantization for FP8 inference
- Datatype(s): FP8 (float8_e4m3fn, float8_e4m3fnuz), with FP32 scales
- Data representation: Per-token, per-tensor, or blockwise quantization
- Target architecture: Generic CUDA and ROCm

## Functionality
These kernels implement various FP8 quantization schemes for LLM inference:
1. Per-tensor quantization: Single scale for entire tensor
2. Per-token quantization: One scale per token (row)
3. Per-token-group (blockwise) quantization: Scales for groups of elements within each token

## Optimization 1: Fused SiLU-and-Mul-and-Quant Kernel
- Commit ID: df5192cff
- Optimization type: Fusion
- Summary: Fuses SiLU activation, element-wise multiplication, and FP8 quantization into a single kernel for MoE down projection input.

- Detailed explanation:
  In MoE layers, the gate-up projection output needs to be:
  1. Split into gate and up components
  2. Apply SiLU to gate: silu(gate)
  3. Multiply: silu(gate) * up
  4. Quantize to FP8 for down projection
  
  This optimization fuses all these operations, eliminating intermediate tensor allocations and memory traffic.

- Code excerpt:
    ```python
    # Fused kernel call
    if _MASKED_GEMM_FAST_ACT:
        down_input, down_input_scale = sglang_per_token_group_quant_8bit(
            x=gateup_output,
            dst_dtype=torch.float8_e4m3fn,
            group_size=scale_block_size,
            masked_m=masked_m,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            fuse_silu_and_mul=True,  # Enable fusion
            enable_v2=True,
        )
    ```

    Previous unfused version:
    ```python
    # Separate allocation and kernel call
    down_input = torch.empty((...), dtype=torch.float8_e4m3fn)
    down_input_scale = torch.empty((...), dtype=torch.float32)
    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        scale_block_size,
        masked_m,
        scale_ue8m0=...,
    )
    ```

- Evidence mapping:
  - "Fused operations" → `fuse_silu_and_mul=True` parameter
  - "Single kernel" → `sglang_per_token_group_quant_8bit` handles all operations
  - "No intermediate allocation" → output tensors created inside the function

## Optimization 2: Unified Per-Token Group Quantization Kernels
- Commit ID: b1b3f0b38
- Optimization type: Code / Compute
- Summary: Unifies multiple per-token group quantization kernel variants into a single configurable implementation.

- Detailed explanation:
  Previously, there were separate kernels for different quantization configurations (column-major vs row-major scales, different group sizes, etc.). This optimization consolidates them into a single kernel with compile-time configuration, reducing code duplication and enabling better optimization.

- Code excerpt:
    ```python
    @triton.jit
    def per_token_group_quant_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        M,
        N,
        group_size: tl.constexpr,
        column_major_scales: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # Unified implementation with constexpr configuration
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Load input block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], ...)
        
        # Compute scale per group
        abs_max = tl.max(tl.abs(x), axis=1)
        scale = abs_max / 448.0  # FP8 E4M3 max value
        
        # Quantize
        out = (x / scale[:, None]).to(tl.float8e4nv)
        
        # Store with configurable layout
        if column_major_scales:
            scale_offs = pid_n * M + offs_m
        else:
            scale_offs = offs_m * (N // group_size) + pid_n
        
        tl.store(out_ptr + ..., out, ...)
        tl.store(scale_ptr + scale_offs, scale, ...)
    ```

- Evidence mapping:
  - "Unified kernel" → single `per_token_group_quant_kernel` function
  - "Compile-time config" → `column_major_scales: tl.constexpr`
  - "Configurable layout" → `if column_major_scales:` branch

## Optimization 3: UE8M0 Scale Format Support
- Commit ID: b58c3c285
- Optimization type: Precision / Memory
- Summary: Adds support for UE8M0 (unsigned 8-bit exponent, 0-bit mantissa) scale format for DeepGEMM compatibility.

- Detailed explanation:
  DeepGEMM uses a specialized scale format (UE8M0) where scales are represented as pure powers of 2. This optimization adds support for this format, enabling efficient integration with DeepGEMM's FP8 GEMM kernels.

- Code excerpt:
    ```python
    @triton.jit
    def quantize_with_ue8m0_scale(
        x,
        scale_ue8m0: tl.constexpr,
    ):
        abs_max = tl.max(tl.abs(x))
        
        if scale_ue8m0:
            # UE8M0: scale is power of 2
            log2_scale = tl.ceil(tl.log2(abs_max / 448.0))
            scale = tl.exp2(log2_scale)
        else:
            # Standard FP32 scale
            scale = abs_max / 448.0
        
        return (x / scale).to(tl.float8e4nv), scale
    ```

- Evidence mapping:
  - "UE8M0 format" → `scale_ue8m0: tl.constexpr` parameter
  - "Power of 2 scale" → `tl.exp2(log2_scale)`
  - "Ceiling for safety" → `tl.ceil(tl.log2(...))` ensures no overflow

## Optimization 4: TMA-Aligned Scale Layout
- Commit ID: c268c11c7
- Optimization type: Memory
- Summary: Aligns scale tensor layout for efficient TMA (Tensor Memory Accelerator) access in subsequent GEMM operations.

- Detailed explanation:
  When scales are used in TMA-based GEMM kernels, they need specific alignment and layout. This optimization ensures the quantization kernel produces scales in a TMA-friendly format, avoiding costly layout transformations before GEMM.

- Code excerpt:
    ```python
    def sglang_per_token_group_quant_8bit(
        x,
        dst_dtype,
        group_size,
        scale_tma_aligned=False,  # New parameter
        ...
    ):
        if scale_tma_aligned:
            # Align scale tensor for TMA
            aligned_size = ((x.shape[-1] // group_size + 127) // 128) * 128
            scale = torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[: x_shape[-2], :]
        else:
            scale = torch.empty(...)
    ```

- Evidence mapping:
  - "TMA alignment" → `scale_tma_aligned=True` parameter
  - "128-byte alignment" → `((... + 127) // 128) * 128`
  - "Layout transpose" → `.transpose(-1, -2)` for column-major access

## Optimization 5: JIT Per-Tensor Quantization
- Commit ID: 0fee6bc63
- Optimization type: Compilation / Latency
- Summary: Uses JIT compilation for per-tensor FP8 quantization to reduce kernel launch overhead.

- Detailed explanation:
  For per-tensor quantization which is simpler than per-token-group, a JIT-compiled CUDA kernel provides lower latency than Triton. This optimization adds a JIT path for per-tensor quantization used in weight quantization and simple activation quantization.

- Code excerpt:
    ```python
    # JIT kernel for per-tensor quantization
    from sglang.jit_kernel import per_tensor_quant_fp8_kernel

    def per_tensor_quant_fp8(x):
        scale = x.abs().max() / 448.0
        out = per_tensor_quant_fp8_kernel(x, scale)
        return out, scale
    ```

- Evidence mapping:
  - "JIT compilation" → import from `sglang.jit_kernel`
  - "Per-tensor scale" → single `scale = x.abs().max() / 448.0`
  - "Lower overhead" → JIT kernel vs Triton for simple operation
