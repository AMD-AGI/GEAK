Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: fused_qk_norm_rope_cache_quant

## Variant Context
- Input semantic type: Fused attention preprocessing (QK normalization, RoPE, KV caching, quantization)
- Datatype(s): BF16/FP16 input, FP8 quantized output
- Data representation: Per-token or per-tensor quantization
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel fuses multiple operations that are typically performed separately in attention preprocessing:
1. QK Normalization (RMSNorm on Q and K)
2. RoPE (Rotary Position Embedding) application
3. KV Cache storage
4. FP8 Quantization

The fusion eliminates intermediate memory accesses between these operations, significantly reducing memory bandwidth requirements.

## Optimization 1: Multi-Operation Fusion
- Commit ID: b8df81e7e
- Optimization type: Fusion
- Summary: Fused QK normalization, RoPE, KV caching, and quantization into a single kernel.

- Detailed explanation:
  The original implementation required 4 separate kernel launches:
  1. RMSNorm on Q and K tensors
  2. RoPE application
  3. KV cache write
  4. FP8 quantization
  
  Each kernel launch involves:
  - Kernel launch overhead
  - Reading input from global memory
  - Writing output to global memory
  
  The fused kernel performs all operations in a single pass:
  1. Load Q, K, V from global memory once
  2. Apply RMSNorm in registers
  3. Apply RoPE in registers
  4. Quantize to FP8 in registers
  5. Write to KV cache once

- Code excerpt:
    ```cpp
    // Fused kernel: qk_norm_rope_cache_quant
    // File: csrc/kernels/fused_qk_norm_rope_cache_quant.cu (637 lines)
    
    // Operations fused:
    // 1. QK Normalization (RMSNorm)
    // 2. RoPE (Rotary Position Embedding)
    // 3. KV Cache storage
    // 4. FP8 Quantization (per-token or per-tensor)
    ```

- Evidence mapping:
  - Multi-operation fusion → Single kernel file with 637 lines
  - Memory reduction → Single read/write pass
  - Commit message → "qk_norm_rope_cache_quant fusion"

## Optimization 2: Per-Token Quantization Support
- Commit ID: b8df81e7e
- Optimization type: Precision
- Summary: Added support for per-token quantization for better accuracy.

- Detailed explanation:
  The kernel supports both per-tensor and per-token quantization modes:
  - Per-tensor: Single scale factor for entire tensor (faster, less accurate)
  - Per-token: Separate scale factor per token (slower, more accurate)
  
  Per-token quantization computes the maximum absolute value for each token and uses it as the scale factor, providing better dynamic range utilization.

- Code excerpt:
    ```cpp
    // support per token quant
    ```

- Evidence mapping:
  - Per-token support → Commit message mentions "per token quant"
  - Accuracy improvement → Finer-grained quantization

## Optimization 3: CK FP8 Type Conversion
- Commit ID: b8df81e7e
- Optimization type: Precision / Compute
- Summary: Adopted CK (Composable Kernel) library's FP8 type conversion for efficient quantization.

- Detailed explanation:
  The kernel uses AMD's Composable Kernel library for FP8 type conversion, which provides:
  1. Hardware-optimized conversion routines
  2. Proper handling of special values (NaN, Inf)
  3. Configurable rounding modes

- Code excerpt:
    ```cpp
    // adopt ck's fp8 type convert
    ```

- Evidence mapping:
  - CK integration → Commit message mentions "ck's fp8 type convert"
  - Hardware optimization → Uses CK library routines

## Optimization 4: Head Dimension Reduction Optimization
- Commit ID: b8df81e7e
- Optimization type: Compute
- Summary: Optimized reduction to operate only over head dimension.

- Detailed explanation:
  For RMSNorm, the reduction (sum of squares) only needs to be computed over the head dimension, not the entire tensor. This optimization:
  1. Reduces the number of elements in the reduction
  2. Enables more efficient warp-level reductions
  3. Improves parallelism across heads

- Code excerpt:
    ```cpp
    // reduce over head_dim only
    ```

- Evidence mapping:
  - Dimension-specific reduction → Commit message mentions "reduce over head_dim only"
  - Efficiency improvement → Smaller reduction scope
