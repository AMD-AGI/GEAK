Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: fmha_fwd (Flash Multi-Head Attention Forward)

## Variant Context
- Input semantic type: Attention (Query, Key, Value matrices for transformer models)
- Datatype(s): FP8 (fp8_e4m3 / fp8_e5m2) with block-wise scaling
- Data representation: Block-scale quantization with per-block descale factors
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The FMHA forward kernel computes scaled dot-product attention: `softmax(Q @ K^T / sqrt(d)) @ V`. This variant adds FP8 block-scale quantization support, where Q, K, V tensors are stored in FP8 format with per-block scaling factors for dequantization. This enables memory bandwidth reduction while maintaining numerical accuracy through block-wise scaling.

## Optimization 1: FP8 Block Scale Quantization Support
- Commit ID: dd0b4294a
- Optimization type: precision / memory
- Summary: Added FP8 block-scale quantization for FMHA forward, enabling reduced memory bandwidth through 8-bit storage with per-block descale factors.

- Detailed explanation:
  The optimization introduces block-scale quantization where FP8 values are dequantized using per-block scale factors. This reduces memory bandwidth by 2x (FP8 vs FP16) while maintaining accuracy through block-wise scaling. The key innovation is computing the descale block index based on the current K/V position and applying the scale factor during the GEMM accumulation phase.

  For the softmax computation, a shift value is precomputed per row to handle the FP8 dynamic range:
  - OCP FP8: shift = 8.0f
  - FNUZ FP8: shift = 7.0f
  
  This shift is subtracted from the row maximum before exp2 computation to keep values in the representable FP8 range.

- Code excerpt:
    ```cpp
    // FP8 shift constants for block scale quantization
    static constexpr float OCP_FP8_SHIFT  = 8.0f;
    static constexpr float FNUZ_FP8_SHIFT = 7.0f;
    
    // Compute descale block index based on K/V position
    float k_descale = 1.0f;
    if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
    {
        // K and V share the same seqlen_k position within a block
        const index_t kv_idx = (kv_load_start + i_total_loops * kN0) / block_scale_size_kv;
        k_descale            = k_descale_ptr[kv_idx];
    }
    
    // Apply dequantization scale during GEMM accumulation
    auto s_acc_element_func_ = [&s_acc_element_func, k_descale]() {
        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
        {
            return s_acc_element_func * k_descale;
        }
        else
            return s_acc_element_func;
    }();
    
    // Precompute (m - shift) once per row for softmax
    auto validated_m = get_validated_m(m[i_idx]);
    auto row_max     = scale_s * validated_m;
    if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
    {
    #if CK_TILE_USE_OCP_FP8
        validated_m -= OCP_FP8_SHIFT; // for Bias/Alibi/SoftCap
        row_max -= OCP_FP8_SHIFT;     // for else branch
    #else
        validated_m -= FNUZ_FP8_SHIFT;
        row_max -= FNUZ_FP8_SHIFT;
    #endif
    }
    
    // Apply V descale during output accumulation
    float v_descale = 1.0f;
    if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
    {
        const index_t kv_idx = (kv_load_start + i_total_loops * kN0) / block_scale_size_kv;
        v_descale            = v_descale_ptr[kv_idx];
    }
    ```

- Evidence mapping:
  - "Block-scale quantization" → `block_scale_size_kv` parameter and `kv_idx` calculation
  - "Per-block descale factors" → `k_descale_ptr[kv_idx]` and `v_descale_ptr[kv_idx]` lookups
  - "FP8 dynamic range handling" → `OCP_FP8_SHIFT` and `FNUZ_FP8_SHIFT` constants
  - "Precompute shift per row" → `validated_m -= OCP_FP8_SHIFT` applied once before sweep

## Optimization 2: Kernel Arguments Structure for Block Scale
- Commit ID: dd0b4294a
- Optimization type: memory / launch
- Summary: Added dedicated kernel argument structures for block-scale quantization with stride information for efficient memory access patterns.

- Detailed explanation:
  The optimization introduces specialized kernel argument structures (`FmhaFwdBatchBlockScaleKargs`, `FmhaFwdGroupBlockScaleKargs`) that carry stride information for descale tensors. This enables efficient strided access to per-head and per-batch descale factors without runtime overhead.

- Code excerpt:
    ```cpp
    struct FmhaFwdCommonBlockScaleKargs : public FmhaFwdCommonQScaleKargs
    {
        ck_tile::index_t nhead_stride_q_descale;
        ck_tile::index_t nhead_stride_k_descale;
        ck_tile::index_t nhead_stride_v_descale;

        ck_tile::index_t block_scale_size_q;
        ck_tile::index_t block_scale_size_kv;
    };

    struct FmhaFwdBatchBlockScaleKargs : public FmhaFwdCommonBlockScaleKargs
    {
        ck_tile::index_t batch_stride_q_descale;
        ck_tile::index_t batch_stride_k_descale;
        ck_tile::index_t batch_stride_v_descale;
    };
    ```

- Evidence mapping:
  - "Stride information for descale tensors" → `nhead_stride_*_descale` and `batch_stride_*_descale` fields
  - "Block scale size parameters" → `block_scale_size_q` and `block_scale_size_kv` fields
