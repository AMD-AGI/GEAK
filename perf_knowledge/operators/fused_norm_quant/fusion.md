---
title: fused_norm_quant — fusion neighbors
kind: technique
operator: fused_norm_quant
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, int8, mxfp4]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/fused_qk_rmsnorm_group_quant.py
  - /sgl-workspace/aiter/aiter/ops/fused_qk_norm_rope_cache_quant.py
---

# fused_norm_quant — fusion

This op IS a fusion (norm+quant). It sits at the seam between the norm family and the quant family, and
stacks both ways:

## 1. ← norm side: which norm is fused
- plain rmsnorm + quant → `rmsnorm2d_fwd_with_dynamicquant` / `rmsnorm2d_fwd_with_smoothquant`.
- residual-add + rmsnorm + quant (triple) → `rmsnorm2d_fwd_with_add_dynamicquant` / `add_rmsnorm_quant`.
- layernorm + quant → `layernorm2d_fwd_with_dynamicquant` (Triton) / `_with_smoothquant`.
- gated rmsnorm + fp8 group quant → `gated_rmsnorm_fp8_group_quant` (group=128).
See [[rmsnorm]], [[fused_add_rmsnorm]], [[layernorm]].

## 2. → quant side: which quant
- per-token dynamic fp8 / int8, group-128 fp8, smoothquant int8 (per-channel), mxfp4 block-32 (gfx950).
Cross-link [[quant_dequant_fp8]], [[quant_int8]], [[quant_fp4_mxfp]].

## 3. → consumer GEMM
The whole point: the quantized norm output feeds a **scaled GEMM** (`gemm_a8w8`, `scaled_quant_gemm`)
reading ½/¼ bytes. The norm-quant scale granularity must match the GEMM dequant. See [[scaled_quant_gemm]],
[[dense_gemm]].

## 4. attention-entry mega-fusion
`fused_qk_rmsnorm_group_quant` (QK-norm + fp8 group quant) and `fused_qk_norm_rope_cache_quant` (QK-norm +
RoPE + KV-write + quant) fold the norm-quant into the attention entry. The Qwen3 win. Cross-link [[rope]],
[[mrope]], [[kv_cache_quant]].

## Fusion map
| upstream norm | quant | aiter entrypoint | consumer |
|---|---|---|---|
| rmsnorm | dyn fp8 | `rmsnorm2d_fwd_with_dynamicquant` | qkv/up-gate GEMM |
| add+rmsnorm | dyn fp8 | `rmsnorm2d_fwd_with_add_dynamicquant` | next-block GEMM |
| rmsnorm | smoothquant int8 | `rmsnorm2d_fwd_with_smoothquant` | int8 GEMM |
| gated rmsnorm | group-128 fp8 | `gated_rmsnorm_fp8_group_quant` | block-scale GEMM |
| QK-norm | group fp8 | `fused_qk_rmsnorm_group_quant` | attention |
| QK-norm+RoPE | fp8 + KV write | `fused_qk_norm_rope_cache_quant` | attention/KV cache |

## torch.compile
ROCm fusion pass (`rocm_aiter_fusion.py`) stitches rms+quant op chains; the fused custom op survives
Inductor. See [[backends/vllm_kernels/aiter_integration]].

## Sources
- aiter norm+quant entrypoints: `/sgl-workspace/aiter/aiter/ops/rmsnorm.py`.
- QK-norm+quant / +RoPE: `/sgl-workspace/aiter/aiter/ops/fused_qk_rmsnorm_group_quant.py`, `fused_qk_norm_rope_cache_quant.py`.
