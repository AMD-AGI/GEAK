---
title: fused_norm_quant — tuning
kind: technique
operator: fused_norm_quant
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, mxfp4]
regimes: [both]
updated: 2026-06-09
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# fused_norm_quant — tuning

Inherits [rmsnorm/tuning.md](../rmsnorm/tuning.md) (bandwidth-bound, persistent grid, 128-bit loads). The
quant adds **one more in-register reduction** (the abs-max for the scale) and **changes the output dtype**.

## 1. The quant is free; the output traffic drops
The norm already has `y` in fp32 registers. Computing `scale = max(|y|)/qmax` is a second wave64 reduce
over the same data — negligible. The win is the **write**: `y_q` is fp8 (½) / int8 (½) / fp4 (¼) of bf16.
And the *consumer* GEMM then reads that quantized input. So tune for: norm bandwidth + correct scale
granularity.

## 2. Scale granularity decides the reduction shape
- **per-token (per-row)**: one abs-max over the row → one extra wave64 reduce, scale `[M]`. Simplest.
- **per-group (e.g. 128)**: abs-max per 128-element group → multiple sub-reductions, scale `[M, N/128]`.
  aiter `gated_rmsnorm_fp8_group_quant` is **head_dim=128, group_size=128 only** — block the kernel so each
  group maps to a contiguous lane span.
- **mxfp4 block-32**: e8m0 scale per 32 elements; block-N must be a multiple of 32.
The granularity must match the consumer GEMM's dequant — a mismatch is a correctness bug, not a perf knob.

## 3. Two-pass when N > block
If N exceeds the LDS block (rare for real hidden), the norm is two-pass and the quant needs the **final
normalized `y`** to compute the abs-max → either a third pass or fuse the abs-max into the normalize pass.
The Triton `_quant_rms_norm_kernel` does the abs-max in the same sweep that produces `y`.

## 4. Knob table
| knob | setting | note |
|---|---|---|
| `num_warps` | 2–4 | memory-bound |
| `BLOCK_SIZE` | next_pow2(N) (×32 mxfp4, group-aligned) | 128-bit loads + scale blocks |
| grid | `min(M, num_sms)` persistent | fill 304 CUs |
| scale granularity | per-token / group-128 / block-32 | **match consumer GEMM** |
| out dtype | fp8 fnuz (gfx942) / mxfp4 (gfx950) | ½ / ¼ write |
| scale dtype | fp32 (per-token/group) / e8m0 (mx) | match dequant |

## 5. Stack with residual-add and activation
- `rmsnorm2d_fwd_with_add_dynamicquant`: residual+norm+quant triple — one read+write does all three.
- On the MLP, the sibling is `act_mul_and_fp8_group_quant` (activation+quant) — see
  [[act_and_mul_silu_gelu]] fusion.
- **PTPC-FP8** (per-token-per-channel) is **up to 2.5× vs naive** on MI300X — prefer it where the consumer
  GEMM supports per-channel weight scales.

## 6. Let the compiler do it (when you can)
- vLLM **ActivationFusionPass +8% throughput**; the `rocm_aiter_fusion` RMSNorm+quant pass feeds the linear
  fp8 directly. ⚠ Inductor torch-op quant can now **auto-fuse some patterns**, so the standalone SiLU+quant /
  RMSNorm+quant passes are **obsolete except custom-op cases** (attention, collectives, sub-byte quant). Check
  the compiled graph before hand-wiring a fused entrypoint.
- **AITER v0.1.12** adds a fused **gated RMSNorm + group-quant** kernel for the gated-MLP path.

## Sources
- Triton `_quant_rms_norm_kernel` (abs-max in the normalize sweep): `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`.
- group-128 fp8 quant constraints: `/sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py`.
- 128-bit loads / persistent grid: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
- PTPC-FP8 up to 2.5× vs naive (MI300X): https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html.
- vLLM Inductor fusion passes (ActivationFusionPass +8%; torch-op quant auto-fuse obsoletes some passes except custom-op): https://docs.vllm.ai/en/latest/design/fusions/.
- AITER v0.1.12 fused gated RMSNorm + group-quant: https://github.com/ROCm/aiter/releases.
