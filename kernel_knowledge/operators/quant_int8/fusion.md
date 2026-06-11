---
title: quant_int8 — fusion
kind: operator_overview
operator: quant_int8
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [int8]
regimes: [both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/include/smoothquant.h
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py
  - https://arxiv.org/abs/2211.10438
---

# quant_int8 — fusion

> Two things fuse: the **SmoothQuant per-channel scale** (fold into the cast or the weights), and the
> **cast itself** (fold into the producing norm/act or the consuming GEMM). The weight-side smooth factor
> is folded **offline** (free at inference); the activation-side work is fused at runtime.

## 1. SmoothQuant scale folding (offline + runtime)
- **Weight side (offline)**: `Ŵ = W · s_smooth` is baked into the quantized weights — zero runtime cost.
- **Activation side (runtime)**: `X̂ = X / s_smooth` is fused into the quant kernel —
  `smoothquant_fwd(out, input, x_scale=s_smooth, y_scale)` applies the per-channel smooth and emits the
  per-token scale in one pass (`csrc/include/smoothquant.h`). No separate smoothing pass.

## 2. norm + int8 quant → [[operators/fused_norm_quant]]
RMSNorm output → SmoothQuant smooth → INT8 cast, all in the norm kernel: the RMS reduction, the smooth
multiply, and the per-token amax happen with the row already in registers. This removes a full
read+write of the activation. (Analogous to the FP8 `fused_rms_*_quant` family.)

## 3. GEMM dequant epilogue (consumer) → [[operators/scaled_quant_gemm]]
The INT8 GEMM accumulates in INT32; the epilogue applies `* s_a[M,1] * s_w[N]` (per-token × per-channel)
and, for asymmetric quant, the **azp correction** `- zp · colsum(W)`, then writes bf16. quant (input) and
dequant (output) bracket the INT8 matrix core with no extra passes.

## 4. MoE int8 grouped GEMM → [[operators/fused_moe_grouped_gemm]]
`moe_gemm_int8_smoothquant` (`aiter/ops/triton/moe/`) fuses per-expert smoothquant + routing into the
grouped GEMM; the per-expert smooth scales use a map/hash (`has_smscale_map`) so each expert's `s_smooth`
is selected inside the kernel. `moe_smooth_per_token_scaled_quant_v1/v2` are the HIP variants.

## Fusion decision
| producer of the INT8 input | fuse into |
|---|---|
| RMSNorm output → linear | fused RMSNorm + smooth + int8 quant (#2) |
| GEMM output → next linear | the GEMM's int8 quant epilogue (#3) |
| MoE expert input | `moe_gemm_int8_smoothquant` (#4) |
| nothing fusible | standalone `smoothquant_fwd` / `dynamic_scaled_int8_quant` ([[tuning.md]]) |

## Pitfalls
- Fusing the per-token amax with the norm is only correct if no residual-add follows the norm — use the
  residual-aware variant otherwise.
- azp fusion must carry the zero-point through to the GEMM epilogue correction term.
- MoE per-expert smooth scales must be selected by the routed expert id, not a single global scale.

## Sources
- aiter smoothquant fused (x_scale/y_scale): `ROCm/aiter@a6bb49937:csrc/include/smoothquant.h`, `csrc/kernels/quant_kernels.cu`.
- Fused MoE int8 smoothquant GEMM: `ROCm/aiter@a6bb49937:aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`.
- SmoothQuant scale migration: https://arxiv.org/abs/2211.10438
