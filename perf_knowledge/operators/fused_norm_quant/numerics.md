---
title: fused_norm_quant — numerics & parity
kind: technique
operator: fused_norm_quant
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, int8, mxfp4]
regimes: [both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# fused_norm_quant — numerics & parity

Inherits the norm numerics ([rmsnorm/numerics.md](../rmsnorm/numerics.md): fp32 Σ, γ promote, ε) plus the
quant numerics ([[quant_dequant_fp8]] / [[quant_int8]]). The quant part is where the bugs are.

## 1. FNUZ vs OCP fp8 — the off-by-2× trap (gfx942)
**CDNA3 (gfx942) MFMA consumes FNUZ fp8** (`e4m3fnuz`/`e5m2fnuz`, exponent bias off-by-one vs OCP).
Quantizing to OCP `e4m3fn` and reading it as fnuz (or vice versa) is wrong by **exactly 2×**. This is the
#1 silent MI300X fp8 bug. SGLang/vLLM normalize checkpoints with `normalize_e4m3fn_to_e4m3fnuz`. On
**CDNA4 (gfx950)** fp8 is OCP. The fused-norm-quant kernel must emit the dialect the consumer GEMM expects.

## 2. Scale granularity must match the consumer GEMM
The norm-quant produces `(y_q, scale)`; the GEMM dequants with `scale`. If the norm writes per-token scale
but the GEMM expects per-group-128 (or block-32 mxfp4), the dequant is wrong. aiter
`gated_rmsnorm_fp8_group_quant` is **fixed at group=128, head_dim=128** for exactly this reason — it pairs
with a specific GEMM. Verify the pairing.

## 3. Compute order: norm in fp32, then quant
`y_fp32 = rmsnorm(x)` in fp32, abs-max in fp32, `scale = max/qmax`, `y_q = round_RNE(y_fp32/scale)`. Don't
quantize a bf16-rounded `y` (double rounding loses a bit). RNE rounding; clamp to the fp8/int8 range.

## 4. Dynamic vs static scale
- **dynamic** (per-token/group, computed in-kernel): adapts to the activation range; best accuracy; the
  norm-quant fusion computes it for free.
- **static** (`scaled_silu_and_mul`-style precomputed scale): faster (no max-reduce) but needs a
  calibration pass; accuracy depends on the calibration set.

## 5. Accuracy gate (task-level, not allclose)
Quant is lossy → never gate on byte parity. Gate on:
1. isolated: dequant(`y_q`)·scale vs fp64 `rmsnorm(x)` within the fp8/int8 error band.
2. e2e: gsm8k / MMLU delta within run-to-run noise (the real gate).
3. fnuz dialect confirmed on gfx942; mxfp4 only on gfx950.
- vLLM AITER MLA fp8 caused a gsm8k loss (aiter #1455) — a concrete reminder to eval, not assume.

## Sources
- group-128 fp8 quant constraints + HIP kernel: `/sgl-workspace/aiter/aiter/ops/gated_rmsnorm_fp8_group_quant.py`.
- abs-max + RNE quant in the norm kernel: `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`.
- FNUZ vs OCP fp8 on CDNA3/4: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html; perf_knowledge [[quant_dequant_fp8]].
