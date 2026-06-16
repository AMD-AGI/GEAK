---
title: quant_int8 — numerics
kind: operator_overview
operator: quant_int8
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [int8]
regimes: [both]
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - https://arxiv.org/abs/2211.10438
---

# quant_int8 — numerics

> INT8 numerics is simpler than FP8 (no FNUZ/OCP split — INT8 is gen-agnostic) but the **outlier problem**
> is sharper: INT8 has uniform spacing, so a single outlier token destroys the resolution of every other
> value unless you smooth (SmoothQuant) and/or go per-token. Accumulate in **INT32**.

## INT8 range & rounding
- **Symmetric**: range [-127, 127] (or [-128, 127]); `s = amax / 127`; `x_i8 = round(x / s)`.
- **Rounding**: round-to-nearest-even via `nearbyint` (vLLM `float_to_int8_rn` uses `nearbyint`; on HIP
  the rounding mode is always `FE_TONEAREST`) to match CUDA. RNE is the standard.
- **Accumulate in INT32** — the MFMA `INT32←INT8` (16×16×32 / 32×32×16) accumulates partial products in
  INT32; never down-convert in the K-loop. `int32_to_int8` saturates the final result.

## Symmetric vs asymmetric (azp / zero-point)
- **Symmetric** (no zero-point): the default; one scale, GEMM is a plain INT8·INT8→INT32.
- **Asymmetric** (azp): `x_i8 = round(x / s) + zp`, where `zp = round(-128 - min_val / s)` (vLLM
  `dynamic_scaled_int8_azp_quant_kernel`). Better for **skewed/one-sided** activations (e.g. post-ReLU,
  GeLU), but requires an **azp correction term** in the GEMM epilogue (subtract `zp · sum(W)`), which
  costs an extra reduction. Use only when the activation distribution is clearly asymmetric.

## SmoothQuant — the outlier migration (the key idea)
INT8 activations fail naively because a few channels have huge outliers that force a coarse scale. SmoothQuant
(arXiv 2211.10438) migrates that difficulty into the weights:
- Per-channel smoothing `s_smooth[c] = max(|X[:,c]|)^α / max(|W[c,:]|)^(1-α)` (α≈0.5, tunable).
- Apply `X̂ = X / s_smooth` (activation easier to quantize) and `Ŵ = W · s_smooth` (weight absorbs it).
- The product `X̂ · Ŵ = X · W` is preserved → no accuracy loss from the migration itself; both tensors
  now quantize cleanly to INT8.
- aiter `smoothquant_fwd(out, input, x_scale=s_smooth[1,n], y_scale[m,1])` applies `s_smooth` and emits
  the per-token scale in one kernel; the smooth factor is folded into the weights offline.

## Granularity vs accuracy
| granularity | accuracy | cost |
|---|---|---|
| per-tensor activation | poor (outliers) | cheapest |
| **per-token activation** | good | one row-reduce — the SmoothQuant default |
| per-tensor weight | ok (weights well-behaved) | cheap |
| **per-channel weight** | best | static, free at inference |
Per-token × per-channel + SmoothQuant is the standard W8A8 recipe.

## INT8 vs FP8 (which to pick)
- **CDNA1/2 (gfx908/gfx90a)**: no FP8 MFMA → INT8 is the only sub-16-bit matrix path.
- **CDNA3/4**: FP8 generally matches INT8 throughput and is **more accurate** (floating-point handles
  dynamic range better than uniform INT8), so FP8 is usually preferred unless the deployment is
  INT8-native or the model was quantized INT8 (compressed-tensors/AWQ-int8). MXINT8 (CDNA4) adds a
  block-scaled INT8 with E8M0 scales — bridges the gap. → [[operators/quant_fp4_mxfp]].

## Accuracy gates
- INT8 W8A8 is lossy → gate on **task accuracy** (gsm8k/mmlu), not byte parity. SmoothQuant + per-token
  typically lands within ~0.5–1 pt of bf16 on large models; small models degrade more.
- err-ratio convention (`tol_err_ratio=0.05`) as the isolated round-trip gate ([[operators/quant_dequant_fp8/numerics]]).

## Pitfalls
- **Naive per-tensor INT8 activations** — outliers wreck resolution; smooth + per-token.
- **azp without the epilogue correction** — wrong GEMM result.
- **Down-converting INT32 accumulator early** — overflow/precision loss.
- **Forgetting α tuning** — α too low/high re-creates the outlier problem on one side.

## Verify
Round-trip INT8 error + INT8 GEMM vs bf16 reference; e2e gsm8k parity. Confirm INT32 accumulate in the MFMA.

## Sources
- vLLM int8 symmetric/azp, nearbyint, int32→int8 saturate: `vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu`.
- aiter smoothquant `(x_scale, y_scale)`: `ROCm/aiter@a6bb49937:csrc/include/smoothquant.h`, `aiter/ops/quant.py`.
- SmoothQuant per-channel migration: https://arxiv.org/abs/2211.10438
- INT8 MFMA shapes / INT32 accumulate: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
