---
title: Scaling strategies — granularity × dynamic/static, and the 224.0 ROCm cap
kind: technique
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, mxfp4, mxfp6, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://arxiv.org/pdf/2211.10438
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html
  - https://docs.vllm.ai/en/stable/features/quantization/quark/
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# Scaling strategies

> **TL;DR.** Two orthogonal axes decide accuracy-vs-cost: **granularity** (how many elements share one
> scale: per-tensor → per-token/per-channel → per-block) and **timing** (**static**/calibrated vs
> **dynamic**/runtime-amax). Coarser scales are cheaper but underflow when outliers dominate amax. The
> safe FP8 default is **per-token dynamic activations × per-channel weights**; block-fp8 (1×128) and
> MXFP (32-elt) go finer. ROCm dynamic FP8 clamps amax to **224.0** for headroom. Element numerics:
> [[operators/quant_dequant_fp8]]; block scaling: [[block_scaling_mxfp.md]].

## Axis 1 — granularity
| granularity | scale shape | accuracy | cost | typical use |
|---|---|---|---|---|
| **per-tensor** | 1 scalar | lowest | cheapest | weights w/ static calib; legacy |
| **per-channel (per-column)** | one per output channel | good for weights | precompute | weight quant default |
| **per-token (per-row)** | one per activation row | **good** | one row-reduce | activation default (SmoothQuant-style) |
| **per-block / group (1×128)** | one per K-block | best (FP8) | per-group reduce | DeepSeek block-fp8 |
| **per-block 32-elt (MXFP)** | E8M0 per 32 elts | best (4/6-bit) | per-block, in-HW | MXFP4/6 ([[block_scaling_mxfp.md]]) |

**Why finer wins.** A per-tensor scale is set by the single largest element; a few outlier tokens force a
coarse scale that **underflows the quiet tokens**. Per-token gives each row its own range; per-block
goes finer still. This is the same argument as MXFP's 32-element block ([[operators/quant_fp4_mxfp]]).

**Activations vs weights.** Activations have *token-varying* outliers ⇒ **per-token dynamic**. Weights
are static ⇒ **per-channel static** (computed once at quantization time). The common FP8 GEMM is
**per-token-act × per-channel-weight** (PTPC), which vLLM ROCm supports natively.

## Axis 2 — dynamic vs static
- **Dynamic**: compute amax at runtime (per-token row-reduce, or per-tensor atomic/segmented max). No
  calibration; adapts to each batch; adds a reduction. vLLM ROCm supports on-the-fly FP8 / PTPC-FP8 at
  server startup — costs ~2–5 min startup but skips pre-quantization.
- **Static**: scales fixed from a **calibration** pass and baked into the checkpoint. Cheapest at
  runtime; needs a representative dataset; risks drift if serving distribution ≠ calib distribution.
  Required for the most aggressive paths and for KV per-head scales ([[calibration_and_quark.md]],
  [[kv_cache_quantization.md]]).
- **Rule of thumb**: weights → static per-channel; activations → dynamic per-token (or static per-token
  via calibration when latency-critical).

## SmoothQuant — the enabler for activation quant
Activation outliers concentrate in a few channels, making activations hard to quantize while weights are
easy. **SmoothQuant** (arXiv 2211.10438) migrates the difficulty by a per-channel diagonal rescale:
`X̂ = X·diag(s)^-1`, `Ŵ = diag(s)·W`, with `s_j = max|X_j|^α / max|W_j|^(1−α)` (smoothing strength α,
typically ~0.5). This **flattens activation outliers into the weights**, enabling W8A8 / per-token FP8
without the outlier blowup. AMD Quark ships SmoothQuant + **AutoSmoothQuant** (per-layer α by MSE)
([[calibration_and_quark.md]]).

## The 224.0 ROCm dynamic cap (a real, sourced gotcha)
vLLM dynamic FP8 quant on ROCm clamps amax to **224.0**, not the FNUZ max 240:
```cpp
// vllm csrc/quantization/fp8/common.cu
// "Using the default max value from pytorch (240.0) will cause accuracy
//  issue when running dynamic quantization. Here use 224.0f for rocm."
constexpr auto FP8_E4M3_MAX = 224.0f;
```
**Why**: leaving headroom below saturation avoids round-to-saturation artifacts on the largest-magnitude
elements during dynamic (amax-derived) quant. It also lines up with the OCP→FNUZ conversion, where OCP's
±448 maps to ±224 ([[fnuz_vs_ocp.md]]). Static/calibrated paths can use the full range; dynamic paths
benefit from the margin. Detail + the zero-scale clamp: [[operators/quant_dequant_fp8]].

## Reciprocal-scale & zero-scale hygiene
- Store **1/s** so the per-element hot path is a multiply, not a divide (`is_scale_inverted` in vLLM).
- **Clamp scale=0 → 1** on all-zero rows (`scale[scale==0]=1` in aiter `pertoken_quant`) to avoid
  divide-by-zero ([[operators/quant_dequant_fp8]]).

## Fusion: do the scale where the data already is
Per-token amax pairs naturally with the preceding norm/activation; fuse the quant into RMSNorm/SiLU to
avoid an extra pass over activations ([[operators/fused_norm_quant]], [[operators/scaled_quant_gemm]]).

## Pitfalls
- **Per-tensor on outlier-heavy activations** — quiet tokens underflow; use per-token + SmoothQuant.
- **Forgetting the 224 cap** on ROCm dynamic quant — saturation artifacts on max elements.
- **Static scales with a mismatched calib set** — distribution drift; revalidate task accuracy.
- **Dividing by a zero scale** — clamp to 1.
- **Mixing weight per-tensor with activation per-token** when the GEMM kernel expects matched layouts —
  check the backend card ([[operators/scaled_quant_gemm]]).

## Verify
- Compare per-token vs per-tensor round-trip error on a real activation batch; the gap is the outlier
  tax. Then gate e2e on task accuracy ([[accuracy_evaluation.md]]).

## Sources
- SmoothQuant (per-channel outlier migration, α): https://arxiv.org/pdf/2211.10438
- ROCm granularity / static-vs-dynamic, PTPC-FP8: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html
- Quark schemes (per-tensor/per-channel/per-token, AutoSmoothQuant): https://docs.vllm.ai/en/stable/features/quantization/quark/
- per-block argument, scaled MFMA: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- 224.0 cap, scale-inverted, zero-scale clamp: [[operators/quant_dequant_fp8]] (`vllm@HEAD:csrc/quantization/fp8/common.cu`, `aiter/ops/quant.py`).
