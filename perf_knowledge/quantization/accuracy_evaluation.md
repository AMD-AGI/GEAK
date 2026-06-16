---
title: Accuracy evaluation — err_ratio gate, task accuracy not byte parity, SR
kind: technique
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, mxfp4, mxfp6, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/artificial-intelligence/quark/README.html
  - https://arxiv.org/html/2511.10909v1
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html
  - https://arxiv.org/pdf/2310.10537
---

# Accuracy evaluation

> **TL;DR.** Quantization is **lossy by construction** — *never gate on byte parity*. Use two tiers: an
> **isolated kernel gate** (round-trip max-rel-error + `err_ratio < 0.05`, the aiter convention) and the
> **decisive e2e gate** (task accuracy: MMLU / GSM8K / perplexity within a band, plus tok/s up).
> Bit-exact parity only applies to *lossless* swaps (bf16↔bf16 library substitution). Element-level
> gating detail: [[operators/quant_dequant_fp8]], [[operators/quant_fp4_mxfp]].

## The two-tier gate
1. **Isolated (kernel/op) gate** — for a quant or quantized-GEMM kernel in isolation:
   - round-trip a representative tensor through the cast vs an fp32 reference; check **max-rel-error**;
   - **`err_ratio`** = fraction of elements outside `(rtol, atol)`; aiter's convention is
     **`checkAllclose(..., tol_err_ratio=0.05)`** (`aiter/test_common.py`) — i.e. `err_ratio < 0.05`.
   - This is the gate the **aiter gradlib** uses for *lossless* library swaps too
     (`err_ratio < 0.05`, `gradlib/gradlib/GemmTuner.py`), where it should be near-zero.
2. **End-to-end (task) gate** — the one that actually decides ship/no-ship:
   - **GSM8K** (reasoning), **MMLU** (knowledge), **perplexity** (LM fit) before vs after quant,
     same seed / temp=0;
   - require the task metric within a band **and** tok/s improved. A reasonable band: GSM8K/MMLU drop
     **< ~0.5–1 pt** for FP8; budget a **larger band for MXFP4** (most aggressive) and prefer
     MXFP6/mixed if it fails ([[block_scaling_mxfp.md]], [[operators/quant_fp4_mxfp]]).

## Why byte parity is wrong as a gate
| swap | lossy? | correct gate |
|---|---|---|
| bf16 → bf16 (library/kernel substitution) | no | byte / err-ratio parity (`err_ratio < 0.05`, ~0) |
| fp16/bf16 → **FP8** | **yes** | task accuracy + isolated err-ratio |
| fp16 → **MXFP4/6** | **yes (more)** | task accuracy (wider band) + per-block error |
| → **INT8/INT4** | **yes** | task accuracy + per-channel/per-group error |
Any FP8/FP6/FP4/INT path changes bits *by design*; demanding parity would reject every working quant.

## Stochastic rounding (SR) vs round-to-nearest-even (RNE)
- **RNE** is the default for inference quant. **SR** rounds up with probability ∝ the residual, removing
  the *systematic bias* RNE accumulates over long reductions. SR is used mainly in **FP8 training**
  (weight updates) and some KV paths — **not** standard inference activation quant.
- CDNA3's MFMA conversion uses an asymmetric **round-down** mode for FP16/BF16 that injects a small
  systematic bias on long-K reductions; the FP8 path was specifically adjusted to mitigate it (MMA-Sim,
  arXiv 2511.10909). Practical default: **RNE + per-token scaling**; reach for SR only when *measured*
  accumulation bias matters. Detail: [[operators/quant_dequant_fp8]].

## What to measure, concretely
- **Per-element / per-block round-trip error** (isolated): max-rel-error + err_ratio vs fp32.
- **Activation outlier check**: per-token vs per-tensor error gap (the outlier tax,
  [[scaling_strategies.md]]).
- **Task metrics** (e2e): GSM8K exact-match, MMLU accuracy, WikiText perplexity — fixed prompts, temp=0,
  same harness (lm-eval-harness / Quark's `--tasks`). Quark's accuracy/perf blog uses exactly this
  vLLM+SGLang MI300X methodology.
- **Throughput**: median of ≥3 warm repeats, tagged `value @ hw, ROCm ver, lib@commit, date`
  ([[index/conventions.md]]).

## MMA-Sim / bit-accurate references
For isolating *where* error comes from (cast vs accumulate vs rounding mode), a bit-accurate matrix-core
simulator (MMA-Sim) reproduces CDNA3 MFMA rounding exactly — useful to separate kernel bugs from
inherent quant loss. See [[operators/quant_dequant_fp8]].

## Pitfalls
- **Gating FP8/MXFP on byte parity** — guarantees a false failure.
- **No task gate** — round-trip error can look fine while GSM8K drops (outlier-sensitive layers).
- **Changing seed/temp between runs** — confounds the accuracy delta.
- **One task only** — MMLU can pass while reasoning (GSM8K) regresses; use a small suite.
- **Ignoring accumulation bias on long K** — RNE round-down can drift; SR or higher-precision accum
  ([[operators/quant_dequant_fp8]]).

## Verify
- Isolated: `checkAllclose(out, ref, tol_err_ratio=0.05)` on a representative input.
- e2e: GSM8K + MMLU + perplexity, before/after, same harness/seed; tok/s median of ≥3 warm runs.

## Sources
- Quark accuracy/perf methodology (vLLM+SGLang, MI300X, task metrics): https://rocm.blogs.amd.com/artificial-intelligence/quark/README.html
- CDNA3 round-down rounding, FP8 adjustment, SR context (MMA-Sim): https://arxiv.org/html/2511.10909v1
- ROCm quantization accuracy guidance: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-quantization.html
- MX paper (low-bit accuracy basis): https://arxiv.org/pdf/2310.10537
- err_ratio<0.05 convention: [[operators/quant_dequant_fp8]] (`aiter/test_common.py:400`, `gradlib/gradlib/GemmTuner.py`).
