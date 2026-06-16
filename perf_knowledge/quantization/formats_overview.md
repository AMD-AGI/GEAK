---
title: Quantization formats overview (bf16 → fp8 → mxfp4/6 → int8/4)
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp4_e2m1, fp6_e2m3, fp6_e3m2, mxfp4, mxfp6, int8, int4]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - https://arxiv.org/pdf/2310.10537
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# Quantization formats overview

> **TL;DR.** Inference on MI GPUs walks a precision ladder: **bf16/fp16** (baseline, lossless swap) →
> **FP8 E4M3** (2× compute, the inference default) → **MXFP4/MXFP6** (4×, block-scaled, CDNA4-only HW) →
> **INT8/INT4** (weight-only, integer). This page is the bit-layout reference; the *behavioral* numerics
> live in [[operators/quant_dequant_fp8]] and [[operators/quant_fp4_mxfp]] — read those for rounding,
> amax, clamps. Dialect (FNUZ vs OCP) is in [[fnuz_vs_ocp.md]]; block scaling in [[block_scaling_mxfp.md]].

## The ladder at a glance
| dtype | bits | exp/mant | element max | HW MFMA | role |
|---|---|---|---|---|---|
| FP16 | 16 | 5/10 | 65504 | CDNA1–4 | baseline, wide mantissa |
| BF16 | 16 | 8/7 | ~3.4e38 | CDNA2–4 (native), CDNA1 emul | baseline, wide range; training/inference default 16-bit |
| FP8 E4M3 | 8 | 4/3 | ±448 (OCP) / ±240 (FNUZ) | CDNA3–4 | **inference default** (weights+activations) |
| FP8 E5M2 | 8 | 5/2 | ±57344 | CDNA3–4 | gradients / very wide range |
| FP6 E2M3 | 6 | 2/3 | ±7.5 | CDNA4 | weights when FP4 too lossy (more mantissa) |
| FP6 E3M2 | 6 | 3/2 | ±28 | CDNA4 | weights/grad (more range) |
| FP4 E2M1 | 4 | 2/1 | ±6 | CDNA4 | aggressive weight (or w4a4) quant |
| INT8 | 8 | integer | ±127 | CDNA1–4 | classic w8a8 / weight-only |
| INT4 | 4 | integer | ±7 | (packed; dequant to fp on most paths) | weight-only (GPTQ/AWQ) |

Element max values per the OCP MX spec / HIP FP8 reference; FP6/FP4 maxima are the OCP MX element formats.

## Floating-point bit layout (sign · exponent · mantissa)
- **FP16** `S EEEEE MMMMMMMMMM` — IEEE half, bias 15.
- **BF16** `S EEEEEEEE MMMMMMM` — FP32's exponent (bias 127) truncated to 7 mantissa bits → same range as
  FP32, ~3 decimal digits precision. Preferred 16-bit type for LLMs (no overflow on attention scores).
- **FP8 E4M3** `S EEEE MMM` — bias 7 (OCP) or 8 (FNUZ). 3 mantissa bits ⇒ ~2× the relative precision of
  E5M2. No inf in OCP E4M3 (NaN only); FNUZ has neither inf nor −0. See [[fnuz_vs_ocp.md]].
- **FP8 E5M2** `S EEEEE MM` — bias 15 (OCP). IEEE-like, has ±inf and ±0 in OCP; FNUZ variant has bias 16,
  no inf.
- **FP6 E2M3** `S EE MMM`, **FP6 E3M2** `S EEE MM` — OCP MX 6-bit element formats; no inf/NaN encodings.
- **FP4 E2M1** `S EE M` — only **15 distinct levels** `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`. Useless
  alone; the per-block E8M0 scale is what makes it usable ([[block_scaling_mxfp.md]],
  [[operators/quant_fp4_mxfp]]).

## Integer formats
- **INT8** — symmetric `[-127,127]` (or asymmetric with zero-point). Pairs with FP scale `x ≈ s·q`.
  Native MFMA on all CDNA gens; classic for w8a8 GEMM and KV cache.
- **INT4** — 4-bit weight-only. GPTQ/AWQ store INT4 weights + group scales; the GEMM usually
  **dequantizes INT4→fp16/fp8 on the fly** rather than running an INT4 matrix core path. Quark's
  two-level **INT4-weight / FP8-compute** scheme gives ~4× compression with FP8-class accuracy
  ([[calibration_and_quark.md]]).

## OCP Microscaling (MX) — the unifying framing
The **OCP MX spec v1.0** defines a *block floating-point* family: a vector of low-bit **element**
values (FP8/FP6/FP4/INT8) sharing one **E8M0** (8-bit, exponent-only) **block scale** over a fixed group
of **32** elements. Named concrete formats: **MXFP8, MXFP6, MXFP4, MXINT8**. The element type sets the
per-value precision; the shared scale restores dynamic range cheaply.
- Effective bits/element = element_bits + 8/32 = element_bits + **0.25** ⇒ MXFP4 ≈ **4.25 b**,
  MXFP6 ≈ **6.25 b**, MXFP8 ≈ **8.25 b**.
- The MX paper (arXiv 2310.10537) is the research basis; the OCP spec is the normative bit-layout.
- AMD CDNA4 implements MXFP4/MXFP6 (and MXFP8) **in the matrix core** via block-scaled MFMA; CDNA3 has no
  FP4/FP6 HW and only *simulates* MX. Full mechanism: [[block_scaling_mxfp.md]],
  [[hardware/cdna4_mi350]].

## Picking a format (one-screen heuristic)
- **Largest models, MI350**: MXFP4 (validate accuracy) → fall back to MXFP6/mixed if it degrades.
- **General inference, MI300/MI350**: FP8 E4M3 per-token/per-block — the safe default.
- **Memory-bound weight-only, any gen**: INT4 GPTQ/AWQ (or INT4-FP8).
- **No FP8 HW (CDNA1/2)**: INT8 w8a8 or stay bf16/fp16.
- Throughput vs accuracy tradeoffs per gen: [[hardware_support_matrix.md]].

## Pitfalls
- **Treating FP8 as one format** — E4M3 ≠ E5M2, and OCP ≠ FNUZ ([[fnuz_vs_ocp.md]]).
- **Raw FP4 without a block scale** — collapses; MX is mandatory.
- **Assuming INT4 runs on an INT4 matrix core** — it is dequantized to fp on MI GPUs in practice.
- **bf16 vs fp16 confusion** — fp16 has more mantissa but can overflow on attention logits; bf16 is the
  LLM default.

## Verify
- Round-trip a tensor through the chosen cast; check max-rel-error per [[operators/quant_dequant_fp8]] /
  [[operators/quant_fp4_mxfp]]. Confirm the saturation point matches the dtype+dialect.

## Sources
- OCP MX formats (element types, E8M0, group 32, MXFP4/6/8): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Microscaling Data Formats for Deep Learning (MX paper): https://arxiv.org/pdf/2310.10537
- FP8 bit layout, E4M3/E5M2, FNUZ: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html
- FP6/FP4 element codes, CDNA matrix-core dtypes: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
