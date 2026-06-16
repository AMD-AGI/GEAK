---
title: scaled_quant_gemm on HipKittens — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: hipkittens
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3, fp8_e5m2]
regimes: [prefill, training]
status: sota
updated: 2026-06-09
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# scaled_quant_gemm × HipKittens

## TL;DR
HipKittens (HK) is the **academic SOTA** for FP8 GEMM on CDNA4: an **8-wave ping-pong** kernel hits
**3222 TFLOPS in just 48 LoC**, and a **4-wave interleave** variant reaches **3327 TFLOPS at 183 LoC**, both
at M=N=K=8192 on MI355X — the latter edging hipBLASLt (~3130) and AMD's own HIP/C++ 8-wave ping-pong (3204).
HK is the **origin of both scheduling patterns** AMD now uses for CDNA FP8 GEMM. Use HK as the perf
reference; for production FP8 GEMM the turnkey path is hipBLASLt / aiter block-scale / the AMD HIP/C++
kernels. Headline numbers gfx950 (MI355X); validated on gfx942.

## SOTA implementation(s)
Two AMD-native schedules, both from HK tile primitives (register/shared tiles + `mma`), replacing NVIDIA
producer/consumer wave specialization (which underperforms on CDNA — see KEY finding below):
- **8-wave ping-pong** — two 4-wave groups alternate MFMA vs memory phases, synchronized with
  shared-memory atomics (no `mbarrier` on CDNA). Compact: **48 LoC**, large tile.
- **4-wave interleave** — one wave per SIMD, full ~512-VGPR budget, instructions interleaved within a wave.
  More code (**183 LoC**) but a few % faster and more robust across ROCm releases (no `#pragma unroll`
  tuning). This is the pattern AMD's CDNA4 GEMM blogs adopt as the ping-pong successor.

See [[languages/hipkittens]] (primitives §scheduling) and arXiv 2511.08083 Table 3.

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| HK FP8 8-wave ping-pong (48 LoC) | arXiv 2511.08083v1 Table 3 | gfx950 (gfx942 validated); fp8_e4m3/e5m2; M=N=K=8192 | **3222 TFLOPS @ MI355X, Hazy Nov 2025 (academic/author-reported)** | minimal-code FP8 GEMM |
| HK FP8 4-wave interleave (183 LoC) | arXiv 2511.08083v1 Table 3 | gfx950; fp8; M=N=K=8192 | **3327 TFLOPS @ MI355X, Hazy Nov 2025 (academic)** — ≥ hipBLASLt ~3130, ≥ AMD HIP/C++ 8-wave 3204 | peak FP8 GEMM, robustness across ROCm |
| (ref) AMD HIP/C++ 8-wave ping-pong | rocm.blogs cdna4-gemm-kernels, ROCm 7.1, MI355X | gfx950; fp8 | 3204 TFLOPS @ 8192 (>hipBLASLt 3130); 2680 @ 4096 (~97% hipBLASLt 2750) | turnkey-ish AMD baseline |

## Config space / knobs (backend-specific)
- **Schedule**: 8-wave ping-pong (compact, 48 LoC) vs 4-wave interleave (peak + robust, 183 LoC) — the core
  trade. Interleave gives full per-wave register budget; ping-pong gives larger tiles with fewer waves.
- **Tile size** (primitive parameter); **MFMA shape** `16×16×32` default for deep pipelines.
- **Pinned register tiles** to bypass the HIPCC AGPR penalty on matmul-heavy loops.
- **Scaling**: FP8 input scales (per-tensor / block); CDNA4 native scaled-MFMA available for FP4/FP6
  variants (see [[operators/quant_fp4_mxfp]]).

## Numerics / parity
- FP8 (e4m3/e5m2) inputs with scales, **FP32 accumulate** → parity within FP8 quant error vs a BF16/FP32
  reference; gate FP8 accuracy on a downstream metric, not just GEMM MSE.

## Integration (how it gets used at serving time)
- HK is the **scheduling origin** for aiter's FP8 GEMM; per the landscape HK is now positioned as an aiter
  backend. Authoring path: build with HIPCC for gfx950/gfx942 from a pinned HK commit, wire via the aiter
  FP8 GEMM dispatch seam (or call-site rebind), then **e2e-gate** through
  [[operators/scaled_quant_gemm/backends/aiter]] / [[operators/scaled_quant_gemm/backends/hip]].

## Pitfalls & anti-patterns
- **Academic maturity**: arXiv 2511.08083v1, Nov 2025; unstable APIs, no support contract — pin a commit,
  bring your own parity + perf gate.
- **gfx950 headline**; gfx942 validated but numbers differ — re-measure on-box.
- **Author-reported, single-source**; treat as vendor-labeled until re-benched.
- The 4-wave 3327 win over hipBLASLt/HIP-C++ is at M=N=K=8192; at smaller shapes the turnkey libs may match
  or win — re-bench per shape.

## How to verify (bench + oracle)
```bash
hipcc --offload-arch=gfx950 ...   # pinned HK commit, FP8 GEMM micro-bench
# bench HK 8-wave vs 4-wave vs hipBLASLt FP8 at target (M,N,K); FP8 accuracy gate downstream
# wire via aiter FP8 seam and re-run A/B + parity gate
```

## Alternatives / cross-links
[[operators/scaled_quant_gemm/backends/hip]] (AMD HIP/C++ 8-wave/4-wave) ·
[[operators/scaled_quant_gemm/backends/hipblaslt]] (bar ~3130 @ 8192) ·
[[operators/scaled_quant_gemm/backends/aiter]] (block-scale + e2e gate) ·
[[operators/scaled_quant_gemm/backends/flydsl]] · [[operators/dense_gemm/backends/hipkittens]] (BF16) ·
[[operators/quant_fp4_mxfp]] · [[languages/hipkittens]] · [[optimization/mfma_scheduling]].

## Sources
- HipKittens — arXiv 2511.08083 (Hazy Research, Nov 2025): https://arxiv.org/abs/2511.08083 ; HTML (Table 3) https://arxiv.org/html/2511.08083v1
- HazyResearch blog: https://hazyresearch.stanford.edu/blog/2025-11-09-hk ; code https://github.com/HazyResearch/HipKittens
- FP8 GEMM Optimization on AMD CDNA4 (8-wave ping-pong, 3204 @ 8192, ROCm 7.1, MI355X): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- Deep Dive Into 4-Wave Interleave FP8 GEMM: https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
- AMD SOTA landscape §1 GEMM (HK FP8 8-wave 3222 / 4-wave 3327; hipBLASLt 3130): [[landscape/amd_sota_2026]]
