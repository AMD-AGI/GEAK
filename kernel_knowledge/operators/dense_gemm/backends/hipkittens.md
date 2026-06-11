---
title: dense_gemm on HipKittens — SOTA card
kind: sota_card
operator: dense_gemm
backend: hipkittens
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, training]
status: sota
updated: 2026-06-09
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083v1
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# dense_gemm × HipKittens

## TL;DR
HipKittens (HK) is the **academic SOTA tile-DSL** for BF16/FP16 dense GEMM on CDNA3/CDNA4: a 256×256-tile,
**no-wave-specialization** kernel reaches **1610 TFLOPS** BF16 at M=N=K=8192 on MI355X — edging the
ThunderKittens-on-B200 (1538) and CUTLASS-on-B200 (1570) NVIDIA references, and matching/beating
hipBLASLt on AMD — while staying <100 LoC in the hot loop. It beats AMD ROCm **Triton 1.3–3.0×**. Use HK
as the **perf reference / idea source** for how to schedule a CDNA GEMM (8-wave ping-pong, XCD/L2-aware
grids); for production today prefer hipBLASLt/aiter and treat HK numbers as author-reported until
re-measured. Headline numbers are gfx950 (MI355X); validated on gfx942 too.

## SOTA implementation(s)
HK writes the GEMM from **tile primitives** (register/shared tiles + `mma`/`load`/`store`), then applies an
AMD-native schedule instead of NVIDIA producer/consumer wave specialization. The headline BF16 kernel uses a
**256×256 output tile with 8-wave ping-pong** scheduling (two wave-groups alternate MFMA / memory phases via
shared-memory atomics, since CDNA has no `mbarrier`). **XCD/L2-aware grid swizzle** (chiplet-aware block
mapping on the 8-XCD MI355X) adds **+3–19%** at large M by lifting L2 hit-rate. See
[[languages/hipkittens]] (overview, primitives, perf_findings) and arXiv 2511.08083 Tables 2 & 4.

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| HK BF16 GEMM, 256×256 tile, 8-wave ping-pong | arXiv 2511.08083v1 Table 2; `github.com/HazyResearch/HipKittens` | gfx950 (gfx942 validated); bf16/fp16; M=N=K=8192 | **1610 TFLOPS @ MI355X, Hazy Nov 2025 (academic/author-reported)** — vs TK-on-B200 1538, CUTLASS-B200 1570 (NVIDIA refs); ≥ hipBLASLt on AMD | large square BF16 GEMM; perf reference |
| HK + XCD/L2-aware grid swizzle | arXiv 2511.08083v1 Table 4 | gfx950; bf16; large M | **+3–19%** (e.g. ~900→1068 TFLOPS @ M=14592; row-major 1113→1145 @ M=N=K=9216, L2 55%→75%, author-reported) | large/skewed M where L2 reuse matters |
| HK vs ROCm Triton | arXiv 2511.08083v1; blog | gfx950; bf16/fp8 | HK **1.3–3.0×** faster than ROCm Triton (author-reported) | shows Triton headroom on AMD |

## Config space / knobs (backend-specific)
- **Schedule**: 8-wave ping-pong (compact, large 256×256 tile, fewer LoC — the default GEMM win) vs 4-wave
  interleave (one wave/SIMD, full register budget, more code — see the FP8 card for where it edges ahead).
- **Output tile**: 256×256 for the BF16 headline; tile is a primitive parameter.
- **MFMA shape**: register tiles default to `16×16×32` for deep, schedulable pipelines.
- **Grid swizzle**: XCD/L2-aware (chiplet) mapping; HBM-address swizzling for conflict-free async HBM→LDS.
- **Pinned register tiles**: less critical for pure GEMM than for attention bwd, but available.

## Numerics / parity
- BF16/FP16 inputs, **FP32 MFMA accumulate** → parity with hipBLASLt/CUTLASS up to tiling rounding. No
  scaling here; block-scaled FP8/FP4 is the separate [[operators/scaled_quant_gemm/backends/hipkittens]] path.

## Integration (how it gets used at serving time)
- HK is now positioned as **an aiter backend** per the landscape (it is the origin of aiter's 8-wave
  ping-pong / 4-wave interleave schedules). Authoring path: build the HK kernel with HIPCC
  `--offload-arch=gfx950` (or gfx942) from a pinned `HazyResearch/HipKittens` commit, then wire it the same
  way an authored kernel is engaged for dense GEMM — via the aiter dispatch seam or a call-site rebind in the
  model `LinearMethod` — and **e2e-gate** through [[operators/dense_gemm/backends/aiter]]'s verification flow
  (isolated win must survive e2e, as the Triton card shows).

## Pitfalls & anti-patterns
- **Academic maturity**: research artifact (arXiv 2511.08083v1, Nov 2025), unstable APIs, no AMD support
  contract — pin a commit; do not deploy as a serving dependency without your own parity + perf gate.
- **gfx950 headline**: 1610 TFLOPS is MI355X; gfx942 is validated but numbers differ — re-measure on-box.
- **Author-reported, single-source** (the paper); treat as vendor-labeled until re-benched on your shapes.
- **Don't assume blanket wins**: HK matches/edges hipBLASLt on BF16 square GEMM; on some shapes the turnkey
  lib is still the no-tune default. Re-bench per shape.

## How to verify (bench + oracle)
```bash
# build HK GEMM micro-bench for your arch, pinned commit
hipcc --offload-arch=gfx950 ...   # or gfx942
# bench HK vs hipBLASLt default at the target (M,N,K); FP32-accumulate parity oracle vs torch/hipBLASLt
# then wire via the aiter seam and re-run the same A/B + parity gate as the aiter dense_gemm card
```

## Alternatives / cross-links
[[operators/dense_gemm/backends/hipblaslt]] (no-tune default / bar) · [[operators/dense_gemm/backends/aiter]]
(dispatch + e2e gate) · [[operators/dense_gemm/backends/flydsl]] · [[operators/dense_gemm/backends/triton]]
(HK beats it 1.3–3.0×) · [[operators/scaled_quant_gemm/backends/hipkittens]] (FP8) ·
[[languages/hipkittens]] · [[optimization/mfma_scheduling]] (8-wave ping-pong / 4-wave interleave; why
wave-specialization fails on CDNA).

## Sources
- HipKittens: Fast and Furious AMD Kernels — arXiv 2511.08083 (Hazy Research, Nov 2025): https://arxiv.org/abs/2511.08083 ; HTML (Tables 2, 4) https://arxiv.org/html/2511.08083v1
- HazyResearch blog "AMD GPUs go brrr": https://hazyresearch.stanford.edu/blog/2025-11-09-hk
- Code: https://github.com/HazyResearch/HipKittens
- AMD SOTA landscape §1 GEMM (HK 1610 BF16; TK-B200 1538 / CUTLASS-B200 1570; XCD +3–19%): [[landscape/amd_sota_2026]]
