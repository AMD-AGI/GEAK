---
title: quant_fp4_mxfp on Gluon — SOTA card
kind: sota_card
operator: quant_fp4_mxfp
backend: gluon
gens: [gfx950]
dtypes: [fp4_e2m1]
regimes: [prefill, training]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
---

# quant_fp4_mxfp × Gluon

## TL;DR (one-line decision)
> On **CDNA4 (gfx950)** Gluon is the SOTA authoring path for MXFP4 GEMM: **5255 TFLOPS @ 92.41% MFMA eff**
> (a4w4, 4096×4096×32768) via the native scaled-MFMA `v_mfma_scale_f32_16x16x128_f8f6f4` with an 8-bit
> per-32-element block scale. CDNA4-only (the op does not exist on gfx942); use when you need peak MXFP4 GEMM
> and can hand-schedule the data + scale pipelines.

## SOTA implementation(s)
Best known MXFP4 GEMM in a Python DSL on gfx950 is AMD's Gluon `a4w4` kernel — same v0→v9 skeleton plus a
dedicated **scale pipeline (GR → LW → LR)** and the 16-cycle scaled MFMA (cite the Gluon GEMM tutorial).

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| Gluon `a4w4` | `ROCm/gfx950-gluon-tutorials:kernels/gemm/a4w4`; Gluon GEMM tutorial | gfx950; MXFP4 (a4w4); 256×256×256 tile; 4096×4096×32768 | **5255 TFLOPS @ 92.41% MFMA eff** @ MI355X gfx950, ROCm 7.0, AMD-measured, 2026 | peak MXFP4 GEMM on CDNA4 with manual scheduling |

MXFP4 GEMMs are ~62% of Llama2-70B e2e cost, so this ceiling matters for FP4 inference; it is the SOTA MXFP4
GEMM cell in [[landscape/amd_sota_2026]].

## Config space / knobs
- **Tile**: `256×256×256` (M×N×K) for a4w4 — same skeleton, larger `BLOCK_K`.
- **Scaled MFMA**: native `gl.amd.cdna4.mfma_scaled` → `v_mfma_scale_f32_16x16x128_f8f6f4`, 16-cycle.
- **Scale pipeline**: explicit **GR → LW → LR** (global-read → LDS-write → LDS-read of block scales)
  alongside the data pipeline — the extra stage that caps eff at 92.41% (LDS port contention).
- **Wave schedule**: ping-pong (8-wave) / interleave (4-wave), authored by hand. [[optimization/mfma_scheduling]]
- **num_stages / pipeline**: explicit 2-/3-stage data pipeline (GR→LW→LR→MFMA).
- **LDS layout**: swizzled/padded shared layouts + `ds_read_tr`; register budget within 512-VGPR/EU.
- **Scheduling passes**: `TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1`.

## Numerics / parity
MXFP4 packs **two 4-bit values per byte** with a **per-group 8-bit scale factor for every 32 elements**
(microscaling block scale; the tutorial says "8-bit per-group scale", does not spell out E8M0). Scaled MFMA
applies the block scale in-instruction, **FP32 accumulate**. Gate FP4 accuracy with a Quark/MXFP4 quant-error
check vs a higher-precision reference — sub-byte quant accuracy is the real risk here, not GEMM rounding.

## Integration (rebind seam)
Authoring path, not a production dispatcher. Wire a Gluon MXFP4 GEMM via the aiter A4W4 / FlyDSL-adjacent
seam or a call-site rebind, then e2e-gate. Relationship: Gluon = low-level Triton dialect authoring of the
MXFP4 kernel; aiter (and FlyDSL for MoE-grouped MXFP4) = the production dispatch the authored kernel feeds.
See [[operators/quant_fp4_mxfp/backends/aiter]].

## Pitfalls & anti-patterns
- **CDNA4-only headline.** `v_mfma_scale_f32_16x16x128_f8f6f4` is a gfx950 op — the 5255 number does **not**
  transfer to gfx942 (no native scaled-MFMA there).
- **92.41% < BF8's 99.72%** because of the extra scale pipeline (GR→LW→LR) and LDS port contention — expect
  the scale movement, not the MFMA, to be the bottleneck.
- **Manual scheduling cost** — two pipelines (data + scale) to hand-author and balance.
- **Large-K compute-bound shape** (K=32768); skinny/decode MXFP4 GEMM is a different regime.

## How to verify (bench + oracle)
```bash
cd kernels/gemm/a4w4
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 9 --K 32768 --dtype mxfp4   # ~5255 TFLOPS @ 92.41%
# parity: MXFP4 vs higher-precision reference; Quark FP4 accuracy gate; e2e-gate via aiter seam
```

## Alternatives / cross-links
[[operators/quant_fp4_mxfp/backends/aiter]] (production MXFP4 dispatch) ·
[[operators/quant_fp4_mxfp/backends/ck]] · [[operators/quant_fp4_mxfp/backends/triton]] ·
[[operators/scaled_quant_gemm/backends/gluon]] (BF8) · [[operators/dense_gemm/backends/gluon]] (FP16) ·
[[languages/gluon]] · [[languages/triton_amd]] · [[optimization/mfma_scheduling]]

## Sources
- From Naive to Near-Peak: GEMM Kernels with Gluon (MXFP4 a4w4 5255 @ 92.41%, 4096×4096×32768, native scaled-MFMA, scale pipeline GR→LW→LR, MI355, ROCm 7.0): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Gluon GEMM tutorial code (a4w4 256×256×256, 16-cycle MFMA): https://github.com/ROCm/gfx950-gluon-tutorials
- CDNA4 GEMM scheduling (ping-pong): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- 4-wave interleave FP8 GEMM: https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
