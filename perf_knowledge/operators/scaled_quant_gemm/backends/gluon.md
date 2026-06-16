---
title: scaled_quant_gemm on Gluon — SOTA card
kind: sota_card
operator: scaled_quant_gemm
backend: gluon
gens: [gfx950]
dtypes: [fp8_e5m2, fp8_e4m3]
regimes: [prefill, training]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
---

# scaled_quant_gemm × Gluon

## TL;DR (one-line decision)
> On **CDNA4 (gfx950)** Gluon authors a **near-peak BF8 (a8w8) GEMM at 3257 TFLOPS @ 99.72% MFMA eff**
> (4096×4096×16384) — essentially the matrix-core ceiling, via CDNA4 scaled-MFMA. Use it to push FP8/BF8
> GEMM past hipBLASLt/aiter when you can hand-schedule; otherwise hipBLASLt (~3130 @ 8192) is the no-tune bar.

## SOTA implementation(s)
Best known BF8 GEMM in a Python DSL on gfx950 is AMD's Gluon `a8w8` kernel — same v0→v9 design as the FP16
a16w16 tutorial, with a larger `BLOCK_K` and the 32-cycle scaled MFMA (cite the Gluon GEMM tutorial).

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| Gluon `a8w8` | `ROCm/gfx950-gluon-tutorials:kernels/gemm/a8w8`; Gluon GEMM tutorial | gfx950; BF8 (a8w8); 256×256×128 tile; 4096×4096×16384 | **3257 TFLOPS @ 99.72% MFMA eff** @ MI355X gfx950, ROCm 7.0, AMD-measured, 2026 | near-peak BF8 GEMM with manual scheduling |

Context on the same HW (FP8, large K): HIP/C++ 8-wave ping-pong hits **3204 TFLOPS** @ 8192 (beats
hipBLASLt 3130, no asm); HipKittens 4-wave FP8 **3327 TFLOPS** (183 LoC). Gluon's 3257 @ 99.72% sits in this
near-peak band as the DSL authoring path. See [[languages/hipkittens]].

## Config space / knobs
- **Tile**: `256×256×128` (M×N×K) for a8w8 — same skeleton as FP16, larger `BLOCK_K`.
- **Wave schedule**: **ping-pong (8-wave)** / **interleave (4-wave)** — the lever to 99.72% MFMA eff;
  authored by hand. See [[optimization/mfma_scheduling]].
- **num_stages / pipeline**: explicit 2-/3-stage software pipeline (GR→LW→LR→MFMA); 32-cycle scaled MFMA.
- **LDS layout**: swizzled/padded shared layouts + `ds_read_tr`.
- **Register budget**: live values within 512-VGPR/EU.
- **Scheduling passes**: `TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1`.

## Numerics / parity
FP8/BF8 operands, **FP32 accumulate**, scaled-MFMA applies the scale in-instruction. Gate FP8 GEMM accuracy
vs a higher-precision reference (the usual block/per-tensor scale parity check); CDNA4 uses OCP fp8 (CDNA3
is FNUZ — relevant if porting back to gfx942).

## Integration (rebind seam)
Authoring path, not a production dispatcher. Wire a Gluon BF8 GEMM via the aiter `triton`/scaled-GEMM seam
or a call-site rebind, then e2e-gate. Relationship: Triton/Gluon = authoring (Gluon = low-level dialect for
the last 20–75%); aiter/hipBLASLt = the production scaled-GEMM dispatch the authored kernel competes with /
plugs into. See [[operators/scaled_quant_gemm/backends/aiter]] and
[[operators/scaled_quant_gemm/backends/hipblaslt]].

## Pitfalls & anti-patterns
- **gfx950 headline.** 3257 is CDNA4; scaled-MFMA op is CDNA4-native. CDNA3 (gfx942) runs FP8 manual
  pipelines but is not the measured headline and uses FNUZ fp8.
- **Manual scheduling cost** — hand-authored layout/pipeline/register budget/schedule.
- **Large-K compute-bound shape** (K=16384). Skinny/decode FP8 GEMM is a different regime.
- **Not a no-tune default** — hipBLASLt ~3130 @ 8192 is the zero-effort bar to clear.

## How to verify (bench + oracle)
```bash
cd kernels/gemm/a8w8
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 9 --K 16384 --dtype fp8   # ~3257 TFLOPS @ 99.72%
# parity: FP8 GEMM vs hipBLASLt scaled reference; accuracy gate on quant error; e2e-gate via aiter seam
```

## Alternatives / cross-links
[[operators/scaled_quant_gemm/backends/hipblaslt]] (no-tune bar ~3130) ·
[[operators/scaled_quant_gemm/backends/hip]] (8-wave ping-pong 3204, 4-wave interleave) ·
[[operators/scaled_quant_gemm/backends/aiter]] (block-scale + dispatch) ·
[[operators/dense_gemm/backends/gluon]] (FP16) · [[operators/quant_fp4_mxfp/backends/gluon]] (MXFP4) ·
[[languages/gluon]] · [[languages/triton_amd]] · [[optimization/mfma_scheduling]]

## Sources
- From Naive to Near-Peak: GEMM Kernels with Gluon (BF8 a8w8 3257 @ 99.72%, 4096×4096×16384, MI355, ROCm 7.0): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Gluon GEMM tutorial code (a8w8 256×256×128, 32-cycle scaled MFMA): https://github.com/ROCm/gfx950-gluon-tutorials
- FP8 GEMM on CDNA4 — 8-wave ping-pong 3204 (> hipBLASLt 3130): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- Deep dive 4-wave interleave FP8 GEMM: https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
