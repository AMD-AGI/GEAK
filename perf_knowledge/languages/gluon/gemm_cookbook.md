---
title: Gluon near-peak GEMM cookbook (CDNA4) — reference
kind: language
gens: [gfx950]
dtypes: [fp16, fp8_e5m2, fp4_e2m1]
regimes: [prefill, training, both]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
---

# Gluon GEMM cookbook (CDNA4, gfx950)

## TL;DR
AMD's `gfx950-gluon-tutorials` walks a single FP16 GEMM from **naive 520 TFLOPS @ 25%** (v0) to
**near-peak 1489 TFLOPS @ 98.75% MFMA eff** (v9) in nine steps, then reuses the same design for BF8 and
MXFP4. The recipe = explicit LDS layout → software pipeline → register budgeting → MFMA scheduling
(interleave) → post-hot-loop cleanup. The measured ceilings below are the **practical low-precision limits**
on MI355X.

## The recipe (a16w16 FP16, v0 → v9)
The tutorial's FP16 kernel introduces every technique; BF8 and MXFP4 reuse the same skeleton with a larger
`BLOCK_K` and the scaled-MFMA op. Progression (v0→v9, `kernels/gemm/a16w16/`):
1. **v0 naive** — straight tiled GEMM, 520 TFLOPS @ 25%. Matrix core idle most of the time.
2. **Explicit LDS layouts** — swizzled/padded shared layouts to remove bank conflicts; `ds_read_tr` to feed
   MFMA operands transposed without an extra pass.
3. **Software pipeline** — explicit 2-/3-stage pipeline (global-read → LDS-write → LDS-read → MFMA),
   authored by hand instead of via Triton `num_stages`.
4. **Register budgeting** — keep live values inside the 512-VGPR budget so occupancy/no-spill holds.
5. **MFMA scheduling** — interleave MFMA with memory so the matrix core never stalls (the CDNA
   ping-pong / 4-wave-interleave patterns; see [[optimization/mfma_scheduling]]).
6. **Beyond the hot loop (v9)** — `llirSched` IR interleave + `amdgcnas` post-assembly peephole, enabled
   with `TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1`.

Tile shapes per dtype (from the tutorial repo):
| kernel | dtype | tile (M×N×K) | notes |
|---|---|---|---|
| a16w16 | FP16 | 256×256×64 | foundation; all techniques introduced here |
| a8w8 | BF8 | 256×256×128 | same design, larger `BLOCK_K`, 32-cycle scaled MFMA |
| a4w4 | MXFP4 | 256×256×256 | adds scale pipeline GR→LW→LR, 16-cycle MFMA, LDS port contention |

Benchmark invocation:
```bash
cd kernels/gemm/a16w16
python bench.py --version 0 --K 8192 --dtype fp16     # naive: 520 TFLOPS @ 25%
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 9 --K 8192 --dtype fp16  # near-peak: 1489 TFLOPS @ 98.75%
```

## Measured ceilings (AMD-measured, single MI355, ROCm 7.0, gfx950, 2026)
| kernel | dtype | shape (M×N×K) | TFLOPS | MFMA eff |
|---|---|---|---|---|
| a16w16 | FP16 | 4096×4096×8192 | **1489** | 98.75% |
| a8w8 | BF8 | 4096×4096×16384 | **3257** | 99.72% |
| a4w4 | MXFP4 | 4096×4096×32768 | **5255** | 92.41% |
| v0 baseline | FP16 | (naive) | 520 | 25% |

FP16 v0→v9 is ~3× to peak. BF8 is essentially peak (99.72%). MXFP4 caps lower (92.41%) because of the extra
**scale pipeline (GR→LW→LR)** and LDS port contention from the 8-bit-per-32-element block scale.

## Numerics
FP16/BF8/FP4 operands, **FP32 accumulate**. MXFP4 = two 4-bit values per byte + an 8-bit per-group scale
(every 32 elements), consumed by `v_mfma_scale_f32_16x16x128_f8f6f4` in one instruction.

## Pitfalls
- The ceilings are **gfx950 headline** — CDNA3 (gfx942) lacks native scaled-MFMA, so the MXFP4 5255 number
  does not transfer. FP16/BF8 manual pipelines do run on gfx942.
- These are **large-K, square-ish shapes** (K=8192/16384/32768). Skinny/decode GEMM is a different regime —
  the near-peak ceilings assume compute-bound dimensions.
- Manual scheduling is the cost: you author the layout, pipeline, register budget, and schedule by hand —
  far more effort than Triton's autoscheduler for the last ~20–75% of peak.

## Cross-links
[[languages/gluon]] · [[languages/triton_amd]] · [[optimization/mfma_scheduling]] ·
[[operators/dense_gemm/backends/gluon]] · [[operators/scaled_quant_gemm/backends/gluon]] ·
[[operators/quant_fp4_mxfp/backends/gluon]]

## Sources
- Gluon GEMM tutorial (v0→v9, ceilings, scale pipeline, env vars): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Tutorial code + per-dtype tile sizes: https://github.com/ROCm/gfx950-gluon-tutorials
- Scheduling patterns (ping-pong / interleave): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html ; https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
