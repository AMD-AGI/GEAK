---
title: Gluon on AMD Instinct (Triton low-level dialect) — overview
kind: language
gens: [gfx942, gfx950]
dtypes: [fp16, bf16, fp8_e5m2, fp8_e4m3, fp4_e2m1]
regimes: [prefill, training, both]
status: sota
updated: 2026-06-09
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
---

# Gluon on AMD — overview

## TL;DR
Gluon is **Triton's low-level, block-level dialect**: same JIT/Python authoring surface as Triton, but it
makes explicit everything Triton's compiler normally schedules for you — tile **layouts** (incl. swizzled/
padded LDS layouts), **software-pipeline stages**, **register budgeting** against the 512-VGPR/EU limit,
and the **MFMA matrix-core intrinsics** themselves (including the CDNA4 *scaled* MFMA). It is the path AMD
used to reach **near-peak GEMM in a Python DSL** on MI355X (gfx950): FP16 **1489 TFLOPS @ 98.75% MFMA
eff**, BF8 **3257 TFLOPS @ 99.72%**, MXFP4 **5255 TFLOPS @ 92.41%** — vs a naive Gluon baseline of
**520 TFLOPS @ 25%** (~3× to peak). Reach for Gluon when Triton's autoscheduler leaves the last 20–75% on
the table and you are willing to hand-place the pipeline; stay on Triton for portability and fast iteration.

## What Gluon is (vs Triton)
- **Same model, lower level.** AMD's tutorial calls Gluon *"a block-level programming model in Triton."*
  Authoring is still tile-level (not thread-level), `@gluon.jit`-style kernels compile through the same
  Triton → TritonGPU → TritonAMDGPU → AMDGCN pipeline. The difference is *who decides* the layout,
  pipeline depth, and instruction schedule: in Triton the compiler does; in Gluon **you** do.
- **What Gluon makes explicit** (per the GEMM tutorial):
  - **Layouts** — tile layouts including swizzled and padded shared (LDS) layouts to kill bank conflicts.
  - **Pipeline stages** — explicit two-/three-stage software pipelining (no autoscheduled `num_stages`).
  - **Register budgeting** — live values managed by hand against the **512-VGPR** budget.
  - **MFMA intrinsics** — incl. the CDNA4 scaled variant exposed as `gl.amd.cdna4.mfma_scaled`
    (hardware op `v_mfma_scale_f32_16x16x128_f8f6f4`).
  - **LDS control** — incl. `ds_read_tr`, the transpose variant of `ds_read`.
  - **Scheduling passes** — `llirSched` (Triton-IR interleaving) and `amdgcnas` (post-assembly peephole),
    enabled at runtime via `TRITON_ENABLE_LLIR_SCHED=1` / `TRITON_ENABLE_AMDGCN_AS=1`.

## Manual scheduling = the two CDNA wave patterns
Gluon is the authoring surface for the scheduling patterns that win on CDNA: **8-wave ping-pong** and
**4-wave interleave** (both originate from HipKittens; see [[languages/hipkittens]] and
[[optimization/mfma_scheduling]]). NVIDIA-style producer/consumer **wave specialization underperforms on
CDNA3/CDNA4** (static register allocation starves producers), so Gluon exposes ping-pong/interleave instead
of warp-specialization as the route to peak.

## Maturity / hardware support
- **CDNA4 (gfx950, MI350X/MI355X)** — headline target; all of the near-peak numbers above are gfx950,
  ROCm 7.0, AMD-measured. Native scaled-MFMA (`mfma_scaled`) is a CDNA4 feature.
- **CDNA3 (gfx942, MI300X/MI325X)** — Gluon runs (same Triton backend, ping-pong/interleave apply), but
  CDNA3 lacks the native `v_mfma_scale_*` op, so MXFP4 scaled-MFMA is a CDNA4-only headline. Use CDNA3
  Gluon for FP16/BF16/FP8 manual-pipeline GEMM.
- **Status: experimental→productized DSL** shipped as a Triton experimental dialect; AMD ships a full
  tutorial repo (`ROCm/gfx950-gluon-tutorials`, MIT) with v0→v9 GEMM kernels.

## When to use Gluon vs Triton
| Use Gluon when | Stay on Triton when |
|---|---|
| Triton's autoscheduler leaves >20% on the table on a hot GEMM | Portability across NVIDIA/AMD matters |
| You need explicit ping-pong/interleave wave scheduling | Rapid prototyping / shape exploration |
| You need CDNA4 native scaled-MFMA (MXFP4 a4w4) | Fused epilogue/attention the library can't express |
| You can budget time for hand layout + pipeline tuning | `torch.compile` / Inductor codegen path |

## Deep-dive map
- [programming_model.md](programming_model.md) — the abstraction: warp/wave-level control, manual
  pipelining, MFMA intrinsics, MXFP4 scaled-MFMA, how it differs from Triton autoscheduling.
- [gemm_cookbook.md](gemm_cookbook.md) — the near-peak GEMM recipe (v0→v9) + the measured ceilings.

## Sources
- From Naive to Near-Peak: GEMM Kernels with Gluon (MI350/MI355, ROCm 7.0): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Gluon GEMM tutorial code (v0→v9, a16w16/a8w8/a4w4, MIT): https://github.com/ROCm/gfx950-gluon-tutorials
- Scheduling-pattern origin (ping-pong/interleave, wave-spec fails on CDNA): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
