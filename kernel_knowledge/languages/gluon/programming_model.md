---
title: Gluon programming model (block-level, manual scheduling) — reference
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

# Gluon programming model

## TL;DR
Gluon keeps Triton's **tile-level** (not thread-level) authoring but hands the programmer the four things
Triton's compiler normally owns: **explicit layouts**, **explicit pipeline stages**, **explicit register
budgeting**, and **direct MFMA intrinsics** (incl. CDNA4 scaled-MFMA). You write the wave schedule
(ping-pong / interleave) yourself instead of getting `num_stages`-driven autoscheduling.

## The abstraction
- **Tile-level, wave/wave-aware.** Like Triton, a Gluon kernel operates on block-shaped tiles. Unlike
  Triton, the **layout** of each tile (registers, LDS, MFMA operand placement) is an explicit object —
  including **swizzled** and **padded** shared (LDS) layouts authored to avoid bank conflicts. CDNA
  wavefront = **64 lanes**; LDS = 64 KB/CU (CDNA3) / 160 KB/CU (CDNA4); 512 VGPR/EU.
- **MFMA intrinsics are first-class.** `tl.dot` in Triton becomes a hand-issued matrix-core op in Gluon.
  The CDNA4 *scaled* MFMA is exposed as `gl.amd.cdna4.mfma_scaled`, lowering to the hardware op
  **`v_mfma_scale_f32_16x16x128_f8f6f4`** (one instruction consumes FP8/FP6/FP4 operands + a block scale).
- **LDS movement is explicit**, including `ds_read_tr` — the transpose variant of `ds_read` — to feed MFMA
  operands in the right layout without an extra transpose.

## Manual pipelining (vs Triton autoscheduling)
| Concern | Triton (autoscheduled) | Gluon (manual) |
|---|---|---|
| Pipeline depth | `num_stages` knob; stream-pipeliner picks placement | **explicit** 2-/3-stage pipeline you author |
| Operand layout | compiler-assigned blocked/MFMA layouts | **explicit** swizzled/padded LDS + register layouts |
| Register pressure | compiler allocates (can spill) | **you** budget live values vs 512-VGPR limit |
| Matrix op | `tl.dot` lowered to MFMA | **you** issue the MFMA intrinsic (incl. `mfma_scaled`) |
| Wave schedule | implicit | **ping-pong / interleave** authored by hand |
| Instruction interleave | compiler | `llirSched` (Triton-IR) + `amdgcnas` (post-asm peephole) |

The interleaving passes are toggled at runtime:
```bash
# enable Gluon's IR-level interleave + post-assembly peephole scheduler
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 9 --K 8192 --dtype fp16
```

## Wave scheduling patterns (why manual)
NVIDIA's producer/consumer **wave specialization underperforms on CDNA3/CDNA4** — AMD's static register
allocation starves the producer waves (only ~80% of peak BF16 GEMM). So Gluon exposes the two patterns that
*do* reach peak on CDNA, both originating in HipKittens:
- **8-wave ping-pong** — two groups of waves alternate MFMA vs memory phases so the matrix core never idles.
- **4-wave interleave** — one wave per SIMD, full 512-VGPR budget, 128×128 tile; the robustness/perf
  successor (no `#pragma unroll` tuning, consistent across ROCm releases).

See [[optimization/mfma_scheduling]] and [[languages/hipkittens]] for the scheduling theory; Gluon is the
Python authoring surface for both.

## MXFP4 numerics (scaled-MFMA path)
MXFP4 packs **two 4-bit values per byte** and uses a **per-group 8-bit scale factor for every 32 elements**
(microscaling block scale). The scaled MFMA consumes the packed FP4 operands and the block scale in one
instruction (`v_mfma_scale_f32_16x16x128_f8f6f4`), accumulating in FP32. The a4w4 kernel adds a dedicated
**scale pipeline (GR → LW → LR)** — global-read → LDS-write → LDS-read of the scales — alongside the data
pipeline; this is the extra complexity that caps MXFP4 at 92.41% (vs ~99% for BF8) due to LDS port
contention. (The tutorial uses the term "8-bit per-group scale"; it does not spell out "E8M0".)

## Sources
- Gluon GEMM tutorial (layouts, pipeline stages, register budget, `mfma_scaled`, `ds_read_tr`, `llirSched`/`amdgcnas`, scale pipeline GR→LW→LR): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Tutorial code (a16w16 256×256×64, a8w8 256×256×128, a4w4 256×256×256): https://github.com/ROCm/gfx950-gluon-tutorials
- Wave-specialization-fails-on-CDNA / ping-pong + interleave origin: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
