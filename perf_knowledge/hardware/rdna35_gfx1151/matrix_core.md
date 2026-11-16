---
title: RDNA3.5 / gfx1151 (Strix Halo) — matrix core (WMMA) in practice
kind: hardware
gens: [gfx1151]
dtypes: [fp16, bf16, int8, int4]
regimes: [both]
updated: 2026-11-06
sources:
  - ../../../kernel_workflow/knowledge/amd_ryzen.md
  - ../rdna4_gfx1201/pitfalls.md
  - repro/triton_caps.py
  - repro/wmma_check.py
  - repro/ceilings.py
---

# RDNA3.5 / gfx1151 — matrix core (WMMA) in practice

> The WMMA dtypes, wave32 and the absence of an fp8 matrix path are in
> [`amd_ryzen.md` §1](../../../kernel_workflow/knowledge/amd_ryzen.md); the RDNA-wide "no MFMA,
> no MX block-scale, no FNUZ" rules are in [`rdna4_gfx1201/pitfalls.md`](../rdna4_gfx1201/pitfalls.md).
> Peaks are in [peak_tables.md](peak_tables.md), instruction details in [isa_notes.md](isa_notes.md).

## TL;DR
> `tl.dot` reaches real WMMA on this part. What costs performance is **how operands reach the
> matrix unit**: 16-bit LDS loads, a tile sized for a different M, a K-block tied to the
> quantization group. Each lever below is a change to the operand path, not to the instruction.

## Concepts
- **Triton lowers to WMMA.** A 2048³ fp16 `tl.dot` compiles to `v_wmma_f32_16x16x16_f16` with zero
  `v_mfma` and zero scalar `v_fma`; elementwise, reduction, fp16/bf16 `tl.dot`, autotune and atomics
  all work and match a float32 reference (Triton 3.6.0, ROCm 7.2;
  [`repro/triton_caps.py`](repro/triton_caps.py), [`repro/wmma_check.py`](repro/wmma_check.py)).
- **WMMA wants 16 contiguous elements per lane.** When the compiler stages a row-major operand
  K-contiguous in LDS, it feeds WMMA with `ds_load_u16`: one kernel issued 384 of them per 24
  `v_wmma`, and LDS instructions were 39–59% of everything issued, with zero bank conflicts (GEAK
  campaign profile; [repro/README.md](repro/README.md)).
- **40 CUs fill early.** A 128×128 tile occupies 2.5 / 10 / 40% of the CUs at 128³ / 256³ / 512³;
  small GEMMs are latency-bound until the tile is sized for CTA count.

## The levers
1. **Make LDS operand loads 128-bit.** Triton's in-thread transpose
   (`TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1`; on by default only for gfx942) turns the 16-bit loads into
   `ds_load_b128` (432 → 96 LDS loads in one kernel). It is process-wide and part of the Triton cache
   key, so a per-shape choice needs a separate process. `[measured]` with a per-shape tile: 2.99× /
   2.97× / 1.82× at 128³ / 256³ / 512³ fp16; tile changes alone were worth 1.86× at 256³ (GEAK
   campaign; [repro/README.md](repro/README.md)).
2. **Size the M tile for decode.** At M = 8 a 64-row tile computes mostly padding. `[measured]` int8
   W8A8 GEMM, BLOCK_M 64 → 8: 2.28× at qkv decode shape (72 → 164 GB/s), prefill also faster (GEAK
   campaign; [repro/README.md](repro/README.md)).
3. **Decouple the K block from the quantization group.** A K block equal to the group (32) loads
   16 contiguous bytes per iteration. `[measured]` int4 weight-only GEMV with K block ≫ group and a
   two-level reduce: N14336·K4096 went from 15.5 to 200 GB/s (7% → 86% of the 232 GB/s read ceiling
   in [peak_tables.md](peak_tables.md)). The K block must be a power-of-two factor of K, so K = 13824 /
   11008 / 29568 gain less (1.25–1.46×) (GEAK campaign; [repro/README.md](repro/README.md)).
4. **Dequantize in the epilogue.** `[measured]` W8A16 prefill GEMM with the scale in the epilogue
   and each packed int32 word split into four dots reaches 29–30 TFLOP/s, the large-shape fp16
   rate in [peak_tables.md](peak_tables.md) (GEAK campaign; [repro/README.md](repro/README.md)).

## Pitfalls
- Reading "no `v_mfma`" as a fallback. On this part it is the correct state; check for `v_wmma`.
- Padding or swizzling LDS when the bank-conflict counter is zero; the cost is load width, not
  conflicts.
- Tuning `matrix_instr_nonkdim` / `kpack`; see [`amd_ryzen.md` §2](../../../kernel_workflow/knowledge/amd_ryzen.md).

## Verify
```bash
grep -c 'v_wmma' kernel.amdgcn   # > 0 on a matrix kernel
grep -c 'v_mfma' kernel.amdgcn   # 0
```
- [`repro/triton_caps.py`](repro/triton_caps.py), [`repro/wmma_check.py`](repro/wmma_check.py).
- `SQ_INSTS_LDS` against total instructions, and `SQC_LDS_BANK_CONFLICT`, before changing LDS layout.

## Sources
- [`repro/triton_caps.py`](repro/triton_caps.py), [`repro/wmma_check.py`](repro/wmma_check.py) —
  re-run 2026-11-06. The in-thread-transpose switch: Triton 3.6.0 `knobs.py` and
  `backends/amd/compiler.py`.
- Levers 1–4: frozen-baseline A/B campaigns on gfx1151 (GEAK kernel workflow, 2026-09 to 2026-11);
  no script in `repro/`, see [repro/README.md](repro/README.md).
