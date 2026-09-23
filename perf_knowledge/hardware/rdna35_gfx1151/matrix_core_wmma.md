---
title: gfx1151 matrix path — WMMA, no AGPRs, and what RDNA does NOT have
kind: hardware
gens: [gfx1151]
dtypes: [bf16, fp16, int8, int4]
regimes: [both]
updated: 2026-09-23
sources:
  - GEAK/kernel_workflow/knowledge/repro/triton_caps.py (Triton 3.6.0 / torch 2.11.0 / ROCm 7.2.3)
  - GEAK/kernel_workflow/knowledge/repro/rdna_roofline.py (disassembly gate)
  - GEAK/kernel_workflow/knowledge/amd_rdna.md §2, §6
---

# gfx1151 matrix path — WMMA

> Peaks in [peak_tables.md](peak_tables.md); register/occupancy model in [arch.md](arch.md);
> assembly conventions in [isa_notes.md](isa_notes.md).

## TL;DR
> The matrix instruction is **WMMA** (`v_wmma_*` / `__builtin_amdgcn_wmma_*`), **not MFMA**. There
> are **no AGPRs**. And the CDNA4 microscaling stack — FP4/FP6, block-scaled MXFP8/6/4, E8M0 scales,
> `v_mfma_scale_f32_*_f8f6f4` — **does not exist on this part at all**. Any strategy phrased in
> MFMA shapes, AGPR budgeting or MXFP block scales must be translated or dropped.

## RDNA vs CDNA, the deltas that change kernel code

| | CDNA (`gfx942`/`gfx950`) | **RDNA (`gfx11xx`)** |
|---|---|---|
| Wavefront | 64 | **32** |
| SIMDs per CU | 4 | **2** |
| Matrix instruction | MFMA | **WMMA** |
| AGPRs | yes (≤256) | **none** — every spill goes to VGPR/scratch |
| Mixed-sign dot4 | `sdot4` only | **`sudot4` available** |
| LLC | small relative to HBM | **32 MB, and it dominates** — see [memory.md](memory.md) |
| Memory | dedicated HBM, 5.3–8 TB/s | **LPDDR5X shared with the CPU**, ~0.23 TB/s |

## What is absent (negative knowledge — this is the expensive half)
The following are CDNA4 features with **no gfx1151 equivalent**. A plan that assumes any of them is
not merely slower here, it will not compile:

- **FP4 (E2M1) / FP6 (E2M3, E3M2) matrix instructions** — absent.
- **Block-scaled MXFP8 / MXFP6 / MXFP4 with E8M0 per-32-element scales** — absent. There is no
  `v_mfma_scale_f32_*_f8f6f4` analogue. Low-precision work here means int8/int4 dot paths or
  software dequantisation feeding WMMA, not a hardware microscaling pipe.
- **FP8 matrix (OCP or FNUZ)** — no WMMA fp8 path on this part.
- **AGPRs** — so the CDNA trick of parking accumulators in AGPRs to relieve VGPR pressure has no
  counterpart; on RDNA that pressure lands on the 1536 VGPR/SIMD budget directly and costs
  occupancy per the table in [arch.md](arch.md).
- **Read-with-transpose `ds` loads** (CDNA4's `ds_read_tr*`) — do not plan a B-operand layout
  around them.

`[measured]` A low-precision win is still available *without* hardware microscaling: an int4
weight-only decode GEMM reached **54.78× geomean** over its frozen naive baseline (22.6× at M=1,
133.1× at M=8, N=K=4096, block_size=32) — but the lever was **fixing the wave mapping first**, not
the dequant algebra. Order matters: the mapping bug dominated everything else.

## Triton on RDNA — verified working
`[measured]` `repro/triton_caps.py`, Triton 3.6.0 / torch 2.11.0 / ROCm 7.2.3: elementwise,
reduction/softmax, **`tl.dot` fp16**, **`tl.dot` bf16**, `triton.autotune`, atomics — **all pass**.

`tl.dot` lowers to genuine matrix instructions: generated AMDGCN for a 2048³ fp16 GEMM contains
**128 × `v_wmma_f32_16x16x16_f16`, zero `v_mfma`, zero scalar `v_fma`**.

## The levers
1. **Grep for `v_wmma`, not `v_mfma`.** Seeing no `v_mfma` is *expected* and is not a fallback.
2. **Pick the compute peak by the matrix instruction the kernel issues, not by its tensor dtype.**
   `rdna_roofline.py` disassembles first and refuses to score anything showing `v_mfma` or no
   `v_wmma`, because a CDNA peak table applied here would be silently wrong.
3. **Budget VGPRs against occupancy explicitly** — with no AGPR relief, the 96/120/144/168 VGPR
   cliffs in [arch.md](arch.md) are the whole story.
4. **Fix wave mapping before instruction-level algebra** on quantized paths.

## Pitfalls
- **"RDNA implies WMMA."** `gfx10` is RDNA with **no** matrix instruction at all. Ask what the
  matrix path *is* — `mfma`, `wmma`, or `none` — and check the capability rather than inferring it
  from the family.
- **Routing on wave size.** RDNA supports wave32 **and** wave64; a wave64 kernel on `gfx11` is not
  a CDNA kernel. Use the compiled wave size as a *check*; if family and wave mode disagree, stop
  and re-detect rather than guessing.
- **Porting an MXFP/FP4 plan from gfx950.** Nothing in that stack exists here.
- **Assuming AGPR-free means more registers.** It means fewer places to put them.

## Verify
```bash
# after compiling, on the generated AMDGCN
grep -c 'v_wmma' kernel.amdgcn      # expect > 0
grep -c 'v_mfma' kernel.amdgcn      # expect 0 — nonzero means you are not on this part
```
`rocprofv3` kernel names also carry the ISA: a gfx1151 Tensile kernel shows `..._ISA1151_...` and
its matrix shape as `MI16x16x16x1`.

## Sources
- `GEAK/kernel_workflow/knowledge/repro/triton_caps.py`, `rdna_roofline.py` — on-box.
- `GEAK/kernel_workflow/knowledge/amd_rdna.md` §2, §6.
- Contrast set: `../cdna4_mi350/matrix_core_blockscale.md`, `../cdna4_mi350/fp4_fp6_microscaling.md`
  — read these to see exactly what is *not* available here.
