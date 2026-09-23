---
title: RDNA 3.5 / Strix Halo / Radeon 8060S (gfx1151) — architecture overview
kind: hardware
gens: [gfx1151]
dtypes: [fp32, bf16, fp16, int8, int4]
regimes: [both]
updated: 2026-09-23
sources:
  - GEAK/kernel_workflow/knowledge/amd_rdna.md (measured reference, this repo)
  - GEAK/kernel_workflow/knowledge/repro/{bw.py,wmma_check.py,triton_caps.py,rdna_roofline.py,noise.py}
  - on-box rocminfo / rocprofv3, ROCm 7.2.3, torch 2.11.0, Triton 3.6.0
---

# RDNA 3.5 / Strix Halo / Radeon 8060S (gfx1151) — architecture overview

> Target: **AMD Ryzen AI MAX+ 395 / Radeon 8060S iGPU**, RDNA 3.5, ISA **gfx1151**.
> Matrix path in [matrix_core_wmma.md](matrix_core_wmma.md); memory hierarchy in
> [memory.md](memory.md); peaks and the scoring bracket in [peak_tables.md](peak_tables.md);
> clocks/power/noise in [clocks_power.md](clocks_power.md); ISA in [isa_notes.md](isa_notes.md);
> profiler defects in [tooling_and_profiling.md](tooling_and_profiling.md).

## TL;DR
> RDNA is **not a smaller CDNA**. Different matrix instruction (**WMMA**, not MFMA), **no AGPRs**,
> **2 SIMDs/CU** and **wave32**, and an APU memory system: **LPDDR5X shared with the CPU** at
> `[measured]` **229–233 GB/s**, fronted by a **32 MB last-level (Infinity) cache** that runs
> `[measured]` **~790 GB/s**. That 3.4× cache/DRAM step is the single largest lever on this part and
> **has no CDNA analogue**. Compute is small: fp16 WMMA datasheet **59.4 TFLOP/s**, best measured
> **38.35–38.52**.

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **32** | CDNA is 64; wave64 is *possible* on RDNA but is not the default |
| CUs | **40** | ~6× fewer than MI350X — grids saturate at small shapes, see levers |
| SIMDs/CU | **2** | CDNA has 4; occupancy math differs |
| Wave slots | **16/SIMD** cap | with 1536 VGPR/SIMD, granule 24 |
| VGPR | 1536 ×4 B/SIMD, granule 24 | **no AGPRs** — every spill goes to VGPR/scratch |
| Matrix instruction | **WMMA** (`v_wmma_*`) | `v_mfma` does not exist here |
| L1 | 32 KB | |
| L2 | **2 MB** | third level — see [memory.md](memory.md) |
| Infinity Cache (MALL) | **32 MB** | **the headline lever**; 3.4× DRAM bandwidth |
| Memory | **LPDDR5X, UMA, shared with the CPU** | 96 GB carve-out of 128 GB; a busy CPU steals GPU bandwidth |
| DRAM BW | 256 GB/s `[vendor]` / **229–233 GB/s `[measured]`** | paper figure is ~12% optimistic |
| LLC BW | **~790 GB/s `[measured]`** (read-read-write) | pure-read goes higher: 782/913/945 |
| fp16/bf16 WMMA peak | **59.4 TFLOP/s `[vendor]`** / **38.35–38.52 `[measured]`** | ratio 64.7% — quote the bracket |
| Engine clock | ~2.9 GHz | basis of the 59.4 peak math |
| Block-scaled FP4/FP6/MXFP | **absent** | no CDNA4 microscaling path; see [matrix_core_wmma.md](matrix_core_wmma.md) |
| Noise floor | **2σ = 0.4% `[measured]`** | on a quiet box; treat <0.4% as noise |

`gfx1150` / `gfx1152` are the same family. `gfx1100` (RDNA 3, discrete, GDDR6) and `gfx120x`
(RDNA 4) have different cache hierarchies and memory — **detect, do not assume**.

## Concepts

### This is an APU, and that changes the memory model
The "VRAM" is a carve-out of system LPDDR5X, not a dedicated pool. Consequences that do not exist
on any Instinct part: the CPU and GPU **share both the memory bandwidth and the power envelope**,
`amd-smi` misreports the memory type (see [tooling_and_profiling.md](tooling_and_profiling.md)),
and an out-of-band idle check is mandatory before any measurement.

### Occupancy vs VGPRs/thread
2 SIMDs/CU, wave32, 1536 VGPR/SIMD, allocation granule 24, hard cap 16 waves/SIMD:

| VGPRs/thread | ≤96 | 120 | 144 | 168 | 192 | 216 |
|---|---:|---:|---:|---:|---:|---:|
| waves/SIMD | 16 | 12 | 10 | 9 | 8 | 7 |

(Source: `perf_knowledge/expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py`, verified
against ROCm 7.2.1 / LLVM 22.)

### 40 CUs is few enough that grid shape becomes a first-order lever
On a 256-CU CDNA part the advice is "≥1024 workgroups". Here the failure mode is the opposite:
small shapes **underfill the device** and the kernel is latency-bound, not bandwidth- or
compute-bound. `[measured]` a 128³/256³/512³ fp16 GEMM moved device fill from **2.5 / 10 / 40%**
of the CUs to **20 / 100 / 100%** purely by re-tiling for CTA count rather than instruction count,
worth **2.53× geomean** against the frozen baseline. Check device fill before touching the inner loop.

## The levers
1. **Get the hot working set under 32 MB.** Worth up to 3.4× on bandwidth-bound work, and it has no
   CDNA equivalent. But check the arithmetic-intensity window first — see [memory.md](memory.md);
   for high-AI kernels this boundary is worth nothing.
2. **Tile for CTA count, not instruction count.** With 40 CUs, underfill is the common small-shape
   bug. `[measured]` 2.53× on small square fp16 GEMM.
3. **Fuse multi-pass row kernels.** `[measured]` softmax 2.21×, add+rmsnorm 1.81× — both by
   collapsing 3 dispatches into 1 and hitting the byte-traffic contract floor (99.4–99.8% of the
   measured DRAM wall).
4. **Keep write-once output out of the LLC** when it will not be re-read — it evicts the stream you
   actually want resident. `[measured]` part of the 1.81× add+rmsnorm win.
5. **Split KV / finer work units to fill the residency ceiling** rather than giving each wave more
   work. `[measured]` GQA decode 2.92×, ending at 83–86% of a demonstrated pure-read ceiling.
6. **Translate or discard every MFMA/AGPR-phrased strategy.** They do not apply.

## Pitfalls
- **Porting a CDNA strategy unchanged.** MFMA shapes, AGPR budgeting, wave64 lane arithmetic and
  "≥1024 workgroups" are all wrong here.
- **Assuming "RDNA implies WMMA".** `gfx10` is RDNA with **no** matrix instruction. Check the
  capability, do not infer it from the family.
- **Routing on wave size.** RDNA *can* run wave64. Route on the `gfx` prefix; use wave size as a
  consistency check only.
- **Building roofline on 256 GB/s.** A kernel already at ~90% of the real 229–233 GB/s wall looks
  like it has 20% headroom against the paper figure. Chasing it is wasted budget.
- **Carrying a "the vendor library is weak on RDNA" prior.** It is not. `[measured]` the vendor path
  wins in 3 of 4 regimes tested; an earlier claim to the contrary in this repo was a benchmark
  artifact and was retracted. See [peak_tables.md](peak_tables.md).

## Verify
```bash
# rocminfo lists the CPU agent FIRST; scope every field to the gfx agent or you get CPU numbers.
rocminfo | awk '/^ *Name: *gfx/{print $2; exit}'                        # -> gfx1151
rocminfo | awk '/Name:.*gfx/{f=1} f&&/Compute Unit:/{print $3; exit}'   # -> 40
rocminfo | awk '/Name:.*gfx/{f=1} f&&/Wavefront Size:/{print $3; exit}' # -> 32
```
Do **not** read memory type or bandwidth from `amd-smi` on this part — it reports `GDDR7` for
LPDDR5X. See [tooling_and_profiling.md](tooling_and_profiling.md).

## Sources
- `GEAK/kernel_workflow/knowledge/amd_rdna.md` — the measured reference this file condenses; every
  `[measured]` number here is traceable to a named script there.
- `GEAK/kernel_workflow/knowledge/learned/_inbox/team-*-kernel-*.json` — the six gfx1151 kernel
  campaigns the lever numbers come from (director-verified, frozen-baseline A/B).
- On-box: ROCm 7.2.3, torch 2.11.0, Triton 3.6.0, `rocminfo`, `rocprofv3`.
- Serving-level (e2e) counterpart: `GEAK/e2e_workflow/knowledge/gemm_tuning/gfx1151_gemm_tuning.md`
  (which backends exist, the two measured GEMM levers) and
  `GEAK/e2e_workflow/knowledge/analysis_skills/roofline/peaks.md` (roofline denominators, incl. the
  32 MB MALL as a third roof).
