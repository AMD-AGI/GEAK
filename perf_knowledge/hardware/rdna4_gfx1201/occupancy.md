---
title: RDNA4 / gfx1201 — occupancy
kind: hardware
gens: [gfx1201]
dtypes: []
regimes: [both]
updated: 2026-09-21
sources:
  - ../../expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json
  - ../../expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py
---

# RDNA4 / gfx1201 — occupancy

Canonical constants: Gluon `hw_constants.json` key `gfx1201`. Re-measure after a
ROCm/LLVM bump with
`amd_occupancy.py --compiler-sweep --arch gfx1201 --format json`.

The table below was regenerated on **ROCm 10 / AMD clang 23.0.0git**
(`llvm-project` `8f497e0992fb7513f7f78a6f6b6f1056c375e961`) in the pinned
public rocm/vllm ROCm 10 image. All breakpoints matched the
earlier ROCm 7.2.1 / LLVM 22 sweep.

## TL;DR
> Cap is **16 waves/SIMD**. VGPR allocation granule 24. Static ≤256 VGPR/wave — `S_ALLOC_VGPR` is
> rejected on gfx1201. `vgpr_file_per_simd` = 1536; **do not** treat 256 as the SIMD file.

## Compiler-derived VGPR → waves/SIMD (ROCm 10 / LLVM 23)

| VGPRs/wave | Max waves/SIMD |
|------------|----------------|
| 96 | 16 |
| 120 | 12 |
| 144 | 10 |
| 168 | 9 |
| 192 | 8 |
| 216 | 7 |
| 240 | 6 |
| 256 | 5 |

LDS: 64 KiB per workgroup. `nW` = `ceil(threads_per_block / 32)` on this family.
