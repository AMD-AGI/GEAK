---
title: RDNA3.5 / gfx1151 (Strix Halo) — ISA notes
kind: hardware
gens: [gfx1151]
dtypes: [fp16, int8]
regimes: [both]
updated: 2026-11-06
sources:
  - ../../../kernel_workflow/knowledge/amd_ryzen.md
  - repro/wmma_check.py
  - repro/blas_backend.sh
---

# RDNA3.5 / gfx1151 — ISA notes

> The register file, LDS and wave size are in [`amd_ryzen.md` §1](../../../kernel_workflow/knowledge/amd_ryzen.md).
> This card lists what is useful when reading or writing gfx1151 code beyond that table. The
> instruction checks are `llvm-mc` one-liners in [repro/README.md](repro/README.md).

## TL;DR
> No AGPR tier, so register pressure spills to scratch in the shared LPDDR5X. Mixed-sign int8 dot
> products exist both as a VALU dot4 and in WMMA. Vendor kernel names carry their target and matrix
> instruction, which is how a measured peak is tied to an instruction.

## Concepts
- **Spills are expensive here.** With no AGPRs, accumulator pressure lands in the VGPR file and then
  in scratch, which is the same LPDDR5X the kernel is streaming (~232 GB/s, [peak_tables.md](peak_tables.md)).
- **Mixed-sign int8 needs no correction pass.** gfx1151 has `v_dot4_i32_iu8` besides
  `v_dot4_i32_i8` / `v_dot4_u32_u8`; gfx90a / gfx942 / gfx950 have only the same-sign two. The matrix
  path takes it too: `v_wmma_i32_16x16x16_iu8` marks an operand signed with `neg_lo`.
- **Tensile kernel names carry the ISA.** `rocprofv3` shows `torch.mm` fp16 running
  `Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_..._ISA1151_...`: fp16 in / fp32 accumulate (`HHS`),
  macro tile 128×128×16, a 16×16×16 matrix instruction, built for gfx1151. That name is in rocBLAS's
  Tensile library ([`repro/blas_backend.sh`](repro/blas_backend.sh)); hipBLASLt names its kernels
  differently (`..._UserArgs_MT..._MI16x16x1_...`). `ISA000` and `_FMA_` in a name mark a kernel
  without matrix instructions.

## The levers
1. Budget VGPRs against the occupancy steps rather than relying on spill headroom.
2. For unsigned activation × signed weight int8, use the mixed-sign dot4 or WMMA `iu8` form instead
   of a bias-correction pass.

## Pitfalls
- Treating a spill as cheap because the VGPR file is large.
- Trusting a peak whose kernel cannot be identified by name or by disassembly.

## Verify
- `rocprofv3 --kernel-trace` for vendor kernel names; the disassembly greps in
  [matrix_core.md](matrix_core.md) for your own kernels.
- `scratch` / `private_segment_size` in the compiled kernel metadata for spills.

## Sources
- `llvm-mc` (ROCm 7.2.3) and `rocprofv3` on gfx1151; [`repro/blas_backend.sh`](repro/blas_backend.sh),
  [`repro/wmma_check.py`](repro/wmma_check.py).
