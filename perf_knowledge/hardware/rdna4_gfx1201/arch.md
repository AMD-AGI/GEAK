---
title: RDNA4 / gfx1201 (R9700, RX 9070 XT) — architecture overview
kind: hardware
gens: [gfx1201]
dtypes: [fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, int8, int4]
regimes: [both]
updated: 2026-09-10
sources:
  - ../../expert_skills/skills/gluon_authoring/references/hardware/hw_constants.json
  - ../../expert_skills/skills/gluon_authoring/references/platform-known-issues.md
  - https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/
---

# RDNA4 / gfx1201 — architecture overview

> Validated client RDNA 4 (`gfx1201`). **Not** Instinct CDNA. Workflow agents should prefer
> [`kernel_workflow/knowledge/amd_rdna4.md`](../../../kernel_workflow/knowledge/amd_rdna4.md).
> Occupancy numbers live in [occupancy.md](occupancy.md); pitfalls in [pitfalls.md](pitfalls.md).
> Do **not** duplicate Gluon `hw_constants.json` beyond the cheat sheet — that file is the source.

## TL;DR
> Wave **32**, matrix ISA **WMMA only** (no MFMA, no scaled/MX, no TDM, no dynamic VGPR).
> LDS 64 KiB/WG (128 KiB/WGP). fp8 is **OCP**, never FNUZ. Use the public R9700
> peaks in the roofline table; do not carry MI300X TFLOPS here. gfx12 also has native
> INT4→INT32 WMMA; do not confuse that with software-dequantized W4A16.

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **32 lanes** | shuffles, `num_warps`, block multiples of 32 |
| Matrix ISA | **WMMA** | `tl.dot` is not MFMA; no VALU co-issue |
| LDS | 64 KiB/WG, 128 KiB/WGP, 32 banks | CDNA4 160 KiB/CU does not apply |
| VGPR/wave | 256 addressable (static) | occupancy from Gluon `vgpr_wave_steps` |
| fp8 | OCP `e4m3fn` / `e5m2` | FNUZ and MXFP4/6/8 are wrong here |
| INT4 | native gfx12 WMMA, INT32 accumulate | verify `v_wmma_i32_16x16x{16,32}_iu4` in ISA |
| Memory | 8 MiB L2 + 64 MiB Infinity Cache + 640 GB/s GDDR6 | cache-resident bandwidth may exceed the external pin rate |

## Concepts
- **Occupancy-first.** Client RDNA wants more resident waves than a typical CDNA MFMA GEMM.
- **No CDNA 512-VGPR combined formula.** Dividing 256 by kernel VGPRs under-reports occupancy.
- **INT4 path identity.** Native gfx12 INT4 WMMA is real, but a framework's `int4_w4a16` label may
  still select an unpack/dequant-to-fp path. Prove the native path from emitted ISA.
- **Roofline placement.** R9700 FP16 ridge is ~298 FLOP/B, between MI300X (~247) and
  MI350/355 (~312); do not assume every MI compute-bound kernel becomes R9700 memory-bound.
- **AITER Navi.** Several Instinct custom kernels are disabled on gfx12; hipBLASLt / Triton are the
  default bake-off, not “must beat AITER CDNA cards.”
- **Roofline peaks.** Public R9700 GDDR6 / WMMA ceilings live in
  [`e2e_workflow/.../roofline/peaks.md`](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md),
  not here.
