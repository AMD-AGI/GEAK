---
title: RDNA4 / gfx1201 ISA — architecture overview
kind: hardware
gens: [gfx1201]
dtypes: [fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, int8, int4]
regimes: [both]
updated: 2026-10-07
sources:
  - ../data/hw_constants.json
  - ../../expert_skills/skills/gluon_authoring/references/pitfalls/platform-known-issues.md
  - https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/
---

# RDNA4 / gfx1201 — architecture overview

> Validated client RDNA 4 (`gfx1201`). **Not** Instinct CDNA. Workflow agents should prefer
> [`kernel_workflow/knowledge/amd_rdna4.md`](../../../kernel_workflow/knowledge/amd_rdna4.md).
> Occupancy numbers live in [occupancy.md](occupancy.md); pitfalls in [pitfalls.md](pitfalls.md).
> Do **not** duplicate [`hw_constants.json`](../data/hw_constants.json) (GEAK's shared hardware data)
> beyond the cheat sheet — that file is the source.
> These ISA facts apply to gfx1201 products. GEAK's product validation, serving
> image, and calibrated peaks exist for **R9700** and **R9600D** (see
> `roofline/peaks.md`); RX 9070 XT and other gfx1201
> products remain uncalibrated.

## TL;DR
> Wave **32**, matrix ISA **WMMA only** (no MFMA, no scaled/MX, no TDM). GEAK
> uses static VGPR occupancy for HIP/Triton; do not mistake that tooling policy
> for ISA rejection of `S_ALLOC_VGPR`.
> LDS 64 KiB/WG (128 KiB/WGP). fp8 is **OCP**, never FNUZ. Use the public R9700
> peaks in the roofline table; do not carry MI300X TFLOPS here. gfx12 also has native
> INT4→INT32 WMMA; do not confuse that with software-dequantized W4A16.

## The one-screen cheat sheet
| Fact | Value | Why it matters |
|---|---|---|
| Wavefront | **32 lanes** | shuffles, `num_warps`, block multiples of 32 |
| Matrix ISA | **WMMA** | `tl.dot` is not MFMA; no VALU co-issue |
| LDS | 64 KiB/WG, 128 KiB/WGP, 32 banks | CDNA4 160 KiB/CU does not apply |
| VGPR/wave | 256 addressable (static) | occupancy from `hw_constants.json` `vgpr_wave_steps` |
| fp8 | OCP `e4m3fn` / `e5m2` | FNUZ and MXFP4/6/8 are wrong here |
| INT4 | native gfx12 WMMA, INT32 accumulate | verify `v_wmma_i32_16x16x{16,32}_iu4` in ISA |
| Memory (R9700, RX 9070 / XT) | 8 MiB L2 + 64 MiB Infinity Cache + 640 GB/s GDDR6 | cache-resident bandwidth may exceed the external pin rate |

R9700, RX 9070 XT, RX 9070 and R9600D all report **gfx1201**. Product clocks,
memory capacity, power and matrix peaks differ; use the exact SKU. R9600D has
8 MiB L2 and 640 GB/s external bandwidth, but its Infinity Cache capacity is
unresolved: the [AMD product page](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9600d.html)
lists 64 MB while the [ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
list 48 MiB. `sku.json` records both sources and leaves `mall_mb` unknown.

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
