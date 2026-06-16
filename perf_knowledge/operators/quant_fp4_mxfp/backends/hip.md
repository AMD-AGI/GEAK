---
title: quant_fp4_mxfp on hip — SOTA card
kind: sota_card
operator: quant_fp4_mxfp
backend: hip
gens: [gfx950]
dtypes: [mxfp4, fp4_e2m1]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/quant_kernels.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/quant.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# quant_fp4_mxfp × hip

## TL;DR
HIP is the editable source for the MXFP4 cast: `dynamic_per_group_scaled_quant_fp4` and the MoE
`mxfp4_quant_moe_sort_kernel` in `csrc/kernels/quant_kernels.cu`, using the `__amd_fp4x2_storage_t` /
`__amd_create_fp4x2` helpers from `hip_ext_ocp.h` (hardware-accelerated on gfx950). Use HIP for a custom
MXFP fusion or to control the E8M0 scale shuffle exactly. The cast packs 2 FP4/byte and emits one E8M0 per
32-block; the craft is the per-block amax reduction and the shuffle layout.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `dynamic_per_group_scaled_quant_fp4` | `csrc/kernels/quant_kernels.cu:846` | gfx950, mxfp4 | group-32 amax + e8m0 + pack | the HIP MXFP4 backend |
| `mxfp4_quant_moe_sort_kernel` / `mxfp4_moe_sort_kernel` | `:1618,1828` | gfx950 | sort + MXFP4 (per-group) | MoE A4W4 |
| `__amd_create_fp4x2` / E8M0 helpers | `hip_ext_ocp.h` | gfx950 | the pack/scale primitives | inside any HIP MXFP kernel |

## Config space / knobs
- group size 32; `shuffle_scale` → `e8m0_shuffle`, scale padded `((m+255)//256*256,((n+31)//32+7)//8*8)`.
- `thread_data_size` / block size; `num_rows`/`num_rows_factor` (ragged/MoE).
- `group_size=32` template param on the MoE sort kernel.

## Numerics / parity
E8M0 power-of-2 scale, `FP4_MAX=4`; FP4 packs 2/byte. Scaled MFMA applies the scale after the dot
([[hardware/cdna4_mi350]]). Task-accuracy gate → [[numerics.md]]. Wrong shuffle → corruption.

## Integration (rebind seam)
Edit the `.cu`, rebuild aiter JIT; bound via `compile_ops`. The MXFP4 + E8M0 scales feed the block-scaled
GEMM. Tier-C seam. gfx950-only HW.

## Pitfalls & anti-patterns
- gfx942: `__amd_fp4*` not HW-accelerated; no FP4 MFMA → simulation only.
- Wrong E8M0 layout/shuffle (Ax/Bx) → silent corruption.
- VGPR spill from oversized tiles; `warpSize==64`.

## How to verify
`-Rpass-analysis=kernel-resource-usage`; `amd_matrix_instruction_calculator --get-register`; round-trip
per-block error.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [ck.md](ck.md) · [[languages/hip_cpp]] ·
[[hardware/cdna4_mi350]] · [overview.md](../overview.md).

## Sources
- HIP MXFP4 group quant + MoE sort: `ROCm/aiter@a6bb49937:csrc/kernels/quant_kernels.cu`, `aiter/ops/quant.py`.
- `__amd_fp4x2`, E8M0, scaled MFMA: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
