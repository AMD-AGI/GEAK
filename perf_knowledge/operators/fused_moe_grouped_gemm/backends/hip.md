---
title: fused_moe_grouped_gemm on HIP/asm — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/fused_moe_bf16_asm.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/tuned_fmoe.csv
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# fused_moe_grouped_gemm × HIP/asm

## TL;DR
> HIP/asm is the **stage-1 (gate+up, g1u1)** of aiter's fused MoE — hand-tuned MFMA assembly
> (`fmoe_stage1_*`) that fuses gate+up + SwiGLU activation. It's the editable Tier-C seam for the hottest
> per-expert GEMM and where the bf16 asm fused-MoE path lives (`fused_moe_bf16_asm.py`). Reach for it to
> own the MFMA schedule/LDS layout of stage-1; consume CK for stage-2.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `fmoe_stage1_*` (hand-tuned asm, g1u1) | aiter `tuned_fmoe.csv` (real name `_ZN5aiter48fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf3E`) | gfx942/950; bf16, per-token fp8 | part of up-to-3× fused MoE (AMD-reported) | stage-1 gate+up of fused MoE |
| `fused_moe_bf16_asm` | `aiter/fused_moe_bf16_asm.py` | gfx942/950 bf16 | — | bf16 (no-quant) fused MoE |
| HIP grouped GEMM (authoring) | `languages/hip_cpp/` patterns + MFMA intrinsics | gfx942/950 | — | a custom per-expert GEMM/fusion |

## Config space / knobs
- **MFMA**: `v_mfma_f32_16x16x16` (prefer 16) / 32×32×8; fp32 acc; fp8 via packed convert (`fmed3f` clip).
- **Tile**: stage-1 64×128 (real kernel), `2tg` (2 thread-groups), `pf3` (prefetch depth 3). Size per-expert
  M-tiles to fill 304 CUs.
- **LDS**: 64 KB CDNA3 / 160 KB CDNA4; XOR-swizzle to avoid bank conflicts; `__launch_bounds__` for VGPR.
- **g1u1**: gate+up share the X load; SwiGLU in the epilogue.
- **`ksplit`** for skinny decode per-expert GEMMs.

## Numerics / parity
fp32 acc; per-token fp8 quant gate; fnuz on gfx942 (wrong dialect = 2× off). bf16 path is parity-safe. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
asm kernels compiled JIT/AOT into `aiter/jit/`; dispatched by `tuned_fmoe.csv`. To edit, modify the asm
source under aiter and rebuild; autotune tile/prefetch/ksplit. bf16 asm via `fused_moe_bf16_asm.py`.

## Pitfalls & anti-patterns
- Decode under-fill: per-expert `m_e < block_m` wastes padding — consider masked grouped GEMM.
- `num_warps=8`-equivalent over-allocation → VGPR spill (3–5× slower).
- fnuz/OCP fp8 dialect mismatch (2× off).

## How to verify
`AITER_LOG_MORE=1` → `fmoe_stage1_*` fired; disassemble the hot loop (`v_mfma`, `buffer_load` width, no
`scratch_` spam); isolated stage-1 timing; bf16 parity / fp8 eval.

## Alternatives / cross-links
[aiter.md](aiter.md) (driver) · [ck.md](ck.md) (stage-2) · [triton.md](triton.md) ·
[`languages/hip_cpp/`](../../../languages/hip_cpp/overview.md) ·
[`languages/asm_mfma/`](../../../languages/asm_mfma/overview.md) · [overview.md](../overview.md).

## Sources
- on-box: `ROCm/aiter@a6bb49937:aiter/fused_moe_bf16_asm.py`, `aiter/configs/tuned_fmoe.csv` (stage-1 names).
- MFMA / fp8 convert on CDNA3: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html

## On gfx1151 (RDNA3.5, Strix Halo)
`gfx1151` appears in `gens:` because this backend's **source is portable** to RDNA — it
compiles/JITs there with no vendor asset table. It is **not** a claim that anything on this
card was measured on RDNA: every perf number, ranking and tuning recipe above is CDNA
(gfx90a/942/950) evidence. Three differences bite before any of it transfers — WMMA not MFMA,
wave32 not wave64, and 40 CU behind a 32 MB MALL on ~229 GB/s shared LPDDR5X rather than HBM —
so tile shapes, occupancy targets and the roofline ceiling all move. Any `fp8_*` entry in
`dtypes:` above is CDNA-only: gfx1151 has **no fp8 matrix instruction and no block-scaled
FP4/FP6**, so an fp8 candidate there runs EMULATED — it passes correctness and loses
performance silently. See [`../../../hardware/rdna35_gfx1151/`](../../../hardware/rdna35_gfx1151/)
and MEASURE on the box.
