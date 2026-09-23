---
title: dense_gemm on rocwmma — SOTA card
kind: sota_card
operator: dense_gemm
backend: rocwmma
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: legacy
updated: 2026-06-08
sources:
  - https://github.com/ROCm/rocWMMA
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# dense_gemm × rocwmma

## TL;DR
> rocWMMA is a header-only C++ WMMA-style wrapper over MFMA — a **portability/teaching** API (CUDA-WMMA-
> like fragments) rather than a SOTA GEMM path. For production dense GEMM it does **not** beat tuned
> hipBLASLt/CK/asm and lacks an autotuner; use it only to port a `wmma`-based CUDA kernel quickly or as
> a readable on-ramp to MFMA. For the live serving path use [aiter.md](aiter.md)/[hipblaslt.md](hipblaslt.md).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| rocWMMA `fragment` GEMM (`load/mma/store_matrix_sync`) | `ROCm/rocWMMA` | gfx942/950; bf16/fp16/fp8 | no published figure competitive with tuned hipBLASLt → **use [hipblaslt.md](hipblaslt.md)/[asm.md](asm.md)** | CUDA-WMMA port / portability, not peak |

## Config space / knobs
- Fragment shape (`16×16×16` recommended over `32×32×8`), `rocwmma::fragment<matrix_a/b/accumulator>`.
- Manual LDS staging + block tiling around the fragment ops (rocWMMA gives the MMA primitive, not the
  full tiled/pipelined kernel — you write the loop, tile multiples of 8, ≥1024 WGs).
- No built-in autotuner; you hand-tune tiles/pipeline like a raw HIP kernel.

## Numerics / parity
fp32 accumulate via the accumulator fragment; parity-safe for bf16. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Build-time C++ extension; call directly from your op. No env-overlay into the live GEMM path.

## Pitfalls & anti-patterns
- Treating rocWMMA as a fast-path — it's an abstraction layer; raw `__builtin_amdgcn_mfma_*` (asm) or
  CK templates give more control and higher perf.
- Default fragment/tile choices left untuned → well below library kernels.

## How to verify
Microbench vs `hipblaslt-bench` on the same shape; expect it to lose — only keep for portability.

## Alternatives / cross-links
[asm.md](asm.md) (raw MFMA, peak) · [ck.md](ck.md) · [hip.md](../../batched_gemm/backends/hip.md) (raw HIP) ·
[aiter.md](aiter.md) · [../overview.md](../overview.md) · language ref [[languages/rocwmma/...]].

## Sources
- rocWMMA project: https://github.com/ROCm/rocWMMA.
- MFMA fragment shape guidance (16×16 > 32×32): ROCm matrix-cores-cdna blog.

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
