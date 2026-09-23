---
title: batched_gemm on hip — SOTA card
kind: sota_card
operator: batched_gemm
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: legacy
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/hipBLASLt
---

# batched_gemm × hip

## TL;DR
> A raw HIP/MFMA batched GEMM (hand-written kernel or a loop over `hipblasGemmStridedBatched`) is a
> **baseline / learning** path — for production it does not beat tuned hipBLASLt/CK/asm and lacks an
> autotuner. Use it only to understand the strided-batched mapping or to wire a quick custom kernel; for
> the live path use [aiter.md](aiter.md)/[hipblaslt.md](hipblaslt.md).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Raw HIP MFMA batched kernel (`__builtin_amdgcn_mfma_*`, batch on blockIdx.z) | hand-authored | gfx942/950; bf16/fp16 | no figure competitive with tuned hipBLASLt → **use [hipblaslt.md](hipblaslt.md)** | learning / custom one-off |
| `hipblasGemmStridedBatched` (library call) | `ROCm/hipBLASLt`/hipBLAS | gfx942/950 | competitive only as a thin wrapper over hipBLASLt solutions | non-aiter stacks |

## Config space / knobs
- Map batch to `blockIdx.z`; per-block tile multiples of 8; 16×16 MFMA; LDS double-buffer; ≥1024 total WGs.
- No autotuner — you hand-pick tile/pipeline like an asm kernel but with less control.
- `hipblasGemmStridedBatched` exposes `algo`/`computeType` only; real tuning lives in hipBLASLt solutions.

## Numerics / parity
fp32 accumulate per batch; bf16 parity-safe. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Build-time; call directly. No env-overlay into serving GEMM.

## Pitfalls & anti-patterns
- A host-side loop of B GEMM launches → launch overhead + CU starvation (the classic anti-pattern).
- Naive global-memory inner loop (no LDS staging / no MFMA) → orders of magnitude off peak.
- Hand-tuning when hipBLASLt already has a faster solution.

## How to verify
Microbench vs `hipblaslt-bench --batch_count`; expect to lose — keep only for learning/custom needs.

## Alternatives / cross-links
[hipblaslt.md](hipblaslt.md) · [asm.md](asm.md) (peak) · [ck.md](ck.md) · [aiter.md](aiter.md) ·
[../overview.md](../overview.md) · language ref [[languages/hip_cpp/...]].

## Sources
- Raw MFMA batched mapping: ROCm matrix-cores-cdna blog.
- Strided-batched API: https://github.com/ROCm/hipBLASLt.

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
