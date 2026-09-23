---
title: grouped_gemm_moe on hip — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://github.com/ROCm/aiter
---

# grouped_gemm_moe × hip

## TL;DR
> A hand-written HIP/C++ grouped-GEMM kernel (MFMA intrinsics + an `expert_offsets` table) is the route
> when you need a fusion or layout the library/Triton paths don't expose, or as the reference for an asm
> rewrite. Most production serving should use [aiter.md](aiter.md); HIP is the authoring substrate.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| HIP MFMA grouped GEMM (custom epilogue/layout) | author against `__builtin_amdgcn_mfma_*` per Matrix Core blog | gfx942/950; bf16, fp8 | no published number — use for custom fusions / as asm reference | bespoke fusion not in aiter/ck |

## Config space / knobs
- MFMA instr choice (`mfma_16x16` vs `32x32`), LDS tiling/double-buffer, register-tile per wave,
  per-expert offset indexing, CU tile balancing (304 MI300X / 256 MI350X).
- Fuse act-and-mul + quant epilogue inline → [../fusion.md](../fusion.md).

## Numerics / parity
- fp32 accumulate; honor per-expert scale order; mask padded rows → [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Expose via a custom op / `hipModule`; wire into the MoE layer call site. Verify by kernel name in trace.

## Pitfalls & anti-patterns
- Reimplementing align&sort/offset logic incorrectly corrupts token routing; reuse the framework's
  `expert_ids`/`num_tokens_post_pad`.
- Hand-HIP rarely beats tuned aiter asm on covered shapes — justify with a measured win.

## How to verify
- Per-expert dense oracle ([../numerics.md](../numerics.md)) + A/B vs aiter.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [ck.md](ck.md) · [tilelang.md](tilelang.md) · [../overview.md](../overview.md)

## Sources
- Matrix Core programming (MFMA intrinsics): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AITER (production grouped path): https://github.com/ROCm/aiter

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
