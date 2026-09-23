---
title: gemm_epilogue_fused on hip — SOTA card
kind: sota_card
operator: gemm_epilogue_fused
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: legacy
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# gemm_epilogue_fused × hip

## TL;DR
> A raw HIP/MFMA kernel with a hand-written epilogue is a **baseline / learning** path — full control
> over the fused op but no autotuner, and the GEMM core won't beat tuned hipBLASLt/CK. Use it to
> understand how the epilogue rides the accumulator before the store, or for a one-off custom fusion
> where you don't want the CK/Triton toolchain. For production, prefer [ck.md](ck.md) (rich epilogue) or
> [aiter.md](aiter.md) (live path).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Raw HIP MFMA GEMM + inline epilogue (`__builtin_amdgcn_mfma_*`, fp32 act before store) | hand-authored | gfx942/950; bf16/fp16/fp8 | no figure competitive with tuned hipBLASLt/CK → **use [ck.md](ck.md)/[aiter.md](aiter.md)** | learning / one-off custom fusion |

## Config space / knobs
- Tile multiples of 8; 16×16 MFMA; LDS double-buffer; ≥1024 WGs; deep K-pipeline.
- Epilogue: apply `α`/bias/residual/act in fp32 on the accumulator registers, then down-cast/quant on
  store; route the store to avoid the 512B Tagram hotspot (the manual `OPTIMIZE_EPILOGUE` equivalent).
- No autotuner — hand-pick tile/pipeline.

## Numerics / parity
fp32 epilogue before down-cast → equal-or-better than unfused; fp8 output task-gated. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
Build-time extension; call directly. No env-overlay into serving GEMM.

## Pitfalls & anti-patterns
- Re-implementing what CK's `CDEElementwise` already gives — large effort, lower perf.
- Naive epilogue writing extra HBM passes (defeats the point of fusing).
- Down-cast before act; store hitting the Tagram hotspot.

## How to verify
Microbench the fused kernel vs (hipBLASLt GEMM + separate elementwise) on the same shape; expect the
core to trail — keep only for learning/custom needs; gate quant on eval.

## Alternatives / cross-links
[ck.md](ck.md) (prefer) · [triton.md](triton.md) · [aiter.md](aiter.md) · [hipblaslt.md](hipblaslt.md) ·
[../overview.md](../overview.md) · language ref [[languages/hip_cpp/...]].

## Sources
- Raw MFMA + epilogue mapping: ROCm matrix-cores-cdna blog.
- Tile/Tagram/OPTIMIZE_EPILOGUE levers: ROCm workload guide.

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
