---
title: skinny_gemv_decode on hip — SOTA card
kind: sota_card
operator: skinny_gemv_decode
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
---

# skinny_gemv_decode × hip

## TL;DR
> A HIP/C++ split-K GEMV (the wvSplitK reference lives here, in ROCm/vLLM custom kernels) is the authorable
> substrate for decode GEMM: thread-per-output-row VALU or mfma_16x16, coalesced W reads, fp32 reduction.
> Use for custom fusion/reference; production decode should use aiter.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| HIP split-K GEMV (wvSplitK-style custom kernel) | ROCm/vLLM custom GEMV kernels (per vLLM ROCm optimization docs) | gfx942/950; bf16, fp16, fp8 | no first-party number reproduced | custom decode fusion, asm reference |

## Config space / knobs
- Split-K factor (≈ CU count), VALU dot vs mfma_16x16, coalesced W tiling, LDS/atomic fp32 reduction,
  occupancy (warps/waves_per_eu).

## Numerics / parity
- fp32 accumulate + reduction; fp8 scale after dot → [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Custom op / hipModule at the decode-GEMM call site; verify kernel name in trace.

## Pitfalls & anti-patterns
- Hand-HIP rarely beats aiter's tuned asm on covered shapes — justify with measured GB/s win.
- Uncoalesced W reads or too-low split-K → leaves bandwidth/CUs on the table.

## How to verify
- HBM GB/s vs peak + A/B vs aiter + dense fp32 oracle ([../numerics.md](../numerics.md)).

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [asm.md](asm.md) · [../overview.md](../overview.md)

## Sources
- Matrix Core programming (MFMA/VALU): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- vLLM ROCm decode GEMV (wvSplitK) kernels: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html

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
