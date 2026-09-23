---
title: splitk_streamk_gemm on hip — SOTA card
kind: sota_card
operator: splitk_streamk_gemm
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://arxiv.org/abs/2301.03598
---

# splitk_streamk_gemm × hip

## TL;DR
> A HIP/C++ MFMA kernel with an explicit persistent grid (≈ CU count) and a fix-up reduction is the
> reference implementation of stream-K on AMD — use it to author/validate a stream-K scheme before an asm
> port, or for a fusion the libraries don't expose. Production should prefer triton/aiter.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| HIP MFMA stream-K (persistent grid + fix-up) | author per Matrix Core blog + Stream-K paper | gfx942/950; bf16, fp8 | no published number — reference/authoring | custom stream-K fusion, asm reference |

## Config space / knobs
- Persistent workgroup count (≈ 304 MI300X / 256 MI350X), tile sizes, MFMA instr (16x16/32x32), fix-up
  reduction (atomic vs workspace), LDS double-buffer.

## Numerics / parity
- fp32 accumulate; fix-up must combine each output tile exactly once → [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Custom op / hipModule wired at the GEMM call site; verify by kernel name in trace.

## Pitfalls & anti-patterns
- Stream-K fix-up bugs (double-count / dropped k-iters) are correctness failures, not just precision.
- Wrong persistent grid (≠ CU count) reintroduces wave quantization.

## How to verify
- Dense fp32 oracle ([../numerics.md](../numerics.md)) + A/B vs triton stream-K.

## Alternatives / cross-links
[triton.md](triton.md) · [ck.md](ck.md) · [hipblaslt.md](hipblaslt.md) · [asm.md](asm.md) · [../overview.md](../overview.md)

## Sources
- Matrix Core programming (MFMA): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Stream-K: https://arxiv.org/abs/2301.03598

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
