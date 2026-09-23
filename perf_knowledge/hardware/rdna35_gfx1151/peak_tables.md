---
title: gfx1151 peak tables — datasheet vs empirical, and the scoring bracket
kind: hardware
gens: [gfx1151]
dtypes: [fp32, bf16, fp16]
regimes: [both]
updated: 2026-09-23
sources:
  - GEAK/kernel_workflow/knowledge/repro/rdna_roofline.py (3 sweeps)
  - GEAK/kernel_workflow/knowledge/repro/wmma_check.py (median of 40 interleaved rounds)
  - GEAK/kernel_workflow/knowledge/amd_rdna.md §3b, §4, §5
---

# gfx1151 peak tables

> Bandwidth half of the roofline in [memory.md](memory.md); the matrix path in
> [matrix_core_wmma.md](matrix_core_wmma.md).

## TL;DR
> **Report a bracket, never a single efficiency number.** Datasheet fp16/bf16 WMMA peak is
> **59.4 TFLOP/s**; the best any kernel measured here reached is **38.35–38.52** (median 38.46),
> i.e. **64.7%**. The datasheet end may be unreachable on this part; the empirical end is a
> *floor* on the true peak, so scoring against it flatters the kernel. Quote both.

## Compute peaks

| | fp16 / bf16 WMMA |
|---|---|
| datasheet peak | **59.4 TFLOP/s** `[vendor]` (40 CU × 64 lanes × 2 × 2.9 GHz × 4) |
| empirical peak | **38.35 – 38.52 TFLOP/s**, median 38.46 `[measured]` — best any kernel reached, over 3 sweeps |
| winning kernel | `torch.mm` @ 3072³, stable across all 3 sweeps, spread 0.4% |
| empirical / datasheet | **64.7%** |

The winner's matrix path is **established, not assumed**: `rocprofv3` names it
`Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_..._ISA1151_...` — a gfx1151 Tensile kernel on a
16×16×16 matrix instruction. A peak whose instruction cannot be identified this way is reported as
**ISA-UNVERIFIED** rather than trusted.

## The bracket, worked
Per-kernel efficiency at 2048³ fp16 (working set 25.2 MB, LLC-resident, AI 683 FLOP/byte):

| kernel | achieved | true efficiency is **bracketed** by |
|---|---|---|
| naive Triton `tl.dot` | 24.36 TFLOP/s | 41.0% (vs datasheet) .. 63.3% (vs empirical) |
| `torch.mm` (Tensile) | 28.38 TFLOP/s | 47.8% (vs datasheet) .. 73.8% (vs empirical) |

**This shape is compute-bound by 9×** — the memory roof is 539 TFLOP/s against a 59.4 TFLOP/s
compute roof. So any argument about which library wins here is an argument about *compute*
efficiency, and neither contender is above half the datasheet peak. **That** is where the headroom
is, not in cache residency.

**An efficiency above 100% of the empirical peak is not a fast kernel — it is proof the denominator
is wrong.** Treat it as a failed measurement and re-derive the peak.

## Vendor-library coverage: uneven, and not predictable from the architecture
`[measured]` Shape-dependent **in both directions**. Do not carry a blanket "the vendor library is
weak on RDNA" prior — and check *which* vendor library you reached, because on this box `torch`
defaults to **rocBLAS** and never touches hipBLASLt.

| shape / regime | naive Triton vs the vendor path | after tuning |
|---|---|---|
| large square, 2048³ fp16 | `torch.mm` **1.16× faster** (27.52 vs 23.59 TFLOP/s) | — |
| small square, 128/256/512³ | vendor path **1.6–2.1× faster** than naive Triton | a tuned kernel beats it **1.51× / 1.83×** at 128³ / 256³ but still **loses 13%** (0.87×) at 512³ |
| skinny / decode, M≤32, N=K=4096 | Triton **~1.8× faster** than `torch.mm` | — |
| short-K (`shortk_512`) | Triton **0.32×** — the vendor path wins outright | — |

> These rows say "vendor path", not a library name, on purpose: only the 2048³ row was re-measured
> with the backend checked. `torch.backends.cuda.preferred_blas_library()` reports
> `_BlasBackend.Cublas`, which on ROCm is **rocBLAS** — hipBLASLt would report `Cublaslt`. **A
> `torch.mm` result on this part says nothing about hipBLASLt**, even though ROCm 7.2 ships
> gfx1151-tuned hipBLASLt kernels.

> **Retraction, kept deliberately.** An earlier version of this table claimed naive Triton was
> **1.98× faster** at 2048³ (24.99 vs 12.60 TFLOP/s). That was a benchmark artifact: the vendor side
> allocated a fresh output every iteration and got no warm-up, while Triton wrote into a
> pre-allocated buffer after five warm-ups, and the two ran in sequence rather than interleaved.
> Pre-allocated, warmed, interleaved, GPU-event-timed, median of 40 rounds → the vendor path is
> **faster**. The withdrawn claim was wrong twice: unfair harness *and* wrong opponent.

Read that as: **the vendor path wins in three of the four regimes here.** The one regime where a
generated kernel clearly wins is **skinny/decode-shaped** GEMM — which is the shape that matters
most for token generation, so it is not a small exception. But note the evidence asymmetry: that
row has **no repro script**, while the row that did get a careful re-measurement is the one that
flipped against Triton. **Re-take the skinny number before building a plan on it.**

Corroboration from the E2E side: on this part, per-shape BLAS routing that mixes rocBLAS and
hipBLASLt gained **+14.3%** while forcing hipBLASLt globally *lost* **8.2%** — a 22-point spread.
Coverage is genuinely per-shape.

## Pitfalls
- **Quoting one column.** Datasheet-only under-reports the kernel; empirical-only flatters it.
- **Trusting a peak whose instruction was not disassembled.** A CDNA peak table applied here is
  silently wrong.
- **Treating the empirical peak as a hard ceiling.** It is a max-of-sweep — a winner's-curse
  estimator, biased high and unstable. Repeated sweeps move it ~1.4% and move the *winning shape*
  between 1024/3072/4096.
- **Assuming a `torch.mm` number characterises "the vendor library".**

## Verify
- `repro/rdna_roofline.py` — disassembles first and **refuses to score** anything that shows
  `v_mfma` or no `v_wmma`.
- `repro/wmma_check.py` — fair-harness GEMM comparison (both sides pre-allocated, warmed,
  interleaved, GPU events, median of 40).
- Note `perf_knowledge/profiling/kernel_roofline.md` is CDNA-scoped and **refuses RDNA outright**
  because it drives `rocprof-compute --roof-only`, whose roofline mode does not support
  gfx10/11/12. The terms here are measured a different way on purpose.

## Sources
- `GEAK/kernel_workflow/knowledge/repro/rdna_roofline.py`, `wmma_check.py` — on-box, ROCm 7.2.3,
  torch 2.11.0, Triton 3.6.0.
- `GEAK/kernel_workflow/knowledge/amd_rdna.md` §3b, §4, §5 (including the retraction).
