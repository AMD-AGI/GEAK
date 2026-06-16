---
title: Mojo on AMD Instinct — enablement status & measured perf
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3]
regimes: [both]
status: experimental
updated: 2026-06-08
sources:
  - https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days
  - https://arxiv.org/html/2511.08083v1
---

# Mojo on AMD — status

## TL;DR
Mojo/MAX **runs on MI300, MI325, MI355** (CDNA3/CDNA4). Modular reports a **day-1 matmul kernel 3% faster
than hipBLASLt** on MI355X and **MAX up to 2.2× vs AMD's optimized vLLM fork** — both vendor-reported (Oct
2025). The honest caveat: **attention is not yet at peak** — the HipKittens paper independently measured
Mojo MHA at **~50% of peak (≈430 TFLOPs at B16/H16/N2048/D128)** from LDS bank conflicts. So as of mid-2026
Mojo is "GEMM-strong, attention-developing" on AMD, and still an *experimental* authoring option vs the
production aiter/CK/hipBLASLt path.

## Enablement timeline (vendor-reported)
| date | event |
|---|---|
| 2025-08 (late) | AMD/TensorWave reach out to Modular |
| 2025-09-01 | MI355X hardware provisioned |
| 2025-09-16 | Demo at AMD Media Tech Day |
| 2025-10-17 | Blog published with results |
| scope | ~20 small PRs, ~1.5 engineers, 2 weeks; "only matmul-category kernels needed changes" for new MI355 HW |

## Measured matmul (vendor-reported, MI355X, M=N=K=8192, GFlop/s)
| kernel | GFlop/s |
|---|---|
| Mojo (day 0) | 1,202,303 |
| hipBLASLt (SOTA baseline) | 1,561,447 |
| **Mojo (day 1)** | **1,610,514** (≈ +3% vs hipBLASLt) |

Kernel was ~500 lines of "well-commented" Mojo. (Cross-ref: HipKittens reports its own HK GEMM at 1610
TFLOPS and hipBLASLt at 1561 at the same M=N=K=8192 — consistent ordering, different units/runs; treat both
as vendor-labeled.)

## Attention status (the honest gap)
- Modular's blog notes they "began exploring the Attention kernel" in week 2 but **published no attention
  TFLOPs**.
- Independent measurement (**HipKittens**, arXiv 2511.08083v1, Nov 2025): Mojo MHA forward "suffers from
  bank conflicts," reaching **~50% of peak**, e.g. **430 TFLOPs at B=16, H=16, N=2048, D=128** on MI355X.
  Root cause is the same AMD LDS swizzle / non-compositional-layout problem HK solves with a swizzle solver
  (see [../hipkittens/primitives.md](../hipkittens/primitives.md)).

## Why Mojo ports fast to new AMD HW
- No hardcoded GPU constants; hardware specifics live in **parameterized library kernels** retuned per
  arch. ~99.9% of the stack is architecture-agnostic, so new-GPU support is "update a few kernels."
- This is the same design philosophy as the tile-abstraction-portable / backend-specific split HipKittens
  argues for — Mojo bets on the *compiler+library* doing the AMD-specific lowering; HK bets on
  hand-written AMD backends. As of Nov 2025 HK's hand-tuned attention beat Mojo's compiler-driven one.

## Decision guidance (perf_knowledge)
- **Production AMD serving today:** aiter / hipBLASLt / CK / asm — not Mojo. Mojo adds a non-ROCm
  toolchain dependency and is single-vendor.
- **Portable cross-vendor authoring with strong GEMM:** Mojo is a credible *experimental* choice; validate
  per-kernel.
- **Attention on AMD:** prefer CK/Triton FA or AITER FA until Mojo closes the bank-conflict gap; re-measure
  if you must use Mojo.

## Sources
- Modular blog (timeline, GFlop/s table, 2.2× vs vLLM, library-directed design):
  https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days
- Mojo MHA ~50% peak / 430 TFLOPs bank-conflict finding: HipKittens paper
  https://arxiv.org/html/2511.08083v1
- overview: [overview.md](overview.md)
