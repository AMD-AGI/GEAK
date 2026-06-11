---
title: layout_shuffle on Triton — SOTA card
kind: sota_card
operator: layout_shuffle
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/shuffle.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# layout_shuffle × Triton

## TL;DR
The shuffle itself is a one-time `view`+`permute`+`contiguous` — **plain torch (or a trivial Triton kernel)
is enough**; there's no Triton-specific advantage for the *shuffle*. Triton's relevance is the **consuming**
GEMM: aiter ships Triton bpreshuffle-preprocess utilities (`ut_afp4wfp4_gemm_preshuffle`) and Triton FP4/FP8
GEMMs that expect a pre-shuffled weight. Use Triton here when authoring a fused Triton GEMM that ingests a
pre-shuffled weight; otherwise the offline shuffle is just `torch.permute`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| offline `torch.permute` / trivial Triton permute | aiter `shuffle.py` (the layout) | both, all dtypes | one-time, negligible — measure the consuming GEMM | the shuffle step |
| Triton GEMM consuming a pre-shuffled weight | aiter `aiter/ops/triton/*` (preshuffle UTs) | gfx950 fp4/fp8 | unlocks the Triton bpreshuffle GEMM | fused Triton FP4/FP8 GEMM |

## Config space / knobs
- The **layout** must match the consuming Triton GEMM's `tl.dot` operand expectation (mirror aiter's
  `shuffle_weight` layout for that kernel).
- AMD knobs (`matrix_instr_nonkdim=16`, `kpack`) live in the consuming GEMM, not the shuffle.

## Numerics / parity
value-preserving (exact); FP4 weight+scale shuffled together; layout-vs-kernel mismatch caught by GEMM
`allclose`. fp8 dialect (fnuz gfx942) is the consumer's concern. See [[operators/layout_shuffle/numerics.md]].

## Integration (rebind seam)
Shuffle at load (torch); the consuming Triton GEMM reads the pre-shuffled weight. Not wired through aiter's
`bpreshuffle` 9-tuple unless routed via aiter dispatch (the aiter Triton GEMM stub is a thin shim —
[[languages/triton_amd/pitfalls.md]]).

## Pitfalls & anti-patterns
- ⚠ Writing a Triton kernel just to permute a weight — `torch.permute` is simpler and one-time.
- ⚠ Mismatched layout vs the consuming `tl.dot` → garbage.
- ⚠ The "Triton GEMM in aiter" is a stub, not a tuned bpreshuffle impl — author needed.

## How to verify
GEMM output `allclose` vs unshuffled; `TRITON_PRINT_AUTOTUNING=1` on the consuming GEMM; rocprofv3 → no
in-kernel reshuffle, conflict-free operand load.

## Alternatives / cross-links
[backends/aiter.md](aiter.md) (the production path) · [backends/hip.md](hip.md) ·
[[operators/dense_gemm/backends/triton.md]] · [[languages/triton_amd/patterns.md]] ·
[[languages/triton_amd/pitfalls.md]].

## Sources
- aiter shuffle layout + Triton preshuffle UTs: ROCm/aiter@a6bb49937:aiter/ops/shuffle.py, aiter/ops/triton/utils/_triton/tunning/ut_afp4wfp4_gemm_preshuffle.py.
- Triton AMD tuning: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
