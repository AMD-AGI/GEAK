---
title: layout_shuffle — numerics
kind: technique
operator: layout_shuffle
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/shuffle.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
---

# layout_shuffle — numerics

## Value-preserving (the shuffle itself is exact)
`shuffle_weight` is a pure `view`+`permute`+`contiguous` — it **reorders elements, never changes values**.
The shuffled tensor is bit-identical to the original under the inverse permutation, for every dtype
(bf16/fp8/fp4/int8). The shuffle introduces **zero** numeric error.

## The real risk: layout mismatch (a correctness bug, not precision)
A shuffled weight is only valid for the kernel that expects **that exact** layout. Wrong `layout=(IN,IK)`,
wrong `pack_n` (FlyDSL), or wrong `NLane`/`gate_up` (FP4 MoE) → the GEMM reads operand fragments in the wrong
order → **garbage output**, caught by the GEMM `allclose` oracle (not a tolerance — a gross mismatch). The
asserts (`N % IN == 0`, `K % BK == 0`, `real_k ≥ 256`) guard shape, not layout-vs-kernel pairing — that you
verify with the GEMM output.

## FP4 / FP8 scale shuffle must travel with the weight
For FP4 MoE, `shuffle_weight_a16w4` (weight) and `shuffle_scale_a16w4` (block scale) are a **pair** — the
block scale must be shuffled into the matching layout or the dequant applies the wrong scale to the wrong
element → wrong values. Shuffle both, together.

## bpreshuffle key parity (engagement, not values)
Setting `is_shuffled` flips `bpreshuffle` in the dispatch key — this is a **dispatch** concern, not a numeric
one, but the gradlib tuner still gates every bpreshuffle candidate on `err_ratio < 0.05` vs a torch reference,
so a numerically-divergent bpreshuffle kernel is never written to the tuned CSV
([[backends/aiter/tuned_gemm.md]]).

## Verify
GEMM-with-shuffled-weight output `allclose` to GEMM-with-original-weight (value-preserving → tight tolerance;
quant variants within the quant tol). A gross mismatch = layout/scale-pairing bug.

## Sources
- view/permute/contiguous (value-preserving), weight+scale pairing: ROCm/aiter@a6bb49937:aiter/ops/shuffle.py.
- err_ratio<0.05 gate on bpreshuffle candidates: ROCm/aiter@a6bb49937:gradlib (see [[backends/aiter/tuned_gemm.md]]).
