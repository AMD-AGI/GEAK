---
title: alibi — numerics & parity
kind: technique
operator: alibi
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2108.12409
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu
---

# alibi — numerics & parity

## 1. Bias added in fp32 before softmax
`S_{ij} = Q_i·K_jᵀ/√d + bias_{ij}`, with `bias` computed and added in **fp32** before the max-subtraction
and exp. Adding it in bf16 loses precision in the score and shifts the softmax. The attention accumulator
is already fp32 (flash softmax) → add the bias there.

## 2. Slope sequence must match the model
The per-head slopes `m_h` are a specific geometric sequence (`2^(−8h/H)`-style, with a defined handling for
non-power-of-2 head counts — the ALiBi paper's interleaving). A wrong slope sequence (or wrong head order)
changes the positional bias → degraded long-context behavior. Match the reference implementation exactly.

## 3. Causal mask interplay
ALiBi is applied with the causal mask: `bias_{ij} = −m_h·(i−j)` for `j ≤ i`, masked `−inf` for `j > i`. The
order (add bias, then apply causal `−inf`, then softmax) matters — masked positions must end at `−inf`
regardless of bias.

## 4. Sign convention
`bias = −m_h·(i−j)` with `i ≥ j` (causal) → the bias is ≤ 0, penalizing distant keys. Getting the sign
wrong inverts the locality bias. Verify against the model.

## 5. Parity
ALiBi is deterministic; cross-backend parity is tight (the bias is exact). A divergence after a backend
swap = wrong slopes / sign / mask order, not rounding — but the underlying attention softmax reduction
order still differs, so greedy e2e re-gate as for any attention swap.

## Parity gate
1. isolated attention-with-bias vs fp64: tight band; confirm slope sequence + sign + mask.
2. e2e greedy parity (attention reduction order differs → re-gate).

## Sources
- ALiBi bias / slopes / causal: https://arxiv.org/abs/2108.12409.
- fp32 score accumulate + bias in FMHA: https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu.
