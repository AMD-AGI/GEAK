---
title: linear_attention_gated_delta — numerics
kind: technique
operator: linear_attention_gated_delta
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
updated: 2026-06-08
sources:
  - https://github.com/fla-org/flash-linear-attention
  - https://arxiv.org/abs/2412.06464
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0
---

# linear_attention_gated_delta — numerics

## Error accumulates along the recurrence
Unlike softmax attention (each output independent), GDN carries a **state S across the whole sequence**, so
numerical error **accumulates over T**. Three consequences:
1. **fp32 state accumulate is mandatory.** Inputs bf16/fp16, but S and the delta update must accumulate in
   fp32. A bf16 state drifts visibly on long sequences.
2. **Chunk-boundary consistency.** The chunked-prefill path and the fused-recurrent decode path must
   produce the **same** state at a chunk boundary, or prefill→decode handoff diverges. Gate by feeding
   prefill's `final_state` into decode and comparing to a pure recurrent reference.
3. **Gate/decay precision.** `α_t` exponential decay and `β_t` write strength are applied per channel; the
   order (L2-norm Q/K → conv1d → gate) must match the reference exactly. A swapped order changes results
   even at fp32.

## The L2-norm + conv1d pre-step
GDN L2-normalizes Q/K and applies a causal conv1d before the scan. aiter fuses L2-norm into the kernel
(`use_qk_l2norm_in_kernel=True` in `fused_recurrent_gated_delta_rule`). If you split the pre-step out,
match its dtype/eps — a different eps in L2-norm shifts the whole scan.

## Parity gate
- Oracle: the FLA reference recurrent scan in fp32 (the `fla-org/flash-linear-attention` PyTorch path), at
  the model's exact d_k/d_v/chunk and gate definitions.
- Two-stage gate: (a) **state parity** — prefill `final_state` vs reference at several T; (b) **output
  parity** — greedy temp=0 generation, ≥10 prompts, several **long** (so the recurrence accumulates).
- Cross-backend bf16 token flips on long greedy decode are benign (equivalence-class) — distinguish from a
  real state-drift regression by comparing the fp32 state, not just tokens.
- aiter's `fused_recurrent_gated_delta_rule` is **forward-only / no grad** — for training use the FLA
  library (the aiter port is an inference kernel).

## AMD-specific
- bf16 MMA type-confusion has bitten linear/recurrent kernels on CDNA3 (cf. the spec-decode rowsum
  bf16/fp16 MFMA mix-up) — verify the MFMA intrinsic matches the dtype, or accumulation silently corrupts.
- No fp8 GDN state path is standard; if you quantize the state, gate hard (fnuz on gfx942).

## Sources
- GDN definition (gated delta rule): https://arxiv.org/abs/2412.06464
- FLA reference scan (parity oracle): https://github.com/fla-org/flash-linear-attention
- aiter forward-only kernel + `use_qk_l2norm_in_kernel`: `ROCm/aiter@a6bb49937:aiter/ops/triton/gated_delta_net/gated_delta_rule.py` (on-box).
