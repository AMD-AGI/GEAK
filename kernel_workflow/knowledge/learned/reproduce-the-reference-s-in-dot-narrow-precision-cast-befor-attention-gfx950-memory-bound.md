---
name: reproduce-the-reference-s-in-dot-narrow-precision-cast-befor-attention-gfx950-memory-bound
description: Read where the reference rounds before funding reassociation: split-KV diverged at max_rel 1.357 and fp8 KV blew worst-element error 36-125x past the gate
keywords: [numerics-gate, reassociation, fp8, dtype-dialect, measurement-method, attention, memory-bound, operand-reuse]
kernels: [_fwd_grouped_kernel_stage1]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: split-KV / flash-decode reassociation and fp8 KV operands on a Triton attention kernel on gfx950 whose golden reference casts P to bf16 inside the V dot, graded by a worst-element max_rel gate
lifecycle: active
type: anti-pattern
confidence: ★★
effect: two structural axes died on numerics, not on performance - the golden bakes a bf16 cast of P INSIDE the V dot (acc += tl.dot(P.to(V.dtype), V)), so the reference is itself max_rel 1.089 against true fp32, and ANY split-KV / flash-decode reassociation then diverged at max_rel 1.357 with 999/8192 elements failing on the graded case, invariant to num_splits; fp8 KV against a max_rel<1e-2 gate blew worst-element max_rel to 36-125x while cosine stayed 0.9992+, driven by ~16% near-zero output rows (weighted-V cancellation) and a 3-bit mantissa, and it held for K-only, V-only and both, with scale granularity irrelevant; with both axes closed the campaign declared its 2.0x target unreachable in gate-safe scope and stopped at ~1.70x
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.63h / 51 passes, 2026-08-11
last_seen: 2026-08-11
---
# Reproduce the reference's in-dot narrow-precision cast before funding reassociation or a narrower operand
- lever: Before spending a deep-explore round on split-K/split-KV, flash-decode style reassociation, or a lower-precision operand, read the reference implementation for where it rounds and measure the reference against a true-fp32 recomputation - if the reference rounds an intermediate inside the accumulation, its exact op order is part of the contract and no reassociated variant can match it. That is a one-hour measurement standing in for a multi-round direction.
- apply: Reproduce the reference's own cast at the same point in the candidate before scoring it, and score any worst-element (max_rel) gate on the near-zero output rows specifically.
- pitfall: fp8 KV looked safe on cosine (0.9992+) -> the output has ~16% near-zero rows from weighted-V cancellation, where a 3-bit mantissa moves worst-element error by two orders of magnitude -> score the cancellation rows directly; scale granularity moves exponent range, not the mantissa that is actually binding.
- verify: Measure the reference against true fp32 first, then the candidate against the reference on the same worst-element metric the gate uses, at several num_splits to show the divergence is invariant.
- caution: Also verify whether the gate or the reference has changed before treating the axis as permanently closed - the cheap tell for reopening is the gate loosening or the reference's internal cast changing.
- source: chuschen 16h time-budget campaign run, 15.63h / 51 passes, 2026-08-11
