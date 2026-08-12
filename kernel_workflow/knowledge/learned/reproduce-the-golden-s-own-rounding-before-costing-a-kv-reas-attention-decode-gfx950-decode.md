---
key: bf16 KV attention decode whose golden casts the probabilities to bf16 inside the PV dot, scored by a worst-element relative-error gate on gfx950
type: anti-pattern
confidence: ★★
effect: closed axis, banked ~1.00x: split-KV per-case diverged at max_rel 1.357 against a 1e-2 gate on all three cases and invariant to the split count; fp8 KV blew worst-element error 36-125x on ~16% near-zero rows
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode
description: Attention decode under a worst-element max_rel gate: split-KV and fp8 KV close on numerics, not speed; split count and scale granularity do not help.
keywords: ['attention', 'decode', 'split-kv', 'flash-decode', 'fp8-kv', 'numerics', 'oracle-parity', 'anti-pattern']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
verified_on: 2026-07-29
---
# Reproduce the golden's own rounding before costing a KV reassociation
- lever: Before spending a round on split-KV/flash-decode or a lower-precision KV cache, recompute the golden against a true-fp32 reference and see how much error budget is already spent.
- apply: If the golden itself sits above the gate on worst-element relative error, any reassociation of the KV reduction inherits that gap; the cast that lives inside the dot cannot be reproduced outside it.
- verify: Score the near-zero output rows separately: cosine similarity stayed above 0.999 while worst-element relative error was orders of magnitude over the gate, so an aggregate metric hides the failure.
- pitfall: fp8 e4m3 KV looked fine on cosine but failed the gate -> weighted-V cancellation leaves a sixth of the rows near zero, where a 3-bit mantissa dominates -> scale granularity changed nothing, because the operands are uniform-magnitude and it is the mantissa that binds.
- caution: Also verify what the parity gate actually scores: this axis reopens if the golden's in-dot cast changes or near-zero rows are scored on an absolute tolerance.
- source: run kernel_unified_attention_2d-ch16h, 2026-07-29, directions split_kv_deep_16h and r1_d0_fp8kv_16h
