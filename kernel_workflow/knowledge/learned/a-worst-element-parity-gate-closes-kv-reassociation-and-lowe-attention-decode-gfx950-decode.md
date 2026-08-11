---
key: attention decode whose golden casts P to bf16 inside the V dot, graded by worst-element relative error on gfx950
type: anti-pattern
confidence: ★★
effect: Disconfirming: KV reassociation lands max_rel 1.357 against a 1e-2 gate, invariant to the number of splits, 999/8192 elements failing; lower-precision KV blows worst-case relative error 36-125x on the ~16% near-zero output rows while cosine stays 0.9992+. Zero speedup shipped from either axis across cases c2/c32/c64.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: a-worst-element-parity-gate-closes-kv-reassociation-and-lowe-attention-decode-gfx950-decode
description: Split-KV/flash-decode and fp8 KV both die on a worst-element max_rel gate when the golden bakes a bf16 cast inside the V dot; cosine stays ~1.0 and hides it
keywords: ['attention', 'decode', 'split-kv', 'flash-decode', 'fp8-kv', 'numerics', 'oracle-parity', 'dead-end']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
---
# A worst-element parity gate closes KV reassociation and lower-precision KV
- lever: Before budgeting rounds on split-KV/flash-decode or a lower-precision KV cache, run a 10-line numpy/torch parity probe of the proposed math against the golden and read the WORST-element relative error, not cosine.
- apply: Read the golden's dot chain first: a cast of the probability tile to the KV dtype placed INSIDE the accumulation makes the reference itself non-associative, so any re-split of the KV axis reproduces a different number by construction.
- verify: Compare the probe's max_rel and the fraction of failing elements against the gate before writing a kernel; also compute the near-zero-output row fraction, since those rows set the worst-case ratio.
- pitfall: Cosine similarity read 0.9992+ and looked like a pass -> weighted-V cancellation makes ~16% of output rows near-zero, where a 3-bit mantissa turns a tiny absolute error into a huge relative one -> switched the acceptance read to worst-element max_rel and the axis closed in one probe instead of several rounds.
- caution: Also verify what the gate actually grades and what the golden's internal cast dtype is before concluding the axis is shut — it reopens if the reference stops casting inside the dot, if the near-zero relative gate is loosened, or if a KV format holds max_rel under the gate on near-zero rows.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11; ledger directions split_kv_deep_16h and r1_d0_fp8kv_16h, both dead_end
