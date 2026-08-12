---
key: how to gate a candidate on a dispatch-dominated op whose per-case timings are bimodal on gfx950
type: instrument
confidence: ★★
effect: The same genuine device-side win was refused 5 consecutive times by the single-sample gate (2.512x / 2.549x / 2.592x / 2.594x / 2.620x against a 2.613x incumbent) while median-of-12 per-case geomean 2.619x-2.717x and fast-window per-case minima 2.96x-2.99x cleared it every time. Per-case signature: the throttle moves batch=32 and batch=64 together (roughly a 25% spread) while batch=2 stays flat, so a case-blind average hides it.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-11
name: bimodal-throttle-defeats-single-shot-gate-method-gfx950-launch-bound
description: On sub-microsecond kernels a bimodal box throttle makes a single-shot gate reject a real win repeatedly; gate on medians or paired same-session A/B.
keywords: ['measurement-noise', 'throttle', 'ab-methodology', 'latency-bound', 'launch-overhead', 'frozen-baseline', 'small-batch']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: method
regime: launch-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-11
---
# bimodal-throttle-defeats-single-shot-gate
- lever: For an op whose per-case time is dominated by dispatch, judge a candidate on a median of >=10 back-to-back samples, or on an interleaved >=8-pair same-session A/B of per-case minima, and additionally look for the mechanism's own fingerprint in the per-case pattern.
- apply: Run candidate and seed alternately inside one process, keep per-case minima, and state in advance which per-case relation the mechanism predicts - here removing a serial per-block dependent-load chain predicted the batch64-vs-batch32 gap collapsing, and it collapsed about 6x while the batch-64 minimum improved 12.7%.
- verify: A single slow sample below the incumbent is a sampling outcome, not a regression: re-sample before discarding a candidate, and re-check that the candidate is still bit-exact against the golden tensor each time it is re-applied.
- pitfall: An earlier round concluded the serial reduction was off the critical path -> that verdict came from a work-adding vectorized variant that measured 0.973x, which disconfirms the variant and not the chain -> eliminating the dependent-load chain at O(1) precompute cost is what pays; bandwidth was never the constraint because the operand is cache-resident.
- caution: Also verify the noise shape before choosing the gate: this signature (two cases moving together, one flat) is what identified a whole-process throttle rather than kernel variance, and a different box may need a different sampling rule.
- source: 16h per-kernel time-budget campaign, 62 resumed passes, gfx950, 2026-08-11
