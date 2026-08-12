---
key: how to gate a candidate on a dispatch-dominated op whose per-case timings are bimodal on gfx950
type: instrument
confidence: ★★
effect: The same genuine device-side win was refused 5 consecutive times by the single-sample gate (2.512x / 2.549x / 2.592x / 2.594x / 2.620x against a 2.613x incumbent) while median-of-12 per-case geomean 2.619x-2.717x and fast-window per-case minima 2.96x-2.99x cleared it every time. Per-case signature: the throttle moves the two larger batch cases together (roughly a 25% spread) while the smallest stays flat, so a case-blind average hides it. Second run, same op and box: interleaving alone is not enough - a FIXED-ORDER interleaved paired A/B (base,V1,V2 x4 reps) reported +3.4/+3.7/+6.4% with 12/12 paired reps favouring the candidate, and the same variants in balanced ABBA order collapsed to -0.08/-0.55/+0.15% with the sign flipping per case (pooled geomean 2.4733 vs 2.4694, i.e. identical); a whole funded direction and an earlier round's positive sub-claim were both retracted as this artifact.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: bimodal-throttle-defeats-single-shot-gate-method-gfx950-launch-bound
description: A bimodal box mode makes single-shot gates reject real wins and fixed-order paired A/B manufacture fake ones; sample medians and balance order (ABBA).
keywords: ['measurement-noise', 'throttle', 'ab-methodology', 'latency-bound', 'launch-overhead', 'frozen-baseline', 'small-batch', 'interleaved-ab', 'measurement-method', 'control-experiment', 'dispatch-floor']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: method
regime: launch-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
---
# bimodal-throttle-defeats-single-shot-gate
- lever: For an op whose per-case time is dominated by dispatch, judge a candidate on a median of >=10 back-to-back samples, or on an ORDER-BALANCED interleaved >=8-pair same-session A/B of per-case minima (fixed-order interleaving is not paired measurement), and additionally look for the mechanism's own fingerprint in the per-case pattern.
- apply: Run candidate and seed alternately inside one process in a BALANCED order (ABBA / order-reversed halves, not A,B,C,A,B,C), pool the reps, keep per-case minima, and state in advance which per-case relation the mechanism predicts - here removing a serial per-block dependent-load chain predicted the largest-vs-middle batch gap collapsing, and it collapsed about 6x while the largest case's minimum improved 12.7%.
- apply: Re-measure a KNOWN quantity inside the new batch as a batch-validity gate before believing any cross-batch delta - a pooled current-best geomean that reproduces the previous run's value to the third digit is what makes a +3.5% salvage readable at all; when it does not reproduce, the box moved and the delta is unreadable.
- verify: A single slow sample below the incumbent is a sampling outcome, not a regression: re-sample before discarding a candidate, and re-check that the candidate is still bit-exact against the golden tensor each time it is re-applied. Read the harness's own report artifact rather than the rounded stdout line - the rounding step can be coarser than every effect still in play on an op this small.
- pitfall: An earlier round concluded the serial reduction was off the critical path -> that verdict came from a work-adding vectorized variant that measured 0.973x, which disconfirms the variant and not the chain -> eliminating the dependent-load chain at O(1) precompute cost is what pays; bandwidth was never the constraint because the operand is cache-resident.
- pitfall: A candidate reads consistently better with every paired rep favouring it, yet the effect is exactly the size it was funded to find -> fixed-order interleaving leaves a systematic position effect (each variant always occupies the same slot after a warm/discard), so it measures position, not the patch -> re-run the identical variants in reversed/ABBA order; a real effect keeps its sign per case, an artifact flips.
- pitfall: Two batches of the same variant disagree by more than the effect -> on this box the timing is bimodal PER PROCESS (some invocations return a flat per-case profile, others a rising one, and which mode a process lands in is stochastic), so one run can fabricate a double-digit percent win on top of the known run-to-run spread -> pool many order-balanced paired reps across independent processes before reading any single-digit delta.
- caution: Also verify the noise shape before choosing the gate: this signature (two cases moving together, one flat) is what identified a whole-process throttle rather than kernel variance, and a different box may need a different sampling rule.
- caution: Also verify a direction's PREMISE with the balanced protocol before funding a round on it - here an axis was closed by a control, re-opened by a fixed-order artifact, and closed a third time by direct attribution, costing a full round.
- source: 16h per-kernel time-budget campaign, 62 resumed passes, gfx950, 2026-08-11
- source: run kernel_20_geak_0811_2h, 3-round lane on the same op and box, ABBA re-measurement, 2026-08-12
