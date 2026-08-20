---
key: certifying sub-1% isolated A/B effects late in a long decode-attention campaign on gfx950, where run-to-run spread is small but arms are compared in a fixed order
type: method
confidence: ★★
effect: single-order pilots carried 0.3-0.4% weighted of pure slot bias against a 0.15% weighted admissibility bar and a ~0.3% per-case floor on stable cases, while full-benchmark repeat spread was under 0.3%; three published per-case numbers were revised under order balancing and one direction flipped sign; the survivor published as +0.469% weighted, 95% CI +0.24%..+0.77%, n=336 per arm per order over 16 balanced pairs
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a
description: A fixed-order A/A control measures load-order bias and then reports it as the noise floor: one bit-identical arm read -2.03% in one order, -0.10% in the other
keywords: ['method', 'ab-methodology', 'measurement', 'measurement-discipline', 'negative-control', 'noise-floor', 'paired-ab-rig', 'frozen-baseline', 'small-effect', 'graph-replay', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-15
levers: ['method.ab-protocol']
origin_kernels: ['mi355x_vllm_triton_unified_attention_gemma4']
---
# A fixed-order A/A control measures order bias and then hides it
- lever: Balance arm order (ABBA) rather than only interleaving, and run the A/A null in the same balanced order; a fixed-order A/A folds slot bias into what it reports as the noise floor, so the bias survives the control that exists to catch it.
- apply: Pair every A/B with a bit-identical control in both orders, in one tree and one session; publish an interval rather than a point, and gate on dispersion as well as on the point estimate.
- verify: The order-balanced A/A should land near zero in both orders; if the two orders disagree by more than the effect being chased, no number off that rig is admissible yet.
- pitfall: An arm published at +0.244% re-measured to -0.004% -> single-order slot bias exceeded the admissibility bar -> require the balanced null and the dispersion before banking a sub-1% result.
A geometry the change cannot reach reads as an unreadable case -> it is a free bit-identical control -> collect that null arm every round and let it honestly zero the case instead of scoring a false win.
An arm reported shipped and verified had been applied inside a private clone -> the label was checked instead of the artifact -> verify by hashing the target file and reading the diff, not by a return code or a status line.
- caution: Also verify what the timed bracket actually contains before funding host-side directions: where the harness times the replay of a captured graph, wrapper-side work never executes inside the measured window at all, which makes that family mechanically unable to move the score on that rig.
- source: run mi355x_vllm_triton_unified_attention_gemma4-bmk7-12h, rounds 8-15 measurement backbone, 2026-08-17
