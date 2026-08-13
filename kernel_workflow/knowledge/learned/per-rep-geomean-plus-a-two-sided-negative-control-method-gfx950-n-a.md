---
key: resolving effects of a couple of percent on a short-running Triton op, where a 3-case geomean dilutes a real per-case device saving
type: method
confidence: ★★
effect: resolved a real, bit-exact per-case effect of +1.37% and +1.29% at 92% sign consistency over 62 paired reps; the control arm moved the same case +0.87% and +4.38% monotonically with injected identity work and flipped the geomean -1.46%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: per-rep-geomean-plus-a-two-sided-negative-control-method-gfx950-n-a
description: Certify or decline a sub-2% kernel effect: per-rep paired geomean plus a two-sided identity-work control and an acceptance bar fixed before measuring
keywords: ['method', 'ab-harness', 'negative-control', 'sign-consistency', 'small-effect', 'dispatch-bound', 'gfx950']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L0
verified_on: 2026-08-12
---
# Per-rep geomean plus a two-sided negative control
- lever: compute the geomean per rep and keep the distribution, rather than pooling the min across reps into a single number.
- apply: paired interleaved A/B over tens of reps; report the median paired ratio and the sign consistency; fix the acceptance bar before the first measurement.
- verify: add a two-sided negative control - inject provably output-neutral work at two magnitudes and confirm the wall clock moves monotonically in the expected direction; a small positive is credible once the harness has been shown to resolve that magnitude both ways.
- pitfall: the pooled-min A/B emitted one number per candidate -> sign consistency and the paired ratio distribution were unobservable -> a genuine ~1.3% effect could be neither certified nor declined, and an unbounded tail of rounds looked justified.
- caution: also verify a certified-real effect against the bar stated in advance; here a real bit-exact +1.3% fell under a +1.5% bar, and declining it is what let the campaign terminate defensibly.
- source: run kernel_20_geak_0808_16h lane _topk_forward, round 6, 2026-08-12, tech_lead_report.md
