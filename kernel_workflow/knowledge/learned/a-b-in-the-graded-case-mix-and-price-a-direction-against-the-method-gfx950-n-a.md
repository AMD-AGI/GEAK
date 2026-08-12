---
key: isolated A/B methodology when the graded case set mixes launch-floor tiny cases with heavy long-context ones, gfx950 attention campaign
type: method
confidence: ★★
effect: same direction measured +2.5% on the heavy-only subset and +0.3% in the graded full mix; two same-session A/B pairs over the tiny cases disagreed 1.107x vs 0.993x, and a baseline-only control of the unchanged path came out ahead of the candidate, so a projected +5.5% on those cases resolved to 0.988x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a
description: A/B in the grader's own case mix and session: a heavy-only subset showed +2.5% that was +0.3% when graded, and launch-floor cases swing ~20% across sessions
keywords: ['measurement', 'noise-floor', 'ab-methodology', 'launch-overhead', 'thermal-drift', 'anti-pattern', 'paged-attention']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: archived
verified_on: 2026-07-30
---
# A/B in the graded case mix, and price a direction against the noise floor first
- lever: Run each candidate A/B in the same case mix and same session as the graded number, and treat launch-floor cases as a noise channel rather than as recoverable geomean.
- apply: Before assigning a direction whose upside lives in the small cases, compare its projected gain to the observed session-to-session spread at the launch floor (~20% here); a projection below that spread is unmeasurable, however sound the mechanism.
- verify: Add a baseline-only control pair inside the same session — if the unchanged path comes out ahead of the candidate, what you measured is session/clock state rather than the kernel.
- pitfall: a self-consistent gain on a heavy-only subset evaporated when graded -> running only heavy shapes holds a different sustained clock/thermal state than interleaving them with tiny ones -> re-measure in the graded mix before reporting.
a correctness-safe, well-founded host-side size-gate still landed below 1.0 -> its premise came from comparing two different sessions -> settle the premise with a same-session pair before spending a round building the gate.
- caution: Here the heavy-case baseline was stable run-to-run (<0.1%) while the tiny-case one was not; also verify which half of your own case set is stable before attributing a small delta to a candidate.
- source: run paged_attention_large-ch16h, 2026-07-30 — same-mode back-to-back A/B pairs plus a baseline-only control
