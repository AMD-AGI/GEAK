---
key: A/B hygiene for attention suites that mix launch-floor-sized cases with long-context heavy cases on one grader geomean
type: instrument
confidence: ★★
effect: Same candidate, two measurement modes: +2.5% on the heavy-only subset vs +0.3% on the grader's full mix (per-case, the heavy shapes were flat in both). At the launch floor, same-session A/B pairs on identical code disagreed 1.107 vs 0.993, and a baseline-only control of the unchanged path timed FASTER than the candidate; the full-mix baseline itself repeats within 0.1%. A projected +5.5% recovery on those cases measured 0.988x.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: measure-in-the-grader-s-own-case-mix-sub-noise-floor-cases-c-method-gfx950-n-a
description: Instrument: a heavy-only subset reported +2.5% where the grader's full mix showed +0.3%; launch-floor cases drift ~20% across sessions and cannot be won
keywords: ['measurement', 'noise-floor', 'ab-testing', 'launch-overhead', 'decode', 'gfx950', 'dead-end']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
verified_on: 2026-08-11
---
# Measure in the grader's own case mix; sub-noise-floor cases carry no signal
- lever: Run every A/B in the same case mix and the same session the grader scores, and treat a case whose duration sits at the dispatch floor as a source of variance rather than a source of geomean.
- apply: Before costing a round on a size-gate or a tiny-case fix, run a null A/B (unchanged path vs itself) across sessions and size the drift; if the claimed gain is under that drift, the gain is the drift.
- verify: Reproduce the claimed delta back-to-back in one session in the scoring mode; a cooler start-of-session baseline compared against a later candidate is not a comparison.
- pitfall: A subset mode gave a self-consistent, repeatable gain that vanished in the full mix → running only the heavy cases holds the part in a different sustained clock/thermal state than interleaving them with tiny ones → A/B in the scoring mix.
A round stopped early at 1.06x on an axis marked dead by a sibling kernel → the verdict was inherited, never measured on this part → re-measuring locally reopened it and reached 1.30x; a stop gate is only as good as the measurement under it.
- caution: Also verify that a host-side gate you are about to build has a payoff that exists at all: the gate here was correct and free (a shape proxy, no device sync, cleanly separating the two populations) and still bought nothing, because the population it routed had no recoverable signal.
- source: 16h single-kernel time-budget campaign, run chuschen16h, 2026-08-11
