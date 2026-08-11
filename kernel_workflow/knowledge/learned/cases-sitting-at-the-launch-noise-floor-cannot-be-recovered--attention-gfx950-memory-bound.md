---
name: cases-sitting-at-the-launch-noise-floor-cannot-be-recovered--attention-gfx950-memory-bound
description: Price the noise before funding a small-case recovery: a size-gate budgeted at 1.37x returned 0.988x, and a baseline-only control moved as much as the candidate
keywords: [measurement-method, control-experiment, interleaved-ab, dispatch-floor, harness-artifact, attention, memory-bound, launch-overhead]
kernels: [paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: pricing a host-side size-gate that routes the small cases of a mixed 16-case attention suite back to the wide-dtype path on gfx950, where those cases already sit at the dispatch floor
lifecycle: active
type: anti-pattern
confidence: ★★
effect: a host-side size-gate routing the small cases back to the wide-dtype path was budgeted at 1.37x off a projected +5.5% recovery on them and returned 0.988x, verdict dead_end; the gate itself was correct and free (proxy from block-table extent, no device-to-host sync, cleanly separating the small set from the large set, correctness PASS) - the premise was the artifact, since three same-session A/B pairs on those cases disagreed at 1.107 vs 0.993 and a baseline-only control of the UNCHANGED path measured FASTER than the candidate, so the small-case latency is session/clock/thermal-driven rather than path-driven; separately a large-cases-only measurement mode showed a self-consistent ~2.5% gain for a compute direction that a same-mode A/B in the graded all-16 mode reduced to ~0.3% (ledger actual 1.00771) while the all-16 baseline itself is stable to <0.1% run-to-run, and a related threshold-branch direction built on a similar false premise returned exactly 1.0
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 16.43h / 6 passes, 2026-08-11
last_seen: 2026-08-11
---
# Cases sitting at the launch/noise floor cannot be recovered, and cross-mode or cross-session A/B will manufacture the signal that says they can
- lever: Before funding a direction whose whole payoff is a recovery on the small/fast cases of a mixed suite, price the noise first - run the same-session A/B two or three times and run a baseline-vs-baseline control on the unchanged path; if the control moves as much as the candidate, those cases are at the dispatch floor and hold no recoverable geomean, so the axis can be closed for the rest of the campaign rather than re-assigned.
- apply: Derive the routing proxy from data the host already has (here block-table extent, no device-to-host sync) so the gate itself costs nothing, then judge it against the repeated control rather than against a single pair.
- pitfall: a compute direction read as a self-consistent ~2.5% gain -> it was A/B'd in a large-cases-only mode that holds the clocks in a different sustained state than the graded mode -> re-run in the same measurement mode the grader scores, where the same direction came back at ~0.3%.
- verify: Same-session interleaved pairs only (a cooler start-of-session baseline is not comparable to a same-session candidate), plus a baseline-vs-baseline control, plus the run-to-run stability of the graded mode itself.
- caution: Also verify the gate's own correctness separately from its premise - here the mechanism was clean and only the projected headroom was imaginary, so the same gate may still pay on a suite whose small cases are not at the floor.
- source: chuschen 16h time-budget campaign run, 16.43h / 6 passes, 2026-08-11
