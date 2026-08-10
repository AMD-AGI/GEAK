---
key: topk / routing · gfx950 · launch-bound
type: lever
confidence: ★★
effect: 1.58x geomean standalone, and per-case it tracks how launch-bound the case is: 2.19x on the smallest grid (~64 CTAs, device time fully hidden behind the launch) / 1.67x mid / 1.05x on the largest (~2048 CTAs, device-exposed). Largest single component of a director-verified 2.28x end state (per-case 2.35 / 2.49 / 2.03).
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: triton 3.6.0 / torch 2.11.0+gitd0c8b1f / gfx950 CDNA4
last_seen: 2026-08-08
---
# Collapse the Python launch path before tuning the body
- lever: On a small op whose per-call device time is single-digit microseconds and whose caller re-invokes it with identical argument objects, Triton's Python launch path (signature binding, arg specialization and hashing, grid canonicalization) can be most of the measured wall; collapse it before spending a round on the kernel body, and re-test config knobs afterwards because a device-side win can score as a regression while a host floor hides it.
- apply: Bind the CompiledKernel once out of the JIT's per-device cache, cache the bound args and grid behind an argument-identity guard with a full fallback to the stock path on any mismatch, and keep the wrapper's call protocol identical to the JITFunction's (subscript-for-grid, plus a .fn whose __globals__ stay reachable) so anything that introspects the callable still works.
- verify: Measure per-call CPU enqueue directly (many back-to-back launches, no sync inside, MIN of several) AND confirm one full dispatch per timed iteration in a profiler trace, since a cache that memoizes or early-returns shows fewer dispatches; then re-measure wall in an order-balanced ABBA interleave against a same-session pristine control, strict A-then-B biased toward the first arm here badly enough that MIN and median disagreed (1.019 vs 0.905).
- caution: The first collapse takes nearly all of it: below roughly 2.5-3.5 us/call of enqueue, d(wall)/d(enqueue) fell to ~0 here, a further verified 24-27% enqueue cut (0.85 us/call) moved the wall 0.0-0.7% at a 53% block win rate, so also price any residual with a 3-arm slope test before spending a round on it. Also verify graph capture on its own terms: a validated 1-node replay cost 2x a cached direct launch, and a silently empty capture reads as +45%.
- source: run kernel_20_geak_0808_4h 2026-08-08
