---
name: past-the-raw-driver-submit-the-remaining-host-submit-ideas-a-unmatched-gfx950-launch-bound
description: Past the raw driver launch, submit rewrites close the axis: a native shim measured 2.61x vs 2.613x and a doorbell kernel was a clean true-negative
keywords: [launch-overhead, dispatch-floor, launch-bound, graph-capture, measurement-method, control-experiment, interleaved-ab]
kernels: []
platforms: [gfx950]
kernel_class: method
regime: launch-bound
key: exhausting host-submission rewrites (ctypes graph launch, native pybind shim, persistent doorbell kernel) on a dispatch-bound Triton kernel on gfx950 that already submits through the raw driver launch entry
lifecycle: active
type: anti-pattern
confidence: ★★
effect: four host-submit disconfirmations on the same three graded batch cases (2 / 32 / 64, whose baselines sit within ~6% of each other, so no case is body-bound) - two on the graph-capture seed, where native ctypes hipGraphLaunch came within 0.1% of Python g.replay() (1.2487x vs 1.322x, logged TRUE-NEG) and flattening the replay bracket to minimal Python gave 1.2814x, <1% and below the timer floor; then two after the raw driver launch landed at 2.613x, where a native pybind submit shim was metric-neutral (2.61x vs 2.613x) and a persistent/doorbell kernel was a clean TRUE-NEG with a doorbell round-trip ~2.1x the raw launch's per-call cost, plus the harness device-wide sync deadlocked the resident kernel; device-side knobs were flat too (num_warps/num_stages 1.000x, BLOCK_SIZE 1024 0.998x), and the run ended with 'no unexplored lever remains' after 62 passes
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: run chuschen16h 2026-08-11
last_seen: 2026-08-11
---
# Past the raw driver submit, the remaining host-submit ideas are a closed axis
- lever: Once a dispatch-bound kernel submits through the driver's own launch entry point with pre-packed arguments, the layer below it is the driver, and further submission rewrites (a second language for the same call, a persistent kernel fed by a doorbell, another wrapper around graph launch) tend to measure neutral or worse rather than incrementally better.
- apply: Close the axis cheaply instead of paying a round per idea - time the mechanism's own round-trip in isolation against the current per-call launch cost before writing it.
- pitfall: a persistent/doorbell kernel was queued as the next submit idea -> its round-trip alone was ~2.1x the raw launch it would replace, and the harness's device-wide sync deadlocked a resident kernel -> check both the mechanism's isolated cost and whether it can coexist with the harness synchronisation model before implementing it.
- verify: Score each rewrite against the current best submit path (not the original baseline), and treat a gap under the timer floor as a TRUE-NEG rather than a small win.
- caution: Also verify the cases really are dispatch-bound - here all three baselines were within a few percent of each other regardless of batch, which is what made the body knobs flat.
- source: run chuschen16h 2026-08-11
