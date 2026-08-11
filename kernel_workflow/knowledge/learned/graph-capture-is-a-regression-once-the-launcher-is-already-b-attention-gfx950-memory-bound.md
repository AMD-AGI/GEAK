---
name: graph-capture-is-a-regression-once-the-launcher-is-already-b-attention-gfx950-memory-bound
description: Price the host path before capturing: with a lean launcher already below graph-launch cost, HIP-graph capture regressed -49.6%/-40.2%/-29.0% at batch 2/32/64
keywords: [graph-capture, launch-overhead, dispatch-floor, measurement-method, control-experiment, attention, memory-bound, interleaved-ab]
kernels: [paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: wrapper-level HIP graph capture/replay on an attention call on gfx950 whose Python launch path had already been collapsed to a lean launcher, enqueued back-to-back
lifecycle: active
type: anti-pattern
confidence: ★★
effect: wrapper-level HIP-graph capture measured a strict regression on every grid - -49.6% / -40.2% / -29.0% at batch 2 / 32 / 64 - and no patch shipped; the lean launcher had already put the host path below graph-launch cost, so the swap added back roughly 1.4-2.3x the entire host contribution it was meant to remove, and async back-to-back enqueues overlap with the previous call's GPU compute anyway, with profiling putting the host share at a small fraction of the call and the wall being GPU compute
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.63h / 51 passes, 2026-08-11
last_seen: 2026-08-11
---
# Graph capture is a regression once the launcher is already below graph-launch cost
- lever: Treat graph capture as competing against the current host path, not against zero - price the per-call host share first, and if it is already smaller than the capture/replay cost the swap can only lose.
- apply: Order the work so the Python/wrapper launch path is collapsed first, then re-measure the host share before deciding whether any capture direction is still funded.
- pitfall: capture was budgeted off an earlier profile -> the lean launcher landed in between and moved the host share below the replay entry cost -> a lever that paid before that collapse inverted after it, so re-price capture against the CURRENT launcher rather than the one it was budgeted against.
- verify: Same-session A/B on every grid, plus a separate measurement of the per-call host share against the replay entry cost.
- caution: Also verify how the caller enqueues - back-to-back async enqueues already hide under the previous dispatch, so capture keeps more of its margin where the caller synchronises between calls or where one capture covers many kernels.
- source: chuschen 16h time-budget campaign run, 15.63h / 51 passes, 2026-08-11
