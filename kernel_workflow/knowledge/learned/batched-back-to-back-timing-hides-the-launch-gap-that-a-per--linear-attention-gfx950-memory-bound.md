---
name: batched-back-to-back-timing-hides-the-launch-gap-that-a-per--linear-attention-gfx950-memory-bound
description: Match the gate's timing shape to the scoring metric: a back-to-back gate silently rejected graph replay, and forcing it on paid exactly 1.0x
keywords: [graph-capture, launch-overhead, dispatch-floor, measurement-method, interleaved-ab, control-experiment, linear-attention, memory-bound]
kernels: [chunk_scaled_dot_kkt_fwd_kernel]
platforms: [gfx950]
kernel_class: linear_attention
regime: memory-bound
key: graph capture/replay on a Triton chunk-scaled-dot linear-attention kernel on gfx950, gated by a back-to-back throughput loop but scored by a per-call event bracket
lifecycle: active
type: anti-pattern
confidence: ★★
effect: a graph capture/replay lever believed to be banked on the smallest case was SILENTLY INACTIVE - its measured-benefit gate timed 20 back-to-back launches under one sync (throughput, host runs ahead) and rejected replay, while the scoring harness brackets each of 100 calls with events; forcing replay to engage and stripping the per-call pointer scan verified correct (cos=1.0) but paid 1.0x - an interleaved alternating bench (7 samples) gave an identical median for eager and replay, and an isolated in-process probe claiming a ~1.5x replay win did not transfer because a fresh short harness process already measures eager within ~10% of that probe's replay figure; the small case sits on the irreducible event-bracket floor and replay on the two larger cases is a fixed-cost pure loss of roughly a quarter of the small case's per-call bracket
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.50h / 48 passes, 2026-08-11
last_seen: 2026-08-11
---
# Batched back-to-back timing hides the launch gap that a per-call event bracket charges
- lever: Before funding host-launch work, make the gate that decides it use the same timing shape as the scoring metric - a throughput-style loop of launches under a single sync pipelines the launch gap away and will reject a lever the per-call event-bracket metric would have paid for.
- apply: Grade with an interleaved alternating A/B inside the harness itself rather than in a side probe, and re-read the gate's own timing loop before trusting a 'no benefit' verdict from it.
- pitfall: an isolated long-lived-process probe showed a clear replay win -> a long-lived process inflates the eager baseline that a fresh short harness process never pays -> reproduce any host-side gain in a fresh harness process before banking it.
- verify: Confirm the lever actually engaged (a gate can silently no-op while still reporting a speedup), then re-time interleaved on every case.
- caution: Also verify where the smallest case sits relative to the event-bracket floor - once it measures at that floor, treat further host-side collapse as unreachable rather than as remaining headroom.
- source: chuschen 16h time-budget campaign run, 15.50h / 48 passes, 2026-08-11
