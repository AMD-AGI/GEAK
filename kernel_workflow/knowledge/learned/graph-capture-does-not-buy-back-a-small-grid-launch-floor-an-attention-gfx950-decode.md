---
name: graph-capture-does-not-buy-back-a-small-grid-launch-floor-an-attention-gfx950-decode
description: Price graph capture instead of assuming it: on decode attention it left the launch-floored small case tied and taxed the two larger cases ~18-21%
keywords: [graph-capture, launch-overhead, dispatch-floor, decode, attention, measurement-method, control-experiment]
kernels: [_fwd_grouped_kernel_stage1]
platforms: [gfx950]
kernel_class: attention
regime: decode
key: wrapper-level HIP-graph capture/replay on a Triton paged attention decode kernel on gfx950, where the smallest case is pinned by per-launch overhead
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Disconfirmed: wrapper-level HIP-graph capture/replay left the smallest, launch-floored case TIED and regressed the two larger cases to 0.819x and 0.786x — a net dead_end against an expected 1.32x. The floor it was aimed at survived: that smallest case (grid ~8 workgroups, ~2% of HBM) stayed at ~1.0x for the whole 15.60h / 61-pass campaign, and the run's end state was 1.0148x / 1.3055x / 1.3451x on the three decode cases (banked geomean 1.2261x, reported max 1.24x).
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 2026-08-11
last_seen: 2026-08-11
---
# Graph capture does not buy back a small-grid launch floor, and it taxes the larger cases
- lever: When a decode case is pinned by per-launch overhead, treat graph capture as a hypothesis to be priced rather than the obvious fix: capture only removes per-dispatch launch submission, so if the residual floor is event/setup cost it moves nothing, while the replay path can add real time to cases that were already device-bound.
- apply: cheaper first probe — read what fraction of the small case's wall is actually dispatch before funding the direction; only then build the capture variant.
- verify: measure the capture variant on every case, not just the one it was written for; the sign of the delta flips with grid size, so a single-case reading cannot tell a win from a net regression.
- pitfall: a direction briefed at 1.32x banked nothing -> capture removed submission cost the small case was not actually paying, and the replay path added time on the device-bound cases -> price the dispatch fraction first and score the variant across the whole case set.
- source: chuschen 16h time-budget campaign run, 2026-08-11
