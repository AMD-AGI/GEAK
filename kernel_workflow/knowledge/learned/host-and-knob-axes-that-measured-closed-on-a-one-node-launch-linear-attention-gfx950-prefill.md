---
key: single-launch Triton chunked linear-attention on gfx950 — the host-side and knob-tuning axes that a plan usually opens with, measured closed
type: anti-pattern
confidence: ★★
effect: ≈1.00x or worse on every batch case: graph replay ≈1.7x costlier than eager for a 1-node launch; a 25-variant config ladder never beat the round-1 meta; d(gain)/d(wave) = 0 ± 1%; static_range −17 to −18%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 8
toolchain: unknown
last_seen: 2026-08-12
name: host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill
description: gfx950 Triton small-grid ops: graph capture, config/occupancy ladders, cache policy and k-loop restructuring all returned ~1.00x or worse
keywords: ['anti-pattern', 'hip-graph', 'occupancy', 'config-sweep', 'cache-modifier', 'triton-pipeliner', 'launch-overhead', 'gfx950']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: prefill
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
origin_kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
---
# Host and knob axes that measured closed on a one-node launch
- lever: treat these as already-measured null axes and price a round accordingly: wrapper-level graph capture for a single-node launch, occupancy/register buys, the config ladder, cache-policy hints, partial-sector read-modify-write, and hand k-loop restructuring.
- apply: if one is opened anyway, open it as a cheap ablation with a stated falsifier rather than a patch — each of these was closed here by a control arm in a fraction of a round.
- verify: the discriminating measurements were: replay vs eager on the identical 1-node graph; a sector-identical control arm for read-modify-write (+18% bytes, 0 extra sectors, hurt identically); waves/SIMD swept to both extremes; and an elasticity probe showing the smallest case is ~0.22 elastic to kernel time, capping it near 4.9x.
- pitfall: an ideal sector count derived as bytes/64 framed two rounds of planning and did not exist → the output rows are aligned so head boundaries never split a sector → enumerate the actual sector set before planning a layout change against an implied prize.
- caution: these were measured on a launch that issues one kernel with a small grid; also verify the graph-capture and occupancy axes again on a multi-node capture or a much larger grid, where the intercept they pay is amortized differently.
- source: run chunk_scaled_dot_kkt_fwd_kernel-own16h, 2026-08-12, rounds 2/4/6/7/8/9/11, all refuted on the isolated A/B
