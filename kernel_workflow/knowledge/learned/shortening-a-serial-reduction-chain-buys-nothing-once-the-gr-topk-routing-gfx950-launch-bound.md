---
name: shortening-a-serial-reduction-chain-buys-nothing-once-the-gr-topk-routing-gfx950-launch-bound
description: Depth reduction on a top-k selection that already fills the device measured 1.008x, while partition-parallel selection on the same code paid 1.185x
keywords: [topk, occupancy, launch-bound, measurement-method, control-experiment, cross-lane, vgpr, dispatch-floor]
kernels: [_topk_forward]
platforms: [gfx950]
kernel_class: topk_routing
regime: launch-bound
key: shortening the serial argmax dependency chain in a Triton top-2 routing selection on gfx950 whose grid already covers the device in one full occupancy wave
lifecycle: active
type: anti-pattern
confidence: ★★
effect: a fused top-2 combiner that halved the serial argmax depth per half (4 reductions + 3 masks down to 2 reductions + 1 mask) passed correctness and produced exactly zero speedup on the one device-exposed case (ledger actual 1.008), and an earlier ground-up rewrite of the same selection re-earned only the state already banked (1.108, that case identical); the disconfirming context is that grid 2048 is one full occupancy wave at 8 blocks per CU with per-SIMD occupancy already at its 8-wave max, ~19-20% of HBM and vgpr 41; the one thing on this axis that did pay was partition-parallel selection at 1.185x on that case, and the campaign's own reading was that the win was ILP overlap, not tree depth
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.59h / 50 passes, 2026-08-11
last_seen: 2026-08-11
---
# Shortening a serial reduction chain buys nothing once the grid already fills the machine
- lever: When a selection or reduction looks latency-bound because of a visible sequential dependency chain, price the premise before funding the rewrite - compute programs per CU and waves per SIMD first, and if the launch already covers the device in a single occupancy wave at max occupancy, depth reduction has no latency to hide and will measure flat.
- apply: Falsify cheaply by shortening the chain with the smallest edit that still passes correctness and looking for any movement at all; a clean zero closes the axis before a full rewrite is written.
- pitfall: a ground-up rewrite of the same selection was scored as a win -> it re-earned only state already banked on that case, because depth was never the cost -> keep depth reduction and overlap-creating restructures as separate ledger entries rather than scoring them as one direction.
- verify: Confirm blocks per CU and per-SIMD wave occupancy on the graded case, then re-measure the minimal depth edit against the frozen baseline.
- caution: Also verify the overlap-creating variant separately - independent partitions merged in registers still paid on this same code where depth reduction did not, so a flat depth result does not close the ILP direction.
- source: chuschen 16h time-budget campaign run, 15.59h / 50 passes, 2026-08-11
