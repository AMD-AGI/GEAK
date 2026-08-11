---
name: move-small-k-selection-onto-a-chunk-axis-off-the-distributed-topk-routing-gfx950-launch-bound
description: Restate small-k selection over a chunk axis instead of Triton's distributed axis: device time -30 to -33% on every case at zero occupancy cost
keywords: [topk, cross-lane, isa-check, vgpr, occupancy, launch-bound, interleaved-ab, control-experiment]
kernels: [_topk_forward]
platforms: [gfx950]
kernel_class: topk_routing
regime: launch-bound
key: small-k selection (top-k) restated over a chunk axis rather than Triton's distributed axis, gfx950/CDNA4, small grids where a launch floor owns half the wall
lifecycle: active
type: lever
confidence: ★★
effect: Device time -30 to -33% on every case (per-case -33% / -30% / -32% on the three grids) at zero occupancy cost; per-case at the wall it paid only where device work was exposed: +26% on the largest grid, ~+3% (inside noise) on the two smaller launch-bound cases, +10.4% geomean against a same-session paired control, verified cumulative step 1.699x -> 1.807x.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: triton 3.6.0 / torch 2.11.0+gitd0c8b1f / gfx950 CDNA4
source: run kernel_20_geak_0808_4h 2026-08-08
last_seen: 2026-08-10
---
# Move small-k selection onto a chunk axis, off the distributed axis
- lever: For a small-k selection in Triton the real cost is often cross-lane traffic, not bytes and not occupancy: tl.topk / tl.sort / tl.softmax / reduce_or along Triton's distributed axis each expand into permute networks. Restating the selection over a separate chunk axis is what paid here; the three attempts that kept the work on the distributed axis (full-width single-shot, chain shortening, mask/select elimination) each measured out between 0 and -1.4%.
- apply: Load chunks into separate tiles, do the elementwise compare/sort work per chunk, collapse the chunks with exactly ONE tl.reduce carrying a tuple combine_fn (a bitonic k-way merge), and keep the k winners as k rank-1 tiles so the whole tail stays elementwise.
- verify: Check the compiled object rather than the source: more registers with waves/SIMD unchanged is a free win (16->28 VGPR held min(8, 512/vgpr) at 8 here, so compute the occupancy step before rejecting a register-hungry rewrite), and dump the ISA to confirm the permute/DPP ops actually went away; then take wall in a paired same-session interleave, because a 31% device win converted to under 5% of wall once the launch floor was over half of it.
- caution: Before spending a round, price the whole arithmetic lane with a delete-the-work probe (keep every load, store and epilogue, remove only the computation), and re-run that probe against current HEAD: ours re-ran 8x smaller after later launch-path work landed, capping the remaining in-kernel lane at +6% on the one device-exposed case and +0.4% geomean. Also verify a wider load tile against bytes-per-lane-per-instruction first, four 64B segments already issued the ISA-max dwordx4.
- source: run kernel_20_geak_0808_4h 2026-08-08
