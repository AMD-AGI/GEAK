---
key: Low-concurrency decode cases of a paged attention on gfx950 whose grid is far smaller than the CU count, so wall time is dominated by dispatch
type: anti-pattern
confidence: ★★
effect: Host graph capture/replay: the smallest case (grid ~8 workgroups vs 256 CUs) stayed tied at 1.00x while the two larger cases regressed to 0.82x (c32) and 0.79x (c64) — a net loss. Dropping a pinned occupancy hint and letting the compiler choose gained only ~0.7% geomean, carried entirely by c32 (~1.9%), with c64 and c2 tied across 13 interleaved runs. 61 passes over one 16h budget moved cumulative from 1.211x to 1.226x, and the 1.32x target stayed out of reach.
confirms_cited: 2
confirms_blind: 0
losses: 2
attempts: 8
toolchain: unknown
last_seen: 2026-08-12
name: a-dispatch-floored-small-case-caps-the-weighted-geomean-attention-decode-gfx950-decode
description: Anti-pattern: when a decode case's grid is far under CU count, graph capture regresses it and occupancy-hint tuning buys ~1%; the geomean target is unreachable.
keywords: ['launch-overhead', 'hip-graph', 'waves-per-eu', 'occupancy', 'attention-decode', 'anti-pattern', 'small-grid', 'decode']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
roofline: smallest case touches ~2% of HBM traffic — pure dispatch floor, not a memory or compute roof
levers: ['host.launch-overhead', 'host.occupancy-hint']
---
# A dispatch-floored small case caps the weighted geomean
- lever: Compute, early, what geomean is reachable if the smallest-grid case stays at 1.00x; here that arithmetic showed the target needed ~1.5x on both larger cases while both were floored at one workgroup per CU, which is the signal to close rather than open another round.
- apply: Grid size versus CU count is the cheap test; a case whose grid is a few percent of the CUs is dispatch-bound, and a pinned occupancy hint on such a kernel is worth at most ~1% either way.
- verify: Sweep the occupancy hint by environment variable with no rebuild and interleave the arms, then read medians per case; a ~1% launcher-only delta is inside a single-shot harness verdict's noise and can be reported as not-improved even when real.
- pitfall: Graph capture removed per-launch overhead yet both larger cases got ~20% slower -> the replayed graph serialized work the stream had been overlapping -> the capture was dropped and the residual overhead treated as irreducible.
- caution: Also verify which case actually carries a sub-1% win before banking the attribution: an earlier wave credited the long-context case and the interleaved re-measurement showed it tied, with the mid case carrying all of it.
- source: 16h per-kernel time-budget campaign (chuschen16h wave, 61 passes), 2026-08-11
