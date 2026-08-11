---
name: stop-shrinking-stored-bytes-once-the-skip-granularity-turns--linear-attention-gfx950-memory-bound
description: Stop shrinking stored bytes once a finer skip mask breaks wide stores: the coarse quadrant skip won -10.8% on the largest case, every finer mask lost
keywords: [linear-attention, memory-bound, coarsening, tile-shape, measurement-method, control-experiment, interleaved-ab]
kernels: [chunk_scaled_dot_kkt_fwd_kernel]
platforms: [gfx950]
kernel_class: linear_attention
regime: memory-bound
key: skip-mask granularity for the triangular output stores of a chunked linear-attention KKT kernel on gfx950, where store issue trades against stored bytes
lifecycle: active
type: anti-pattern
confidence: ★★
effect: The coarse quadrant-level skip (stores 75% of the triangular output, two wide coalesced stores) cut 25% of fp32 store bytes for -10.8% on the largest case and geomean 13.89 -> 14.60x. Every finer variant lost the wide store and failed: 4x4 skip (62.5% stored) showed ~2% in medians but whole-run verify came back 14.5841 vs 14.5993 and was not banked; 16x16 skip failed whole-run verify outright; an alternative left column-band split regressed (128B rows below the 256B burst); element-wise predicated/masked triangular stores break coalescing. Axis recorded CLOSED at coarse 2x2.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.50h / 48 passes, 2026-08-11
last_seen: 2026-08-11
---
# Stop shrinking stored bytes once the skip granularity turns the store issue-bound
- lever: On a triangular or otherwise sparsely-written output, byte reduction and store-issue efficiency trade against each other: fewer bytes bought by a finer skip mask cost coalescing and extra store instructions, and past a point the kernel is store-ISSUE-bound rather than store-byte-bound. Price the coarsest mask that still keeps full-width contiguous stores first.
- apply: keep each surviving store at least a full burst wide (a split that dropped rows below the burst width regressed even though it stored fewer bytes), and enumerate mask granularities coarse-to-fine, stopping at the first one that breaks the wide store.
- verify: score every candidate on whole-run verify, not on per-case medians.
- pitfall: a finer mask read ~2% faster in medians and never banked -> the median window hid the coalescing loss that whole-run verify exposed -> treat a medians-only win on this axis as the signal the axis has closed rather than as a tuning target.
- source: chuschen 16h time-budget campaign run, 15.50h / 48 passes, 2026-08-11
