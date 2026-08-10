---
key: dense gemm · gfx950 · compute-bound
type: lever
confidence: ★★
effect: launch-config plus pid-order retuning alone carried a captured GEMM from 1.00x to 2.64x with a byte-identical kernel body, and a fixed-area aspect re-sweep (wide to square) then added +7.5% on all three cases at once (per-case 2.22x / 3.21x / 3.21x, small-M and large-M shapes alike); by contrast the best kernel-body change of the entire run was +2.8%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
---
# Tile aspect at fixed area, under a hard accumulator-per-lane wall
- lever: On a captured Triton GEMM, treat the launch config as the first-order lever and sweep tile ASPECT at fixed area before growing area: accumulators per lane = tile_area / (64 * num_warps) with num_warps capped at 16, so a tile above ~64 fp32 accumulators per lane spills and lands 3-4x SLOWER than the untuned baseline (measured 0.27-0.38x), and at equal legal area square beat wide-N beat tall-M (2.84 / 2.71 / 2.62) because the N-side operand is shared across the whole M-grid through L2 while the M-side operand is not.
- apply: Enumerate the legal lattice first (tl.arange forces power-of-2 block sizes, so the tile and BLOCK_K ladders are short and finite), sweep aspect at fixed area, and retune the pid-group size after every tile change: a group size tuned against a 64-row tile spans 8x the intended rows under a 512-row one.
- verify: Rank candidates on wall-clock ms from an interleaved in-process A/B that re-times the control inside every rep, and read VGPR count and scratch off the compile to separate a spilled config (a cliff) from a merely poor one (a few percent).
- caution: Also verify each cache-modifier and pid-grouping choice again after the tile moves, since they are tuned against a traffic mix rather than against the code: a write-through epilogue hint worth +4-6% at one tile cost -5% at the next, and pid grouping went from worth 15x to flat across a single aspect change.
- source: run kernel_20_geak_0808_4h 2026-08-08
