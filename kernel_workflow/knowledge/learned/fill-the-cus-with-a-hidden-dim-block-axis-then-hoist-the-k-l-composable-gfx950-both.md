---
name: fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both
description: Composable TileLang pre/GEMM/post chain: add a hidden-dim block axis to the token-only grid, hoist the k-loop bounds guard out of the GEMM; 1.88x stacked
keywords: ['cu-underfill', 'grid-occupancy', 'loop-hoisting', 'tile-geometry', 'kernel-fusion', 'anti-pattern', 'oracle-parity', 'measurement-discipline', 'gfx950']
kernels: ['mhc_post_tilelang_kernel', 'mhc_pre_big_fuse_tilelang_kernel', 'hc_prenorm_gemm_block_m_v2_tilelang_kernel']
platforms: ['gfx950']
kernel_class: composable
regime: both
key: multi-dispatch TileLang pre-norm/GEMM/post chain in vLLM, one grid axis over tokens, decode+prefill graded together on gfx950
layer: learned
levers: ['grid.hidden-dim-block-axis', 'compute.guard-hoist']
cost: L3
lifecycle: active
type: lever
confidence: ★★
effect: 1.88x isolated geomean, director-verified vs frozen baseline, non-overlapping; per case decode 2.25x and 1.96x, prefill 1.76x and 1.69x, all four oracle-gated
roofline: decode issue/fill-bound (CTAs below CU count) -> memory-bound; prefill after the win has two of three dispatches at 70-101% of achievable HBM and the GEMM at ~57% of achievable HBM simultaneously with ~48% of vector-fp32 peak
verified_on: 2026-08-17
last_seen: 2026-08-17
---
# Fill the CUs with a hidden-dim block axis, then hoist the k-loop guard
- lever: on a chained pre/GEMM/post op whose elementwise stages are gridded over tokens only, add a second block axis over the hidden dim so small-batch CTA count crosses CU count; independently, hoist the per-iteration bounds guard out of the GEMM k-loop before touching tiles or staging.
- apply: two disjoint patches on disjoint dispatches - a 2-D (token, hidden-block) grid plus wider workgroups on the elementwise stages, and a guard-free k-loop with a re-swept block_m on the GEMM. Keep the k-linear walk; the achieved bandwidth is a property of it.
- stack: total 1.88x isolated (geomean, director-verified) = two directions integrated
  - 1. grid.hidden-dim-block-axis - 1.34x standalone (round 1, verified) - carries the decode cases
  - 2. compute.guard-hoist + block_m retune - 1.36x standalone (round 1, verified) - carries the prefill cases; the hoist alone is ~2.2x on that dispatch at the narrow block_m
  - note: the two own disjoint dispatches and multiplied cleanly; a third direction (collapsing dispatch count behind an existing fusion gate) verified ~1.01x alone and dropped the merge, so it was rejected. A later grid-level uniform-branch split measured +2.9% weighted on a paired control but did not clear the promotion gate and is NOT in the total.
- verify: check launched CTAs against CU count per case before and after, and re-run the frozen-baseline A/B per case - the two levers land on different regimes, so a geomean alone hides both.
- pitfall: kernel edits changed nothing measurable -> the symbol resolved from the INSTALLED package rather than the workspace tree -> repoint the loader, then grep EVERY import site; one site stayed uncovered a full round after the first fix.
- pitfall: the fastest sweep configs benchmarked ~20% faster with ~94.5% element mismatch -> the bench driver checks no outputs -> correctness-gate every reported timing at the same tree state it was timed in.
- pitfall: a thread-count raise delivered its predicted gain for the wrong reason -> the limiter was an intra-CTA divergent branch, not occupancy -> what ports is the grid-level uniform branch per CTA; wider workgroups also hit TileLang layout-inference failure past a threshold.
- pitfall: unrolling a loop that loads a weight tile per iteration put all iterations' tiles in flight (occupancy 1, AGPR overflow) -> use a serial loop; and a thread-local tile indexed by a serial loop variable silently degraded to scratch with hundreds of spills -> move that tile to LDS.
- caution: also verify a proposed post-into-GEMM fusion with a no-op-epilogue ablation first - with the epilogue deleted entirely the fused form was still ~2.3x the unfused GEMM, because the access order the fusion forces costs more than the dispatch it deletes.
- caution: also verify sub-1.5% deltas with a paired interleaved control - the same case drifted and returned across one session with no relevant change.
- source: run kernel_20_geak_0811_2h, TileLang MHC post+pre lane, 2 rounds, director-validated 2026-08-17
