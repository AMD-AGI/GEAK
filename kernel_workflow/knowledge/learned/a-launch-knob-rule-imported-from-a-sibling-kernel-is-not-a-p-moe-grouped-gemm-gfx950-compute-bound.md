---
name: a-launch-knob-rule-imported-from-a-sibling-kernel-is-not-a-p-moe-grouped-gemm-gfx950-compute-bound
description: Re-measure an imported launch-knob rule at your own tile: every rule ported from a sibling MoE kernel lost, one of them up to 6.5x worse
keywords: [launch-config, config-sweep, moe, tile-shape, mfma, vgpr, compute-bound, measurement-method]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: imported launch-knob folklore (MFMA shape, kpack, num_warps, BLOCK_K/num_stages) on a Triton fp8 MoE grouped GEMM pinned at a [128,256] tile, gfx950
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Every imported rule tested here lost, and each cost a round that ended in an empty patch at cumulative 3.3326x. The 16x16 -> 32x32 MFMA-shape win from a neighbouring MoE stage-1 kernel was a pure regression at this tile (+18.4% / +1.2% / +1.7% on the three cases), and kpack 0/1/2 had no observable MFMA-feed effect on the [128,256] x BLOCK_K=128 tile. The 'wave64 wants num_warps=4' rule from the vLLM configs was up to 6.5x worse — it spills the fp32 accumulator — with num_warps=8 the sweep winner. On the pipelining axis the incumbent BLOCK_K=128 / num_stages=2 won in every M-bucket by a wide margin: num_stages=1 was ~3x worse at that tile, BLOCK_K=64 uniformly +12-58%, and BLOCK_K>=256 or num_stages=3 aborted on LDS overflow on the two large cases. Every combination that compiled and ran stayed bit-correct (recorded explicitly for the MFMA-shape/kpack sweep), so these were perf losses rather than correctness filters; the BLOCK_K>=256 / num_stages=3 points never ran at all.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.72h / 49 passes, 2026-08-11
last_seen: 2026-08-11
---
# A launch-knob rule imported from a sibling kernel is not a prior at a different tile
- lever: Config folklore travels by kernel family, but these knobs are coupled to the accumulator footprint of the tile actually instantiated, not to the family or the wavefront width, so a rule that won next door is worth one measurement rather than a starting point.
- apply: A/B the imported value against the incumbent at the real tile in one M-bucket first, and treat a knob whose failure mode is a spilled fp32 accumulator or an LDS overflow as a closed axis once the tile is fixed; recording the axis as closed is what lets a later pass stop instead of re-funding it.
- verify: Confirm each combination that compiled stayed bit-correct before reading its loss as a perf result, and keep configs that never ran (LDS-overflow abort) separate from configs that ran and lost — only the second kind is evidence about the tile.
- pitfall: three consecutive rounds ended in an empty patch -> imported MFMA-shape, warp-count and stage/BLOCK_K rules were each entered as priors rather than as one measurement each -> price each import with a single-bucket A/B, then declare the axis closed and move the round elsewhere.
- caution: Also verify wavefront-width folklore against the accumulator footprint before adopting it: the 'wave64 wants num_warps=4' rule spilled the fp32 accumulator here and num_warps=8 was the sweep winner.
- source: chuschen 16h time-budget campaign run, 15.72h / 49 passes, 2026-08-11
