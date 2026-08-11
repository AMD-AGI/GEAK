---
name: price-an-inferred-padding-or-grid-quantization-tax-with-a-bl-moe-grouped-gemm-gfx950-mixed
description: Price an inferred padding or grid-quantization tax with a block-count sweep before funding a round: both inferred taxes were fictions and the lane returned 0%
keywords: [grid-geometry, control-experiment, measurement-method, moe, tile-shape, memory-bound, launch-config]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: mixed
key: pricing a suspected row-padding or grid-quantization tax on an aligned Triton MoE grouped GEMM (gfx950) from the alignment metadata before funding a round
lifecycle: active
type: method
confidence: ★★
effect: Two taxes inferred from grid arithmetic (a 1.5x row-padding tax and a grid staircase) survived four consecutive round plans unmeasured and both turned out to be fictions: a 120..416 block sweep gave latency LINEAR in block count with per-block cost improving ~1.17x across the sweep instead of worsening, and the alignment metadata had ZERO fully-padded blocks (mean 42.6 valid rows of 64). The lane returned 0% over 3 budget units while the small-batch case stayed at 2.49x against 3.14x / 3.18x on the large-batch cases (run 2.92x geomean, director-verified).
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
source: run kb_on_0810 2026-08-10
last_seen: 2026-08-10
---
# Price an inferred padding or grid-quantization tax with a block-count sweep before funding a round on it
- lever: Before funding a direction against padding or grid-quantization overhead on a grouped/aligned op, measure both: sweep the number of row-blocks and check latency for a staircase, and histogram valid rows per block from the alignment metadata.
- apply: Drive the sweep through the real launch path so the swizzle and dispatcher are included; a routing pass that pads only each group's LAST block to the tile leaves few or no fully-empty blocks, so an early-exit has nothing to skip and the residue is intra-block row padding that a smaller tile_m would have to pay for elsewhere.
- verify: A real quantization tax appears as a staircase in latency vs block count and as per-block cost that worsens near a wave boundary; a real padding tax appears as a nonzero count of fully-invalid blocks. If neither shows, the lag is elsewhere.
- pitfall: four consecutive round plans budgeted against a padding tax that did not exist -> both taxes were inferred from grid arithmetic and never measured -> spend one sweep plus one metadata histogram first, and read a linear latency curve with improving per-block cost as the refutation.
- caution: Also locate where the lagging case's wall actually goes with a loads-only replica using the kernel's exact addressing: here it ran ~73% of the real wall (71% memory), which moved the remaining work from padding to memory overlap, and a cache hint that paid on the zero-reuse operand cost 8-12% on the re-read ones.
- source: run kb_on_0810 2026-08-10
