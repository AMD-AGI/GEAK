---
key: dequant-heavy packed-int4 weight GEMM on gfx950, Triton, where the same dequantised weight tile can serve several row-blocks
type: lever
confidence: ★★
effect: +24.4% at fusion width 2 and a further +23.8% widening 2 to 4, both non-overlapping vs frozen baseline; the gain lands on the large-M cases (M=32768/65536), the small-M case (M=2048) stays flat because it is memory-latency bound
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: share-the-dequantised-weight-tile-across-row-blocks-widen-m--moe-grouped-gemm-gfx950-prefill
description: Amortize int4 dequant by reusing one dequantised weight tile across several row-blocks; widen the dot COUNT along M, not the tile extent: ~+24% twice.
keywords: ['moe-grouped-gemm', 'w4a16', 'int4-dequant', 'fusion-width', 'lds-tiling', 'triton', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: prefill
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
---
# Share the dequantised weight tile across row-blocks, widen M first
- lever: reuse ONE dequantised weight tile across a group of row-blocks so the dequant cost is paid once per group; the dequant tile is shared across M and replicated across N, so widening along M buys more than widening along N at the same product.
- apply: widen the NUMBER of dots at a small row-tile instead of raising the row-tile extent, and concatenate the sub-blocks into one tile-major [BM, MM*BK] activation operand so the LDS round trip plus barrier pair is paid once per k-iteration.
- verify: isolated A/B per shape against the frozen baseline plus bit-exactness against the golden path, and check registers: the widened form held at 124 VGPR / 4 waves, beating a pre-declared register kill gate.
- pitfall: raising the row-tile extent instead of the dot count flipped warpsPerCTA and doubled the dequant work -> measured 0.64x -> keep the tile extent, widen the count.
a width knob that degrades to width 1 on the small shape lost -> degrade to the next-narrower FUSED form instead of unfusing.
- caution: also verify the activation-concat still pays at the widest group - at group width 8 the concat reversed sign and the unconcatenated form won, so re-measure the concat at each width.
- source: run kernel_20_geak_0808_16h, proposal fused_moe_int4_w4a16-own16h, 2026-08-12; rounds 3/5/18 verified geomeans 1.6379 -> 2.2597, director-accepted final 2.8019
