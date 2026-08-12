---
key: streaming fp8 quantize/cast on gfx950 already running near the practical HBM ceiling, deciding whether another memory round is worth spending
type: anti-pattern
confidence: ★★
effect: ~1.00x (inside run-to-run noise) on both large token-count cases across six directions - wider num_warps, larger and smaller tiles, store cache modifiers, load hints, num_stages, flat 1D cross-row tiling, manual store repack; none cleared the >2%-on-both-cases gate at ~62-63% of nameplate HBM bandwidth
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound
description: Above ~60% of nameplate HBM, six bandwidth directions all returned ~1.00x on an fp8 quant cast; the store already lowered to one 128-bit instruction
keywords: ['memory-bound', 'quantize-cast', 'fp8', 'closed-axis', 'cache-modifier', 'num-warps', 'tiling', 'store-vectorization', 'assembly-inspection']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: memory-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-07-30
roofline: ~62-63% of nameplate HBM bandwidth, i.e. at the practical ceiling for a 3-pass traffic mix (bf16 read + fp8 write + fp32 scale)
---
# Near the practical HBM ceiling the bandwidth knobs are a closed axis
- lever: compute fraction-of-nameplate bandwidth from the traffic the op has to move before planning a memory round; once it sits around 60%+, the remaining gap is read/write turnaround plus the fixed traffic mix, so a knob sweep is a low-yield place to spend a round and the budget is better aimed at the launch/host side.
- apply: if a store-side idea survives that check, settle it from the ISA dump before writing a candidate: read the generated store width, VGPR/SGPR counts, occupancy and spill count.
- verify: gate any memory candidate on beating the frozen baseline by a margin on EVERY large case, not on a geomean - four of these tied on one case and lost on the other.
- pitfall: store vectorization looked wide open → the ISA showed the quantized store already lowering to one 128-bit buffer_store_dwordx4 because the output tile is contiguous and gets max-width coalescing → a manual uint32/uint4 repack cannot exceed that width and only adds shift/or work, so it can at best tie.
- caution: also verify the ceiling claim on your own shapes when counters are unavailable - here bandwidth was derived analytically because the profiler counters were uncollectible, and the derivation was only cross-checked against one earlier measurement.
- source: run _per_token_group_quant_fp8-ch16h, 16h per-kernel time-budget campaign, 2026-07-30
