---
name: a-frozen-launch-config-still-leaves-the-program-to-tile-map--moe-grouped-gemm-gfx950-mixed
description: With grid and block sizes frozen, fuse G tiles per program along the shared-operand-invariant axis: +107% on top of an already-fused MoE grouped GEMM
keywords: [operand-reuse, tile-shape, mfma, occupancy, vgpr, moe, dequant]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: mixed
key: work-per-program fusion in a Triton fused-MoE grouped GEMM whose grid/block/num_warps/num_stages the harness has frozen, gfx950, small- and large-batch cases
lifecycle: archived
type: lever
confidence: ★★
effect: two fusion steps took the verified geomean 17.2x -> 46.5x (+30%, then +107%); per-case at that point the two large-batch cases sat at 57-60x while the small-batch case, already at its DRAM compulsory floor, reached only ~28x
confirms_cited: 0
confirms_blind: 1
losses: 1
attempts: 8
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
---
# A frozen launch config still leaves the program-to-tile map open
- lever: When the harness freezes grid, block sizes, num_warps and num_stages, work-per-program is still free: hand the first 1/G of the programs G tiles each along an axis the shared operand does not depend on and let the surplus programs retire at the top. The produced tile set is bit-identical, and the shared operand is fetched once per G tiles instead of once per tile.
- apply: choose the fusion axis by which operand is invariant along it, then collapse the G tiles into ONE wide dot (BM = tile_M * G) and merge their index, scale and mask plumbing into a single set rather than keeping G separate dots - the merged form is what lets the compiler pick the wider MFMA shape and what keeps the register cost affordable.
- verify: measure L1-from-L2 bytes, and bytes per MFMA per output tile, read .vgpr_count and shared out of the compiled artifact before benchmarking, and confirm the wider MFMA opcode appears.
- pitfall: a fused variant was rejected on a predicted register count, and a wait-percentage counter went the wrong way while the kernel got 30% faster -> each surviving wave now carries G times the work, so resource and stall models mis-score the merged form -> re-measure wall clock once the plumbing is merged; the same fusion factor measured -14% with per-tile dots and +107% with one merged dot.
- caution: this buys reuse by spending occupancy (registers roughly doubled, waves per SIMD halved), so also verify wall clock rather than a resource model.
- source: run kernel_20_geak_0808_4h 2026-08-08
