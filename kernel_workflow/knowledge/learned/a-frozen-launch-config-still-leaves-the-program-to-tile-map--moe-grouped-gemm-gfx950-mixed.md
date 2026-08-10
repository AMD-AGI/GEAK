---
key: moe grouped gemm · gfx950 · mixed
type: lever
confidence: ★★
effect: two fusion steps took the verified geomean 17.2x -> 46.5x (+30%, then +107%); per-case at that point the two large-batch cases sat at 57-60x while the small-batch case, already at its DRAM compulsory floor, reached only ~28x
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-08
---
# A frozen launch config still leaves the program-to-tile map open
- lever: When the harness freezes grid, block sizes, num_warps and num_stages, work-per-program is still free: hand the first 1/G of the programs G tiles each along an axis the shared operand does not depend on and let the surplus programs retire at the top. The produced tile set is bit-identical, and the shared operand is fetched once per G tiles instead of once per tile.
- apply: choose the fusion axis by which operand is invariant along it, then collapse the G tiles into ONE wide dot (BM = tile_M * G) and merge their index, scale and mask plumbing into a single set rather than keeping G separate dots — the merged form is what lets the compiler pick the wider MFMA shape and what keeps the register cost affordable.
- verify: measure L1-from-L2 bytes and bytes per MFMA per output tile, read .vgpr_count and shared out of the compiled artifact before benchmarking, and confirm the wider MFMA opcode appears; wait-percentage counters are not an objective function here — one went the wrong way while the kernel got 30% faster, because each surviving wave now carries G times the work.
- caution: this buys reuse by spending occupancy (registers roughly doubled, waves per SIMD halved), so also verify wall clock rather than a resource model; and when a fused variant was rejected on a predicted register count rather than a measured clock, re-measure it once the plumbing is merged — the same fusion factor measured -14% with per-tile dots and +107% with one merged dot.
- source: run kernel_20_geak_0808_4h 2026-08-08
