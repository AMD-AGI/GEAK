---
name: disassemble-the-store-before-funding-a-vectorization-round-o-quantize-cast-gfx950-memory-bound
description: Read the ISA for the widest store before a vectorization round on a streaming fp8 quantize: two directions produced no patch against 2.55x/2.40x budgets
keywords: [isa-check, quant, fp8, memory-bound, roofline, occupancy, vgpr, tile-shape, measurement-method]
kernels: [_per_token_group_quant_fp8]
platforms: [gfx950]
kernel_class: quantize_cast
regime: memory-bound
key: store-vectorization and tiling rounds on a streaming fp8 quantize/cast kernel on gfx950 whose 32x128 output tile is already contiguous
lifecycle: active
type: anti-pattern
confidence: ★★
effect: two separate directions on this axis returned no patch at all (verdict dead_end, actual 0.0, expected 2.55 and 2.40) - the assembly showed the fp8 store already lowering to a single buffer_store_dwordx4 (128-bit, 16 fp8 bytes per lane) because the 32x128 output tile is contiguous, with the load side at 2x buffer_load_dwordx4 and the only scalar store being the unavoidable fp32 scale write, at VGPR=27, SGPR=23, occupancy=8 (max on this part) and zero spills, so a manual uint32/uint4 repack can only add pack ops at that width and failed the >2%-on-both-large-cases gate; in the same sweep num_warps=8 was worse, both smaller and larger tile widths were worse, alternate store cache modifiers tied or lost, load hints were no-ops, num_stages was inert, and a flat 1D cross-row tiling was bit-exact but slower at every tile width, with the ending state at ~59-62% of nameplate HBM bandwidth
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.70h / 56 passes, 2026-08-11
last_seen: 2026-08-11
---
# Disassemble the store before funding a vectorization round on a tile that is already contiguous
- lever: Once a streaming quantize/cast is within ~1.6x of nameplate HBM, the remaining gap is usually read/write turnaround plus the op's fixed traffic ratio (here 2 bytes read + 1 byte written + an fp32 scale per element), not store efficiency, and no tiling or cache-hint knob moves it - so dump the ISA first and look for the widest store the part offers.
- apply: If the store is already at the widest instruction and occupancy is at max with no spills, treat the bandwidth axis as closed and spend the round elsewhere; deriving achieved bandwidth analytically from the traffic model is enough to make the call when hardware counters are uncollectible.
- pitfall: a manual uint32/uint4 repack was budgeted as a vectorization win -> the compiler was already emitting the widest store because the output tile is contiguous -> the repack could only add pack ops and produced no patch that cleared the >2%-on-both-large-cases gate.
- verify: Confirm the emitted store/load widths and the spill count in the code object on the same case the bandwidth estimate came from, and score against the large cases the gate scores.
- caution: Also verify the tile really is contiguous for your shape - the same repack is worth trying where the output rows are strided and the compiler has to narrow the store.
- source: chuschen 16h time-budget campaign run, 15.70h / 56 passes, 2026-08-11
