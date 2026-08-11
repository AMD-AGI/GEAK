---
name: stop-widening-tiles-and-stages-once-the-accumulator-owns-the-quantized-gemm-gfx950-compute-bound
description: Tile/stage/occupancy closes as one axis under a big fp32 accumulator: a 256x64 tile fell to 0.64x, a wider MFMA non-k dim to 0.916x, num_stages=3 no build
keywords: [tile-shape, occupancy, vgpr, lds, mfma, pipeline-stages, compute-bound, fp8, config-sweep]
kernels: [_gemm_a8w8_blockscale_kernel, _w8a8_triton_block_scaled_mm]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: tile-shape, num_stages and occupancy escapes on an fp8 block-scaled GEMM on gfx950 whose fp32 128x128 accumulator sets the register floor and whose LDS is at the compile wall
lifecycle: active
type: anti-pattern
confidence: ★★
effect: four directions on the tile / stage / occupancy axis all closed on the graded compute-bound cases of an fp8 GEMM sitting at 88 VGPR/lane, per-SIMD occupancy ~5 waves and ~2 workgroups/CU - a 256x64 tile regressed to 0.64x; widening the MFMA non-k dim from 16 to 32 regressed to 0.916x (16x16x128 is already the widest-K instruction); num_stages=3 and a deeper k-group both overflow the 160KB/CU LDS and fail to compile against a hard unroll*num_stages<=5, and num_stages=3 fits only at a halved N tile, which halves fp8 arithmetic intensity; reaching a 6-wave per-SIMD occupancy needs <=85 VGPR/lane against an fp32 128x128 accumulator that is ~32 VGPR/lane on its own over 8 warps; a hand-scheduled register-resident rewrite that fed MFMA from global loads to bypass the LDS read chain was also attempted and banked no win
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign, 2026-08-11
last_seen: 2026-08-11
---
# Stop widening tiles and stages once the accumulator owns the register budget and LDS is at the compile wall
- lever: When the residual on an MFMA loop is an operand-feed stall (LDS read gated by an lgkmcnt wait, serial with the MFMA that consumes it), the current-iteration read cannot be prefetched, so the usual escape is more independent waves - which is exactly what a large fp32 accumulator rules out. Price that escape before funding it.
- apply: Divide the accumulator's own per-lane register cost into the occupancy step you would need, and check whether the stage or tile change that would hide the latency still fits the LDS budget under the unroll*num_stages product the compiler enforces; if neither the register nor the LDS side has room, the tile-shape and stage axes are closed together.
- pitfall: enlarging the tile and the stage count was expected to plateau at worst -> both push against a register floor and a hard LDS/unroll product, and the one stage setting that fits does so only at a halved N tile -> expect regressions (0.64x, 0.916x) rather than a flat result, and read the compile failure as the budget speaking.
- verify: Read VGPR/lane and the emitted MFMA opcode from the code object, and confirm the achieved per-SIMD wave count rather than the requested one.
- caution: Also verify the widest-K instruction for your dtype first - widening the non-k dim here lost precisely because 16x16x128 was already the widest-K form available.
- source: chuschen 16h time-budget campaign, 2026-08-11
