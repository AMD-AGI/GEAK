---
name: specialise-tile-height-per-batch-bucket-when-packed-weight-d-moe-grouped-gemm-gfx950-compute-bound
description: Specialise tile height per batch bucket when packed-weight dequant reloads per tile: a 1.80x step took the verified MoE GEMM geomean from 2.23x to 4.00x
keywords: [tile-shape, dequant, moe, occupancy, vgpr, config-sweep, mfma, prefetch]
kernels: [fused_moe_kernel_gptq_awq]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: per-batch-bucket tile height / warp count / mfma width selection for a packed sub-byte weight fused-MoE grouped GEMM on gfx950, batch 2-64
lifecycle: active
type: lever
confidence: ★★
effect: Per-bucket tile height (64 at batch 2, 256 at batch 32, 512 at batch 64) took the verified geomean 2.2255x -> 4.0023x in one direction (a 1.7984x step, the largest inside the campaign; only the pre-campaign byte-once dual-nibble dequant step, 1.8154x, was larger); end state per-case 3.4287x at batch 2, 4.8761x at batch 32, 5.7367x at batch 64, geomean 4.5769x and a best pass of 5.1149x. Two smaller companion wins on the same axis: warp count 4->8 plus a 16-wide mfma instruction size measured 2.2255x, and gating the software prefetch off for the small bucket measured 1.1339x overall / 1.458x on that bucket alone. Roofline-emp went 0.100 -> 0.600.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.74h / 32 passes, 2026-08-11
last_seen: 2026-08-11
---
# Specialise tile height per batch bucket when packed-weight dequant is reloaded per tile
- lever: When each output tile re-unpacks and rescales the same packed sub-byte weights, the dequant cost per output element falls as 1/tile-height, so tile height is the lever the N and K tiles cannot substitute for - and its best value differs by an order of magnitude between the latency-bound small batch and the large batches.
- apply: Treat tile height, warp count and mfma instruction width as a per-bucket tuple selected at dispatch rather than one global config, and re-sweep the small bucket separately after the large ones move.
- verify: Check where the accumulator register floor puts occupancy for each chosen height before trusting a sweep row.
- caution: Also verify the small bucket under every knob elected at the large end: there the accumulator is small enough to sit at high occupancy, and bigger tiles, more warps and software prefetch each cost a few percent - gating prefetch OFF for that bucket alone was worth 1.458x on it.
- source: chuschen 16h time-budget campaign run, 15.74h / 32 passes, 2026-08-11
