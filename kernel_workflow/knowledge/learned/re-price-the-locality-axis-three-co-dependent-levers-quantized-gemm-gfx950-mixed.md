---
key: off-chip locality on a per-group block-scaled fp8 A8W8 GEMM, Triton, harness-frozen launch config, multi-XCD gfx950 — the axis re-opened after the in-loop dequant tax is already gone
type: lever
confidence: ★★
effect: 1.5145x marginal on the integrated tree (director-verified, non-overlapping; run-to-run spread 0.06%), taking the run 14.25x -> 22.03x isolated geomean vs the frozen baseline; per-case 22.66x / 21.64x / 21.81x over three growing-M cases, i.e. the trio pays uniformly — the smallest case (a tail/fill grid at ~1.3 workgroups per CU, ~3% of the weighted sum) gains as much as the largest
roofline: off-chip-locality-bound -> on-SIMD-issue-bound. L2 hit 55.5% -> 82.6%; HBM read amplification over the unique DRAM footprint 13.9x -> 2.72x, leaving reads at ~20% of nameplate BW (memory a closed axis); MFMA pipe 21.7% -> 34.7% and VALU issue 22.6% -> 36.2% at an unchanged 1.90 waves/SIMD against a 2-wave register ceiling; WAIT_ANY 46.1% -> 33.0% of wave cycles; combined pipe occupancy 44.4% -> 70.9%, ~33% of the fp8 matrix-core peak at the measured clock, with the instruction stream byte-unchanged
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: triton-on-rocm / gfx950
last_seen: 2026-08-12
verified_on: 2026-08-12
name: re-price-the-locality-axis-three-co-dependent-levers-quantized-gemm-gfx950-mixed
description: Bind the dead XCD program remap + in-body supergroup swizzle + drop the streaming hint on the most-reused operand: 1.51x marginal, each is noise alone
keywords: ['l2-locality', 'pid-remap', 'xcd', 'cache-modifier', 'l2-reorder', 'operand-reuse', 'group-size-m', 'fp8', 'block-scale', 'quantized-gemm', 'measurement-method', 'gfx950']
kernels: ['_gemm_a8w8_blockscale_kernel', 'gemm_a8w8_blockscale']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: mixed
layer: learned
levers: [mem.l2-locality, mem.cache-policy]
cost: L1
lifecycle: active
---
# Re-price the locality axis after the dominant in-loop term is gone — three co-dependent levers
- lever: With grid, block sizes and warp count frozen, the program-to-tile map and the per-operand cache policy are still open. Three levers — bind the XCD round-robin program remap, recompute the supergroup grouping in the body, and take the streaming/non-temporal hint OFF the largest most-reused operand — are strongly CO-DEPENDENT rather than additive: each reads as ~1% noise alone and they compound, so fund and measure them as ONE direction.
- apply: (a) use the XCD remap helper's return value — the program count was already a multiple of the XCD count, so no grid padding was needed; (b) when the harness pins the supergroup width to 1 the map degenerates into a row-major sweep, so recompute the grouping in the body and apply it AFTER the remap, so each XCD's contiguous program range covers whole supergroups; (c) promote the per-operand cache policy to constexpr parameters and sweep it per operand — the plain default on the large reused operand carried the round, while the streaming hint on it, and both write hints on the output store, each measured negative.
- stack: total 22.03x isolated (geomean vs frozen baseline, director-verified) = three directions compounded, incremental in landing order.
  - 1. bitcast the fp8 operands to the part's native OCP dialect, folding the resulting 4x into an existing scale — 9.19x standalone (round 1, verified), the big lever
  - 2. rank-1 row-scale collapse where the scale group width equals the N tile — 1.007x standalone but +1.56x on top of (1) (round 1, verified); a masked-target patch is only scoreable in combination
  - 3. this locality trio — +1.5145x on top of (1,2) (round 2, verified)
  - within (3), incremental on weighted time: remap bind +0.9%, + in-body supergroup swizzle +15.7% cumulative, + default cache policy on the reused operand +51.5% cumulative. Leave-one-out at the end point: removing the remap costs 5.7%, removing the swizzle 6.5% — each worth 6-10x more in the presence of the others than alone.
  - note: attribution is incremental in landing order, not independent.
- verify: Toggle each lever in the SAME binary and read L2 hit rate and HBM read bytes against the unique footprint — a real locality win appears at a byte-identical instruction / VALU / MFMA / wave count (register count and MFMA opcode were unchanged across the whole trio here). Confirm the remap's return value is actually consumed, then re-time the three TOGETHER against the frozen baseline rather than one at a time.
- pitfall: this exact axis measured 0.998x one round earlier and was filed as a dead end, then returned 1.5145x -> it had been priced against a profile taken before a dominant in-loop dequant term was removed, and its salvaged sub-findings were logged at 1.5-2% when one of them was worth ~15x that -> re-price every salvaged sub-finding on the CURRENT integrated tree before trusting an earlier noise verdict.
- pitfall: a program remap sat in the source, read as engaged, and had never been in effect -> the kernel discarded the helper's return value -> grep for the binding and toggle it in one binary; if the L2 counters do not move at an identical instruction count, it never fired.
- pitfall: an approximate epilogue rescale missed the parity bar by 28-36x at every chunk size -> the per-group scales are drawn i.i.d. across the K groups, so there is no smoothness in K for a coarser reference scale to exploit; the predicted relative error CV/sqrt(1+CV^2)*sqrt(1-1/C) matched the measured worst-element error at both ends of the chunk dial -> model the error from the scale distribution first, and keep any dequant hoist exact.
- pitfall: a floor reading came back ~15% high and read as a 5.5% regression -> the GPU-lock preflight did detect a co-tenant and returned non-zero, but piping it into grep swallowed the exit status and an `&&` chain then scored a SKIPPED correctness run as a pass -> check the preflight's exit status explicitly, take two readings, and discard the first timing after a cold device.
- caution: The supergroup width optimum is flat-bottomed and non-monotonic here, and it MOVED when the cache policy changed (powers of two were the worse choices under the streaming policy and among the better ones under the default), so also re-sweep the two neighbouring widths after any further structural merge.
- caution: This held with occupancy pinned at a 2-wave register ceiling and the fp32 accumulator resident in AGPRs; on a variant that is not accumulator-gated, also verify the cache policy operand by operand rather than globally, since the same hint helped on the reused operand and hurt on the streamed one and on the store.
- source: run kernel_20_geak_0811_2h_kb_new, 2 rounds / 6 directions, director-verified, 2026-08-12
