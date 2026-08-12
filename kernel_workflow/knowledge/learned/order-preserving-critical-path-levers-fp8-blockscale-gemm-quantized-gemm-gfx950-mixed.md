---
key: Triton fp8 block-scaled W8A8 GEMM upcast+MFMA path on gfx950/CDNA4, under a near-bit-exact parity gate
type: lever
confidence: ★★
effect: 10.66x isolated vs the frozen baseline, bit-exact (max_rel=0); per-case 9.81x / 11.15x / 11.08x over three M scales (smallest M gains least); roofline-emp 0.03 -> 0.34, latency-bound -> compute-bound. Reproduced independently on the same key by the hand-written-SWAR variant of the same lever: 11.60x geomean isolated, bit-identical (max_abs_diff 0), per-case 9.59x smallest-M / 12.42x / 13.11x, matrix-core share of SIMD cycles ~0.13 -> ~0.23 of the fp16 matrix-core peak fraction and VALU:MFMA 306:1 -> 24:1.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 10
toolchain: unknown
last_seen: 2026-08-12
verified_on: 2026-08-12
name: order-preserving-critical-path-levers-fp8-blockscale-gemm-quantized-gemm-gfx950-mixed
description: fp8 block-scaled GEMM, gfx950: fold the scale fixup into a packed upcast (native cvt or 1-instr SWAR), re-tile on accum/lane, chain sub-K dots - 10.7-11.6x
keywords: ['fp8', 'block-scale', 'quantized-gemm', 'dep-chain', 'sub-k-coarsening', 'scaled-cvt', 'swar', 'inline-asm', 'num-warps', 'bit-exact', 'gfx950', 'critical-path']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: mixed
layer: learned
lifecycle: active
---
# order-preserving-critical-path-levers-fp8-blockscale-gemm
- lever: On a dequant-then-MFMA fp8 path the payoff is DEP-CHAIN LENGTH, not op count: derive magnitude and sign from ONE packed upcast, regroup the inner K sub-tiles into fewer wider dots on one linear fp32 accumulator, and re-tile so fp32 accumulators per lane (tile_area / (64 * num_warps)) is the swept quantity.
- apply: Fold the scale-format fixup into the fp32 scale operand as an exact power of two, feed int32-packed 4-values-per-dword loads, and lift sign from bit7 rather than a second cvt. Two verified upcast routes: the native packed scaled-cvt, or a hand-written branchless SWAR that collapses to ONE add-shift per packed dword (add the sign bit to itself so the carry lands on the wide format's sign bit) - brute-force it over all 256 codes. Chain sub-K with dot(a,b,acc) (the acc operand continues the same MFMA chain, so it is bit-identical), and unpin the launch through an exported launcher OBJECT with an env-selectable config so the tile/num_warps sweep needs no source edit.
- stack: total 11.60x isolated (geomean, director-verified, bit-identical) = three directions compounded on a 7.25x first round (packed upcast + max_contiguous/multiple_of address hints, worth 1.03x alone but +10.2% stacked)
  - 1. launcher re-tile to a square tile at HALVED num_warps - to 8.95x (round 2, verified), best individual
  - 2. upcast collapsed to one add-shift - +13.4% on top of (1) (round 2, verified; 8.67x standalone)
  - 3. intra-kernel K-slab split at 4 sub-slabs - +14.3% on top of (1,2) (round 2, verified; 7.92x standalone, the round's WEAKEST standalone and its LARGEST stacked contributor)
  - note: attribution is incremental in landing order, not independent. The earlier campaign on this same key reached 10.66x by the native-cvt route, where sub-K coarsening was the largest single step.
- verify: Re-time on the official runner rather than the authoring loop, take a median of >=3 runs (a single 4-6% high outlier appears while the other runs agree within 0.2%), report max_rel rather than cosine (cosine read 1.00000000 over failing outputs), and read VGPR/LDS/MFMA counts off the compile metadata before timing an arm.
- pitfall: a stacked pair applied cleanly, compiled, ran and returned garbage (max_rel ~6e5) -> an inline-asm output constraint that reads its own result register is only register-legal at the launch config it was authored under, and a re-tile changes that -> give the mask a dedicated early-clobber scratch output, and re-run parity on every COMBINATION, not just each patch alone.
- pitfall: growing the tile at the harness-pinned num_warps regressed -> the tile only pays once fp32 accumulators per lane reaches ~128, and doubling num_warps halves them -> sweep tile and num_warps JOINTLY on the accumulators-per-lane invariant.
- pitfall: coarsening past the sweet spot spills VGPRs and regresses, and the earlier-measured optimum flipped after the packed-upcast rewrite landed -> a knob optimum is conditional on the surrounding structure -> re-sweep it after any structural change instead of trusting the recorded value.
- caution: The occupancy drop this stack causes is not a regression signal here; also verify whether the op is issue/dep-chain-bound before reading occupancy as the score, and re-check the shared-memory figure against a residency claim (it was invariant to the sub-slab split, so that premise was false while the win was real).
- caution: Also verify that the native low-precision matrix-core path clears the parity gate before funding it: here the reinterpretation was value-wrong at the saturating code, and even a value-exact operand map changed the fp32 accumulation order and failed.
- source: 16h single-kernel time-budget campaign, chuschen16h wave, 31 STATE-resumed passes / 4 rounds, 2026-08-11
- source: run kernel_20_geak_0811_2h_kb_new, 2 rounds / 6 directions, 2026-08-12
