---
key: Triton fp8 block-scaled W8A8 GEMM upcast+MFMA path on gfx950/CDNA4, under a near-bit-exact parity gate
type: lever
confidence: ★★
effect: 10.66x isolated vs the frozen baseline, bit-exact (max_rel=0); per-case 9.81x / 11.15x / 11.08x over three M scales (smallest M gains least); roofline-emp 0.03 -> 0.34, latency-bound -> compute-bound
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-11
name: order-preserving-critical-path-levers-fp8-blockscale-gemm-quantized-gemm-gfx950-mixed
description: fp8 block-scaled GEMM, gfx950: fold the scale-format fixup into the native scaled-cvt operand, pack 4 fp8 per dword, coarsen sub-K - 10.66x bit-exact
keywords: ['fp8', 'block-scale', 'quantized-gemm', 'dep-chain', 'sub-k-coarsening', 'scaled-cvt', 'lds-tiling', 'bit-exact', 'gfx950', 'critical-path']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: mixed
layer: learned
lifecycle: active
---
# order-preserving-critical-path-levers-fp8-blockscale-gemm
- lever: On a dequant-then-MFMA fp8 path the payoff is DEP-CHAIN LENGTH, not op count: derive magnitude and sign from ONE native packed scaled-cvt, and regroup the inner K sub-tiles into fewer wider dots while keeping one linear fp32 accumulator.
- apply: Fold the fnuz->OCP 2^-1 fixup into the fp32 scale operand of the native scaled-cvt; feed it int32-packed 4-fp8-per-dword loads (cast the pointer, pack=1, two f16x2 outputs); lift sign from each fp8 byte's bit7 instead of a second cvt + abs; load B transposed [BLOCK_N,SUB_K] and transpose back so it rides LDS; sweep NSUB over the values that keep SUB_K a multiple of the MFMA K (here {1,2,4,8}).
- stack: total 10.66x isolated (bit-exact, task-runner re-verified) = four order-preserving directions compounded
  - 1. drop redundant inner sub-K masks when K is an exact multiple of BLOCK_K - +2.7% (round 1, verified)
  - 2. scale-operand fold + int32-packed fp8 loads + transposed-B/LDS path - to 8.98x (verified); the LDS B-path alone carried ~+15% on the two larger cases
  - 3. single-cvt signed upcast (cvts 192->96, v_and 153->9) - +1.5-1.8% on top of (2) (verified)
  - 4. sub-K coarsening NSUB 8->4 - 9.17x -> 10.66x, the largest single step (verified)
  - note: attribution is incremental in landing order; (3) is small because the win is pipeline balance, and the 96-cvt cut alone measured ~0.
- verify: Re-time on the official runner rather than the authoring loop (self-reports drifted; ~1% run-to-run there), confirm parity reports max_rel=0, and read VGPR/occupancy off the static AMDGCN of a PERF-only build - the golden's emulated upcast is a phantom target.
- pitfall: pitfall: coarsening past the sweet spot (NSUB<=2) spills VGPRs and regresses -> wider dots grow the live set -> keep the widest sub-K that holds occupancy 3 at zero spill.
pitfall: the earlier-measured optimum NSUB flipped after the packed-cvt/LDS-B rewrite landed -> a knob optimum is conditional on the surrounding structure -> re-sweep the knob after any structural change instead of trusting the recorded value.
- caution: The occupancy drop this stack causes (4->3) is not a regression signal here; also verify on your shapes whether the op is issue/dep-chain-bound before reading occupancy as the score.
- source: 16h single-kernel time-budget campaign, chuschen16h wave, 31 STATE-resumed passes / 4 rounds, 2026-08-11
