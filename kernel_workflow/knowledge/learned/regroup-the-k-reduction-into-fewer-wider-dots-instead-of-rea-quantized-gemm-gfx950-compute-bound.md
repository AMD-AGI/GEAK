---
name: regroup-the-k-reduction-into-fewer-wider-dots-instead-of-rea-quantized-gemm-gfx950-compute-bound
description: Regroup the k reduction into fewer wider dots instead of reassociating it: cumulative geomean 9.17x -> 10.66x at max_rel=0 on all three cases
keywords: [mfma, quantized-gemm, vgpr, occupancy, tile-shape, compute-bound, operand-reuse, correctness-gate, config-sweep]
kernels: [_w8a8_triton_block_scaled_mm]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: order-preserving regrouping of the inner k reduction in a Triton block-scaled quantized GEMM on gfx950, under a gate that forecloses split-k and atomics
lifecycle: active
type: lever
confidence: ★★
effect: one-line change of the inner sub-k factor (8 sub-dots of 16 -> 4 sub-dots of 32, single linear fp32 accumulator kept) took the cumulative geomean 9.1678x -> 10.6607x, per-case 9.81x / 11.1466x / 11.0802x on the small / mid / large case, at max_rel=0 against the golden; VGPR 148 -> 154 with occupancy unchanged at 3. It halves the per-k-block dequant-convert and LDS-transpose invocations without changing summation order, so it passes a gate (cos>=0.99, max_rel<1e-2, denominator clamp 1e-6) under which every reassociating variant of the same reduction failed: 2-way contiguous-half split max_rel 0.1146 at cos=1.0, even/odd split identical, Kahan ~100% relative error.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.73h / 31 passes, 2026-08-11
last_seen: 2026-08-11
---
# Regroup the k reduction into fewer wider dots instead of reassociating it
- lever: when a tight correctness gate forecloses split-k, atomics and any other reordering of the k reduction, one order-preserving axis is left: keep the single linear accumulator and regroup the same k order into fewer, wider dependent dots. The payoff is critical-path balance, not op count -- wider dots give the backend more room to overlap operand load, dequant-convert and LDS traffic against the matrix-core chain -- so it can pay even when the instruction census barely moves.
- apply: sweep the sub-k factor over the values that keep it a multiple of the matrix instruction k (32x32x16 here), on the current body rather than a remembered one, and read compiler register/occupancy alongside time; coarsening past the register budget reverses the sign (two sub-dots spilled and regressed).
- verify: confirm max_rel is exactly 0 (the regrouping is meant to be order-preserving, so anything nonzero means the accumulator was also changed), and check VGPR count and occupancy did not step down.
- caution: also re-measure this axis after any change to the operand-conversion structure - the optimum here moved from 8 sub-dots to 4 once packed-dword loads and the LDS B-path landed, so a previously measured optimum from an earlier body is stale evidence rather than a settled value.
- source: chuschen 16h time-budget campaign run, 15.73h / 31 passes, 2026-08-11
