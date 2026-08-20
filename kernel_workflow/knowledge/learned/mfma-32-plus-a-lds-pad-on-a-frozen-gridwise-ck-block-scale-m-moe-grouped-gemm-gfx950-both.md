---
key: fp8 block-scale MoE second grouped GEMM on gfx950, Composable-Kernel 2-stage codegen behind an aiter JIT include path (gridwise frozen, only the instance .cuh is editable)
type: lever
confidence: ★★
effect: 1.25x cumulative isolated vs frozen baseline, bit-exact; per-case 1.18x on the smallest token case and 1.29x/1.30x on the two larger cases (non-overlapping)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both
description: Widen MFMA 16x16->32x32 on a CK block-scale MoE grouped GEMM (with matching host weight re-preshuffle) and pad A-LDS: ~1.25x, bit-exact.
keywords: ['mfma', 'lds-padding', 'bank-conflict', 'moe', 'block-scale', 'composable-kernel', 'fp8', 'preshuffle', 'grouped-gemm']
kernels: ['moe_stage2']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
roofline: compute-bound 0.24 -> 0.31 of the empirical roof
origin_kernels: ['moe_stage2']
---
# MFMA-32 plus A-LDS pad on a frozen-gridwise CK block-scale MoE GEMM
- lever: On the V1 (M64/N128/K128) block-scale instance, widen the XDL/MFMA shape from 16x16 to 32x32, then de-conflict A-side LDS reads with one extra row of A-block LDS padding.
- apply: Both live in the editable codegen instance header + its generator: flip the MFMA size and re-preshuffle the second-GEMM weights on the host to the layout the wider MFMA expects; add the A-block-LDS-extra-M=1 knob.
- stack: stack: total 1.25x isolated (bit-exact) = two directions compounded
  - 1. compute.mfma-32 + host weight re-preshuffle - 1.26x standalone (verified) - carries the win
  - 2. mem.lds-pad A-block extra-M 0->1 - +1.4% on top of (1) (verified) - only pays once the MFMA-32 read pattern exists
  - note: attribution is incremental in landing order; (2) was not measured alone.
- verify: Re-time every candidate against the baseline frozen at benchmark time, and confirm oracle parity is bit-exact; grep the built artifact for the config/marker delta to prove the instance actually changed.
- pitfall: Rounds that improved were reported as not-improved by the evaluator -> verdict scraping missed the marker -> grep-confirm the config delta and diff against the canonical instance before discarding a round.
pitfall: MFMA widened without re-preshuffling the host-side weights -> operand layout mismatch -> re-preshuffle to the pairing the wider MFMA expects.
- caution: Also verify the tile's M-per-block still equals the expert-sort granularity: a larger M-per-block let one tile span two expert groups and silently fetched the wrong expert's weights (error ratio ~0.7).
- source: run moe_stage2-ch16h, 2026-08-12, 16h budget, rounds R2 and R6
