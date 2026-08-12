---
key: fp8 block-scaled Triton GEMM on gfx950, gated on bit-exact fp32 MFMA accumulation order, dequant-emulated on the VALU
type: lever
confidence: ★★
effect: 1.16x on top of the previous round's seed (cumulative 9.17x -> 10.66x vs the frozen baseline, non-overlapping); per-case 9.81x / 11.15x / 11.08x on the three shapes; bit-exact, max_rel=0
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: sub-k-coarsening-regroup-the-same-reduction-order-into-fewer-quantized-gemm-gfx950-compute-bound
description: Under a bit-exact accumulation gate, coarsen the inner K sub-tile (fewer, wider dots, one linear fp32 accumulator) — pipeline-balance win, parity untouched.
keywords: ['fp8', 'block-scaled-gemm', 'bit-exact-gate', 'critical-path', 'sub-k-coarsening', 'mfma', 'gfx950', 'num-stages']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: archived
cost: L1
verified_on: 2026-08-12
---
# Sub-K coarsening: regroup the same reduction order into fewer wider dots
- lever: Sweep the inner sub-K split (SUB_K a multiple of the MFMA K, so only a few values are legal) while keeping ONE linear fp32 accumulator: it regroups the identical summation order, so a bit-exact parity gate stays satisfied.
- apply: One-line config change to the sub-K factor; the coarser split halves per-K-block convert / repack / transpose invocations and lets the compiler overlap load+convert+LDS against the MFMA chain.
- verify: Confirm max_rel=0 against golden, then re-time on the frozen isolated A/B; check VGPR count and occupancy in the AMDGCN ISA to see the coarser tile did not spill.
- pitfall: Too-coarse a split (below the legal midpoint) spills VGPRs and regresses -> register pressure, not issue rate, is the limit -> keep the measured optimum and re-sweep after any convert-path rewrite, since the optimum moved by one step once the packed-convert path landed.
- caution: The win is critical-path balance, not op count, so also verify it holds when the auto-pipeliner stage count changes; on a shape with a different K it can shift.
- source: run _w8a8_triton_block_scaled_mm-ch16h, 2026-08-12, 16h per-kernel budget, 31 passes
