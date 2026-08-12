---
key: a block-scaled fp8 (a8w8) Triton GEMM on gfx950/MI355X already sitting at a 2-wave/high-VGPR ILP optimum, where the remaining directions are closed
type: anti-pattern
confidence: ★★
effect: ~1.00x on all five directions, measured not assumed: doubling warps reached a 2.5x-higher occupancy class yet came out ~4% slower on both large per-case shapes; a warps-per-eu hint spilled and cost ~14-15% there; a narrower N tile cost ~11% on the tiny case; hand-scheduled ping-pong and host-side graph capture were byte-identical or slightly worse
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound
description: Above a tuned block-scaled fp8 GEMM on gfx950 five axes returned ~1.00x: occupancy raise, Gluon/HIP ping-pong, host graph capture, body microtune, tile shrink
keywords: ['anti-pattern', 'occupancy', 'ilp', 'quantized-gemm', 'fp8', 'block-scale', 'launch-overhead', 'tile-size', 'gfx950']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
---
# Five closed axes above an ILP-bound block-scaled fp8 GEMM
- lever: Before spending a round on occupancy or host-side capture for a quantized GEMM whose profile is compute-bound, test whether the compiler's natural high-VGPR / 2-wave pick is already the register-for-ILP optimum: here reaching a higher occupancy class needed either more warps (which halves per-warp ILP and breaks the second-MFMA/dequant overlap) or a forced register cap (which spills), and both net-lose.
- apply: Read the generated ISA for VGPR count, occupancy class and scratch spill first; that is a minutes-long check that decides whether the occupancy axis is open at all.
- verify: Re-time any occupancy change per case rather than trusting the occupancy number, and confirm no scratch spill appeared; for host-side capture, first check whether a GPU-idle bubble exists at all — the tiny case was grid/compute-bound, not launch-bound.
- pitfall: A microtune sweep and a hand-scheduled body both came back with the source byte-identical to the seed — the honest reading is a true negative, confirmed by md5-matching the workspace against the banked best, not a broken harness.
- caution: Also verify these axes on your own scale granularity: they closed here for arbitrary fp32 per-1x128 scales, which forbid native scaled-MFMA; an E8M0 per-32 layout with a pinned tile reopens the compute axis, and the arithmetic here already sits at ~59% of the native-fp8 MFMA partition peak and ~91% of the dequant-limited floor, with HBM traffic near 6% of achievable bandwidth.
- source: run _gemm_a8w8_blockscale_kernel-ch16h, 2026-08-12 — 16h time-scaling campaign, five dead-end directions (occupancy, Gluon/HIP, host runtime, microtune, grid/tail)
