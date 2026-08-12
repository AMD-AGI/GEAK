---
key: per-1x128 block-scaled fp8 (A8W8) GEMM on gfx950/MI355X in Triton, seed emulating the fp8 upcast in VALU
type: lever
confidence: ★★
effect: 24.08x weighted cumulative vs frozen baseline; per-case c2 24.29x, c32 23.76x, c64 23.88x (all three within 2%); roofline-emp 0.05 latency-bound -> 0.55 compute-bound
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 7
toolchain: unknown
last_seen: 2026-08-12
name: fp8-hardware-cvt-upcast-is-the-wall-under-a-block-scaled-fp8-quantized-gemm-gfx950-compute-bound
description: Block-scaled fp8 GEMM pinned near the latency-bound floor: hardware fnuz->OCP cvt upcast plus fused split-K reduce lifts it to compute-bound
keywords: ['fp8', 'block-scale', 'dequant', 'split-k', 'upcast', 'triton', 'compute-bound', 'quantized-gemm']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
verified_on: 2026-08-11
---
# fp8 hardware-cvt upcast is the wall under a block-scaled fp8 GEMM, not occupancy
- lever: When a block-scaled fp8 GEMM sits at a few percent of achievable peak, suspect an emulated fnuz->OCP fp8 upcast in the VALU before suspecting occupancy: swap it for the hardware cvt and fold the constant 0.25 rescale into the accumulator (exact for finite positive-normal fp8).
- apply: Rewrite the operand-upcast in the K-loop to the hw convert intrinsic; for tiny-M cases add NUM_KSPLIT=2 to fill the grid (roughly doubles resident CTAs and halves per-CTA K iterations, which re-tunes num_warps/num_stages downward) and fuse the bf16 partial reduction into a companion reduce kernel.
- stack: total 24.08x weighted = three directions compounded, attribution incremental in landing order
  1. fp8 hw-cvt upcast — 12.95x standalone (pass 1, verified) — carries the bulk
  2. small-case split-K + fused reduce — +21.7% on top of (1) (pass 2, verified)
  3. rank-1 scale collapse + 2-deep overlap unroll — 1.5263x on top of (1,2) (pass 3, verified)
- verify: Grep the generated ISA for the cvt opcode replacing the emulation sequence, then re-time every case against the frozen baseline; the bound class should move off latency-bound and the small-batch case should stop being grid-starved.
- pitfall: split-K win vanished -> the partial reduction was a 3-kernel torch.sum epilogue whose launches and HBM round-trips exceeded the split-K gain -> fuse the reduction into one bf16 reduce kernel; also KSPLIT=3/4 lose to 2 on the tiny case.
- caution: The 0.25 fold is exact only for finite positive-normal fp8 inputs, so also verify bit-exactness against the oracle on denormal and saturating operands before banking the win.
- source: 16h per-kernel time-budget campaign (chuschen16h lane), 2026-08-11, 49 passes, Director-validated geomean
