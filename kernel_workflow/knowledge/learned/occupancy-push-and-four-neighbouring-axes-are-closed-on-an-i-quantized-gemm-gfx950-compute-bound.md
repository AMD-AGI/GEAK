---
key: a block-scaled fp8 GEMM on gfx950 already at 2 waves/SIMD with no scratch spill, where the critical chain is MFMA -> scale -> accumulate
type: anti-pattern
confidence: ★★
effect: 5 of 7 directions in the campaign returned 0% over the 24.08x seed: doubling warps reaches 5 waves/SIMD but regresses c32 and c64 by ~4% each; capping registers to force a third wave spills and regresses c32 ~14% / c64 ~14%; shrinking the N tile to remove a half-wasted tile column regresses the tiny case ~11%; hand-scheduled ping-pong and host-side graph capture/replay are byte-identical / slightly worse on all three cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 7
toolchain: unknown
last_seen: 2026-08-12
name: occupancy-push-and-four-neighbouring-axes-are-closed-on-an-i-quantized-gemm-gfx950-compute-bound
description: At 2 waves/SIMD an ILP-bound block-scaled fp8 GEMM loses from any occupancy push; grid/tail, hand-scheduling, graph capture, microtune all zero
keywords: ['occupancy', 'vgpr', 'ilp', 'spill', 'hip-graph', 'grid-tail', 'anti-pattern', 'quantized-gemm']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
verified_on: 2026-08-11
---
# Occupancy push and four neighbouring axes are closed on an ILP-bound dequant GEMM
- lever: Before spending a round raising occupancy on a dequant GEMM, read the ISA: if the seed is already at 8 warps with no scratch spill and the accumulator plus per-K partial are algorithm-fixed in fp32, the 2-wave floor is likely the optimum ILP-for-registers trade and the profitable axes are elsewhere (dequant math, split-K on the tiny case).
- apply: Grep the compiled .amdgcn for the VGPR count, occupancy annotation and any scratch usage; treat a natural ~184-VGPR/2-wave pick with zero spill as evidence the compiler already made the trade, and budget the round for the dequant chain instead.
- verify: Re-time both directions against the frozen baseline rather than trusting the occupancy number: a config that reaches a higher wave count and still loses is the signal that the op is latency/ILP-bound on the accumulate chain.
- pitfall: A round reported IMPROVED=false and looked like an apply failure -> it was a genuine true negative, the workspace md5 matched the seed and no winning diff was produced -> confirm the patch marker and the config delta in the verify workspace, and compare against baseline, before recording either a win or a harness bug.
- caution: This closure was measured with the fp32 accumulator and arbitrary per-1x128 fp32 scales that rule out native scaled-MFMA; if the harness pins a tile shape that admits an E8M0-style scaled MFMA, also re-open the occupancy and tile axes before assuming they stay closed.
- source: 16h per-kernel time-budget campaign (chuschen16h lane), 2026-08-11, five directions closed as dead ends
