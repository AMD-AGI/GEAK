---
key: which axes are already closed on a frozen CK/aiter fp8 block-scaled MoE grouped GEMM at gfx950 whose MFMA path is already native
type: anti-pattern
confidence: ★★
effect: 7 of 11 issued directions returned <=1.01x on all three batch cases (2/32/64): HIP-graph replay -3 to -4%, alternate and compiler-default hot-loop schedules -0.4 to -1.4%, LDS-pad and epilogue-cluster rewrites 1.000x, occupancy co-cut -18%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 11
toolchain: unknown
last_seen: 2026-08-12
name: where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed
description: Frozen vendor fp8 block-scaled MoE GEMM at ~0.37 of roof: LDS padding, epilogue rewrite, vector width, HIP-graph and fp4 weights all returned ~1.00x
keywords: ['moe', 'grouped-gemm', 'fp8-blockscale', 'composable-kernel', 'anti-pattern', 'closed-axis', 'hip-graph', 'lds-padding', 'occupancy', 'compute-bound']
kernels: ['moe_gemm_fp8_blockscale']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-07-30
roofline: stayed compute-bound at ~0.37 of the empirical roof through every direction listed here
---
# Where a native-MFMA block-scaled MoE GEMM has no headroom left
- lever: When the disassembly shows the native f8f6f4 MFMA is already being emitted and the profile says the stall is dependency wait rather than a schedulable gap, the remaining headroom is in the per-128-K fp32 rescale that co-issues with the MFMA — an axis the frozen pipeline does not expose — so the classic memory/epilogue/graph axes are worth at most one cheap probe each.
- apply: Before budgeting rounds, disassemble to confirm the MFMA is native (an fnuz/ocp naming difference is cosmetic) and count the independent accumulators: with only two per K-scale block, low ILP is intrinsic and re-scheduling cannot recover it.
- verify: Treat a direction as closed only after re-timing on the frozen baseline across all batch cases; several of these read as small wins on one case and regressions on another before the frozen re-time.
- pitfall: Storing the expert weights as fp4 promised a large win but failed the numeric gate at 14-19 dB SNR against a 25 dB threshold → post-MFMA per-block rescale leaves no headroom for a 4-bit mantissa here → a sibling MoE kernel's fp4-weight win did transfer numerically only because it was cosine-gated, not SNR-gated.
- caution: Also verify the scheduler axis is even reachable before spending a round: the fused pipeline header here specializes exactly one scheduler enum and leaves the primary template body empty, so flipping the enum fails the build rather than producing a slower kernel, and the generator exposes no hook for it.
- source: run moe_gemm_fp8_blockscale-ch16h, 2026-07-30 — 16h time-budget campaign; ledger of 11 directions with per-direction verdicts
