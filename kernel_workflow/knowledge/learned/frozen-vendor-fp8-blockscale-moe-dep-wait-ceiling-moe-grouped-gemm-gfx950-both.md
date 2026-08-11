---
key: fp8 per-1x128 block-scaled two-stage MoE grouped GEMM on gfx950, built from a frozen vendor CK pipeline (only instance-generation knobs reachable)
type: anti-pattern
confidence: ★★
effect: ceiling ~1.29x weighted over the frozen baseline, flat per-case (small-batch case 1.23x, mid 1.30x, large 1.31x, so no shape escapes it); the disconfirming half: wrapper graph capture replays 3-4% slower, alternate/compiler-default hot-loop schedules regress 0.4-1.4% (numerically neutral, SNR ~32dB), LDS-pad flags and epilogue write-out clustering and per-stage M-tile split all returned 0, and fp4 weight storage failed the numeric gate at 14-19dB SNR vs a 25dB bar; roofline fraction only 0.37 -> 0.52 of its own roof, still compute/latency-bound after 24 passes
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 24
toolchain: unknown
last_seen: 2026-08-11
name: frozen-vendor-fp8-blockscale-moe-dep-wait-ceiling-moe-grouped-gemm-gfx950-both
description: On frozen vendor fp8 block-scale MoE grouped GEMM the post-MFMA VALU rescale caps gains near 1.3x; six further axes measured zero or negative.
keywords: ['moe', 'fp8', 'blockscale', 'grouped-gemm', 'mfma', 'latency-bound', 'dep-wait', 'closed-axis', 'gfx950', 'anti-pattern']
kernels: ['moe_gemm_fp8_blockscale']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
---
# frozen-vendor-fp8-blockscale-moe-dep-wait-ceiling
- lever: when the hot loop is a vendor-frozen fp8-dequant -> MFMA -> fp32-accumulate chain with the block scale applied POST-MFMA, treat the residual gap as intrinsic low-ILP dep-wait (~56% of cycles) rather than a schedulable gap, and spend the round elsewhere
- apply: diagnose first: dump the ISA to confirm native f8 MFMA is already emitted (no software emulation to remove), and count independent accumulators per K-scale block — 2 XDL accumulators feeding one fp32 rescale is the signature
- verify: an A/B on this axis is worth one cheap round at most: re-time per-case against the frozen baseline and look for a non-overlapping distribution, not a sub-1% mean shift
- pitfall: a scheduler axis looked open but the vendor pipeline header only specializes the Intrawave variant (primary template body empty) -> flipping the scheduler enum compiles to a struct with no Run() -> build failure, so the axis is closed at compile time, not at measurement
- caution: measured on a 3-case weighted workload with instance-generation as the only reachable seam; also verify on your own build whether the scale granularity is hoistable or the block-K is tunable, since either would reopen the rescale axis
- source: run moe-blockscale-16h campaign, 2026-07-29..2026-08-11, 24 resumed passes / 11 ledger directions
