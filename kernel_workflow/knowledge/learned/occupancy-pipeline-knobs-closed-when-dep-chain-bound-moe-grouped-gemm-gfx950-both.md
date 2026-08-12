---
key: low-precision weight-streaming grouped GEMM sitting at occupancy 1 on gfx950, where the residual is an unpack-to-MFMA dependency chain rather than exposed load latency
type: anti-pattern
confidence: ★★
effect: Whole axis disconfirmed against a 42.2x incumbent: a genuine double-buffer of the streamed operand -14% (both larger-batch cases regressed per-case), a second occupancy wave -76%, sequential sub-tiling with operand reload -55%, pipeline depth 2 net-negative, and disabling the existing 1-deep prefetch -43%; the L2/XCD reorder that does help is a ~1% ceiling (+1.2% at mid batch, neutral at the largest, unchanged at the smallest).
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-11
name: occupancy-pipeline-knobs-closed-when-dep-chain-bound-moe-grouped-gemm-gfx950-both
description: Anti-pattern: at occupancy 1 with a reused operand far larger than L2, occupancy, pipeline-depth and L2-swizzle knobs are all negative or ~1% ceilings.
keywords: ['occupancy', 'software-pipelining', 'double-buffer', 'prefetch', 'num-stages', 'l2-residency', 'xcd-swizzle', 'dep-chain', 'moe', 'grouped-gemm']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
verified_on: 2026-08-11
---
# occupancy-pipeline-knobs-closed-when-dep-chain-bound
- lever: Treat occupancy, pipeline depth and cache-residency hints as a single axis that is only worth a round while the kernel is single-wave latency-STALLED; spend the round elsewhere once a cheap probe says otherwise.
- apply: Cheap probe first: add one software double-buffer of the operand that is actually on the memory critical path. If it REGRESSES, the residual is a dependency chain plus reuse traffic, and no depth knob, extra wave or eviction hint reaches it — the extra buffer only adds register pressure.
- verify: Read the prefetch knob's own source before sweeping it: here the values were not a depth, only {on, off}, so three of five settings silently disabled a 1-deep buffer and looked like a depth regression. Re-time each setting against the frozen baseline per case.
- pitfall: A second occupancy wave passed the parity gate and still cost ~76% -> the reused operand set per compute unit doubled and is two orders of magnitude larger than L2, so the second wave only thrashed -> keep one wave and size tiles for reuse instead.
- caution: Also verify the reused operand set against L2 capacity before borrowing an eviction-policy or non-temporal lever from another op: when reuse far exceeds L2 the op is traffic-bound, not pollution-bound, and the hint has nothing to protect.
- source: 16h per-kernel time-budget campaign, lane chuschen16h, ledger dead_end / TRUE-NEG entries, 2026-08-11
