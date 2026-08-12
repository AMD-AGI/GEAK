---
key: fp8 per-block-scale grouped MoE GEMM on gfx950 whose gridwise lives in a frozen third-party library outside the editable surface
type: anti-pattern
confidence: ★★
effect: disconfirming: host dispatch collapse 1.0036x and wrapper graph capture/replay 1.0075x, both inside case-to-case spread on the smallest case; occupancy re-sweep, K-per-block widening and LDS halving all flat on every case; epilogue store-vector widening a monotonic regression; 9 of 11 directions dead-end
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 9
toolchain: unknown
last_seen: 2026-08-11
name: a-frozen-vendor-gridwise-closes-the-occupancy-host-launch-an-moe-grouped-gemm-gfx950-both
description: When the hot gridwise sits in a frozen vendor library, occupancy/host-launch/epilogue/low-precision-B axes each returned ~1.00x: a ~1% ceiling, not a lever
keywords: ['anti-pattern', 'reachability', 'occupancy', 'launch-overhead', 'hip-graph', 'epilogue', 'mxfp4', 'moe', 'grouped-gemm', 'composable-kernel']
kernels: ['moe_stage2']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-08-11
---
# A frozen vendor gridwise closes the occupancy, host-launch and epilogue axes at once
- lever: Before spending rounds on occupancy, host launch, epilogue staging or a lower-precision B operand, check reachability first: establish which files actually compile into the dispatched object and which template the harness dtype/scale combination selects.
- apply: Compile-canary an intentional error into each candidate file to find the decorative mirrors; then trace the dispatch heuristic from the harness dtypes to the single instantiable template, and read what that gridwise special-cases.
- verify: A direction that survives reachability still needs the isolated A/B; measure host-side probes interleaved in the same process, because a sequential A/B on a compute-bound kernel reports launch wins that vanish when interleaved.
- pitfall: A mirror source tree accepted edits and produced bit-identical results -> it was never compiled -> the compiled surface was a different directory, found with a canary error.
A loose correctness gate suggested a low-precision-B swap was affordable, but that operand format is a distinct gridwise with an incompatible scale granularity the harness never emits -> the round was spent proving unreachability, not measuring.
- caution: Also verify the profiler's improvement verdict against the raw per-case numbers: this lane saw improved=false emitted on rounds that were genuinely improving, so a config delta plus a diff against the canonical build is the safer read.
- source: GEAK per-kernel time-budget campaign, chuschen16h lane, 2026-08-11
