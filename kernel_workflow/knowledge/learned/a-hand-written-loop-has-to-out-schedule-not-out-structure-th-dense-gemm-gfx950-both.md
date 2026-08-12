---
key: latency-bound block-scaled fp8 dense GEMM on gfx950 after the compute fix, deciding whether to hand-schedule below Triton's pipeliner or spend the round elsewhere
type: anti-pattern
confidence: ★★
effect: six rounds of structural directions returned 1.00x or worse on all cases (M=2048/32768/65536): hand-scheduled LDS double buffer 0.78x, narrow-nonkdim MFMA reshape 0.86x, split-k 0.85x, graph capture 0.97x; best real gain +0.4..0.7%, inside the ~1.2-1.7% in-window floor
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-12
name: a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both
description: Closed axis: on a latency-bound fp8 GEMM with zero bank conflicts, hand-scheduled double buffer, mfma reshape, split-k and graphs all return <=1.00x
keywords: ['anti-pattern', 'latency-bound', 'double-buffer', 'split-k', 'cuda-graph', 'mfma', 'dense-gemm', 'fp8', 'triton', 'gfx950']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
---
# A hand-written loop has to out-schedule, not out-structure, the compiler
- lever: when the loop is latency-bound with zero bank conflicts and LDS traffic already at its derived minimum, treat structure as a closed axis and spend the round on a different algorithm or a compiler with scheduling controls
- apply: before committing to a hand-scheduled rewrite, build the cheap control first: a faithful single-buffer mimic of the compiler's own loop; if the mimic already sits far below the compiler, the structural gain inside the rewrite cannot recover the gap
- verify: a structural claim needs bit-exact output plus the resource proof (shared bytes, VGPR, spills, ds_write count) before the frozen A/B, and a diff against canonical so an unchanged incumbent cannot read as a small win
- pitfall: a launch-overhead attack looked promising on CPU-per-call -> injecting extra host delay per launch moved the score 0%, because the queue is permanently backlogged and host cost never enters the event bracket -> close host directions with that injection control, not with a geomean
- caution: also verify what the compiler exposes before rating a hand-scheduled direction: the residual here turned out to be instruction interleaving the pipeliner already does, and the single-buffer mimic control is what separates that from the LDS structure
- source: run _gemm_a8w8_blockscale_kernel-own16h, 2026-08-12, rounds 4-9
