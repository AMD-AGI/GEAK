---
key: store-bandwidth-saturated chunked linear-attention forward on gfx950 where the small case sits on the harness per-call event-bracket floor
type: anti-pattern
confidence: ★★
effect: four axes closed on the same shape set: occupancy 3->4 waves gave 1.00x on both large cases; persistent grid-stride cost +16% on the largest case even at a logically no-op stride factor; 4 heads per program cost +21%; forced per-call graph replay on the smallest case measured a median identical to eager
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound
description: Once a kernel sits at its store roofline, occupancy lift, persistent grid-stride, finer store-skip granularity and graph replay all returned ~1.00x or regressed
keywords: ['anti-pattern', 'closed-axis', 'occupancy', 'persistent-kernel', 'grid-stride', 'graph-replay', 'launch-overhead', 'store-bandwidth', 'memory-bound', 'gfx950', 'linear-attention']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: memory-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
roofline: store-bandwidth-bound at ~0.8-0.97 of achievable store roof before and after all four attempts
levers: ['compute.occupancy', 'host.persistent-grid', 'host.graph-replay']
---
# Axes that stay closed once the store pipe is saturated
- lever: When the dominant case is already near its store roofline, rank occupancy/VGPR lift, persistent grid-stride workgroup collapse, finer triangular store-skip granularity and per-call graph replay as low-prior seeds - all four were measured out here - and spend the round on store bytes and store cache policy instead.
- apply: Cheap disconfirmations first: read the bound class off a trace-based roofline, then check whether the candidate axis can move it at all before authoring a patch for it.
- verify: Benchmark inside the scoring harness with interleaved alternating samples; an isolated in-process probe showed replay clearly ahead of eager while the harness measured them identical, because the isolated eager was inflated by long-lived-process clock effects.
- pitfall: A host-side lever was recorded as banked but was silently inactive -> its measured-benefit gate timed batched back-to-back launches with one sync (throughput) while scoring used per-call event brackets -> probe the runtime mode after the real harness warmup rather than trusting the gate.
Finer store-skip granularity looked strictly better on bytes but regressed -> at fine granularity the case turns store-issue-bound rather than store-byte-bound -> the coarse 2x2 skip stayed optimal.
- caution: Also verify where the smallest case's floor is before spending a round on host-side levers: below the harness per-call event bracket no launch-side change can score, and the required per-case ratio is computable up front.
- source: run chunk_scaled_dot_kkt_fwd_kernel-ch16h, 2026-08-12 (16h per-kernel time-budget campaign, dead_end ledger entries across compute / memory / host_runtime)
