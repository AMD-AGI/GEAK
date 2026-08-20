---
key: paged / grouped-query attention decode stage-1 in Triton on gfx950, latency-floored at about one workgroup per CU
type: lever
confidence: ★★
effect: ~1.21x isolated geomean on decode from launch-meta alone (per-case: mid and large KV-context cases carry it, the tiny-grid case ~1.00x); unpinning waves_per_eu then adds ~+0.7% geomean, carried entirely by the mid case at ~+1.9% with the other two tied; banked total ~1.23x vs frozen baseline
confirms_cited: 1
confirms_blind: 0
losses: 1
attempts: 5
toolchain: unknown
last_seen: 2026-08-17
name: launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode
description: Launch-meta is the primary lever on latency-bound paged grouped-attention decode; unpinning waves_per_eu beats a pinned hint, numerically exact.
keywords: ['launch-meta', 'num-stages', 'waves-per-eu', 'occupancy', 'attention-decode', 'paged-kv', 'latency-bound', 'gfx950']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
roofline: memory/latency-bound at ~0.18 of empirical roofline before -> ~0.39 and compute-bound after
origin_kernels: ['_fwd_grouped_kernel_stage1']
---
# Launch-meta first on latency-floored paged decode, and let the backend pick occupancy
- lever: On a decode attention kernel whose profile says latency-floored at ~1 workgroup/CU, put the round into launcher metadata (num_warps, num_stages, matrix-instr dim) and treat a pinned waves_per_eu hint as one more ablatable field rather than part of the winning set.
- apply: Tune the launcher meta as a group, then re-ablate each field on its own; leaving waves_per_eu unset lets the compiler choose occupancy. Gate the hint injection behind an env-read flag so both arms come from one build.
- stack: total ~1.23x isolated (weighted geomean, frozen baseline) = three directions compounded
  - 1. launch-meta group (num_warps/num_stages/matrix-dim) — ~1.21x standalone, the bulk of the win; 2-stage pipelining is load-bearing
  - 2. scalar page-index load — ~+1.3% on top of (1), bit-identical, orthogonal
  - 3. unpinning the occupancy hint — ~+0.7% on top of (1,2), numerically exact
  - note: attribution is incremental in landing order; (3) was only visible after (1) was banked
- verify: Sweep the two arms through one env switch so no rebuild separates them, interleave ~a dozen alternating repeats per arm and compare medians against the frozen baseline; for a launch-only change also require bit-exact output (zero error ratio).
- pitfall: sub-1% launch-only delta came back as not-improved from the automated verdict -> the delta sits inside the harness noise band -> bank it on interleaved medians and grep the built artifact to confirm the config actually flipped
the pinned occupancy hint looked like part of the winning launch-meta set -> its gain was carried by the other fields -> re-ablate each launcher field separately once the group is banked
- caution: Also verify how load-bearing the pipeliner depth is before restructuring the loop: on this shape a loop-split cost ~4-5% on the large case by breaking 2-stage pipelining.
- source: GEAK 16h per-kernel time-budget campaign, run _fwd_grouped_kernel_stage1-ch16h, 2026-08-12, 61 passes
