---
key: host binding hit-path caching for a bf16 skinny GEMV decode linear on gfx950 where per-call host work is a large share of the wall
type: lever
confidence: ★★
effect: ~1.08-1.10x same-session geomean across the three decode cases (tokens=2 and 4) from allocation reuse plus a lock-free last-hit shortcut, collapsing a host share that was ~45% of the per-call wall; every device-side direction in the same run returned ~1.00x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode
description: On a launch-floored decode GEMV, host-side alloc reuse plus a lock-free last-hit shortcut gave ~1.08-1.10x while every device-side direction returned ~1.00x.
keywords: ['launch-overhead', 'dispatch-floor', 'host-runtime', 'decode', 'gemv', 'caching', 'timing-drift']
kernels: ['wvSplitK_hf_sml_']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: archived
---
# Cache the per-call host work when the host owns a large share of a tiny op's wall
- lever: reuse allocations across calls and add a thread_local last-hit shortcut that skips the mutex and the table scan on the binding hit path
- apply: in the C++ binding: compare the cached last-hit descriptor before taking the lock, fall through to the locked lookup on a miss; the device kernel is untouched
- verify: confirm the dispatch count is unchanged so the gain is provably host-side, then run a same-session interleaved A/B per case against the raw unwrapped frozen baseline
- pitfall: the cumulative-vs-frozen number jumped from ~1.10x to ~1.31x with no new patch -> box timing drift between sessions, not an optimization -> re-A/B in one session before crediting any cumulative delta
- caution: also verify there is host work left to remove: trimming further integer compares and branches off the same hit path measured +-3-4%, inside the ~8% within-config spread, i.e. measured-closed rather than helpful
- source: run wvSplitK-ch16h, 16h per-kernel time-budget campaign, 2026-08-12
