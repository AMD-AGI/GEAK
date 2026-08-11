---
key: host binding hit path (allocation reuse + lookup shortcut) for a repeatedly called thin dense linear on gfx950, where the device kernel is untouched
type: lever
confidence: ★★
effect: 1.08-1.10x attributable on same-session A/B, collapsing ~45% of the per-call host work; per-case cumulative vs the frozen baseline 1.27x / 1.34x / 1.32x at batch 2 / 32 / 64. The follow-on trim of redundant validation on the same hit path measured 1.00x (+-3-4%, inside noise), so the host axis closes after the cache.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-11
name: cache-the-host-hit-path-first-the-rest-of-the-host-path-is-f-dense-gemm-gfx950-decode
description: Allocation reuse plus a lock-free last-hit shortcut was the only paying lever on a dispatch-floored dense linear; further host trimming measured zero.
keywords: ['host-runtime', 'launch-overhead', 'dispatch-floor', 'caching', 'decode', 'skinny-m', 'measurement-drift']
kernels: ['wvSplitK']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
---
# Cache the host hit path first; the rest of the host path is floor
- lever: on a repeatedly called small op, reuse the output/workspace allocation and shortcut the config lookup with a lock-free thread-local last-hit slot before touching the device kernel.
- apply: in the C++ binding, keep the previous key and dispatch target in thread-local storage and skip the mutex plus table scan on a repeat hit; confirm the device side is untouched (dispatch count per call stays 1).
- verify: same-session interleaved A/B against the seed across every case, repeated enough to see the within-config spread; only credit a delta that survives interleaving.
- pitfall: the cumulative number jumped 1.10x -> 1.31x with no new patch -> the box was simply running faster than when the baseline snapshot was taken -> re-measured the cache-only path in the same session and credited the ~1.08-1.10x, keeping the drifted figure out of the attribution.
- caution: also verify the residual host work is really below the launch/enqueue floor before booking another host round: here integer-compare trimming on the same path was correct and measured zero.
- source: chuschen 16h per-kernel time-budget campaign, 18 passes, 2026-08-11
