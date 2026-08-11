---
name: credit-a-cumulative-gain-only-from-a-same-session-a-b-never--dense-gemm-gfx950-memory-bound
description: Credit a gain only from a same-session interleaved A/B: a cross-session baseline inflated a real ~1.09x into a reported 1.31x on a tiny dense GEMM
keywords: [measurement-method, interleaved-ab, clock-drift, noise-band, dense-gemm, memory-bound, dispatch-floor]
kernels: [_gemm_a16_w16_kernel]
platforms: [gfx950]
kernel_class: dense_gemm
regime: memory-bound
key: crediting a speedup on a tiny dispatch-floored bf16 dense GEMM (Triton) on gfx950, where box clock drift between sessions is wider than the effect
lifecycle: active
type: instrument
confidence: ★★
effect: The cumulative number against the frozen baseline jumped 1.1026x -> 1.3097x with no new optimization: the cache-only path (no new patch) re-measured ~1.30x on the later box, while the genuine attributable win in a same-session A/B stayed ~1.08-1.10x. Across 18 passes the same artefact reported between 1.1021x and 1.3419x on accepted passes. On the same kernel one honest zero-delta change reported a 0.9745 same-session geomean that was one process landing in a cool-clock window: within-config std ~8% on the largest case and ~29% run-to-run spread (slowest repeat ~1.29x the fastest), against a change that provably removes host work and so cannot slow the path.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: time-budget-16h campaign run, 2026-07-30
last_seen: 2026-08-11
---
# Credit a cumulative gain only from a same-session A/B, never from a delta against a baseline timed on another day
- lever: For a few-microsecond op, treat any speedup computed against a stored baseline timing captured in an earlier session as an upper bound contaminated by clock-state drift, and attribute a gain only from an interleaved same-session A/B of seed vs candidate; re-measure the unchanged cache/seed path alongside the candidate each pass, because if the no-new-patch arm has moved, the difference is the box.
- apply: One process, one lock, alternating arms over several repeats per case, with the frozen seed re-timed as its own arm; report the seed-vs-seed spread next to the candidate delta, and size the per-config noise band before reading a small number as a result.
- verify: The stored-baseline delta and the same-session delta agree; where they do not, the same-session one is the claim and the difference is drift to be recorded rather than banked.
- pitfall: a change that only removes host work reported a sub-1.0x same-session geomean -> that process landed in a cool-clock window and the noise band was wider than the effect -> read a below-1.0x result on a provably-work-removing patch as a band-sizing signal and re-measure, rather than as a regression.
- caution: Also verify this on longer-running kernels before assuming the drift is negligible there — the effect is proportionally largest when the whole call is a few microseconds, but the box drift itself is not kernel-specific.
- source: time-budget-16h campaign run, 2026-07-30
