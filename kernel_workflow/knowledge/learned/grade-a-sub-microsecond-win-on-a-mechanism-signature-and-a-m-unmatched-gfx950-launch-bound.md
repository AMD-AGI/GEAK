---
name: grade-a-sub-microsecond-win-on-a-mechanism-signature-and-a-m-unmatched-gfx950-launch-bound
description: Grade a tiny win on the per-case signature its mechanism predicts plus a median: five straight false rejections became a confirmed 3.00x
keywords: [measurement-method, interleaved-ab, noise-band, clock-drift, launch-bound, dispatch-floor, control-experiment]
kernels: [write_req_to_token_pool_triton]
platforms: [gfx950]
kernel_class: memory_movement
regime: launch-bound
key: grading a few-percent win on a tiny int64 single-dispatch copy kernel (Triton, gfx950) whose whole graded time is bimodally throttled by the box
lifecycle: active
type: instrument
confidence: ★★
effect: A bit-exact int64 win was rejected five consecutive times by the single-shot harness gate, scoring 2.512 / 2.5493 / 2.5916 / 2.594 / 2.62 against a 2.613 bar, purely from bimodal whole-process throttle (the slow mode ran ~1.3x the fast mode, moving batch 32 and 64 together while batch 2 stayed flat). The same patch under better instruments: 12-run median per-case geomean 2.619 and per-run median 2.631, fast-window per-case minima 2.986, and an interleaved 8-pair same-session A/B in which the batch64-batch32 gap collapsed to ~1/6 of its size exactly as the mechanism predicted, the batch-64 minimum improving 12.7% and geomean-on-minima 2.827 -> 3.000. An earlier verdict derived from a single noisy shot ('the scan is off the critical path') was recorded as wrong.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: run chuschen16h 2026-08-11
last_seen: 2026-08-11
---
# Grade a sub-microsecond win on a mechanism signature and a median, not on one harness shot
- lever: When the whole graded call is dominated by a fixed per-call bracket, a box with bimodal throttling can swamp a real few-percent win, and the failure looks like a reproducible rejection rather than like noise; the tell is several cases moving together while an unrelated case stays flat.
- apply: Take a median over ten or more back-to-back samples, or an interleaved pair-wise A/B in one session, and above all state the per-case signature the mechanism predicts before measuring — here, replacing a serial per-program dependent-load prefix scan with an O(1) precomputed exclusive prefix should close the gap between the two largest batch cases, and it did.
- verify: A predicted per-case signature that shows up is evidence a single geomean cannot give; check the fast-window minima as well as the medians, and re-read any earlier one-shot verdict about the same mechanism before trusting it.
- pitfall: a real win was rejected five times in a row -> single-shot grading on a bimodally throttled box, with the slow mode ~1.3x the fast mode -> grade on medians plus an interleaved same-session pair-wise A/B, and check the predicted per-case signature.
- caution: Also verify which part of the mechanism actually costs: vectorising the same scan with a masked tl.sum added work and measured 0.973x, because the cost was the dependency chain rather than bandwidth.
- source: run chuschen16h 2026-08-11
