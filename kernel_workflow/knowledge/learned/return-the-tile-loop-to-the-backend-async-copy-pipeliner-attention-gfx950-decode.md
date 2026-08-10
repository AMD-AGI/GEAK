---
key: attention · gfx950 · decode
type: lever
confidence: ★★
effect: director-verified 1.86x geomean end state (1.82 / 2.11 / 1.67 at batch 2 / 32 / 64, decode q_len=1); this lever's own A/B was +11.0% over the then-best body, concentrated on the latency-bound small/mid-batch shapes (kernel 18.0->14.1 us at batch=2, 19.4->16.9 us at batch=32) and 0% on the large-batch shape already running at ~78% of nameplate bandwidth
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm7.2.3 / triton3.6.0 / torch2.11.0
last_seen: 2026-08-08
---
# Return the tile loop to the backend async-copy pipeliner
- lever: When several rounds of hand-written prefetch / double-buffering have accreted on a tiled loop, measure DELETING them and handing the loop to the backend async-copy path with a per-loop stage count: register relief and issue depth can turn out to be one edit rather than two problems, and a hand-rolled prefetch that beat the compiler early can be the thing blocking it later.
- apply: Enable the AMD async-copy knob around compilation (it is a triton.knobs global with no compile-option surface, so set and restore it) and put the stage count on the loop range itself, which leaves a caller-pinned kernel-level num_stages kwarg untouched; then drop the manual prefetch state the pipeliner now owns.
- verify: Read the cached ISA: global-to-LDS staging loads appear, per-tile LDS writes and barriers collapse (13->0 and 6->1 here), architectural VGPRs fall (250->220) and outstanding loads at the first vector wait rise (0->9); then re-measure wall clock per case, because an ISA success criterion can be met in full while the clock stays flat.
- caution: Also verify that no loop-carried load survives in the body — one such load switched the async lowering off entirely here, reverting the whole structure — and re-sweep the schedule hint and stage count afterwards, since both optima moved when the copy path changed.
- source: run kernel_20_geak_0808_4h 2026-08-08
