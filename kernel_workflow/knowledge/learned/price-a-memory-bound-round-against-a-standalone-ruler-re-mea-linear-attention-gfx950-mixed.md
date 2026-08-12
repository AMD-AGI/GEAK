---
key: roofline denominators for a chunked linear-attention (delta-rule / KKT) kernel on gfx950, store-dominated large-batch case
type: method
confidence: ★★
effect: Instrument, not a speedup: end state director-verified 25.55x geomean (3.21x on the smallest batch case, 47.2x and 110.2x on the two larger). Its value showed on the largest, store-dominated case, where a freshly built ruler re-priced remaining headroom from a funded 1.4x to a measured 1.18x (that case sat at 85% of its own achievable roofline) and the structural round it graded returned +2.3% interleaved-A/B. A carried absolute-bandwidth figure had inflated three rounds of roofline percentages by ~20%, and a flat-fill ruler read 33% off the mixed read+write ruler at the same byte count.
confirms_cited: 1
confirms_blind: 0
losses: 2
attempts: 9
toolchain: rocm 7.2.3 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: price-a-memory-bound-round-against-a-standalone-ruler-re-mea-linear-attention-gfx950-mixed
description: Price a memory-bound round against a standalone ruler at the kernel's exact read+write byte count: re-priced 1.4x of funded headroom to a measured 1.18x
keywords: ['roofline', 'measurement-method', 'memory-bound', 'control-experiment', 'dispatch-floor', 'interleaved-ab', 'linear-attention']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: mixed
lifecycle: active
---
# Price a memory-bound round against a standalone ruler re-measured at the kernel's exact mixed read+write byte count
- lever: Before funding a structural memory round, spend an hour on a standalone ~60-line kernel that reproduces this kernel's grid, addressing and EXACT traffic (bytes read at the read dtype + bytes written at the write dtype), and use it — not a carried absolute-bandwidth number and not a flat fill — as the denominator for every expected_speedup in the plan.
- apply: Re-measure the ruler on the box being benchmarked in the current round. Build a small ladder at the same geometry — empty body / store-only / mixed read+write at the exact byte count / full kernel — so the residual splits into host issue rate, store side and read side; an empty-body time identical across two very different case sizes is host issue rate, not device work. Variants of the ruler (wider bursts, fewer bytes) price geometry levers before any kernel edit.
- verify: The ruler is trustworthy when its full-kernel rung reproduces the measured kernel time and its empty rung reproduces the known dispatch floor; a lane is worth funding only if the ruler's variant beats its baseline rung by more than the round's noise band. Confirm any resulting patch with a same-session interleaved A/B, since cross-session geomeans on the same code drifted ~3% here and cannot support a claim smaller than that drift.
- pitfall: three rounds of roofline percentages came out ~20% optimistic -> a bandwidth figure carried from an earlier box/session was being used as the denominator, and a flat-fill ruler read 33% off the mixed read+write one -> re-measure the ruler on this box at this round's exact byte mix.
- caution: Also verify the caller's allocator before pricing any output-byte-elision lever the ruler makes look attractive — here the ruler valued skipping unwritten output at 23-33%, but the output buffer came from an uninitialised allocation, so unwritten bytes would have been garbage; and also verify that a byte-cutting scheme keeps the contiguous row width, since narrowing it can cancel the saved bytes exactly.
- source: run kb_on_0810 2026-08-10
