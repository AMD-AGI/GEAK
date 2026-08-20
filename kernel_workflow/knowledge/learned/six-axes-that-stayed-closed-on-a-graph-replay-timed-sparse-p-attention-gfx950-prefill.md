---
key: which directions are already spent on a Triton gathered sparse prefill attention op on gfx950 once the retile and the rendezvous deletions have landed
type: anti-pattern
confidence: ★★
effect: host/runtime measured exactly 0.000% in every form; the <=128 VGPR occupancy escape is 4.20x slower through spills; the symmetric 2-stage prefetch is -12%; num_stages>1 is a -255% cliff; on the tiny case the whole remaining knob space is worth 1.6% at 0.77% of the weighted metric
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill
description: Closed axis: host/runtime, occupancy, prefetch, LDS order, launcher knobs and LDS-for-bandwidth all returned <=1.00x on gathered sparse prefill attention
keywords: ['anti-pattern', 'closed-axis', 'attention', 'prefill', 'triton', 'host-runtime', 'graph-replay', 'occupancy', 'vgpr-pressure', 'software-prefetch', 'num-stages', 'static-isa-screen', 'gfx950']
kernels: ['_sparse_attn_prefill_ragged_kernel']
platforms: ['gfx950']
kernel_class: attention
regime: prefill
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-17
origin_kernels: ['mi355x_vllm_triton_sparse_attn_prefill_ragged']
---
# Six axes that stayed closed on a graph-replay-timed sparse prefill attention op
- lever: Price these six axes before staffing a round on them - each measured closed with a mechanism across 16 rounds here: host/runtime; occupancy via a lower VGPR ceiling; prefetch in four forms; statement-order steering of LDS; launcher knobs; and trading LDS traffic for global bandwidth.
- apply: Occupancy is closed from both ends (the leanest expressible body is 194 VGPR with zero spills, while 128 is already the accumulator plus the duplicated q before any score, tile or pointer exists). schedule_hint is silently dropped at num_stages=1; async-copy and pingpong lower to nothing; kpack is reset to 1 on this arch; nonkdim=32 spills at compile time and nonkdim=16/kpack=2 emit byte-identical asm. There is no global-bandwidth cost to trade into either: forcing every gather to one cache slot is slower than the control, refuting the latency-limiter premise that had steered three rounds.
- verify: The static asm compile screen answers most of this off-GPU - 12-20 variants build in the time of two benchmark runs - but it is only valid within one process/module instance: loading the same source as two modules yields different hashes with byte-identical statistics and manufactures fake 'differs' verdicts. Take statics from the driver's own build cache, not a hand-built AST source.
- pitfall: Host work looked like the obvious first lever -> the harness runs the whole Python launcher at graph-capture time and times only the replay -> an injected per-call delay moved the eager path by the injected amount and moved replay by nothing; price the timed bracket before the launcher.
Instruction-count intuition inverted three times -> deleting an always-true load predicate costs 3.1-3.7% (a masked load folds into the buffer descriptor for free), an unsigned-compare rewrite with 58 fewer asm lines is -5.8%, and 18 static barriers beat 9 -> count rendezvous per trip, not instructions.
- caution: Also verify a stuck small case is kernel-bound rather than launch-bound before spending on it: here graph replay of the tiny shape already exceeded the host+launch cost the graph removes.
- source: run mi355x_vllm_triton_sparse_attn_prefill_ragged-bmk7-12h, 2026-08-17, rounds 2-16, director-validated
