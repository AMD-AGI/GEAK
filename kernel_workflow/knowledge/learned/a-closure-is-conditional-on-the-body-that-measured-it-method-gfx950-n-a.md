---
key: deciding when a 'closed' verdict has expired, on a long multi-round Triton attention run where tile constants and dispatch targets keep moving
type: method
confidence: ★★
effect: three previously confirmed directions reversed sign on the heavy prefill case when the k-block doubled - one gather form had been measured at -6.5% to delete and became +6.1% to delete - and a landed locality remap that a dispatch rewire dropped was ~1% of the weighted metric left on the floor for six rounds
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a
description: After a structural move, re-audit closures and landed hunks: three confirmed wins inverted when the k-block doubled; a dispatch rewire dropped a remap
keywords: ['method', 'bottleneck-shift', 'dead-list', 'stacking', 'measurement-discipline', 'verification', 'xcd-remap', 'attention', 'triton', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L0
verified_on: 2026-08-17
---
# A closure is conditional on the body that measured it
- lever: Treat every 'closed' verdict as conditional on the structural constants of the body that measured it, and re-open the list whenever one of them - trip count, tile shape, dispatch target - moves.
- apply: Record the precondition next to each closure; after a patch that changes the k-block or rewires dispatch, re-run the cheap closed knobs on the new body and grep the executed path for the hunks earlier rounds landed.
- verify: Diff the live kernel body against the shipped patch rather than trusting a clean apply; confirm re-engagement with a counter that names the mechanism (cache hit rate and DRAM read count for a locality remap) alongside an interleaved same-session control.
- pitfall: A stacked pair of individually-positive patches came out negative -> one added a new dispatch target, so the other's hunks were dead code on the measured shapes -> hand-merge one mechanism at a time and re-measure each.
A landed locality remap silently vanished -> a later dispatch rewire re-created the launcher without it and nothing re-read the locality counters for six rounds -> re-audit dispatch-level optimizations after every dispatch rewrite.
- caution: Also verify a deadline-truncated round's claims against the tree itself rather than the engineer's prose: two blackboard claims here were false of the artifact, and the round's real win took extra rounds to book.
- source: run mi355x_vllm_triton_sparse_attn_prefill_ragged-bmk7-12h, 2026-08-17, rounds 7-15
