---
key: Layout round-trips and serial argmax depth in the tail of a top-k select on gfx950, once the grid is one full occupancy wave and the load dominates
type: anti-pattern
confidence: ★★
effect: 1.00-1.01x across two independent clean negatives on the largest case: (a) eliminating 100% of the transposed-store layout round-trips and 9 barriers moved it <1%, inside jitter, and a word-split store variant went the wrong way ~3%; (b) halving serial reduction depth (4 reductions+3 masks -> 2+1) passed correctness and gave exactly 0%. Smaller cases unaffected (dispatch-floored). Achieved roofline stayed ~0.10 -> ~0.14 of peak.
confirms_cited: 1
confirms_blind: 0
losses: 1
attempts: 6
toolchain: unknown
last_seen: 2026-08-11
name: the-tail-is-free-once-the-grid-is-one-full-occupancy-wave-moe-router-topk-gfx950-decode
description: Anti-pattern: at one full occupancy wave, tail store-layout round-trips and serial reduction depth are hidden behind the load path; removing them buys ~0.
keywords: ['convert-layout', 'lds', 'occupancy', 'memory-bound', 'anti-pattern', 'reduction', 'triton']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: moe_router_topk
regime: decode
layer: learned
lifecycle: active
---
# The tail is free once the grid is one full occupancy wave
- lever: Before spending rounds on tail store layout or on shortening a serial reduction chain, ablate them to zero and measure: if the grid is one full occupancy wave and a large operand load sits upstream, the tail overlaps and both directions return ~0 - spend the budget on the load or on the launch path instead.
- apply: L3 diagnostic ablation: drop the outputs to trivial coalesced stores (correctness intentionally broken) to price the tail, and separately land a depth-halving reduction combiner that keeps correctness.
- verify: A/B each ablation against the frozen baseline on the largest case only; a change inside the jitter band of that case is a zero, not a small win.
- pitfall: Rewriting output layout looked available but the OUTPUT TENSORS ARE ALLOCATED BY THE HARNESS, so any per-row store just re-converts -> price the tail by ablation before planning a wrapper-side allocation change that the interface does not permit.
- caution: Also verify the ~1% class of results against the case's jitter band before banking them: an exp2-native softmax rewrite here verified at ~+1% and did not survive as a durable win.
- source: 16h per-kernel time-budget campaign, lane chuschen16h, two dead_end directions, 2026-08-11
