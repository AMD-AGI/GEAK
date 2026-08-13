---
key: an already-tuned split-KV decode attention that ships as two dispatches and whose main dispatch is at its measured DRAM roof with all three resource budgets full, gfx950/MI355 HIP
type: anti-pattern
confidence: ★★
effect: four structural directions returned ~1.00x or worse on the decode cases: two independent reduce-fusion implementations at -4.8% and -3.9%, grid right-sizing / empty-workgroup deletion at 0.0 +/- 0.1% in two separate attempts, 5 waves/SIMD at -18%, partition-size doubling at -6.4%; a two-oracle decomposition then bounded any fusion at +1.2% weighted, below the round's 4% kill gate
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode
description: Closed axis: at ~99.8% of the DRAM roof, decode-attention reduce fusion, grid right-sizing, occupancy and cache-residency steering all return ~1.00x
keywords: ['gfx950', 'attention-decode', 'paged-attention', 'decode', 'anti-pattern', 'closed-axis', 'kernel-fusion', 'dispatch-collapse', 'occupancy', 'grid-occupancy', 'empty-workgroups', 'l2-residency', 'roofline', 'oracle', 'memory-bound']
kernels: ['aiter_paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
roofline: main dispatch at ~99.8% of a re-measured blended DRAM roof; registers 124/128 unified, LDS 100% allocated, waves held at 4/SIMD
levers: ['host.dispatch-structure', 'compute.occupancy']
---
# Price the residual before funding fusion or geometry work at the roof
- lever: Before spending a direction on fusing the split-KV reduce, right-sizing the grid, chasing occupancy or steering last-level-cache residency, price what is left: measure the fraction of the re-measured blended DRAM roof the main dispatch already reaches, and convert the residual into a ceiling on the weighted metric. Near the roof these axes were arithmetic dead ends here, and the number tells you that in one hour instead of three rounds.
- apply: Price a second dispatch with TWO oracles, not one: an empty-body kernel, and an empty-body-plus-one-real-workgroup kernel. Their difference separates a launch ramp (attackable by grid geometry, batching or graph capture) from a fixed pipeline barrier / cache flush (only fusion touches it). Here the fixed part was ~41% of the inter-dispatch gap and the merge body ~59%, so the fusion prize was ~1.2% weighted at best.
- verify: Check that a 1-workgroup and a full-grid empty kernel really do cost the same before calling the cost fixed, and sanity-check the ceiling with a deliberately wrong-answer oracle that deletes the second dispatch entirely - it reached ~1.41x here against a delivered ~1.35x, which is the whole remaining headroom in that seam.
- pitfall: two independent fusion implementations both lost (64-bit data atomics, then a ticket-counter last-arriver merge) -> at full residency the merging workgroup still holds its LDS slot and its registers, so the epilogue displaces the next slab instead of overlapping it; the protocol was cheap and the merge body was the cost -> price the slot, not the arithmetic.
empty-workgroup deletion measured 0.0 +/- 0.1% even at 4 LDS slots per CU -> empty workgroups retire for free once the grid is co-resident -> the premise was wrong, not the implementation.
coalescing and page-granularity 'defects' were falsified three ways (over-fetch ratio 1.01, widest load already issued) and the fix was 6-8% worse -> a plausible mechanism outlived its effect -> measure the mechanism, not only the effect.
- caution: Also verify the roof is re-measured on the same box and blended for the actual mix before quoting a fraction of it. Also verify that a cache-residency knee exists before steering for it: no large last-level knee was found here, only a smaller L2 one, which left the hint space one bit wide and already optimally spent.
- source: run paged_attention_ragged-own16h, 2026-08-12 (16h per-kernel campaign, rounds 1-7 including a dedicated deep_explore round; director-validated accepted)
