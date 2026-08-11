---
name: layout-round-trips-that-sit-in-the-tail-behind-the-critical--topk-routing-gfx950-launch-bound
description: Layout round-trips hidden in the tail behind the critical path are free to keep: deleting all of them moved a launch-bound top-k case inside jitter
keywords: [lds, cross-lane, topk, launch-bound, control-experiment, isa-check]
kernels: [_topk_forward]
platforms: [gfx950]
kernel_class: topk_routing
regime: launch-bound
key: transposed-store LDS round-trips in a caller-allocated Triton top-k / routing op on gfx950 whose device time already sits at its tail floor
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Eliminating 100% of the transposed-store layout round-trips (all three outputs reduced to trivial coalesced stores, 9 s_barriers removed) moved the device-exposed case by ~1%, inside jitter, and produced no patch. A word-split store variant of the same idea went from 9 to 11 barriers and cost +3%. The round-trips were downstream of, and fully hidden behind, a large operand load plus the iterative select; the tail work that remains (softmax ~8% of the case's device time, pack ~6%, select tail ~13%) overlaps into an irreducible floor at ~81-86% of it. The output tensors were allocated by the caller, so restructuring the wrapper's allocation to remove the transpose was never available.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.59h / 50 passes, 2026-08-11
last_seen: 2026-08-11
---
# Layout round-trips that sit in the tail behind the critical path are free to keep and costly to reshape
- lever: Cross-lane and LDS traffic visible in the ISA is only worth a round if it is on the critical path. Locate it in the schedule first: staging that happens after the last dependent load and after the main reduction is hidden, and deleting it will read as jitter no matter how many barriers it removes.
- apply: Run the delete-the-work probe as the cheap version of this — remove only the suspect stage, keep every load and store, and look for movement before spending the round. Check who allocates the output tensors before planning a store-layout fix.
- verify: Compare the measured movement against the case's own re-measurement spread; a change of about a percent on a case already at its tail floor is not a result.
- pitfall: a store-layout rewrite came back +3% slower -> the caller owns the output shape, so a per-row store just re-converts and the reshaped variant added barriers (9 -> 11) -> establish output-buffer ownership before authoring the reshape.
- caution: Also verify where the remaining tail actually is — here it overlaps into a floor at four fifths or more of the case's device time, so even a perfect store-layout fix had a very small ceiling.
- source: chuschen 16h time-budget campaign run, 15.59h / 50 passes, 2026-08-11
