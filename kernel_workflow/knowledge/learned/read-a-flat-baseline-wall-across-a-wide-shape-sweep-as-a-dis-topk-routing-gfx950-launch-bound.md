---
name: read-a-flat-baseline-wall-across-a-wide-shape-sweep-as-a-dis-topk-routing-gfx950-launch-bound
description: A flat baseline wall across a 32x shape sweep is a dispatch floor, not the roofline label: the host lane paid 2.29x on the smallest case, 1.0x on the biggest
keywords: [dispatch-floor, launch-overhead, host-launch, launch-bound, roofline, topk, measurement-method, control-experiment]
kernels: [_topk_forward]
platforms: [gfx950]
kernel_class: topk_routing
regime: launch-bound
key: reading the per-case baseline profile of a small top-k routing op on gfx950 whose automated classifier says memory-bound while the wall is a host dispatch floor
lifecycle: active
type: instrument
confidence: ★★
effect: Three baseline cases spanning a 32x row range all landed within ~8% of each other in wall time (1.081x spread end to end), while the automated classifier labelled the kernel memory-bound at roofline-emp 0.100. Acting on the flat-wall reading instead, the host lane supplied two of the three largest banked steps in the ledger (1.6489x for a cached direct launch closure and 1.7654x for a trusted steady state, with an algorithm step of 1.719x between them) and paid 2.29x on the smallest case, 1.89x on the mid case, and nothing at all on the largest. End state 2.1169x cumulative, best validated pass 2.1806x, and the opt-side classifier still says memory-bound at roofline-emp 0.140 even though the per-case profile is a dispatch floor plus one device-exposed case.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.59h / 50 passes, 2026-08-11
last_seen: 2026-08-11
---
# Read a flat baseline wall across a wide shape sweep as a dispatch floor, not as the roofline label
- lever: Before believing a roofline verdict on a small op, put the per-case baselines side by side. If wall time barely moves while the work per case moves by an order of magnitude, the constant part is host dispatch and the roofline number is measuring a body that is not on the critical path.
- apply: Grade any host-lane change per case rather than on the geomean - the payoff concentrates in the smallest cases and disappears on the largest, so a geomean understates the lever and hides where it stops working.
- verify: The same reading tells you when to stop: once the small cases sit on the dispatch floor, only the one device-exposed case is still worth a body round.
- caution: Also verify the classifier label against the per-case wall spread before funding a body round - here both the analyze-side and the opt-side classifier kept reporting memory-bound while the per-case profile was a dispatch floor.
- source: chuschen 16h time-budget campaign run, 15.59h / 50 passes, 2026-08-11
