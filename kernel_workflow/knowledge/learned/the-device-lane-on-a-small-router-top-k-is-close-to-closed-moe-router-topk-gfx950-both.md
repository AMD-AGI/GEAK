---
key: small MoE router top-k + softmax + bitmatrix pack on gfx950 Triton - which device-side reformulations were already measured out on this op shape
type: anti-pattern
confidence: ★★
effect: ~1.00x across eight funded device directions: selection top-k 1.54-1.62x against a 1.80x incumbent, bitmatrix pack +1.6% standalone and null-to-negative stacked (three attempts), topk-network reformulation +0.24% vs a same-session control, per-launch BLOCK_M 0.0%, whole-op rewrite nothing shippable; per case the largest shape moved least
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both
description: Device-side rewrites of a small MoE router top-k (selection topk, pack, BLOCK_M, whole-op rewrite) all returned ~1.00x on gfx950; the win is host-side
keywords: ['anti-pattern', 'moe-router', 'topk', 'dispatch-bound', 'static-isa-screen', 'launch-overhead', 'triton', 'gfx950']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: moe_router_topk
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
origin_kernels: ['_topk_forward']
---
# The device lane on a small router top-k is close to closed
- lever: screen device candidates with a static v_* count per region before benchmarking, and spend the freed rounds on dispatch; on this op shape the merge-tree top-k networks were ~61% of static VALU and the bitmatrix pack ~5%.
- apply: attribute static VALU by region first; a candidate that can only touch a 5% region is capped near 1% of the largest case, which is inside what a 3-case geomean can resolve.
- verify: give every device candidate a same-session control arm; a candidate within ~0.5% of its control is null however good its ISA looks.
- pitfall: three patches merged with zero textual conflict and every stack lost -> all of them targeted the same bottleneck resource -> judge orthogonality by bottleneck, not by whether the patches apply cleanly.
- pitfall: the static screen under-predicted a mask-free specialization -> predicate/exec-mask setup and the early-exit branch are invisible to the histogram -> that arm carried more static VALU and still won ~0.9% at the largest case.
- caution: also verify whether the smallest case is host-issue bound before chasing its occupancy - here it contributed exactly 1.0000 to the unweighted geomean three times, which caps every device win at ~2/3 of its per-case size.
- source: run kernel_20_geak_0808_16h lane _topk_forward, rounds 2-6, 2026-08-12, tech_lead_report.md
