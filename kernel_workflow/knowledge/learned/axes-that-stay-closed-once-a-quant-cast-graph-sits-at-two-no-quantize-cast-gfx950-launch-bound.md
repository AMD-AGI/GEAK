---
key: deciding whether another fusion, cache-policy or launch-parameter round is worth funding on a small two-node graph-captured fp8 quantize/cast at gfx950, where the node cost itself is measured-irreducible
type: anti-pattern
confidence: ★★
effect: ~1.00x across five attacks on unconditional 2->1 node fusion and four on per-node cost; the marginal dependent node cost is invariant across 38 configurations and 9 factors; (VEC,BS) closed to ±0.2% geomean over 9 grid points x 2 sessions with identity controls; non-temporal on the consumer's scratch read costs +38%; a ground-up single-node rewrite read 0.5085x with its worst case +426%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound
description: On a 2-node graph-captured fp8 quant cast, unconditional fusion, per-node cost knobs, cache policy and (VEC,BS) all returned ~1.00x; price the node first
keywords: ['closed-axis', 'anti-pattern', 'hip-graph', 'dispatch-floor', 'launch-overhead', 'quantize-cast', 'cache-modifier', 'cross-workgroup', 'atomics', 'block-size', 'gfx950']
kernels: ['data_to_scale_kernel', 'scaled_quant_kernel']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: launch-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-14
---
# Axes that stay closed once a quant-cast graph sits at two nodes
- lever: before funding fusion, cache-policy or launch-parameter rounds on a small graph-captured op, price the graph node itself: a node dependency here is simultaneously the cheapest grid-wide sync available and an irreducible floor, and one cheap measurement decides both axes
- apply: sweep the marginal cost of one dependent node across the factors you would otherwise tune — kernarg size, pointer count, LDS, grid, block, launch bounds, VGPR, node identity, node ordering; if the cost does not move, the remaining headroom is the node count, not the node, and launch-bounds annotations price as a small net cost rather than a win
- verify: compute the zero-work ceiling — the wall divided by what would remain with all device work removed at the current dispatch shape; here that ceiling was 2.30x against 1.71x achieved and the node constant was 83.6% of the largest case's wall, which is what justified stopping
- pitfall: grid barrier, u64 atomicMax handoff and a ticket protocol all lost → a software grid-wide rendezvous costs roughly 9x a graph-node dependency, and a u64 atomic serialises across the 8 XCDs (+19-21% at 112-128 blocks) → only a shape-conditional fusion needing no cross-block sync survived
a ground-up rewrite had every block re-read the whole tensor → redundant read volume scales as nblocks x tensor bytes while a node cost is flat → unprofitable at every size, and its mechanism diagnosis (amplification, not fusion) is what later reopened the axis in conditional form
non-temporal on the handoff scratch read looked symmetric with the winning non-temporal output store → it destroys the L2/MALL residency the handoff depends on → keep policy bits off streams that something re-reads
- caution: also verify the closure against your own footprint: the fused path's working set never pressured a cache here, so no policy bit could pay, and also verify any re-swept knob grid against a byte-identical same-session control arm before calling it closed
- source: run kernel_20_geak_0811_2h_bmk7_long lane, 2026-08-14, rounds 2-7 and 11, TechLead report + director validation (accepted)
