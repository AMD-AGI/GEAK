---
key: two-stage fp8 block-scale fused-MoE grouped GEMM on gfx950/CDNA4, CK templates behind a Python host router that can pick an instance per token bucket
type: method
confidence: ★★
effect: +1.44% over the pre-routing tree, reproduced five times; per-case the large tile is +1.4% at 65536 tokens but -2.2..-2.4% at 2048, and the epilogue-fusion threshold is +1.5% at 2048 and inert above it — shipped globally the pair nets ~1.00x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed
description: Shape-conditional wins that read as noise on a 3-case geomean cancel when shipped globally; routed per token bucket they add exactly.
keywords: ['bucket-routing', 'host-runtime', 'moe', 'grouped-gemm', 'tile-selection', 'sub-variance', 'fp8-blockscale']
kernels: ['fmoe_fp8_blockscale_g1u1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-08-09
---
# Route discarded sub-noise knobs per shape instead of shipping them globally
- lever: re-score candidates already discarded as sub-variance: a knob that helps one shape and hurts another is a routing-table entry, not a global default.
- apply: one host-side table keyed by token bucket selecting the instance and the fusion threshold; single-source each value so the host re-slicer and the device define cannot drift, and give each value its own compiled module name.
- verify: score each knob on its own bucket — a one-case move of ~1.5% dilutes to ~0.5% on a 3-case geomean against ~2% box variance — then re-measure the pairwise matrix and check additivity rather than assuming it.
- pitfall: a stacked patch measured slower than its best single part -> the fusion predicate was compile-time-only and therefore bucket-blind, cancelling the other direction's small-token win -> make availability compile-time and the taking runtime (one comparison on token count).
engineer and verifier disagreed by ~1% on the same tree -> one instance carries a 0.85% run-to-run spread against a 0.09% base -> settle sub-2% items with interleaved paired cycles, not one full run per arm.
- caution: also verify both halves of a host/device coupling are per-call: a knob whose partner (a weight preshuffle fixed once at startup) is not per-call is not per-bucket expressible, however well it scores.
- source: run kernel_20_geak_0808_16h, 2026-08-08..09, gfx950/MI355X; director-validated 1.41x geomean, correctness PASS
