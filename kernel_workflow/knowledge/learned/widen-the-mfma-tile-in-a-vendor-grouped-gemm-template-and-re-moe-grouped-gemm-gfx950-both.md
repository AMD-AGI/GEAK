---
key: fp8 per-block-scale grouped MoE second-stage GEMM on gfx950, vendor CK 2-stage codegen template, compute-bound
type: lever
confidence: ★★
effect: 1.25x isolated weighted vs frozen baseline; per-case 1.18x at the smallest token count, 1.29x and 1.30x at the two larger ones; bit-exact on every case
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 11
toolchain: unknown
last_seen: 2026-08-11
name: widen-the-mfma-tile-in-a-vendor-grouped-gemm-template-and-re-moe-grouped-gemm-gfx950-both
description: Widen MFMA tile 16x16->32x32 with a matching host B-preshuffle (then pad A-LDS) on fp8 per-block-scale grouped MoE GEMM: 1.25x isolated
keywords: ['mfma', 'moe', 'grouped-gemm', 'fp8', 'blockscale', 'lds-padding', 'bank-conflict', 'weight-preshuffle', 'composable-kernel']
kernels: ['moe_stage2']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
cost: L1
verified_on: 2026-08-11
roofline: compute-bound 0.24 -> compute-bound 0.31 of its own roof
---
# Widen the MFMA tile in a vendor grouped-GEMM template and re-preshuffle B to match
- lever: On the vendor 2-stage block-scale grouped-GEMM codegen template, raise the MFMA instruction tile from 16x16 to 32x32 on the M64/N128/K128 v1 instance and re-preshuffle the B weights on the host into the matching (32,16) layout; then de-conflict A-operand LDS reads with a one-element extra-M pad.
- apply: Two knobs in the codegen template header (MFMA size, ABlockLdsExtraM 0->1) plus the host weight-preshuffle argument, all in the same patch; keep the block tile equal to the expert-sort granularity.
- stack: total 1.25x isolated (weighted, verified) = two directions compounded
  - 1. compute.mfma-32x32 + matching host preshuffle — 1.264x standalone on the GEMM — carries the win
  - 2. mem.lds-extra-m pad — +1.4% on top of (1) — only pays once the wider MFMA makes A-LDS the conflicting reader
  - note: attribution is incremental in landing order, not independent.
- verify: Bit-exact against the frozen oracle plus a non-overlapping isolated A/B on all three token-count cases; grep the built object for the new tile marker, since the config can silently not engage.
- pitfall: Widened the MFMA tile while the host preshuffle stayed on the narrow pairing -> B fragments consumed in the wrong order -> parity failure; fixing the preshuffle in the same patch restored bit-exactness.
Raising the block tile above the expert-sort granularity built cleanly but let one tile span two expert groups -> wrong-expert weights, large error ratio; keeping block tile == sort granularity fixed it.
- caution: Also verify the epilogue store-vector width separately once the MFMA tile changes: widening it regressed monotonically here even though the same width paid off on the sibling first-stage kernel.
- source: GEAK per-kernel time-budget campaign, chuschen16h lane, 2026-08-11
