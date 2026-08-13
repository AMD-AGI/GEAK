---
key: int4 w4a16 GPTQ/AWQ-style fused-MoE grouped GEMM with packed-nibble weights and per-group scales, Triton on gfx950/MI355 under a package power cap
type: lever
confidence: ★★
effect: 2.61x geomean isolated vs frozen baseline, non-overlapping (per-case 2.04x at M=2048, 2.93x at M=32768, 2.99x at M=65536; bit-exact vs golden; reproduced under a cold compiler cache)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both
description: Amortize (do not shrink) int4 weight dequant in a MoE grouped GEMM: one dequantized B tile per several M blocks, tails split not padded; 2.61x geomean
keywords: ['int4', 'w4a16', 'moe-grouped-gemm', 'dequant', 'amortization', 'triton', 'gfx950', 'bit-exact', 'tile-geometry']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
levers: ['compute.dequant-amortization', 'mem.nibble-split', 'host.two-specialisation-dispatch']
---
# Amortize int4 dequant across M-blocks instead of shrinking it
- lever: Make one dequantized weight tile serve more MFMA work: fuse several M blocks of the same expert behind one dequant pass, round a short expert tail UP into a padded merge, and express an odd rung as two accumulators (2+1) in one K loop rather than padding to the next power of two; on top of that a magic-number nibble->fp32 conversion that is bit-exact and removes every in-loop int->float convert.
- apply: Load the packed weight tile once as [BK/2, BN] uint8 and split low/high nibbles with constant masks (fold the 0.0625 into the high-nibble scale); dispatch on the host to a merged vs unmerged specialisation by an expanded/valid row ratio; re-sweep BLOCK_SIZE_K, num_warps, num_stages and matrix_instr_nonkdim after every body rewrite.
- stack: total 2.61x isolated (geomean, director-verified) = four groups compounded
  - 1. nibble split + launch re-tile — 1.72x cumulative; the split alone is 1.10x and the re-sweep on the merged body carries it (stacking the two bodies without re-tuning scored only 1.23x)
  - 2. magic-number dequant + nonkdim16 — 1.85x cumulative, bit-exact
  - 3. M-block fusion (+16.2%) and the run-length tail ladder (7.4% total across rungs)
  - 4. shader-engine phase-lock fix (+2.6 to +3.0%) and the 2+1 split rung (+3.0% at M=32768, +1.6% at M=65536)
  - note: attribution is incremental in landing order, not independent.
- verify: Expect the MFMA instruction count per output tile to stay identical while VALU drops (-61% at the fusion step) and the output stays bit-exact against the golden; re-time the smallest shape separately, since it gains least.
- pitfall: Two verified wins hand-merged, each keeping its own tuning constants, scored below the better one alone -> tuning metaparams are body-specific and expire on a body rewrite -> resolve every shared constant by re-measurement on the merged body.
- caution: Also verify that a skinny / low-expansion shape is routed to its own specialisation: merging there measured +41% to +271% because that shape wants more workgroups, not less operand traffic.
- source: run fused_moe_kernel_gptq_awq-own16h, 19-round campaign 2026-08-08/09, director-validated 2026-08-12 (bit-exact, cold-cache cross-check)
