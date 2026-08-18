---
key: int4 weight-only (W4A16) MoE grouped GEMM in Triton on gfx950, dequant-dominated body, batch swept across small and large M buckets
type: lever
confidence: ★★
effect: 3.33x weighted geomean vs frozen baseline, non-overlapping; per-case 2.58x on the small-M case, 3.68x and 3.89x on the two larger-M cases; parity clean on all 8 cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 15
toolchain: unknown
last_seen: 2026-08-12
name: per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound
description: Per-M-bucket host-side launch-config retune on int4 W4A16 MoE grouped GEMM: 3.33x weighted, per-case 2.58-3.89x, kernel body byte-identical
keywords: ['moe', 'grouped-gemm', 'int4', 'weight-only-quant', 'w4a16', 'launch-config', 'host-tuning', 'm-bucket', 'num-warps', 'block-size-k', 'compute-bound']
kernels: ['fused_moe_int4_w4a16']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-07-29
roofline: compute-bound 0.26 -> compute-bound 0.98 of its own empirical roof
---
# Per-M-bucket launch config on an int4 weight-only grouped GEMM
- lever: When the kernel body is M-independent (cost dominated by int4 dequant per output tile), stop tuning one global launch config and pick a separate config per M bucket: tile M x N, BLOCK_K, group-size-M, num_warps, num_stages.
- apply: Host/wrapper-only edit: bucket M at dispatch and select a per-bucket config dict; the compiled kernel source can stay byte-identical to the golden, which keeps parity risk near zero.
- stack: total 3.33x cumulative = two non-overlapping host-side patches. 1. per-M-bucket config retune - 3.26x standalone (verified), carries essentially the whole win. 2. second host patch integrated on top - 3.31x (verified), i.e. a small increment. Attribution is incremental in landing order; 12 later kernel-side directions added exactly 0.
- verify: Frozen-baseline isolated A/B per case (not just the geomean) plus oracle parity on every case; confirm each bucket actually resolves to its own config, since a single mis-bucketed dispatch hides the whole effect.
- pitfall: Buckets disagreed on group-size-M (small-M kept a larger value than the big cases) -> a single global config underserves one end of the M range -> tune each bucket independently rather than averaging.
- caution: An algorithmic variant that de-interleaves the packed weights to drop a K-reorder shuffle measured only ~1.03x and was incompatible with the winning large tile - also verify a body-level idea composes with the incumbent tile before spending a round on it.
- source: run fused_moe_int4_w4a16-ch16h (16h per-kernel time-budget campaign, 2026-07-28/29), 49 resumed passes
