---
key: int4-weight / 16-bit-activation fused-MoE grouped GEMM on gfx950, Triton — the win lives in the host wrapper's per-M-bucket launch config, not in the kernel body
type: lever
confidence: ★★
effect: 3.33x cumulative isolated geomean vs frozen baseline, parity-clean on all 8 cases; per-case 2.58x on the small-M case and 3.68x / 3.89x on the two large-M cases; roofline fraction 0.26 -> 0.98 of its own ceiling, compute-bound both sides
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: per-m-bucket-host-launch-config-int4-moe-gemm-moe-grouped-gemm-gfx950-both
description: Per-M-bucket host launch-config retune on int4-weight MoE grouped GEMM: 3.33x isolated with the JIT body left byte-identical.
keywords: ['moe-grouped-gemm', 'int4-dequant', 'w4a16', 'launch-config', 'host-tuning', 'per-m-bucket', 'block-size', 'group-size-m']
kernels: ['fused_moe_int4_w4a16']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
---
# per-m-bucket-host-launch-config-int4-moe-gemm
- lever: Retune the launch config per M-bucket in the host wrapper (block sizes, group-size-M, num_warps, num_stages) and leave the compiled body untouched; when the dequant inner loop is M-independent, one config per bucket captures most of the available win.
- apply: L2, wrapper-only: a config table keyed by M-bucket; two independent host patches stacked here without conflict, and the kernel source stayed byte-identical to the golden baseline.
- stack: total 3.33x from two non-overlapping host-side directions - 1. per-M-bucket launch-config retune, 3.26x standalone (carries essentially the whole win); 2. a second wrapper patch, +~2% on top of (1); attribution is incremental in landing order, not independent.
- verify: Frozen-baseline isolated A/B per case plus oracle parity on every case, then confirm the config actually engaged by diffing the emitted launch arguments - a host-only patch can silently no-op and still look neutral.
- pitfall: An inner de-interleave rewrite that removed the int4 K-reorder shuffle measured only ~1.03x -> it was mutually exclusive with the large-tile config that carried the win -> evaluate an algorithmic tweak jointly with the config winner, not against the untuned baseline.
- caution: Also verify the small-M case on its own: it settled on a different group-size-M from the two large cases, so a single global config understates the achievable win.
- source: 16h per-kernel time-budget campaign, 49 resumed passes, run id in this proposal, 2026-08-11
