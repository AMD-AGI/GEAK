---
key: paged bf16 KV attention decode in Triton on gfx950, where each KV tile is loaded exactly once per workgroup
type: lever
confidence: ★★
effect: +8.2% geomean isolated vs frozen baseline, bit-identical; per-case +15% and +8% on the two bandwidth-bound cases, ~0% on the latency-floored small-batch case
confirms_cited: 2
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-17
name: drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode
description: The non-temporal KV hint is a per-call decision: predicate it on KV element width and tile-loop trip count; a blanket drop or keep each loses a case.
keywords: ['attention', 'cache-modifier', 'decode', 'fp8-kv', 'kv-cache', 'memory-bound', 'paged-attention', 'size-gating', 'triton']
kernels: ['kernel_unified_attention_2d', 'kernel_unified_attention_3d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-07-29
origin_kernels: ['kernel_unified_attention_2d', 'mi355x_vllm_triton_unified_attention']
---
# Drop the non-temporal cache hint on once-read KV streams
- lever: On the K/V tile loads inside the KV loop, try clearing the '.cg' cache_modifier so the streaming reads go through the normal cache path instead of the non-temporal one.
- apply: One argument on the tl.load of the K/V tiles; no math changes, so the output stays bit-identical and parity is free.
- verify: Interleaved min/median A/B against the frozen baseline, per case: expect the gain on the bandwidth-bound long-context cases and roughly nothing on a latency-floored small-batch case.
- pitfall: The geomean read as a uniform win -> it is entirely carried by the two bandwidth-bound cases -> attribute per case before claiming the lever transfers to short contexts.
- caution: Also verify the tiles really are read once under your blocking; a tile re-read across iterations may genuinely prefer the streaming hint.
- source: run kernel_unified_attention_2d-ch16h, 2026-07-29, 16h time-budget campaign, direction r1_d0_cache_modifier

- source: GEAK 12h per-kernel time-budget campaign, run mi355x_vllm_triton_unified_attention-bmk7-12h, 2026-08-17, rounds 3/4/14/15, director-validated accepted, correctness PASS
