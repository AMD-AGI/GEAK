---
key: paged KV-cache attention decode with a split-KV two-dispatch wrapper, fp8-capable KV store, on gfx950/CDNA4
type: lever
confidence: ★★
effect: 3.18x geomean isolated vs frozen baseline; per case 3.63x (small ctx) / 3.13x (mid ctx) / 2.84x (large ctx decode), non-overlapping
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode
description: Paged-KV attention decode on gfx950: host alloc/scale hoist first, then fp8 KV storage with bf16 in-register dequant, then occupancy + NT loads; 3.18x stacked.
keywords: ['attention-decode', 'paged-kv', 'fp8-kv', 'host-overhead', 'non-temporal-loads', 'occupancy', 'launch-bounds', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
roofline: large-ctx case moves from an apparent ~0.5 of achievable HBM roof to VALU+LDS bound
levers: ['host.alloc-cache', 'mem.storage-dtype', 'mem.non-temporal', 'compute.occupancy']
origin_kernels: ['paged_attention_decode']
---
# Decode attention: pay the host tax first, then halve KV bytes, then re-tune occupancy
- lever: Order the round set host-wrapper overhead -> HBM byte count -> occupancy/load policy: a per-call allocation + scale-recompute cache in the python wrapper was worth ~2.08x alone, far more than any in-kernel axis that followed.
- apply: Cache the scratch allocations and hoist scale computation out of the per-call wrapper (a 'frozen baseline' wrapper may still be in scope); switch KV STORAGE to fp8 while dequantizing to bf16 in-register (keep math bf16); add conditional __launch_bounds__ on the fp8 path only; gate non-temporal/streaming KV loads on (small partition) OR (KV dtype != auto).
- stack: total 3.18x isolated (weighted geomean, frozen baseline) = four directions compounded
  - 1. host alloc/scale cache - 2.08x standalone - carries the bulk of the win
  - 2. fp8 KV storage + bf16 in-register dequant - +5.3% on top of (1)
  - 3. conditional launch_bounds occupancy bump on the fp8 path - +5.7% on top of (1,2), a ceiling
  - 4. non-temporal KV loads extended to the fp8 large-ctx case - +1.9% on top of (1,2,3)
  - note: attribution is incremental in landing order, not independent; (4) measured -7% before (2) landed.
- verify: Re-time every direction per case against the frozen baseline (small/mid/large ctx), read the ISA for VGPR+AGPR counts and absence of spill (count AGPRs separately on this arch), and confirm the correctness SNR gate did not move.
- pitfall: Storage-dtype change silently invalidated occupancy tuning -> register footprint dropped and the auto-selected occupancy moved -> re-tune launch bounds after any dtype change.
A storage-dtype change also INVERTED the non-temporal-vs-L2-residency tradeoff (-7% at the wider dtype, +1.9% once the set halved and stopped being L2-resident) -> re-measure NT loads after every byte-count change.
The build cache was keyed on a compile-param hash and returned a stale binary -> forced rebuild each iteration; and the harness improved-flag read false on genuinely improving rounds from small-case drift -> re-verify on the large-ctx median.
- caution: Also verify the harness data distribution before spending a round on scale granularity: uniform synthetic KV made every finer-granularity scale byte-identical to per-tensor, so the axis returned ~1.00x for distributional reasons, not architectural ones.
- source: run paged_attention_decode-ch16h, 16h time-scaling campaign, 2026-08-12
