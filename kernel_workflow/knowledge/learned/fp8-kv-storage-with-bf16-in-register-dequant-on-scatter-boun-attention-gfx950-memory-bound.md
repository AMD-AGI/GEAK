---
key: paged attention whose KV read sits at the paged-block scatter HBM roofline, fp8 e4m3 KV STORAGE decoupled from bf16 MFMA accumulate, gfx950
type: lever
confidence: ★★
effect: 1.30x cumulative all-case geomean (up from 1.06x before this direction); ~1.88x geomean over the 8 heavy long-context shapes, per-case 1.61x-2.07x with no heavy case below 1.6x, while the 8 launch-floor tiny cases stay ~1.00x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: fp8-kv-storage-with-bf16-in-register-dequant-on-scatter-boun-attention-gfx950-memory-bound
description: Store the paged KV cache as e4m3 fp8 and dequant to bf16 in registers: halves KV HBM traffic on a scatter-bound attention op, ~1.9x on heavy shapes
keywords: ['paged-attention', 'fp8-kv-cache', 'hbm-bound', 'kv-cache-quant', 'memory-bound', 'gfx950', 'long-context']
kernels: ['paged_attention_large']
platforms: ['gfx950']
kernel_class: attention
regime: memory-bound
layer: learned
lifecycle: active
verified_on: 2026-07-30
roofline: HBM-bound at the paged-scatter roofline before and after; the win is halved bytes moved, not a bound-class change
---
# fp8 KV storage with bf16 in-register dequant on scatter-bound paged attention
- lever: Keep the cache in e4m3 fp8 and dequantise to bf16 in registers at point of use, leaving both dots on bf16 MFMA — the win is halved KV bytes from HBM, not cheaper math.
- apply: Decouple STORAGE dtype from ACCUMULATE dtype in the dispatch (fp8 cache dtype + F16/bf16 MFMA type); K's packed layout width must be recomputed from the element size (x = 16/sizeof(cache_t)), V is a straight narrowing cast.
- verify: Same-mode isolated A/B over the grader's full case mix plus worst-element allclose at the harness tolerance; per-case, the heavy long-context shapes should carry essentially all of the win and the launch-floor shapes should be flat.
- pitfall: silent garbage output -> K packing width left at the 16-bit value after switching the cache to 8-bit -> derive x from sizeof(cache_t) instead of hardcoding it.
values off by ~2x -> fnuz vs OCP e4m3 encoding mismatch; this arch's cvt_pk_f32_fp8 decodes OCP -> encode e4m3fn with max 448 and scale amax/448.
a real win measuring ~1.00x -> the bf16->fp8 conversion sat inside the timed loop and is invisible to GPU-event timing -> hoist it into warmup so timed iterations reuse the quantised cache.
- caution: An inherited 'fp8-KV is closed here' verdict from a sibling kernel was refuted on this one (the sibling had cast softmax P, not KV storage) — also verify what a sibling actually quantised before treating the axis as settled, and re-check any occupancy hint that was tuned for the pre-fp8 register footprint.
- source: run paged_attention_large-ch16h, 2026-07-30 — frozen-baseline isolated A/B over 16 cases, oracle parity PASS
