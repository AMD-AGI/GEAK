---
key: fp8 KV-cache storage with bf16 compute in a paged decode attention device kernel on gfx950/CDNA4
type: lever
confidence: ★★
effect: 1.14x on top of the host-optimized state = 1.053x (fp8 KV storage) x 1.057x (occupancy-4 bounds) x 1.019x (non-temporal loads); per-case the whole gain lands on the longest-context case, while the short-context cases stay byte-identical because they are host-bound.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-11
name: fp8-kv-storage-with-in-register-bf16-dequant-then-re-tune-oc-attention-decode-gfx950-decode
description: fp8 KV storage + in-register bf16 dequant on decode attention, stacked with occupancy-4 bounds and non-temporal loads: 1.14x on top of the host-optimized state
keywords: ['decode', 'attention', 'fp8', 'kv-cache', 'occupancy', 'non-temporal-loads', 'dequant']
kernels: ['paged_attention_decode']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
---
# fp8 KV storage with in-register bf16 dequant, then re-tune occupancy and non-temporal loads
- lever: Store KV in fp8 and dequantize to bf16 in-register — the byte saving lives on the HBM read while compute stays bf16 — then re-tune occupancy and re-test non-temporal loads under the new storage dtype.
- apply: Conditional launch bounds for the fp8 path only, so the bf16 path keeps its own tuning; gate the non-temporal/streaming load flag on the storage dtype rather than on partition size alone.
- stack: total 1.14x over the host-optimized state = three directions compounded
  - 1. fp8 KV storage + in-register bf16 dequant — 1.053x standalone (verified)
  - 2. conditional occupancy-4 launch bounds on the fp8 path — +5.7% on top of (1); occupancy 4 was the ceiling
  - 3. non-temporal KV loads on the long-context case — +1.9% on top of (1,2)
  - note: attribution is incremental in landing order, not independent.
- verify: Keep the parity/SNR gate pinned across the change, and count AGPRs as well as VGPRs on this arch to confirm the occupancy change engaged with no spill.
- pitfall: fp8 numerics collapsed far below the parity gate -> the culprit was casting the softmax probabilities to fp8 (subnormal floor), not KV storage -> keep the probabilities in bf16 and quantize only the stored KV.
- caution: A storage-dtype change invalidates prior occupancy tuning and can flip the sign of the non-temporal-vs-L2-residency tradeoff — the same flag measured -7% under bf16 storage and +1.9% under fp8 — so also re-measure both after any dtype change. Narrowing further (finer scale granularity, fp6) returned nothing here, and the harness data distribution is worth checking first: uniform inputs make per-block scales byte-identical to per-tensor.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11
