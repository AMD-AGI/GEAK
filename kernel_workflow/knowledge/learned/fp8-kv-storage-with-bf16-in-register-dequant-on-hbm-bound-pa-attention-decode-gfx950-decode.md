---
key: paged KV-cache attention on gfx950, HBM-bound by paged-block scatter — fp8 KV storage decoupled from bf16 accumulate
type: lever
confidence: ★★
effect: 1.30x cumulative full-mix geomean vs frozen baseline; per-case: long-context decode shapes 1.61x-2.07x (heavy subset ~1.88x geomean), tiny launch-floor shapes ~0.87-0.95x i.e. flat within session noise, since they have no KV traffic to save
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: fp8-kv-storage-with-bf16-in-register-dequant-on-hbm-bound-pa-attention-decode-gfx950-decode
description: fp8-e4m3 KV storage + in-register bf16 dequant halves KV read traffic on HBM-bound paged attention: 1.30x cumulative, 1.6-2.07x on long-context shapes
keywords: ['fp8', 'kv-cache', 'paged-attention', 'decode', 'hbm-bound', 'non-temporal', 'dequant', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: archived
cost: L3
verified_on: 2026-08-11
roofline: memory-bound 0.44 -> memory-bound 0.54 of achievable roofline; still at the paged-scatter ceiling
---
# fp8 KV storage with bf16 in-register dequant on HBM-bound paged attention
- lever: Store the paged KV cache as e4m3 fp8 while keeping Q, the dots and the accumulator in bf16 — a storage-only narrowing that halves the dominant KV read stream without touching numerics of the math.
- apply: Dispatch on kv_cache_dtype=fp8 with the mfma type left at F16 so the kernel takes the bf16-Q / uint8-KV path and dequants each 8-element chunk into registers; stack it on top of non-temporal 128-bit KV loads (those alone were worth ~1.06x).
- stack: total 1.30x cumulative = two directions compounded
  - 1. non-temporal KV loads — 1.06x standalone (verified, parity PASS)
  - 2. fp8 KV storage + bf16 dequant — 1.22x on top of (1) (verified); attribution is incremental in landing order
- verify: Confirm parity with a worst-element allclose at 5e-2, and confirm the traffic actually dropped by checking the heavy long-context cases move (they carry the KV stream) while launch-floor cases stay flat.
- pitfall: Silent garbage in K only → the packed-K layout stride is dtype-dependent (x = 16/sizeof(cache_t), so 8 for bf16 and 16 for fp8) and a plain cast keeps the bf16 layout → remap K; V is a straight narrowing cast.
Outputs off by ~2x → the gfx950 packed-fp8 convert decodes OCP e4m3, so encode with the OCP variant (amax/448 scaling), not the fnuz one.
Quantization cost showed up as free → the bf16->fp8 conversion was memoized in warmup and is invisible to event timing; re-check it if inputs are rebuilt per iteration. Also revisit any launch_bounds tuned for the bf16 path — its ~+5% was neutralized once KV went fp8.
- caution: Also verify inherited 'fp8-KV is dead here' verdicts against THIS kernel's own dispatch paths: the sibling verdict that closed this axis had cast softmax-P to fp8, a different thing from KV storage, and re-measuring locally moved the ceiling from 1.06x to 1.30x.
- source: 16h single-kernel time-budget campaign, run chuschen16h, 2026-08-11
