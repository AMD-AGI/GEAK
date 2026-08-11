---
name: mark-write-once-output-stores-non-temporal-when-the-store-pi-linear-attention-gfx950-memory-bound
description: Mark write-once output stores non-temporal when the store pipe is the roofline: +6.4% bit-identical on the largest case, cumulative 15.09 -> 15.36x
keywords: [non-temporal, l2-locality, isa-check, memory-bound, linear-attention, roofline, interleaved-ab]
kernels: [chunk_scaled_dot_kkt_fwd_kernel]
platforms: [gfx950]
kernel_class: linear_attention
regime: memory-bound
key: write-once output stores of a Triton chunked linear-attention kernel on gfx950, largest case HBM-store-bandwidth-bound with stores already at full dword width
lifecycle: active
type: lever
confidence: ★★
effect: A write-combining cache_modifier ('.cs', lowering to buffer_store_dwordx4 ... sc0 nt) on the two output stores took the largest case +6.4% (store bandwidth +6%, reaching ~80-97% of the achievable HBM store roofline) and the cumulative geomean 15.09 -> 15.36x, bit-identical (cos=1.0), rock-stable across 8+ runs. The stores were ALREADY dwordx4, so this was pure cache policy, not vectorization. End state 15.36x banked, per-case 1.56x / 27.14x / 86.0x on the small / mid / large case (the campaign row for this kernel reports 16.61x, its max single-pass geomean).
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.50h / 48 passes, 2026-08-11
last_seen: 2026-08-11
---
# Mark write-once output stores non-temporal when the store pipe is the roofline
- lever: When the output buffer is written once and never re-read, and the profile says the case is HBM-store-bandwidth-bound with stores already at the widest dword width, the remaining headroom is store cache policy: a write-combining / non-temporal cache_modifier on the store bypasses L2 write-allocate (read-for-ownership) and stops the output polluting L2.
- apply: One-line cache_modifier on the store ('.wt' measured equivalent here); the eviction_policy argument is a different knob and moved nothing on its own.
- verify: Check the ISA actually gained the nt/sc0 bits, and confirm the output is bit-identical rather than merely within tolerance.
- caution: Also verify this cheap axis before funding further tiling or byte-reduction work - it is a one-line, bit-identical edit, so it prices the store-policy ceiling for almost nothing.
- source: chuschen 16h time-budget campaign run, 15.50h / 48 passes, 2026-08-11
