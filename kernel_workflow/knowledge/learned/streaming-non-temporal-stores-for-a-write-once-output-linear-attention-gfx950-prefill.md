---
key: store-bandwidth-bound Triton op on gfx950 whose fp32 output buffer is written once and never re-read inside the launch
type: lever
confidence: ★★
effect: +6.4% on the largest per-case shape (store-bandwidth-bound), cumulative 15.09x -> 15.36x; ~80-97% of achievable HBM store roofline after; smaller cases flat (they sit on the harness event-timing floor); bit-identical, cos=1.0
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-11
name: streaming-non-temporal-stores-for-a-write-once-output-linear-attention-gfx950-prefill
description: Write-combining cache modifier on write-once fp32 output stores bypasses L2 write-allocate on gfx950: +6.4% on the store-bound case, bit-identical
keywords: ['non-temporal-store', 'cache-modifier', 'store-bandwidth', 'l2-write-allocate', 'linear-attention', 'memory-bound']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: prefill
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-11
roofline: store-bandwidth-bound both sides; store utilization ~0.85 -> ~0.9+ of the achievable store roof
---
# streaming (non-temporal) stores for a write-once output
- lever: for an output written once and never re-read, tag the stores with a write-combining / non-temporal cache modifier ('.cs', '.wt' in Triton) so they lower to a streaming buffer_store with the nt bit set and skip L2 write-allocate / read-for-ownership
- apply: one argument on the store calls; no layout, dtype or vectorization change — where the stores are already the widest dword form this is a pure cache-POLICY win, not a vectorization one
- verify: inspect the lowered store for the nt bit, confirm bit-identical output, and A/B the store-bound case on its own, since a geomean over tiny cases hides a single-digit percent move
- pitfall: the eviction_policy argument alone changed nothing -> it does not select write-combining -> the cache_modifier argument is the one that engages the lever
- caution: holds only where the buffer really is write-once for the launch; also verify nothing re-reads it later in the same kernel, and re-check parity bit-for-bit rather than by tolerance
- source: 16h single-kernel time-budget campaign (48 passes), 2026-08-11; deep_explore direction, status verified, stable across 8+ repeats
