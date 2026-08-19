---
key: write-once fp32 output stores in a chunked linear-attention forward, HBM-store-bandwidth-bound large-batch case on gfx950 / Triton
type: lever
confidence: ★★
effect: +6.4% on the largest case (store-bandwidth-bound), stable across 8+ repeats and a clean A/B; ~80-97% of achievable HBM store roofline after; cumulative geomean 15.09x -> 15.36x, bit-identical output
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound
description: Non-temporal '.cs' cache_modifier on write-once output stores lifts store bandwidth on gfx950 store-bound kernels; a cache-policy win, not vectorization
keywords: ['cache-modifier', 'non-temporal-store', 'store-bandwidth', 'memory-bound', 'linear-attention', 'gfx950', 'roofline', 'triton']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: memory-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
roofline: store-byte-bound -> store-bandwidth-bound at ~0.8-0.97 of the achievable store roof
levers: ['mem.store-cache-policy']
origin_kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
---
# Streaming non-temporal store for write-once output
- lever: When the output tile is written once and never re-read, try a write-combining cache_modifier ('.cs', '.wt' equivalent) on the output store: on gfx950 it lowers to a non-temporal buffer store that skips the L2 write-allocate / read-for-ownership path.
- apply: One argument on the store call(s) in the Triton kernel; tiling, vector width and grid stay untouched, so it stacks on top of any store-byte reduction already banked.
- verify: Confirm the emitted ISA store carries the non-temporal bits (a config that silently did not engage looks identical to a null result), then A/B the store-bound case against the frozen baseline and check parity is bit-identical.
- pitfall: eviction_policy set alone moved nothing -> it does not touch the write-allocate path -> the write-combining cache_modifier is the part that lowers to a non-temporal store.
The whole-run improvement gate read false -> the net geomean move sat inside the small-case event-bracket noise band while the store-bound component was stable across repeats -> re-check the per-case component plus the ISA marker before discarding the round.
- caution: Also verify the stores are already at the widest vector width before crediting this lever, otherwise a plain vectorization gain gets attributed to cache policy; and also verify the buffer really is never re-read downstream.
- source: run chunk_scaled_dot_kkt_fwd_kernel-ch16h, 2026-08-12 (16h per-kernel time-budget campaign, 48 passes, deep_explore memory direction)
