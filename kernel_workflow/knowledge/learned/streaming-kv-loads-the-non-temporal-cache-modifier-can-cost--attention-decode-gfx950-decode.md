---
key: Triton paged/streaming-KV attention inner loop, bf16 KV cache, memory-bound decode on CDNA gfx950
type: lever
confidence: ★★
effect: +8.2% geomean on the frozen-baseline isolated A/B; per-case c32 +15%, c64 +8%, small-batch c2 ~wash (latency-floored, not bandwidth-limited); outputs bit-identical so parity is free
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-11
name: streaming-kv-loads-the-non-temporal-cache-modifier-can-cost--attention-decode-gfx950-decode
description: Dropping the '.cg' non-temporal modifier on once-read KV loads in paged attention: +8.2% geomean, bit-identical, on memory-bound cases only
keywords: ['attention', 'decode', 'paged-kv', 'cache-modifier', 'memory-bound', 'streaming-loads', 'triton']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
---
# Streaming KV loads: the non-temporal cache modifier can cost bandwidth
- lever: On a once-read streaming KV tile load, try clearing the non-temporal/'.cg' cache modifier; the nt bit can throttle achieved read bandwidth on CDNA even though the data truly is read once.
- apply: L1 edit inside the KV load in the attention inner loop: cache_modifier '.cg' -> '' (default). No host or numerics change; load width stays at the widest dwordx4 form.
- verify: Interleaved min/median A/B with >=8 reps per case against the frozen baseline, plus a bitwise output compare — the change is value-preserving, so any output delta means something else moved.
- pitfall: The win looked like noise under an average-of-100 harness -> the averaging harness is noisier than an interleaved A/B and washed the mid-batch case -> re-measured interleaved, where the largest-batch case was the most robust.
- caution: Also verify the case is actually bandwidth-limited before spending a round: on a latency-floored small-batch case the same edit measured as a wash, and on cases with heavier address arithmetic the gain can be eaten by register pressure.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 51 passes, 2026-08-11; ledger direction r1_d0_cache_modifier, verdict confirmed
