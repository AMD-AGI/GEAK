---
key: choosing cache policy for the KV stream and the split-KV scratch round trip of a paged decode attention kernel on gfx950, across cache-resident small shapes and streaming large ones
type: method
confidence: ★★
effect: size-gated non-temporal KV loads gave +2.5% on the cache-resident small-batch cases with the large-context cases provably untouched (0.9997), and dropping the hint everywhere costs the large-context half 0.9644; the same class of hint on the split-KV scratch round trip measured an increment of 0.00 +/- 0.01, while deleting that stream outright was worth +18.6%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode
description: Non-temporal / cache-policy hints move a decode kernel only on read-once traffic no restructure can delete, and only when the working set exceeds cache.
keywords: ['decode', 'paged-attention', 'non-temporal-loads', 'cache-modifier', 'kv-cache', 'memory-bound', 'size-gating']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: archived
cost: L1
verified_on: 2026-08-12
---
# Ask whether the traffic is removable before you tune how it is cached
- lever: split the candidate streams into removable and irreducible first; spend the cache-policy direction only on the irreducible one, and host-select the hint per call on a working-set threshold via a template bool rather than baking it in
- apply: compute the working-set footprint on the host from the shape, pass a compile-time flag to the kernel, and let cache-resident shapes keep normal caching while streaming shapes bypass the last-level cache
- verify: carry an A/A null inside the same measurement, and ablate the hint off everywhere - if off-everywhere costs the streaming shapes, the hint is load-bearing where it applies; if the threshold sweep is byte-identical across a wide window, say so rather than reporting a tuned value
- pitfall: a cache-policy direction on the scratch round trip returned a flat null twice in consecutive rounds -> that traffic was an artifact of the two-phase decomposition, not of the problem -> the restructure that deleted the stream landed in the same round and was worth an order of magnitude more; also verify a redundant-load reduction is not just L1 hits on one cacheline before costing it
- caution: also verify how the harness re-calls the op: replaying the same tensors makes small shapes cache-resident across calls, so a non-temporal hint discards residency the next call wants - the gate, not the hint, is what transfers
- source: run paged_attention_large-own16h, 2026-08-12, rounds 4/5/6/9, director-validated (accepted, correctness PASS)
