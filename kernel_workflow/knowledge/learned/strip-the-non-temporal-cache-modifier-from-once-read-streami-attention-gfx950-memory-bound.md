---
name: strip-the-non-temporal-cache-modifier-from-once-read-streami-attention-gfx950-memory-bound
description: Strip an inherited non-temporal / L1-bypass cache modifier from once-read streaming loads: +8.2% geomean, bit-identical, +15% on the mid batch case
keywords: [cache-policy, streaming-operand, memory-bound, attention, decode, isa-check]
kernels: [paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: an inherited '.cg' cache-policy bit on once-read streaming KV loads in a memory-bound attention kernel on gfx950, close to its bandwidth roof
lifecycle: active
type: lever
confidence: ★★
effect: Dropping the '.cg' modifier on the streaming KV loads was bit-identical and worth +8.2% geomean on its own (+15% on the mid batch case, +8% on the largest); it is the largest win recorded in the run's direction ledger (the pass trace also shows a larger unattributed first-pass jump, 1.00 -> 1.49 geomean). Campaign end state is 1.72x max validated geomean over 51 passes / 15.63h, per-case 2.20x / 1.67x / 1.31x at batch 2 / 32 / 64, roofline-emp 0.210 -> 0.310, still classified memory-bound at both ends. The tail was thin: the last ~40 passes moved the max only 1.6816 -> 1.7185 (+2.2%), and the final resume round was logged at +0.8-1.15%.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.63h / 51 passes, 2026-08-11
last_seen: 2026-08-11
---
# Strip the non-temporal cache modifier from once-read streaming operand loads
- lever: Grep the load sites for a cache-policy/cache_modifier argument before touching tiles or scheduling. A '.cg'-style non-temporal / L1-bypass bit is often inherited from a variant where the operand was re-read, and on a streaming operand that is read exactly once it throttles achieved bandwidth instead of protecting cache.
- apply: Flipping it back to the default is a constexpr-level, bit-identical edit, so it can be A/B'd in one build; it pays most on the cases that are already closest to the bandwidth roof.
- verify: Confirm the edit is bit-identical on the output and read the ISA to see the cache-policy suffix disappear from the streaming loads, then rank on the cases nearest the roof.
- caution: Worth pricing the operand-width axis in the same round — also verify whether the load is already lowering to a 128-bit buffer_load, in which case that axis has nothing left. Also verify whether the store side matters at all: an output that is ~0.05% of traffic and written once washed out here.
- source: chuschen 16h time-budget campaign run, 15.63h / 51 passes, 2026-08-11
