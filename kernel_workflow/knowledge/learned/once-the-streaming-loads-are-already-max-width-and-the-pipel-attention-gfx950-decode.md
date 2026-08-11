---
name: once-the-streaming-loads-are-already-max-width-and-the-pipel-attention-gfx950-decode
description: When ISA already shows max-width loads and the pipeliner overlaps them, close the load axis: cache hints and manual prefetch cost 3-20% here
keywords: [pipeline-stages, l2-locality, isa-check, decode, attention, memory-bound, counters, roofline, operand-reuse]
kernels: [_fwd_grouped_kernel_stage1]
platforms: [gfx950]
kernel_class: attention
regime: decode
key: cache hints, manual prefetch and loop splitting in the KV streaming loop of a Triton attention decode kernel on gfx950 that already emits 128-bit vector loads
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Whole memory axis returned nothing on a latency-bound streaming loop (SQ_WAIT_ANY/SQ_BUSY ~3.2, ~47% of nameplate HBM bandwidth on the largest case): cache_modifier '.cs' is a hard compile error on this Triton, '.cg' made the largest case 12-20% worse (K +20%, V +12%, all-.cg in the same band) by adding an L2 round-trip, a loop-carried manual double-buffer prefetch of the page index was +3% worse, and splitting the loop cost 4-5%. ISA showed the loads were already buffer_load_dwordx4 (128-bit) x10. The only thing this axis yielded was a bit-identical +2% scalar page-load (1.0127x).
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 2026-08-11
last_seen: 2026-08-11
---
# Once the streaming loads are already max-width and the pipeliner is doing the overlap, the load region is closed
- lever: Before spending rounds on cache hints, manual prefetch or loop restructuring in a KV/streaming region, check two facts: whether the generated ISA already emits max-width vector loads, and whether the backend pipeliner (num_stages>1) is already overlapping the address chain with the load. If both hold, the region is at its on-box ceiling and every remaining edit in it competes with the pipeliner instead of helping it.
- verify: read the load width straight out of the ISA (a 128-bit buffer load per operand means the width axis is spent) and confirm any candidate did not displace the pipeliner's overlap before attributing its regression to something else.
- pitfall: every negative on this axis looked like a different bug -> all of them displaced the backend overlap: an explicit-caching hint pushed traffic through L2 and cost 12-20% on the largest case, and manual prefetch of a scalar index the pipeliner already hoists was pure overhead -> stop at the two-fact screen and spend the round elsewhere.
- caution: also verify which cache hint the local Triton even accepts before designing around one — one modifier here was a hard compile error, not a slow path.
- source: chuschen 16h time-budget campaign run, 2026-08-11
