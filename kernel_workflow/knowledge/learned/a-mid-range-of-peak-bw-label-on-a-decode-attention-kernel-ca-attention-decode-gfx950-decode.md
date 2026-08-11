---
key: split-KV paged decode attention on gfx950/CDNA4 that profiles as memory-bound but is really VALU+LDS-bound with its loads front-loaded
type: anti-pattern
confidence: ★★
effect: Two full deep-explore rounds returned 0 gain: the double-buffered streaming rewrite measured 1.00x against a 1.2x expectation on the only non-byte-identical case, and split-KV reduce fusion measured a ~7x regression on that same case. Counters on it read VALU:VMEM 34:1 with LDS-wait above busy cycles, and that case can contribute at most ~1.09x geomean even at 90% of achievable peak.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-11
name: a-mid-range-of-peak-bw-label-on-a-decode-attention-kernel-ca-attention-decode-gfx950-decode
description: Decode attention reading as memory-bound was VALU+LDS-bound with loads already in flight: streaming/prefetch and reduce-fusion rounds both returned zero
keywords: ['decode', 'attention', 'roofline', 'double-buffer', 'prefetch', 'reduce-fusion', 'anti-pattern', 'hardware-counters']
kernels: ['paged_attention_decode']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: archived
---
# A mid-range %-of-peak-BW label on a decode attention kernel can be a compute-tail artifact
- lever: Before planning a load-streaming / prefetch / double-buffer round on a decode attention kernel, read the VALU:VMEM instruction ratio and LDS-wait vs busy cycles, and treat a mid-range %-of-peak-BW figure as unclassified rather than as headroom.
- apply: Check whether the K and V tiles already fit in the per-wave register footprint at the achieved occupancy: if they do, the loads are front-loaded and in flight, and a second buffer has nothing to hide — the BW fraction is a compute tail (dequant, LDS transpose, softmax) during which HBM sits idle.
- verify: Bound the prize before spending the round: compute what the one case that can move contributes to the weighted geomean at 90% of achievable peak, and compare that to the target the direction was issued with.
- pitfall: Fusing the split-KV reduce into the main-kernel epilogue needed a device-scope fence for cross-block partial-output visibility across ~1024 co-resident blocks -> the fence serialized the grid through L2 writeback -> ~7x regression, and the fence-free atomic inline reduce still cost more than the extra dispatch it removed.
- caution: This closure was measured with the loads already fully in flight at occupancy ~5 and a two-dispatch split-KV structure — also re-check the counter ratio on your own shapes, since a lower-occupancy or much longer-context configuration can re-expose the loads and reopen the axis.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11
