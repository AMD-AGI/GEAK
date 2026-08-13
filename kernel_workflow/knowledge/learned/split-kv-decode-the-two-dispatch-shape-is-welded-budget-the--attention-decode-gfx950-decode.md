---
key: removing the second dispatch of a split-KV decode attention (reduce folded into the main kernel) on gfx950/CDNA4 with ~1024 co-resident blocks
type: anti-pattern
confidence: ★★
effect: ~0.14x (about a 7x regression) on the large-ctx decode case for the fenced variant; the fence-free atomic+inline-reduce variant was still net-negative vs the dispatch it removes, so ~1.00x is the ceiling of the whole axis
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode
description: Fusing the split-KV reduce into the attention epilogue is a closed axis on gfx950: the cross-block fence L2-serializes the grid for a ~7x regression.
keywords: ['attention-decode', 'paged-kv', 'split-kv', 'kernel-fusion', 'threadfence', 'dispatch-overhead', 'anti-pattern', 'gfx950']
kernels: []
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
levers: ['host.dispatch-count', 'algo.fusion']
---
# Split-KV decode: the two-dispatch shape is welded, budget the round elsewhere
- lever: Before planning a fusion round, price the second dispatch as a fraction of the case it sits in: here the reduce dispatch is almost entirely launch overhead and moves a tiny fraction of the bytes the main kernel does, so even perfect removal caps the axis in the low single-digit percent.
- apply: The fused form needs cross-block visibility of partial outputs over ~1024 co-resident blocks; the device-scope fence that provides it forces an L2 writeback that serializes the grid. The fence-free alternative (atomic counter + last-block inline reduce) costs more than the dispatch it deletes.
- verify: Time the fused candidate against the frozen baseline per case rather than trusting the removed-dispatch arithmetic, and check the ISA for the emitted fence.
- pitfall: Fused candidate looked structurally strictly better (one fewer dispatch) yet regressed ~7x -> the agent-scope fence serialized ~1024 blocks through L2 writeback -> no fix found; the two-dispatch split stands.
- caution: Also verify block co-residency before generalizing: at a much smaller grid the fence cost is not the same, so this closure is about the wide-grid decode shape and not about fusion in general.
- source: run paged_attention_decode-ch16h, 16h time-scaling campaign, 2026-08-12
