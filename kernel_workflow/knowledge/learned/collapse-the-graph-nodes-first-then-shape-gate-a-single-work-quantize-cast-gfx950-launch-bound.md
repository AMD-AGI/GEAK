---
key: dynamic per-tensor fp8 quantize/cast on gfx950/MI355X, captured as a multi-kernel HIP graph whose wall is dominated by a flat per-dispatch-node cost
type: lever
confidence: ★★
effect: 1.71x unweighted geomean isolated vs frozen baseline, director-verified at 1.7112x with two runs agreeing within 0.73%; per case 1.46x at (64,4096), 2.02x at (64,512), 1.70x at (64,2,1792); no case regressed, correctness PASS
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound
description: Delete a graph dispatch node via self-resetting device scratch, then shape-gate a single-workgroup fusion: ~1.71x on a graph-captured fp8 quant cast
keywords: ['dispatch-collapse', 'hip-graph', 'launch-overhead', 'quantize-cast', 'fp8', 'kernel-fusion', 'size-gating', 'grid-stride', 'non-temporal-store', 'raw-hip', 'gfx950']
kernels: ['data_to_scale_kernel', 'scaled_quant_kernel', 'initializeScale']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: launch-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-14
levers: ['host.launch-overhead', 'algo.kernel-fusion', 'mem.grid-stride']
---
# Collapse the graph nodes first, then shape-gate a single-workgroup fusion
- lever: on a graph-captured multi-kernel quantize/cast, count the dispatch nodes before touching any device body: a node costs a flat amount that is independent of the work inside it, so deleting a node outranks every per-node lever until the count stops falling
- apply: delete an initialization node by making the device scratch scale self-resetting (its last consumer restores the sentinel); then gate small shapes onto a single-node, single-workgroup path holding the tensor in registers with one read and one write and no cross-block handoff; place that gate by measuring the fused/multi-node crossover rather than inheriting the first threshold that worked
- stack: total 1.71x unweighted geomean isolated (director-verified) = five directions compounded
  - 1. dispatch 3 -> 2 nodes via self-resetting device scratch scale — 1.44x standalone (round 2, verified) — by far the largest single lever
  - 2. flat grid-stride decomposition replacing row-per-block — +3.4 to +5.0% per case on top of (1) (round 3, verified); the same idea scored ~1.07x and was dropped at 3 nodes
  - 3. peel the consumer's grid-stride load above the dependent scratch reduce, plus a non-temporal output store — -2.20% and -0.65% (round 6, verified)
  - 4. shape-gated single-workgroup fusion below the measured crossover — +8.1% cumulative, -23% on the smallest case (rounds 8-9, verified)
  - 5. host-side exact-tile bounds-guard deletion on the fused path — +3.2% (round 9, verified)
  - note: attribution is incremental in landing order, not independent
- verify: re-profile after each landing and confirm the deleted node is actually gone from the captured node list; re-time every case against the frozen baseline, because the small fully-fused case and the large multi-block cases move for different reasons and only one of them can show the fusion gate
- pitfall: a merge-on-top memory candidate measured ~1.07x in round 1 and was dropped → per-node device work is invisible while dispatch count dominates → re-price shelved candidates against each new incumbent; that same candidate won its round once the node count fell
a host-side trim claimed ~-1.1% and was banked as sub-bar → a later round priced its physical mechanism far too small to produce that reading → it was inside the cross-session drift band and had to be retracted
- caution: also verify the concurrency story of a self-resetting device scratch: publishing per-block partials through one global buffer is safe only while calls are stream-ordered inside the captured graph, and also verify a fallback path exists for shapes above the scratch's block cap
- source: run kernel_20_geak_0811_2h_bmk7_long lane, 2026-08-14, TechLead report + director validation (accepted, 11 rounds / 26 directions)
