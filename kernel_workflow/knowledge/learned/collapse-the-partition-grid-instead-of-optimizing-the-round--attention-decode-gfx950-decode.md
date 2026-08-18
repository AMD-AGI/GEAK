---
key: split-KV paged decode attention in HIP/C++ on gfx950 whose large-context shapes launch thousands of workgroups across a partition grid and then a second reduce dispatch
type: lever
confidence: ★★
effect: reproduced on a second, independent lane and harness at 1.4302x geomean over 7 bf16 decode cases (arithmetic 1.4638x), 7/7 parity, no case regressed, per case 1.2446x on the weakest mid-context shape up to 1.5895x at the longest context; first lane 1.2042x geomean over 16 cases, 15/16 >= 1.00x, non-overlapping vs the frozen baseline; split by regime the large-context half is 1.3738x (per case 1.32-1.46x) while the launch-floored small-batch half is 1.0556x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 9
toolchain: unknown
last_seen: 2026-08-17
name: collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode
description: One workgroup walks every KV partition of a sequence: the split-K round trip and its reduce dispatch both vanish; 1.27x then 1.43x on two lanes.
keywords: ['decode', 'paged-attention', 'split-kv', 'dispatch-collapse', 'online-softmax', 'grid-gating', 'hip', 'partition', 'stacking', 'register-pressure']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel', 'paged_attention_ll4mi_reduce_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-17
---
# Collapse the partition grid instead of optimizing the round trip it creates
- lever: one workgroup walks all partitions of a sequence with gridDim.y -> 1, carrying online-softmax max/sum/accumulator in registers and writing the final output directly
- apply: gate the collapsed path on a workgroup-count threshold so only large grids take it; keep the two-dispatch path for small grids; three effects compound - the partial-output/exp-sums/max-logits round trip is deleted, the reduce dispatch is deleted, and per-partition Q fetch and scale setup hoist to once per sequence
- stack: total 1.2042x geomean = four directions compounded, attribution incremental in landing order
  - 1. partition collapse - 1.2737x on the large-context half standalone (verified) - the bulk of the win
  - 2. grid-gated fused reduce epilogue - 1.0525x (verified), and it is what carries the small-batch half (+7% by ablation)
  - 3. occupancy: one fewer wave requested so the register budget stops spilling - 1.0388x (verified)
  - 4. software-pipelined collapsed loop with Q staged in LDS - +4.2% on top (verified, 16/16 positive)
  - note: the last merge was exactly multiplicative in its parts (1.0501 measured vs 1.0544 predicted), earlier ones were not
  - second lane, same lever, different stack: total 1.4302x = a read-once non-temporal value-stream load 1.2121x (round 1) then the collapse at 1.3956x standalone (round 2, verified) then re-sweeping the partition-size knob inside the collapsed world for +2.2% over the best individual
  - note: the collapsed world INVERTS the partition-size tradeoff - the value that won before the collapse scored ~0.93 of collapse-alone, and halving it twice recovered ~+10%, since extra partitions are near-free once the combine is a register rescale and a smaller one frees live registers
- verify: confirm the collapsed path actually engages per shape (a path trace beats a speedup), check the reduce dispatch count drops to zero on the shapes you expect, and re-time against the frozen baseline interleaved with an unpatched copy
- pitfall: collapse forced onto small grids measured 0.58x, worst case 0.475x -> one workgroup per sequence starves a 256-CU grid -> keep it behind a minimum-workgroup gate; an epilogue edit that never changed timing -> it landed in the build-regenerated HIP copy of the .cu and never reached the compiler -> diff the .cu and grep the saved patch for generated-file hunks
- pitfall: two winning patches merged with only line offsets, yet the pair lost to one of them alone -> the second patch's whole contribution (fewer partitions, so a smaller scratch round trip) is exactly what the collapse deletes, leaving only its cost in live registers -> keep the merged patch's generalization (partition size as a correctly derived compile-time knob) and re-sweep that knob after merging
- pitfall: changing the partition constant compiled clean and surfaced only as a parity failure -> two derived extents were literals that were merely correct at the old size (a straddling value block, a hardcoded token-axis extent) -> re-derive every literal the old size made a coincidence, and treat the size where per-lane tokens fall below one native vector fetch as the correctness floor
- caution: also verify what you are actually pricing before funding the fusion: deleting only the second launch while keeping the scratch round trip (an arrival-counter handshake) measured 6-13% slower on every case here, because the two dispatches overlap the attention tail rather than serialize
- caution: also verify the caller's allocation contract before shrinking the partition size: the partial-output buffers are sized by the caller for a fixed partition count, so a smaller one corrupts memory silently unless it uses private scratch; and also verify against the other engineer's base arm, since identical binaries swung bimodally on two large-context cases
- source: run paged_attention_large-own16h, 2026-08-12, director-validated (geomean 1.1965 re-measured vs 1.2042 reported), correctness PASS; reproduced on a 2h lane, run kernel_20_geak_0811_2h, 2026-08-17, director-validated 1.4302x geomean, 7/7 correctness PASS, single fused dispatch confirmed by kernel trace
