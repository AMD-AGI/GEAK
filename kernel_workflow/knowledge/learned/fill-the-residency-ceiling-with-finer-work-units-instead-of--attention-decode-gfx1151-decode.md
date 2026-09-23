---
key: fp16 GQA decode attention at batch 1 (many query heads per KV head, head_dim 128, long history) in Triton on a 40-CU RDNA3.5 wave32 iGPU with WMMA and a shared LPDDR memory path, starting from a multi-pass baseline that materialises the score matrix
type: lever
confidence: ★★
effect: 2.9199x geomean vs the frozen baseline, director re-measured on a workspace rebuilt from the true original (in-run verify 2.9072x, 3-run median band 2.85-2.89, across-run median spread 0.5-2.2%); per case 2.49x at history length 2048, 2.82x at 4096, 3.55x at 8192; correctness PASS on all 8 gate cases including 4 ragged lengths, error identical to the baseline's own
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-09-17
name: fill-the-residency-ceiling-with-finer-work-units-instead-of--attention-decode-gfx1151-decode
description: GQA decode on a 40-CU RDNA3.5 iGPU: fuse to a group-shared split-KV flash decode, then size the split to the residency ceiling and sub-tile the tail - 2.92x
keywords: ['split-kv', 'flash-decode', 'attention-decode', 'decode', 'occupancy', 'grid-occupancy', 'wave-quantization', 'launch-shape', 'register-pressure', 'roofline', 'read-only-twin', 'gqa-head-sharing', 'kernel-fusion', 'dispatch-collapse', 'closed-axis', 'isa-inspection', 'anti-pattern', 'triton', 'fp16', 'gfx1151', 'rdna']
kernels: ['gqa_decode', '_fd_split_group_kernel', '_fd_combine_kernel']
platforms: ['gfx1151']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
origin_kernels: ['gqa_decode_kernel']
cost: L3
verified_on: 2026-09-17
roofline: latency / device-fill bound before (the output pass ran at roughly a tenth of achievable occupancy over 3 dispatches, and the KV stream was re-read once per query head in the group) -> memory-bound after at 83/83/86% of a DEMONSTRATED pure-read ceiling, with fetched/unique bytes exactly 1.00x, per-wave load instructions within 2% of the ideal count, and achieved 4.9-5.6 waves/SIMD against a 6-wave register ceiling; about 1.2-1.4x of bandwidth headroom is still open at every case
levers: ['algo.fusion', 'compute.launch-shape', 'compute.work-granularity', 'host.dispatch-collapse']
---
# Fill the residency ceiling with finer work units instead of lifting it
- lever: On a register-capped split-KV decode kernel, treat the workgroups-per-CU residency ceiling as fixed and spend the round on filling it: pick the split count so (kv_heads x split) lands exactly on (residency ceiling x CU count), and hand history out in units of HALF a KV tile so a workgroup runs whole tiles plus at most one cheap partial one. That raises wave-lifetime/duration (here 0.77/0.86/0.89 -> 0.81/0.89/0.94) and achieved occupancy with the same workgroup count, same register count and no spills. It is the second move; the first is the fusion that gets you onto the byte wall at all - one KV tile read once per group and fanned to every query head of that group through the matrix pipe (pad the group dim up to the matrix tile), online softmax carried in registers, the materialised score buffer and its extra pass deleted.
- apply: All host-side plus constexpr once the fused body exists: split = min(want, ceil(history/unit)) with want derived from CU count and the measured workgroups-per-CU ceiling; a unit counter per workgroup rather than a contiguous row range; and give the partial-reduce kernel a real grid by chunking it along head_dim at one warp per program instead of one program per query head.
- stack: total 2.9199x isolated (director-verified) = three directions compounded, attributed incrementally in landing order
  - 1. fused group-shared split-KV flash decode (score buffer + its softmax pass deleted, 3 dispatches -> 2, KV redundancy 4x -> 1x) - 2.6499x standalone (round 1, verified) - the bulk of the win
  - 2. wrapper launch-plan cache with pre-marshalled args and preallocated scratch - 1.0042x standalone before (1), but +4.6% on top of it at integrate (round 1, verified), essentially all on the shortest case, because (1) shrank the GPU time the host submit had been hiding under
  - 3. split count set to the residency ceiling, half-tile work units, and a grid on the reduce kernel - +5.3% on top of (1,2) (round 2, verified)
  - note: attribution is incremental, not independent; (3)'s three mechanisms were never isolated from each other, though the reduce-kernel grid alone measured about +1%.
- verify: Check that the launched workgroup count equals (residency ceiling x CU count) and that wave-lifetime/duration and achieved waves/SIMD actually rose, with register count and spill count unchanged; confirm fetched bytes equal unique KV bytes by cross-checking the last-level-cache read-request counter against a hand count, and re-run parity on the ragged history lengths, not only the benchmarked ones.
- pitfall: The longest case was declared finished at ~95% of roofline in round 1 -> the published bandwidth figure it divided by was a read-read-WRITE number while this kernel is a pure read stream, understating the ceiling by 15-20% -> demonstrate the ceiling with an ablated twin (same skeleton, same grid, same byte stream, attention math deleted); against that twin the shipped kernel sat at 83-86%, with headroom at EVERY case rather than only the short one.
The round-2 direction was funded on the prescription "cut registers so one more workgroup per CU fits" and every route to it lost -> the occupancy hint reached the register target only by spilling (down to ~0.6x of the incumbent), two scheduling hints were no-ops, and a head-dim split that genuinely reached the target with zero spills still lost ~8% -> occupancy was not the binding constraint; fund the diagnosis and leave the mechanism to the engineer.
Making the split tile-exact by going one step past the residency ceiling cost ~13% -> the scheduling tail of the extra wave-front exceeds the tile tail it removes -> land on the ceiling exactly and fix the tail with granularity instead.
The tile-quantisation model forecast about ten times what the fix delivered (~+25% predicted, +2-4% measured) -> achieved occupancy is just the wave-lifetime ratio times the register ceiling, so removing the tail only buys back that ratio -> price a balance direction against the ratio before funding it.
Fusing the two dispatches behind a last-CTA atomic arrival protocol lost, correctly implemented (acquire-release counter, exchange reset, volatile partial reads) -> the protocol cost more than the dispatch boundary it removed -> the productive form of the same lever was a few lines: give the reduce kernel a grid.
Two rounds argued the inner loop was the constraint -> stage ablation showed the FULL kernel beats the partially-ablated ones, so the softmax, the second dot and the transpose all price at zero or less under the memory stream -> ablate stage by stage before spending a direction on inner-loop work.
RDNA tooling: "the hot loop issues zero LDS instructions" was a false negative -> gfx11 spells LDS ops ds_load_* / ds_store_*, not the CDNA ds_read / ds_write -> grep the RDNA mnemonics; also the traffic counter reports exactly half the real bytes, the global-activity counter carries a fixed per-dispatch offset (normalise per-wave counters by fitted clock x duration), and three derived metrics in one counter pass abort and orphan the child process, so collect them one per pass.
- caution: Also verify the shortest case with at least three full benchmark runs and quote the median band: within-run rep spread there breached the 5% bar in 3 of 3 runs with a ~6%-wide speedup band, so a single-case gain under about 6% there is not measurable; and also verify how much of a short-case win is host-launch reduction rather than kernel work, since that part would not survive into a caller that already replays the op inside a captured graph.
- source: run team_gqa_decode_kernel_20260917_144205_38421_2926, 2026-09-17, TechLead report + director re-measurement, geomean 2.9199x at parity over 2 rounds
