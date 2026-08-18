---
key: top-k-gathered MLA-latent sparse attention, ragged prefill, Triton bf16 on gfx950/MI355X under vLLM V1
type: lever
confidence: ★★
effect: 6.90x time-weighted / 5.19x unweighted geomean isolated vs the frozen baseline (n=72 medians, non-overlapping 95% CIs); per case 7.06x on the long ragged prefill, 6.42x mid, 3.08x on the tiny kernel-bound case
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-17
name: retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill
description: Gathered MLA sparse prefill attention on gfx950: one program per query position, then halve k-trips and delete each per-trip cross-warp reduce - 6.90x weighted
keywords: ['attention', 'prefill', 'triton', 'top-k', 'tiling', 'tile-geometry', 'online-softmax', 'cross-workgroup', 'vgpr-pressure', 'num-warps', 'xcd-remap', 'l2-locality', 'unroll', 'gfx950']
kernels: ['_sparse_attn_prefill_ragged_kernel', '_rocm_sparse_attn_prefill_ragged_triton']
platforms: ['gfx950']
kernel_class: attention
regime: prefill
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-17
roofline: issue/rendezvous-bound at a few percent of achievable bf16 matrix peak -> latency/dependency-bound after the retile; the gathered latent is last-level-cache resident, so bytes were never the wall
levers: ['mem.lds-tiling', 'compute.trip-count', 'compute.rendezvous-deletion']
---
# Retile to one program per query, then delete every per-trip rendezvous
- lever: Give one program all query heads of one query position and raise the k-block until the loop is a handful of trips; then delete the per-k-trip cross-warp rendezvous one at a time - that count, not instruction count, is what moved wall clock here.
- apply: Retile BLOCK_H 16->64 with BLOCK_K 64 as one coupled 5-knob change (num_warps/num_stages/matrix_instr_nonkdim move with it); wrap the PV dot in an scf.if so the chained-dot warpsPerCTA=[num_warps,1] inheritance breaks (67% VGPR drop, 2x residency); then, in order: lazy acc*=alpha, two per-trip workgroup reduces collapsed into one, the softmax denominator deferred to a register tile, the running row max replaced by a bounded-warmup frame, and the frame itself deleted behind an exp2 argument clamp (finite saturation, strictly safer than the inf/NaN original).
- stack: total 6.90x weighted, ~9 directions compounded over 16 rounds
  - 1. retile to one program per query - 2.15x standalone, the largest single step
  - 2. score-path VALU deletion (exp2, scale folded into q, EVEN_D/EVEN_H constexpr, fp32 epilogue divide) - to 2.34x
  - 3. warp-partition scf.if around the PV dot - to 2.59x, and it re-opened tile/launch knobs the ledger had closed; deleting it later is -84%
  - 4. trip-count halving (k-block 32->64, only reachable once the value prefetch pinning it was removed) - -11.6% on the heavy case, the largest late step
  - 5. five rendezvous deletions ending in the clamp - cumulative 3.05x -> 4.95x
  - 6. D-split of the PV dot (pure scheduling, instruction-identical LDS traffic, LDS wait -24.8%) +7% heavy case; per-launch UNROLL=2 constexpr; waves_per_eu=2; XCD pid remap gated at large query counts (L2 hit +0.64 pp, DRAM reads -1.5%, -0.93%)
  - note: attribution is incremental in landing order, not independent
- verify: One protocol per A/B (the in-process vs cross-process offset is shape-dependent, 1.1-2.3%, so mixing invents a fake shape-dependent win), medians not means, interleaved same-session controls (drift is the same decimal place as any late effect), and gate on the weighted metric - a 0.6% cross-session offset on the two small cases cancels a 1% real win on the case carrying 86% of the weight.
- pitfall: A naive stack of two individually-positive patches came out negative -> one patch added a new dispatch target, so every hunk of the other was dead code on the measured shapes -> hand-merge mechanism by mechanism and check which hunks are still on the executed path.
The transposed-accumulator and prefetch family kept looking attractive -> the MLA latent is both K and V, so MFMA inherently needs two LDS layouts and a genuine second gather is GVN-merged, while register/LDS prefetch went over the VGPR ceiling -> the trip count was the cheaper knob for the same per-trip cost.
- caution: Also verify the ragged contract survives the retile: that shape-derived launch knobs come from host-side tensor sizes rather than from indptr contents, and that the k-loop stays masked.
- source: run mi355x_vllm_triton_sparse_attn_prefill_ragged-bmk7-12h, 2026-08-17, director-validated (rounds 1-16, isolated A/B vs frozen baseline, correctness 3/3)
