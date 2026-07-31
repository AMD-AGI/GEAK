---
id: flydsl_decode_moe_stage1_blkmap
title: Halve grouped-MoE stage-1 weight HBM traffic at decode by fusing paired same-expert sort-blocks into one double-height compute tile (expert weight streamed once, reused across both blocks)
kind: expert_skill
authors:
- zhengy
scope: kernel
match:
  operator: grouped_gemm_moe
  arch_class:
  - '*'
  gens:
  - gfx950
  dtypes:
  - mxfp4
  - fp4_e2m1
  - fp8_e4m3
  - fp8_e4m3_fnuz
  regimes:
  - decode
  from_backend: ''
  to_backend: ''
  profile_signature:
    op_name_regex: ''
    min_pct_gpu: 0.0
expects:
  isolated_speedup_min: 1.05
  isolated_scope: 'stage-1 ISOLATED SEGMENT = the gate+up GEMM kernel PLUS the descriptor-producer
    kernel this recipe adds. Do not include pre-existing sort/align/quant helpers in this isolated
    gate; report the full stage-1 window separately. The producer has an unrelated name, so a GEMM-only
    filter drops it and overstates the win by ~1.9pp (1.103x GEMM-only vs 1.080x with producer).'
  e2e_delta_min_pct: 1.0
  parity: required
validation:
  status: validated
  last_verified: '2026-07-27'
  gpu: 'gfx950 / MI355X'
  model: 'fp8-act / fp4-wt grouped-MoE decode (provenance only; selection is by shape/bottleneck, not model)'
  measured:
    isolated: 'stage-1 gate+up GEMM 185.7 -> 168.4 us = 1.103x (-9.3%) on a base tile_m=64 kernel with 86
      disjoint pairs among 347 blocks (172 participating blocks); 3-trial average 179.5+-0.3 -> 162.5+-0.5
      us (-9.5%), independent repeat -9.1%. Counting the +3.62us descriptor producer the segment is
      185.7 -> 172.0 us = 1.080x (-7.4%).
      Weight-HBM read units 347 -> 261 (analytic count over the real routing distribution, floor 259 --
      NOT a hardware counter). Device-time sum over the whole fused decode path 313.9 -> 301.3 us (-4.0%).
      INDEPENDENT REPRODUCTION on a second tree, 4 interleaved trials, selection driven purely by the
      tuner config (no env override): stage-1 segment 192.69 -> 174.59 us = 1.1037x (-9.4%), arm ranges
      191.38-195.32 vs 173.20-174.76 (non-overlapping); GEMM-only 192.69 -> 170.92 = 1.1274x; descriptor
      producer 3.68 us; whole fused decode path 352.10 -> 333.25 us = 1.0566x.'
    e2e_pct: ''
    parity: 'pass - numerically equivalent to the unpaired tile (logits_diff delta <4e-6 vs baseline, cos_sim 0.99893); differs only by reduction order'
  artifact: ''
role: advisory_prior
supersedes: []
---

## When to use
Trigger on the **problem signature, not a specific model**: a **grouped-GEMM MoE stage-1 (gate+up)** kernel
at **decode / small-M** where repeated expert-weight HBM traffic is a **material** part of latency (the
baseline may be mixed rather than purely memory-bound), the MoE token-sort pads each expert to a fixed
sort-block granularity `B` (commonly 64 rows), and routing gives many experts **>=2 sort-blocks**. A
per-sort-block compute tile then re-reads the same low-precision expert weight once per block. Do not infer
"compute-bound" merely because the MFMA dispatch dominates wall time, and do not infer "memory-bound" from
analytic bytes alone.

If the workflow has not already distinguished MoE phase/stage, resolve it before applying this skill. Do not
match on `grouped_gemm_moe` alone. This recipe is for the **decode stage-1 gate/up** path: look for the
stage-1 GEMM (`mfma_moe1` / gate-up naming), sorted-block leader mapping, and no top-k reduce kernel in the
scored isolated segment. If the evidence instead points at stage-2 down/reduce (`mfma_moe2`, partial-buffer
materialization, separate top-k reduce), use the stage-2 recipe instead. If phase or stage remains ambiguous,
treat this skill as not applicable and let the normal workflow exploration classify the bottleneck first.

**Three applicability gates — check ALL before claiming the reference-sized win, and report them either way:**
1. **The baseline must select a `tile_m = 64` compute tile.** If the baseline already picks a 128-row tile,
   doubling it needs ~263 KB of LDS against a ~160 KB limit and **will not build**. The recipe is
   inapplicable on that config, not merely unprofitable.
2. **A substantial fraction of sort-blocks must be pairable** (adjacent same-expert pairs). Count them from
   the sorted block->expert table *first*. Report both definitions explicitly: the reference had
   `pair_count_fraction = 86 / 347 = 24.8%` (the analytic weight-read-unit reduction) and
   `paired_block_fraction = 2*86 / 347 = 49.6%` (blocks participating in pairs). A 2%
   `pair_count_fraction` has only a `1/(1-0.02) = 1.020x` traffic-only ceiling before descriptor/resource
   overhead, generally below this skill's own `expects`.
3. **Repeated-weight traffic must be latency-relevant.** Prefer HBM/L2 counters. If counters are unavailable,
   classify the baseline as **mixed / unknown**, run a same-session paired/unpaired ablation, and let measured
   A/B decide; a profiler label with null bandwidth/cache counters is not evidence. After pairing, separately
   check whether the doubled tile shifts the winner to VGPR/LDS residency or MFMA throughput.

A prior re-verification that reported **1.0000x / out-of-applicability** was itself an invalid diagnosis:
its local tuning CSV had been hand-edited, uncommitted, from the committed `tile_m=64` baseline to `tile_m=128`.
Restoring the committed `tile_m=64` row reproduced `1.1037x` on the isolated segment. Treat baseline config
provenance and a clean worktree as hard preflight; do not relabel a valid shape from a contaminated local row.

## Mechanism
Why the redundancy exists: MoE align/sort rounds each expert's token count **up** to the sort-block `B` so the
grouped GEMM can index fixed-size row-blocks; the kernel streams the expert's full weight tile per `B`-row
block. An expert holding `k` sort-blocks therefore reads its weight from HBM `k` times even though the weight
is identical across those blocks. At decode each block is a thin sliver of useful rows, so this redundant
weight read dominates stage-1 HBM traffic.

The lever: **fuse two adjacent same-expert sort-blocks into one `2B`-row compute tile.** The expert weight is
loaded to registers/LDS **once** and MFMA'd against both `B`-row halves, so weight HBM reads for paired experts
are halved (approaching the floor where each expert's weight is read exactly once). The decisive property is
that the **sort-block granularity `B` is left unchanged** — only the stage-1 compute tile's row-height doubles
— so the downstream stage-2 (down) padding and work are identical; the reuse is free of any stage-2 cost. It is
a pure **data-reuse / traffic** optimization: the arithmetic is the same accumulation over a larger `BLOCK_M`,
so the result is numerically equivalent modulo reduction order (not a new numeric scheme).

## Procedure
1. **Establish applicability, not a guessed bottleneck label.** Record base `tile_m`, pair count/fraction,
   VGPR/LDS, and HBM/L2 counters when available. If counters fail, say **mixed / unknown** and use controlled
   same-session A/B. If measured core reduction is much smaller than the analytic weight-read reduction,
   weight traffic is material but not the sole limiter; do not call either arm purely memory- or compute-bound.
2. **Build a compact leader-block descriptor.** Scan the sorted-block -> expert table; for each run of
   same-expert blocks emit `ceil(k/2)` "leader" tiles, each covering two `B`-row sub-blocks (a trailing odd
   block becomes a solo leader covering one). Compute this **device-side and CUDA-graph-safe** — preallocated
   outputs, no host sync, deterministic shapes — so it works under graph capture; an in-kernel parity scan or a
   host-side build is a slower fallback.
3. **Size the leader launch honestly.** `n_leaders = total_blocks - pair_count`; it equals
   `ceil(total_blocks/2)` only when nearly every block pairs. A worst-case full grid with sentinel early exits
   is graph-safe but can add an entire scheduling round when the merged kernel permits only one workgroup/CU.
   Report actual `Grid_Size_Y`; prefer a graph-safe persistent queue, indirect launch, or a producer-integrated
   leader count when the full-grid overhead is material.
4. **In the kernel, set `BLOCK_M = 2B`:** load the expert weight tile once and MFMA it against both `B`-row
   halves; accumulate / run the epilogue exactly as the unpaired kernel does per half.
5. **Store masking (the one correctness-critical step).** A solo leader (odd tail) must mask the store of its
   absent upper half, else it double-writes / goes out of bounds. Carry a per-leader "store-second-half"
   predicate and honor it at the store.
6. **Leave stage-2 (down) on the original `B`-row blocks** — do not widen the sort-block.
7. **Apply a runtime-signature gate before timing.** A valid paired arm must emit the paired GEMM suffix
   (for example `_am2_bmap`) and exactly one descriptor-producer dispatch. If either is absent, the arm is the
   baseline or a plumbing failure — reject its timing. Verify config-selected and forced diagnostic arms emit
   the same signature, then score the config-selected arm with all force-enable environment variables cleared.
8. **Joint-sweep the paired configuration.** `waves_per_eu`, `b_nt`, async-copy/cache policy, and pairing are
   not independently composable winners. Sweep the relevant `w2/w3/w4 x bnt0/bnt2` space with pairing active;
   do not select an unpaired scheduling winner and mechanically stack `_am2` afterward.
9. **Validate parity at the SAME tile granularity as the baseline** (the paired kernel is numerically
   equivalent, so logits must match within reduction-order noise), then measure same-session A/B and confirm
   the weight-HBM-read drop with hardware counters when available. The primary isolated gate is only
   `mfma_moe1 + descriptor`; pre-existing sort/align/quant helpers are diagnostics and the full stage-1 window
   plus whole fused path are secondary transfer metrics.

## Knobs & pitfalls
- **Pairing (merge factor 2) is the validated setting.** Merging `>2` blocks widens `BLOCK_M` further and risks
  LDS / register / occupancy limits — measure before assuming it helps.
- **The validated merged kernel is resource-heavy by design.** About 248 VGPR and 132,096 B LDS for the
  128-row effective tile is an observed reference envelope, not a universal target, and was also seen in the
  hand-tuned implementation. Do not reject it merely for that signature, but account for one-workgroup/CU
  residency and grid scheduling rounds. **Any scratch/spill is a hard rejection** for this validated path.
- **Keep the base sort-block granularity as the baseline uses it.** Forcing the paired (double-height) path on
  top of an already-large base compute tile exceeds the compile-time LDS/register budget and fails to build
  (measured: a 128-row base tile would need ~263 KB LDS against a ~160 KB limit).
- **The descriptor producer must be ONE kernel.** The reference first built it with eager tensor ops: ~13 tiny
  dispatches, **~47 us/iter**, which turned the whole optimization into a net **loss**. A single fused
  producer costs **~3.6 us**. An in-kernel parity scan instead of a producer measured *worse than baseline*
  (202 us vs 162 us). Budget the producer explicitly against the GEMM saving (~17 us) before starting.
- **Time the producer together with the GEMM.** The producer is a separate dispatch with an unrelated kernel
  name, so a GEMM-name timing filter silently drops it and reports 1.103x where the honest segment number is
  1.080x. See `expects.isolated_scope`.
- **The win scales with the fraction of `>=2`-block experts.** Synthetic / uniform routing overstates it vs
  skewed real decode routing — report real-routing numbers.
- **A `tile_m = 128` tuning entry, where one exists, beats this recipe — compare against it before claiming a
  win.** Pairing reaches a `2B`-row compute tile from `B`-row sort blocks; a config that simply *sorts* at
  `2B` reaches the same tile with no descriptor kernel and fewer weight reads (measured on the reference
  shape: paired-`t64` segment 174.6 us vs `t128` 168.6 us, because only 89 of 379 leaders actually pair, so
  290 solo leaders still read a weight tile for just `B` rows). This recipe's value is therefore confined to
  baselines that are **pinned** to `tile_m = B` — precondition 1 is about buildability, this is about
  profitability. State which baseline the speedup is measured against, and never compare a paired arm to a
  differently-tuned arm.
- **Make the merge factor selectable from the tuner's config, never from an environment variable alone.** In
  the reference the config tag was parsed but then dropped on the way to the kernel, so the tuner could not
  actually select pairing and the path silently never activated — a config-driven arm and an env-forced arm
  must be verified to emit the identical kernel before any A/B is trusted.
- **Every codegen-affecting knob belongs in codegen cache identity.** Include `b_nt`, `a_merge`, `blk_map`, and
  the other scheduling/cache modifiers in both the emitted module name (or an equivalent unique tag) and the
  JIT/AOT cache tuple. Omitting `b_nt`, for example, can make `bnt0` and `bnt2` silently reuse one compiled
  closure and invalidates the sweep.
- **The descriptor producer must be graph-capturable;** a host-side scan defeats CUDA-graph use.

## Do-no-harm notes
- **Numerically equivalent** to the unpaired kernel (same math, larger tile) — but only if the odd-tail store
  mask is correct; a missing mask double-counts. Parity-gate every build.
- **Clearly compute-bound stage-1 (usually prefill / large-M) sees little or no win** and the larger tile can
  cost occupancy. Small-M decode may be mixed: pairing can help before the merged kernel shifts to
  compute/residency limits. The workflow's on-box A/B picks the winner (advisory prior, never overrides
  measurement).
- **The paired mode is a pure add-on** — when not selected the kernel is byte-identical to the generic path, so
  a non-matching shape regresses nothing.

## Sources
Evidence is external and cited for **provenance only** — GEAK does not depend on any tree, and exact
commits / file paths are intentionally omitted. The portable knowledge is the signature, mechanism, and
measured numbers below.

- **Reference measurement** (same-session interleaved A/B; decode shape with **86 disjoint pairs among 347
  blocks (172 participating blocks)** on a **base `tile_m=64`** kernel; fp8 activations / fp4 weights on
  gfx950 / MI355X):
  - stage-1 gate+up GEMM **185.7 -> 168.4 us = 1.103x (-9.3%)**; 3-trial average **179.5 -> 162.5 us (-9.5%)**,
    independent repeat **-9.1%**;
  - **counting the +3.62us descriptor producer: 185.7 -> 172.0 us = 1.080x (-7.4%)** — this is the honest
    segment number;
  - sorting / align kernels unchanged (18.30 -> 18.26 us, i.e. noise), confirming the sort-block granularity
    was untouched;
  - weight-HBM read units **347 -> 261** (floor 259), an **analytic count over the real routing
    distribution — not a hardware counter reading**;
  - **parity numerically equivalent** (logits_diff delta `<4e-6` vs the unpaired baseline, cos_sim `0.99893`)
    — the output differs only by reduction order.
- **Autonomous-reproduction evidence** (same shape/signature): packaged-run audit confirmed the skill matched,
  emitted `_blkmap_kernel` plus an `_am2_bmap` GEMM, preserved sort64, and passed full fused-MoE parity
  (`logits_diff` about `0.001061-0.001070`, `cos_sim` about `0.99893`). The honest isolated segment was
  `191.060 -> 177.267 us = 1.0778x` (`173.292 us` GEMM + `3.975 us` descriptor). This clears the
  conservative `1.05x` skill gate and is a valid skill reproduction, but it is below a stricter prompt goal of
  `1.094x`; report that as **target_missed**, not as "skill did not inject". Future attempts must prove arm
  identity and jointly measure scheduling/cache choices using the runtime-signature, joint-sweep,
  cache-identity, and grid-size guardrails above.
- Implemented as an optional double-height-tile mode in a FlyDSL grouped-MoE stage-1 core with a device-side
  leader-block descriptor producer; the default (unpaired) path is unchanged. Any staged-rollout gating is an
  implementation detail of the reference, **not** part of this recipe — the transferable content is the
  pair-fuse-and-reuse technique above.
