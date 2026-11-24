# Flash Attention — Structural Insights

Read before importing a published attention technique (especially FA-4). Operational how-to:
`attention.md`. Full FA-4 catalogue: `tile-programming-cutedsl` (B200 / sm_100 only).

## Structural model

`QK MMA -> online softmax (running m, l) -> PV MMA -> epilogue normalize`. When the row max grows,
`O` must be rescaled before the next PV accumulate; `P` is the unnormalized per-block numerator.

## Tile-primitive graph (one K/V block)

GMEM -> LDS staging (async / buffer load) -> S (QK MMA) -> P (exp + row reduce, in a layout the PV
MMA can read) -> O (PV MMA) -> rescale / epilogue normalize. For each arrow state the operand
layouts, the LDS buffers and their producer, and the wait that orders it
(`../gluon/layout-reference.md`, `../gluon/memory-reference.md`, `../tile-programming/pipeline.md`).

## Why GEMM-only levers may not transfer here

- Softmax VALU sits on the critical path between the matmuls: the fork-era GEMM throughput-pairing
  scheduler (gfx950-tutorial-v0.1) emitted invalid IR on an FA forward (reproduced); later builds of that
  pass roll back silently and upstream MFMA scheduling leaves the loop untouched — check the ISA for the
  schedule on the toolchain in use.
- The `(m, l, O)` recurrence blocks same-phase symmetric ping-pong (not the stage-offset warp pipeline).
- The PV accumulator is read-modified every iteration (rescale), not write-only until the epilogue.

## Bound-class porting filter

FA-4 (arXiv:2603.05451) forward techniques: (1) TMEM + async MMA + warp-specialized pipeline,
(2) polynomial exp, (3) conditional rescale skip, (4) 2-CTA backward, (5) LPT causal.

| the technique fights | Blackwell (B200) | gfx950 / gfx942 |
| --- | --- | --- |
| exp / shared-memory throughput | primary FA-4 fight | applies when the profile shows exp/LDS saturated — uncommon on occupancy-first CDNA forward in the kernels measured |
| accumulator read-modify-write latency | correction warpgroup | low-occupancy decode, or a 2-wave prefill where `acc *= alpha` sits on the PV MFMA dependency chain |
| grid makespan (LPT) | yes | yes — causal imbalance |
| TMEM / `tcgen05` / separate MUFU / 2-CTA cluster MMA | native | no such hardware on CDNA3 / CDNA4 |

## gfx942 / gfx950 (CDNA3/CDNA4) — validated strategies

Occupancy-first, as observed in the gfx942 / gfx950 attention kernels measured here (Triton 3.8.0): latency
was mostly hidden by resident waves rather than by warp-specialized pipelines.
FA-4 is a catalogue, not a porting checklist; each strategy below names the bottleneck class (profile
+ asm signal) it targets. gfx942 measurements are labelled as downgrade evidence.

### LPT causal (longest-processing-time-first)

- **Grid-makespan lever:** reverse `mblock` order — a pure `pid -> worktile` remap, no body change,
  so it is independent of softmax or pipeline work on causal fwd/bwd; pairing and the varlen uniformity gate:
  `attention.md ## Scheduling (load-imbalanced grids)`.
- Conflicts with an L2 `pid` swizzle (not additive): A/B LPT-only / swizzle-only / both.
- *Verify:* spread in per-worktile time; re-run causal correctness.

### Conditional rescale skip

Mechanism, the two paths and the branch-not-select requirement:
`attention.md ### Conditional rescaling (skip the per-block acc *= alpha)`.

- **Path A (latency), all should hold:** low occupancy (often 1 wave/SIMD decode); the rescale is
  exposed on the PV path in asm (plus `v_accvgpr_read` where the `v_mfma` destination is `a[..]`;
  on gfx950 / Triton 3.8 it is normally `v[..]`); running max usually stable.
- **Path B (co-execution budget):** two waves/SIMD with the softmax in the compute clusters, and a
  cluster's VALU demand over its MFMA-shadow capacity
  (`../tile-programming/llir-codesign.md ## Attention: the co-execution budget`).
- **Expected not to pay (reasoned; A/B before closing):** multi-wave with no over-budget cluster and
  no `acc *= alpha` on the PV dependency chain (the branch is overhead); under the plain-Triton
  auto-pipeliner (`num_stages>1` dense prefill — a data-dependent skip can break the steady
  schedule); a hand-built prefill with neither profile in hand gives no basis to predict it. Measured
  negative with an added cross-lane vote:
  `../pitfalls/negative-patterns.md ## Conditional rescale skip — measured negative at the occupancy its own path A asks for`.
- *Verify:* occupancy; `v_mfma` destination and `v_accvgpr_read` in the PV loop; race-test after any
  handoff change; on path B confirm something spends the freed budget (alone it reads as null).

VALU folds (`log2e`, `qk_scale` into Q) are not FA-4: `attention.md ## Online softmax (the recurrence to preserve)`.

### Software-pipeline reproduction — the kernel shape re-injection needs

A diagnostic route — it reproduces plain's schedule; the other route is the hand-written overlap
(`../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`;
re-injection: `../tile-programming/pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`).
Its target is plain's schedule, and it is not designed for a loop that already hand-stages (check the
TTGIR for the stages it added). The loop shape that gave the pipeliner room (*injected*
measurements: gfx942 FA forward, plus one gfx950 / Triton 3.8.0 FA-forward body — A/B before relying on the
thresholds; not a plain-Triton `num_stages` recommendation):

- **K/V loaded in-body, no hand register-prefetch** — the pipeliner is the prefetcher; a
  hand-prefetched loop gained about nothing.
- **Split the causal mask into two loops**, not one loop with an `scf.if`: the branch-free loop
  pipelines like plain's, and it un-blocks `BlockPingpong` (which bails on a loop-variant mask).
- **Composes with the LPT + XCD remap.**
- **Depth** is a parameter of the re-injected pipeliner, not a Gluon knob. `ns=3` was worse than `ns=2`
  on the gfx942 kernels and, on the gfx950 body, at the lengths where pipelining won. `ns=4` takes the
  chained-dot path: on the gfx942 kernel it hit the occupancy cliff (so `ns=2` there); on the one gfx950 /
  Triton 3.8.0 body measured it was the best arm, with no cliff — whether its extra registers cross the next
  waves/CU boundary decides, so read VGPR / `shared` per arm. **Length threshold:** dispatch the
  non-pipelined variant at short `S`. On gfx942 the injected overhead lost below `S = 2048`; the gfx950 body
  crossed near the same length — one body, so A/B it per arch and per body. Give each arm its own constexpr
  or process: the in-process JIT cache does not key on the re-injection hook.

### Do not plan on AMD

TMEM / warp-specialized FA-4 pipelines; 2-CTA MMA backward tricks (no such hardware on CDNA3 /
CDNA4). Polynomial exp is not an FA-4 import here (no separate exp issue port — it adds VALU on the
same SIMD); the profile signal it targets is a saturated exp unit, and one recipe losing does not
close other fast-exp forms (`attention.md ## Online softmax (the recurrence to preserve)`)
(`../method/profile.md ### Raising exp throughput by partial FMA emulation (when the exp unit is the bound)`).
gfx1250: out of scope.
