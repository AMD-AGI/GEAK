# Workload router — which archetype is this kernel, and what must I read

**This page is reached by an archetype name, not by browsing.** `scripts/hw_budget.py --workload
<archetype>` already requires you to name one before it will price a ceiling, and it prints the
row below for whatever you named. The archetype keys here are exactly that tool's vocabulary, so
the name you had to declare anyway is the key that routes your reading.

Two rules before any row applies.

**Settle the archetype against the body, not the file name.** A kernel named `..._attention_...`
whose `q_len == 1` is a gather, and a kernel named `..._gemm_...` whose operand assignment is
data-dependent is the `moe` row. The check is
`intake.md ## 0. First: is it the archetype its name says?`, and it is worth the two
minutes: every row below routes layout and pipeline decisions, and both are wrong for the whole run
if the archetype is wrong.

Three questions answer it from the body, and each one has a consequence that is already recorded
rather than a preference:

| Read from the body | What it settles |
| --- | --- |
| **Which matrix op does the accumulator chain issue** — none, the regular instruction, or the scaled one? | The row, and then the counter caliber (`## Before any bound claim`, precheck 3). On CDNA4 fp4 has no regular matrix intrinsic at all, and fp8 has one only at the registered shapes (`[16, 16, 32]` / `[32, 32, 16]` in `v3.8.0`'s MFMA v4 table) — at `[16, 16, 128]` / `[32, 32, 64]` an fp8 operand means either the scaled path or an upcast that widens the operands first, so read the issued op rather than inferring it from the dtype (`../tile-programming/low-precision.md`). A kernel whose name says `gemm` and whose loop issues the scaled op is the `gemm` row **plus** that modifier, not a different row. **gfx942 downgrade:** there is no scaled matrix op at all — fp8 is the FNUZ variant (`e4m3fnuz` / `e5m2fnuz`) on the regular `cdna3.mfma`, fp4 has no matrix path, so the question collapses to "none or regular" |
| **Is there VALU work between the matrix ops** (a softmax, a descale, a norm epilogue)? | Whether the compiler scheduling ladder is available. On a VALU-between-matmul body the in-tree toggle emits **invalid IR / a verifier assertion**, not a slowdown, so this is a legality question rather than a tuning one (`../hardware/optimization-gotchas.md` row 7) |
| **Is the operand assignment data-dependent** — does a runtime index decide which weight this tile multiplies? | `moe` versus `gemm`. The two do not share a cost variable: MoE's is the weight bytes of the experts actually touched, not the token count, so pricing one as the other is wrong by the expert-count ratio (`moe.md ## Deciding the bound before sizing a prize`) |

A kernel whose name matches the row it lands in is the common case; the point of the three
questions is that answering them costs less than the round a wrong row spends.

**The archetype routes reading. It does not rank levers.** The bound your profile named does that
(`../hardware/bound-class-signals.md`), and the budget says which lever you can still afford
(`../method/budget.md`). A row here is the shortest path to the pages that can answer a decision,
not a claim that the decision is already made.

**Every row is written gfx950-first** (CDNA4, MI350X / MI355X); where gfx942 (CDNA3, MI300X / MI325X)
changes the answer the row says so as a *gfx942 downgrade*. The recurring downgrades, stated once:
no `ds_read_b64_tr`, no scaled MFMA / MXFP, direct-to-LDS async copy only at 32-bit with
`order=[1,0]` (so the authored ring is usually sync-staged), 64 KiB LDS / 32 banks instead of
160 KiB / 64 banks, and fp8 in the FNUZ encoding rather than OCP. Peaks and CU counts come from
`perf_knowledge/hardware/data/sku.json`, never from a number typed into a row.

## How to read a row

| Column | What it is |
| --- | --- |
| **shape** | the pages that decide the tile and the layer order for this archetype |
| **pipeline** | what this archetype's loop does to the overlap decision — the part that does not transfer between archetypes |
| **gate** | the capability or version cell to check BEFORE spending a round, not after a flat result |

Every archetype also owns the DSL-neutral spine: `../hardware/atlas.md` Table B for the layer ×
resource map, `../gluon/index.md` for mechanism → API, and
`../tile-programming/pipeline.md ## Where the overlap comes from, and it is not the same question per tier` for the overlap route. Those are not
repeated per row. The overlap order on the Gluon path is defined once, in that page, and every
**pipeline** cell below assumes it: hand-written first — register-level prefetch, then an authored
LDS ring, then `warp_pipeline_stage` with a scheduling-model choice (`num_warps >= 8`) — and
re-injecting plain's pipeliner last, only as a below-parity diagnostic or a last resort whose
numbers are labelled *injected* and never on an incumbent Gluon kernel. `num_stages` is dead on the
Gluon path in 3.8.0 (no pass consumes it); it survives only as a budget parameter and a champion
record.

**Where the acceptance criteria are, on whichever page a row sends you.** Two forms are in use and
both are load-bearing, so look for the second before concluding a page does not state one. Some
pages attach the check to the claim it qualifies, inline, as a `Verify:` clause in the sentence
that proposes the lever. Others collect them in a closing section — `## How to verify any of this`,
`## Acceptance`, `## Before recording a result here`. A page with no inline `Verify:` has usually
put its checks in the closing section rather than omitted them; a page with neither is a gap worth
recording through `../method/triage.md`.

## Before any bound claim

Three prechecks are shared by every row and repeated in none of them. They are ordered: each one
can make the next meaningless, so failing to run the first does not leave you with a slightly
imprecise number — it leaves you arguing against the wrong denominator.

1. **Does a roofline apply to this kernel at all?**
   `../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`.
   A working set inside the memory-side last level voids the **memory** floor specifically — the
   compute floor survives and becomes the binding one. The tell is a measured time *below* the
   memory floor, which is the model announcing it does not apply rather than an impossibility.
2. **Which dtype's ridge, against which denominator.** The ridge is
   `peak_tflops[dtype] / peak_hbm_tb_s`, so it is a different number per dtype and per SKU, and the
   SKU table — the single peak source, `perf_knowledge/hardware/data/sku.json` — deliberately stores
   none (`_no_derived_ridge`); `scripts/hw_budget.py` recomputes it per call. A kernel whose
   intensity sits between two dtypes' ridges is compute-bound in one and memory-bound in the other,
   so a ridge quoted from the wrong row moves the whole dispatch boundary. Any "% of roofline" that
   follows carries its `numerator_basis` (model | counters) and `denominator_basis` (datasheet |
   empirical@tool-version | in-shape probe): a datasheet denominator ranks only; only a measured
   numerator over a probed or calibrated denominator may gate or close (`../method/budget.md`,
   `../hardware/roofline-models.md`).
3. **Which counter convention the reading was taken under.** Matrix engagement is a union test
   rather than one counter — the scaled pipe and the generic one answer for different matrix ops,
   and the applicable caliber is a property of the op this kernel issues rather than of the
   architecture (`../hardware/bound-class-signals.md ## Rules (necessary-not-sufficient)`,
   `../hardware/optimization-gotchas.md` rows 2 and 5). Mixing two conventions inside one
   before/after does not add noise, it inverts the conclusion.

Arriving with a finished kernel behind you adds a fourth question — which of its conclusions
still hold on this one. `../gluon/technique-transfer.md` answers that per layer, and its
`## Cross-layer: does a roofline apply` and `## Cross-layer: the counter convention` sections are
the same two prechecks stated as transfer rules.

## Before any correctness-gated lever

One lever class is not decided on time at all. Read the **type** of the task's correctness oracle,
not just its abs/rel tolerance, before choosing any precision reduction: an angular/cosine gate can
be stricter than an abs/rel one and can reject a low-precision matrix path that abs/rel accepts
(`../tile-programming/low-precision.md`). Deciding this after the rewrite is how a round gets spent
on a change that was never admissible.

Two rows carry an extra form of this. On `linear_attention` the reduced value is carried across
steps, so the error compounds along the generated sequence rather than staying inside a tile
(`linear-attention.md`). On `collective` the gate is not numeric at all: a protocol that produced
the right answer on every launch you ran is the expected appearance of a broken one
(`collective.md ## How to verify any of this`).

## Before any compiler-invisible-ordering lever

One more lever class is decided before the rows below, and this is the only place its trigger is
keyed on the same variable you already had to name. Where a kernel needs an ordering, a
machine-state bit, a wave-collective or a whole publish/consume protocol that the tile abstraction
does not spell, the single door is `gl.inline_asm_elementwise`
(`../gluon/inline-asm-reference.md ## Classifying a site: mechanism × intent`). It has no layer of
its own; it is reached from layers 2, 3, 4, 4+ and 7.

**How strongly a workload of your shape tends to reach for it**, strongest first. This calibrates
expectations; it is not a recommendation, and it is not a performance statement:

| workload family | how often it appears | why |
| --- | --- | --- |
| cross-CTA / cross-device sync and state machines | **nearly always** | correctness depends on execution order and on cache scopes the binding will not emit |
| large matrix-multiply families | **a substantial minority** | staging and scheduling pressure, not correctness |
| pure element-wise epilogues | **effectively never** | nothing here depends on order beyond data dependence |

**Read the gradient as one sentence: density tracks how much compiler-invisible ordering the kernel
has to control.** It does not track difficulty, importance, or payoff.

**And read the adoption column as a count, never as a recommendation.** There is no kernel-level
timing behind any of these numbers — whether a site is worth its cost is `not established` here and
has to be measured on your kernel. The same survey records a tutorial tree that writes essentially
none, so a whole correct codebase is a live option; and the mechanism class with the *highest*
production reach is the one this pack records as least durable across a compiler upgrade
(`../gluon/inline-asm-reference.md ## What it costs you, on every class`). Three archetypes sit at
**0%** and that is the finding, not the gap: a pure elementwise epilogue has no ordering the
compiler cannot already see. Before reaching, pass the gate in
`../pitfalls/negative-patterns.md ## Inline asm: justify the reach before taking it`.

## gemm

- **shape** — `gemm.md ## Layer roadmap (a16w16 FP16)`, then the regime:
  `gemm.md ## Small-M (decode / GEMV) regime — memory/occupancy-bound` for decode-shaped M,
  `gemm.md ## Mid-M ridge (regime transition) — needs its own tile` at the transition. Operand and
  accumulator layouts: `../gluon/matrix-reference.md`. Variant substitution:
  `gemm.md ## Applying the framework to GEMM variants`.
  **Settle the regime before the tile**, because the two branches below disagree about what the
  rest of this row means.
- **pipeline** — **compute-bound (mid/large M):** the canonical case for the vetted skeleton
  (`../tile-programming/pipeline.md ### Vetted double-buffer skeleton (copy, then specialize)`),
  written on gfx950 as an `async_copy` ring with `commit_group` / `wait_group`; **gfx942
  downgrade:** the same ring sync-staged (register load → `ds_write`), since direct-to-LDS async
  there is 32-bit only.
  Decide `groups_per_stage` before `N`: per-tensor lead distance is bought with commit placement,
  not with buffer count.
  **Memory-bound (decode / small-M):** depth is the wrong question — confirm there is a loop worth
  deepening at all, and expect the answer to be the authored-staging route rather than a ring
  (`../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`).
  A K-loop that is waiting on weights it will never reuse does not get faster by holding more of
  them in flight.
- **gate** — `../hardware/capability-matrix.md` async-copy + MFMA dtype rows (gfx950 first; the
  gfx942 column drops scaled MFMA and the wide async widths). Beyond the hot loop,
  L2/XCD locality is a real lever here and not elsewhere: `gemm.md ## Beyond hot loop (L2 / XCD locality)`.
  Two modifiers that change this row rather than replacing it:
  **block-scaled (fp8/fp4)** — fp4 has no regular matrix-core intrinsic, and fp8 has none at the
  `[16, 16, 128]` / `[32, 32, 64]` shapes (it does at `[16, 16, 32]` / `[32, 32, 16]`), so on those
  shapes the path is `cdna4.mfma_scaled` and not a dtype swap
  (`../gluon/matrix-reference.md ## Matrix-Family Details` owns the per-`(version, M, N, K)` table);
  recompute the ridge at the fp8 rate
  rather than inheriting the f16 one; and note that the scaled pipe does not increment the generic
  MFMA counter, so a near-zero matrix utilization reading is a caliber artifact and not evidence
  of a memory bound.
  **Decode-shaped fp8 specifically** — plain `tl.dot(fp8)` lowers to the scale-less
  `tt.dot_scaled`, while the Gluon path must go through `mfma_scaled` with a unit e8m0 scale **when
  the tile's `(M, N, K)` is not one of the registered regular fp8 shapes**. At
  GEMV-like M that is an instruction **downgrade**, so "escalate this to Gluon" is the wrong
  direction for the fp8 decode band — the structural gap is stated at
  `../gluon/matrix-reference.md ## Matrix-Family Details`, and
  `../method/entry.md` is where the decision belongs. Decode / GEMV and paged shapes also run the
  Quick Reject (`../pitfalls/negative-patterns.md ## Quick Reject Checklist (= gate skip-transcription conditions)`)
  as an admission hint before any transcription.

## attention

- **shape** — `attention.md ## Layer roadmap` and
  `attention.md ## Online softmax (preserve, do not rewrite first)`. The structural model and what
  fails when a GEMM lever is ported in: `flash-attention-structural-insights.md ## Why GEMM-only levers fail here`.
  Paged / varlen / sparse bodies: `attention.md ### The dynamic-body variable (sparse / paged / varlen)`;
  decode / paged bodies run the Quick Reject in `../pitfalls/negative-patterns.md` first, as an
  admission hint (`## Paged / Indirect Decode` there is the paged instance).
  MLA and other head-dim substitutions: `attention.md ### Substituting into the co-execution variable (head dim, MLA)`.
- **pipeline** — two dots chained through a softmax, so the overlap question is where the softmax
  goes rather than how deep the ring is (`attention.md ## Where the softmax goes, and why it rides with the matrix op`).
  The overlap is authored (the hand-written order above); re-injecting plain's pipeliner on this
  body is a below-parity diagnostic only, and what it does here is measured, not assumed:
  `../gluon/pipeline-reference.md ## And on attention — two dots chained through a softmax`.
- **gate** — the wave-level model is decided BEFORE the ring depth and gates warps/CTA:
  `../tile-programming/scheduling-model.md ## How to choose`, then
  `../tile-programming/warp-pipeline.md ## Gate 1: is this kernel a candidate?` **before**
  `## Gate 2: does your toolchain have it?`. Gate 1 is where this archetype's two forks live and
  skipping to Gate 2 hides both: a **decode** body with a large per-lane accumulator is usually
  disqualified on the occupancy row (one resident wave is one instruction stream), and on a
  **scaled fp8/fp4** path the presumption flips rather than merely weakening. For the scaled case
  the two pages this row sends you to are in **open conflict** — attention needs the two-wave
  structure that the shared-read-bus hazard argues against — so it is measured, not chosen:
  `attention-lowprec.md ## What does NOT transfer, and why`. Load-imbalanced
  grids and `waves_per_eu`: `attention.md ## Scheduling (load-imbalanced grids)`.

## attention_bwd

- **shape** — `attention.md ## Backward pass (general)`. The forward roadmap does not carry over
  unchanged: the reduction lands on a different axis and the recompute decision is upstream of the
  tile choice.
- **pipeline** — read the forward row's pipeline note first, then treat the extra operand as a
  third commit chain rather than a deeper ring.
- **gate** — same wave-level gate as `attention`, plus the register budget:
  `../tile-programming/slicing.md ## Budget to compute first`.

## moe

- **shape** — `moe.md`. It is a GEMM whose operand assignment is
  data-dependent, so the GEMM roadmap still applies to the inner tile — but that framing predicts
  none of the decisions, so **type the kernel first** on fusion boundary, quantization form and
  shape band (`moe.md ## Type the kernel before advising it — three questions`); advice given per
  operator class is given at the wrong level here. The bucketing variable
  is tokens-per-expert (`moe.md ## Tile choice: tokens-per-expert is the bucketing variable`),
  and the small-M regime is reached per expert rather than per launch
  (`gemm.md ## Small-M (decode / GEMV) regime — memory/occupancy-bound`). The metadata that carries
  that distribution is itself built by a counting sort, and three shapes of it are viable — what
  selects one is availability and ordering guarantees rather than speed
  (`moe.md ## Bucketing: three shapes, and what selects one`).
- **pipeline** — the epilogue is non-uniform for three separable reasons
  (`moe.md ## The three real causes of a non-uniform epilogue`); fixing the
  wrong one buys nothing. Gathered operands separate the index layer from the transport layer
  (`moe.md ## Gather: separate the index layer from the transport layer`).
  The long tail is a *schedule* rather than a tile, and its gate is jobs-per-CU
  (`moe.md ## Load balance is a schedule, not a tile`).
- **gate** — `moe.md ## Version gates`. LDS `.gather`/`.scatter` are 3.7.0+, LDS
  atomic scatter 3.8.0+. Two more cells belong here rather than after a flat result. **The bound:**
  this archetype's byte model defaults to `memory` and its own metadata says the default is not
  actionable until `E_touched` is measured, and `stage` is a 1.5x error in either direction
  (`moe.md ## Deciding the bound before sizing a prize`). **Four
  implementation-level facts:** the scheduler ladder is *excluded* on a scaled expert GEMM (invalid
  IR, not a slowdown), matrix-engagement needs a dtype-aware counter read or the kernel
  misclassifies, launch geometry carries a recorded hazard (whose single-warp half is the decode
  band's deliberate default), and **padding changes sign across the bands** — latency at decode,
  ignorable in the middle, wasted compute at prefill
  (`moe.md ## Implementation-level facts that decide rounds, and are not in the layer ladder`).
  The routing kernel is **not** budgeted under this row at all — it reads
  none of the weight bytes this row's model is made of, so pricing it here overstates it by orders
  of magnitude. It has its own row: `## routing`.

## norm

- **shape** — `reduction-elementwise.md ## Group-wise quantization: the three-stage shape` when a
  quantized GEMM consumes the output, which is the usual case. The three stages have separate
  pages' worth of decisions: `reduction-elementwise.md ## Stage 1 — the group absmax reduction`,
  `reduction-elementwise.md ## Stage 2 — scale dtype, and the group size that is not the hardware's`,
  `reduction-elementwise.md ## Stage 3 — dequant + residual + norm`.
- **pipeline** — usually there is no ring to build: the binding question is whether the row is
  register-resident, because that decides whether stage 1 costs a second pass over the data. Read
  the residency paragraph in Stage 1 before proposing an overlap.
- **gate** — the most expensive misconception here is the group size: the CDNA4 hardware scale
  group is **32**, a group of 128 is a software convention the matrix instruction has no notion of,
  and NVFP4's is 16. On the explicit-tile path the rank-3 reshape carries a layout you did not
  author: `reduction-elementwise.md ## The group reduction and the broadcast back (explicit-layout path)`.
  Whether Gluon is justified at all: `reduction-elementwise.md ## When Gluon is justified here`.
  Inline asm is **not** on this row's menu: the surveyed epilogue archetypes (`rmsnorm`,
  `fused_add_rmsnorm_*`, `topk_softmax`, `paged_attention_output_gate`) sit at **0%** adoption, and
  the reason is structural rather than stylistic — there is no ordering here the compiler cannot
  already see. If you are reaching for it, the thing to re-check is the archetype
  (`## Before any compiler-invisible-ordering lever`).

## reduction

- **shape** — `reduction-elementwise.md ## Logical tiles` and `## What helps (in order)`.
- **pipeline** — a dot-free loop cannot be reached by re-injection (there is no anchor to inject
  against), so authored staging is the only overlap route:
  `../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`.
- **gate** — `reduction-elementwise.md ## When Gluon is justified here`. A reduction has no matrix
  multiply, so pricing it at the matrix rate puts the compute floor one to two orders too low and
  `memory` then falls out as a default rather than a finding.
  Inline asm is **not** on this row's menu, for the structural reason stated under `## norm`.

## elementwise

- **shape** — `reduction-elementwise.md ## What helps (in order)`. The tile is a bandwidth
  decision; there is no reduction axis to place.
- **pipeline** — same authored-staging note as `reduction`. Usually the answer is no ring at all: fuse the
  producer/consumer and remove a temporary instead.
- **gate** — as `reduction`: no matrix multiply, so declare the engine
  (`--flops-engine valu`) or the floor is meaningless.
  Inline asm is **not** on this row's menu, for the structural reason stated under `## norm`.

## scan

**Mechanism row, not a workload archetype.** Unlike the route-only rows below, this key has a byte
model and prices fine — what it does not have is a kernel family behind it. A scan is almost always
*inside* another archetype (a decay accumulation in a linear-attention chunk, a prefix sum in MoE
bucketing), so if you arrived here by naming `scan` for a kernel whose body does something else,
go back and settle the archetype against the body. Read this row for the carried-value mechanism;
take the tile and the bound from the row that actually describes the kernel.

- **shape** — `reduction-elementwise.md ## Logical tiles` for the tile, then
  `../gluon/appendix-api.md` for `gl.associative_scan`'s surface and the combiner's constraints.
- **pipeline** — the carried value serializes the loop, so overlap comes from independent lanes /
  rows rather than from a deeper ring. Treat it like the `linear_attention` row's recurrence.
- **gate** — the combiner must be associative and is traced, not called. A reverse scan is a
  parameter, not a second pass.

## gather

**Mechanism row, not a workload archetype** — with one exception worth keeping straight. Indexed
access is usually the *implementation* of another archetype's operand movement (MoE's token-to-expert
transport, paged attention's KV addressing), and in that case this row is the mechanism reference
while the bound and the tile come from the owning row. Price it as `gather` **only** when the
indexed movement is the whole kernel. Do not fold this key into `reduction` or `elementwise` to
tidy the list: its byte model is `b*n_gathered` against their `b*(n_in+n_out)`, and `n_gathered` is
irregular and cache-sensitive, so the merge would misprice a standalone gather rather than simplify
it. For the same reason, calibrate with `--measured-dram-mb` before quoting a ceiling here.

- **shape** — `../gluon/layout-reference.md` (register-level, layout-conditional) and
  `../gluon/smem-lds-reference.md` (LDS-level) — the two levels of runtime-indexed access, routed
  in `../gluon/index.md`.
- **pipeline** — the index chain is the dependency, not the payload. Overlap the index load with
  the previous payload's consumption rather than deepening the payload ring.
- **gate** — `gl.gather` is warp-local **only** under the layout condition on the layout page;
  otherwise it falls back to a shared-memory path whose scratch is sized by the whole source
  tensor, so the cost lands on occupancy. LDS `.gather`/`.scatter` are 3.7.0+.

## linear_attention

**Route-only archetype: `hw_budget.py` has no byte/FLOP model for it.** Declare what the kernel
moves with `--tensors` and `--flops <n> --flops-engine mfma|valu` instead. That is not a gap to
work around silently: the chunked recurrence's traffic depends on the chunk size and on whether
the state stays in registers, which no name-keyed model can know.

- **shape** — `linear-attention.md ## The loop shape, and where the state lives`.
  Gated DeltaNet, delta-rule and the linear-attention family carry a state across chunks instead of
  normalizing a score matrix within one, so the attention rows above do not describe this loop.
  The rescale-fold form: `../gluon/matrix-reference.md`.
- **pipeline** — `linear-attention.md ## The Independence rule is not what blocks you here`,
  then `linear-attention.md ## Getting the pipeline, and two ways of being misled about it`.
- **gate** — none in the capability matrix: every primitive is present on all four versions, so the
  gate here is a budget rather than a capability. The register-resident state is the budget line
  that moves; whether it is also the **limiter** is a separate reading, and it can come back empty
  — in which case the occupancy levers are inert and the fork sends you the other way. Do not read
  the budget line as the answer:
  `linear-attention.md ## The state is an occupancy budget line; whether it is the limiter is a separate question`
  and `../tile-programming/slicing.md ## Budget to compute first`.

## collective

**Route-only archetype: no byte/FLOP model.** Cross-GPU traffic is not in the six resources
`hw_budget.py` prices, and a single-GPU ceiling quoted for a kernel whose critical path is a peer
wait is worse than no ceiling. Scope the measurement to the local work and say so — concretely,
`--workload collective` will ask you for `--tensors`, and what you declare there is **this rank's
local traffic only**, with the peer traffic named in prose as an unpriced term rather than folded
in. A manifest that quietly includes the remote side turns a communication bound into an apparent
memory bound.

- **shape** — `collective.md`. For the fused case only: an allreduce-norm-quant,
  a decode step consuming a peer's KV shard. If the communication is a separate launch, this is the
  `norm` or `gemm` row plus a host-side schedule.
- **pipeline** — ordering comes from the atomic protocol, not from a ring
  (`collective.md ## The one primitive: an atomic carrying sem and scope`).
  Spinning has a hoist that will eat the wait:
  `collective.md ## Spinning, and the hoist that will eat your wait`.
  Forward progress between blocks is a launch option, not an assumption — and that option is the
  only language-level guarantee there is, while both surveyed production stacks chose structures
  that do not need it:
  `collective.md ## Co-residency is a launch option, not an assumption`.
  This is the one archetype where inline asm is routinely a *program* and not a missing
  instruction: surveyed cross-GPU collectives carry labelled spin loops, `s_cbranch`, EXEC
  save/restore and `buffer_inv` / `buffer_wbl2` inside a single asm body
  (`../gluon/inline-asm-reference.md ## Class 4b — protocol blocks`). Read that section **before** the per-instruction
  advice on the rest of that page, which does not reach a protocol block. The ordinary
  publish/consume pair still needs none of it — `collective.md ## Publish and consume` first,
  always.
- **gate** — no capability cell: the primitives are plain builtins on all four versions. The trap
  is the **default** `scope="gpu"`, which is correct on one GPU, usually correct at TP=2, and wrong
  at TP=8 — silently, in all three cases.

## prologue

**Route-only archetype: no byte/FLOP model, and often no hot loop either.** This is the
per-token, no-loop, no-LDS kernel that precedes attention — QK-norm, RoPE, the paged KV-cache
write, per-token quantization. It matters as its own row because the layer backbone assumes a hot
loop to pipeline, and **two of its layers go inert while the rest do not**: the missing loop takes
out layer 4 and, with it, layer 4+ — everything else still has something to act on, and this row's
own output usually lands there. Which layer is live, and why:
`prologue.md ## Which layers act when there is no loop`. Price it with **`--tensors ... --flops 0`**
— both flags, because `--tensors` alone gives bytes only and the tool refuses to guess a FLOP count.
Typing the zero is the point: it is a real answer, not a missing one, and it says the compute floor
cannot bind.

- **shape** — the bound is lane-exchange and layout: `prologue.md ## Vector width is arithmetic, not a knob`
  for what sets the per-lane extent, then `../gluon/layout-reference.md` and
  `../tile-programming/layout-recipes.md` for the spellings. Rotary pairing is a lane exchange
  **in two of its three layouts and a real gather in the third** —
  `prologue.md ## Rotary pairing has three layouts`. The paged index chain is routed by
  `../gluon/index.md`'s runtime-indexed-access row.
- **pipeline** — usually **none to build**, and that is a finding rather than a gap. Before
  proposing one, confirm there is a loop at all:
  `../tile-programming/pipeline.md ## Where the overlap comes from, and it is not the same question per tier` has no row for a body without one,
  and a launch-per-token grid is a layer-6 scheduling question instead. The one overlap that does
  exist here is the index chain, not the payload ring: `prologue.md ## The paged index chain`.
- **gate** — `../hardware/capability-matrix.md` for the transpose / lane-exchange rows. If the
  quantization epilogue writes a scale for a downstream GEMM, the group-size trap in the `norm`
  row applies here too, and the scale layout is a contract with the consumer's instruction shape:
  `prologue.md ## The quant epilogue's scale layout is a contract`.

## routing

**Route-only archetype: no byte/FLOP model, and the gap points in both directions.** This is the
MoE metadata kernel — topk over the router logits, the counting sort or prefix sum that buckets
tokens by expert, the expert-offset table. It is its own row because the two nearest modelled rows
fail it in opposite directions: `moe`'s entire model is expert **weight** bytes and this kernel
reads none of them, while `reduction`/`elementwise` assume bandwidth is the floor when the
constraint here is the **length of the sort/scan dependency chain**, which no byte formula on this
page has a term for. Price it with `--tensors` (logits in, indices and offsets out) plus
`--flops --flops-engine valu`, and expect the floor to sit far below the measured time — that gap
is latency, so a byte-removal round collects nothing from it.

- **shape** — the kernel is `moe.md ## Which tier the routing kernel belongs on`;
  what it must *produce* is `moe.md ## Routing metadata: precompute it once, look it up in O(1)`,
  and that output shape is the real design decision here, because it is what makes the consumer's
  lookup O(1) instead of a search. The consumers are two other rows: `## gather` owns the index
  chain that reads this table, `## moe` owns the weight traffic this kernel never touches.
- **pipeline** — usually **nothing to build**, and for a different reason than `prologue`'s. There
  may well be a loop; it is the serial dependency *inside* the body that leaves no independent work
  to overlap with. The lever that exists is shortening the chain, not staging it — and a prefix sum
  in a Gluon body has to be built from `gl.associative_scan` with an explicit combine, which is a
  mechanism the `## scan` row owns rather than an archetype of its own.
- **gate** — the hardest kind: a **language** boundary, not a capability cell or a tuning result.
  `gl.sort` / `gl.topk` / `gl.cumsum` are absent on all four versions
  (`moe.md ## Version gates`). **Which way that cuts depends on how the top-k is written, and both
  answers are real.** A sort-shaped top-k loses its two primitives and gains no tile structure by
  moving, so leave it in plain Triton and say why. A **wave-collective** top-k — ballot, find-first-
  set, lane extract, prefix popcount — is missing from plain Triton too, and only Gluon can pin the
  lane-to-element map those constructions require; in surveyed production source the shipped routers
  are overwhelmingly of that second kind. Decide by expressibility, not by kernel size, and read
  `moe.md ## Which tier the routing kernel belongs on` before recording either verdict.

## When no row fits

Say so rather than forcing the nearest name — a stretched archetype gives `hw_budget.py` the wrong
byte model and gives you the wrong reading list. Declare the tensors and FLOPs directly
(`--tensors name:dir:dtype:dims[:xN] --flops <n> --flops-engine ...`; **`dir` is `r` / `w` / `rw`,
dims are `x`-separated and tensors comma-separated** — `"q:r:bf16:32x8192x128, o:w:bf16:32x8192x128"`
— and a wrong `dir` or a comma inside the dims exits with an argparse error, not a budget), record
the mismatch through
`../method/triage.md`, and route by the BOUND instead of by the name
(`../hardware/bound-class-signals.md`).
