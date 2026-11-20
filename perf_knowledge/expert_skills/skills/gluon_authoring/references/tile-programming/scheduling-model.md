# Scheduling-model choice (layer 1.5) — where the scheduling intelligence lives

After the equivalence anchor and before the pipeline layer, decide **who overlaps
memory with compute**: the *compiler* (interleave MFMA among the loads within one wave)
or the *kernel's wave structure* (two waves kept phase-offset so one's compute hides the
other's memory). The choice sets `warpsPerCTA`, the occupancy target, and the
register-file use, so it **gates** the pipeline (layer 4) and register/occupancy
(layer 5) layers. It is not a hot-loop tweak — decide it here, realize it there.

Both models answer the *same* bound class — **compute / mfma-issue** (keep the matrix
pipe issuing an MFMA every cycle); latency is hidden by prefetch in either. They are two
places to put the same intelligence.

**Where this sits in the overlap order.** The order itself is defined once, in
`pipeline.md ### The order to reach for these in`: (1) register-level prefetch, (2) the authored
LDS ring, (3) this choice of scheduling model and its realization — `warp_pipeline_stage` for
`inter_wave`, instruction pacing for `compiler_interleave` — and (4, lowest) re-injecting plain's
pipeliner as a diagnostic or last resort. This page decides rung 3; the scheduling mechanisms are
stated once, here (the model choice), in `warp-pipeline.md` (the `inter_wave` realization) and in
`instruction-scheduling.md` (fence, `s_nop`, `llvm_fn_attrs`, hand-rolled `s_setprio`, and the
`sched_barrier` / `sched_group_barrier` / `iglp_opt` family that is not reachable). Other pages
point to these three rather than restating them. Everything below is written for gfx950 first;
gfx942 differences are flagged as downgrade notes.

**The problem here is throughput, not latency, and that is not the usual framing.** Memory
latency is the *easy* half — prefetching solves it by issuing a load many stages ahead of the
MFMA that consumes it. What is hard is that the SIMD must also *issue* those memory
instructions, and every issue slot they take is one the matrix pipe does not get. So the
question is never "how do I hide the latency" but **"which wave issues the memory, and when,
relative to the compute"** — and that is what the models below answer differently.

## Four paradigms, and why this target has three

The choice is one point on a spectrum: **at what granularity is the scheduling decision made**,
and correspondingly how much of the work lands on the compiler rather than on the kernel's
structure.

| Model | Scheduling unit | Compiler involvement | Here |
| --- | --- | --- | --- |
| **`compiler_interleave`** | individual instructions | **very high** | ✓ but build-pinned |
| **`authored_stage`** | one async commit group | **low** | ✓ **and stock** |
| **`inter_wave`** | pipeline stages, across waves | **medium** | ✓ (Gluon only) |
| **warp specialization** | functional roles | **lowest** | **✗ — no hardware primitive** |

> **`compiler_interleave` was called `intra_wave` in earlier notes**, and the rename is not
> cosmetic. "intra_wave" names *where the overlap happens* — inside one wave's instruction stream —
> which is true of **two** of the models above, so using it as the name of the compiler-driven one
> made the hand-authored one unnameable at this node. It went missing as a result: a reader choosing
> between "compiler-interleave, which needs passes your build may not have" and "wave-ping-pong,
> which runs on stock" never saw the third option, which is the one most production Gluon
> kernels on gfx950 actually use. The lever cards and the realize sections below now carry the
> current names (`scheduling_model: authored_stage` / `compiler_interleave` / `inter_wave`);
> `intra_wave` survives only in this note and in older records.
>
> **And the old word is still live elsewhere as a different referent, so the rename does not finish
> the job — disambiguate by asking "layer or model?".** The Gluon pack's pipeline page runs a
> four-layer table whose **layer 2 is called "intra-wave"**, and it means the empty-asm scheduling
> fence, a placed `s_nop`, and the per-compile `llvm_fn_attrs` strategy. That is **not** this node's
> `compiler_interleave`: it is the set of mechanisms in which `compiler_interleave` *and* the
> hand-authored forms are both realized, one layer down from this choice. A model is what you pick
> here; a layer is where you spend a round. Nothing is inconsistent between the two pages, but the
> shared word means a reader can carry a decision from one to the other that neither page made
> (`pipeline.md ### The four layers, ordered by what production
> reaches for`).

**Warp specialization is the absent third, and knowing *why* it is absent is what stops you
reaching for it.** It splits waves by **role** rather than by tile: a compute wave issues only
MFMA, a producer wave issues only the loads and feeds it through shared memory, so the
producer→consumer dependency crosses *between* waves. That is the cleanest overlap of the three
— the compute wave's matrix pipe carries zero memory-issue overhead — and it is what the two
available models are *approximating*. It needs a hardware **async-copy plus cross-wave
synchronization** primitive (on NVIDIA: TMA with named-barrier / warpgroup specialization).
**CDNA has no such primitive**, and the DSL surface reflects that rather than hiding it: the
`warp_specialize` symbol imports on every version, `gluon_to_ttgir` even runs the pass that would
allocate the warp groups, and the lowering then **hard-errors** with `Warp specialization is only
supported on gfx1250, got <arch>` — an explicit arch check, not a subtle miscompile. Read the exact
message as the evidence: it is the difference between "absent" and "present for a different chip"
(`../gluon/pipeline-reference.md ## Authored overlap (no compiler patch)`). Read it as a
**structural** absence,
not a missing binding — no amount of kernel work or compiler sanction reaches it, and a plan
that assumes a producer wave has to be re-planned, not debugged.

## The unit is the region, not the kernel

The table below reads as a whole-kernel choice, and for a GEMM it is one. It stops being one as
soon as a loop body has regions that differ: a kernel can be **inter_wave between its memory and
compute stages and compiler-interleaved inside each compute stage**, which is what happens when a third
instruction category (softmax vector math, a gate, a dequant) has to share a wave with the MFMA.

The discriminator is one question, asked **per region**: *does this region need two instruction
categories to issue from the same wave?* No — the overlap is supplied by the ping-pong and there
is nothing to schedule. Yes — the overlap has to be manufactured by instruction ordering, and
only the compiler can do it.

Two consequences worth carrying: the same intra-wave granularity covers two problems of very
different difficulty (MFMA↔memory is a **throughput** pairing, MFMA↔VALU is a **co-execution**
assignment where every vector op needs a specific MFMA's window), and a tool operating below the
DSL routes by region content rather than by kernel identity. Both in
`llir-codesign.md ## Region routing: the model is a property of the region`. Decide the
whole-kernel default here; re-ask per region at the pipeline layer.

## The three models

| | **`authored_stage`** (hand-built ring) | **`compiler_interleave`** | **`inter_wave`** (wave-ping-pong) |
| --- | --- | --- | --- |
| scheduling intelligence | **you**, via commit-group placement | the **compiler** spreads MFMA among the loads | the **kernel wave structure** (two phase-offset groups) |
| resident waves / SIMD | one is enough | one | **two, or it degenerates** |
| register file | prefetch registers + LDS buffers are the cost | leans on **AGPR** accumulators (frees VGPR for the interleave's extended live ranges) | accumulators stay in **VGPR** (`amdgpu-agpr-alloc=0,0`) |
| toolchain | **stock** — the async-copy surface is a plain builtin (gfx950: 128/32-bit direct-to-LDS; gfx942 downgrade: sync staging, or 32-bit async with an `order=[1,0]` destination) | the RA half is stock per compile (`llvm_fn_attrs`, 3.8.0); the stock `coexec` strategy is opt-in on gfx950 / gfx942 (`TRITON_HIP_USE_COEXEC_SCHEDULER=1` or `llvm_fn_attrs`; automatic only on gfx1250 at `num_warps <= 4`); any other *automatic* interleave needs a pass you author | **stock** (3.7.0+), but Gluon-only: the pass that groups the stages runs in `gluon_to_ttgir` and nowhere else; Gate 0 `num_warps >= 8` |
| lever card | `manual_pipeline_prefetch` | `gemm_compiler_stack` | `warp_pipeline_schedule` |
| mechanism | `commit_group` / `wait_group` rings — `pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)` | per-compile attributes plus an authored pass — `compiler-contract.md ## What upstream 3.8.0 actually gives you` | `warp_pipeline_stage(...)` clusters |
| production use, gfx950 | **dominant** | the portable `llvm_fn_attrs` half: common, decided per kernel (`llvm-fn-attrs.md`) | rare |

**They are not a ladder and they are not exclusive in the way the earlier two-model framing
implied.** `authored_stage` is the base — it creates the overlap. The other two *schedule what it
created*, from opposite directions, and production layers them: instruction scheduling commonly
appears on top of the ring, and the hand-rolled priority form of `inter_wave` **usually** appears on
top of a ring — but not always, so do not read its presence as proof that a ring is there. The one genuine either/or is at the instruction level: a
compiler-interleave pass and a hand-placed fence are competing claims on the same instruction order.

## How to choose

**Start from who is going to build the overlap, not from which scheduler you have.** The earlier
version of this list started at step 2 below, which asked a toolchain question first and therefore
answered "stock build -> `inter_wave`". That is the wrong default: `inter_wave` needs two resident
waves and a stage body free of waits, and when either fails the portable answer on a stock build is
`authored_stage`, not a scoped ceiling.

1. **Is there a K-loop with operands to stage?** If yes, `authored_stage` is the default and the
   base for anything else: the async-copy surface is a plain builtin on the Gluon path, it needs one
   resident wave rather than two, and it is what expresses per-tensor depth and staggered chains at
   all. Realize it at the pipeline layer (`## authored_stage — realize at the pipeline layer`). If
   there is **no** loop — a per-token prologue kernel, a single-shot reduction — none of the three
   models applies and this layer has nothing to decide.
2. **Does the launch have `num_warps >= 8`, is 2 waves/SIMD reachable, and is the stage body free
   of waits?** All three are required before `inter_wave` adds anything on top. Below 8 warps the
   CTA is one wave group and the phase offset does not exist at all — silently, by compile-time
   arithmetic (`warp-pipeline.md ## Gate 0: does the launch have two wave groups at all?`). One
   resident wave leaves no second stream, so it degenerates.
   And the pass **refuses to compile** a stage containing a barrier or a wait — which is exactly
   what an authored ring contains, so the ring's `wait_group` has to sit *between* stages. Where the
   ring cannot be arranged that way, `inter_wave` is unavailable for this kernel even though the API
   exists (details in `warp-pipeline.md`). This is also why the
   production kernels that use `warp_pipeline_stage` without a ring are the small-M ones: no ring
   means no waits means the stages are legal.
3. **Accumulator read cadence.** `compiler_interleave`'s `+ra` rung pins accumulators in AGPR,
   which pays only for a **write-only-until-epilogue** accumulator (GEMM). A
   read-modify-every-iteration accumulator (online-softmax rescale, gating) must stay in
   VGPR — which is what `inter_wave` does natively — so for VALU-between-matmul ops
   `inter_wave` or hand-authored interleave is the fit, and the GEMM-class compiler
   answers do not apply (`compiler-contract.md ## What upstream 3.8.0 actually gives you`).
4. **Toolchain availability, last.** `compiler_interleave` splits on this axis: the RA-hint half
   is reachable per compile on 3.8.0, the *automatic* interleave is not reachable at all without a
   pass you author — apart from the stock `coexec` strategy, which on gfx950 / gfx942 is an
   opt-in to A/B rather than something the build already does. **That is not a ceiling here**,
   because steps 1 and 2 have already given you two stock-reachable models. What a missing pass
   costs you is the *automatic* interleave; the portable per-compile strategy attribute and the
   hand-placed fences remain (`instruction-scheduling.md`).

**Record the model you chose, and record `authored_stage` when that is what you built** — a kernel
logged under a compiler-driven model invites the GEMM-class answers on the next round, which is the trap
the neither-model cell below is about.

### When steps 1 and 2 both say no — the neither-model cell

Steps 1 and 2 can each rule out a different model, and on one real kernel class they do it
simultaneously. **1 wave/SIMD *and* a read-modify-every-iteration accumulator** is the cell:
`inter_wave` is out because there is no second wave, and `compiler_interleave`'s GEMM ladder is out because
it assumes an MFMA -> MFMA accumulator chain — on VALU-between-matmul it does not merely fail to
help, the in-tree scheduler **asserts**. Routing such a kernel to `compiler_interleave` and then reaching
for that ladder is the trap; the ladder is not the model, it is one realization of it.

A large per-lane accumulator is what puts a kernel here — a compressed-latent / large-head-dim
attention forward pushes the accumulator's register count up, and registers are exactly what
caps resident waves, so the same property that makes the kernel interesting closes both doors.
Recognise it from the budget before the profile: predict waves/SIMD from the accumulator's
per-lane registers, and if it lands at one, do not compute a co-execution budget at all — its
first gate has already failed.

What remains, in order:

1. **Buy the wave back, if it is cheap.** The occupancy cliff is the whole problem, so anything
   that sheds accumulator liveness may restore two waves and re-open both models — attack the
   accumulator traffic itself (tighter dtype, fewer live fragments, restructured accumulation
   order) rather than trying to raise occupancy directly. If it works, re-enter at step 1.
2. **Author the interleave by hand.** With one instruction stream the overlap has to come from
   the order you write, not from a second wave — independent `AC`/`LR`/`DOT` staggered
   explicitly (`pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`).
   This is the portable answer and it needs no plugin.
3. **The portable scheduler strategy is *actually* indicated here.** A single low-occupancy wave
   with independent work the default scheduler leaves unscheduled is the ILP-starved case that
   `instruction-scheduling.md` is for — the one place it is a
   first-class candidate rather than a long shot. Sweep and A/B it; it applies to
   VALU-between-matmul, unlike the GEMM ladder.

Record the model as **neither**, not as `compiler_interleave`. The distinction matters downstream: a
kernel logged under a compiler-driven model invites the GEMM knobs on the next round. Note that
`authored_stage` is still available in this cell -- it needs neither a second wave nor the GEMM
ladder -- so "neither model" means neither of the two *scheduling* models, not no overlap.

## authored_stage — realize at the pipeline layer

You place the async copies, the commit groups and the waits, so the overlap exists whether or not
any scheduler cooperates. This is the base model: the other two schedule what this one built, and on
a stock build it is the only one of the three that is unconditionally available.

The decision that defines the model is **`groups_per_stage`** — how many async copies you close
under one `commit_group()`. One copy per commit makes the wait depth equal the stage count and is
the readable starting point; several copies under one commit retire together (cheaper, no per-tensor
control), and several commits per stage buy per-tensor lead distances at the cost of a deeper wait
count. Decide that first and derive `wait_group(N)` from it, because `N` counts **commit groups**.

The rules, the vetted skeleton (gfx950 async ring, and the gfx942 sync-staged downgrade) and the
three production ring shapes are at
`pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`; the API
surface and the per-architecture availability are in the Gluon pack's
`../gluon/pipeline-reference.md ## Authored overlap (no compiler patch)`. Lever:
`manual_pipeline_prefetch`.

One model-level property to carry from there: `wait_group` provides **no CTA synchronization**,
so the `gl.barrier()` placement is a separate decision per hazard
(`pipeline.md ### Three shapes production actually builds, and the barrier placement that differs between them`).

## compiler_interleave — realize at the pipeline layer

Write the explicit independent AC/LR/DOT + prefetched multi-buffer pipeline, then let the
compiler stack interleave it. The knobs **exploit** the structure, they do not create it
(gating law `compiler_lever_needs_paired_structure`) — so this model presupposes
`authored_stage`'s output as its input (or, only as a labelled last resort below the parity gate, a
re-injected pipeline — `pipeline.md ### The order to reach for these in`, rung 4). Details, what upstream 3.8.0
reaches for each capability, the GEMM-only assumption, and Scenario B authoring:
`compiler-contract.md`. The portable half is
`instruction-scheduling.md`. Lever: `gemm_compiler_stack`.

Copy the structural pipeline skeleton, then specialize the independent regions
and compiler-stack settings:

```text
# Keep AC/LR/DOT regions independent; the compiler interleaves them.
AC -> LR -> DOT, with independent prefetched multi-buffer regions
# Specialize prefetch depth, then enable the validated compiler ladder -- the SCHEDULER
# and RA-HINT capabilities. Upstream reaches the RA half per compile and the scheduling
# half only from a pass you author, so take the route from the table rather than from here:
#   compiler-contract.md ## What upstream 3.8.0 actually gives you
```

## inter_wave — realize with warp_pipeline_stage

Split the workgroup's waves into two groups kept permanently a stage apart: while one
group runs its MFMA cluster, the other issues that region's memory, then they swap.
Overlap is a property of the ping-pong, not of within-wave instruction order — so it
needs **no** interleave pass. Lever: `warp_pipeline_schedule`.

**The mechanism is stated once, in `warp-pipeline.md`** — Gate 0 (`num_warps >= 8`), the
candidacy gates, the Gluon-only call site (the stage-grouping pass runs only in `gluon_to_ttgir`,
so this is one of the few genuine reasons to *be* on the explicit-tile tier), how the phase offset
is built and torn down, what the compiler emits at each boundary, the launch configuration, the
authoring rules that fail conversion rather than merely costing time, and a symptom-to-cause
table. Read it before the first build of an `inter_wave` kernel, not after the first disappointing
measurement. What this node needs from it, as a checklist rather than a restatement:

| decide before building | why it matters at the model choice | stated in |
| --- | --- | --- |
| `num_warps >= 8`, then 2 waves/SIMD achieved | below either, there is no second group / stream and the offset silently degenerates | `warp-pipeline.md ## Gate 0: does the launch have two wave groups at all?`, `warp-pipeline.md ## The launch configuration` |
| the ring's `wait_group`s fall on stage boundaries | a stage may hold no barrier, no wait and no loop, and needs at least two stages — a ring that cannot be cut that way rules this model out for the kernel | `warp-pipeline.md ## Authoring rules`, `warp-pipeline.md ### The no-wait-inside-a-stage rule is what makes layering this over an authored ring hard` |
| memory stage at the higher priority | on the marker path, a compute stage above memory starves the address VALU and removes the overlap | `warp-pipeline.md ## Stage priority: memory outranks compute` |
| LDS hazards resolved a stage early (`S-1 -> S+1`) | one group runs a full stage behind; a fence landing in the compute stage fragments MFMA issue | `warp-pipeline.md ## LDS hazards close a stage early` |
| redundant compute-stage barrier elision | relaxed/synced read or per-region allocations — each gated per kernel by the determinism race-test | `warp-pipeline.md ## Membar is a standing risk, not a one-time bug`, `../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)` |

Lever: `warp_pipeline_schedule`; the copyable skeleton is
`warp-pipeline.md ## Writing one: the six steps, and a skeleton to copy`.

### inter_wave hazard: scaled-MFMA vs the shared-read bus (arch fact, low-precision path)

On the scaled-MFMA (fp8/fp4) path, the hardware decouples each scaled MFMA into a hidden
scale-load window + the compute window, and paired SIMDs share one shared-read issue bus.
In the two-waves-in-phase ping-pong the groups' shared-read slots can collide, so the
scaled path can go shared-read-throughput-bound where the unscaled path does not (a single
resident wave is structurally immune because the next MFMA simply waits for its own read).

This hazard means the inter-wave candidate is not automatically better for low precision:
measure scaled-MFMA issue + shared-read utilization separately, and keep the single-wave
`authored_stage` or `compiler_interleave` candidate if the bus is the bound class.

**Abandoning inter-wave is not the only response, and it is usually not the first one.** The
collision is a *window* problem: each scaled MFMA yields only the co-issue slots left after its
hidden scale-load window, and on a narrow instruction shape that remainder is small enough that
the two groups contend for it. A **wider MFMA shape** takes longer per instruction, so the
absolute window it leaves is larger and it admits several shared-read slots instead of one — the
contention disappears rather than being scheduled around.

What makes this cheap enough to try before restructuring: the scaled MFMA is a block operation, so
changing its shape does **not** rewrite the loop body — the compiler re-tiles the same K work into
fewer, larger instructions. The costs to check are the operand layouts it implies and the
resulting register pressure. Diagnose it first: a shared-read stall with the bank-conflict counter
at **zero** is this hazard rather than a layout problem, and the width of the co-issue window for
your shape is the quantity to compare (`../hardware/isa-mechanisms.md`, and note the window is per
*shape*, not per architecture — do not derive one shape's window from another's).

- **Do not combine with `compiler_interleave`'s `+ra` rung by default.** They solve different cadence
  problems and the AGPR pressure can make the two-wave resident count drop from 2 -> 1,
  destroying the ping-pong. Only combine after re-checking occupancy and the determinism
  race-test.

## Cross-refs

- `warp-pipeline.md` — the `inter_wave` mechanism in full: phase shift, boundary emission, launch
  configuration, authoring rules, the BlockPingpong lineage, and diagnosis
- `compiler-contract.md` — `compiler_interleave` passes, ladder, Scenario B authoring; the portable
  half is `instruction-scheduling.md`
- `pipeline.md` — where all three models are realized, and which tier builds the overlap at all
- `llir-codesign.md` — what the models look like **below** the DSL: region routing, the
  throughput pairing vs the co-execution budget, and the gates that decide whether a schedule
  tool can act on this loop at all
- `../hardware/roofline-models.md` — occupancy / issue / LDS diagnostics
- `../hardware/lever-cards.json` — lever cards `gemm_compiler_stack`, `warp_pipeline_schedule`
- `../method/benchmark-hygiene.md` — race-test and determinism gate
- `../method/profile.md` — re-profile and classify the bound class after each candidate
- `../gluon/memory-reference.md` — async-copy and shared-read contracts
