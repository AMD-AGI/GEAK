# Warp-pipelining — the wave-level schedule, and what the compiler does with it

Read this once `scheduling-model.md` has picked **inter_wave**. That page decides *which* model;
this one is the mechanism: how the two-group phase offset is actually built, the rules the kernel
author owns, what the compiler emits at each boundary, and the failure modes that read as "the
marker did nothing". In the overlap order (`pipeline.md ### The order to reach for these in`) this
is part of rung 3: it schedules a structure that register prefetch and the authored ring (rungs 1–2)
already built, and it ranks above re-injecting plain's pipeliner, which is rung 4. Written for
gfx950 (the CDNA4 tutorial kernels and the 3.8.0 marker spelling); gfx942 differences are noted
where they change something.

Warp-pipelining is a **phase-shifted, barrier-rendezvous scheduling scheme**. It splits a
workgroup's waves into two groups and keeps them permanently out of phase: one group runs a
compute stage while the other runs a memory stage, then they swap at a rendezvous. It does not
reduce memory latency — it hides latency behind *the other group's* compute. That makes it a
compute-utilization technique and nothing else, which is what the first gate below tests.

## Gate 0: does the launch have two wave groups at all?

**One question, and it is arithmetic rather than pressure: is `num_warps >= 8`?**

The phase offset's group size is computed in the pass as `warpSize * 4` threads, and the prelude
computes `warpIDX = tid.x / (warpSize * 4)`. **The pass never reads `num_warps`.** So at
`num_warps <= 4` a CTA is at most one group wide, `warpIDX` is identically 0, the
`cond_barrier(warpHigh)` never fires, and the offset does not exist — **with no diagnostic of any
kind**. The markers still compile and still emit their cluster barriers.

**Carry the arithmetic, not the 256, because the 256 is the part that does not travel and the
threshold is the part that does.** A CTA is `num_warps * warpSize` threads and a group is
`warpSize * 4`, so the group count is `num_warps / 4` and **`warpSize` cancels out**. More than one
group therefore needs `num_warps > 4`, and `num_warps` is asserted to be a power of two, so the
threshold is `num_warps >= 8` on wave64 **and on wave32 alike** — what differs between them is the
thread count behind it (512 on CDNA, 256 on a wave32 target), not the warp count. Writing the gate
down as "256 threads" invites the wave32 mis-derivation "the group is only 128 threads there, and
`num_warps = 4` already reaches 128, so 4 is enough" — it reaches exactly **one** group, `warpIDX`
is still identically 0, and the offset is still absent. That mis-derivation is off by one power of
two and it fails exactly as silently as the case this gate exists to catch, which is why the rule
is written as a warp count rather than a thread count.

### The upstream sample you will find for this on CDNA says 4

Worth knowing before you copy one, because the search a CDNA author runs lands on the wrong file.
Upstream's AMD Gluon *examples* (`third_party/amd/python/examples/gluon/`, non-empty on all four
versions) all target gfx1250, and they do launch the marker at `num_warps = 8` — but the whole
kernel around them is WMMA + TDM + `mbarrier`, so there is nothing there to lift onto gfx942 or
gfx950 (`../gluon/appendix-api.md ## What upstream cannot tell you here`). The one upstream artifact
that compiles `warp_pipeline_stage` **for a gfx9 target** is a test — `test_warp_pipeline_gfx9.py`,
3.8.0 only — and it pins `NUM_WARPS = 4` with `arch="gfx950"`.

**That 4 is correct for what the test is doing and wrong for what you would be copying it for.** Its
assertions count `s_setprio` occurrences and cluster structure in the `amdgcn`, and by the previous
section those are emitted at any `num_warps` — it is grading the *wall*, not the offset, so it never
needed two groups (`### Before reading a "no" as a defect` below). Take its spelling of the marker
and its `tl.range(..., loop_unroll_factor=)` interaction; do not take its launch. The API docstring,
which is the only upstream material written to *teach* this marker, names no `num_warps` at all.

This is why it cannot ride on Gate 1's occupancy row below. **2 waves/SIMD and 8 warps/CTA are not
the same thing:** two 4-warp CTAs resident on the same SIMD satisfy the occupancy question and
still give you one wave *group*, because the grouping is within a CTA. Occupancy is a background
condition; `num_warps` is a construction condition, and a kernel can pass all four of Gate 1's
questions while failing this one.

**The `num_warps` axis also splits the stock compiler help, so decide it knowingly.** Upstream
3.8.0's `coexec` machine-scheduler hook only fires at `num_warps <= 4` (automatically on gfx1250;
opt-in on gfx950 / gfx942 via `TRITON_HIP_USE_COEXEC_SCHEDULER=1` or `llvm_fn_attrs`), while the
phase offset only exists at `num_warps >= 8`. A launch picks at most one of the two; the per-compile
`llvm_fn_attrs` strategy is the one scheduler lever available on both sides
(`instruction-scheduling.md ## llvm_fn_attrs — the portable, per-compile scheduler strategy`).

**A "no" here is not a defect report.** If the marker is in the source anyway, or you were about to
add one, go to `### Before reading a "no" as a defect: are you here for the wall, not the offset?` —
the marker has a second use that does not need two groups, and reading a dead offset as four
separate Gate-1 failures is the mis-read that section exists to prevent.

## Gate 1: is this kernel a candidate?

Four questions, in this order. A "no" at any of them means the rest of this page describes
something that will not pay, and two of the four are budget facts you already have before writing
any code.

| Ask | Disqualifying answer | Why it is fatal rather than merely weak |
| --- | --- | --- |
| **Is the bound class matrix issue?** | memory- or bandwidth-bound | The technique buys compute utilization by covering memory with *another group's* compute. With no compute to hide behind, the added barriers and register pressure are the entire effect. |
| **Does the loop stage operands through LDS?** | no LDS traffic in the loop | With nothing staged there is nothing for the other group to overlap. The marker still compiles and still emits its cluster barriers, so this failure is silent — a census showing markers present and `ds_read`/`ds_write` at zero is this case, and it is a kernel-structure problem, not a scheduling one. |
| **Can occupancy reach 2 waves/SIMD?** | register or LDS pressure caps it at one | One resident wave is one instruction stream, and there is nothing to interleave. Also silent: it compiles, runs, and does not overlap. Predict it from the accumulator's per-lane registers *before* building. |
| **Is the matrix op the unscaled path?** | scaled MFMA (fp8/fp4) | Not an automatic no, but the presumption flips. Scaled MFMA decouples into a hidden scale-load window plus the compute window, and paired SIMDs share one shared-read issue bus, so two in-phase groups can collide there and go shared-read-throughput-bound where the unscaled path does not. A single resident wave is structurally immune. Measure the shared-read utilization separately before choosing this model (`scheduling-model.md ## inter_wave hazard: scaled-MFMA vs the shared-read bus (arch fact, low-precision path)`). |

By kernel family, what that resolves to:

- **GEMM** — the designed-for case. Two stages, one moving LDS to registers and one issuing the
  next global loads and computing.
- **Attention** — a candidate, and the interesting one, because a third instruction category
  (the softmax's vector math) competes for the same SIMD. Four stages, and the placement of that
  third category is a derivation rather than a preference (`../workloads/attention.md`).
- **Elementwise, reduction, and other non-matrix loops** — not a candidate. There is no matrix
  pipe to keep busy, so the premise is absent. These loops want the buffering and prefetch of
  `pipeline.md`, not a wave-level phase offset.
- **Decode / small-batch attention with a large per-lane accumulator** — usually disqualified by
  the occupancy row above, and it is worth predicting rather than discovering: the same property
  that makes the kernel interesting caps residency at one wave
  (`scheduling-model.md ## When steps 1 and 2 both say no — the neither-model cell`).

### The fifth question: is this better than spending the same resources on the ring?

The four gates ask whether the technique *can* pay. This one asks whether it pays *more than the
alternative*, and it is the question that decides where this layer ranks rather than whether it
works. Two facts make it load-bearing on this target:

- **Going to two groups is not additive on top of a tuned single-wave kernel — it spends the same
  budget.** Doubling the resident waves halves the register file each group gets at equal
  occupancy, and registers are what a deep authored ring and a register-prefetch schedule were
  already consuming. So the comparison is not "ring versus ring plus phase offset"; it is "deep
  ring at one wave per SIMD" versus "shallower ring at two".
- **Measured on this target, the four-wave build won that comparison.** The same GEMM built
  four-wave — deeper ring plus instruction scheduling, one wave per SIMD — **led the eight-wave
  two-group build at every K tested**, while the eight-wave build led on in-loop matrix efficiency
  at the shortest K only and decayed from there. The eight-wave build also ran at a much lower
  register count, which is the tell: it was not register-bound, and the four-wave build was
  converting that headroom into something worth more than the phase offset.

The reading to carry is the *fragility*, not the winner: that result **reversed** the conclusion
taken on an earlier toolchain pin of the same two kernels. So treat the choice between deep-ring
single-wave and shallow-ring two-group as a per-build A/B with the build identity recorded beside
it, and do not port the verdict across pins. What does generalize is the ordering it implies —
exhaust the ring and the instruction schedule first, because they are cheaper to build, cheaper
to attribute, and on this evidence not obviously worse
(`pipeline.md ### The four layers, ordered by what production reaches for`).

## Gate 2: does your toolchain have it?

`gl.amd.warp_pipeline_stage` is **version-gated, not probe-gated**, and the gate is sharp:

| | 3.8.0 | 3.7.1 | 3.7.0 | 3.6.0 (downgrade) |
| --- | --- | --- | --- | --- |
| `gl.amd.warp_pipeline_stage` | yes | yes | yes | **absent** |
| `add_warp_pipeline` in `libtriton` | yes | yes | yes | **absent** |

**3.6.0 downgrade:** there is **no marker path at all** — this page does not apply, and the
available pipelining is rungs 1–2 of the order (register prefetch and the authored ring), with
re-injecting plain's own software pipeliner only as the labelled rung-4 diagnostic / last resort
(`../tile-programming/pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`). Check this
before reading further: it is one `hasattr` and it decides whether the rest is actionable.

From 3.7 on, `gluon_to_ttgir` already calls `add_warp_pipeline`, so **no compiler patch is
needed** — the pass bails out unless the loop body carries a border marker, so an un-annotated
loop is untouched.

**And that call site is the only one.** `add_warp_pipeline` appears exactly once in the AMD
backend, inside the Gluon lowering; `make_ttgir` never calls it. So this mechanism is **Gluon-only
by construction**, not by packaging — a marker emitted from plain Triton compiles and is never
grouped, which is the quietest possible failure. Two consequences worth carrying:

- It is one of the few genuinely *positive* reasons to be on the explicit-tile tier at all. Most
  levers here recover something plain had; this one plain cannot express.
- The lowering half runs later and in a **shared** stage (`make_llir`), which is where the
  `s_setprio` and `cond_barrier` sequence is actually emitted. So "the marker survived" and "the
  schedule was built" are two different checks, and the first does not imply the second. The full capability matrix, including the async-copy entry points and the
fork-only `gl.warp_predicate`, is in the Gluon API reference for this pack (the pipeline page
under `references/gluon/`, section ### `gl.amd.warp_pipeline_stage` — the official marker path).

### Before reading a "no" as a defect: are you here for the wall, not the offset?

Gate 1 assumes the marker was written to get a **phase offset**. There is a second, unrelated reason
to write one, and on this target it is the only way to get the thing: `sched.barrier` has **no direct
Gluon spelling**, and a stage boundary emits one as a side effect
(`instruction-scheduling.md ## The declarative hints that are not there` — "reachable indirectly on
the Gluon path ... you get it at the boundary and you do not choose the mask"). So a marker can be a
**scheduling wall**, authored to stop the backend from moving instructions across a point, with the
phase offset neither wanted nor reachable.

The two uses look identical in source and are told apart by the launch, not by the marker:

| | phase offset (what Gate 1 grades) | scheduling wall (Gate 1 does not apply) |
| --- | --- | --- |
| `num_warps` | 8, so 2 waves/SIMD exist to offset | often 1 or 4 — one resident wave, nothing to offset |
| loop | dynamic `range`, so the pass finds an `scf.for` | frequently `gl.static_range` |
| LDS in loop | required, it is what the other group overlaps | may be absent entirely |
| bound class | matrix issue | any, including non-matrix kernels with no MFMA at all |
| what you are buying | overlap between two wave groups | a fence the scheduler may not cross |

**Do not "fix" a kernel into the first column when it was written for the second.** At
`num_warps=1` the four Gate-1 rows all fail by construction and the diagnostic table below will read
that as four separate defects; on a single-wave, no-LDS, non-matrix loop it is instead the expected
shape. Settle which one you are looking at before spending a round: if the launch pins one wave per
SIMD, the marker cannot be an offset, so grade it as a fence — does the `sched_barrier` appear at the
cluster boundaries, and did anything need to stop crossing there? A census cannot separate the two
uses either, but not for the reason a marker census suggests: the stage markers emit `s_setprio`
**only when at least one stage in the loop passes `priority=`** (`## Stage priority` below), and
that is orthogonal to which of the two uses the marker was written for. Where no stage passes one,
the census sees markers and zero `s_setprio`; where one does, every stage gets a `s_setprio` whether
it was authored as an offset or as a wall. So **`s_setprio` in a dump does not prove a phase offset
took effect, and its absence does not prove the marker is missing** — the mnemonic tracks
`priority=`, not intent. Settle it on `num_warps`, not on the mnemonic.

**How much of the wall survives when the offset does not — and what it costs either way.** Three
facts, because "the offset did not happen" and "the marker did nothing" are different statements:

1. **The wall is laid down at parse time, so it outlives most of the failure modes.** The border
   marker is *itself* a `ROCDL::SchedBarrier(0)` the moment the frontend builds it, not a thing
   `WarpPipeliner` emits later — upstream's own frontend test prints it. So if the pass declines
   the loop entirely, or the launch has one wave group, or the tile gate closes, **the
   `sched_barrier(0)` is still in the IR and still fences the scheduler.** The single exception is
   a marker under a constexpr condition that traces false: there the `with` block never exists, so
   there is nothing to leave behind.
2. **When clustering *does* happen, it is not free, and a dead offset does not refund it.** Each
   cluster boundary emits its barrier triple, and the prelude emits an unconditional local
   `ttg.barrier` before any of it. Those are paid whether or not `warpIDX` ever differs across
   waves — so a marker on a `num_warps <= 4` launch buys the fence and pays for a rendezvous
   nothing rendezvouses at.
3. **The exemption that avoids the extra barrier does not recognise hand-issued asm.** If a
   boundary already carries an ignorable barrier — `ttg.async_wait` and friends, the ops with
   `MemWaitOpTrait` — the pass wraps it instead, adding only the two `sched_barrier(0)`s and no new
   barrier. **A hand-issued `s_waitcnt` does not carry that trait.** So layering
   `warp_pipeline_stage` over an authored ring that drains through inline asm (`pipeline.md
   ### Draining below the group: bare s_waitcnt instead of wait_group`) adds the full barrier set
   rather than wrapping what is there: landing your waits on the stage boundaries is necessary, and
   it is not sufficient to make the combination free.

Two counting notes that follow. A marker on the flat path can expand well past its source count —
three authored stage sites becoming twelve clusters is a shape that occurs — so **a census of
markers is not a census of clusters**. And nothing above says what the surviving wall is *worth*:
whether a given residual `sched_barrier(0)` was authored deliberately or is the leftover of an
offset attempt that failed is not recoverable from the IR, and cluster labels carry no semantics to
appeal to. Read these as mechanism, and price the fence by measuring it.

## When the gates say no: partition space instead of offsetting time

A "no" at any gate above does not mean the kernel gives up on wave-level structure — it means the
structure has to be **spatial rather than temporal**. This is the route production attention takes
on this target, and it is worth reading as a designed alternative rather than as a layer somebody
forgot: in production, explicit phase-offset markers are all but unused on attention, and
`cond_barrier`, named barriers and producer/consumer wave roles do not appear at all.

The shape is: give each wave in the CTA its own slice of the data, let each run the *whole*
pipeline on its slice with its own private state, and merge once after the loop.

- **For attention that means a private `(m, l, acc)` triple per wave** — each wave runs a complete
  online softmax over its own share of the keys, and the loop ends with one weighted merge across
  waves. The precondition is exactly the mergeability of online softmax: the triples combine by a
  rescaled sum, so partial results from disjoint key ranges are recoverable. Check it before
  designing around it — an epilogue that gates or quantizes must be deferred until after the
  merge, or it cannot be merged at all.
- **The partition degree is a function of the M tile, not a constant.** Production derives it from
  the query-block size — fewer parts at the largest block, more as it shrinks — and the criterion
  is structural: once `BQ / warps_per_cta[0]` already fills the matrix instruction's M extent,
  partitioning further buys no additional independent work and only adds merge cost.
- **It buys the loop body's synchronization back.** With disjoint slices and private state there
  is nothing to rendezvous on inside the loop, so a fully partitioned key loop can carry no
  barrier and no wait at all — which is the opposite trade from this page's mechanism, where the
  rendezvous *is* the mechanism.

The reason this is the AMD answer rather than one option among several: the primitive a
producer/consumer split needs is a per-wave predicate, and the Gluon language surface has none —
no `warp_id`, no `lane_id`, and every route from a tile to a 0-d condition is a whole-tile
reduction, so a condition is necessarily CTA-uniform. The fork-only marker that would provide it
is not in any upstream version (`scheduling-model.md ## Four paradigms, and why this target has three`).
So on this target: phase offset where the gates pass, spatial partition where they do not, and
warp specialization nowhere.

## What it is not

Five nearby things share vocabulary with it, and each confusion sends you somewhere different.

- **vs. a scheduling fence you happen to get for free.** The marker's *side effect* is a
  `sched_barrier` at each cluster boundary, and because Gluon exposes no other way to emit one, a
  kernel may carry the marker purely for that. Then none of Gate 1 applies: there is no phase offset
  to grade, and the launch usually proves it by pinning one wave per SIMD. Grade it as a fence
  instead — see `### Before reading a "no" as a defect` above.

- **vs. instruction scheduling.** Warp-pipelining decides *which stage boundaries exist and which
  stages may overlap*. Instruction scheduling reorders *within* the synchronization those
  boundaries already established. Stage relationships are semantic, so they have to be explicit
  before lowering flattens the program into instructions — a backend scheduler cannot recover
  them. The two compose: `llir-codesign.md` operates inside the regions this page creates.
- **vs. warp specialization.** Both give different roles to different wave groups, but
  specialization runs **intentionally different code** per group (a producer doing DMA, a
  consumer doing math) while warp-pipelining runs the **same** kernel logic at a temporal offset.
  Specialization is a work-partitioning shape; this is a scheduling shape. CDNA lacks the
  primitive specialization needs — see `scheduling-model.md ## Four paradigms, and why this target has three`.
- **vs. double buffering / "ping-pong buffers".** A ping-pong *buffer* swaps two memory banks so a
  copy can fill one while compute drains the other. That is data movement and is **orthogonal**:
  a warp-pipelined kernel normally does both, and conflating them is how a buffering change gets
  credited to the schedule. Buffering lives in `pipeline.md`.
- **vs. BlockPingpong.** Same runtime target, different representation layer. BlockPingpong
  encodes a mostly concrete schedule directly in transformed IR and inserts synchronization as it
  builds it; warp-pipelining encodes **stage structure first** and decides concrete
  synchronization during lowering. The lineage is worth knowing and is at the end of this page.

## The phase offset, and how it is built

This is the load-bearing part, and it is invisible in the kernel source: the author writes
stages, the conversion builds the offset.

The enabling primitive is a **conditional barrier** (`amdg.cond_barrier`), which has three
properties that matter and no others:

1. it executes a barrier **only for the selected waves**,
2. it deliberately **diverges** execution to create the offset, and therefore requires an explicit
   **reconvergence** later, and
3. it sets **no memory fence** — it is an execution rendezvous only.

Property 3 is why the offset is cheap. Shifting two groups relative to each other must not drag
memory-ordering waits into the schedule; if it did, the shift itself would cost what the schedule
is trying to save.

The conversion wraps the pipelined loop in three parts:

- **Pre-loop** — a local barrier drains outstanding synchronization, then one group is held with
  `cond_barrier`. The split is computed from the thread id: `warpIDX = threadIdX /
  threadsPerPipelineGroup`, and the barrier is taken by `warpIDX != 0`.
- **In-loop** — every stage boundary re-locks the phase (next section).
- **Post-loop** — a complementary `cond_barrier` reconverges the groups.

Conceptually the offset needs nothing but control flow and a standard barrier: route some waves
through a helper block containing a barrier while the others skip it, then have everyone reach a
common barrier inside the loop. **Convergence happens because all participating waves reach the
same barrier PC, not because they followed the same path to get there.** `cond_barrier` is the
analyzable form of that trick. Knowing this is what lets you read the lowered code: if you see an
`s_barrier` under an `exec`-masked branch before the loop, that is the phase shift, not a bug.

### Why the legacy priority-only ping-pong was not enough

Before conditional barriers, the AMDGPU-style approach was priority toggling alone: `s_setprio 0`
before the MFMA block, `s_setprio 1` after the first MFMA, `s_setprio 0` after the last. Once one
wave enters the MFMA series its priority rises, so a competing wave cannot issue its own MFMA
until the first returns to low priority. That produces a crude compute/memory overlap and
**leaves synchronization inefficiency untouched**, which is usually the binding constraint once
shared memory is involved. Priority is still used — but as a stabilizer on top of a rendezvous,
not as the rendezvous.

## Stage priority: memory outranks compute

The single most consequential parameter, and the easy one to get backwards.

```python
with warp_pipeline_stage("mem", priority=1):     # LDS reads, refills, address VALU
    ...
with warp_pipeline_stage("mfma", priority=0):    # the matrix ops
    ...
```

The reason is issue-slot contention, and it is worth stating in full because the intuition
("compute is the important work, give it priority") points the wrong way. **Both clusters contain
VALU** — the memory cluster needs it for address arithmetic — and the two resident waves share
one SIMD's issue ports. If the compute cluster has the higher priority, its wave monopolizes
those slots for every vector op, the memory wave cannot advance its address updates, its next
loads never launch, and the overlap the whole scheme exists for is eliminated. Giving the
**memory** cluster the higher priority keeps it able to make forward progress underneath the
compute group.

Three independent sources agree on the ordering: the upstream pass's own rationale (inherited
from the BlockPingpong chained-dot schedule), the shipped warp-pipeline GEMM example, and the API
docstring's example. So if a reference sets priorities with **compute above memory** *on the marker
path*, treat it as a probe or a slip rather than a pattern to copy.

**That qualifier carries the whole rule, so state it explicitly: this is a claim about the marker
path**, where two wave groups contend for the same issue slots and starving the memory group's
address updates costs the overlap. The hand-rolled form — bare `s_setprio` through inline asm —
usually runs in a single-group kernel at `num_warps` of 1 or 4, where **there is no second group to
starve**, and there both polarities are shipped forms. Do not carry this ordering across to that
case; the polarity there is a per-shape parameter, not a default
(`../tile-programming/instruction-scheduling.md`).

**But "compute above memory" and "memory above compute" are not the only two options — the third
is to set nothing at all, and shipped kernels do use it.** Both of the gfx950 tutorial attention
kernels write bare `warp_pipeline_stage("dot1")` with no priority anywhere, leaving the pass to
place `setprio` at section ends on its own. That is a deliberate choice, not an omission: the
`s_setprio` the conversion emits at the boundaries may already be what the schedule wants, and an
explicit priority overrides a decision the pass was making correctly. Read an absent priority as
"the author let the pass decide", and reach for an explicit one when a measurement says the
default is losing.

Three mechanical details. **The second one is load-bearing for two tables elsewhere on this page**
— the Gate-1 row on LDS traffic and the Diagnosing table both used to pair "markers present" with
"`s_setprio` emitted", and that co-occurrence does not hold: with no `priority=` anywhere the
markers emit none, so the presence or absence of the mnemonic in a dump is evidence about
`priority=`, not about whether the marker is there or which of its two uses it was written for
(`### Before reading a "no" as a defect`).

- **Valid range is 0–3**, lowered directly to the `s_setprio` operand. Outside that range is an
  assertion, not a clamp.
- **If any stage in the loop sets a priority, every stage that does not is reset to 0.** If no
  stage sets one, no `s_setprio` is emitted from the stage markers at all. So `mem=1` with `mfma`
  left bare is equivalent to the pair above, and "I removed the priority from one stage" is not a
  no-op — it silently pins that stage at 0.
- **Where the hint lands is not fixed across builds.** On CDNA, MFMA lowering relocates a
  `s_setprio` that sits immediately before a matrix op to *after the first one*, so the priority
  applies during the sequence. That relocation is pass behaviour, and pass behaviour moves: one
  toolchain re-pin changed where the conversion places `setprio` relative to the loop's
  wrap-around barrier and cost several points of in-loop matrix efficiency on attention kernels
  with no kernel change at all (`../pitfalls/platform-known-issues.md`). Record the build identity beside
  any priority measurement.

Priority is a hint to the hardware scheduler, so its effect depends on the dynamic interaction of
the two instruction streams. Treat a priority change as a measurement, not a derivation.

## The launch configuration

The conversion's model is concrete: **a workgroup runs on 4 SIMDs with 2 waves per SIMD**, and
warp-pipelining splits those into two groups of one wave per SIMD. The group size is computed as
`warpSize * 4`, so it scales with the architecture rather than being fixed.

On CDNA (64-lane waves) that is `num_warps = 8` and a 256-thread group; typical launches also pin
`waves_per_eu = num_warps // 4 = 2`. **That is the launch a phase offset needs, not a precondition of
writing the marker at all** — a kernel using the marker as a scheduling fence commonly launches at
`num_warps = 1` or `4`, which is a correct configuration for that use and a disqualifier only for the
offset (`### Before reading a "no" as a defect` above). The layout consequence is smaller than it looks: doubling the
waves changes `warpsPerCTA` from `[2,2]` to `[2,4]`, and if the layouts are built **parametrically**
from the wave-grid constants, every other layout follows mechanically — the global-load layouts
gain a warp dimension while the shared, dot-operand and MFMA layouts are wave-count-independent
and reused verbatim. Build them parametrically before converting a kernel, or this is a rewrite
instead of a one-line edit.

**One class of operand does not follow mechanically, and it fails quietly: a side operand whose
layout inherits the warp tiling.** The scale tensors of a scaled-MFMA kernel are the case. Their
layout is derived per warp, so doubling the warps along a dimension **halves the bytes each thread
contributes** on that side — and the wide transposing LDS read the 4-wave kernel used has a
minimum per-thread width. Fall below it and the read silently decomposes into many narrow reads
plus a pile of lane-shuffles to reassemble what one instruction used to fetch. Nothing errors; the
matrix efficiency just lands far below what the same kernel reached at 4 waves, and the symptom
(a slow inner loop) does not point at the layout.

The fix is not to re-tile the scale: **load the full un-split scale block so the per-thread width
is preserved, then recover the halves with a register-only slice**, which is free when it keeps
the source layout (`llir-codesign.md ## Attention: the co-execution budget` for the slice
discipline). Check this before porting any scaled kernel: compute bytes-per-thread for every side
operand at the new wave grid and compare against the width the LDS read you depend on requires.

**Occupancy is the background condition for everything above — the *second* reason a schedule can
degenerate, not the first.** The first is `num_warps <= 4` (`## Gate 0`), which kills the offset by
compile-time arithmetic no matter how much occupancy you have. Given two groups, occupancy still
has to hold: if register or LDS pressure caps residency below 2 waves/SIMD there is one instruction
stream and nothing to interleave, and the schedule degenerates silently — it still compiles, it
still runs, it just does not overlap. Confirm both after the first build: the achieved wave count,
and that the launch really asked for `num_warps >= 8`.

## Writing one: the six steps, and a skeleton to copy

The steps are short because most of the difficulty is in the two decisions already made above
(is it a candidate, and what priority goes where). What remains is mechanical, and the mechanical
part has four places to get it wrong — all four are in the skeleton.

1. **Identify the hot loop.** Start from a *working, non-pipelined* kernel and take the loop that
   dominates. Do not write a pipelined kernel from scratch: correctness and the schedule fail in
   different ways, and debugging them together is what makes this expensive.
2. **Choose the stage count.** Two unless the loop has multiple dependent compute phases — see
   the next section.
3. **Wrap each group of operations in a stage.** Every op in the body must land in one.
4. **Put async waits *between* stages, never inside one.** Inside is a conversion failure; the
   placement rule for *which* boundary is in `## LDS hazards close a stage early`.
5. **Set up multi-buffering with its prologue and epilogue.** At least double, triple is common.
   The pipelined loop does not cover every tile and the leftovers are a correctness obligation.
6. **Set the launch parameters** — `num_warps=8`, `waves_per_eu=2` — and then *confirm* the
   achieved occupancy rather than assuming the launch got it.

```python
import triton.experimental.gluon.language as gl

NUM_BUFFERS: gl.constexpr = 3          # >= 2; 3 is the common choice
LOADS_PER_ITER: gl.constexpr = 2       # A and B, for the wait counts below

# --- Prologue: fill all but one buffer, then wait for the FIRST tile only ------------
for _ in gl.static_range(NUM_BUFFERS - 1):
    producer = issue_global_to_lds(producer, ...)      # async copy, no wait here
wait_for_outstanding((NUM_BUFFERS - 2) * LOADS_PER_ITER)

# --- Main loop: one tile per iteration, two stages ------------------------------------
# NOTE the trip count. The prologue already consumed NUM_BUFFERS-1 tiles' worth of copies,
# so this loop is SHORT by that many and the epilogue below owes the remainder.
for k in range(0, K_ITERS - (NUM_BUFFERS - 1)):        # a real `range`, NOT static_range

    with gl.amd.warp_pipeline_stage("mem", priority=1):     # memory: the HIGHER priority
        a = a_buf.index(consumer % NUM_BUFFERS).load(dot_layout_a)
        b = b_buf.index(consumer % NUM_BUFFERS).load(dot_layout_b)
        consumer += 1                                  # address VALU belongs HERE

    wait_for_outstanding(0)                            # BETWEEN stages, never inside one

    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        producer = issue_global_to_lds(producer, ...)  # refill for a future iteration
        acc = dot(a, b, acc)

# --- Epilogue: drain the remaining buffers, OUTSIDE any stage -------------------------
for i in gl.static_range(NUM_BUFFERS - 1):
    wait_for_outstanding((NUM_BUFFERS - 2 - i) * LOADS_PER_ITER)
    acc = consume_one_tile(consumer, acc, ...)
    consumer += 1

# --- Launch ---------------------------------------------------------------------------
# num_warps=8 -> 4 SIMDs x 2 waves -> two pipeline groups of one wave per SIMD.
kernel[grid](..., num_warps=8, waves_per_eu=2)
```

Four things in that skeleton are the ones that actually break, and none of them is the `with`
block:

- **The trip count and the epilogue.** The loop is short by however many tiles are already in
  flight when it starts, and the epilogue owes exactly that many. Getting it wrong is a
  wrong-results bug, not a slow kernel, and the epilogue runs **outside** any stage. **The count
  is not always `NUM_BUFFERS - 1`** — see below.
- **The wait between the stages.** It is outside both `with` blocks. Moving it inside either one
  fails conversion.
- **The address arithmetic sits with the memory stage.** Pointer and index updates are VALU, and
  VALU in the compute stage competes with the matrix ops for issue slots.
- **`range`, not `static_range`.** The pass looks for an `scf.for`; a statically unrolled loop
  produces none, so nothing is pipelined and nothing warns.

The `wait_for_outstanding` and `issue_global_to_lds` above stand for the arch-specific async-copy
surface, which differs sharply between CDNA and gfx1250 (`## Architecture differences worth checking before porting`).
**On gfx950** they are `gl.amd.cdna4.async_copy.buffer_load_to_shared(...)` +
`async_copy.commit_group()` and `async_copy.wait_group(N)`, with `N` counted in commit groups
(`pipeline.md ### Hand-built buffering rules (correctness + scheduling footguns)`); the runnable,
numerics-checked form is `scripts/pipeline_examples_cdna4.py` C5. **gfx942 downgrade:** with sync
staging there is no group to wait on — the refill is a register round trip (`.store(gl.load(...))`)
and the ordering is a `gl.barrier()`, which is legal *between* stages exactly like the wait;
`scripts/pipeline_examples_cdna3.py` B1 is the runnable marker example there. The gfx1250
descriptor path is in this pack's Gluon API reference under `references/gluon/`.

### How long the drain is, and why the skeleton's formula does not generalize

The skeleton uses `NUM_BUFFERS - 1` because in a two-stage loop those two quantities coincide.
They are not the same thing, and on a deeper pipeline they diverge:

**The drain is however many tiles are in flight when the loop exits**, which is set by the
**pipeline depth** — how many tiles the loop body is simultaneously working on — not by the
buffer count. Read it off the loop body: if one iteration copies tile `j+3`, reads tile `j+2`,
runs the first matrix chain on `j+1` and the second on `j`, the body spans four tiles, so three
are unfinished at exit and the epilogue owes three. A four-stage attention loop does exactly
this with only **two** LDS buffers, so `NUM_BUFFERS - 1` would say one and be short by two.

The same count applies at the other end: a depth-`d` pipeline needs `d - 1` prologue steps to
fill, and those are the straight-line code a trace reports separately from the loop.

| | depth | prologue | drain |
| --- | --- | --- | --- |
| two-stage GEMM, `NUM_BUFFERS` buffers | `NUM_BUFFERS` | `NUM_BUFFERS - 1` | `NUM_BUFFERS - 1` |
| four-stage attention, 2 buffers | 4 | 3 | **3**, not 1 |

So: **count the tile indices live in one iteration of your own body.** `NUM_BUFFERS` answers a
different question — how many tiles can be resident in LDS at once — and it only has to be large
enough that no stage overwrites a buffer another stage is still reading.

This also prices the depth. Each unit of depth costs one prologue step and one drain step outside
the pipelined loop, so a deep pipeline on a short loop spends a large fraction of the dispatch in
code that is not overlapped. That is the `## Diagnosing` row about good in-loop numbers and a
disappointing whole dispatch.

## Authoring rules

Each of these is a conversion failure or a silent loss, not a slowdown.

- **Two stages minimum.** One stage means nothing to overlap, and the pass errors below two.
- **The loop must be a dynamic `range`, not a static/unrolled one.** The pass looks for an
  `scf.for`; a statically unrolled loop produces none, so nothing is pipelined and nothing warns.
- **Barriers and async waits go *between* stages, never inside one.** Inside a stage they fail
  conversion. This is also where the `wait_group` placement rule below lands.
- **Every operation in the loop body must be inside a stage**, be a recognised inter-stage op
  (async wait, barrier), or be the loop's own iterator update. A stray scalar op between stages
  fails conversion.
- **No `scf.for` / `scf.while` inside a stage.**
- **The pipelined loop is short by the number of tiles still in flight**, and those need a
  sequential epilogue after the loop, outside any stage. That number is the pipeline **depth**
  minus one, which equals `NUM_BUFFERS - 1` only in the two-stage case
  (`## How long the drain is`). Forgetting it is a correctness bug.
- **Do not reach for this on a memory-bound kernel.** It converts another group's compute into
  cover for your memory. With no compute to hide behind, the added barriers and register pressure
  are the entire effect.

One deliberate escape hatch: if two stage borders are placed back to back, the frontend inserts a
**dummy cluster** (tagged `triton.warp_pipeline.empty_cluster`) rather than collapsing them, so an
intentional pipeline bubble survives. Use it when you want an empty slot in the schedule; do not
be surprised by it in a dump.

### The no-wait-inside-a-stage rule is what makes layering this over an authored ring hard

Read that rule together with the `authored_stage` model
(`../tile-programming/scheduling-model.md ## authored_stage — realize at the pipeline layer`),
because the interaction decides whether this page applies to your kernel at all. An authored ring's
whole structure is `commit_group()` / `wait_group(N)` inside the loop body — and `wait_group` is
precisely an op that fails conversion inside a stage. So the ring's waits have to land at stage
**boundaries**, which constrains where the stages can be cut, and a ring whose waits cannot be
arranged that way makes this mechanism unavailable for the kernel even though the API is present.

Production bears this out in a way that is worth knowing before you plan a round. On gfx950,
explicit stage markers are rare — and **most of the kernels that carry them carry no
`commit_group` at all**. They are small-M kernels: no ring, therefore no waits, therefore the
stages are trivially legal. The hand-rolled priority form (bare `s_setprio` through inline asm)
shows the mirror image: it **usually** sits on top of a ring, because inline asm is subject to
none of these conversion rules. Usually, not always — one surveyed group of files issues a
single-slot `wait_group(0)` full drain and is *not* on a ring, and the whole-CTA arbitration form
(raised once, never lowered) is not on one either. If you need priority control inside a body that also has waits,
that is the route — with the corresponding loss of everything the pass would have checked for you
(`../gluon/inline-asm-reference.md ## Class 2 — scheduling control`).

A campaign on gfx950 hit this rule directly: a first attempt failed to compile with exactly the
diagnostic above, a second attempt with repaired borders compiled and then produced **numerically
wrong output** (nonfinite values and non-zero empty rows, with the softmax denominator still
correct), traced to a PV/shared-layout synchronization the re-cut stages no longer ordered. The
first failure is cheap; the second is the one to design against.

## What the compiler emits at each boundary

Every stage boundary lowers to a three-part sequence: a scheduler wall, one barrier, another
scheduler wall. The middle element is chosen by whether the boundary actually carries an LDS
dependency.

| Boundary | What is emitted | Why |
| --- | --- | --- |
| stages share an LDS allocation (read/write overlap) | `sched_barrier(0)` + **local-fencing barrier** + `sched_barrier(0)` | LDS writes must be visible before the consumer stage reads |
| stages have no LDS overlap | `sched_barrier(0)` + **bare `s_barrier`** + `sched_barrier(0)` | execution rendezvous only; no fence needed |
| pre-loop | local barrier + `cond_barrier` (one group) | drain outstanding sync, then phase-shift |
| post-loop | `cond_barrier` (the other group) | reconverge |

The scheduler walls on both sides are the point: they stop the backend from shuffling the cluster
structure back together after the conversion built it.

**Fewer local-fencing barriers is better**, and an unexpected one is a diagnosable signal rather
than noise — it means the analysis believes two stages touch the same LDS interval at the same
time. The fix is structural (separate allocations, or buffer indexing that keeps producer and
consumer apart), not a flag.

### Why a fence inside the compute stage is a performance bug

A local-fencing barrier does not only synchronize execution; it pulls in the wait needed to make
LDS writes visible. If that wait lands inside or at the head of a **compute** cluster, the MFMA
stream drains before proceeding — fragmenting exactly the continuous issue the scheme protects.
Hence the rule: **fence only at boundaries that genuinely carry an LDS hazard, and keep the wait
at the end of the memory cluster, never inside compute.**

## LDS hazards close a stage early

Because one group runs a full stage behind the other, an LDS dependency **cannot** be resolved by
an adjacent-only `S -> S+1` barrier. Consider same-slot reuse:

- stage `S` writes an LDS slot,
- stage `S+1` reads it,
- but wave 0 can reach `S+1` while wave 1 is still at `S`.

If the producer-side write is only guaranteed complete *at* `S`, wave 0's read at `S+1` races it.
The counterexample is the cleanest way to hold this: with a one-stage lag, "producer in `S`,
consumer in `S+1`" is exactly the arrangement that breaks, because the consumer arrives while the
producer is still active. The fix is to require the producer to finish by `S-1` and to reason
about the synchronization window across **`S-1 -> S+1`** — two stages apart, not one.

This is the structural reason every `wait_group(...)` sits **before** the MFMA cluster that
consumes the data rather than at the boundary that immediately follows the producing stage. That
placement looks wrong on first reading (draining right before the consuming load would seem
natural) and it is the crux of the design: the wait has to guarantee that *both* groups have
committed their outstanding copies, not just the reader's own group.

**Ring dependencies.** Stages form a logical **ring** across iterations, not a one-way chain: the
first stage of iteration `i+1` can depend on the last stage of iteration `i`, because the buffer
it refills is the one the next iteration reads. A schedule that looks safe within a single
iteration can still violate cross-iteration safety if the wrap-around edge is ignored. This is
why these loops prime the ring with a prologue of copies plus staged waits before entering steady
state.

## Barrier taxonomy

The distinction that matters throughout is **execution synchronization** (everyone reaches the
same point) versus **memory visibility** (a fence or wait that makes prior memory effects
observable). Mixing them up is how a fence ends up in a compute stage.

| Primitive | Role here | Memory semantics |
| --- | --- | --- |
| `s_barrier` | execution rendezvous | **none** — does not order memory by itself |
| `cond_barrier` | conditional rendezvous for the phase shift; needs reconvergence | **none** — explicitly no fence |
| local-scope `ttg.barrier` | rendezvous **and** CTA-wide visibility of shared-memory writes | LDS fence |
| `rocdl.barrier` | barrier that expands like HIP's `__syncthreads()` | includes a threadfence |
| `gpu.barrier memfence [workgroup]` | the MLIR-level spelling of an LDS barrier | LDS fence when required |
| `sched_barrier` (mask 0) | compile-time scheduler wall | none at runtime — it constrains the *compiler* |

## Membar is a standing risk, not a one-time bug

The membar analysis inserts local-address-space barriers where it believes a hazard exists, and it
can also insert one after an async wait. Two behaviours are worth carrying:

- **It can reorder a wait before a barrier.** The intended memory-cluster opening is barrier, then
  wait, then priority; membar may put the wait first, blurring the stage separation you built.
- **The AMD membar filter early-returns when both hazard ops are async**, so a barrier between two
  async LDS loads is never filtered out (`compiler-contract.md`).

Three escape hatches when the analysis inserts a redundant compute-stage barrier guarding a read an
explicit wait already orders:

- **A relaxed/synced shared load**, whose async-wait token the filter recognises and skips. This
  is sound only because the caller already paired the copy with the wait that synchronised the
  wave — used without that pairing it is a race.
- **Separate LDS allocations per region.** The analysis cannot disambiguate sub-buffers of one
  allocation, so a ring driven from a single allocation gets a conservative fence; four
  quadrant half-tiles in four allocations with distinct buffer IDs are disambiguated by
  allocation and the fence is never inserted. This is the structural fix and needs no relaxed
  load.

- **Extending a buffer's live range past the conflict**, with `_keep_alive()` — whose mechanism is
  the opposite of its name and whose cost is a larger LDS interference graph:
  `../gluon/smem-lds-reference.md ## _keep_alive() — the name is the effect, the mechanism is its opposite`.
  Use it where the barrier exists because the allocator *reused* the bytes, not where two live
  buffers genuinely race.

Gate any of them per kernel with the determinism race-test (`../method/benchmark-hygiene.md`). All three
trade a barrier the analysis wanted for a guarantee you are now making yourself.

## Where it runs in the compiler

For orientation when reading dumps:

- **Gluon to TTGIR** — `add_warp_pipeline` runs immediately **before** warp-group allocation. It
  reads the stage markers and builds the cluster structure.
- **TTGIR to LLVM** — `add_warp_pipeline_conversion` runs **after** the async-wait counts are
  finalized and **before** `scf` is lowered to control flow. This is where the `cond_barrier`s,
  the per-boundary scheduler walls, and the fence-vs-rendezvous choice are emitted.

The ordering is the part to rely on; line numbers in the upstream tree move.

## The hand-built schedules this generalizes

Warp-pipelining generalizes a family of hand-built ping-pong schedules, and each primitive it
now selects automatically was introduced to solve a concrete problem in one of them -- so a
symptom you hit maps onto the variant that first hit it. Those variants are **plain Triton's**
lineage (the automatic ping-pong pass runs in `make_ttgir`, not in the Gluon lowering), so the
table and the three transferable details live with the plain tier:
`../tile-programming/pipeline.md ### The hand-built schedules this generalizes (plain lineage)`.

## Partitioning stages

Five principles, in the order they bind:

1. **Separate compute from memory.** Different hardware units, so they can overlap. This is the
   whole point and everything else is refinement.
2. **Keep stage durations roughly balanced.** The shorter stage's group idles at the barrier, so
   imbalance is paid twice per iteration.
3. **Minimize data crossing a boundary.** A value produced in one stage and consumed in the next
   is live across the barrier, and register pressure is what caps the residency the scheme needs.
4. **Put address arithmetic with memory, not compute.** Pointer-update VALU competes with the
   matrix ops for issue slots; keeping it in the memory stage is what keeps the compute stage
   clean — and it is the same fact the priority rule rests on.
5. **Prefer fewer stages.** Each boundary costs a barrier plus two scheduler walls. Start at two
   and only add more when profiling shows imbalance you cannot fix by moving work.

### How many stages

The count follows from how many **dependent compute phases** the loop body has, not from how much
work it contains.

| Stages | Shape | Use when |
| --- | --- | --- |
| **2** | `mem` then `mfma` | One compute phase. Every GEMM, scaled or not. The default; start here. |
| **4** | compute, memory, compute, memory | Two dependent compute phases, so each gets its own memory stage to face. Attention (QK then PV) is the case. |

Going from two to four is not a tuning sweep. It is not something you arrive at by measuring
imbalance and adding boundaries until it goes away — it falls out of the loop's dependency
structure, and the derivation below is what to run on a loop that is neither a GEMM nor an
attention.

### Deriving the stage count from the loop's dependencies

The rule from the priority section — every cluster must pair matrix work with memory work, so
that a wave in one kind of cluster always faces a wave in the other — is also what fixes the
*number* of clusters. Work it in four steps.

**1. List the loop body's operations and the dependencies between them.** For an attention tile
there are eight: copy K and copy V from global to LDS; read K and read V back out; the two matrix
chains (scores, then the accumulator); and the two halves of the vector math between them. Name
them, because the names are what the rest of the reasoning manipulates.

**2. Count the dependent *compute* phases.** Not the operations, not the work — the matrix chains
that cannot overlap each other because one consumes the other's output. A GEMM has one. Attention
has two: the second matrix chain needs the first one's scores after the vector math has
transformed them.

**3. Give each compute phase a memory stage to face.** This is the pairing rule, applied once per
compute phase, and it is what makes the count `2 x compute_phases` rather than anything else.
One compute phase gives `mem`/`mfma`. Two gives compute, memory, compute, memory — and the
memory stages are where the copies and LDS reads for *future* tiles go, so there is real work to
put in them.

**4. Read off the depth, which is not the stage count.** Software-pipeline the result so each
stage works on a different tile, then look at one iteration of the body: it copies one tile,
reads back an earlier one, and runs the two matrix chains on two earlier ones still. That span —
four tiles for the attention shape — is the **pipeline depth**, and it sets the prologue and
drain (`## How long the drain is`), the number of buffers that must not collide, and how much of
the dispatch is outside the pipelined loop.

The output of this derivation is a stage count *and* a depth *and* a drain length, which is why
it is worth doing on paper before writing the loop: getting the depth wrong is a correctness bug,
and discovering it after the fact means rewriting the prologue and the epilogue together.

**What measurement is still for.** Not the count — that is structural. Per-stage duration tells
you whether the stages you derived are *balanced*, which is principle 2 above and is fixed by
moving work between stages, not by adding boundaries. Adding a fifth stage to fix imbalance in a
four-stage loop is the move this section exists to prevent.

### The four-stage shape, and a difference worth knowing about

Two shipped attention kernels use four stages and **place the softmax differently**. Both are
real; the divergence is instructive rather than a contradiction to resolve by picking one.

- **Softmax rides in the compute stages** (the CDNA/gfx950 tutorial kernels). The two compute
  clusters are QK and PV, the two memory clusters carry the copies and LDS reads, and the softmax
  is split in half across the two compute clusters and *balanced* between them by slicing the
  score tile. The derivation — why the vector math must share a wave with the matrix op rather
  than ride with the loads — is in `../workloads/attention.md ## Where the softmax goes, and why it rides with the matrix op`, and the
  budget arithmetic that balances the halves is in `llir-codesign.md`.
- **Softmax rides in the memory stages** (the gfx1250 example): `stage0` QK compute at
  `priority=0`, `stage1` softmax-part-1 plus the V LDS read plus the next K copy at `priority=1`,
  `stage2` PV compute at `priority=0`, `stage3` softmax-part-0 plus the K LDS read plus the next
  V copy at `priority=1`.

What is common to both is the alternation and the priorities: compute stages at `0`, memory
stages at `1`. What differs is which stage the vector math joins, and that is exactly the
decision the co-execution argument turns on — so **do not port the placement across
architectures without re-deriving it**. The gfx950 argument rests on CDNA issue rules (one
instruction per wave per cycle, VALU and memory on separate ports fed from different waves) and
on `s_setprio` being relocated after the first matrix op by MFMA lowering. gfx1250 has a
different wave size and does not relocate `s_setprio` (`## Architecture differences`), so the
same reasoning does not automatically produce the same answer.

## Architecture differences worth checking before porting

The mechanism is the same on CDNA and on gfx1250, but several facts underneath it are not. The
group size itself scales correctly on its own — the conversion computes it as `warpSize * 4`
rather than assuming 64-lane waves — so what needs attention is everything built on top.

| Dimension | CDNA (gfx9) | gfx1250 | What it changes |
| --- | --- | --- | --- |
| wave size / group threads | 64 / 256 | 32 / 128 | half the register file per group at equal occupancy, so tile sizes may not carry over |
| matrix op | MFMA | WMMA | use the arch-appropriate dot |
| `s_setprio` relocation | MFMA lowering moves it **after the first** matrix op | **no relocation** | the priority hint takes effect at a different point; re-tune the values rather than porting them |
| wait model | packed `s_waitcnt` | split per-counter waits | finer-grained in principle; handled by the compiler |
| global-to-LDS path | async copy (`commit_group` / `wait_group`) | descriptor-based mover | the copy and wait calls are different APIs, not renames |
| LDS capacity | gfx950: 160 KiB / 64 banks; gfx942 downgrade: 64 KiB / 32 banks | larger again | more buffers or larger tiles become reachable; a depth sized on gfx950 can cap occupancy on gfx942 |
| BlockPingpong | works | **broken** — it hard-codes a 256-thread split | on gfx1250 the Gluon marker path is the only correct route |

The BlockPingpong row is the one that bites silently: its asymmetric sync assumes 64-lane waves,
so on a 32-lane target it splits the wrong set and the ping-pong is not what the pass thinks it
is.

**RDNA (gfx11 / gfx12 client parts) is out of scope for this page.** Those targets expose `wmma`
and no async-copy surface at all, so the memory stage has nothing asynchronous to issue and the
authored-overlap answer there is ordinary shared-memory staging with an explicit barrier. Treat a
warp-pipeline plan on RDNA as unsupported rather than untested.

**Low precision is a per-kernel question on either architecture.** On the scaled-MFMA path the
two groups' shared-read slots can collide, which is the fourth gate at the top of this page; it
is a property of the scaled path rather than of the target.

## Diagnosing

The failure modes are few and each has a distinct signature. **First check that the kernel was
written for a phase offset at all**: on a marker authored as a scheduling fence the first three rows
below all fire at once and none of them is a defect
(`### Before reading a "no" as a defect`). Three simultaneous "failures" plus a one-wave launch is
that case, not a kernel with three bugs.

| Symptom | Likely cause | Where to look |
| --- | --- | --- |
| markers present, no LDS traffic in the census | nothing was staged through LDS — the marker reorders, it does not move data | this is a kernel-structure problem, not a scheduling one |
| compiles, runs, no overlap in the trace | residency below 2 waves/SIMD | occupancy and the register/LDS resource that capped it |
| overlap present but MFMA issue fragmented | a fence or wait landed inside a compute stage | the boundary table above, then the membar section |
| worse than the non-pipelined version | memory-bound kernel, or a priority assignment with compute on top | the gates at the top, and the priority section |
| overlap works on the unscaled path, not on the fp8/fp4 one | the two groups' shared-read slots are colliding on the scaled-MFMA path | measure shared-read utilization separately (`scheduling-model.md ## inter_wave hazard: scaled-MFMA vs the shared-read bus (arch fact, low-precision path)`) |
| conversion fails outright | a barrier/wait inside a stage, a stray op between stages, one stage only, or a statically unrolled loop | the authoring rules |
| correct in-loop behaviour, wrong results | the epilogue that drains the remaining buffers is missing or mis-counted | the authoring rules |

Measure **per-SIMD MFMA efficiency**, not only wall time: it is the frequency-independent signal
that the overlap is working, and the two can move in opposite directions
(`../hardware/roofline-models.md`).

**And apply the factor.** A trace reports the figure **per wave**. This page's whole subject is
running *two* waves per SIMD, so on every kernel described here the per-SIMD number is
`per_wave x 2`. Forgetting it halves the reading, which reads as a schedule that is not working —
the exact conclusion this page exists to help you avoid drawing wrongly.

Two numbers, not one, and they answer different questions:

- **in-loop efficiency** — how well the loop body is scheduled.
- **loop fraction** — how much of the dispatch is *in* that loop rather than in the prologue and
  drain. A deeper pipeline buys the first and spends the second, so a kernel can be at its
  in-loop ceiling and still lose on wall time to one with a shallower pipeline. If in-loop numbers
  look finished and the dispatch does not, this is the term to check before concluding the kernel
  is done.

## Cross-refs

- `scheduling-model.md` — whether to be on this model at all, and the `authored_stage` /
  `compiler_interleave` alternatives
- `pipeline.md` — the buffering and prefetch structure underneath the stages
- `llir-codesign.md` — what happens *inside* a region once the stages exist
- `compiler-contract.md` — membar behaviour, toolchain identity, build generations
- `../workloads/attention.md` — the four-stage shape and the third instruction category
- `../hardware/lever-cards.json` — lever `warp_pipeline_schedule`
- `../method/benchmark-hygiene.md` — the determinism race-test that gates the barrier-elision escapes
