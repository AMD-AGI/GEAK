# Distributed Fusion — Collapsing a Multi-Launch Operator into One Persistent Kernel

The playbook for the `distributed` specialty: an operator that today issues **several serialized
kernel launches per rank** and coordinates with peer ranks, and whose target form is **one persistent
kernel per rank** with computation and communication genuinely overlapped (the "DeepGEMM-style"
shape). `geomean_levers.md` Lever 1 tells you to collapse dispatch count on a single-GPU op; this
file is what changes when the stages also talk to *other ranks*, because then the thing you are
removing is not launch latency — it is a **wait**.

Every number below is measured on a live 8×MI355X (gfx950/CDNA4, 256 CU, 8 XCDs, wave64) EP8 MoE
workload. Treat them as calibration for the *shape* of each effect, not as portable constants.

**Read these three first; this file assumes them.** They are method cards — how to derive the
answer — where this file is a lever list for a shape of problem you have already diagnosed:

- `tile_task_graph.md` — building the tile-level dependency DAG (edge rule, edge scope, what
  enforces each edge today, critical path, slack). **The Analyze phase must emit this as an
  artifact.** Every lever below is an intervention on some edge of that graph; if you cannot name
  the edge, you are not ready to pull the lever.
- `fusion_preconditions.md` — the three-condition test each candidate edge must pass, and the
  cheaper levers to rule out first. It exists so that "do not fuse" remains a reachable conclusion.
- `resource_partition.md` — who gets the CUs once anything overlaps, and where the critical path
  moves next.

**A note on what this file deliberately does not contain.** Earlier revisions carried the specific
missing edges, per-guard outturns and instrumented latencies from the operator it was written
against. Those were removed on purpose. A knowledge card that hands over another operator's answer
does not teach an analysis, it substitutes for one — and a run that reproduces a pre-supplied answer
proves nothing about whether the workflow could have derived it. What remains is the *method* and
the *failure modes*, which transfer. Do not re-add measured results for a specific operator here;
they belong in that operator's run artifacts.

## When this file applies

All three must hold, or use `geomean_levers.md` instead:
- ≥2 GPU kernels per call that are **strictly serialized** (kernel-level DAG shows zero observed
  overlap between them);
- at least one of them exchanges data with peer ranks;
- the profile shows **waiting**, not arithmetic, dominating — e.g. accounted-cycle share for
  `waitcnt`+`barrier` above ~35% inside the kernels, or a cross-rank barrier whose cost explodes
  under load skew.

Diagnostic that settles it in one measurement: run a **no-payload control** (same compute, peer
stores removed). If the stage collapses, the transfer/visibility cost is real and currently
*exposed*, i.e. not hidden under compute, and **that gap is the budget for hiding this transfer** —
no overlap scheme can recover more communication time than the control removes. If the control
barely moves, there is nothing to hide on that payload/visibility edge; it does **not** rule out
CTA-local tail filling or a better phase transition/resource partition (Lever 1b). Report the
narrow conclusion and rank that edge behind the scheduling mechanisms. This measurement bounds the
communication-hiding part of the project, not every possible benefit of fusion.

## Lever 1 — Enumerate the MISSING readiness edges before writing any code

Serialized stages are usually not serialized because they must be. They are serialized because **no
producer publishes and no consumer waits**, so the only ordering is kernel termination + stream
order. You can enumerate every fusion opportunity statically, before touching the kernel:

- Read the producer's epilogue. If it ends at a plain `buffer_store` — no release fence, no atomic,
  a cache hint at most — it publishes **nothing**.
- Grep the consumer for `wait_until` / any acquire. If there is none, the edge does not exist.

Each such (producer, consumer) pair is one missing edge, and each is a candidate. Enumerate them
yourself for the operator in front of you — the count is usually small, and knowing it exactly is
the difference between a plan and a guess. Record each one in the `enforced_by` column of the
tile-level graph (`tile_task_graph.md`): an edge whose only enforcement is `launch_boundary` is a
missing edge in this sense.

Do not assume the intra-rank edges and the cross-rank edges have the same character. They usually
do not: one is a visibility question inside a device, the other is a visibility question across an
interconnect, and they sit at different rows of the edge-scope table. Classify before you plan.

**An edge that does not exist cannot be optimized — it must first be added.** Budget for that: the
publication itself costs something (Levers 5 and 6 are about making it cheap).

## Lever 1b — Build a CTA-local phase pipeline, not a grid-wide sequence

Two stages using the same binding engine do not justify a permanent CU split, but that fact does not
justify a global drain either. The useful persistent schedule is often a **phase-staggered CTA
pipeline**:

1. Give every compute CTA a shard of the upstream and downstream work queues.
2. A CTA drains only its own upstream shard; when that shard has no claimable work, that CTA begins
   downstream work whose readiness edge is satisfied.
3. It does not wait for other shards or for in-flight upstream CTAs. Thus one CTA's local order stays
   producer-before-consumer while the grid contains both stages concurrently.
4. Apply the same rule at the next stage boundary: a CTA whose downstream shard is drained may enter
   reduction/combine while other CTAs still execute downstream compute.

The benefit is not imaginary simultaneous MFMA throughput. It is partial-wave tail filling,
overlap of downstream VMEM/P2P/epilogue work, and removal of grid-wide idle intervals. Therefore the
static `max(A/f, B/(1-f))` argument in `resource_partition.md` rejects only a fixed `f/(1-f)` role
partition; it cannot reject this dynamic phase-staggered schedule.

Load-bearing invariants:

- queue heads are sharded or hierarchical; no single global claim atomic sits on every CTA's path;
- “my shard is empty” is a CTA-local transition, never evidence that the whole grid drained;
- all startup-role CTAs eventually join the compute queues after their startup duty;
- each downstream item waits on its own producer readiness, with the required acquire;
- no global “all producer work done” or “all downstream work done” barrier is inserted between the
  queues;
- the grid remains resident and the queue/readiness graph is acyclic.

Reject a proposed fusion if its pseudocode performs `drain_all(stage1) -> barrier ->
drain_all(stage2)`. It may have one launch and still preserve the original serialization. Require a
same-timeline trace or block-level counters showing upstream and downstream CTA intervals overlap;
stage timer sums are not valid evidence once this happens.

## Lever 2 — Replace whole-phase joins with per-item readiness (highest payoff *only* when the publication is cheaper than the wait it removes)

An all-rank barrier, or any single counter every consumer gates on, forces **every** consumer to pay
`max` over **all** producers. It is where load skew gets exposed, and it is almost always the largest
single line item.

**Caveat that inverts the ranking on multi-die parts — read before you put this first.** "Highest
payoff" is the *potential*, not the *net*. Per-item readiness only pays when the readiness
publication itself is cheaper than the wait it removes, and on an 8-XCD part the naive publication
is *not* cheap: a per-item **system-scope** atomic or release fence lowers to a cross-die LLC
flush (Lever 6 measured `fence_system_release` at **+0.411 ms** over ~200 blocks), which can exceed
the exposed wait outright — a measured **regression**, not a win. So the correct order is: (a) size
the exposed wait (the no-payload control + an instrumented peer-wait timer), (b) size the
per-item publication cost at the scope you would actually need for cross-die visibility (Lever 5)
using the cheap form (Lever 6), and only (c) rank this lever first if (a) > (b) with margin. Where
(b) ≥ (a) — which happens on well-tuned operators whose cross-rank tail is already short — the gain
is **not** on this edge at all; it is in the **resource partition** (who owns which CUs once the
stages are co-resident: `resource_partition.md`) and in the launch-structure wiring, and the
per-item readiness edge should be attempted **last, hardware-gated, with a local (agent/workgroup)
publication and a sharded arrival counter**, never a system-scope per-item atomic as the first move.
For an operator with an existing distilled recipe, defer to that operator's expert skill for where
this lever actually ranked when measured — it overrides the generic "usually" below.

**Measure it, do not assume it.** Instrument the peer wait itself (an in-kernel cycle counter around
the wait, rank-max p95 over ≥20 replays) and compare a uniform route against a skewed one. The
diagnostic you are looking for is a wait that grows sharply under skew while the *bytes* do not:
if logical remote bytes are within a fraction of a percent across routes and interconnect traffic is
flat or lower on the slow route, the cost is **serialization, not transfer** — fast ranks blocking
on slow ones — and it is a scheduling/granularity problem. If instead the bytes moved, you have a
transfer problem and this lever is the wrong one. That distinction costs one experiment and decides
the whole direction; skipping it is how projects spend a month optimizing the wrong quantity.

The fix is granularity, not removal: publish **per destination item** (per token, per tile) and have
each consumer wait only on the items it is about to read, then start its reduction. Fast items
proceed while slow ones are still arriving.

**Prerequisite most people miss:** a single-buffered staging buffer is *why* the barrier was needed —
the barrier is what stops iteration N+1 from overwriting rows iteration N has not consumed. Removing
the barrier therefore **requires** either double-buffering with a parity index or per-slot completion
counters. Do that first, or the fusion is correct in a single shot and silently wrong in a loop.

Keep the barrier behind a compile-time fallback flag. It gives the A/B an honest control and a
liveness failure an immediate escape hatch.

## Lever 3 — Charge the fusion for the kernel boundary it removes

This is the lever that decides most *negative* results, so read it before proposing a fusion.

A kernel boundary is not free ordering — it is a **hardware-managed, load-balanced join** over many
small workgroups. Replacing it with an in-kernel counter substitutes a **software max-of-N** over the
resident blocks, and the resident blocks are big, few, and unevenly dispatched across dies.

Measured: a quantization pass costing `0.0009 ms` standalone was folded into the persistent kernel as
a static grid-strided partition. The pass itself was free. The join was not:

| route | separate launch | fused ingress | Δ |
|---|---|---|---|
| 512 uniform | 0.6796 | 0.6806 | +1.0 µs (+0.15%) |
| 512 skew | 0.7549 | 0.7657 | +10.8 µs (+1.4%) |
| 8192 uniform | 4.4877 | 4.5042 | +16.5 µs (+0.37%) |
| 8192 skew | 5.3472 | 5.4122 | +65.0 µs (+1.2%) |

The delta is worst on the skewed routes — exactly where the grid is least evenly occupied, i.e. where
max-of-N is worst.

**Rule: fusion pays when it removes a *wait*, not when it merely removes a *launch*.** Launch is
~5–15 µs; if that is all you are removing and the phase must then be joined in-kernel, expect to
lose.

**Corollary — launch count is not an objective, so do not adopt it as one.** The pass above was
pursued to reach "exactly one launch per rank", which sounded like the definition of full fusion. It
is not, for three independent reasons, and the project later withdrew the milestone outright rather
than repair it:

1. **The measurement said no from the start.** The pass was 0.64% of e2e; the join it created cost
   more than the launch it removed. Nothing about that ratio was going to improve.
2. **Check the operator boundary before fusing across it.** That pass was an activation *cast*, which
   in a real serving graph belongs in the epilogue of whatever produced the activation — producer
   quantizes what it just wrote. Pulling it into the consumer megakernel competes with the better
   placement and forces the consumer to ingest unquantized input, defeating any pre-quantized fast
   path the operator already offers. **Ask "who is the natural producer of this data?" before
   fusing a phase in; if the answer is a different operator, the phase is not yours to fuse.**
3. **The reference architecture does not do it either.** "Fuse it like <fast library X>" was read as
   "one launch for everything". Fast GEMM libraries of this class consume pre-quantized inputs with
   their scales and ship the cast as a separate helper; collective dispatch/combine likewise live
   outside. What they maximize is occupancy and pipelining *inside* the main body, not the number of
   launches in the surrounding graph. **If you are citing a reference design as justification, state
   the specific mechanism you are copying. "It's one kernel" is usually not what the reference did.**

If you build such a fusion anyway, keep it in-tree **default-off as a labelled control experiment**,
not as an unfinished feature — the negative result is the deliverable, and leaving it filed as
"almost done" causes the next person to spend effort finishing something that should not ship.

**The corollary runs in BOTH directions — a small inter-kernel gap is NOT grounds to reject a
fusion.** The rule above says launch count is not a reason to fuse. It does not say launch gap is a
reason not to. These are different claims and conflating them has already killed a correctly-scoped
fusion on the wrong evidence: a planning role measured the total inter-kernel gap at 12.4 µs/iter
(0.22% of e2e at the large size, 1.42% at the small one) and closed the direction as
"net-negative before it starts", citing this very file. That reasoning is invalid. The gap is what
fusion removes **incidentally**; the wait is what it removes **on purpose**. On that same operator
the two quantities differed by orders of magnitude: the inter-kernel gap was a fraction of a percent
of e2e, while the exposed cross-rank wait was hundreds of microseconds at rank-max p95. Rejecting a
fusion on the basis of the first number, when the second is the one it targets, is measuring the
wrong side of the change. Measure both, and say which one your proposal is about.

So the decision test is fixed, and it is not a launch-gap measurement:

- **Run the no-payload control** (Priority item 1). Replace the transferred payload with nothing and
  keep everything else identical. The delta is the exposed communication cost — the ceiling on what
  overlap can buy. Report it as a percentage of the stages it covers, not in milliseconds.
- **Then measure the exposed wait directly** — an instrumented peer-wait timer, or the barrier's
  own duration. If a consumer is blocked on a producer for a time that is large relative to the
  target, fusion has something to absorb, *regardless of how small the launch gap is*.
- Only if BOTH come back near zero is "no fusion win available" the right conclusion. **"The kernels
  launch back-to-back" is evidence about the host, not about whether the GPU is idle waiting.**
  Kernel boundaries can be 12 µs apart and still serialize a millisecond of dependent work, which is
  exactly the case a persistent kernel with per-item readiness exists to fix.

Write the decision down with the number that drove it. "Rejected: no-payload control showed 0.3% of
e2e" is a durable result. "Rejected: launch gap is 0.22%" is a category error and will be re-litigated.

## Lever 4 — Straggler-gating has a variance signature; use it to diagnose

Same code, 8192 uniform, 100 iters, 3 interleaved reps, rank-max e2e:

| arm | rep1 | rep2 | rep3 | spread |
|---|---|---|---|---|
| separate launch | 4.4208 | 4.4217 | 4.4200 | **1.7 µs** |
| fused ingress | 4.4566 | 4.4310 | 4.4476 | **25.6 µs** |

Slower **and 15× noisier** ⇒ a gate paying max-of-N, not a throughput deficit. Slower with an equally
tight spread ⇒ genuinely less throughput. These call for opposite fixes, so **always report per-rep
spread alongside the median** — a median-only table cannot distinguish them.

Corollary on the straggler reading: adding participants makes max-of-N *worse*, not better; removing
them eventually runs out of bandwidth. Neither direction is the fix — the fix is Lever 2 (make it not
a join at all) or Lever 3 (do not fuse it).

## Lever 5 — Publication scope must match the producer/consumer domain

Classify the edge before choosing scope:

1. A rank-local edge between CTAs on the same GPU uses **agent scope**. Do not escalate an
   intra-rank producer→consumer readiness counter to a system-scope RMW merely because the GPU has multiple
   XCDs; that adds an unnecessary peer-visible operation and may be illegal for an ordinary
   rank-local allocation.
2. A peer-rank edge whose producer and consumer are different GPUs uses **system scope** and
   peer-visible symmetric storage.
3. Because a wait is a relaxed load, it does **not** invalidate L1. Every wait must be paired with
   an explicit **acquire fence** before the guarded data is consumed. No exceptions — this is the
   single most common source of "works at bs=128, garbage at bs=8192".

Record the coherence-domain count in the profile alongside the cache sizes; the sizes do not predict
either of these.

## Lever 6 — Write-through stores instead of per-block release fences

The obvious way to publish is `fence_system_release()` then the completion atomic. On this part that
fence lowers to a **full LLC writeback (`buffer_wbl2`) per block**. With ~200 participating blocks it
measured **+0.411 ms**.

Cheaper and equivalent for this pattern: mark the payload and scale stores **write-through**, then
`s_waitcnt vmcnt(0)` before the completion atomic. On gfx95x the buffer cache-policy bits are
`bit0=sc0`, `bit1=nt`, `bit4=sc1`; write-through is `sc0|sc1 = 1|16 = 17`. Applied to the ingress
this recovered most of an 8.4 µs residual and removed the fence entirely.

Same trade the GEMM epilogue already makes for its activations — check whether your codebase has an
`out_cache_modifier`-style hook before adding a fence.

## Lever 6b — A wait the compiler is allowed to move is not a wait

Once producer and consumer live in the same kernel, the acquire side stops being a runtime property
and becomes a **compiler** property, and this failure passes every functional test you will think to
run. The reported case: a consumer's input pointer was declared `const __restrict__`. Both
qualifiers are promises — *nothing writes through another pointer, the value does not change* — and
they are false here, because the whole point is that a peer block writes that buffer. The compiler
took them at face value and hoisted the load **above the readiness spin**. The kernel then read
whatever was in the buffer from the previous iteration. On a warm graph replay the previous
iteration's data is usually *nearly right*, so the error is small, shape-correct, and passes a
tolerance check.

Three consequences worth adopting wholesale:

- **Do not put `const`/`__restrict__` (or a language's equivalent aliasing promise) on any pointer
  whose contents are produced inside the same kernel by another block.** The qualifier is a lie the
  moment the buffer becomes cross-block state. This is the single highest-yield thing to grep for
  after a fusion.
- **Verify the ordering in the emitted ISA, not in the source.** Find the readiness spin and confirm
  the dependent loads are *after* it, and that the appropriate `s_waitcnt`/acquire sits between. A
  source-level wait that the scheduler sank past is invisible at every level above the disassembly.
- **Test it with poisoned inputs, not with real ones.** Fill the producer's output buffer with a
  value that cannot be correct (NaN, a sentinel, the wrong iteration's data) *before* launch, and
  run the same graph repeatedly. A hoisted load reads the poison on replay 1 and then hides forever
  behind plausible stale data. Real inputs are exactly the inputs that cannot detect this.

Generalize the rule: **anything that tells the compiler a value is invariant defeats a
synchronization built on that value changing.** Ordering added at the source level must be re-proved
at the machine level after every fusion, because fusion is precisely the transformation that puts
the two sides within the optimizer's reach for the first time.

## Lever 7 — Residency is a correctness invariant, not a tuning knob

In a persistent kernel, roles are assigned by **arrival ticket** (`ticket = atomic_add(counter)`;
ticket 0 = plan owner, `1..K` = producers, and so on). If any block gates on a participant index that
is **not resident**, the grid deadlocks: the unscheduled block cannot run until a resident block
retires, and none can, because they are all waiting on it.

Measured: the largest stage ran at **1 WG/CU**, so exactly tickets `0..num_cu-1` were resident.
Excluding the owner from the participant set without shrinking the participant count pushed the last
share onto ticket `num_cu` → hard hang, all 8 GPUs at 100%.

**Invariant: every participant index must land inside `[0, resident_blocks)`.** And note the trap:
the resident count depends on the occupancy-binding resource and on per-bucket grid sizing, so a
participant count that is legal at one problem size can hang at another. Derive it, never hardcode
it, and make any override env-gated for bisection only.

## Lever 8 — Acyclicity: the coordinator must not owe the workers anything

If the block holding a coordinating role (emitting a plan, seeding counters) is also assigned a share
of work that the workers gate on, and that share is scheduled *after* the coordination point in
program order, you have closed a cycle: coordinator → workers → coordinator. Hard hang, at every
participant count you try.

**Rule: any work the workers gate on must complete before the coordinator reaches its coordination
point.** Placement, not logic — moving the phase earlier in the same block fixed it.

Bisection tell: if forcing the participant target to a value the *workers themselves* satisfy always
passes while every real value hangs, it is this, not a scope bug.

## Lever 9 — Reset-free counters

Never clear an arrival counter between generations; the clear races the next generation's publishers.
Instead let the counter rise monotonically and raise the *target* by the per-generation increment.
Pad any slot outside the live range so the target is still reachable.

Where a clear is genuinely unavoidable, do it in the **ticket-0 block before it publishes the epoch
gate** — that orders the clear ahead of every other block without needing a grid barrier.

Where a generation key is needed, prefer an explicit monotone epoch with a parity-selected buffer
half (ABA-safe). **Trap:** do not key a generation on a counter that also counts launches which
publish nothing — e.g. a variant compiled without the publishing role will bump the epoch and desync
every waiter.

## Anti-patterns (measured, do not re-walk)

**A — A work-stealing claim queue for an in-kernel phase.** Measured **~89 ns of end-to-end cost per
claimed chunk** (+0.020 ms at 512 tokens, +0.30 ms at 8192, scaling exactly with chunk count), because
every block's *next* claim atomic sits on its own critical path. A static grid-strided split with no
claim atomic at all beat it outright. Work-stealing is for a queue whose items have wildly unequal
cost; a uniform sweep does not qualify.

**B — Grid-stride unrolling to restore outstanding-load count.** A phase folded into a kernel pinned
at 1 WG/CU runs far fewer waves than the standalone launcher did, so more independent groups per
iteration "should" help. Measured at 8192 tokens, 100 iters, 3 reps: `4.4535 ms` at U=4 vs
`4.4476 ms` at U=1 — no. **And that is information:** if unrolling does not help, the residual is not
latency-bound, so stop looking there and go read Lever 4's spread instead.

**C — Cutting LDS to raise occupancy without checking which resource actually binds.** The stage was
at 97.5% of LDS/CU, which reads as an obvious 1→2 WG/CU opportunity. It was not: at **256
VGPR/thread** a 512-thread block is pinned to 1 WG/CU *regardless of LDS*. The only configuration
that cleared the LDS budget cost **+2.33 ms** e2e. **Report which resource binds, not just the LDS
figure** — and when occupancy is VGPR-bound, an LDS direction is a dead end before it is dispatched.

**D — A per-rank load test to detect intra-rank skew.** Skew that matters is often an *intra*-rank
property (a few hot experts inside one rank). A detector comparing rows received against
`tokens × topk` never fires, because the rank-level totals are balanced. Use a statistic at the
granularity the imbalance lives at — e.g. `max_expert_tiles` versus `rows / tile_m / experts_per_rank`.

**E — Pursuing "one launch" as the definition of full fusion.** See Lever 3's corollary. The
milestone was built, was correct, reached exactly one launch, measured slower on all four guards, and
was withdrawn as a goal rather than repaired. Cost: the fusion work itself, plus a planned follow-up
to convert its ingress join into per-item readiness that was only ever needed to rescue the wrong
target. **Before adopting a structural target, ask what it is worth in the profile.** If the phase in
question is under ~1% of e2e, the target is bookkeeping, not performance.

## Measurement discipline (this domain breaks the default assumptions)

- **Interleave the arms in ONE script.** Batch-to-batch drift measured up to **4%** at small sizes —
  larger than most of the effects here. A 1% apparent regression from sequential runs became 3-of-4
  paired **wins** when interleaved. Any margin under ~2% measured sequentially is noise.
- **Report per-rep spread with every median** (Lever 4).
- **Rank-max, not rank-mean,** is the number that matters: the collective is gated by the slowest rank.
- **Guard all four corners.** Small/large × uniform/skew. A fusion that wins at large uniform can lose
  at small skew, and small-batch regimes have proportionally more fixed cost to amortize.
- **Liveness is a separate gate from correctness.** 1000 CUDA-Graph replays per route, plus an
  explicit stale-read check that the readiness counters are epoch/parity-correct across back-to-back
  iterations. A fusion can be numerically correct on a single shot and deadlock or read stale data at
  a different block count (Levers 7–9 are all failures of exactly this kind).
- **A latency win with no measured overlap change is suspicious, not accepted.** Re-collect the
  kernel-level DAG and the instrumented peer-wait and show the wait actually shrank. Otherwise you
  have measured drift.
- **Name the denominator.** Compare against the frozen upstream baseline under an identical
  route/iteration command, never against a different library.

- **Two denominators, two different questions — run both.** "How much faster than upstream" and "how
  much did the fusion buy" are not the same number, and on a branch carrying other work the first
  can be several times the second. That has happened: on one guard the improvement over the frozen
  baseline was mostly *unrelated* changes already sitting on the branch, and only a small part of it
  was the fusion. Reporting the first number as the fusion's result is a real, easy, and common
  attribution error. Attribute a fusion win by running the
  **same tree with only the fusion flag toggled**, A,B,A,B. Comparing the fused build against a frozen baseline credits fusion with every
  other change on the branch — including the ones you would keep if the fusion were reverted.

- **Fusing a phase deletes the instrument that measured it.** A peer-wait timer living in the
  standalone kernel reads a flat `0.0` once that kernel is no longer called — which looks exactly
  like "the wait went to zero" and is really "nothing was measured". Before quoting an improvement in
  a wait/occupancy/bandwidth counter after a fusion, **prove the instrumented code path still
  executes.** Budget an in-kernel replacement instrument as part of the fusion, not as follow-up.

- **State the aggregation unit of an in-kernel timer before reading its output.** Two bugs, both
  producing confident wrong numbers, neither a timing bug:
  - *Mean vs total.* A per-item readiness scheme waits many short times where a barrier waited once
    for long. Reporting the per-wait **mean** made the fusion look worse for free. The comparable
    statistic to a barrier timer is the **per-launch total**. Publish both.
  - *Per-block vs per-wave slots.* One accumulator per block sums waves that wait **concurrently**,
    turning parallel time into serial time. It reported 8470 µs of wait inside a 5.4 ms kernel.
    **Any in-kernel wait exceeding the kernel's own wall time is an aggregation error** — that
    impossibility is the cheapest available self-check, so compute the ratio every time.

- **Once anything overlaps, the SUM of per-stage timers stops being a cost and becomes an
  artifact — and it moves the WRONG WAY.** This is the single easiest way to read a successful
  overlap as a regression, and it needs no bug to happen. A consumer stage that is allowed to enter
  early charges its own *waiting* to itself, and stages running concurrently contend for the same
  units, so every individual stage gets slower on paper while the operator gets faster. One
  published measurement of exactly this: summed kernel time rose 62.4 → 102.9 µs across the same
  transformation in which the end-to-end execution span fell 61.6 → 37.0 µs. Both numbers are real;
  only one of them is the operator's latency.
  **The comparable quantity across a fusion is the span** — first stage start to last stage end,
  or equivalently the graph's critical path — never `Σ stage_i`. If your harness reports per-stage
  timers (most do, and they are the natural thing to watch), say explicitly in the report that they
  are diagnostic under overlap and non-comparable to their pre-fusion values. Keep them: a stage
  timer that inflates while the span shrinks is *positive evidence that overlap happened*, and it is
  often the cheapest such evidence you have. Just never add them up.

- **An opt-in fast path that falls back silently will be measured in its slow form.** A megakernel
  gated on `ENV == "1"` **and** a config predicate silently took the scattered path when either
  failed, and a full closeout was filed against the wrong path — it does not look like a bug, it
  looks like "the optimization didn't help". Make the kernel **announce which path it took**, once
  per process, and make the harness **refuse to report a number without that marker from the same
  run**. Do not use an unrelated field (a variant name, a config string) as a proxy for "fusion is
  on"; it will read plausible on both paths.

- **The static-ISA screen is a screen for RESOURCE hypotheses only. It cannot refute a latency-hiding
  one.** When the GPU group is a serialized lease, dumping the final ISA and diffing
  `amdhsa_private_segment_fixed_size` / `amdhsa_next_free_vgpr` / `amdhsa_group_segment_fixed_size`
  plus instruction-class counts is an excellent way to kill directions in minutes without queueing.
  It is sound for claims of the form *"this will free registers / shrink LDS / raise occupancy"* —
  those claims are **defined** by those numbers, so unchanged numbers do disprove them.
  It is **not** sound for claims of the form *"this will shorten the dependency chain / overlap a
  load with an MFMA / reduce time spent stalled"*. Those are properties of the dynamic schedule.
  Measured counterexample on this operator: an index-scalarization + prefetch-lead change left
  scratch, VGPR, LDS, `v_mfma` and `buffer_load` counts **all bit-identical**, was refuted on that
  basis, and then — when it was accidentally measured anyway — came back favourable on **4 of 4
  paired reps at the large size (−0.39% to −1.61%)**, three to four times that guard's calibrated
  noise floor. The static screen was not wrong about the ISA; it was answering a different question
  than the one asked.
  So: use the screen to **order** the queue and to catch compile-illegality for free, never as the
  sole grounds to close a scheduling-class direction. Recording "ISA unchanged" is useful; recording
  "ISA unchanged, therefore no effect" is a claim the instrument cannot support. If a scheduling
  hypothesis survives compile, it needs wall clock.

## Lever 10 — Some fused kernels cannot be compile-screened AT ALL. Find out on day one.

The normal cheap loop is "compile it, screen the ISA, only then spend a run". A persistent
multi-rank kernel often **cannot enter that loop**, because its body calls the communication runtime
(`shmem`-style symmetric-heap accessors, peer pointer resolution) **at trace/codegen time**, not just
at run time. Forcing codegen without an initialized heap aborts inside the runtime's own status check.
There is no partial screen: you get an abort, not a kernel.

Such a kernel is **lease-only**. Every hypothesis about it — including ones that are obviously true —
costs a full multi-rank run. That single fact dominates the schedule: it is the difference between
"try five things this afternoon" and "try five things this week", and a plan written as though static
screening were available will silently blow its budget.

Act on it explicitly:

- **Test it once, at the start**, and record the verdict where the next round can see it. Attempting
  a static screen every round is a recurring tax on a question already answered.
- **Say so in the direction's plan.** A lease-only kernel changes what a round should contain: one
  instrumented run that answers several questions at once beats three runs that each answer one.
- **Build the instrument to record-and-continue**, never halt-on-first-failure. If the kernel is
  lease-only, the lease is the scarce resource, and a bisection that stops at the first fault has
  spent the whole lease to learn one bit.
- **Compile screens are still valuable for every other kernel**, and they remain a way to *reject*,
  never a substitute for a run: two directions on this workflow were compile-clean on every
  {shape}×{phase} combination and dead on hardware.

The inverse is also worth knowing: if the fused kernel *does* compile standalone, you have a cheap
screen the scattered version never needed, and register/LDS regressions from fusion show up there for
free.

## Priority

0. **Establish whether the kernel is compile-screenable** (Lever 10). It sets the cost of every
   iteration that follows, so it is the first thing to know and the cheapest to find out.
1. **Run the no-payload control** and the DAG. No exposed wait closes only the
   communication-hiding claim; continue evaluating CTA-local tail filling and
   launch-structure/resource-partition value.
2. **Enumerate the missing readiness edges** (Lever 1). This is static and cheap.
3. **Test the CTA-local phase pipeline** (Lever 1b) whenever downstream items
   become ready before the whole upstream grid drains. A shared MFMA engine
   rejects a permanent CU split, not this schedule.
4. **Attack the largest exposed wait first** with per-item readiness (Lever 2) — usually the cross-rank
   barrier, and usually most of the available gain — **but only after confirming the readiness
   publication is cheaper than that wait** (Lever 2's multi-die caveat + Lever 6). On an 8-XCD part a
   system-scope per-item atomic can cost more than the wait it removes; when it does, this lever is a
   regression and the gain is in the resource partition (item 3b), not here. Defer to the operator's
   expert skill for where this ranked when it was actually measured.
   4b. **When the exposed cross-rank wait is already short**, skip straight to the launch-structure
   wiring and CU-role partition (`resource_partition.md`) — co-residency of the stages with the right
   CU ownership, and folding combine in as a third concurrent queue, is what pays there. This is the
   headline mechanism on a well-tuned distributed operator, and it does **not** depend on a per-item
   cross-rank readiness edge.
5. **Then the intra-rank producer→consumer edge**, agent scope, which is cheaper to publish.
6. **Do not fuse launch-only phases at all** (Lever 3) unless something other than launch count pays
   for it. If you build one to settle the question, land it default-off and labelled as a control.
7. Levers 5–9 are **prerequisites, not options** — get scope, fence cost, residency, acyclicity and
   counter-reset right, or the above will hang rather than run slowly.

Honest ceiling: perfectly hiding the *entire* worst-case combine wait was 13.46% of e2e, and the
no-payload delta is not removable (a correct implementation still sends the data). Real gains are a
fraction of these. A cumulative **2–5%** rank-max improvement from a full fusion of a
well-tuned distributed operator is a good outcome — size the effort accordingly.

**What the outturn looks like, in shape rather than in numbers.** Across guards, a distributed
fusion's gain **tracks how much exposed cross-rank wait there was to absorb, and nothing else.**
Expect, and check for, all four of these:

- Guards with the largest exposed wait gain the most. If your best guard is not your most
  wait-exposed guard, something other than the fusion is moving the number — go find it.
- Small batch sizes sit near the noise floor: the fixed costs a fusion removes are the same order as
  run-to-run drift there, and individual pairs can come back **negative** without the fusion being
  wrong. Report the per-pair spread, not only the median, or you will over-claim.
- **A skewed route can gain less than a uniform one even with a perfectly tuned fusion**, because
  under skew the bottleneck moves to a *producer-side imbalance* — a long tail on the ranks holding
  the hot work. Fusion overlaps a consumer with a producer; it cannot make an overloaded producer
  finish sooner. That regime needs the producer's tile→block mapping changed so a destination's
  contributions arrive progressively, which is a different change with a different risk profile.
  Diagnose which regime you are in (Lever 4's spread, plus the per-rank wait distribution) **before**
  spending more effort on the fusion itself.
- The total stays inside the 2–5% band above. A result far outside it, in either direction, is more
  likely a measurement or attribution fault than a discovery — check the path marker, check the
  denominator, check that both arms ran in the same rep.

**Overlap claims have a proof ceiling you should know before promising one.** Once the operator is a
single kernel there are no boundaries left for a kernel-granularity trace to show overlap across, and
a wait timer shows that a wave *waited*, not that the CU executed something else meanwhile. Structural
evidence (kernel count in the trace, per-item readiness replacing a barrier) plus a bounded in-kernel
wait (measured: 8.6%–23% of kernel time) supports overlap but does not demonstrate it. A direct claim
needs a **residency-side** instrument — executing-wave sampling across the timed window, or an ATT
pass correlating wait regions against issued VALU. Plan for that instrument up front, or state the
overlap result as partial and say what is missing.

## Four traps this operator's producer→consumer edge hits (established on-box, gfx950/MI355X)

These cost a whole round to rediscover; treat them as preconditions, not surprises.

1. **A `@flyc.kernel` that calls mori shmem runtime primitives (`wait_until_equals`,
   `fence_system_acquire`, the combine's `CrossDeviceBarrier`) binds/executes those primitives at
   **trace/codegen time**, so it cannot be compiled standalone without live peers — it can only be
   exercised under a full multi-rank lease.** Do not plan a cheap compile-screen for the combine or
   any consumer that waits on a peer; there is no compile-only gate for it, only a lease. (This does
   NOT mean you cannot *distinct-hash* it — a JIT-key anchor still produces distinct binaries; it
   means the trace itself needs the peers present.)

2. **Threading a producer→consumer arg requires editing the combine HOST OP wrapper
   (`flydsl_dispatch_combine_intranode_op.py`, `_run_combine_kernel`), not only the kernel body.**
   The kernel-body file and the op-wrapper file are two separate files; the arg list is assembled in
   the op wrapper. If the op wrapper is outside your modifiable-files scope, the producer→consumer
   edge is structurally unbuildable — surface that as a scope blocker on round 1, do not bank a
   kernel-only half that can never be wired.

3. **A host-toggled double-buffer parity index is FROZEN at CUDA-graph capture time**, so host-side
   buffer swapping between launches is a NO-OP under the 1000-replay captured-graph harness (stale
   read / deadlock risk). Double-buffering a peer-written shmem buffer (`shmem_comb_inp`) needs a
   **device-side epoch/parity counter** that advances inside the kernel, not host logic. Budget this
   as a lease-free prerequisite before the gain arm, not as something to discover mid-lease.

4. **The producer publish spans multiple blocks.** The hidden vector spans `num_n_blocks` blocks and a
   last-block fence orders only its own slice, so a per-token consumer wait must expect
   `expected = topk * num_n_blocks` arrivals (plus the device epoch from #3), not `topk`. A wait sized
   for a single block deadlocks or reads stale under skew.

Corollary (timing/meter): `s_memrealtime` is **not** exposed through the flydsl rocdl binding — emit
the raw `llvm.amdgcn.s.memrealtime` intrinsic directly (`read_memrealtime()` helper pattern). And an
in-kernel meter accumulator is capture-safe but its **host readback (device sync + `.item()` + print)
is illegal during CUDA-graph capture** — guard it with `is_current_stream_capturing()`; the meter's
scattered(~0)/forced-concurrency(high) controls are therefore only observable in EAGER mode, so run
the meter validation eager, not inside the captured bench.
