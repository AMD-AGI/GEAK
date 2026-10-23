# When NOT to Fuse — The Precondition Test for Fine-Grained Dependencies

A **method** card, and deliberately an adversarial one. Every other document in this tree tells you
how to fuse. This one exists so that "fuse it" has to survive a test before it becomes a plan.

The bias it corrects is structural, not personal: a workflow whose task statement says *"collapse N
launches into one"* will produce an analysis that concludes *"collapse N launches into one"*,
because that conclusion was in the prompt. An analysis that cannot reach the opposite conclusion on
evidence is not an analysis. **You are permitted — and expected — to return "the fine-grained form
does not pay here, and here is the measurement that says so."** That is a successful Analyze phase.

## The three conditions

Fusing to a persistent/megakernel form buys you exactly one thing the multi-launch form cannot have:
a consumer may start on *part* of a producer's output before the producer has finished. Everything
else fusion gives you (fewer launches, warm caches, no re-load of weights) is available more cheaply
by other means. So the fine-grained dependency pays **only if all three hold**:

1. **The producer's output can be handed over at a smaller granularity than "all of it."** There
   must exist a partial result that is independently consumable. If the consumer needs a reduction
   over the producer's entire output, there is no partial handover and no window.
2. **There is idle hardware for the consumer to run on while the producer is still running.** If the
   producer already saturates the machine, starting the consumer early does not make it finish
   earlier; it just interleaves. Check occupancy and CU utilization during the producer, not after.
   **But "the machine" is not one scalar resource.** Model the producer and consumer as demand
   vectors over MFMA/VALU issue, VMEM/LDS issue, HBM and the interconnect, then ask whether the
   *consumer's required resources* have headroom while the producer runs. When both stages demand
   the same saturated engine (two throughput-limited GEMMs both wanting MFMA), a static `f/(1-f)`
   split runs in `max(A/f, B/(1-f))`, minimized at `A+B`: exactly the serial time, before overhead.
   When their binding resources differ, the lower bound may instead be a per-engine maximum and a
   real overlap prize can exist (see `resource_partition.md`).

   Do not turn that distinction into the opposite shortcut. A GPU P2P operation issued by shader
   `buffer_store` is not a free DMA copy: it still consumes resident waves, issue slots, VMEM
   resources, registers and often a compute tail before bytes reach the interconnect. Likewise a
   "communication" reduce normally contains CU-side loads and arithmetic. Split each stage by
   engine from counters or controlled arms; do not label the whole stage `comm` from its name and
   assume `floor = max(compute_us, comm_us)`. The useful test is measured headroom on every engine
   the consumer actually needs, plus a controlled in-kernel overlap reading.
3. **The producer does not complete in a single wave.** This is the condition that is skipped most
   often and kills the most proposals. If the producer's grid fits in one wave, its first output
   tile becomes ready at almost exactly the moment the whole producer finishes — the early-start
   window is ~0. There is nothing to overlap. Compute it: `ceil(blocks / (CUs × blocks_per_CU))`. If
   that is 1, stop; write it in the report and stop.

Condition 3 has a corollary worth stating separately: **a producer with many waves has a window
proportional to (waves − 1)/waves of its own duration.** That is your upper bound on what
fine-grained handover can recover from that edge, before any overhead. Compare it to the sync cost
per edge (`tile_task_graph.md`) *before* proposing the change. If the window is 3 µs and you need
200 edges each costing ≳1 µs of global sync, the arithmetic has already answered you.

Evaluate all three **per edge**, not for the operator as a whole. It is normal for one edge in a
chain to satisfy all three and the rest to fail condition 3.

### Condition 1 is cheaper to check than the others, and it fails more often than you expect

Do it first, statically, from the source — it costs minutes and needs no GPU. **Look at the first
operation the consumer performs on the producer's output.** If it is a reduction over an axis the
producer partitions across, there is no partial handover, and no barrier design, event scheme, or
scheduler can create one. The recurring shapes: a normalization or softmax whose statistic spans the
full hidden vector; a dot product over the full contraction dimension; any accumulation over
`reduction_col` before the value is final. In each case the *last* contributing tile is what the
consumer needs, so "partial" readiness is worth nothing.

How often this bites, from a published profile of an open-source megakernel implementation of a
1B-parameter decode layer: of six consecutive operator pairs, **exactly one** admitted a partial
start (QKV → partial-attention, where attention for one KV head needs only that head's Q/K/V
blocks), and the measured overlap there covered **28% of the producer's duration**. Every other pair
was a hard full-grid wait for a structural reason of the kind above, and the implementation's own
barriers correctly said so. Do not assume your operator is denser in exploitable edges than that
one; find out, per edge, before the fusion is designed around an overlap that cannot exist.

### Attribute the win, or you will not know whether you needed the fusion

A fused form changes at least three things at once: launch count, kernel-boundary synchronization,
and cross-operator overlap. Reporting one end-to-end number does not tell you which of them paid,
and the standing critique of this whole technique — a fair one — is that published results routinely
omit that breakdown. This matters practically and not just rhetorically: if the win is mostly
launch-stall removal, the same win is available from graph capture and a dependent-launch mechanism
at a fraction of the complexity and none of the maintenance cost, and the megakernel was the wrong
build.

The ladder in the next section doubles as the instrument. Run the rungs as an ablation and report
the increment each one bought:

```
baseline (kernel-by-kernel)             T0
+ adjacent-kernel fusion                T1     -> boundary + intermediate-traffic removal
+ graph capture                         T2     -> host launch cost
+ concurrent independent branches       T3     -> DAG structure  (usually the largest single step)
+ dependent-launch / partial handover   T4     -> fine-grained overlap
fused persistent kernel                 T5     -> whatever is left, minus the framework cost
```

`T4 → T5` is what the megakernel is actually worth. Report it separately, and report **what fraction
of each producer was actually overlapped** (producer duration covered by a consumer that started
early ÷ producer duration) rather than the word "overlap". That fraction is the honest form of an
overlap claim and the only one that distinguishes real pipelining from serialization rearranged.

## The cheaper-lever ladder

Fusion is the most expensive lever in the drawer — in engineering time, in verification risk, in
deadlock surface, in how hard the result is to change later. Work up the ladder and stop at the
first rung that closes the gap you measured:

1. **Remove the work.** Algorithmic or layout change, redundant compute, an unnecessary
   materialization, a precision choice. Consistently the largest wins, and consistently the last
   thing anyone tries.
2. **Fuse *adjacent* kernels** that share operands — trivially safe, no dependency machinery, gets
   the cache reuse without the scheduler.
3. **Run independent branches concurrently** (multiple streams/queues). This costs nothing in
   correctness and directly attacks the case where the DAG has parallelism the launch order is
   throwing away. It is the right answer whenever `tile_task_graph.md` found unconnected subgraphs.
4. **Cut the launch overhead itself** — graph capture / replay. Launch and teardown is on the order
   of ~2 µs uncaptured and ~1.3 µs captured; the residual bubble between adjacent nodes in a
   captured graph has been measured at **~300 ns**. Note what that means: *if your proposal's whole
   thesis is "we remove N launch boundaries", the prize is N × 300 ns.* For a handful of launches
   that is noise. Say so rather than shipping it.
5. **Overlap dependent consumers with their producers** without going persistent, where the platform
   offers it. On NVIDIA this is programmatic dependent launch (PDL), worth ~0.5–2 µs per dependency
   edge. **ROCm has no direct PDL equivalent** — so on AMD hardware this rung is thinner, and its
   substitutes (cooperative launch, in-kernel producer/consumer flags between co-resident
   workgroups, multi-queue) each carry more of fusion's complexity. That gap is a legitimate
   *argument for* fusion on this hardware, and it is one you have to make explicitly rather than
   assume.
6. **Persistent / megakernel form.** Only after 1–5, and only for the edges that passed the
   three-condition test.

## The calibration that should make you cautious

Published four-way ablation on a dependency chain, same work throughout (graph time / kernel time,
µs):

| form | time |
|---|---|
| linear chain, no dependent-launch | 65.5 / 61.6 |
| linear chain + dependent-launch | 59.4 / 54.2 |
| restructured branching DAG, no dependent-launch | 49.2 / 44.4 |
| restructured branching DAG + dependent-launch | **41.0 / 37.0** |

Read the second column against the third: **restructuring the DAG so independent work is expressed
as independent (rung 3) beat the fine-grained-dependency mechanism (rung 5) on the original
structure.** The mechanism and the structure compose, but the structure was worth more.

And a replication that should be quoted at anyone proposing a megakernel on launch-overhead grounds
alone: a small transformer decode step, hand-written as one megakernel, versus the same step split
into 81 ordinary kernels under graph capture — 983 µs fused, 1169 µs captured-with-dependencies,
**1010 µs captured + dependent launch**. The cheap mechanism recovered ~83% of what the kernel
boundaries cost; the persistent form led by **~2.7%**.

Three further results from the same literature, all pointing the same way — the sophisticated
mechanism is not free and is not universally better:

- **On-GPU dynamic scheduling loses to static scheduling on regular dense work**, measured at
  **0.82–0.89×** — i.e. materially *slower than the baseline* — while the same scheduler wins on
  data-dependent routed work. Dynamic scheduling is a treatment for load imbalance, and applied to a
  workload that does not have imbalance it is pure overhead plus queue contention. Pick the
  scheduler from the workload's variance, not from its sophistication (`resource_partition.md`).
- **Compiler-generated tiles inside a fused kernel routinely lose to a tuned library kernel** for
  the same GEMM. A published system attributes several of its own losing configurations to exactly
  this. Fusing a GEMM means giving up whatever the vendor library was doing for it, and that
  regression is charged against the fusion's gain, not waived.
- The technique's benefit **decays as the regime becomes compute-bound**. It is strongest where
  per-operator work is small relative to fixed cost — low batch, decode, latency-bound — and shrinks
  as batch grows and the bubbles it removes become a smaller fraction of a longer kernel. If your
  guard set spans both regimes, expect the answer to differ between them, and do not generalize a
  small-shape win.

The honest summary of the public evidence: **fusion of adjacent kernels + concurrent independent
branches + a dependent-launch mechanism typically reaches 90–95% of an ideal megakernel.** A
megakernel proposal is therefore a proposal to spend most of the project's risk budget on the last
5–10% — which is sometimes exactly right (when rung 5 is unavailable, when the three conditions hold
strongly on a long chain, when the remaining 5% is the difference that matters), and which you must
argue *as such*, with the ladder above enumerated and each rung's expected recovery stated.

## Concurrency is not free of decisions

The moment you run branches concurrently (rung 3) or split roles inside a persistent kernel, you
have created a **resource allocation problem** that did not exist while everything was serialized:

- Which branch should get priority?
- How many CUs should each branch occupy?
- After you give a branch more resource and it gets faster — **has the critical path moved?**

Concurrency without answering these often produces *no speedup at all*, because the branch you
accelerated had slack and the one you starved did not. Serialized code has one bottleneck and it is
easy to find; concurrent code has a bottleneck that moves. See `resource_partition.md`.

## Three ways a fusion result gets accepted for the wrong reason

Each of these actually happened in this campaign. Check all three before you propose promoting a
fused candidate.

**1. A whole-grid-join arm is a control, not a candidate.** If the two roles are separated by one
grid-wide barrier — every block finishes role A, all blocks join, then role B — the execution order
is identical to two launches. The dependency edge has been moved from the launch boundary into the
kernel, not relaxed. Such an arm has, by construction, *zero* overlap; what it measures is the
**cost** of unifying the two roles (one register budget for both, sized by the larger role; one LDS
allocation, sized by the larger role; scratch/spill growth; the join itself). It is worth building
— it is the subtraction baseline that makes a fine-grained arm's number interpretable — but it can
never satisfy an overlap acceptance criterion, and a wave that only ships this arm has not tested
the hypothesis. Say so in the report in those words rather than reporting it as "overlap not yet
measured".

Corollary, which explains a shape you will otherwise misread: a whole-join arm tends to *win* on
routes with load imbalance and *lose* on balanced routes. Imbalance leaves idle time that absorbs
the unification cost; a balanced route has none, so the cost is fully exposed. That asymmetry is
evidence the arm is the control arm. It is not evidence that fusion is route-dependent.

**2. Attribute the win before you claim its mechanism.** Compute, at the same guard and on one
comparable timeline/rank when the harness can provide it:

```
fused_kernel_us   vs   sum of the kernels it replaced
residual_us = e2e - sum(all kernel_us)   for BOTH arms
```

Do not synthesize this comparison by summing timers captured in separate graphs or independently
reduced across ranks: `max(sum)` and `sum(max)` are different quantities, and their residual can be
negative. If a valid fused-kernel measure is slower while operator e2e is faster, the fused body
does not explain the win; locate the mechanism before calling it overlap. A grid-wide join can be an unintended
cross-rank synchroniser: it aligns every rank's role-B start, which tightens the arrival window of
whatever comes next and can shrink a downstream peer-wait by more than the fusion itself costs.
That is a real operator effect and possibly worth shipping, but it is not evidence that the target
edge overlapped. Under an operator-e2e contract keep the result and withhold the mechanism claim;
under a changed-kernel contract reject it from that score. Mixing those contracts mid-run poisons
the next wave's priors.

**3. A route gate resolved once and cached is not shippable.** Under graph capture you cannot branch
inside the captured region and you cannot device-to-host sync inside it, so the natural
implementation resolves the statistic on the first eager forward and caches it forever. In serving
that means whichever batch happens to arrive first permanently selects the path for every batch
after it — including the batches the gate would have routed the other way. Such a gate is a
temporary stop-loss for a regression you have not fixed yet, not a feature, and a gated result does
not satisfy an "improves on all guards" criterion. The only shippable form is one captured graph
per path plus a host-side per-batch statistic, at 2x graph memory — which is only worth building if
the regression it hides is genuinely unfixable.

## What to write in the report

Whatever your conclusion, the Analyze phase should contain, per candidate edge:

```
edge: <producer tile> -> <consumer tile>
  (1) partial handover possible?   yes/no + why
  (2) idle hardware during producer? measured occupancy/CU util
  (3) producer waves = ceil(blocks / (CUs * per_CU)) = N   window <= (N-1)/N * producer_us
  window_us: X    per-edge sync cost: Y (scope from tile_task_graph.md)
  verdict: PAYS / DOES NOT PAY / UNKNOWN(what measurement would settle it)
```

Plus the ladder, with the rung you are proposing and one line on why each cheaper rung is
insufficient. A proposal that skips the ladder is not ranked.

## A fusion built in stages: what each stage is judged on

Once a fusion passes the three conditions and gets built, it is normally built in more than one
step: the producer publishes readiness, then the consumer waits on it, then the two are folded into
one launch. **Only the last of those steps can be faster.** The earlier ones add completion
signalling and a second buffer and have nothing yet to hand their work to, so the only effect they
can measure is their own overhead. They are supposed to be slower.

If they are judged on speed anyway, they are rejected, they do not enter the next round's tree, and
the consumer half is then written against a tree where the producer half does not exist. It is
therefore never written. The result is a permanently half-built fusion whose report reads as a
failed search. That has happened in this project: the producer side of a combine fold was authored
and compile-checked, no GPU was allocated to it, it could not have shown a gain in isolation even
if one had been, it was not carried forward, and the consumer side was never begun.

So each step declares what it is, and is judged accordingly.

| | enabling step (a prerequisite) | terminal step (closes the chain) |
|---|---|---|
| What it is | producer-side readiness, buffering, signalling — anything with no consumer yet | the step that makes the fused path actually run end to end |
| Judged on | function | speed |
| Acceptance | compiles; the path marker appears; results correct; 1000 replays without deadlock | the full protocol below |
| Its timing | recorded as a cost against a declared budget, never used to reject it | the result |
| Kept for the next round | yes, committed to the canonical tree | yes, if it wins |

**The terminal step's protocol, in full.** Base / candidate / **blank control** interleaved, all four
route guards, rank-max, paired within a repetition. Plus the measured overlap fraction on the exact
edge the fusion claims to have created — a fused kernel that is faster with zero overlap on its own
edge got its win somewhere else (see "Three ways a fusion result gets accepted for the wrong
reason"). Then an independent re-check: correctness, 1000 replays of the real call sequence without
deadlock, performance reproduced, path marker reproduced.

**Compare against the pre-chain baseline, not the current tree.** Every landed prerequisite made the
canonical tree slower. Measuring the terminal step against that tree credits the fusion with
removing overhead the chain itself installed. The number to beat is the one from before the first
prerequisite landed.

**A prerequisite is a debt, and it comes due.** It is allowed to be slower; it is not allowed to cost
more than the fusion can plausibly return, and it is not allowed to sit unpaid indefinitely. A chain
abandoned half-built is worse than a chain never started: the tree ends up slower with nothing to
show for it. Either close the chain or revert the prerequisites and say why.

## Provenance

Distilled from public analyses arguing *against* the megakernel-by-default position, including
measured launch/bubble costs under graph capture, the dependent-launch-vs-persistence ablations
above, an independent replication of a published megakernel result, an independent profile of an
open-source megakernel decode implementation that found only one of six operator pairs admitted a
partial start, and published systems' own reported losing configurations (dynamic scheduling on
dense work, compiler-emitted tiles against a tuned library). Retained here specifically
because the rest of this knowledge tree argues the other way and an Analyze phase needs both. All
numbers are other people's, on other hardware, quoted as order-of-magnitude calibration.
**Measure yours.**
