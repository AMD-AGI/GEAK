# Pipe Occupancy — The Half of the Analysis the Dependency Graph Cannot Give You

A **method** card, and the second half of `tile_task_graph.md`. It contains no operator's answer.

The dependency graph answers one question: **may** these two pieces of work overlap. That question
is necessary and it is not sufficient, and a program that only asks it will spend every wave
proposing overlaps that the hardware was already achieving, or overlaps that fill a pipe which was
never the constraint. This card is the other question: **which functional unit is idle, at what
point in time, and what dependency-free work exists that could be issued into it.**

Keep the two separate in your head and in your artifacts. Dependency is a property of the *problem*.
Occupancy is a property of the *schedule*. Fusing changes the second; it never changes the first.

## The failure this exists to prevent

An Analyze phase emits a correct tile-level DAG, finds a set of `launch_boundary` edges whose true
scope is narrower than grid-wide, and proposes fusing them. The plan ranks the direction by
"expected speedup", which is a number somebody guessed. Nowhere in that chain did anyone ask what
the machine was doing during the interval the fusion is supposed to improve. If the answer is "the
arithmetic pipe was already saturated", the fusion buys nothing and the wave is spent proving it. If
the answer is "no pipe was near saturation and memory bandwidth was mostly idle", the fusion is not
about launch overhead at all — it is about making a *prefetch expressible* — and the direction should
have been written that way, because the two implementations do not resemble each other.

Both of those are real outcomes from one program. The second one cost four waves of negative results
before anyone measured a pipe. Which one *your* operator is in is a measurement, not a guess, and it
is cheap relative to a wave; the rest of this card is how to take it.

## The measurement

Do not estimate this. Three collections, all cheap relative to a wave.

**0. The counter-free first pass: an active-CU upper bound from a trace you already have.** Before
touching a profiler, take the kernel trace you collected for timing, and for each kernel read its
**grid size in blocks** and its **duration**. At any instant a kernel can have at most
`min(grid_blocks × blocks_per_CU, CUs)` CUs doing its work; average that over the timeline weighted
by duration. This is an *optimistic upper bound* — it assumes every resident block is issuing every
cycle — which is exactly what makes it useful: **a low upper bound is conclusive.** A published
example: eleven kernels on a 148-SM part gave a duration-weighted bound of 64.65/148 = **43.7%**,
before a single counter was read, and that one number said the operator's problem was that most of
the machine was idle rather than that any kernel was slow.

The characteristic pattern to look for is a handful of kernels with tiny grids — 12, 16, 32 blocks
on a device holding hundreds — sitting on the critical path. Each of them pins the whole device to
its own grid size for its whole duration, and no amount of tuning *inside* such a kernel can raise
that; only letting other work run beside it can. Report the bound per stage and the weighted mean.
It costs one pass over a trace and it frequently reframes the analysis before you spend a lease.

**1. Per-pipe utilization, from hardware counters.** On AMD, `rocprofv3 --pmc`:

```
SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_WAIT_ANY SQ_LDS_IDX_ACTIVE SQ_LDS_BANK_CONFLICT
```

Two traps that have each cost a day:

- **SQ counters aggregate every shader engine** — they are a whole-GPU number, not a per-CU one.
  Divide by the CU count before you compare against a per-CU issue rate, or every utilization you
  compute will be off by two orders of magnitude in the reassuring direction.
- **`VGPR_Count` is reported in units of 2 on some targets** (gfx950 prints 128 where the ISA's
  `.vgpr_count` is 256). Cross-check any occupancy you derive from it against the compiled ISA
  metadata before you build an argument on it.
- Get the **engine clock under load**, not the boot clock, or every cycles→time conversion is wrong.

**2. The inter-kernel gap**, from the trace: end of kernel N to start of kernel N+1, per boundary,
reported as a distribution and not a mean. This one number prices the entire launch-count argument.

## Reading the table

Lay the pipes out side by side per stage. Then classify — the class decides which levers are even
candidates, and it is a small closed set:

| observation | class | what actually helps |
|---|---|---|
| one pipe ≥ ~80% | **throughput-bound on that pipe** | do less work on that pipe, or move work to an idle one |
| all pipes low, `SQ_WAIT_ANY` high, gaps ~0 | **latency / dependency-stalled** | more independent work *in flight per wave* — prefetch, software pipelining, deeper unroll, async copy |
| all pipes low, gaps large | **launch / host-bound** | fuse launches, graph capture, reduce host work |
| pipes low, occupancy already maxed | **not an occupancy problem** — see below | stop adding waves |

The middle row is the one people misdiagnose, and they misdiagnose it in a specific direction: they
reach for **occupancy**. More waves per SIMD is the standard answer to "the machine is stalled", and
it is the wrong answer here. Occupancy hides latency by having *another wave* ready to issue. That
only works if the stall is a long-latency load with other waves available to cover it. If every wave
is stalled on the *same* dependency at the *same* point — which is what a tight GEMM inner loop with
no prefetch looks like — then adding waves adds stalled waves. Worse, it is actively harmful:
occupancy and per-wave register/LDS budget trade against each other directly, and software
pipelining is exactly the thing that needs those registers. **Raising occupancy can buy out the fix.**

A program that has kill-gated occupancy at *both* ends — proved fewer workgroups is not slower, and
proved more waves per SIMD is measurably worse — has proved it is in the middle row. Record that as a
closed axis (`tech_lead.md` rule 3e) and stop spending leases there.

## Instruction-level overlap is the lever the tile graph cannot see

`tile_task_graph.md` says one instruction is "too fine" to be a node, and for *scheduling* that is
right — there is no hardware scheduler entry per instruction. But it is exactly the granularity at
which overlap is *won*, because what fills an idle pipe is not another workgroup, it is another
**instruction issued from the same wave** while the first one's result is still in flight.

So for each stage, ask this and write the answer down:

> At the point where this stage stalls, what work exists that (a) the DAG says has no incoming edge
> from anything not yet complete, and (b) uses a *different* pipe from the one the stall is waiting on?

The candidates are almost always the same shapes, and they are almost always sitting on the far side
of a kernel boundary:

- **Weight / constant loads for the next GEMM.** Weights are typically module-resident tensors,
  addressed statically, with no dependency on any activation. `B` of GEMM2 can be in flight during
  GEMM1's MFMAs — the DAG has no edge there at all. Verify it by finding where the pointer comes
  from: a constructor-set tensor is dependency-free; one written by a previous stage is not.
- **Scale / quantization metadata**, same argument, and usually small enough to be pure latency.
- **Index, offset, and shape preprocessing** — expert descriptors, cumsum, sorted-token-id tables.
  These are VALU and scalar work, they do not touch MFMA, and they are frequently recomputed inside
  a stage that is MFMA-bound.
- **The epilogue of the previous tile** overlapped with the load of the next.

The reason none of these are happening today is usually not that the compiler refused. It is that
**a kernel boundary is a full grid-wide barrier plus a pipeline drain**: nothing on the far side of
it can be in flight, so the prefetch is not merely unscheduled, it is *inexpressible*. That is a
different and much stronger argument for fusion than launch overhead, and it survives in exactly the
case where the launch-overhead argument dies — a measured inter-kernel gap of zero.

State it that way in the proposal. "Fuse to remove 0.6 µs of launch cost" and "fuse so that GEMM2's
weight loads can be issued during GEMM1's MFMA shadow" are different claims with different
implementations and different falsifications, and only the second one is worth a lease when the gap
is zero.

### The dependency is on the first READ, not on the task

This is the reframing that turns the list above into a recipe. A consumer task B that depends on
producer A is usually modelled as "B waits for A". That is too coarse. What actually depends on A is
**B's first load of A's output** — and in a real GEMM-shaped task that load is preceded by a
substantial amount of work that touches nothing A produces:

- reading B's own task descriptor / instruction,
- constructing scheduler and barrier state, initialising the stage barriers,
- building descriptors for the async-copy engine,
- **loading B's weights and scales** — the largest item, and pure memory traffic.

All of it is dependency-free, and in the multi-launch shape all of it is stuck behind the barrier.
So the fused form is not "B starts earlier"; it is **B's prologue runs during A**, and only B's
activation load waits.

The implementation shape that makes this real is a **split loader**: give the input operand and the
weight operand *different* loader warps with different wait conditions. The activation loader
(`ALoader`) waits on the readiness signal for A's tile; the weight loader (`WLoader`) does not wait
at all, and runs ahead filling the pipeline up to `kNumStages` deep. Both then meet at the
per-stage full barrier that the MMA loop already waits on, so the main loop is unchanged and the
correctness argument is unchanged — the activation half still cannot outrun its dependency. The
same trick applies to a paged shared-memory scheme: release a page as soon as its stage is consumed
and the *next* instruction's weight load can start into it immediately, so weights are always
loading.

Two things to check before proposing it. First, **the weights must actually be dependency-free** —
trace the pointer to a constructor-set or module-resident tensor; one written by a previous stage is
not. Second, running ahead costs registers and shared memory per stage in flight, which is the same
budget occupancy wants; that trade is the subject of the previous section, and it is normally the
right way to spend it.

### Charge the framework, then subtract it

A persistent/fused kernel has a **fixed cost that exists even with no work in it**: the resident
blocks, the instruction/queue machinery, the page and barrier bookkeeping. Measure it directly by
running the framework with an empty instruction stream — one published implementation measured
**4.671 µs** out of a 44.272 µs run, i.e. more than 10% of the fused time was scaffolding. Report
the fused number **both ways** (raw, and minus framework), because the two support different claims:
the raw number is what the operator costs, the net number is what the *scheduling idea* is worth,
and quoting only the second one is the standard way this technique gets oversold.

That fixed cost is also the reason a fused kernel can lose to the multi-launch form on small shapes
while winning on large ones — it is amortized, so it is a per-guard effect, not a global one.

### The cost of one resource shape for every op

A fused kernel's resident blocks have **one** register allocation, one block size, one shared-memory
budget, and every op it swallows must live inside them. That is a real cost and it is the substance
of the standing counter-argument to megakernels (stated as far back as 2013, and independently in
the graph-processing literature: differing register demands per op mean fusing *everything* is not
optimal, and some ops are better left outside). Concretely: an MMA op that needs paired blocks
constrains how blocks can be handed out to everything else; a top-k or reduction op that wants 1024
threads runs at a fraction of its natural width inside a resident block sized 384 for the GEMM.

So when you fuse, name the op with the *worst* fit and price it. If the answer is that one op loses
more than the fusion gains, the correct design is a fused kernel that does not contain that op —
which is a legitimate and defensible outcome, not a failure to finish the job.

## Pricing a direction against the table

Every GPU direction should carry, before it is ranked:

- **which pipe it fills** (or which pipe's work it removes),
- **that pipe's measured current utilization**,
- **the headroom it can plausibly claim** — bounded above by the idle fraction of that pipe over the
  interval it applies to, and by the critical path from the DAG. A direction claiming more than
  `min(pipe headroom, e2e − critical_path)` is arithmetically wrong and can be rejected without a
  run.

Two directions filling the *same* pipe are not additive, and the second one should be planned as a
follow-up rather than a parallel arm. Two directions filling *different* pipes may be.

A direction that cannot name a pipe is not yet an optimization; it is a hunch. Send it back.

## Cross-stage cost asymmetry is a finding, not noise

If two stages do comparable MFMA work at different cost per MFMA, that ratio is a first-class
result and it is usually more actionable than anything in the DAG. Normalize: nanoseconds per
1e3 MFMA instructions, per stage. Then attribute the difference by walking the other counters —
VALU per MFMA, LDS index cycles per MFMA, bank conflicts per LDS access. A stage costing materially
more per MFMA while dragging a much larger bank-conflict rate has told you both the size of the prize (bring it to
parity) and the mechanism (LDS layout / swizzle), and neither of those is visible in a dependency
graph.

Price it in end-to-end terms before you propose it: `(cost_ratio − 1) × that stage's share of e2e`.

## The artifact

Emit alongside the task graph, not instead of it:

- `pipes[]`: `{stage, pipe, utilization_pct, source}` for at minimum VALU / MFMA / LDS / HBM.
  `source` is the counter expression, so a reader can recompute it.
- `interkernel_gap_us`: `{median, max, n_boundaries}` — the launch-overhead argument, priced.
- `class`: one of `throughput_bound | latency_bound | launch_bound | mixed`, with the row of the
  table above that produced it.
- `stall_reason` per stage: what the waves are waiting on, and the counter that says so.
- `idle_pipe_opportunities[]`: `{stage, idle_pipe, candidate_work, dag_edge_status, blocked_by}` —
  `dag_edge_status` cites the task-graph finding that there is no edge; `blocked_by` is what makes
  it inexpressible today (`launch_boundary`, `register_pressure`, `no_async_copy`, ...).
- `closed_axes[]`: levers this table rules out, with the counter that rules them out. This is the
  highest-value field on the page, because it is what stops the next wave from re-running them.
- `unknowns[]`: counters you could not collect and what would settle them. A missing counter is a
  fact; an estimated utilization presented as a measurement is not.

## Provenance

The active-CU bound, the split-loader/prologue recipe, the empty-framework cost and the
one-resource-shape argument are from public megakernel implementation write-ups and from the
long-standing counter-literature on fusing everything into one kernel; their numbers are other
people's, on other hardware, quoted for the *shape* of the effect.

The rest is distilled from a GPU optimization program that spent four waves on occupancy, launch-count and
communication-overlap directions before collecting a single hardware counter, and then found that
the counters had foreclosed all three before any of them was bought. What the counters said is
deliberately **not** reproduced here: this is a method card, and a card that ships one operator's
pipe table teaches the next run to expect that table instead of collecting its own. The illustrative
percentages in the sections above are shape, not data.

**Collect your own before you classify.** An operator whose table you have not measured is
`unknowns`, not `latency_bound`.
