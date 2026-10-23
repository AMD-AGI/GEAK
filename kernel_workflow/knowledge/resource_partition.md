# Resource Partition — Who Runs Where, and Why That Is a Decision

A **method** card. The companion to `tile_task_graph.md`: that one asks *what may run concurrently*,
this one asks *what actually gets hardware, and at whose expense*. They are different questions and
conflating them is the most common way a correctly-derived DAG turns into a kernel that is no
faster.

Once the graph says two things may run at once, someone must decide who gets the CUs. In
multi-launch code that decision is made for you (badly, but consistently: everything, then
everything else). The moment you overlap anything, **you own it** — and an unowned allocation
decision defaults to whatever the launch configuration happened to be, which is a decision made
before the analysis existed.

## The specialization ladder

Homogeneous parallelism — every worker does the same thing on different data — is the default and
is often wrong for the parts of a workload that are small in compute but globally depended-upon.
Specialization is available at four levels, and they compose:

| level | unit specialized | typical use |
|---|---|---|
| **warp/wave** | waves within a workgroup | producer waves issue loads, consumer waves do MFMA |
| **workgroup/block** | blocks within a grid | a block does reduction/scheduling while others compute |
| **CU/SM role** | partition of the device's CUs | dedicated scheduler CUs vs worker CUs in a persistent kernel |
| **device/rank** | whole GPUs in a multi-GPU group | one rank does routing/indexing, the rest do bulk compute |

The last row is the one nearly every distributed implementation skips, because the frameworks make
symmetric execution the path of least resistance: every rank runs the same program on its shard. Ask
explicitly whether that is right. **A stage that is small in FLOPs, globally depended-upon, and
sync-heavy is the worst possible thing to replicate across all ranks** — replicating it multiplies
the sync traffic, duplicates the compute, and puts the result on the critical path of every rank
instead of one. Assigning it to a single rank that broadcasts the result can be a larger win than
anything done to the bulk compute, and it is invisible to any analysis that assumes symmetry.

This is a *direction to evaluate*, not a recommendation. It costs load balance and it costs a
broadcast. Evaluate it: `duplicated_cost × ranks` vs `broadcast_cost + idle_on_specialized_rank`.

## Fixing roles: when, and why not later

In a persistent kernel, the split between roles (worker CUs, scheduler CU(s)) is **fixed at launch
to the physical CU count, and does not change during execution.** That is a real constraint, not an
implementation shortcut: dynamic role switching requires a global agreement protocol on the critical
path, which costs more than the imbalance it fixes. Consequences to plan for:

- The partition must be right for the *whole* execution, including its worst phase. Size it for the
  phase with the least parallelism, not the average.
- Over-provisioning the scheduler is cheap and under-provisioning it is catastrophic — a scheduler
  that becomes the bottleneck serializes everything behind it.
- You are launching *exactly* as many workgroups as the device can hold co-resident. Occupancy
  arithmetic (registers, LDS) stops being an optimization detail and becomes a correctness
  precondition: if one workgroup fails to be resident, a dependent task never runs and the kernel
  hangs. Compute it, don't tune it.

## Static vs dynamic scheduling

Given a legal execution order (the BFS linearization from `tile_task_graph.md`), who assigns tasks
to hardware?

| | static (host builds per-CU queues) | dynamic (on-GPU scheduler hands out work) |
|---|---|---|
| runtime overhead | ~none | scheduler contention, a shared structure to poll |
| load imbalance | fully exposed — a slow queue stalls its CU | absorbed — idle CU takes the next task |
| scaling | fine at any CU count | contention grows with CU count if the queue is centralized |
| needs | task durations predictable | nothing |

The rule that falls out: **static wins for regular, uniform work; dynamic wins when task durations
are data-dependent.** If your workload's per-task duration varies with runtime data (variable
segment lengths, skewed distributions, anything routed), static scheduling converts that skew
directly into idle CUs and a static schedule built from *average* durations will look excellent in a
uniform benchmark and collapse in a skewed one. If your guards include a skewed case, this is very
likely the mechanism behind the gap between them — and it is a scheduling finding, not a
communication finding, so no amount of overlap work will fix it.

If you go dynamic, the queue is the contention point. A single global queue polled by every CU is
the obvious design and the one that stops scaling first; hierarchical or work-stealing structures
cost more code and defer the wall.

## Wave quantization — the tail nobody profiles

`waves = ceil(blocks / (CUs × blocks_per_CU))`. The **last wave is almost always partial**, and
during it the device is running at `(blocks mod capacity) / capacity` utilization. A grid of 512
blocks on a device holding 148 concurrently runs 3 full waves and a final wave of 68 — the tail
executes at 46% of the machine for a full wave's duration.

Two things follow:

- **This tail is invisible in a per-kernel time and obvious in a task graph.** It is one of the
  largest bubbles in most multi-launch operators and it is not a launch-overhead problem, so
  removing launches does not remove it.
- **It is the bubble that fusion actually fixes**, and the mechanism is worth naming precisely: with
  independent work available, the partial tail wave of one stage is filled by tasks from another.
  That is a *scheduling* effect, which is why it needs both a dependency graph and a resource
  policy — the graph says the other work is legal, the policy makes it resident.

Report `waves` and the tail occupancy per stage. It is three numbers and it frequently reframes the
whole analysis.

## Resources that outlive a task

In a persistent kernel, on-chip state can span task boundaries — that is most of what makes fusion
worth its cost, and it needs explicit management:

- **Paged / arena-allocated LDS.** If LDS is statically partitioned per task type, the kernel's
  occupancy is set by the *worst* task and every other task wastes the difference. Treating LDS as a
  pool that tasks allocate from lets a buffer's lifetime span tasks — a producer's output stays
  on-chip for its consumer instead of round-tripping through HBM, which is the single largest
  concrete benefit of the persistent form.
- **Cross-task software pipelining.** With multiple tasks resident, one task's loads are issued
  while another's math runs. This is the same idea as intra-kernel prefetch, one level up, and it is
  where a persistent kernel recovers latency that a sequence of kernels structurally cannot.
- **Continuous prefetch.** Weights/operands for a *later* task can stream in during an earlier one;
  in a multi-launch form each kernel re-establishes its own working set from cold.

Each of these is a reason fusion might pay that is *independent of launch overhead*. If your fusion
argument rests only on launch count (`fusion_preconditions.md`, rung 4), these are the arguments you
should be making instead — and they are testable: measure the HBM traffic the round-trip costs.

## The floor is per engine, not a scalar

A device has several execution resources — MFMA/VALU issue, VMEM/LDS issue, HBM and, on a multi-GPU
rank, the interconnect. Their use can overlap, but they are not automatically independent. Represent
each stage as a demand vector over those resources; the lower bound is the maximum load on any
binding engine, plus dependencies that prohibit overlap. This distinction decides whether a fusion
can pay:

- **Same engine, static split** (two GEMMs, both MFMA-bound). A static `f/(1-f)` CU split of two
  throughput-limited stages of durations A and B runs in `max(A/f, B/(1-f))`, minimized at `A+B` —
  exactly the serial time. This disproves a permanent role partition; it does **not** disprove a
  phase-staggered CTA pipeline. If each CTA switches to ready downstream work when its own upstream
  shard drains, other CTAs may still execute the upstream stage. That schedule fills partial-wave
  tails and can overlap downstream VMEM/P2P/epilogue work without claiming two simultaneous units of
  MFMA throughput. Evaluate its span and tail occupancy; do not close it from engine labels alone.
- **Different binding engines.** If measured counters show the stages do not contend on any binding
  resource, summing their times overstates the floor: the smaller engine's work may hide under the
  larger's. The prize is `serial − max_engine_floor`, and it is captured only if the consumer really
  runs during the producer. Relocating it into the same launch behind a whole-grid join preserves
  the serial order and overlaps nothing.

The failure mode this prevents: computing one scalar floor by summing every stage's time, proving
`overlap = A+B` against it, and reporting "fusion does not pay" — when the two stages were on
different binding engines and the real floor was closer to a maximum than a sum.

The inverse failure is just as costly: calling a shader-issued P2P store "DMA" and treating it as
free overlap. Such a store still needs resident waves, issue/VMEM capacity, registers and usually a
CU-side epilogue; a cross-rank reduce also contains local loads and arithmetic. Measure those
components before assigning them to an engine. Report the floor as a vector backed by counters or
controlled arms, not as one scalar guessed from stage names.

## The moving bottleneck

State this in every proposal that changes an allocation:

> Before: critical path runs through X, taking N µs.
> After: X is given more CUs / overlapped; expected N′ µs.
> **The critical path then runs through Y**, and the next bound is M µs.

If you cannot name Y, you have not modelled the graph — you have modelled one node of it. Adding
resource to a path that had slack changes nothing; adding it to the critical path moves the
bottleneck somewhere you did not measure. Both outcomes look like "the optimization didn't work"
from the benchmark, and only the graph distinguishes them.

## Checklist for the artifact

- `waves` and tail occupancy per stage.
- Proposed role partition, if any: unit (wave/block/CU/rank), split, and the phase it is sized for.
- Scheduling policy: static or dynamic, with the data-dependence argument for the choice.
- On-chip state that spans tasks, and the HBM traffic that saves.
- Predicted post-change critical path — the node, not just the number.
- For a symmetric multi-rank operator: one line on whether asymmetric rank roles were considered and
  the arithmetic that rejected (or accepted) them.

## Provenance

Distilled from public work on persistent-kernel runtimes and task-graph compilers: fixed
worker/scheduler CU partition at launch, paged shared memory and cross-task software pipelining;
device-level specialization breaking the symmetric-rank assumption in a multi-GPU serving runtime;
static-vs-dynamic scheduling results on data-dependent workloads; and standard wave-quantization
arithmetic. Numbers are illustrative and from other hardware. **Measure yours.**
