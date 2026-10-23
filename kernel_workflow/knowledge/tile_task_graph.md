# Tile-Level Task Graph — Deriving the Dependency DAG Before Proposing Anything

A **method** card. It contains no operator's answer. It tells you how to build the object that an
optimization proposal must be derived *from*, so that the proposal is a conclusion rather than an
assertion.

The failure this exists to prevent: an Analyze phase that writes "the stages are serialized, so fuse
them" and calls that analysis. That is a *restatement of the launch count*, not a dependency
analysis. It cannot tell you which edges are real, which are artifacts of how the code is written
today, which are on the critical path, or how much slack a given edge is hiding. Every one of those
questions has a determinate answer, and none of them can be answered at kernel granularity.

## The granularity rule

**A kernel is not a node.** A kernel is a *batch of nodes that happen to have been launched
together*, plus an implicit all-to-all barrier at each end that nobody asked for.

Build the graph at the granularity at which data actually becomes ready:

| level | node | typical unit |
|---|---|---|
| too coarse | one kernel | "GEMM1" |
| **usable** | one **tile** of one stage | one output block of GEMM1 for one expert |
| too fine | one instruction | not schedulable, no scheduler entry per instr |

Pick the tile as the unit the *producer already writes atomically* — the block/workgroup output
tile. If a stage's output tile is consumed in a different tiling than it was produced in, that
mismatch is itself a finding: it is the reason the edge is currently a full barrier.

**Partition by tile, not by whatever semantic group the operator names.** Operators come with a
natural-looking unit that is *not* the tile — an expert, a sequence, a head, a bucket, a segment —
and it is tempting to make that the scheduling unit because the code is already organized that way.
Resist it. A semantic group has data-dependent size, so partitioning the schedule by group inherits
the group's imbalance directly: the largest group becomes the tail, and no amount of overlap fixes a
unit that is intrinsically uneven. A tile has a fixed shape by construction. Where the two disagree,
the group boundary belongs in the *indexing*, and the tile stays the scheduling unit. This is a
recurring result rather than a subtlety — several independent systems converged on it after first
shipping the group-partitioned version.

## The edge rule

There is an edge from task A to task B **iff A's output region overlaps B's input region.** That is
the whole rule (it is the rule Mirage Persistent Kernel uses to build its tGraph, and it is the only
rule you need). Two consequences engineers routinely miss:

- **Overlap is per-region, not per-buffer.** "B reads the tensor A writes" is not an edge; "B reads
  *the bytes* A writes" is. If B's tile reads rows 0–127 and A's tile writes rows 128–255, there is
  no edge between those two tasks, however much the kernel-level picture suggests one. Most of the
  false serialization in a multi-launch operator lives exactly here.
- **The absence of an edge is a finding.** The pairs of tasks with no path between them in either
  direction are the parallelism that exists in the problem. You are not creating parallelism by
  fusing; you are *stopping the current code from destroying it*.

## The operands the graph structurally cannot see

Nodes are output tiles, so an edge only ever describes the operand that flows along it. Every other
operand the consumer reads — the next stage's weight matrix, its scales, the descriptors, the index
and offset arrays — is produced by nobody. It has no node, so it has no edge, so no amount of
correct edge analysis will ever surface it, and the graph is at its most convincing exactly when it
is silent about this.

It matters because those operands **depend on nothing**. They can be loaded while the producer is
still running. That is a different lever from making the dependent operand arrive earlier: it does
not need a readiness flag, a fence, or a counter — only address arithmetic that does not wait — and
it is usually the cheaper of the two to build.

So on every edge with a `fan_in`, record `producer_independent_operands`: the list of things the
consumer needs that this edge does not carry. It is one static read of the consumer's signature.
`[]` is a real answer.

The cost of not asking, measured on a real operator: the graph had six nodes and five edges, found
both fusable edges correctly, priced them, and never once mentioned that the second GEMM's weights
depend on nothing at all — while its own pipe table reported both stages at ~30% HBM, which is the
headroom that would have paid for exactly that prefetch.

## Edge scope — the column people forget

For each edge, record **what has to be true for the consumer to safely proceed**, at the narrowest
level that suffices:

| scope | what must be visible | rough cost class |
|---|---|---|
| register / lane | value in a register | free |
| LDS / shared, same workgroup | shared store + workgroup barrier | ~tens of ns |
| L2 / same die | release store + acquire load, device scope | ~hundreds of ns |
| HBM / cross-die (multi-XCD) | device-scope release + cache maintenance | higher; measure it |
| cross-rank / interconnect | peer store visible + a flag the peer polls | µs-class; measure it |

Two rules follow, and they are the point of the column:

1. **Sync at the lowest level that is sufficient.** An edge whose producer and consumer land on the
   same workgroup does not need a device-scope fence. An edge inside one die does not need
   cross-die cache maintenance. Multi-die GPUs make this a *hierarchy*, not a binary, and the
   default in most code is to fence at the top of the hierarchy for every edge because that is what
   is always correct. "Always correct" is where the time goes.
2. **A cheap fence is not a free fence.** A published counter-datum to keep you honest: a
   global-memory sync between producer and consumer has been measured at **≳1.2 µs** on real
   hardware — i.e. an edge you sync naively is *latency-bound* unless something else is overlapped
   with it. If your graph has hundreds of fine-grained edges and you plan to enforce each with a
   global flag, do that multiplication before you write the kernel. Others have measured total sync
   cost as low as ~2% of a megakernel; both numbers can be true, and which one you get is decided by
   the scope column, not by whether you fused.

## What enforces this edge *today*

For every edge, one of:

- `launch_boundary` — the only thing enforcing it is that the consumer is in a later kernel. **This
  is the interesting class.** It is an edge the hardware is enforcing far more strongly than the
  data requires: a launch boundary is a full grid-wide barrier plus a pipeline drain.
- `barrier` — an explicit in-kernel barrier or grid sync.
- `fence/flag` — an explicit release/acquire or a polled readiness flag; already fine-grained.
- `none_needed` — you determined the regions do not actually overlap; the apparent edge is false.

The count of `launch_boundary` edges whose true scope (previous section) is *narrower* than
grid-wide is the size of the opportunity. Report that number. If it is small, say so — that is a
legitimate and valuable Analyze result, and it is one that will save the project a wave.

## How many events does an edge need — the pairing decision

An edge in the graph is realized at runtime as a **counter**: the event lowers to an ordinary
integer in memory, the producer does an atomic decrement (or increment) when its tile is done, and
the consumer spin-waits until the counter hits its target. That is the whole mechanism; there is
nothing exotic underneath, which is why the interesting decision is not *how* to signal but **how
many counters to allocate for one producer/consumer pair.** Four strategies, in increasing
precision and increasing cost:

| strategy | one counter per | consumer waits for | when it is right |
|---|---|---|---|
| `whole` | producer stage | all producer tiles | tilings unrelated, or the consumer's first op is a full reduction |
| `tile` | producer tile | the one tile it reads | tilings identical, 1:1 |
| `tile_cover` | consumer tile | every producer tile intersecting its input region | tilings differ; the general case |
| `tile_reduce` | consumer tile | the N producers accumulating into it | producer writes are a partial-sum fan-in |

Two rules that follow, and both are load-bearing:

- **`whole` is the correct default for any pair you have not analyzed.** It is exactly the barrier
  the multi-launch form already has, so it is never a regression, and it lets you fuse an operator
  incrementally — convert the edges you have proven, leave the rest at `whole`, and the kernel is
  correct at every step. A fused kernel that must have all its edges refined before it runs at all
  is a fused kernel you cannot bisect.
- **Counters are not free and the count is the cost.** `tile_cover` on a pair with fine tiling on
  both sides can allocate thousands of counters and issue an atomic per producer tile per consumer;
  price that against the window from `fusion_preconditions.md` before choosing it. Refining an edge
  whose window is 3 µs into 2000 counters is a measurable loss with a correct-looking design.

Record the chosen strategy per edge in the artifact. It is the field that most directly determines
whether the implementation matches the analysis, and the one most likely to silently drift back to
`whole` during debugging.

## Critical path and slack

With node durations (measured, from the profile — not estimated) and the edge set:

- **Critical path**: the longest path through the DAG. Its length is the floor on end-to-end time
  for *any* schedule, fused or not. Compare it to the measured end-to-end time. The gap between them
  is the total addressable inefficiency; **an optimization that claims more than that gap is
  wrong.** This one comparison kills more bad proposals than any profile.
- **Slack** per node = latest-start − earliest-start. A node with slack is not worth optimizing;
  making it faster changes nothing until its slack is exhausted. Nodes with **zero** slack are the
  only legitimate targets.
- **Where the critical path runs** is more informative than its length. If it runs entirely through
  compute nodes, communication is already hidden and the fusion argument is dead. If it alternates
  compute → wait → compute, the waits are exposed and the alternation tells you the tile granularity
  at which handover would help.
- **Slack is not static.** Speeding up the critical path moves it somewhere else. Re-derive after
  each accepted change, and state in the proposal *where you expect the critical path to move to*.
  A proposal that does not predict the next bottleneck has not modelled the graph.

## Making the graph smaller before you act on it

Three transformations, in order (this is MPK's normalization pipeline, and it is generic):

1. **Event fusion.** If several edges share the same producer set and the same consumer set, they
   are one event, not N. Do this first; it usually collapses the graph by an order of magnitude and
   most "the graph is too big to schedule" conclusions are really "the graph was never normalized".
2. **Normalization.** Rewrite until each task has at most one triggering event and at most one
   dependent event, inserting trivial events where needed. Now every task's readiness is a single
   integer counter, which is the only form that lowers to cheap hardware: an atomic decrement by
   each producer, and a consumer that proceeds when it hits zero. Anything more complex than a
   counter means a scheduler doing real work on the critical path.
3. **Linearization.** BFS the normalized graph to get an execution order that respects dependencies;
   that order is what a worker queue holds. Note this is a *legal order*, not an optimal one —
   whether the assignment of that order to hardware is any good is the resource question, not the
   dependency question. Keep them separate; see `resource_partition.md`.

## Dynamic shape and data-dependent structure

If the graph's *shape* depends on runtime data (routing decisions, variable-length segments, top-k
selections), you cannot fully build it on the host. Two mechanisms, both usable independently:

- **Data-dependent event update** — the graph is structurally fixed but the *wait counts* are
  computed at runtime from a routing/index tensor. Cheap: the counters are just written by an
  earlier task. Use this whenever the set of tasks is knowable but their fan-in is not.
- **Data-dependent triggering** — the *set* of tasks to run is decided at runtime; the scheduler
  admits tasks as their existence becomes known. Necessary when whole branches may be empty.

Both are ordinary integer-tensor work, and both are far cheaper than the alternative of falling back
to a host-side barrier per layer. When the shape is dynamic in *size* but static in *structure*,
prefer a symbolic-shape template compiled once over recompilation per shape — the compile cost is
otherwise paid on the critical path of the first request of every new shape.

## The artifact

An Analyze phase that has done this work can emit it. One that has not, cannot. Emit:

- `nodes[]`: `{id, stage, tile, duration_us, source}` — `source` says where the duration came from
  (profile trace / derived / assumed; an assumed duration is a declared unknown, not a fact).
- `edges[]`: `{from, to, scope, enforced_by, bytes, pairing}` using the three vocabularies above
  (`pairing` ∈ `whole | tile | tile_cover | tile_reduce`, with the counter count it implies).
- `critical_path[]`: node ids in order, plus `critical_path_us` and the measured end-to-end for
  comparison.
- `slack_us` per node, or at minimum the set of zero-slack nodes.
- `false_edges[]`: pairs the kernel-level view implies but the region analysis rules out.
- `unknowns[]`: every edge whose scope or duration you could not determine. **A short honest graph
  beats a complete invented one**; an unknown is a fact, an estimate presented as a measurement is
  not.

Then, and only then, argue for a change — and see `fusion_preconditions.md` before you argue for
fusion specifically, because the graph being serial today is not by itself an argument.

## Provenance

Distilled from public literature on megakernel/task-graph compilation (Mirage Persistent Kernel's
tGraph construction, event fusion, normalization and BFS linearization; EventTensor's event-as-tensor
lowering and its data-dependent event update / task triggering; the event-pairing strategies and the
partition-by-tile-not-by-group result from published megakernel implementations and their
follow-ons; published measurements of hierarchical sync cost on multi-die GPUs). The numbers in this file are other people's, cited as
order-of-magnitude calibration for the *shape* of an effect. **Measure your own before you rely on
one.**
