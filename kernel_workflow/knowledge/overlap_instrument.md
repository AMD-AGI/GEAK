# Measuring overlap inside a fused kernel — and why nothing you already have can do it

Read this together with `tile_task_graph.md` (which edges could overlap) and `fusion_preconditions.md`
(whether a given edge pays). This file is about the separate question of **how you would know**.

## The problem: every instrument you have goes blind exactly when the thing you want starts existing

Before fusion, "is there overlap?" is answerable for free. Four launches, four kernel records in the
trace, and if the intervals do not intersect there is no overlap. That is where the observation "the
four stages are strictly serialized — measured kernel-level overlap is zero" comes from.

Now fuse them. There is one kernel. The trace has one record. Its begin and end tell you the wall
time and nothing whatsoever about what happened inside, and the per-stage timers are no better:
they are separate rank-reduced timers over stages that used to be serialized, and once a stage may
start early it charges its own waiting to itself. **Both stage timers can rise while the operator
gets faster.** So at the exact moment overlap becomes possible, all three of the instruments that
would have detected it stop being able to.

This has a consequence that has to be said plainly, because it is the failure mode this whole file
exists to prevent:

> **A fused kernel that is faster is not evidence of overlap.** It is evidence of *something*. Fusing
> also removes launch overhead, keeps data in L2 across a boundary, and lets a scheduler fill gaps
> that the old grid shape could not. Those are real wins and none of them is overlap. If the
> acceptance bar asks for "genuine compute/communication overlap, not serialization disguised by
> kernel boundaries", a latency number cannot answer it — in the fused shape, latency is *precisely*
> the quantity that cannot distinguish the two.

The existing doctrine says "a latency win with no measured change in overlap is suspicious, not
accepted." That sentence has been in the task text for several waves with no instrument behind it,
which makes it unenforceable: there was no way to produce the measurement it demands, so the honest
answer was always "not measured" and the rule never bit. Building the meter is what makes the rule
real.

## What to build

An in-kernel, per-workgroup phase log. Nothing exotic:

1. **A cycle counter.** This snapshot does *not* define a `read_memrealtime` helper — you must add
   the `s_memrealtime` inline-asm helper yourself (`communication_ops_utils.py` is where the other
   fences and atomics live). `s_memrealtime` is a constant-rate counter, which is what you want;
   `s_getreg(HW_REG_SHADER_CYCLES)` is a clock-rate counter and drifts with DPM state.
2. **A preallocated device buffer**, sized `n_workgroups × max_slots × 4` u64, plus one overflow
   counter. Preallocated and fixed-size on purpose: a meter that can allocate is a meter that can
   perturb the allocator you are measuring.
3. **Two writes per phase per workgroup.** On entering and leaving each *role* — dispatch-wait,
   gemm1, gemm2, p2p-publish, combine — one lane of the workgroup writes `(role_id, t)`. One lane,
   not one per wave: you are timing the workgroup's occupancy of a role, not each wave's.
4. **Read it back on the host** and build, per role, the set of intervals during which at least one
   workgroup was inside it.

Gate the whole thing behind the same kind of env-var switch as the fused path itself, and give it
its own one-line marker. A meter that silently did not compile in is worse than no meter.

## The two numbers, and why one alone is not enough

Let `A(t)` = the number of *distinct roles* with at least one active workgroup at time `t`.

- **`overlap_fraction`** = `|{t : A(t) ≥ 2}| / |{t : A(t) ≥ 1}|`.
  Simple, and it reads ~0 on the scattered path, which is what makes it checkable.

- **`co_resident_cu_fraction`** = the fraction of *CU-time* (not wall time) spent by workgroups of
  role `r` while at least one workgroup of a different role was resident. Wall-clock concurrency
  between one straggler block and a full grid is not overlap in any sense that buys latency.

**Report both, because `overlap_fraction` alone can be manufactured.** Let one dispatch-wait
workgroup linger for the whole kernel and `overlap_fraction` goes to 1.0 while nothing whatsoever
has been overlapped. The CU-weighted number is what says how much machine was actually doing two
things at once.

And the third quantity, which is not from the meter at all: **`mega_e2e` rank-max must fall.** A rise
in overlap with no fall in latency is *contention*, not overlap — two roles co-resident and fighting
for the same CUs, issue slots or LDS. That is a real and common outcome, it is a finding, and it
must be reported as one rather than as a partial success.

## Three traps. Each one has produced a plausible wrong number somewhere.

### 1. Clock coherence across XCDs

MI355X is multi-XCD. `s_memrealtime` is constant-rate, but **whether it is synchronised across XCDs
is not something you may assume** — and every interval comparison between two workgroups that landed
on different XCDs is invalid if it is not. This is not a corner case: with a persistent kernel the
roles are deliberately spread across the whole device, so *most* of the comparisons the meter makes
are cross-XCD.

Measure the skew before you trust the meter. Put a grid-wide barrier in a throwaway kernel, have
every workgroup read the counter immediately after it, and look at the spread. That spread is your
comparison floor. Report it as `clock_skew_ns`. If it is comparable to the intervals you are
measuring, you have two honest options — restrict the analysis to same-XCD pairs and say so, or
report the meter as `unknown` — and one dishonest one, which is to present the number anyway.

### 2. The meter perturbs what it measures

Timestamp stores are memory traffic and a serialisation point. Run the same guard with the meter on
and with it off, paired, and report `meter_overhead_pct`. Two consequences:

- If the overhead is comparable to the effect you are claiming, the *overlap* number still stands
  (it is about structure) but the *latency* number must come from a meter-off run. Never quote an
  e2e from a metered run as the result.
- If the overhead is large, suspect that you are logging too finely. Per-workgroup-per-role is
  enough. Per-tile is not, and it will change the schedule you are trying to observe.

### 3. Only-inside-the-lease clocks

Timestamps are only comparable within one kernel launch on one device. Do not compare a timestamp
from rank 3 with one from rank 5; there is no synchronised cross-rank clock here. Cross-rank overlap
has to be inferred from each rank's own local picture plus the arrival counters, not from a shared
timeline. Say which of the two you did.

## The meter needs its own controls, for the same reason the benchmark does

The workflow already refuses to trust a benchmark whose instrument has never read a known value.
The same rule applies here, and it is cheap:

- **Negative control — the scattered path.** Run the meter against the *unfused* four-launch
  baseline. The true answer there is known: kernel-level overlap is zero. **If your meter reads
  anything materially above zero on the scattered path, the meter is broken and every number it
  produces afterwards is void.** This is the single most valuable check in this file, it costs one
  run, and it is the one most likely to be skipped because it is expected to be boring.
- **Positive control — forced concurrency.** Make two roles run at the same time on purpose for a
  known duration (a bounded spin in one while the other works). The meter should report an overlap
  close to what you constructed. A meter that reads 0 on the scattered path and *also* reads ~0 on
  forced concurrency is not a conservative meter, it is a dead one, and the two cases are
  indistinguishable without this second control.

Record both readings. `overlap_fraction` with no control readings beside it is an unverified
instrument, and the workflow treats it the same way it treats a 1.000x with no positive control:
not as a zero, as an untested instrument.

## What to report

```
overlap: {
  measured: "yes" | "no" | "unknown",   // `no` = built the meter, it says there is no overlap.
                                        // `unknown` = could not measure. They are NOT the same and
                                        // must never collapse into one value.
  fraction: 0.0,                        // overlap_fraction, wall-clock, ≥2 distinct roles
  cu_fraction: 0.0,                     // CU-weighted
  method: "in-kernel s_memrealtime per-workgroup phase log | rocprofv3 kernel trace | ...",
  base_reading: 0.0,                    // the negative control. REQUIRED to believe the rest.
  forced_reading: 0.0,                  // the positive control, and what was constructed
  clock_skew_ns: 0,                     // trap 1
  meter_overhead_pct: 0.0,              // trap 2
  note: "what could not be measured and why"
}
```

`measured: "unknown"` is a first-class answer and you should use it rather than reaching for a
number. Naming the collection experiment you could not run is a result. Synthesising a plausible
fraction to fill the field is the specific failure this instrument exists to make unnecessary.

## Trap 4 (added after it fired): a summed `overlap_fraction` reads high on a kernel with zero overlap

A meter that emits one `overlap_fraction` by summing every pair of concurrently-active roles will
report a large, entirely honest, entirely useless number, because most of the pairs it sums have
nothing to do with the edge you fused. Real reading from a whole-grid-join arm — an arm whose own
source comment says it has zero overlap by construction:

```
overlap_fraction=0.43399
pairwise={'dispatch_wait|gemm1': 5636044, 'gemm1|join': 1730112, 'gemm2_p2p|join': 2204}
join_exit_spread_ticks=26  n=256
```

`dispatch_wait|gemm1` (76.5%) is intra-producer overlap that the *unfused* kernel already had.
`gemm1|join` (23.5%) is blocks queueing at the barrier — waiting, not working. The one pair that is
the fused edge, `gemm2_p2p|join`, is **2204 ticks, 0.03%**. The headline said 43%; the answer is 0.

Two rules follow.

**Partition the pairs before you normalise.** Every pair is either ON the edge under test
(producer role concurrent with consumer role) or OFF it (pre-existing concurrency, or queueing).
Only the ON group may be normalised into a headline. Emit the OFF group too — it is useful context —
but never in the same scalar. A composite overlap number is a false positive generator.

**Always emit `join_exit_spread_ticks` (or the equivalent for your sync primitive)** — the spread in
exit timestamps across all participating blocks. It is a one-line veto: a spread of tens of ticks
across hundreds of blocks means everything downstream of that sync is strictly serialised, and any
overlap claim can be rejected on the spot without reading the pairwise table at all. Cheap to
collect, and it fails loudly in the one case where the composite fraction fails quietly.

## If you cannot build it

Then criterion "genuine overlap" is **unmeasured**, and that is what goes in the report — not
"passed", and not silence. A fused kernel with a real latency win and an unmeasured overlap claim is
a good result with a named hole in it, which is worth much more than the same result with the hole
papered over. The workflow will attach the caveat for you; do not pre-empt it by guessing.
