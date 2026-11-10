# Generic lessons for whole-operator fusion runs

You get this file as `MEGA_GENERIC_LESSONS` in a `mode=mega` run with Expert Skills disabled, and in
a MoE `mode=mega` run with a pinned Expert Skill. With a pinned Skill, where a lesson below points at
`MEGA_MEASUREMENT_GUIDE` or the generic graph contract, use the Skill's runtime and the workflow's
Verify instead; the process lessons themselves still apply.

Each lesson below cost at least one full round in an earlier run. None of them names an operator or
a design. They are failure patterns of the *process*: measuring, debugging, and choosing what to
build next. They never override COMMANDMENT, your role file, or a measured result. If one conflicts
with a measurement, the measurement wins; say so in your notes.

Which sections to read:

| role | sections |
|---|---|
| planner (`mega_search_lead`) | all |
| engineer | §1 – §5 |
| verify / director | §1, §2, §5.4 – §5.6 |

---

## §1 Measurement: what counts as a number

1. **Only an on-card, measured result counts.**
   - A plan, a handoff line, "should be faster", or "compiles" is not a result.
   - Carry an unmeasured claim forward as `unverified`, never as progress.
2. **The official metric is the rank-max.**
   - One slow rank sets the operator's time.
   - A change that speeds up the median rank but leaves one straggler is a regression.
   - Always look at the per-rank spread, not only the reduced number.
3. **Paired and interleaved, or it did not happen.**
   - Arms that are timed sequentially drift by more than the effects hunted here.
   - Compare only base/candidate pairs interleaved inside one lease.
4. **Every variant in a sweep gets fresh inputs and fresh output buffers, then a fresh process.**
   - A variant that silently skips work can read the previous variant's correct results from a
     reused buffer. It then looks both correct and fast.
   - Confirm every sweep winner in a separate fresh-process run before you report it.
5. **Prove the activation, don't assume it.**
   - A fused path behind a switch runs only if the measured arm actually exported the switch *and*
     every rank printed the path marker.
   - A flag you added but never exported in the harness measures the old path.
   - Map each new switch to the arm's environment explicitly.
6. **Measure the denominator; never copy it.**
   - If the baseline itself moves (machine state, clocks, a neighbour on the GPUs), every speedup
     moves with it.
   - Before believing a surprising gain or loss, re-time the frozen baseline against its recorded
     number. A whole-run shift in the baseline is a machine problem, not a code effect.
7. **Every guard, not only the target.**
   - A design tuned at the largest shape often regresses the small ones: more setup cost, worse
     occupancy, a heavier synchronization path.
   - A fallback that is *allowed* at small shapes is not *free*:
     - once fallback is allowed, the search tends to settle for it;
     - say per guard whether the fused path ran;
     - report the measured ratio there too.

## §2 Correctness and liveness of a fused multi-rank kernel

1. **Correct once is not correct.**
   - Readiness counters, flags and queues must be correct on replay N of a captured graph, with
     different inputs, not only on the first call.
   - Two workable patterns:
     - monotonic generation counting, where every producer bumps **exactly once per generation**
       and waiters target `generation × expected_count`;
     - an in-graph reset that is properly ordered against every reader.
   - A producer whose bump count depends on data or on how work was tiled breaks generation targets:
     the first run passes, and a later replay hangs.
2. **Everyone you wait on must be resident.**
   - A workgroup spinning on a producer that is not scheduled yet deadlocks.
   - Size a persistent grid from the measured occupancy, not from the problem size.
   - Keep the spin bounded while debugging.
3. **Debug hangs with instruments, not hypotheses.**
   - A blind "fix, run, hang, kill" loop costs a full lease per guess and returns no information.
   - Build three instruments first:
     - a host-readable dump of `{observed, target}` for every counter;
     - a bounded spin that sets a failure flag and exits;
     - the smallest reproduction (e.g. capture plus a few replays of one small case).
   - The graph-contract runner's watchdog already names the stuck phase.
   - If `observed` is 0 everywhere, the publishing path is not running (wiring, gate, dead code). It
     is not an address or offset bug.
4. **Do not abandon a correct protocol because of a black-box signal.**
   - When a design that is known to be sound hangs, first localize which counter never reaches its
     target.
   - Replacing the protocol under an unexplained hang usually trades one hang for another.
5. **Check correctness at every configuration you actually select.**
   - A tile or launch parameter past a supported limit can silently drop part of the output.
   - Measure accuracy per selected configuration (shape class), with fresh buffers (§1.4).
6. **Direct evidence only.** Compare the graph-captured candidate output with the task's numeric
   reference. "Equal to another variant that was correct" is not correctness evidence.
7. **Compare the first call after an input change with a repeated call.**
   - A consumer that reads per-launch metadata (an ordering, a grouping table, counts) before it
     is published sees the previous launch's value. Only the first call after the input changes
     goes wrong; a repeated call heals.
   - The damage can stay under an accuracy gate against the reference (measured: whole rows wrong,
     relL2 0.045–0.107 against a 0.10 gate), or show as a hang.
   - Symmetric inputs hide it, even when the heavy part moves: every rank's stale metadata still
     looks like its current one. Measured on one defect: all ranks skewed alike, 0/32 caught;
     one hot source rank that moves on every call, 32/32 (or a hang).
   - Gate on the first call against a repeated call of the same input at a tight tolerance, and
     make every reader of the metadata wait for it, not only the ones in a particular role.
8. **A baseline that is not bit-exact on repeated input has a bug; find it before tolerating it.**
   - Fixed-order reductions repeat bit-exactly. If two calls with the same input differ, some
     reader sees the wrong value. Measured once: two repeats differed by relL2 0.04–0.08 and the
     cause was one shared on-chip control word that a writer overwrote without a barrier while slow
     readers still needed its first value.
   - Do not widen a tolerance to fit that drift: it also hides the stale reads that item 7 exists
     to catch. Fix the race on every path the candidate runs, including the fallback path.

## §3 Code generation and compiler traps

1. **Keep atomics and other heavy side effects out of the hot loop.**
   - An atomic read-modify-write inside the main loop, or in its tail, can change register
     allocation for the whole kernel.
   - The result is a slowdown or a deterministic fault far from the edit.
   - Publish once per tile or work item, after the loop.
2. **When one edit makes a distant site fault or slow down, compare the kernel metadata first**
   (VGPR/SGPR count, spills, scratch, LDS) before and after. Register pressure is usually the link.
3. **Trace-based DSLs lower loops from the traced program.**
   - A loop-carried variable must exist on every path.
   - Creating one only under a compile-time-constant branch can break the loop lowering.
   - Derive it from values that already exist, and use integer 0/1 instead of a boolean.
4. **Static checks do not catch trace-time or compile-time errors.**
   - A structural checker can pass a kernel that fails the first time it is traced on the card.
   - Compile and run the smallest case on the card early in every lease.
5. **JIT caches are per machine, not per tree.**
   - A flag- or environment-gated arm must prove it compiled to a distinct binary.
   - Run every arm in its own fresh process.
6. **Read the metadata for three common code-generation losses** (§3.2 says where to look):
   - a compile-time-constant address offset that the compiler hoists out of the loop and then
     spills; anchor it to a per-tile runtime base;
   - a value uniform across the workgroup held in vector registers, which costs a
     `readfirstlane` loop per use; load it into a scalar register;
   - an in-order memory-count wait that also drains a prefetch, because an independent load was
     issued after it; issue independent loads before the barrier.
   - Each fix can help one bucket and hurt another. Measure every bucket it reaches.

## §4 Choosing what to build next

1. **Separate structural levers from tuned constants.**
   - Two kinds of change:
     - a structural lever is a switch: a new stage order, a fused boundary, a queue;
     - a constant is a tuned number: tile size, wave count, prefetch depth.
   - Adopt or reject a lever with its constants retuned for the new structure. Untuned constants
     make a good lever look bad.
2. **Wire the baseline's tuning into the fused path before inventing new overlap.**
   - The unfused baseline usually ships per-shape tuned configurations for each stage.
   - If the fused path runs a stage with a default configuration, that stage is slower than the
     baseline's. No amount of overlap recovers that.
   - Check each stage's configuration first.
3. **Overlap is not free.**
   - Running stages concurrently helps only when the time hidden exceeds the machinery added:
     readiness polling, extra synchronization, lost occupancy.
   - Measure `wait` against `busy` with the stage meter (MEGA_MEASUREMENT_GUIDE §2) before and after.
4. **Build in stages, each proven on the card.**
   - For a multi-step rebuild, land and measure each stage.
   - Do not scaffold every stage and hand off an untested whole.
   - A rebuild may regress on the way to its goal. Keep its branch across rounds instead of
     reverting it at the first slower measurement, and state the stage it is at.
5. **Name the blocker class honestly.** For each failed round, record one class:
   - measurement (void or noisy evidence);
   - correctness;
   - liveness;
   - speed;
   - flow (budget or timeout).

   The next plan should target that class, not repeat the previous plan with a new name.
6. **In a persistent kernel, the claim order is a tuning surface.**
   - The persistent scheduler replaces the original grid's block-to-tile mapping. That mapping
     often gave operand reuse in cache; a plain linear claim order loses it, and the fused stage can
     end up no faster than the unfused one even with every wait removed.
   - Group claims so that the workers running at the same time share one operand slice, and sweep
     the group size on fresh inputs.
   - Keep the mapping a bijection over all units. If some worker can never claim some unit, a
     worker that waits on that unit never finishes.
   - measured: on one operator the claim order alone was worth more than all the fusion's removed
     waiting.
7. **Chunk size trades tail balance against reuse.**
   - Larger claims cut claim atomics and can extend reuse, but when units per worker are few they
     quantize the tail.
   - Chunk size and claim order interact: sweep them together, not one with the other fixed at a
     value tuned for a different order.
8. **A reuse-friendly order can collapse on skewed input.**
   - With one hot item, the same grouping can send all concurrent work to one link, one peer or
     one memory channel.
   - Measure every order on the skewed routes as well, and gate the order on a runtime skew measure
     so balanced inputs keep the tuned order.
9. **Credit efficiency changes to the right cause.**
   - An edit inside the fused candidate may also speed the unfused path (shared code), or could
     apply to it just as well (scheduling, codegen).
   - Measure the same tree with fusion off. A gain that shows in both arms is kernel efficiency,
     not fusion; fusion itself still only removes waiting.
   - Both kinds ship. Mixing them up sends the next plan to the wrong lever.

## §5 Adapting to the shape

The same operator is a different problem at different sizes. The guards of one run can span two or
three of them, so a single design rarely wins everywhere.

1. **Place every guard on the regime map before planning.**
   - From the shape alone you can compute, per guard: rows per expert (or per group), the bytes
     that must be read once (weights), the math, and the cross-rank bytes.
   - Each gives a lower bound on time. The largest is the *binding* resource. A measured baseline far
     above every bound means fixed costs (launches, syncs, handshakes, start latency) dominate.
   - `tools/mega_regime.py` does this from explicit inputs (MEGA_MEASUREMENT_GUIDE §1 step 0). The
     profile writes it to `EVAL_DIR/regime.json` and `PROFILE_SUMMARY.key_metrics.regime_by_guard`.
   - Guards with the same regime form a bucket. A bucket is a hypothesis; the paired A/B decides.
2. **Choose the lever family from the regime, not from the largest guard.**
   - Bound by a fixed read (few rows per expert): every expert's weights are read whatever the row
     count. Keep that stream at achieved bandwidth on every compute unit, and hide or remove
     everything else. Faster math changes little.
   - Bound by math (many rows per expert): tile shapes, occupancy, operand reuse, overlap of
     dependent stages.
   - Bound by links: overlap transfers with math, send fewer bytes, balance ranks.
   - Fixed-cost dominated: count launches, syncs and handshakes on the critical path of one call;
     shrink that count before tuning any throughput.
3. **Fixed costs do not shrink with the problem.**
   - A handshake, a barrier, a cache-maintenance operation or a launch costs about the same at 32
     rows as at 32 000. At small sizes they are a large share of the time; at large sizes they hide.
   - Measure them at the small guards. A design that is "overhead-free" at the largest guard can
     lose at the smallest one for this reason alone.
   - Synchronization scope is a cost too. A wider memory scope than the data needs (device-wide
     instead of the cache level the producer and consumer share, or invalidating caches that the
     data never went through) is pure fixed cost. Place producer and consumer so the narrowest
     correct scope suffices, and read the hardware's memory model for what each scope flushes.
   - Do not compute padding. A tile far larger than the real rows of an expert multiplies the work;
     check the tile-efficiency column of the regime map.
4. **One path per bucket, chosen by an explicit rule.**
   - It is normal for the fastest design at a small size to differ from the fastest at a large size
     (different tiles, different stage overlap, even a different launch structure).
   - Implement the bucket paths behind a dispatch rule that depends only on quantities known at
     call time (token count, capacity), and state the rule.
   - A bucket that cannot beat the baseline keeps the baseline path *honestly*: report it as a
     fallback with its measured ratio (§1.7). Never let a fallback bucket hide inside a geomean.
5. **Switching between buckets must not leak state.**
   - A serving process alternates sizes. State left by one path (counters, flags, queue heads,
     cached metadata, buffers sized for another capacity) must not corrupt the next call on another
     path, eager or graph-replayed.
   - Ranks can receive different token counts in the same call (ragged). The dispatch rule must give
     every rank a consistent decision, or the ranks must tolerate mixed paths.
   - The graph-contract runner's switch test interleaves every operator and size, forward and
     reverse, graph and eager, with ragged ranks when the adapter supports it. It must pass on the
     candidate.
6. **Per-bucket constraints, not one global one.**
   - The target guard has a speedup goal; every other guard has its own floor. Satisfy each.
   - A launch target is a ceiling: fewer launches is fine, more is not.
   - Report the per-guard ratio, the path that ran, and the launches per forward for every guard.
7. **Residency of waiting work.** A workgroup that waits on another (§2.2) must not occupy a slot
   the producer needs. Count residency per scheduling domain (per chiplet / per cache domain where
   the hardware has them), not only per device.
8. **Planner: say it per bucket.** In every plan, name the bucket of each guard (from the regime
   map, or "unmapped" if the profile did not produce one), the path each bucket will run (new, tuned,
   or the baseline as a declared fallback), the dispatch rule between them, and the lever family you
   chose for the target's bucket and why. A plan that changes only the largest guard's path must say
   what keeps the small guards at or above their floors.
9. **Targets in different regimes: one regime per run, then integrate.**
   - Do not build one shared structure for buckets in different regimes to reuse code. The
     structure that wins in one regime carries its fixed costs into the other. measured: a run that
     did this reached its large-bucket goal early, but held its small bucket within about 1% of the
     baseline for thirteen rounds and after fifteen was still below what a run targeting that
     bucket alone had reached with a different launch structure.
   - Prefer one run per regime: that regime's buckets are the targets, every other guard is a
     regression guard at its floor (an honest fallback is allowed). The plan, the round budget and
     the attribution then belong to one regime.
   - Finish with an integration step: put each regime's path behind the dispatch rule (§5.4) in
     one operator and verify every target, every floor and the switch test (§5.5) together. Shared
     code is where the regimes conflict; re-measure every bucket after any change to it.
   - When one run must cover several regimes, give each its own structure: the plan carries a
     target per bucket, and each direction names the bucket it changes.
