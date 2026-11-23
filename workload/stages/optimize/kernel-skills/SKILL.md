---
name: kernel-skills
description: Delegate a qualified kernel to a supervisor that assembles each round's team from v4 role skills and delivers a candidate through workload's frozen native UT.
---

# Kernel skills

Select `stages.optimize.skill=kernel-skills` or name this skill in the plan. Follow
the [Optimize kernel calls](../README.md#kernel-calls) for task admission, dispatch,
cold verification and delivery. Pass this directory to the scheduler as `--skill-root`.
The runner needs file, shell and subagent tools for the supervisor and its delegates.

For a one-kernel delegation, Brain supplies the task working copy, objective, qualified
baseline/UT, permitted edits, device set, budget and report destination in the call's
`PLAN.md`. Forward its path and evidence locations through `EXTRA_STEER`; use the
configured worker hooks for each attempt. In a full campaign the scheduler dispatches
each admitted kernel separately. Each session runs the [kernel supervisor](../../../../kernel_skills/v4/kernel-supervisor/SKILL.md).
The supervisor selects the v4 skills, team and handoffs from evidence each round.
The search strategy defaults to `evo`; the plan may select `hillclimb` or `avo`.

## Worker contract

Read the prior handoff and worker status. Give every role the following workload
contract along with its [v4 instructions](../../../../kernel_skills/v4/COMMON.md).

1. Set `INPUT` to the worker's current task, `RUN` to this attempt's search directory
   and `APPLY=0`. On first entry materialize `RUN/orig` with
   `bash "$KW/scripts/materialize_workspace.sh" --src "$INPUT" --dst "$RUN/orig"`.
   Compare the source identity, frozen files and required fixtures before starting
   the search or resuming a saved archive. Candidate patches are relative to this
   attempt's starting source, which may already contain a previous KEEP. Record the
   native score of that starting source in `c000` against the UT's original baseline.
   Use the editable leaves declared by the worker.
2. Use the worker's complete assigned GPU set for each native UT, including all ranks
   for collectives. The worker owns the slot. Serialize GPU commands from all roles
   with `flock "$RUN/measurement.lock" <command...>` and inherit the worker's device
   visibility. This replaces standalone GPU selection and cross-GPU pairing in the
   role instructions. Run processes within the worker's timeout group and join them
   before handoff. Count alternative optimization directions against `BUDGET` across
   all roles. Reserve time within `ATTEMPT_CAP` for the ship audit and delivery.
3. Qualify and score each candidate with the full frozen native UT, including its
   gates, inputs, case weights and candidate binding. Use the same measurement hook
   the worker uses. For a candidate workspace `ws` and a fresh receipt directory
   `receipt` outside it, run

   ```bash
   mkdir -p "$receipt"
   TASK_DIR="$ws" RUN_DIR="$receipt" MEASURE_TIMEOUT="$MEASURE_TIMEOUT" \
     MEASURE_NATIVE_RC_FILE="$receipt/native.rc" \
     flock "$RUN/measurement.lock" bash "$METHOD/measure.sh" > "$receipt/fields"
   ```

   Check hook exit, native exit and `STATUS` before recording `SPEEDUP`. Retain
   `.measure.log` and generated task receipts before the next measurement. Record
   available per-case evidence for the archive. For an aggregate-only UT, use its
   single measured aggregate as one search observation with unit weight and label
   it as such. Fast probes and archive composites can guide search. Use the native
   score for promotion and preserve `MIN_IMPROVE` for advancing the inner incumbent.
   For a task whose native binding requires its canonical path, follow the existing
   [frozen UT procedure](../kernel-dynamic-loop/stages/verify/frozen-ut/SKILL.md)
   and hold the slot lock across installation, measurement and restoration.
4. After the independent reviewer accepts the candidate with the v4 ship audit, run
   `python3 "$SOURCE_TREE" check "$RUN/orig/kernel_src" "$ws/kernel_src" --allowed "$KERNEL_SRC_ALL"`
   on the exact verified workspace. Exit 1 reports permitted changes. Deliver those
   with `python3 "$SOURCE_TREE" copy "$ws/kernel_src" "$INPUT/kernel_src"`.
   If the audit fails or time runs out before verification, retain the starting
   source. Record the verdict, archive, decision and receipt paths,
   costs and unfinished work in the worker's `HANDOFF.md`. The worker remeasures the
   delivery and owns KEEP/revert.

## Complete the call

After search, Optimize runs its [cold verification](../README.md#cold-verification)
and delivers `optimized.json` and `REPORT.md` for the declared scope. Return the
accepted source identity and cold receipts, or the measured no-gain, failure or
unfinished work with its evidence and cost. Brain uses that result to decide on
Apply back and serving evaluation.
