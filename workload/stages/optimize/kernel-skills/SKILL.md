---
name: kernel-skills
description: Optimize a qualified kernel with GEAK's v4 role skills, candidate archive and search strategies using the workload's frozen native UT.
---

# Kernel skills

Select `stages.optimize.skill=kernel-skills` or name this skill in the plan. Follow
the [Optimize kernel calls](../README.md#kernel-calls) for task admission, dispatch,
cold verification and delivery. Pass this directory to the scheduler as `--skill-root`.
The runner needs file, shell and subagent tools for the lead and independent roles.

Each worker attempt invokes the [v4 lead](../../../../kernel_skills/v4/geak-lead/SKILL.md)
from the [kernel skills package](../../../../kernel_skills/README.md) imported from
`feat/kernel-roles-skills` at `b7135788`. The lead selects parents from measured
candidates, records decisions and dispatches the benchmark, roofline, investigator,
engineer and ship roles. It uses `evo` by default. A plan can name `hillclimb` or
`avo` in the attempt guidance (`EXTRA_STEER`). Record the strategy in the decisions.
The package also contains the earlier role sets and the standalone EAT instructions.
This workload entry invokes v4 directly.

## Running one attempt

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
4. Follow the [search shell](../../../../kernel_skills/v4/search/SHELL.md) through the
   independent ship audit. After an accepted final verdict, run
   `python3 "$SOURCE_TREE" check "$RUN/orig/kernel_src" "$ws/kernel_src" --allowed "$KERNEL_SRC_ALL"`
   on the exact verified workspace. Exit 1 reports permitted changes. Deliver those
   with `python3 "$SOURCE_TREE" copy "$ws/kernel_src" "$INPUT/kernel_src"`.
   If the audit fails or time runs out before verification, retain the starting
   source. Record the verdict, archive, decision and receipt paths,
   costs and unfinished work in the worker's `HANDOFF.md`. The worker remeasures the
   delivery and owns KEEP/revert. Optimize owns the final cold repeats.
