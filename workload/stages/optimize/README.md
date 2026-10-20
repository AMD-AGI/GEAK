# The `optimize` slot

**What it does**: takes each frozen-harness task extracted by the profile stage
(`tasks/<kernel>/`), optimizes them one kernel per GPU slot until the time budget is
spent, then runs each task's own unit test once more from cold and hands on only what
clears its noise band.

**Input**: `$RUN_ROOT/profile/out/tasks/` and `$RUN_ROOT/roofline/out/roofline.json`
(order by expected e2e gain; absent, directory order — ranking degrades, correctness
does not).

**Output**: `$OUT/optimized.json`, `$OUT/result.json`.

## Implementations

| Skill | Notes |
| --- | --- |
| `geak-kernel-workflow/` | **default**. Drives GEAK's `kernel_workflow/kernel_workflow.js`. |
| `subagent-raw/` | the "bare agent" option required by `instruct.md`; also the control group for the GEAK path. |
| `geak-kernel-gpu-groups/` | groups tasks by how many GPUs their harness needs and runs each group in turn. |

Switch with `stages.optimize.skill`, or `--set stages.optimize.skill=subagent-raw`.

## This is an agent-shape stage (contract §2B)

There is no `run.sh`. An agent reads this document and performs the three steps below.
That is deliberate: the two steps that are mechanism — dispatching unattended work for
hours across containers, and running a Python entry whose directory shadows the standard
library — are code, and everything else is judgement. The judgement used to be code too:
a `verify.py` that re-implemented the verdict, parsed three output dialects, chose which
source snapshot to run, and decided correctness for itself. It disagreed with the test it
was wrapping and won the disagreement, which is how a task whose harness had **crashed
before comparing any output** came to be recorded as a correctness failure. Prose does not
accrete that way, and an agent reading it can tell a crash from a wrong answer.

### Step 1 — dispatch

```
python3 $WORKLOAD_ROOT/scripts/scheduler.py run \
    --work-root $OUT/work --tasks-dir $RUN_ROOT/profile/out/tasks \
    --skill-root $SKILL_ROOT --runner $RUNNER \
    --gpus <list> --gpus-per-task <n> --target <sec> --backend docker --image <img> \
    [--order-file $RUN_ROOT/roofline/out/roofline.json] [--exclude a,b] \
    [--keep-threshold 1.045] [--keep-threshold-override task=1.35]
```

Every flag has a counterpart in `$PARAMS`; read them from there rather than inventing
values. The scheduler owns one container per slot, dedup, credit, heartbeat, stale
reclamation and bad-node blacklisting — see its module docstring. `scheduler.py
--work-root <dir> status` prints per-task credit and attempts.

### Step 2 — run each task's unit test again, from cold

For every task with a `state/<kernel>.status.json`, run it **twice** in a fresh process:

```
python3 $WORKLOAD_ROOT/scripts/run_task_entry.py $TASKS_DIR/<kernel>
```

Never `cd <task> && python3 unittest.py`: the task directory lands on `sys.path[0]` and
shadows the stdlib `unittest` package, and the failure surfaces far away as
`No module named 'unittest.mock'` with an empty measurement. `run_task_entry.py` resolves
the stdlib first and then prints a one-line summary of the task's own verdict:

```json
{"ut_result": "<task>/run/ut_result.json", "ok": true, "gates": {"correctness": true, ...}}
```

The receipt it points at holds the gates and the per-case timings. Its serving-weighted
aggregate is `gates.timing.aggregate.primary`; where the harness reports the
`serving_weighted_speedup_px_causal` / `px_alloc` pair instead, that is an interval and
**the low end is the figure of record**.

Take the **minimum** speedup across the two runs. A mean is pulled up by one lucky run,
and only what survives every cold start counts.

### Step 3 — write `optimized.json` and `result.json`

One row per task, then `result.json` per the stage contract. Keep this shape — applyback
reads it:

```json
{"tasks_total": 4, "gpu_hours": 10.7, "verdict_counts": {"pass": 1, "noise": 3},
 "kept": [{"kernel": "opus_mla_v4_sparse_prefill", "speedup": 1.1096}],
 "kernels": [{"kernel": "...", "verdict": "pass", "verified_speedup": 1.1096,
              "claimed_speedup": 1.1116, "attempts": 7, "credited_seconds": 7333,
              "best_kernel_src": "work/runs/<kernel>/best_kernel_src", "notes": []}]}
```

`result.json`: `metrics.kept` counts `verdict == "pass"`, `metrics.best_speedup` is the
highest among them, `metrics.gpu_hours` sums credited seconds. `status` is `blocked` when
every task was blocked (runner exit 78, an unmet precondition), `fail` when the scheduler
never dispatched anything, otherwise `ok`. **Keeping zero kernels is not a failure** — the
stage ran, the numbers are real, and the conclusion is "no win". Calling that `fail` sends
someone to fix something that is not broken.

## The verdict vocabulary

Four words, and the distinction between the last two is the one that matters:

| verdict | means |
| --- | --- |
| `pass` | the test ran, every gate held, and the cold minimum clears the band |
| `noise` | the test ran, every gate held, and it does not clear the band |
| `incorrect` | **the test ran and compared outputs, and they differed** |
| `harness_error` | the test could not deliver a judgement |

A nonzero exit, a timeout, no verdict on stdout, or any failing gate other than
`correctness` is `harness_error` — never `incorrect`. It says nothing about the kernel's
arithmetic; it says the extracted test, or what the optimizer did to the source that test
anchors on, needs fixing. Record it as such and send it back to the owning module: a
negative control anchored on literal source text stops matching the moment an attempt
renames the symbol it quotes, and the harness then refuses to patch blind — correctly.
That is a profile-stage defect, and reporting it as a wrong kernel buries it.

Before trusting any number, two things are worth checking and neither needs code:

* the worker's frozen snapshot, `work/runs/<kernel>/frozen_harness`, still matches the
  task's harness files — a number measured by an edited harness measures the edit;
* the only sources that changed under `kernel_src/` are the ones `meta.json` declares
  editable (`kernel_extraction.editable_leaves` / `candidate_bind.editable_leaves`).
  `python3 $WORKLOAD_ROOT/scripts/source_tree.py check-frozen <task> <snapshot>` does the
  first. For the second, `source_tree.py editable <task>` prints the declaration and
  `source_tree.py check <before> <task>/kernel_src --allowed "<those names>"` exits 2
  naming every file it will not accept.

## The `prompt.sh` interface

An implementation supplies the optimization method and nothing else. `scripts/worker.sh`
passes context through the environment and expects the prompt at `$PROMPT_OUT`:

| Variable | Meaning |
| --- | --- |
| `K` | kernel name |
| `TASK_DIR` | task directory (frozen harness) |
| `KERNEL_SRC` | the editable file under `kernel_src/` |
| `ATTEMPT` | attempt number |
| `BEST_SPEEDUP` | this kernel's best result so far |
| `KEEP_THRESHOLD` | the noise band |
| `RUN_DIR` / `EVAL_DIR` | the kernel's run root / this attempt's own directory |
| `ATTEMPT_CAP` | this attempt's second limit (enforced externally) |
| `PROMPT_OUT` | where to write the prompt |

`measure.sh` is optional and must print `STATUS=<0|1> SPEEDUP=<x>`; without it the worker
runs the task entry as in step 2.

## Conventions that must not be changed

* **Report only re-verified numbers.** `claimed_speedup` is for contrast and must never be
  quoted externally. A previous campaign produced a kernel that passed hot and failed cold
  on correctness — a unit-test leg allocates from a virgin allocator, so zeroed pages let
  "skip part of the output" pass while warm.
* **This stage claims no end-to-end gain.** MoE sorting at 1.80× on the microbenchmark was
  +0.2356% whole-machine; split4 at 1.342× was **−0.0872%**. Only applyback may speak
  about e2e.
* **One kernel is never dispatched to two slots.** Both would edit the same `kernel_src/`
  and trample each other's KEEP/revert. The scheduler dedups with the node as the source
  of truth.
* **Multi-GPU parallelism requires the docker backend.** The local backend pins only
  through the environment, and an agent resets `ROCR_VISIBLE_DEVICES` in its own
  subprocesses, putting two workers on one physical GPU. Use local only for single-GPU
  smoke tests.

## Resuming

Credit and heartbeat live on shared storage (`$OUT/work/state/`). After a node failure, a
Ctrl-C or a killed scheduler, re-issue the same command: credited seconds are not reset
and edits already committed into `kernel_src/` keep accumulating. This is also why
attempts are capped at 1500 s — measured, a 2 h cap gave 69 attempts / 11 keeps (16%),
a 1500 s cap gave 85 attempts / 22 keeps (26%) in under half the wall-clock.
