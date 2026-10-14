# The `optimize` slot

**What it does**: takes each frozen-harness task extracted by the profile stage
(`tasks/<kernel>/`), optimizes them in parallel at one kernel per GPU for a default of 48
hours each, then **re-verifies from cold** and hands on only what survives.

**Input**: `$RUN_ROOT/profile/out/tasks/` (task directories) and
`$RUN_ROOT/roofline/out/roofline.json` (optimization order by expected e2e gain; if
absent, directory order is used — ranking degrades, correctness does not).

**Output**: `$OUT/optimized.json`, `$OUT/verified.json`, `$OUT/result.json`.

## Implementations

| Skill | Notes |
| --- | --- |
| `geak-kernel-workflow/` | **default**. Drives GEAK's `kernel_workflow/kernel_workflow.js`. |
| `subagent-raw/` | the "bare agent" option required by `instruct.md`. No dependency on the GEAK runtime, and simultaneously the control group for the GEAK path. |

To switch: change the directory name under `stages.optimize.skill`, or pass
`--set stages.optimize.skill=subagent-raw`.

## Separation of concerns (the part of this slot worth understanding)

```
         ┌───────────── slot (unchanged when the method changes) ─────────────────┐
         │  orchestrate.sh   schedule -> re-verify -> summarize                   │
         │  finalize.py      artifacts -> result.json; trusts only re-verification│
         │  ../../scripts/scheduler.py   one container per GPU, dedup, stale      │
         │                               reclamation, bad-node blacklisting       │
         │  ../../scripts/worker.sh      attempt loop, credit/heartbeat,          │
         │                               KEEP/revert                              │
         │  ../../scripts/verify.py      cold re-verification gate                │
         └──────────────────────────────────┬────────────────────────────────────┘
                                            │ env
                                            ▼
              ┌──────── implementation skill (only "how to optimize") ────────┐
              │  prompt.sh    required: build this attempt's prompt           │
              │  measure.sh   optional: custom measurement; default runs      │
              │               unittest.py                                     │
              │  run.sh       three lines, delegating to ../orchestrate.sh    │
              └───────────────────────────────────────────────────────────────┘
```

**Why the line is drawn here**: "one unit test per GPU / 48 hours / fault tolerance /
nothing counts until re-verified" is what `instruct.md` demands of this **stage**, and it
holds for any optimization method; whether the work is fed to `kernel_workflow.js` or
performed by a bare agent is the method. With the line drawn here, adding a method is a
directory containing a `prompt.sh` with no change to scheduling or accounting — and
because both methods run on the same harness, the same noise band and the same
verification gate, their results are directly comparable.

### The `prompt.sh` interface

The worker passes context through environment variables and expects the prompt written to
`$PROMPT_OUT`:

| Variable | Meaning |
| --- | --- |
| `K` | kernel name |
| `TASK_DIR` | task directory (frozen harness) |
| `KERNEL_SRC` | the filename of the one editable file under `kernel_src/` |
| `ATTEMPT` | attempt number |
| `BEST_SPEEDUP` | this kernel's best result so far |
| `KEEP_THRESHOLD` | the noise band |
| `RUN_DIR` / `EVAL_DIR` | the kernel's run root / this attempt's own directory |
| `ATTEMPT_CAP` | this attempt's second limit (enforced externally) |
| `PROMPT_OUT` | where to write the prompt |

`measure.sh` is optional and must print a line containing `STATUS=<0|1> SPEEDUP=<x>`.
Without it the default applies: run the task's entry through
`scripts/run_task_entry.py $TASK_DIR` and read whichever contract it prints —
`CORRECTNESS: PASS` plus `GEAK_GEOMEAN_SPEEDUP <x>`, or a JSON line
`{"exit": 0, "weighted": <x>}` (the form the extraction skill emits, where a zero exit
means every gate passed and `weighted` is the serving-weighted speedup over the sweep).

The entry is **not** run as `cd $TASK_DIR && python3 unittest.py`: the task directory
contains its own `unittest.py`, which lands on `sys.path[0]` and shadows the standard
library package. Nothing fails at import; it fails later inside torch with
`No module named 'unittest.mock'`, the measurement comes back empty, and the attempt is
reverted for a reason that has nothing to do with the kernel.

## Artifacts → `result.json`

`finalize.py`'s mapping, shared by both implementations:

| result.json | Source |
| --- | --- |
| `artifacts.optimized` | `optimized.json` (per kernel: verdict, verified speedup, attempts, credited seconds) |
| `artifacts.verified` | `verified.json` (`scripts/verify.py`'s cold re-verification) |
| `metrics.kept` | number of kernels that **passed re-verification** |
| `metrics.best_speedup` | the highest among those |
| `metrics.gpu_hours` | total credited seconds |
| `status` | all tasks blocked → `blocked`; no worker state → `fail`; otherwise → `ok` (including when nothing won) |

Keeping zero kernels is **not** a failure: the stage ran, the numbers are real, and the
conclusion happens to be "no win". Marking that `fail` sends someone to fix something
that is not broken.

## Conventions that must not be changed

* **Report only re-verified numbers.** `claimed_speedup` (self-reported during
  optimization) is for contrast only and must never be quoted externally. A previous
  campaign produced a kernel that passed hot and failed cold on correctness — a unit-test
  leg allocates from a virgin allocator, so zeroed pages let "skip part of the output"
  pass while warm.
* **This stage claims no end-to-end gain.** An operator win is not a system win: MoE
  sorting at 1.80× in microbenchmark corresponds to +0.2356% whole-machine, and MoE
  split4 at 1.342× to **−0.0872%**. Only applyback may speak about e2e.
* **One kernel is never dispatched to two GPUs.** Both sides would edit the same
  `kernel_src/`, their KEEP/revert decisions would trample each other, and both sets of
  numbers would be void. The scheduler dedups through a busy set, with the node as the
  source of truth.
* **Multi-GPU parallelism requires the docker backend** (one container per GPU, mounting
  a single `/dev/dri/renderD<128+8i>`). The local backend can only pin through
  environment variables, and agents reset `ROCR_VISIBLE_DEVICES` inside their own
  subprocesses, putting two workers on the same physical GPU. Use local only for
  single-GPU smoke tests.

## Resuming

Credit and heartbeat both live on shared storage (`$OUT/work/state/`). After a node
failure, a manual Ctrl-C or a killed scheduler, re-issue the same command: credited
seconds are not reset, and changes already committed into `kernel_src/` keep
accumulating. This is also why attempts are capped at 1500 s — measured, a 2 h cap gave
69 attempts / 11 keeps (16%), while a 1500 s cap gave 85 attempts / 22 keeps (26%) in
under half the wall-clock.
