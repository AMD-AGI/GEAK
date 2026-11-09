# The `optimize` slot

**What it does**: takes each frozen-harness task extracted by the profile stage
(`tasks/<kernel>/`), optimizes them one kernel per GPU slot until the time budget is
spent, then runs each task's own unit test once more from cold and hands on only what
clears its noise band.

**Input**: `$RUN_ROOT/profile/out/tasks/` and `$RUN_ROOT/roofline/out/roofline.json`
(order by expected e2e gain; absent, directory order — ranking degrades, correctness
does not).

The slot resolves tasks from profile, independently of `$IN` (roofline's output in the
normal order). For a standalone smoke run, set `TASKS_DIR` to the fixture task directory;
`finalize.py --tasks-dir` remains available for inspecting a delivery directly.

**Output**: `$OUT/optimized.json`, `$OUT/result.json`.

## Implementations

| Skill | Notes |
| --- | --- |
| `geak-kernel-workflow/` | **default**. Drives GEAK's `kernel_workflow/kernel_workflow.js`. Requires a runner that exposes the Workflow tool. |
| `native-frozen/` | the same worker and frozen harness, optimized with the runner's own coding tools. Use it when the runner is a plain coding CLI, and for collectives: `kernel_workflow.js` takes a pool lock on one GPU. |
| `subagent-raw/` | the "bare agent" option required by `instruct.md`; also the control group for the GEAK path. |

Switch with `stages.optimize.skill`, or `--set stages.optimize.skill=subagent-raw`.

They differ only in `prompt.sh`. Dispatch, timing, credit, cold verification and the
verdict are the slot's and are identical for all three.

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

**Account for every extracted task.** The profile task plan is the input set; roofline
orders that set and never removes a task because its predicted gain is small or unknown.
`params.task_limit` (default 8) is **this stage's in-flight capacity** — how many
independent tasks the campaign can hold at once, one per GPU — and not a cap on what
profile may extract. Extraction stops where the evidence stops: the same skill run by
hand delivered ten complete tasks, and a limit applied upstream would have made that
delivery a defect. Submit the complete task set to the scheduler in roofline order; it
fills released slots from that queue. Resume the same work root so completed tasks retain
their credit and later tasks enter the slots they free.

Queue admission is not an optimization attempt. Record actual attempts and remaining
credit for every task, including tasks that have not started when a deadline forces a
checkpoint. A task already admitted to the scheduler queue is not also
`capacity_deferred`. If a partial dispatch has not admitted a task, preserve its name,
roofline rank and continuation reason in `capacity_deferred`; it is unattempted and
prevents a claim that the full delivery is complete. Do not use `max_tasks` or
`exclude_tasks` to silently optimize only a convenient subset, and do not leave runnable
queued tasks waiting while the scheduler has an unused slot.

Before dispatch, ask the slot's verdict what it will require:

```
python3 $WORKLOAD_ROOT/stages/optimize/finalize.py \
    --tasks-dir $RUN_ROOT/profile/out/tasks
```

An extracted task that cannot run still needs an explicit entry and blocker in the final
report. Repair a missing precondition when possible and resume that task; it must not
vanish from the input set or be counted as attempted merely because another task won.

**Parallel is the default.** Submit all ready tasks together, one task per physically
isolated GPU slot, retaining roofline order when there are fewer slots than tasks. Do not
serialize independent kernels through a shell loop or put two optimization workers on the
same GPU to simulate parallelism. Record the assigned GPUs and start/end times so progress
reports describe actual overlap.

**Width is the task's own declaration.** Each task's `meta.json` says how many GPUs it
needs in `required_gpus`, and the scheduler gives it that many consecutive devices.
Single-rank kernels and a TP8/EP8 collective therefore run in one campaign with no
grouping step: narrow tasks keep the machine busy while a wide one waits for its ranks to
come free. A task that declares nothing gets `--gpus-per-task`. There is no campaign-wide
GPU width and no second scheduler — there used to be both, and partitioning tasks into
same-width batches was the only thing that made them necessary.

Before spending the search budget, read the profile qualification receipts: the frozen
harness must represent serving inputs and be able to distinguish a useful change from
measurement noise. In particular, routed workloads need observed local expert occupancy,
and short collectives need device timing that is not dominated by host launch overhead.
Use the already-qualified evidence; repeat qualification only when it is missing or a
specific defect is found. An unchanged-source control establishes noise, not a kernel win.
Check that the selected method's tools are available through `$RUNNER`; `native-frozen`
and `subagent-raw` remain available when the GEAK Workflow tool is absent.

If a task's harness needs repair, preserve its old receipts and source snapshots, send
the defect to the profile slot, and requalify that task before restarting its search.
Healthy tasks on disjoint GPUs can continue. A changed harness starts a separate worker
work root so an old frozen snapshot or KEEP cannot silently be adopted into the new
measurement basis. Retain old credit in cost accounting; compare performance only within
the matching harness version.

Before the first worker starts, compare the frozen snapshot with the task's declared
immutable inputs. `check-frozen` checks bytes only for existing snapshot members;
it does not prove coverage. If the default freeze omits inputs, complete the snapshot
in a new work root, including the manifest that declares them, while preserving the
declared edit boundary. Do not rewrite an existing work root's frozen snapshot.

```
python3 $WORKLOAD_ROOT/engine/campaign/scheduler.py run \
    --work-root $OUT/work --tasks-dir $RUN_ROOT/profile/out/tasks \
    --skill-root $SKILL_ROOT --runner $RUNNER \
    --gpus <list> [--gpus-per-task <n>] --target <sec> --backend docker --image <img> \
    [--order-file $RUN_ROOT/roofline/out/roofline.json] [--exclude a,b] \
    [--keep-threshold 1.045] [--keep-threshold-override task=1.35]
```

Every flag has a counterpart in `$PARAMS`; read them from there rather than inventing
values. The scheduler owns one container per slot, dedup, credit, heartbeat, stale
reclamation and bad-GPU blacklisting — see its module docstring. `scheduler.py
--work-root <dir> status` prints per-task credit and attempts.

A worker container sees the task tree, the skill and the run root at the same absolute
paths the scheduler was given, so those filesystems are bind-mounted into it. Which ones
is **derived**, not named: the mount point of each configured path. It used to be a
literal `-v /shared_nfs:/shared_nfs` — one site's NFS export written into the tool,
correct here and wrong on the next machine, and invisible until a container started and
found an empty directory. `--mount <host path>` (repeatable) replaces the derived set for
anything outside the configured tree; `/` is never mounted wholesale. The scheduler logs
what it mounted at startup.

Two optional preservation parameters are worth knowing about. `retain_containers: true`
(`--retain-containers`, docker backend only) stops owned workers but keeps them, and an
existing retained name blocks another dispatch until its evidence has been preserved and
the name explicitly retired; the default is ordinary cleanup. `evidence_root`
(`--evidence-root`, absolute) selects storage for new measurement and recovery archives,
and that storage must be mounted at the same path in every worker; existing archives,
pending evidence and task ownership are untouched. Omit it to keep the default
placement. When the storage is node-local, record it and its retention limits in the
scenario config.

### Step 2 — run each task's unit test again, from cold

For every task with a `state/<kernel>.status.json`, run it **twice** in a fresh process:

```
python3 $WORKLOAD_ROOT/engine/campaign/run_task_entry.py $TASKS_DIR/<kernel>
```

Use the retained source snapshot for that task, with the same frozen harness that scored
it. Place that selected computation in its final snapshot directory before cold testing;
the live search working tree may still contain an unaccepted attempt. Check the source
and harness before and after each invocation. Run independent tasks
concurrently on disjoint GPU slots using the same physical isolation as search. Keep a
task's two repeats sequential: its `run/` directory is shared by those invocations.
Archive the complete receipt, stdout, exit status and source hashes into a separate
`$OUT` directory after each repeat, before the next invocation overwrites `run/`. Each
repeat starts a fresh process and fresh task runtime/compile caches; do not delete model
weights or prior measurement artifacts. Check the task's own build paths: native builders
may ignore standard cache environment variables. Archive each mutable task-local cache
before the next repeat while preserving declared immutable inputs, including inputs
stored under `run/`. Moving the entire directory can remove a required anchor or capture.
Preserve generated reference outputs too; a final working-file copy does not preserve
versions overwritten within one UT. Record the actual archive coverage and retain any
missing-evidence limitation rather than inferring prior byte identity from equal seeds.
A whole-node collective waits for other GPU work to finish, and E2E waits for all cold tests.

Derive the reserved wall time from measured full-test durations and the planned parallel
slots. A coding-attempt cap does not bound the worker's subsequent full measurement.
Preserve the entire cold-test and E2E reservation until those phases actually execute;
finishing a search does not spend its verification allowance.

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

### Step 3 — write `optimized.json`

One `kernels` row per task admitted to the scheduler, including no-gain, failed and
blocked tasks. A queued task that has not started records `attempts: 0` and its
continuation in `notes`. Only tasks not admitted in a partial dispatch appear in
`capacity_deferred`, with their rank and reason; neither form counts as attempted.
Keep this shape — applyback reads it:

```json
{"tasks_total": 5, "gpu_hours": 10.7, "verdict_counts": {"pass": 1, "noise": 3},
 "kept": [{"kernel": "opus_mla_v4_sparse_prefill", "speedup": 1.1096}],
 "kernels": [{"kernel": "...", "verdict": "pass", "verified_speedup": 1.1096,
              "claimed_speedup": 1.1116, "attempts": 7, "credited_seconds": 7333,
              "kernel_src": "sources/<kernel>/kernel_src",
              "task_meta_sha256": "<recorded meta.json SHA-256>",
              "kernel_src_hashes": {"compute.py": "<recorded SHA-256>"}, "notes": []}],
 "capacity_deferred": [{"kernel": "quant_mx_g32_e8m0", "roofline_rank": 9,
                        "reason": "not admitted before the checkpoint; resume in the next allocation"}]}
```

For every passing row, `kernel_src` names the complete source directory measured by
both cold runs. Relative paths resolve against `optimized.json`; `best_kernel_src`
remains a supported legacy name for that directory. `kernel_src_hashes` is the complete
relative-file/SHA-256 map, checked before and after both runs. Obtain it using
`source_tree.py hashes <selected-directory>` with the same file policy as snapshots.
Record `task_meta_sha256` from the same frozen `meta.json`, including its callable
binding, so later metadata edits cannot redirect a verified source to another target.
If `candidate_bind.file` is a frozen task-local adapter outside `kernel_src`, also record
its `binding_file_sha256` from those checks. Applyback consumes these exact bytes and
rejects missing or changed identities before building an overlay. Recover an older
report's identity from its retained cold evidence; hashing present-day WIP does not
retroactively verify it.

A serving integration failure goes back to the owning stage with its failing input or
dispatch, native error and selected source identity. The optimization agent owns kernel
implementation fixes; a deficient frozen test or source boundary belongs to profile.
After a source fix, repeat cold qualification for that changed source and publish a new
snapshot. Prior scores remain attached to their prior source. Compatible extraction and
unchanged kernels' evidence can be reused; do not restart the entire search or rewrite
frozen tests merely to get past applyback.

**You do not write `result.json`.** `stages/optimize/finalize.py` does, after the session,
and it is the stage's conclusion (`stages/CONTRACT.md` 3.1). What it accepts is written
only there — read the file, or run it early with the command in step 1, rather than
looking for a second copy of its rules in this document. It checks names, counts and
budget; it never re-judges a unit test.

Report extracted, attempted and kept counts separately, and explain every unattempted
task and its continuation. Distinguish device kernel identities from task boundaries: two
attention dispatch branches or a real fused chain can belong to one optimization task,
with their mapping recorded in `task_plan.json`.

Report three distinct time quantities: elapsed wall time (with concurrent intervals
counted once), summed worker credited seconds, and GPU-slot hours (worker duration times
the number of GPUs reserved). The historical `gpu_hours` field above is credited
task-hours; label that basis explicitly and report device reservation separately for
multi-GPU tasks. Separate search, cold testing and E2E, include unsuccessful attempts in
costs, and state whether the requested budget was completed. A deadline-bounded checkpoint
must not be described as a completed 48-hour search.

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
  `python3 $WORKLOAD_ROOT/engine/campaign/source_tree.py check-frozen <task> <snapshot>` does the
  first. For the second, `source_tree.py editable <task>` prints the declaration and
  `source_tree.py check <before> <task>/kernel_src --allowed "<those names>"` exits 2
  naming every file it will not accept.

## The `prompt.sh` interface

An implementation supplies the optimization method and nothing else. `engine/campaign/worker.sh`
passes context through the environment and expects the prompt at `$PROMPT_OUT`:

| Variable | Meaning |
| --- | --- |
| `K` | kernel name |
| `TASK_DIR` | task directory (frozen harness) |
| `KERNEL_SRC` | the editable file under `kernel_src/` |
| `KERNEL_SRC_ALL` | every editable file, when the task declares more than one |
| `GPU_IDS` | the logical GPUs assigned to this task, as wide as its `required_gpus` |
| `ATTEMPT` | attempt number |
| `BEST_SPEEDUP` | this kernel's best result so far |
| `KEEP_THRESHOLD` | the noise band |
| `RUN_DIR` / `EVAL_DIR` | the kernel's run root / this attempt's own directory |
| `ATTEMPT_CAP` | this attempt's second limit (enforced externally) |
| `PROMPT_OUT` | where to write the prompt |

`measure.sh` is required. The worker passes `TASK_DIR`, `RUN_DIR`, `MEASURE_TIMEOUT`,
and `MEASURE_NATIVE_RC_FILE`; the hook prints
`STATUS=<0|1> SPEEDUP=<x> CLEAN=<x-or-empty> SINGLE=<count>`.
The three native-task implementations (`geak-kernel-workflow`, `native-frozen`,
`subagent-raw`) delegate to the single reader in `stages/optimize/measure.sh`.
That reader runs the frozen task entry and interprets its output; the worker only
invokes the hook, retains evidence, and consumes those fields. The slot's final
acceptance criterion remains in `stages/optimize/finalize.py`.

The hook's exit code describes hook execution. A completed native UT with a failing
verdict is reported through `STATUS=0`; a missing hook or nonzero hook exit stops the
worker. After a hook invocation, its logs are retained before stopping. A hook may
write the native process exit code to `MEASURE_NATIVE_RC_FILE` so retention can record
it separately from the hook exit code. This optional file is transport metadata;
it neither adds a verdict nor changes the four stdout fields.

Hook or retention failures leave the task's existing status `blocked`, which tells the
scheduler to stop redispatching it. After repairing the cause, retain the failure logs
and set that status back to `pending` to resume; keep its credit, history and evidence.

`KEEP_THRESHOLD` is the noise gate against the original baseline, not a requirement to
multiply the current best by that threshold on every iteration. Give the session the
configured incremental adoption rule as well; the outer worker owns the KEEP decision.
Confusing these two tests can discard useful directions before they are measured.

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
* **A collective gets its whole device set.** `kernel_workflow.js` holds a pool lock that
  selects one GPU; run it on an eight-rank task and seven ranks are missing while the
  numbers still look like measurements. That is what `native-frozen/` is for.
* **Multi-GPU parallelism requires the docker backend.** The local backend pins only
  through the environment, and an agent resets `ROCR_VISIBLE_DEVICES` in its own
  subprocesses, putting two workers on one physical GPU. Use local only for single-GPU
  smoke tests.

## Resuming

When the user requires a full compute budget, set `require_full_budget=true` and pass
`--require-full-budget` to the scheduler, which preserves the per-task target across
allocation windows. A short window contributes real worker time, then checkpoints with
`checkpoint_reserve_sec` left for owned-worker cleanup. It reports remaining credit as
`blocked`; cold/E2E wait until all targets are complete and enough time is available for
validation. No allocation is changed or implicitly renewed.
Workers finish one last bounded attempt and its scoring
sweep even if that slightly exceeds the target; an early `.done` marker or time spent
idle is not completion. The outer allocation deadline still applies, so interruption
retains credit and requires continuation rather than a full-budget claim. Reserve time
for this final attempt as well as cold verification and E2E.

When validation must fit in the same allocation, set `checkpoint_reserve_sec` to at least
the complete cold-test and E2E reservation derived above. Otherwise checkpoint mode may
retain only its cleanup allowance when the conservative search estimate does not fit.
Keep the full per-task target; a protected validation window does not make unfinished
search complete.

Credit and heartbeat live on shared storage (`$OUT/work/state/`). After a node failure, a
Ctrl-C or a killed scheduler, re-issue the same command: credited seconds are not reset
and edits already committed into `kernel_src/` keep accumulating. This is also why
attempts are capped at 1500 s — measured, a 2 h cap gave 69 attempts / 11 keeps (16%),
a 1500 s cap gave 85 attempts / 22 keeps (26%) in under half the wall-clock.
