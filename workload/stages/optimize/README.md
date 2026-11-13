# Optimize

Produce an isolated candidate under the [invocation contract](../CONTRACT.md#direct-module-calls).
**Input:** declared change scope, qualified inputs, budget and validation contract.
**Output:** candidate identity, native evidence, costs and blockers.
**Acceptance:** the selected method preserves the module's verification and budget contract.
A local improvement does not establish serving gain.

## Implementations

| Skill | Method |
| --- | --- |
| [geak-kernel-workflow](geak-kernel-workflow/SKILL.md) | Existing Kernel Workflow runtime and roles; default kernel method |
| [kernel-dynamic-loop](kernel-dynamic-loop/SKILL.md) | Repeated reasoning over interchangeable kernel modules |
| [oob](oob/SKILL.md) | Native coding tools for a qualified kernel or declared system experiment |

Select by skill path in the plan or `stages.optimize.skill` in the sequential driver.
System experiments use the plan's native quality/effect checks and need no artificial UT.
Kernel methods share the contract below. Historical method labels remain in their receipts.

## Kernel calls

Use separate working copies of the plan's qualified tasks and a fresh scheduler work
root. The sequential driver supplies the full configured Profile delivery; a smaller
direct scope does not establish full-campaign completion. Roofline orders the tasks by
expected system savings; unknown bounds retain task-plan order and do not remove work.

Read existing qualification before dispatch. Compare `source_tree.py editable <task>`
with the declared computation; a binding adapter is not necessarily an editable kernel.
The frozen snapshot must cover the immutable inputs, including their declarations.
`check-frozen` compares only snapshot members, so it does not prove coverage.
Missing/conflicting qualification returns through Reasoning to UT.
Use the selected method's tools and preserve its complete verification/integration path;
an attempt cap is not permission to substitute a different method.

### Dispatch and recovery

Submit ready tasks together. `params.task_limit` is in-flight capacity, not an extraction
cap. Account for every task; queued tasks remain visible with zero attempts until started.
Only tasks not admitted to a partial dispatch belong in `capacity_deferred`, with their
rank and continuation reason. Use available slots for ready queued work.

Each task declares `meta.json.required_gpus`; `--gpus-per-task` is the fallback.
A collective needs its complete device set, and the selected method must preserve it.
Use disjoint physically isolated GPU slots; the local backend is for CPU/single-GPU smoke.
Do not edit launchers or shared inputs while active processes are reading them.

```bash
python3 "$WORKLOAD_ROOT/engine/campaign/scheduler.py" run \
    --work-root "$OUT/work" --tasks-dir "$TASKS_DIR" \
    --skill-root "$SKILL_ROOT" --runner "$RUNNER" \
    --gpus <list> --target <seconds> --backend docker --image <image>
```

Resolve flags from the plan/scenario, including order file, task widths, attempt/measurement
limits, KEEP policy, mounts and retention. Inspect `scheduler.py --help` for the interface.
The scheduler handles queueing, worker liveness, stale reclamation and durable credit;
Reasoning owns the next inspection and decisions about reported defects.
Attempt bounds must allow observed method-completion and full-test costs.

For a deficient task, preserve its failure, source and credit history and report promptly.
Hold that branch, requalify the repaired delivery through UT, then use a separate work root
for the new measurement basis. Healthy branches continue on disjoint resources.
A verified immutable candidate can proceed to integration/evaluation while other work
continues if their measurement resources do not overlap.

Containers mount configured paths at the same absolute locations. `--mount` overrides
filesystem-derived mounts. Render devices come from ROCm topology or `--render-devices`
in host GPU order; verify that mapping before measurement.
`--retain-containers` preserves stopped containers and requires
evidence preservation and explicit cleanup before reuse; `--evidence-root` selects storage
for new archives. Record node-local storage and its retention limits.

### Cold verification

For each selected retained source, run the frozen task **twice**, each in a fresh process:

```bash
python3 "$WORKLOAD_ROOT/engine/campaign/run_task_entry.py" "$TASKS_DIR/<task>"
```

Place selected sources in their final snapshot directory before testing; the live search
tree may hold an unaccepted attempt. Check source/harness identity before and after each
run. Test independent tasks concurrently on disjoint slots; a task's repeats are sequential.

Retain each receipt, stdout, exit status and source hashes before another run overwrites
them. Use fresh runtime/compile caches, including native task-local build paths; preserve
weights, immutable inputs and previous measurement data. Retain generated oracle versions
and state any archive coverage limits. A retained symlink records its literal target,
not the external contents. Evidence helpers are described in [engine](../../engine/README.md).

The task's native gates own correctness and timing. Use its serving-weighted aggregate;
when reported as a causal/allocation interval, record the low end.
The minimum speedup across the cold repeats is the verified value.
Reserve cold-test and E2E time from measured full-test costs and actual parallel slots;
a coding-attempt cap does not bound the subsequent full measurement.

### Delivery

Write `optimized.json` with one `kernels` row per admitted task, including no-gain,
failed, blocked and still-queued tasks. Record attempted/kept/deferred counts separately.

```json
{
  "tasks_total": 1,
  "kept": [{"kernel": "<task>", "speedup": 1.1}],
  "kernels": [{
    "kernel": "<task>", "verdict": "pass", "verified_speedup": 1.1,
    "claimed_speedup": 1.12, "attempts": 3, "credited_seconds": 100,
    "kernel_src": "sources/<task>/kernel_src",
    "task_meta_sha256": "<sha256>", "kernel_src_hashes": {"compute.py": "<sha256>"},
    "notes": []
  }],
  "capacity_deferred": []
}
```

`kernel_src` is the complete directory used by both cold runs, relative to this report
or absolute. `best_kernel_src` remains a legacy alias.
Record the complete file/hash map with `source_tree.py hashes`, the frozen metadata hash
and any frozen task-local `candidate_bind.file` hash as `binding_file_sha256`.
Recover older identities only from retained cold evidence, not present-day working files.
Apply back consumes these exact bytes.

| Verdict | Meaning |
| --- | --- |
| `pass` | Native gates pass and the cold minimum clears the configured band |
| `noise` | Native gates pass without clearing the band |
| `incorrect` | The test compared outputs and found a correctness failure |
| `harness_error` | No valid judgement: execution, timing or other non-correctness failure |

A serving integration defect returns through Reasoning with the failing input/dispatch,
native error and selected source identity. Source fixes require fresh cold qualification;
a deficient frozen task belongs to UT. Preserve unaffected tasks' evidence.

[finalize.py](finalize.py) owns task accounting and budget admission; it does not rejudge
the native UT. For a scoped call, use `--tasks-dir <scoped-tasks> --optimized <report>`
with the invocation's `PARAMS` and `OUT`. All supplied tasks must appear.
Report only reverified numbers externally. Distinguish elapsed wall time (overlap counted
once), summed credited task time and reserved GPU-hours. Include failed/superseded work
in cost accounting, separate search/cold/E2E, and state unfinished budgets.
The legacy `gpu_hours` summary field holds credited task-hours; label it accordingly
and report physical GPU reservation separately.

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
Native-task implementations delegate to the single reader in
`stages/optimize/measure.sh`.
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

## Budget and resume

For a required full search, pass `--require-full-budget`; it preserves each task's
target across allocation windows. Resume the same work root for compatible inputs.
A changed harness needs its own work root and qualification, with earlier costs retained.

`checkpoint_reserve_sec` is a planning allowance, not an automatic shutdown mechanism.
`--max-wall-sec` stops the scheduler loop and can leave workers running.
Reasoning must arrange a checkpoint before the allocation deadline: stop new dispatch,
allow active measurements to finish, retain evidence and stop owned workers.
Reserve final-attempt, cold-test and E2E costs as well as cleanup; revise bounds from
observed durations. Incomplete search is a continuation, never a completed time budget.
