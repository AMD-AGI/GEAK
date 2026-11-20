# UT

Define and qualify the computation boundary for an optimization task under the
[invocation contract](../CONTRACT.md#direct-module-calls).

**Input:** one `ut_request.json`: per task its kernels, entry callable, contract family,
scaffold arguments and concrete cases with their weights, plus any blocked identities.
Its caller writes it from a profile's `kernels.json` under the
[request contract](kernel-ut/references/request-authoring.md);
this module reads nothing else upstream
and does not know how many configs a case's weight pools. The call also passes its GPU
allocation and the deployment's `serving.json` (environment and source path the UT must
reproduce) explicitly; see [kernel-ut](kernel-ut/SKILL.md).
**Output:** independently runnable frozen tasks, editable-source declarations, native
qualification/delivery receipts and the covered or deferred identities. Report separately
the computation edited, computation timed, served regimes covered, and differences from
the selected opportunity.
**Acceptance:** the native task and delivery gates pass for the requested cases,
the actual device width and the declared edit boundary. CPU preflight alone is incomplete.

Implementation: [kernel-ut](kernel-ut/SKILL.md). `scripts/scaffold_tasks.py` installs and
scaffolds the requested tasks; CPU author lanes write them in parallel and borrow GPUs
per validation command (`ut_wave.py gpu -- ...`); every task qualifies with
`unittest.py --validation-mode correctness --reuse`. Receipts record
`performance_measured: false`; the optimizer runs the same driver in performance mode.

A boundary, harness or case change produces a new request and a new qualified delivery.
Preserve the old version and report which dependent tasks need replacement; unrelated
tasks stay valid. A requested boundary that the live seam check refutes is reported
`blocked` with its verdict, not redrawn here.

| Artifact | Purpose |
| --- | --- |
| `task_plan.json` | Installed tasks and blocked identities, from the request |
| `selection_targets.json` | Only when the call passes the profile's selection (`scaffold_tasks.py --selection`), for Roofline |
| `tasks/<name>/` | Qualified tasks with `unittest.py`, `meta.json`, `kernel_src/` and `cases.py` |
| `tasks/<name>/run/ut_result.json` | Native full-test verdict |
| `tasks/<name>/run/RETURN.json` | Native delivery-gate receipt |
| `results.json`, `REPORT.md` | Identity dispositions, per-task case/gate results, limitations |

[finalize.py](finalize.py) checks that every requested identity is owned by an existing
task directory or carries a stated blocker, and that each task has its UT verdict and
delivery receipt. Whole-serving coverage is not inferred from a UT pass.

Scenario route:

```json
{
  "order": ["baseline", "profile", "ut", "roofline", "optimize", "applyback"],
  "task_source_stage": "ut",
  "stages": {
    "profile": {"skill": "kernel-profile", "mode": "agent", "params": {"selection_top_k": 10}},
    "ut": {"skill": "kernel-ut", "mode": "agent", "params": {"request": "<path>/ut_request.json"}}
  }
}
```
