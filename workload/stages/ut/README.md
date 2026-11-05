# Stage `ut` — generate and qualify standalone kernel tests

**Input** `$IN/profile_handoff.json` from the profile stage.
**Output** `$OUT/tasks/`, `task_plan.json`, per-task GPU/source receipts and `result.json`.

The `kernel-ut` skill consumes dispatch/shape/source evidence and uses the existing
fixed family contracts and driver. It does not start serving or collect profiles.
Source-boundary review and operator-specific hooks remain part of generation; a
template alone cannot infer a valid cache/routing object or a native build boundary.

`prepare.py` verifies the [handoff](../profile/HANDOFF.md) and prepares read-only input
references in a separate workspace. `kernel-ut/scripts`, `shared`, `ut_contracts` and
`references` link to the single existing tool bundle; frozen task helpers are still
copied and hashed by the normal scaffold.

`finalize.py` checks the handoff and input hashes, then delegates to the existing
extraction finalizer's task ownership, all non-timing GPU UT and source-delivery checks. It uses
the profile's frozen selection policy. Exact whole-serving coverage is reported
separately and may remain unknown; missing task gates or source-delivery receipts
still block UT completion. Do not turn a quick or CPU-only test into full qualification.

The default wave uses `--timing none`; tasks finish with
`unittest.py --validation-mode correctness --reuse`. This is a complete correctness
qualification, not a partial performance run. Receipts carry the explicit
`kernel_ut_validation/v1` contract and `performance_measured: false`. Keep all weighted
shapes and the timing implementation for optimization, which runs
`unittest.py --validation-mode performance --reuse` and must obtain real timings.
The unqualified `unittest.py` command retains performance mode for existing optimizer
callers. Plain `--skip timing` in performance mode remains a partial result.

For split scenarios:

```json
{
  "order": ["baseline", "profile", "ut", "optimize", "applyback"],
  "task_source_stage": "ut",
  "stages": {
    "profile": {
      "skill": "kernel-profile",
      "mode": "agent",
      "params": {"output_kind": "dispatch_evidence"}
    },
    "ut": {"skill": "kernel-ut", "mode": "agent", "params": {"timing": "none"}}
  }
}
```

The driver exports `TASKS_DIR` from `task_source_stage`; consumers use the declared
producer instead of assuming tasks live under profile. An optional roofline stage can
read UT output but must collect missing performance inputs itself. Legacy five-stage
scenarios keep `profile: kernel-extract` and their previous combined contract.

Run only profiling with `stage.py run profile`, and generate/retry UTs later with
`stage.py run ut`. Resume revalidates stage input receipts. A UT-only retry uses the
same handoff and does not invoke baseline/profile again.
