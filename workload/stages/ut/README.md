# UT

Define and qualify the computation boundary for an optimization task under the
[invocation contract](../CONTRACT.md#direct-module-calls).

**Input:** selected workload evidence, source/callable identities, planned scope and runtime.
**Output:** independently runnable frozen tasks, editable-source declarations, native
qualification/delivery receipts and the covered or deferred identities.
**Acceptance:** the native task and delivery gates pass for representative inputs,
the actual device width and the declared edit boundary. CPU preflight alone is incomplete.

Implementation: [kernel-extract](kernel-extract/SKILL.md), reusing the vendored extractor.
A boundary, harness or case change produces a new qualified delivery. Preserve the old
version and report which dependent tasks need replacement; unrelated tasks stay valid.

## `kernel-ut`: UTs from a profile handoff (split route)

[kernel-ut](kernel-ut/SKILL.md) is the implementation for the split `profile` → `ut` route
(multi-config scenarios); [kernel-extract](kernel-extract/SKILL.md) above serves direct
module calls and Brain-planned repairs. Both qualify tasks with the same native gates.

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

The UT wave performs correctness qualification; tasks finish with
`unittest.py --validation-mode correctness --reuse`. This is a complete correctness
qualification, not a partial performance run. Receipts carry the explicit
`kernel_ut_validation/v1` contract and `performance_measured: false`. Keep all weighted
shapes and the timing implementation for optimization, which runs
`unittest.py --validation-mode performance --reuse` and must obtain real timings.
The unqualified `unittest.py` command retains performance mode for existing optimizer
callers. Plain `--skip timing` in performance mode remains a partial result.

GPU scheduling uses command-scoped leases: up to `--author-workers` CPU-only authors
(default 8), one writer per task, submit `ut_wave.py gpu -- <command>` for validation.
The host broker leases all required cards together and drains workers before returning
them. See [setup](kernel-ut/references/setup.md) for locks, mounts and environment.
Existing frozen tasks keep their own helpers; new tasks use the fixed contract driver.
The retired fixed-lane and timing-tail wave modes are no longer accepted.

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

Run only profiling with `run.py run profile`, and generate/retry UTs later with
`run.py run ut`. Resume revalidates stage input receipts. A UT-only retry uses the
same handoff and does not invoke baseline/profile again.
