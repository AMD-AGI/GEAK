---
name: kernel-ut
description: Generate source-backed standalone GPU unit tests from a verified kernel-profile handoff, using the shared fixed family contracts, templates and driver. Author and validate tasks in parallel without serving, profiling or timing; performance measurements belong to optimization.
---

# Template-based kernel UT generation

Read the profile stage's `profile_handoff.json`. Own task boundary review, source
extraction and template-based UT generation/qualification. Treat profile evidence as
read-only. Do not launch serving, profile kernels, rediscover shapes or invent missing
workload weights. Missing evidence becomes a task blocker or a request back to profile.

The pipeline supplies `$IN`, `$OUT`, `$PARAMS`, `$CONFIG`, `$RUNNER` and
`$WORKLOAD_ROOT`. This skill's `scripts/`, `shared/`, `ut_contracts/` and `references/`
reuse the existing kernel-extract tool bundle via relative symlinks: there is one
maintained template/driver implementation, not a fork. Read
[UT contracts](ut_contracts/README.md), [source extraction](references/source-extraction.md),
[task authoring](references/task-authoring.md) and
[source delivery records](references/source-delivery-records.md).
The scope in this skill overrides the combined workflow instructions in shared references.

## Verify and prepare

```bash
python3 "$WORKLOAD_ROOT/stages/ut/prepare.py"
# standalone: add --handoff <profile_handoff.json> --out <ut-output-dir>
```

Preparation verifies the handoff and all structured evidence hashes, copies small
top-level descriptors and references the shared evidence. It never copies old tasks or
changes upstream files. Output is separate from the profile tree. `profile_input.json`
binds this UT run to one handoff; a changed handoff needs a fresh UT output directory.

## Declare and scaffold

```bash
python3 "$SKILL_ROOT/scripts/draft_task_decls.py" --eval-dir "$OUT" \
  --out "$OUT/task_decls.json"
```

Review the draft using only handed-off evidence and installed source in the pinned
environment. A template does not decide the actual callable boundary:

- One owner per complete operator; keep mandatory companion chains together and
  mutually exclusive dispatch branches in separate `joint_phases` rows. Batch-size
  branches can belong to one UT; do not split the task merely because entries differ.
  Record source/ownership blockers.
- Select the fixed UT family and bind axes/layout/dtype/context to observed evidence.
  Preserve config IDs, effective call counts and normalization. Use invocation anchors
  when several leaf launches implement one call. Clear each `needs_review` with evidence.
- Resolve the real callable, objects/cache state and native build dependencies.
  Deliver computation source in `kernel_src/`, not an importing dispatcher alone.
  A custom family uses the same contract schema with an explicit reason.

For a multi-config group, apply [the UT merge policy](references/multi-config-ut-merge.md)
before final scaffolding. Build the provisional plan it consumes, preserve dispatcher,
rebuilt-instance or scope-equalized branch metadata, and rebuild after applying its
decisions. Use only the handed-off evidence and matching pinned source/baseline.

```bash
python3 "$SKILL_ROOT/scripts/build_task_plan.py" --eval-dir "$OUT" --check
python3 "$SKILL_ROOT/scripts/build_task_plan.py" --eval-dir "$OUT"
```

This writes `task_plan.json` and the family scaffolds: `cases.py`, `meta.json`,
`unittest.py`, `run/ut_cases.json`, `run/ut_contract.json` and task evidence. Fill only the
template hooks and case metadata. Generate ordinary values from fixed seeds; retain
valid structural/routing/page semantics. Do not save large ordinary input/output `.pt`
fixtures. Use the shared template's bounded, private weight cache and the driver's
batch output handling; do not replace them with suite-wide golden-output files.
Preserve all cases and draws. Read [temporary tensor lifecycle](ut_contracts/README.md#temporary-tensor-lifecycle)
for the frozen generation recipe and budgets. Report `ut_report.json.data_handling`;
one oversized case and inline recorded tensors can still be large.

## Parallel authoring and validation

Hold the allocated GPU pool through the existing campaign lease wrapper and use one
agent lane per task. A task directory has one writer:

```bash
python3 "$SKILL_ROOT/scripts/ut_wave.py" plan --eval-dir "$OUT" --gpus <allocated-ids>
python3 "$SKILL_ROOT/scripts/ut_wave.py" start --eval-dir "$OUT" --runner "$RUNNER" \
  --backend docker --image <pinned-image> --gpus <allocated-ids> \
  --timing none
```

Lanes follow [the UT lane procedure](references/ut-lane.md), author and fix in parallel,
and run CPU lint, quick iterations, then `unittest.py --validation-mode correctness --reuse`.
This runs every non-timing gate and emits a passing UT qualification without starting
any timing workers. Reuse unchanged passing gates; preserve the cases, draws and
per-config weights for later optimization. Wait/status the wave and retain each failure.

Every accepted task needs full correctness, binding/launch, server-call/variant,
isolation/replay, per-leaf negative controls and restored-source evidence as required
by its family. Timing is not required for UT generation. The driver records
`validation_contract: "kernel_ut_validation/v1"`, `validation_mode: "correctness"`,
all eleven `required_gates`, and `performance_measured: false` in both UT receipts;
there is no timing gate or fabricated 1.0 ratio. Keep the weighted timing interface
executable for the optimizer. Use the real source-delivery verifier to create
`run/RETURN.json`. A quick pass, missing GPU access or a numerical control that only
breaks compilation is not a full UT pass.

## Deliver

Write `REPORT.md` and `results.json` with selected identity ownership, blockers,
per-task case/gate results and coverage limitations. Preserve each reviewed task's
`selection_pct_gpu_time` from `task_plan.json` as `results.tasks[].selection_pct_gpu_time`
(with its `task` label); the optimizer uses this measured share for its default order.
Merged tasks retain branch cases, boundary cases, expected kernels, branch tolerances
and per-branch weights. Report performance as unmeasured; optimization later supplies
per-branch timings. Then:

```bash
python3 "$WORKLOAD_ROOT/stages/ut/prepare.py" --check
python3 "$WORKLOAD_ROOT/stages/ut/finalize.py"
```

The finalizer rechecks immutable inputs and uses the existing task/UT/source-delivery
acceptance checks. It does not require inventing an exact whole-serving coverage
percentage to qualify a valid UT; unknown serving coverage remains unknown. The
profile stage reports measured sampling evidence, while this stage reports tested
case/source coverage. Downstream optimization receives this stage's output containing
`tasks/`, `task_plan.json`, selection and ranking, and performs its own timing. An optional
roofline stage must collect any missing performance inputs. No optimization or E2E gain
is claimed.
