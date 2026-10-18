# Native primary scores in the common GEAK lane

GLM's primary score is a median of complete-process phase folds. It cannot be
reconstructed by folding per-case median times, or by combining decode and
prefill geometric means. The common caller now preserves `external_primary` for
both authentic GEAK (`optimizer: "greedy"`) and GEAK-EVO. GEAK keeps its existing
planner, engineer/verify/integrate stages, thresholds, budget and stopping rules.

The caller supplies these fields in addition to its normal admitted task inputs:

```json
{
  "objective": {"kind": "external_primary", "primary_regime": "prefill_extend"},
  "external_measurement_command": ["/installed-owner/existing-evaluate-command"],
  "external_result_command": ["/installed-owner/read-control-result"],
  "optimizer_task_ref": {"path": "/admitted/task.json", "sha256": "..."}
}
```

These paths are examples of owner-supplied commands, not newly installed programs.
The measurement command retains the existing Arena evaluation handshake and
returns its real `candidate_id` and `evaluation_id`. Each complete measurement is
the unchanged UT's three fresh processes, with native `min_runs=2`, graph/eager
settings and all current checks preserved. The lane tells roles to use this
combined operation in place of their generic repeated correctness/benchmark
instructions. No extra numerical or phase coverage is introduced.

The result command is a small installed binding to
`geak_evo.control_result.main(argv, evaluator=<existing ArenaEvaluator>)`.
It starts no native work. The lane passes `--candidate-id`, `--evaluation-id`,
`--workspace`, and an optional `--patch`. The reader rechecks the existing settled
collection through `read_evaluation`, including `read_external_report` over raw
UT artifacts. It verifies that the recorded source matches the relevant workspace
or the supplied patch applied to that workspace. It does not trust a model's
reported scalar or `OBSERVATIONS.json` alone. No new store, identity, admission,
device, clock or cleanup system is introduced.

For baseline setup, engineers, independent verification, integration and final
director validation, return the existing IDs as:

```json
{"external_evaluation": {"candidate_id": "...", "evaluation_id": "..."}}
```

An integrator places this under `best`. A warm-start check uses the same fixed
remapped-or-original patch selection as adoption. The host reader supplies the primary score, both
phase folds, descriptive case pairs, native gain/regression verdicts and retained
receipt reference. Missing or invalid evidence has no scalar fallback. A valid
ORIGINAL or no-gain evaluation remains a measured observation; adoption and
export require the unchanged native gain gate and the existing GEAK improvement
threshold. The final score is independently reread, and an absent final evaluation
does not fall back to the planner's cumulative claim. Final no-gain observations
retain their measured score with `outcome: "no_improvement"` and no promoted patch.

The common GEAK-EVO dispatch result carries the same explicit native objective
and gain verdict. Its caller no longer demands that descriptive case medians
reproduce the native fold. Default geomean/time-weighted callers retain their
existing selection and result fields.

This is prospective source support. The sole operator must bind these command
entrypoints to the already admitted owner, expose IDs from the real evaluation
command, and qualify the complete ORIGINAL path before launching a comparison.
The GLM task uses the pinned `ac30…` image and private writable AITER/Triton/Torch
caches; the DeepSeek image and previous measurements do not qualify that path.
Preserve exact task hashes, source versions, model profile and exposure in both
arms. No source test here constitutes installed owner support or a GPU result.
