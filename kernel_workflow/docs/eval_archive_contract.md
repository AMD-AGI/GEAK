# Eval archive contract (issue #429 — corrected root cause)

Outer harnesses (AgentKernelArena / Hyperloom goal loops) MUST NOT full-tree
`cp -a` / `tar` an `EVAL_DIR` into `wave*_archive_*` without exclusions.

## Required excludes when archiving

- `**/*.so`, `**/*.o`
- `**/aiter/jit/**`, `**/aiter/aiter/jit/**`
- `**/.torch_ext/**`, `**/build/**`, `**/__pycache__/**`
- Prefer archiving only: `final_patch.diff`, metrics JSON, `STATE/`,
  `best_patch.diff` per direction, and `storage_telemetry.jsonl`.

## Retention

- At most one archive generation retained live; delete or lighten the previous
  `wave*_archive_*` when a new wave starts.
- Disk pressure must trigger reclaim (see `scripts/reclaim_eval_artifacts.sh`),
  never abort the optimize loop.

## GEAK-side helpers

```bash
bash "$WORKFLOW_DIR/scripts/materialize_workspace.sh" --src "$CANONICAL" --dst "$OUT/workspace" \
  --shared-root "$EVAL_DIR/_shared" --link-aiter
bash "$WORKFLOW_DIR/scripts/reclaim_eval_artifacts.sh" --eval-dir "$EVAL_DIR"
# Under pressure (still continue optimizing):
bash "$WORKFLOW_DIR/scripts/reclaim_eval_artifacts.sh" --eval-dir "$EVAL_DIR" --force-heavy
```

## Candidate source and reported evidence

Save exact measured source with `scripts/save_candidate_patch.sh`, using the supplied workspace
base commit (EVO's `workspace_base_commit`, not the selected parent's HEAD) and declared editable
paths. Include new backend/build source dependencies; exclude measurement files, logs and binaries.
Keep the best measured bytes before further edits; never overwrite their patch with a different
final workspace. Source retention does not establish correctness or authorize promotion.

Optional `evidence_context` uses `tested_applicability` and `next_separating_observation` strings,
plus `observations`, `unresolved`, `rival_explanations` and `evidence_refs` arrays of strings.
Keep useful existing notes when it is absent. `reported_context` and contributor records quote
author interpretations tied to their own sources; their hashes establish custody, not truth.
Check applicability to a child and verify every combined source independently. Local donor paths
are unavailable unless shared; use bounded-context overflow references where relevant. These
records guide the next experiment and never change the official evaluator or candidate score.

### Unverified source recovery

When `UNVERIFIED_SOURCE_RECOVERY:true` is supplied, this rule replaces save-only-on-improvement
instructions for an implementation whose evaluation has not completed. Before entering a potentially
blocking GPU evaluation, save the concrete changed source with `save_candidate_patch.sh`. Use the
initial private-workspace commit from `OUTPUT_DIR/workspace_base_commit.txt`, never a later HEAD.
If no correct measured development beyond the starting program is saved, use `best_patch.diff` so
the ordinary verifier can qualify it even after an interrupted engineer call. Never overwrite a
saved correct measured development with an unmeasured iteration; retain that additional attempt
under a distinct patch/report filename instead. Do not change the canonical incumbent.

If returning an unmeasured attempt, leave the workspace at its saved bytes and return the same
`worker_result.json` through StructuredOutput: `status:partial`, `measurement_valid:false`, empty
`per_case`, no inherited evaluation IDs or scores, and required numeric speedups as zero placeholders.
State explicitly that timing is unavailable, not measured zero. Preserve any completed correctness
result with its source and receipt; otherwise correctness is unknown. Never infer a pass from missing timing.
Record the immutable base, source/patch paths, available hashes, evaluation failure/timeout evidence
and next required qualification in `evidence_context.evidence_refs` and `notes`. Keep a known
correctness failure distinct from an evaluation that did not execute. Available evaluation remains
required; source recovery does not justify stopping early or consuming final-validation reserves.
The host's existing source pin and independent verifier decide custody and qualification. An
unmeasured source has no fitness and cannot replace the incumbent or enter delivery as a gain.

### Measurement conditions and repeated observations

In the report and returned `evidence_context`, label each timing observation's
`execution_mode` (for example, the actual native compiled path or eager execution),
`buffer_policy` (reuse, rotation and reset), `timing_boundary` (what the clock includes,
units and repeats), and source/evaluation receipt bindings. Use strings or nested objects;
extra top-level result fields need not survive the archive projection. Associate each
`per_case` row with its observation's context or receipt reference, including the baseline
and candidate bindings. Preserve the producer's exact labels; unspecified conditions are
`UNKNOWN`, not inferred from source equality, correctness or a kernel name.

### Native aggregate and case measurements

When COMMANDMENT declares a native primary, keep that scalar, its case rows and
its repeat reduction associated with the same measurement cohort. Preserve the
native ORIGINAL and candidate times, ratios, counts and weights. A median of
aggregate scores need not equal an aggregate recomputed from casewise medians;
record the declared reduction instead of silently forcing equality.

Return compact provenance in `evidence_context.case_measurement`: `basis`
(`native`, `frozen_reference` or `unknown`), `cohort_ref`, `reduction`, and
`weight_source`. These are author-reported descriptions, not host verification.
Put full source/result receipts in `evidence_context.evidence_refs`. Keep long
explanations in the referenced report. Do not author `observed_evidence` or
`reported_per_case`; those fields belong to the external host result reader.

Preserve native geometric/arithmetic metrics when supplied; otherwise derive them
only from the same native cohort under its declared reduction. In weighted mode,
an unavailable secondary metric is `null` with an explanation, never a measured
zero or a copy of the weighted primary. An unweighted primary still requires its
actual numeric measurement. If native rows are unavailable, return `per_case:[]`;
that omission alone neither invalidates nor qualifies the native primary. Required
correctness, source, timing and acceptance rules still apply.

Keep frozen-reference rows under
`evidence_context.frozen_reference_per_case`, with their definition and receipts.
Only calculate them when case identity, units, baseline times, candidate reduction
and weight provenance are known. They describe a different denominator and must
not replace native rows in stored candidate evidence. Unavailable inputs remain
unknown. A diagnostic table disagreeing with the primary does not by itself prove
the native measurement failed.
Copy each native row's `optimized_ms` verbatim into its frozen-reference row; do
not round it independently or substitute another repeat. Both tables must refer
to the same candidate timing reduction, even though their ORIGINAL times differ.

An archive entry holds the latest verifier verdict for a candidate; recording that candidate
again replaces its verifier evidence. Before remeasurement, preserve the complete earlier
result, case rows and conditions under a distinct report/JSON filename in the existing role
output directory. Save the new observation separately and carry both retained paths and
available hashes in `evidence_context.evidence_refs` on the next return. Do not overwrite the
only copy or assume the archive supplies a multi-protocol history. References to unavailable
artifacts remain explicitly unavailable.

Keep native-compiled/native-buffer and eager/rotated-buffer observations separately labelled,
even for identical source. Apply the declared repeat reduction only within matching conditions;
do not pool incompatible observations or select a faster protocol as the official score.
Historical observations and author-supplied conditions do not replace the required independent
verification, establish deployment engagement or change promotion.

## Reusing prepared ORIGINAL evidence

For supplied `PREPARED_ORIGINAL_REF={path,sha256}`, verify exact bytes, successful producer role
returns and supporting raw evidence before reuse. Match ORIGINAL source, oracle, cases/weights,
loaded state, runtime, measurement definitions and physical device/group; timing reuse also needs
the declared freshness/conditions. A standalone timing/setup file is not a completed role return.
Copy applicable artifacts into the private workspace/eval directory and rebind execution paths,
GPU locks and source bindings; preserve raw receipts and never execute the producer's workspace
or share mutable build caches. Record reused versus locally satisfied obligations and missing
evidence in existing notes. Reuse matching analysis/profile instead of repeating it; incompatible
receipts remain historical context and required local measurements still run. Return the normal
role schema when its obligations are complete.
