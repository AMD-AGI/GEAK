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
