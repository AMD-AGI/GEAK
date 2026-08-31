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
