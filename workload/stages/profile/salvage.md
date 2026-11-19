Profile salvage: deliver only the UT tasks that already qualify.

`finalize.py` judges this as a **partial** delivery (README.md, "When the budget runs
out"). Work through these steps in order, using only what is already in `$OUT`:

1. **Census, merge and selection over what was captured.** If `kernel_ranking.json` and
   `selection_targets.json` are complete, keep them. Otherwise finish them over the
   configs whose captures are already on disk (multi-config: `multi_config.py merge` /
   `select` with only those `--config` entries) at the configured scope. Do not capture a
   missing config. Name every config that was not captured in `REPORT.md` and in
   `results.json` under `salvage.uncaptured_configs`; finalize derives the same list from
   `$IN/baseline.json` and reports it.
2. **Keep the tasks that are already green.** A task qualifies only when its directory
   is complete (`unittest.py`, `meta.json`, `cases.py`, `kernel_src/`), its
   `run/ut_result.json` reports `ok: true` with no failed gates, and its
   `run/RETURN.json` delivery receipt names it. Do not modify a qualified task: changing
   it after its GPU run invalidates that evidence.
3. **Move every other task directory out of `tasks/`**, into
   `$OUT/tasks_unqualified/<name>/` (move, never delete). Optimize and applyback treat
   every directory under `tasks/` as delivered, so finalize refuses a salvage while an
   unqualified one remains there. Do not write new UTs, and do not try to fix red ones.
4. **Record every selected identity that no qualified task owns as blocked** in
   `results.json` `identities[]`: no `task`, `status: "blocked"`, and `why` set to
   `profile budget exhausted (salvage)` (append the specific cause when known, e.g.
   which gate was red or that its config was never captured). Remove moved tasks from `results.json` `tasks[]`. Keep the
   identities that were already blocked for another reason as they are. An identity
   missing from both lists is the one thing finalize will not accept.
5. **Restate coverage for the delivered tasks only**:
   `coverage.delivered_declared_pct_gpu_time` and
   `coverage.delivered_achieved_pct_gpu_time` (or the per-identity fields) over the
   tasks still under `tasks/`.
6. Label `REPORT.md` as a partial delivery (profile budget salvage), listing delivered
   tasks, blocked identities and uncaptured configs, then run
   `python3 "$WORKLOAD_ROOT/stages/profile/finalize.py"` and fix what it names.
