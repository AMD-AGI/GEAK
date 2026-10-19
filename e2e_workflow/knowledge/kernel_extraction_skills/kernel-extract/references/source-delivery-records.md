# Source delivery records

For `KERNEL.source_delivery_required=true`, return a task-local
`source_delivery_ref={path,sha256}` pointing to the following manifest:

```json
{
  "contract": "kernel_source_delivery/v1",
  "target_callable": "package:complete_op",
  "snapshot_sha256": "...",
  "files": [{"path": "cases.py", "sha256": "..."}],
  "ut_result_ref": {"path": "run/ut_result.json", "sha256": "..."},
  "kernels": [{
    "device_kernel": "exact_selected_symbol",
    "source_files": ["kernel_src/implementation.py"],
    "execution_ref": {"path": "run/leaf_execution.json", "sha256": "..."},
    "negative_control_ref": {"path": "run/leaf_negative.json", "sha256": "..."}
  }]
}
```

Compute the full `files` and `snapshot_sha256` with
`node "$SKILL_ROOT/scripts/kernel_source_delivery.mjs" snapshot "$TASK"`.
It covers `meta.json`, `cases.py`, `unittest.py` and every source file under
`kernel_src/`. Include other task-local harness/build/input adapters through
`meta.kernel_extraction.source_delivery_files`. Freeze these files before GPU
validation. Compute file references with the same script's `ref TASK FILE`.
Do not edit frozen files after validation; a changed file invalidates the records.

The full GPU UT JSON must report `ok:true` (or `smoke:"pass"`), `exit_code:0`,
the actual `target_callable` and the snapshot digest used by the test. It must
result from the standard UT and complete the gates in the main contract.

Each per-kernel execution/negative-control record has `kind` equal to
`candidate_execution`/`numerical_rejection`, `ok:true`, `device_kernel`,
`target_callable`, `snapshot_sha256`, the executed `command`, the actual
locked `gpu` set and a hashed task-local `log_ref`. Execution records include
positive `invocations` at the actual editable leaf. Negative controls include
`launch_succeeded:true` and positive `numerical_mismatches`; build/import errors
do not count. Record the clean snapshot the reversible mutation tested; retain
the actual mutation and mutated-source identity in the log, restore it, and
run the normal final UT. For a joint task, record each requested leaf separately.

Generate records from actual test outcomes. Hashing a file or copying these
field names does not prove GPU execution. The checker enforces consistency,
not independent truth of the recorded tests; preserve raw logs for review.
It rejects missing members, changed sources/entry, stale UT results and
negative controls that only fail compilation. The standalone CLI requires `source_delivery_required=true`; it does not accept
a source-free declaration as an extraction delivery.

<!-- Standalone adaptation, 2026-10-12: local tools and agent-owned stages. -->
