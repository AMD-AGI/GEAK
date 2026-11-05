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
`node "$SHARED_TOOLS/kernel_source_delivery.mjs" snapshot "$TASK"`.
It covers `meta.json`, `cases.py`, `unittest.py` and every source file under
`kernel_src/`. Include other task-local harness/build/input adapters through
`meta.kernel_extraction.source_delivery_files`. Freeze these files before GPU
validation. Compute file references with the same script's `ref TASK FILE`.
Do not edit frozen files after validation; a changed file invalidates the records.

The full GPU UT JSON must report `ok:true` (or `smoke:"pass"`), `exit_code:0`,
the actual `target_callable` and the snapshot digest used by the test. It must
result from the standard UT and complete the gates in the main contract.

### Existing `ut_driver.py` receipts

The standard Python UT driver uses a different snapshot algorithm from the
canonical source-delivery snapshot: it hashes relative paths followed by their
file SHA-256 strings, in a fixed order. Do not rewrite its receipt, replace the
canonical `snapshot_sha256`, or accept two unrelated digests merely because
both are present. For an existing driver receipt, explicitly add this field to
the **source-delivery manifest**:

```json
{
  "ut_snapshot": {
    "contract": "ut_driver/path_sha256_concat/v1",
    "report_ref": {"path": "run/ut_report.json", "sha256": "..."}
  }
}
```

Keep `ut_result_ref` pointing to the original, hashed `run/ut_result.json`.
Hash the actual full `run/ut_report.json` using `ref TASK FILE`. This opt-in
contract is accepted only when all of the following hold:

- The canonical `sourceSnapshot`, `files`, target callable, per-leaf source
  hashes, execution records and numerical rejection records still validate
  under the original source-delivery rules.
- Both Python receipts have the current driver snapshot: concatenate
  `meta.json`, `cases.py`, `unittest.py`, `run/ut_cases.json`,
  `run/ut_contract.json`, then the sorted files under `kernel_src/`, excluding
  paths containing `__pycache__`. Each contribution is the relative path in
  UTF-8 followed by the file's lowercase hexadecimal SHA-256 in UTF-8; missing
  optional snapshot paths are skipped, matching `_snapshot`.
- The full report's `reuse_key` equals a fresh computation over the driver's
  exact `_reuse_key` inputs: sorted top-level `.py`/`.json` files, sorted
  `run/ut_cases.json`, `run/ut_contract.json` and `run/server_*` entries, sorted
  files immediately under `run/server_calls/`, then the sorted `kernel_src/`
  files. These groups are concatenated in that order, hashing regular files
  only. This additionally binds frozen helpers and server evidence that the
  older driver snapshot alone omitted. Symlink file targets must stay within
  the task, and symlink directories are not traversed.
- The result is `ok:true`, both result and report have `verdict:"pass"` and
  `exit_code:0`, and they identify the same `started`, `target_callable`,
  `family` and `contract_sha256`. The full report must not be a quick run.
  New receipts explicitly carry `validation_contract: "kernel_ut_validation/v1"`,
  matching `validation_mode` and `required_gates`. In `correctness` mode all eleven
  non-timing gates must appear and pass (or be `not_required`), the timing gate is
  absent, and `performance_measured` is false. In `performance` mode all twelve gates
  including timing must pass and `performance_measured` is true. Legacy receipts
  without a validation mode still require all twelve gates. Missing/skipped/failed
  required gates, mismatched modes, altered hashes and unknown contracts are rejected.
  A correctness pass never supplies a speedup or an invented baseline time.

Inspect the current Python-compatible hashes without running Python or a GPU:

```bash
node "$SHARED_TOOLS/kernel_source_delivery.mjs" ut-driver-snapshot "$TASK"
```

This command is diagnostic; it does not certify execution. The verifier
recomputes the hashes itself rather than trusting values copied into a manifest.
Per-leaf execution and negative-control records continue to name the **canonical
source-delivery snapshot**, not the Python driver snapshot.

This explicit compatibility contract also accepts the Python JSON serializer's
literal positive `Infinity` in two successful negative-control margin locations:

- Full report: `gates.negative_controls.controls[i].margin`, where `controls`
  is an array, the gate has `status:"pass"`, and that control has `caught:true`,
  `ok:true` and `restored:true`.
- Per-leaf numerical rejection record: `ut_negative_controls_gate.margin`,
  where the outer record has `kind:"numerical_rejection"`, `ok:true`,
  `launch_succeeded:true` and finite positive `numerical_mismatches`, and the
  nested control has `caught:true`, `ok:true` and `restored:true`.

The verifier hashes the original artifact bytes before parsing. It normalizes
only lexical tokens outside JSON strings, without evaluating code or rewriting
the artifact. Quoted or escaped string contents remain unchanged. `NaN`,
`-Infinity`, numeric overflow and positive `Infinity` at every other location
are rejected; duplicate properties cannot hide an `Infinity` token. The
manifest, UT result summary and candidate-execution records still require
strict JSON, as does every artifact without this explicit compatibility
contract. All snapshot, reuse-key, source and full-gate checks still apply.

After an isolated timing tail, hash the updated real result/report and update
only delivery metadata stored under an excluded location such as
`run/source_delivery.json`. Adding a top-level `.json`/`.py` file or a
`run/server_*` file after validation changes `_reuse_key` and correctly makes
the evidence stale. Do not edit `meta.json`, task helpers, inputs or receipts to
force compatibility; changed test inputs still require fresh validation.

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
negative controls that only fail compilation. Declare `source_delivery_required=true` for every task; do not accept
a source-free declaration as an extraction delivery.

<!-- Standalone adaptation, 2026-10-12: local tools and agent-owned stages. -->
