# Measure an external task from the ordinary GEAK workspace

`scripts/native_runtime/external_measurement.py` connects the common GEAK lane
to its existing admitted `ResearchStore`, `ArenaEvaluator`, `NativeCallbacks`
and `control_result` reader. The operational owner installs/registers the script
with the native sources and supplies an existing sealed admission, ordinary
state directories, authorized sudo/helper invocation and study clock.

Configure these argv prefixes in the common lane:

```json
{
  "external_measurement_command": ["/usr/bin/python3", "-I", "-B", "INSTALLED_SCRIPT", "--admission", "ADMISSION_PATH", "ADMISSION_SHA256", "measure"],
  "external_result_command": ["/usr/bin/python3", "-I", "-B", "INSTALLED_SCRIPT", "--admission", "ADMISSION_PATH", "ADMISSION_SHA256", "read"]
}
```

The engineer appends `--workspace ABSOLUTE_WORKSPACE` to `measure`. That workspace
must contain the admitted task files. Protected UT, baseline, metadata and
harness bytes must remain unchanged. The script captures only declared source
files and their executable bits, creates the actual candidate, and evaluates
that candidate with a fresh ID. Missing source fails without ORIGINAL fallback.

The first stdout JSON record contains `candidate_id`, `source_id` and
`evaluation_id`, flushed before evaluation. The second contains the same IDs,
`external_evaluation`, receipt and costs. Return the exact
`external_evaluation:{candidate_id,evaluation_id}` to the common lane. A settled
native suite pass returns exit code zero even when it has no reproducible gain.
The existing native result reader and optimizer apply the relevant metrics and
selection rules.

An optional `--evaluation-id ID` uses the existing evaluator's reconciliation
semantics. The same candidate and ID are collected without relaunch; another
candidate using that ID is refused. The script preserves the initial ID output
if evaluation raises or is interrupted and performs no automatic retry or
cleanup operation.

The common lane appends the reader's existing arguments:

```text
--candidate-id ID --evaluation-id ID --workspace PATH [--patch PATCH_PATH]
```

`read` checks protected workspace artifacts and calls
`control_result.main(..., evaluator=the_existing_evaluator)`. Its stdout is the
reader's single JSON result. The reader checks source or source-plus-patch bytes,
exact candidate and evaluation identity, native settlement, runtime/device
evidence and raw timing evidence. It launches no evaluation.

The admission supplies the native installation, immutable task, complete source
registration, epoch, source-object root, evaluation root and existing clock.
Before importing code, the script checks root ownership, source hashes and its
own registration. It rejects unregistered Python files and importable caches or
native extensions, then rechecks actual imported paths. Importing the named
package does not add native directories to Python's standard-library search
path. The loaded evaluator hash must match the admission. Store and evaluation
directories must already belong to the ordinary host user.

This entry creates no admission or clock and does not extend a deadline or add
a GPU lifecycle. Callback launch, settlement and clock enforcement remain with
the existing owner. A prospective source/admission closure and installed-owner
qualification are required before use. CPU tests exercise source capture,
protected files, IDs, reader argument wiring and same-ID reconciliation with
synthetic callbacks; they do not establish installed custody or native results.

## Search and final deadlines

A model-process timeout does not cancel the independently owned native GPU
operation. For a bounded search, apply `native-phase.patch` to the exact helper
in `native-phase-base.json`, then register the resulting source through the
existing owner. The optional immutable admission field `phase_deadline_seconds`
is an offset from the original monotonic start. It requires the full evaluation
envelope to fit before that phase ends; it never shortens the native session,
settlement reserve, or original clock. Admissions without the field retain their
existing behavior.

Use separate admitted command prefixes for search and Director validation:

```json
{
  "external_measurement_command": ["/owner/measure-search"],
  "external_result_command": ["/owner/read-search"],
  "external_final_measurement_command": ["/owner/measure-final"],
  "external_final_result_command": ["/owner/read-final"]
}
```

Both final prefixes must be supplied together. Only `director:validate` receives
the final measurement prefix, and its result is checked with the final reader.
All other native measurements use the search prefix. The two admissions use the
same task, source store, measurement epoch and original clock; their immutable
phase deadlines differ. The owner must install both before launch. Paths above
are examples, not installed commands.

GEAK-EVO's command also performs its own baseline/selected final pair. Keep that
existing final evaluator on its appropriate final admission, not on GEAK's
search-only admission. Its original shared call allowance must retain the final
calls; its smaller search limit already has a separate configuration field.

The common search window and latest permitted native start must be declared for
both arms. Account for the full evaluation envelope and any existing preparation
margin. A short pipeline check establishes execution compatibility; it does not
establish a long-horizon advantage for either optimizer.
