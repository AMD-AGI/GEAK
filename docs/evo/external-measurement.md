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
