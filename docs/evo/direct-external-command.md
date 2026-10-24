# Native measurements

GEAK and GEAK-EVO can use the same supplied native task evaluator through
`external_measurement_command` and `external_result_command`.
[scripts/native_runtime/direct_external_command.py](../../scripts/native_runtime/direct_external_command.py)
captures candidate source, calls `geak.native.external_runner`, and reads its raw
evidence through `geak.native.external`. The caller supplies an existing GPU
environment, an immutable task, a writable result directory, and the GPU lock.

The supported task schema is `geak-arena-external-task-v1` with the
`glm-weight-lib-v2` report method. Other task formats use their existing workflow
evaluators; the retired Python driver's other evaluation schemas are not included.

New task preparation names `geak.native.external_runner` in the contract, which
changes the task digest relative to the former runner name. Existing pinned
contracts keep their identities. Replay historical studies with their frozen
sources; this cleanup does not migrate their requests or results. The former
runner's alternate `--work`/`--storage-manifest` route is retired; use the private
workspace and optional shared-input mapping described below.

## Configure the workflow

The following paths are illustrative. Select the Python interpreter that has
the task dependencies, and make this checkout's `geak` package importable by
that interpreter, for example with `PYTHONPATH=/workspace/GEAK`.

```json
{
  "objective": {"kind": "external_primary", "primary_regime": "<task-primary-regime>"},
  "external_measurement_command": ["python3", "-B", "/workspace/GEAK/scripts/native_runtime/direct_external_command.py", "--task-root", "/task", "--state-root", "/results/native", "--gpu-lock-file", "/tmp/team_gpu_locks/gpu_0.lock", "measure"],
  "external_result_command": ["python3", "-B", "/workspace/GEAK/scripts/native_runtime/direct_external_command.py", "--task-root", "/task", "--state-root", "/results/native", "--gpu-lock-file", "/tmp/team_gpu_locks/gpu_0.lock", "read"]
}
```

Take `primary_regime` from the supplied `task.json` contract. Use the same state
root and measurement epoch for capture, measurement, and reading. The optional
`--epoch` defaults to `direct-native-v1`. Optional
`external_final_measurement_command` and `external_final_result_command` must
be supplied together if final validation needs distinct command prefixes.

All engineers must share the workflow's GPU lock. In a singleton-GPU container,
`/tmp/team_gpu_locks/gpu_0.lock` is the usual stock lock path. Invoke the adapter
directly: it acquires that lock itself. Wrapping it in `gpu_lock.sh` would acquire
the same lock twice. The enclosing workflow owns device access and deadlines.

## Command protocol

The complete command tails have these forms:

```text
measure --workspace W --capture-only
measure --workspace W --candidate-id C --evaluation-id E
read --workspace ORIGINAL --candidate-id C --evaluation-id E --patch PATCH
```

A plain `measure --workspace W` allocates fresh IDs and runs the measurement.
Capture-only mode returns IDs without running the task. Measuring those IDs
requires the captured source to be unchanged. Existing native operations are
read again; partial or failed operations are not silently rerun.

The reader checks the workspace and optional patch against the captured source,
then reconstructs the result from the native request, receipts, reports, and
history. Saved summary scalars alone are insufficient. The task contract supplies
the numerical checks, cases, timing protocol, and primary objective. The checked
primary metric controls ranking; the all-case geometric mean is diagnostic.
A valid measurement can still show no improvement.

See [shared immutable inputs](shared-task-inputs.md) for optional fixture sharing.
These helpers provide measurement and source checks; deployment and GPU execution
must be established in the caller's actual environment. This interface makes no
claim of search or performance benefit.
