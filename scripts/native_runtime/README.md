# Native task measurement adapter

`direct_external_command.py` connects a supplied native task evaluator to the
ordinary GEAK and GEAK-EVO measurement/result command interface. It uses
`geak.native` for task validation, source capture, native execution, and raw
report reading. The caller provides the task environment, GPU access, and lock.

Use either `--gpu-lock-file PATH` for the existing pinned-GPU command or
`--gpu-pool 0,1,...` for an allocated pool in one runtime context. Pool mode runs
the complete native UT through the stock `gpu_lock.sh`, which picks a free GPU
and uses the same locks as profiling. Invoke the adapter directly in either
mode; an additional outer GPU wrapper would acquire the same lock twice. The
caller supplies a consistent local-to-physical device mapping and allocation.
Each pooled evaluation gets private Torch extension, Triton, XDG and AITER JIT
cache directories even when the caller inherited shared cache settings.
`AITER_ROOT_DIR` continues to name the installed dependency.
The stock pool wait limit and use log still apply; pool acquisition has no FIFO
or starvation guarantee. More engineers can increase time spent waiting.

To continue editing while a measurement waits or runs, first capture its source:

```text
measure --workspace W --capture-only
measure --captured --candidate-id C --evaluation-id E
read --captured --candidate-id C --evaluation-id E
```

These tails follow the same configured command prefix (task root, state root,
epoch and GPU lock or pool). Capture returns `candidate_id`, `evaluation_id`, `source_id`
and `source_workspace`; it saves only the declared editable files, including
their executable bits, before publishing the request. Protected task files and
any declared shared inputs keep their existing task-root binding. `--captured`
requires the returned ID pair and forbids a workspace or patch argument. Its
result describes that retained source even when W has moved on. Associate the
result with those IDs; export or reconstruct that source before promoting it.
The normal `measure/read --workspace W` calls still reject a different source.

The adapter remains synchronous. A caller may use its existing owned command
handle to run the second command in the background while doing independent CPU
work, then wait for that specific command and run the third command to read its
result. Keep the handle together with the captured ID pair. Capture and read
emit one JSON object; measure emits an identity line followed by its final
checked result as JSON Lines. After successful process settlement, keep the
last complete checked result; do not parse the entire measurement stdout as a
single JSON object or treat its first identity line as a score.

The caller must retain process ownership through completion or
record cancellation; an agent-owned command can be stopped when that agent
exits. Capture does not create a background service, preserve a process across
agent exits, or deliver a later notification. Failed or partial evaluations keep
their IDs and are never silently rerun. Missing or incomplete evidence is not a
valid measurement.

- [Configure native measurements](../../docs/evo/direct-external-command.md)
- [Share immutable task inputs](../../docs/evo/shared-task-inputs.md)
- [Run GEAK-EVO](../../docs/evo/stock-workflow.md)

The former pilot launchers, provider patches, and research receipts have been
removed from this directory. Historical deployments keep their pinned sources;
this adapter does not update them.
