# Run GEAK-EVO

GEAK-EVO runs through the ordinary kernel workflow. Set `optimizer` to `evo` in
the workflow arguments; `greedy` is the default for GEAK. Install the runtime
using the [installation guide](../install/install.md), and prepare a task that
builds and passes its original correctness checks in the intended GPU environment.

From the repository root:

```sh
node interface/runtime/engine/run_workflow.mjs \
  kernel_workflow/kernel_workflow.js \
  --agent codex --args-file /absolute/path/run-args.json \
  --result-file /absolute/path/result.json
```

An example `run-args.json`:

```json
{
  "kernel_path": "/absolute/path/task",
  "workflow_dir": "/absolute/path/GEAK/kernel_workflow",
  "eval_dir": "/absolute/path/fresh-evaluation",
  "optimizer": "evo",
  "budget": 6,
  "gpu_ids": "0"
}
```

Configure the model/backend and task dependencies using the existing runtime
and [kernel workflow](../../kernel_workflow/README.md) instructions. The example
paths and GPU ID are placeholders for the caller's environment.

## Relevant options

| Argument | Default | Meaning |
| --- | --- | --- |
| `optimizer` | `"greedy"` | `"evo"` enables the retained population in the shared loop. |
| `budget` | `6` | Direction budget; each deep engineer costs `deep_cost` units. |
| `deep_cost` | `2` | Budget charge for a deep engineer. |
| `max_no_improve` | `2` | Consecutive rounds without qualifying progress before stopping. |
| `min_improve` | `0.02` | Required relative improvement over the current best for promotion. |
| `evolution_evidence_handoff` | `false` | Boolean `true` adds selected parent/donor evidence to engineer prompts. |

GEAK-EVO can verify and retain slower candidates without promoting them. It also
allows parallel deep engineers. Account for these choices when comparing search
costs. Use the same task, model, numerical contract, objective, and measurement
rules for a GEAK comparison. For a timed search, configure the existing deadline
and finalization reserves, and choose `max_no_improve` for the intended stopping
policy in both runs; a large direction budget alone does not prevent early stops.

## Outputs and limits

The ordinary reports and final patch remain under the evaluation directory.
GEAK-EVO also writes `evolution/archive.json`, its retained patches, and a fixed
run base. See the [archive guide](archive.md) to inspect source and evidence.

Use a fresh evaluation directory for each run. Archive files survive ordinary
round cleanup, but there is no protocol for resuming this population after a
terminated run. A hard termination can prevent final measurement.

For tasks with a supplied native evaluator, use the existing measurement and
result command fields described in [native measurements](direct-external-command.md).
They are shared by GEAK and GEAK-EVO. The former Python EVO driver is retired.
Remove `optimizer_command`, `optimizer_task_ref`, and `optimizer_reserve_s` from
old configurations; these arguments are rejected at startup.
