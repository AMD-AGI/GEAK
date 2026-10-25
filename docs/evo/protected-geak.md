# Include a complete GEAK run

`interface/runtime/engine/compose_kernel.mjs` is an opt-in standalone composition:
one complete GEAK run, one or more complete extra runs, and a fresh common
native evaluation of their returned finalists. Extras normally use GEAK-EVO;
independent GEAK controls are also supported. It does not change the meaning of
`optimizer=greedy` or `optimizer=evo` in existing workflows or frozen studies.

The GEAK child keeps its full arguments, prompts, model settings, search budget,
stopping policy, knowledge settings and finalization stages. Extra runs receive
additional execution contexts and resources. The composer selects from actual
final programs; it does not merge search histories or inject EVO findings into
GEAK. This first version compares eligible **workflow finals**, not every program
in an EVO archive. End-to-end scheduling and deployment are not part of this CLI.

## Prepare the execution contexts

The existing operational owner must supply isolated worker contexts and an
available common final-evaluation context. The composer does not create
containers, assign GPUs or attest isolation. Distinct names in a JSON file are
not evidence that mounts or physical devices are isolated.

- Keep GEAK's original agent capacity, physical GPU grant, CPU service and full
  clock allowance. Give the extra runs separate grants. Two processes sharing a
  GPU lock are measurement-safe but do not preserve GEAK's former GPU service.
- Expose only each worker's own writable task copy, experiment/state directories,
  runtime sessions, caches, local KB and learned-card writes. Common immutable
  inputs may be mounted read-only. Do not let extra workers modify knowledge or
  feedback that GEAK can read, including shared/remote KB writes.
- Keep both existing GEAK and EVO argument files complete. In particular, bind
  `exp_root`, `eval_dir`, optional `state_dir`, writable KB roots, evaluator result
  roots and GPU locks to the correct private context. Preserve GEAK's settings;
  do not disable its KB or shorten its budget for the composition.
- Paths used for arguments, exports, the immutable original task and final native
  evidence must be visible to the composer at the same absolute paths. Optional
  per-lane `output_dir` supports mounting a private export root into each worker.
  The common evaluator may read both retained exports; worker agents must not.
- Worker launch commands must retain ownership until the full runtime and its
  children settle. A remote/container client whose exit leaves work running is
  insufficient. The existing process-bounded worker deadlines remain required.
- Reserve a common final-evaluation service/device after the searches. All
  returned finalists use the same command prefix and execute sequentially there.
  Preserve device/image/measurement provenance in the existing owner configuration
  and native evidence; matching task bytes alone does not establish a common device.

## Configuration

Use the native task and command protocol in [Native measurements](direct-external-command.md).
The supported schema is `geak-arena-external-task-v1`, using `glm-weight-lib-v2`.
The composer validates the actual task artifacts through the existing native
validator, checks the exact `task.json` SHA256, and binds candidate identities to
the resulting task digest, measurement epoch and source digest.

Run:

```sh
node /workspace/GEAK/interface/runtime/engine/compose_kernel.mjs /operator/composition.json
```

The following example uses already owned contexts. Replace every illustrative
path, container name, physical device identity, SHA256 and model with the
operator's actual bindings. Both contexts need the same checkout at
`/workspace/GEAK`; the composer invokes its existing `kernel_workflow.js`.

```json
{
  "schema": "geak.kernel-composition/1",
  "task_root": "/inputs/task",
  "task_sha256": "<SHA256 of the exact task.json bytes>",
  "out_dir": "/exports/comparison-001",
  "evaluation": {
    "measurement_command": ["python3", "-B", "/workspace/GEAK/scripts/native_runtime/direct_external_command.py", "--task-root", "/inputs/task", "--state-root", "/exports/common-native", "--gpu-lock-file", "/tmp/team_gpu_locks/gpu_2.lock", "--epoch", "comparison-001", "measure"],
    "result_command": ["python3", "-B", "/workspace/GEAK/scripts/native_runtime/direct_external_command.py", "--task-root", "/inputs/task", "--state-root", "/exports/common-native", "--gpu-lock-file", "/tmp/team_gpu_locks/gpu_2.lock", "--epoch", "comparison-001", "read"],
    "timeout_ms": 3600000
  },
  "lanes": [
    {
      "id": "geak",
      "args_file": "/operator/geak-args.json",
      "output_dir": "/exports/geak/comparison-001",
      "runtime_command": ["docker", "exec", "owned-geak", "node", "/workspace/GEAK/interface/runtime/engine/run_workflow.mjs"],
      "runtime_options": {"agent": "codex", "model": "<model>", "concurrency": 4, "agent_timeout_ms": 3600000},
      "timeout_ms": 32400000,
      "resources": {"context_id": "owned-geak", "device_ids": ["<GEAK physical device identity>"]}
    },
    {
      "id": "evo",
      "args_file": "/operator/evo-args.json",
      "output_dir": "/exports/evo/comparison-001",
      "runtime_command": ["docker", "exec", "owned-evo", "node", "/workspace/GEAK/interface/runtime/engine/run_workflow.mjs"],
      "runtime_options": {"agent": "codex", "model": "<model>", "concurrency": 4, "agent_timeout_ms": 3600000},
      "timeout_ms": 32400000,
      "resources": {"context_id": "owned-evo", "device_ids": ["<additional physical device identity>"]}
    }
  ]
}
```

Exactly one lane must be named `geak`, with `optimizer="greedy"` in its argument
file. Additional lanes may use `optimizer="evo"` or `optimizer="greedy"`. An extra
independent GEAK control makes it possible to compare additional GEAK capacity
with additional GEAK-EVO capacity using the same composition and measurement
rule. Each full argument file must
use `mode="optimize"` (or omit it), `apply_to_original=false`, the pinned native
objective, and existing `deadline_epoch`/`search_reserve_s` settings. Final native
reader commands must explicitly include `--task-root`, `--state-root`,
`--gpu-lock-file` and `--epoch`, ending in `read`. Existing measurement commands
end in `measure`. Supply options as separate argv entries, not shell strings.

`runtime_options` supports `agent`, `profile`, `model`, `registry`, `concurrency`
and `agent_timeout_ms`. Optional lane `cwd`/`env` configure the existing process
launcher. Optional evaluation `cwd`/`env` apply consistently to capture,
measurement and reading. Keep credentials in private owner configuration, not
public manifests. Each runtime has its own concurrency semaphore and call budget.
This does not remove external provider rate limits.

The output directory and optional lane output directories must be fresh, with
their parent directories already present. Arguments are copied byte-for-byte to
each lane's output before launch. Without `output_dir`, a lane writes under
`out_dir/<id>`. Grant each worker access only to its own exports.

The lane `timeout_ms` must encompass its full absolute deadline plus at least
seven seconds of process cleanup; it is not a replacement search budget.
Common evaluation begins after all worker processes settle. Its `timeout_ms`
applies separately to capture, measurement and reading for each finalist, so the
owner must allow the sum of these calls, staging and final publication. The
composer does not shorten GEAK to make room for comparison.

## Selection and retained evidence

Each lane preserves its runtime result, metrics, stdout/stderr, immutable argument
snapshot and copied final patch. Its native final reader verifies the patch
against the actual original source and retained child evaluation. A director
rejection is preserved. A missing or invalid child is recorded explicitly and
does not cancel another child.

The composer then uses the stock workspace materializer and native source/patch
helpers to stage a fresh ORIGINAL plus each saved patch. It captures new native
IDs, measures the complete task, and re-reads the checked native reports. Among
returned finalists with a fresh native reproducible gain and no reproducible
regression, the largest common `primary_speedup` decides the winner; ties prefer
the included GEAK. The GEAK fallback below takes precedence if its common replay
fails. The development geomean and child-reported scalar are not comparison
scores.

`result.json` contains all child records, `protected_geak`, `comparison_complete`,
`selection_basis` and the selected patch/source/evaluation identities. Per-lane
`candidate.json` is written when its final patch is retained and updated after
common qualification. The original child results are never overwritten by a
comparison scalar.

- If an extra fails, GEAK's valid result remains available.
- If GEAK produced an eligible retained program but its common replay fails,
  that GEAK program is the fallback even if an extra qualifies. The result says
  `comparison_incomplete_geak_retained`, with `primary_speedup: null`; its earlier
  child score remains separately recorded under `child_evidence`.
- If GEAK itself is missing/invalid, an extra may still supply a usable result,
  but `protected_geak.qualified=false` and the CLI exits nonzero. This is not
  reported as successful GEAK containment.
- Valid no-gain measurements are `no_improvement`; missing measurements are
  failures/unavailable, not measured zero speedup. Without a common qualified
  gain or an eligible retained GEAK fallback, `winner` is `null` and
  `selection_basis` is `original_retained`. A valid no-gain GEAK final may retain
  the existing `host_final_patch.diff` even though the workflow suppresses patch
  export.

When the included GEAK finalist has a qualified gain in the fresh common native
measurement, the selected score is the maximum over the eligible returned
workflow finalists and is at least GEAK's score in that comparison. Equality is
allowed. This guarantee covers the actual included run and the common
measurement; it excludes unevaluated archive candidates and separately sampled
stochastic GEAK runs.
Selecting a maximum from noisy scores can overstate repeatable gain, even after
the native reproducibility checks. The selected score does not establish the
lowest true latency or an end-to-end throughput gain.
