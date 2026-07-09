---
myst:
    html_meta:
        "description": "Run a GEAK workflow: optimize whole-model sglang/vLLM serving throughput or a single GPU kernel from Claude Code, or via the run_e2e.py integration interface."
        "keywords": "GEAK, run workflow, serving throughput, single kernel, Claude Code, run_e2e, sglang, vLLM, ROCm"
---

# Run a workflow

GEAK exposes two workflows. Start a run by describing the task in natural language to
[Claude Code](../install/install.md) — it resolves the paths and invokes the `Workflow` tool. For an
external orchestrator, use the [`run_e2e.py` interface](#external-integration-run_e2epy) instead.

## Optimize whole-model serving throughput (`e2e_workflow`)

Point at the repo's `e2e_workflow/`, the model, the serving backend, the workload, and the GPUs:

```
use path_to_GEAK/e2e_workflow to optimize inference for /models/Qwen3.5-27B-FP8, sglang, ISL/OSL=1024, conc=64, gpus 0,1,2,3
```

Output lands under `e2e_workflow/exp/e2e_<model>_<timestamp>/` — `final_report.md`,
`architect_report.md`, a `final/` bundle (overlay + `final_patch.diff` + `final_launch.sh`), and
per-stage artifacts.

## Optimize a single kernel (`kernel_workflow`)

Point at `kernel_workflow/` and a kernel task directory or source file:

```
use path_to_GEAK/kernel_workflow to optimize path_to_GEAK/examples/tasks/knn
```

```
use path_to_GEAK/kernel_workflow to optimize /path/to/silu, budget 8, focus on wrapper overhead
```

To optimize many kernels at once, spawn one agent per kernel; GPU access is serialized via
`scripts/gpu_lock.sh` (flock-based), so kernels can safely share GPUs.

## External integration (`run_e2e.py`)

`interface/run_e2e.py` is the single, version-stable surface for an external orchestrator. It hides the
`Workflow` invocation and the `e2e_workflow.js` argument names behind one command and two JSON files:

```bash
python interface/run_e2e.py <handoff.json> <result.json> [--dry-run]
```

- `handoff.json` (caller → workflow): `model_path` and `exp_root` are required; everything else
  (`framework`, `tp`, `gpu_ids`, `workload`, seed config, measurement protocol) has a default.
- `result.json` (workflow → caller): `status` (`ok` / `no_gain` / `error`), baseline and final
  throughput, speedup, output parity, and paths to the deliverable bundle.
- `--dry-run` prints the mapped arguments and prompt without running any GPU work — use it to validate
  the mapping in CI.

Exit codes: `0` = `ok`/`no_gain`, `1` = crash (`status: error`), `2` = bad usage. See
[Reference](../reference/api-reference.md) for the full handoff/result schema.

## Related topics

- [Install GEAK](../install/install.md) — set up the environment and Claude Code.
- [GEAK pipeline](../conceptual/geak-pipeline.md) — the phases of an end-to-end run.
- [Reference](../reference/api-reference.md) — Workflow arguments, the integration contract, and artifacts.
