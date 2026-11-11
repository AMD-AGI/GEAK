---
myst:
    html_meta:
        "description": "Run GEAK-evo kernel and end-to-end serving workflows with the bundled Node runtime or the standard Python handoff entrypoint."
        "keywords": "GEAK-evo, run workflow, serving throughput, kernel, Node, Claude Code, sglang, vLLM, gsm8k"
---

# Run a GEAK-evo workflow

GEAK-evo uses the bundled Node runtime to orchestrate the agent CLI and nested
kernel work. Use this runtime even when Claude is the selected backend: kernel
delivery needs `runProcess`, which the native Claude `Workflow` tool lacks.
Omitting `optimizer` selects `evo`; add `"optimizer": "greedy"` to the workflow
arguments or E2E handoff to select the original search policy.

## Prerequisites

Complete the [installation guide](../install/install.md), including Node.js,
ROCm, a profiler and agent credentials. E2E runs also need a working serving
backend and model weights. Run the commands below from the release checkout.
Replace example paths and GPU IDs with your own; use a fresh evaluation directory
for each run.

## Single kernel

Save this as `kernel-args.json`. The task directory must contain the kernel and its
correctness/benchmark harness; see the [kernel workflow guide](../../kernel_workflow/README.md).

```json
{
  "kernel_path": "/absolute/path/GEAK/examples/tasks/knn",
  "workflow_dir": "/absolute/path/GEAK/kernel_workflow",
  "eval_dir": "/absolute/path/runs/knn-001",
  "gpu_ids": "0",
  "budget": 6
}
```

```bash
node interface/runtime/engine/run_workflow.mjs \
  kernel_workflow/kernel_workflow.js \
  --agent claude --args-file kernel-args.json \
  --result-file kernel-workflow-result.json
```

The evaluation directory contains the reports, verified final patch and EVO
archive. The result file contains the workflow's return value. See
[Run GEAK-evo](../evo/stock-workflow.md) for retained-candidate options and output details.
For concurrent runs, assign separate GPU sets and evaluation directories.

## End-to-end serving

For Hyperloom or another integration, keep the standard two-file invocation:

```bash
python interface/run_e2e.py /path/to/handoff.json /path/to/result.json
```

The wrapper detects the hardware identity, invokes the bundled runtime and writes
the existing normalized `result.json`. See the [integration contract](../../interface/run_e2e.md)
for the handoff fields, AgentX configuration and backend selection.

For a direct run, save the following as `e2e-args.json`. This example targets four
MI355X GPUs; set `expected_gfx` and `expected_target` to the actual device identity.
Direct model-mode runs require both fields.

```json
{
  "model_path": "/models/Qwen3.5-27B-FP8",
  "workflow_dir": "/absolute/path/GEAK/e2e_workflow",
  "eval_dir": "/absolute/path/runs/qwen-001",
  "expected_gfx": "gfx950",
  "expected_target": "mi355x",
  "backend": "sglang",
  "tp": 4,
  "gpu_ids": "0,1,2,3",
  "isl": 1024,
  "osl": 1024,
  "conc": 64,
  "budget": 4,
  "kernel_budget": 6
}
```

```bash
node interface/runtime/engine/run_workflow.mjs \
  e2e_workflow/e2e_workflow.js \
  --agent claude --args-file e2e-args.json \
  --result-file e2e-workflow-result.json
```

For fixed-load workloads, the workflow profiles the server, tunes configuration
and hot kernels, then re-measures integrated changes. AgentX trace replay uses
baseline measurement, kernel extraction and standalone task validation, kernel
optimization, then serving validation; it skips the outer configuration sweep.
Reports and the final overlay, patch and launch
script are written under `eval_dir`. The direct Node return is separate from the
normalized integration result produced by `interface/run_e2e.py`.

## Depth and accuracy options

Add these fields to the E2E arguments as needed. The table describes the fixed-load
serving pipeline. AgentX retains the head/milestone scheduling of its selected
mode and skips ConfigSweep. `optimizer` selects the nested kernel search policy.

| Mode | Fields | Phases |
|---|---|---|
| Default | Omit `fast_mode` and `deep_mode`. | ConfigSweep, HeadKernel and Milestone. |
| Fast | `"fast_mode": true` | HeadKernel only, with `fast_budget_ms` defaulting to five hours. |
| Deep | `"deep_mode": true` | ConfigSweep and a multi-round HeadKernel pool, with `deep_head_budget_ms` defaulting to 24 hours. |

Set only one mode; if both are true, deep takes precedence. For quantized workloads
where task accuracy is the intended acceptance criterion, add these fields:

```json
{
  "accuracy_gate": "gsm8k",
  "accuracy_limit": 200,
  "accuracy_tol": 0.01
}
```

This selects sampled gsm8k acceptance within the declared tolerance instead of
greedy output parity. See the [E2E guide](../../e2e_workflow/README.md) for the full
measurement contract and [runtime setup](../../interface/runtime/SETUP.md) for
agent and model selection. Existing `GEAK_*` settings remain supported.
