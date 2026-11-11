# Standalone full e2e loop

Read this when the user needs ConfigSweep or direct workflow controls that the
Python handoff does not forward. Keep orchestration in GEAK. Native Claude's
`Workflow` and the standalone Node host both run the unchanged workflow.

## Prepare direct args

After Claude/environment setup, set `GEAK_ROOT`, `GEAK_PYTHON` and `RUN_DIR`
to absolute paths; create a fresh `RUN_DIR`. On the allocated GPU host, using
the same GPU visibility as the planned run:

```bash
"$GEAK_PYTHON" "$GEAK_ROOT/scripts/gpu_identity.py" > "$RUN_DIR/gpu-identity.json"
```

Read the detected `gfx`, `target`, `marketing_name` and `physical_cu_count`.
Both `expected_gfx` and `expected_target` are mandatory in a direct model-mode
call. For Instinct, `target` can legitimately be `unknown`; generic gfx1201
does not imply R9700. Follow current product/backend policy in the checkout.

Write `$RUN_DIR/e2e-args.json` as an args object. Below is a **MI355X example**;
replace its identity with actual detector output and all workload/path values
with the resolved request:

```json
{
  "model_path": "/models/Qwen3.5-27B-FP8",
  "workflow_dir": "/work/GEAK/e2e_workflow",
  "expected_gfx": "gfx950",
  "expected_target": "unknown",
  "expected_device_name": "AMD Instinct MI355X",
  "expected_physical_cu_count": 256,
  "backend": "sglang",
  "tp": 1,
  "gpu_ids": "0",
  "isl": 1024,
  "osl": 1024,
  "conc": 64,
  "config_tune": "true",
  "budget": 4,
  "kernel_budget": 6,
  "time_budget_s": 14400,
  "exp_root": "/work/geak-run/geak",
  "eval_dir": "/work/geak-run/geak/e2e_run_001",
  "apply_to_original": "true"
}
```

The e2e `apply_to_original` flag requests a reusable overlay/launch bundle;
unlike the standalone kernel flag it does not mean modifying site-packages.
Use a fresh `eval_dir` unless deliberately resuming that exact run.

`budget` counts kernel tasks; `kernel_budget` is the direction budget for each
recursive kernel run. ConfigSweep does not consume those task counts.
`time_budget_s` provides workflow pacing/reserve but is not an OS kill timer.
Match a requested hard deadline in the scheduler/supervisor and leave enough
time for Finalize/Report/Validate. Do not interpret task counts as hours.

If supplied, translate existing baseline settings to direct workflow keys:
`launch_script`, `initial_extra_server_args`, `initial_extra_env`, and
`initial_overlay_pythonpath`. Carry benchmark protocol/client settings exactly
as documented in the checkout; do not rename handoff fields by guesswork.

## Native Claude Workflow

In a running native Claude session with Workflow enabled, read `e2e-args.json`
and call the tool with the **parsed object**, not the filename or encoded JSON:

```javascript
Workflow({
  scriptPath: "/work/GEAK/e2e_workflow/e2e_workflow.js",
  args: resolvedArgs
})
```

Here `resolvedArgs` denotes the complete resolved object above. It is not a
literal tool argument to leave unresolved. Keep the Claude session and any
background Workflow task alive; wait for the workflow terminal result. Do not
launch a native background Workflow through a short-lived `claude -p` wrapper.

## Headless Node host with Claude agents

Use this when the current agent/shell has no native Workflow tool:

```bash
node "$GEAK_ROOT/interface/runtime/engine/selftest.mjs"
node "$GEAK_ROOT/interface/runtime/engine/run_workflow.mjs" \
  "$GEAK_ROOT/e2e_workflow/e2e_workflow.js" \
  --agent claude --args-file "$RUN_DIR/e2e-args.json" \
  --result-file "$RUN_DIR/workflow-result.json" \
  --metrics-file "$RUN_DIR/runtime-metrics.json" \
  > "$RUN_DIR/run.log" 2>&1
```

Run under a persistent session/job. This host owns the workflow lifetime and
uses Claude CLI for individual agents; it does not use Python `claude-agent-sdk`.
Do not describe it as the SDK path merely because `--agent claude` is selected.

Node's result file is the **raw workflow return**, not the normalized
`interface/run_e2e.py` result schema. Read it alongside the pinned
`workflow_return.json`, final report and Director validation; do not assume
interface-only fields or normalize a success from a process exit code alone.

Changing `--agent` to another provider is a deliberate backend choice. Follow
the selected checkout's runtime setup when the user requests that change.
