---
name: geak-e2e-workflow
description: >-
  Set up and run GEAK e2e_workflow to optimize whole-model inference on AMD GPUs
  through profiling, configuration tuning, hot-kernel extraction, recursive
  kernel optimization and measured end-to-end validation. Use for GEAK setup,
  vLLM, SGLang or ATOM workload optimization, AgentX trace replay, or resuming
  a GEAK e2e run. Use geak-kernel-workflow for a standalone kernel task; this
  skill is not for ordinary serving or a one-off benchmark without optimization.
---

# GEAK E2E Workflow

Bootstrap the environment, resolve the workload, launch GEAK's deterministic
workflow, and return its validated result and reusable launch bundle. GEAK owns
profiling, Amdahl triage, tuning, kernel extraction, optimization and acceptance.
Do not replace these phases with hand-written optimization in chat.

## Source of truth

Read the selected checkout's `README.md`, `e2e_workflow/README.md` and
`interface/run_e2e.md` before preparing a run. Verify interface behavior in
`interface/run_e2e.py`; `map_args` is an allowlist, not arbitrary passthrough.
Checked against [GEAK main, 001c5c7f](https://github.com/AMD-AGI/GEAK/tree/001c5c7f75df43c0c5b9c94bf980d2645e398ee9)
on 2026-11-11. Refresh these instructions from source when a newer checkout differs.

## Step 1: Prepare the checkout and Claude runtime

Follow [references/claude-setup.md](references/claude-setup.md) for first setup,
updates or CLI/SDK failures. It covers the latest branch, Python dependencies,
Claude CLI, the separate **`claude-agent-sdk`** installation, credentials, and
the CLI actually selected by the SDK. Install on the execution host/container,
using the same Python that will run `interface/run_e2e.py`.

Check these on that host before a GPU run:

- Supported AMD GPU, ROCm, `/dev/kfd` and `/dev/dri`; working `rocminfo` and
  `amd-smi` or `rocm-smi`. Detect the product and ISA rather than guessing.
- Model weights and the requested serving stack are accessible and runnable.
  GEAK does not install a matching PyTorch/Triton/serving toolchain for the user.
- A supported profiling path is usable. Let GEAK preflight record missing
  optional profiling capabilities.
- The allocated GPUs, memory and disk space fit the workload. Paths must be
  valid inside the execution container; keep results on a durable mount.

For a scheduler submission, follow the user's CPU/GPU allocation policy and
check directives and environment overrides. Keep runtime thread counts and
Docker CPU quotas independent of the scheduler's CPU request.

## Step 2: Resolve the workload and entry point

Reuse supplied values. Resolve model, framework, GPU IDs and TP/EP, synthetic
ISL/OSL/concurrency or trace-replay contract, objective, time budget and output
location. Preserve precision, launch flags/env and the baseline recipe. Ask only
for missing run-critical information. Show the resolved configuration before
launch; existing authorization to run remains valid. A setup-only request ends
after setup and validation.

| Requested run | Entry point |
| --- | --- |
| Handoff from Hyperloom/another caller; optimize an accepted configuration | `interface/run_e2e.py`, below |
| Standalone full loop including ConfigSweep, or explicit task/kernel budgets | [Full workflow](references/full-workflow.md): native Claude `Workflow` or Node runtime with Claude agents |
| AgentX replay or phase resume | Select the appropriate entry above, then read [Replay and resume](references/replay-resume.md) |

**Current interface limitation:** `run_e2e.py` fixes `config_tune="false"`,
assuming the caller already searched flags/env. The separate TuningSkillset
phase remains enabled by default. Adding `config_tune`, `budget`,
`kernel_budget` or `task` to a handoff does not forward those keys at the checked
revision. Use the full-workflow route when needed; do not silently omit a
requested configuration search or patch GEAK merely to launch it.

## Step 3: Prepare and validate an interface handoff

This is an example, not a preset. Replace paths/workload values with the request.
Keep the basename of `exp_root` exactly `geak`, under a unique experiment parent.

```json
{
  "schema_version": 2,
  "model_path": "/models/Qwen3.5-27B-FP8",
  "framework": "sglang",
  "tp": 1,
  "gpu_ids": "0",
  "workload": {"isl": 1024, "osl": 1024, "conc": 64},
  "exp_root": "/work/geak-run/geak",
  "bench_client": "native"
}
```

Save as the run's `handoff.json`. For upstream handoffs preserve
`accepted_flags`, `accepted_env`, `launch_recipe`, complete configuration and
benchmark protocol. Upstream baseline numbers are audit metadata; the headline
speedup uses GEAK's matched local validation pair.

With `GEAK_ROOT`, `GEAK_PYTHON` and `RUN_DIR` set to absolute paths:

```bash
"$GEAK_PYTHON" "$GEAK_ROOT/interface/run_e2e.py" \
  "$RUN_DIR/handoff.json" "$RUN_DIR/result.json" \
  --timeout-s 14400 --dry-run > "$RUN_DIR/dry-run.json"
```

Replace 14400 with the requested seconds. If `GEAK_E2E_TIMEOUT_S` is also set,
the **smaller** timeout wins. With neither stated the default is 12 hours.
The interface forwards this wall-clock limit to GEAK for pacing and final
validation reserve; task/direction counts are a different kind of budget.

Inspect `mapped_args`, `agent_backend`, `runtime_selection`, benchmark client
and workload exports. A GPU-less dry run may report
`gpu_identity_status=unavailable_dry_run`; a real run still probes identity.
Do not post a full dry-run document when its handoff contains sensitive env values.

For the native SDK route, clear `GEAK_AGENT_BACKEND` and `GEAK_AGENT_PROFILE`
in the launch environment and set `GEAK_AGENT_AUTO=0`. Explicit
`GEAK_AGENT_BACKEND=claude` selects the **Node runtime**, not the native SDK.
Preserve an explicitly requested runtime selection.

## Step 4: Launch and monitor

Run under a persistent job/session, keeping the SDK parent alive to completion:

```bash
"$GEAK_PYTHON" -u "$GEAK_ROOT/interface/run_e2e.py" \
  "$RUN_DIR/handoff.json" "$RUN_DIR/result.json" \
  --timeout-s 14400 > "$RUN_DIR/run.log" 2>&1
```

Use the same timeout as the dry run. Record job ID/PID, host/container, GEAK SHA,
command, handoff, log and exact `eval_dir`. Read the pinned run's disk artifacts
while active and `result.json` when available; do not identify a run by newest
directory mtime.

Check promptly that Setup starts. Periodically summarize phase, elapsed/remaining
time, baseline and accepted metric, latest gate decision and failures. Use short
log excerpts rather than streaming logs. A Claude turn saying “running in the
background” is not workflow completion. Check terminal artifacts and the owning
process before relaunching.

Carry forward only GEAK-accepted changes with measured e2e gates, engagement
checks and configured correctness checks. Keep the benchmark contract and warm
server lifecycle fixed. Isolated kernel speedup alone is not an e2e gain. Use
GEAK's reversible overlay/final bundle instead of editing installed packages.

## Step 5: Deliver the measured result

Exit code zero is not proof of improvement: it also covers `no_gain` and
`timeout`. Read status, validation evidence, metric basis and correctness together.
Report:

- Exact model/workload, GPU/TP/EP, framework, versions and GEAK SHA.
- Matched baseline/final metric, units, `metric_basis`, `throughput_speedup`,
  noise/validation limits and parity or accuracy outcome.
- Accepted configuration/kernel changes and their measured contributions when available.
- Existing `result.json`, `workflow_return.json`, `final_report.md`,
  `director_e2e_validation.json` and final launch/overlay/patch artifacts.
  Interface fields `final_launch_script`, `final_overlay`, `final_patch` may
  be empty when that deliverable does not exist.
- Validated regimes and how to reuse the final bundle. Revalidate new workloads.
  State no gain or incomplete validation plainly.

## Troubleshooting

- SDK import failure, early `workflow_parse_error`, or termination after Setup:
  check the SDK in the **execution Python/container**; follow
  [Claude setup](references/claude-setup.md). A global CLI alone is insufficient.
- `--effort ultracode` rejected or missing `Workflow`: check actual CLI and
  workflow settings. Current `run_e2e.py` translates the legacy effort default
  instead of passing an unsupported effort flag.
- GPU identity refusal: inspect `scripts/gpu_identity.py` on the target. Supply
  `expected_gfx` and `expected_target` together from observed facts. Generic
  `gfx1201` alone establishes neither product identity nor serving qualification.
- Timeout/incomplete A/B: inspect the pinned `eval_dir` and use phase resume.
  Do not promote provisional engineer results to validated gains.
