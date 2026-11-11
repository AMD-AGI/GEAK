---
name: geak-kernel-workflow
description: >-
  Set up and run GEAK kernel_workflow to optimize a standalone AMD GPU kernel
  or fused operator, author an implementation against a frozen oracle, or
  compare kernel backends in bakeoff mode. Use for GEAK kernel task directories,
  kernel latency optimization, wrapper overhead tuning, or HIP/Triton/FlyDSL/CK
  implementation comparisons. Use geak-e2e-workflow for whole-model serving
  throughput optimization and reintegration into vLLM, SGLang or ATOM.
---

# GEAK Kernel Workflow

Prepare a runnable task, launch `kernel_workflow/kernel_workflow.js`, and deliver
the independently validated patch and measurements. The dispatcher and its
`kernel_lane.js` worker own Director → TechLead → optimization/verification
rounds → integration → final validation. Do not reproduce that loop in chat.

## Source of truth

Read the checkout's `README.md`, `kernel_workflow/README.md` and the task's
instructions. Use `kernel_workflow.js` as the public entry; `kernel_lane.js` is
the internal worker. For headless launches also read
`interface/runtime/engine/run_workflow.mjs` and its `registry.json`.
Checked against [GEAK main, 001c5c7f](https://github.com/AMD-AGI/GEAK/tree/001c5c7f75df43c0c5b9c94bf980d2645e398ee9)
on 2026-11-11. Follow the selected checkout when its interface changes.

## Step 1: Set up the execution environment

For first setup, updates or Claude failures, follow
[references/claude-setup.md](references/claude-setup.md). It distinguishes
the CLI from **Python `claude-agent-sdk`**: native interactive Workflow and the
Node runtime do not import the Python SDK; e2e's Python interface needs it for
reliable background-workflow lifetime. `pip install GEAK` does not install it.

Verify on the actual GPU host/container:

- ROCm, device visibility and observed GPU ISA/product; matching compiler,
  PyTorch and kernel language dependencies for this task.
- A working baseline build, correctness check and benchmark. GEAK installs
  orchestration dependencies, not every kernel's toolchain.
- Allocated GPU pool and durable experiment directory. Respect logical vs
  physical device numbering under the scheduler/container.

For scheduler submissions, follow the user's CPU/GPU allocation policy and
check directives and environment overrides. Keep runtime threads, build
parallelism and Docker CPU quotas independent of the scheduler's CPU request.

## Step 2: Prepare the task and choose a mode

`kernel_path` is a directory containing implementation, dependency information
and reproducible correctness/benchmark commands. Hand-written tasks need not
match one layout: Benchmark establishes `COMMANDMENT.md`. Extracted tasks must
retain the complete oracle, metadata, baseline sources/overlays, helper imports
and captured-data links. A copied kernel file alone may not be a runnable task.

| Mode | Use when | Required distinction |
| --- | --- | --- |
| `optimize` | Improve existing implementation | Default; preserve its true original baseline |
| `author` | Write a new implementation | Frozen runnable oracle and `target_language` |
| `bakeoff` | Compare implementation languages | Shared frozen baseline; original language always competes |

Read [references/modes.md](references/modes.md) for authoring, bakeoff,
workload-weighted scoring, multi-GPU tasks or recovery. Set `mode` explicitly
for author/bakeoff; prose such as “try HIP and Triton” does not select it.

Resolve task path, GPU IDs, mode, direction budget, focus, output root and whether
applying the patch to the original is requested. Ask only for missing constraints.
Keep `apply_to_original="false"` unless applying is within the request. Setup-only
requests stop after setup/preflight.

**Budget:** `budget=6` means six optimization directions across rounds, not
six hours or rounds. Benchmark/profile/verify/integrate/final validation are
outside this count. `deep_explore` costs `deep_cost` (default 2). Bakeoff budget
is **per lane**, so cost grows with the number of languages. A wall-clock limit
needs a separate job/supervisor deadline; `--agent-timeout-ms` is per agent,
not a total-run limit.

## Step 3: Launch the workflow

Use the full checkout: bakeoff references sibling e2e roles/scripts. Set
`GEAK_ROOT`, `RUN_DIR`, `GEAK_PYTHON` to absolute paths; use a unique run
directory. Resolve all example paths before launch.

With a native Claude `Workflow` tool, call it with an object:

```javascript
Workflow({
  scriptPath: "/work/GEAK/kernel_workflow/kernel_workflow.js",
  args: {
    kernel_path: "/work/tasks/knn",
    workflow_dir: "/work/GEAK/kernel_workflow",
    mode: "optimize",
    budget: 6,
    gpu_ids: "0",
    exp_root: "/work/kernel-run/exp",
    task: "Optimize measured latency while preserving correctness",
    apply_to_original: "false"
  }
})
```

`args` must be an object, not a JSON-encoded string. `workflow_dir` is required
even with an absolute `scriptPath`.

For a shell/agent without native `Workflow`, save that **args object only** as
`$RUN_DIR/kernel-args.json`, then use GEAK's Node host with Claude agents:

```bash
node "$GEAK_ROOT/interface/runtime/engine/selftest.mjs"
node "$GEAK_ROOT/interface/runtime/engine/run_workflow.mjs" \
  "$GEAK_ROOT/kernel_workflow/kernel_workflow.js" \
  --agent claude --args-file "$RUN_DIR/kernel-args.json" \
  --result-file "$RUN_DIR/result.json" \
  --metrics-file "$RUN_DIR/runtime-metrics.json" \
  > "$RUN_DIR/run.log" 2>&1
```

The selftest needs no GPU or key. Run the real command under a persistent
job/session; record job ID/PID, CLI/model, GEAK SHA, task identity, args and log.
Do not run `node kernel_workflow.js` directly: it needs injected Workflow globals.

When the user specifies a Git identity, propagate it to GEAK's workspace
commits. Role examples may contain `team@workflow`; see the environment-override
mechanism in [Claude setup](references/claude-setup.md). Check effective author
and committer identities in the environment that actually commits.

## Step 4: Monitor and preserve verification

Confirm Setup/Benchmark start and baseline measurements exist. Periodically
report round, directions used, current independently verified metric, latest
gate and compilation/correctness failures. Use short log excerpts. Check the
parent and GPU workers before relaunching a stopped-looking run.

Let GEAK use `kernel_workflow/scripts/gpu_lock.sh` for measurements. This lock
serializes cooperating GEAK jobs, not arbitrary GPU users. Preserve the baseline,
oracle, case set and benchmark contract. Never alter reference, tolerances,
seeds or timer to make a candidate pass. Claims need independent verification.

Default primary scoring is geometric-mean baseline/candidate latency. With a
workload spec use returned workload-weighted `final_speedup` and report
`final_geomean` separately. Check `timing_basis` before calling a host/runtime
latency gain a device-execution gain.

## Step 5: Deliver the validated kernel

Read `result.json` or the native return and its exact `eval_dir`. Default lane
outputs are under `<exp_root>/team_<kernel>_<timestamp>/<kernel>/`; use returned
paths rather than constructing them for bakeoff or custom runs.

Report task/GPU/mode, baseline and final per-case latency, primary speedup,
correctness/validation status, rounds and directions used. Link existing
`tech_lead_report.md`, `director_validation.json`, `final_patch.diff` and
`optimized/`. The final Director result takes precedence over engineer claims
and preliminary TechLead estimates.

For bakeoff report `winner`, comparable candidates and winner kind. An env-tune
winner is delivered as `apply_env` plus its tuning artifact and may have no
source patch. `winner=null` / `no_winner` means no accepted improvement. State
failed or incomplete validation plainly even if a promising patch exists.
Applying/deploying follows the user's authorized scope.

## Troubleshooting

- Missing paths: check absolute `workflow_dir`/`kernel_path` and the args object;
  `--args-file` contains args alone, not the `Workflow` envelope.
- Broken baseline: fix its declared dependency/build setup first. A new
  implementation cannot replace the original performance denominator.
- `author_failed`/`freeze_failed`: retain complete oracle and real baseline;
  do not substitute an easier synthetic reference.
- CLI/SDK/auth failure: follow [Claude setup](references/claude-setup.md) in the
  execution environment. Do not print credentials or reroute providers silently.
