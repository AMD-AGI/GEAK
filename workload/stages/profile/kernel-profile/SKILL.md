---
name: kernel-profile
description: Profile one serving config's GPU kernels from AgentX replay evidence -- sampling, live capture, census, ranking and selection -- and deliver kernels.json, each selected kernel's source path, served shapes and workload weights. Excludes task boundaries, UT authoring and optimization.
---

# AgentX kernel profile

One config, one deployment. The agent performs the steps below with this skill's
scripts. Set `SKILL_ROOT` to this directory and `SHARED_TOOLS` to `$SKILL_ROOT/shared`.
Read [setup and inputs](references/setup.md) first. When run as the pipeline's `profile`
stage, also read [pipeline handoff](references/pipeline-handoff.md) for inputs and outputs.

The objective is a measured account of what this deployment's GPU time is spent on and
where each selected kernel is launched from: `kernels.json`. Preserve the installed stack,
accepted overlays, baseline, TP layout, allocation and deployed graph/compile modes. Do
not silently start a new full replay. Missing GPU access permits useful CPU work, but
leaves capture explicitly incomplete. Several configs are several profile calls; a
capture of an identical deployment can be reused (`plan_points.py --reuse-capture`).
This skill decides no task boundary and writes no UT.

For vLLM, first read [vLLM observation and profiling](references/vllm.md).
Its throughput logger and request/profiler APIs differ from the SGLang defaults.

## 1. Establish the run and resume state

Use the completed measured baseline logs, benchmark exports, effective model
and serving configuration, workload specification and compatible raw traces.
Exclude warmup/discarded legs from cost and weight measurement (the compile evidence
of §2 keeps them: warmup is where a server first compiles); record exact files, interval, rank/stream scope,
image/packages, graph modes and accepted overlays in `environment.json`.
Resolve the actual installed sources in that serving environment, not an
assumed host checkout. Source trees and build resources may live inside its container.

Create `EVAL_DIR` outside this skill. Keep `PROGRESS.md` with completed stages,
evidence paths and unresolved items; reuse compatible captures and passing
checks on resume. Changed code, inputs, entry or deployment mode invalidates
the affected checks. Ask only for missing environment/data/allocation needed
for the next dependent step while continuing independent inspection.

## 2. Measure the workload and select targets

Read [workload measurement](references/dynamic-workloads.md) and
[selection coverage](references/selection-coverage.md). Logs determine case distributions
and sampling points; runtime traces determine actual kernels and costs. Keep request,
token, call and device-time mass distinct; target at least 90% of device-time mass per
material phase and disclose achieved coverage, approximations and missing mass. Counts
come from the serving log. Missing observation logs are an explicit evidence gap.

This stage is a fixed template. **The input you write is `serving.json` (step 1).**
Everything else is produced by the scripts below; do not hand-write census, rank-merge,
ranking, shape-manifest or source-map scripts -- every run that did produced numbers no other run could compare
with, and spent most of the hour doing it.

1. **Declare the deployment.** `python3 "$SKILL_ROOT/scripts/probe_server.py" init --out
   "$EVAL_DIR/serving.json"`, then copy model, TP, GPUs, flags and environment from the
   baseline's launch exactly. Leave `decode_capture` at `profile_cuda_graph`.
2. **Observe and plan.** `observed_workload.py`, then `plan_points.py`
   ([sampling](references/sampling.md)) with the scenario's sampling budget. To reuse the
   captures of another profile of the identical deployment, pass `--reuse-capture`.
3. **Capture and freeze profile evidence.** Under the allocated GPU-set lease,
   `probe_server.py run ... -- run_points.py` captures the planned points: prefill on the graph
   server, decode normally on the eager server, per-point census, achieved-shape
   relabeling, ranking and dispatch/source evidence. Resume reuses signed censuses;
   `--redo` requests recapture. Failed or hidden windows stay explicit gaps.
   `compile_watch.py` records startup, warmup and graph-capture compile variants and
   call descriptors as well as timed-window evidence. Profile stops at that evidence.
4. Write `kernel_workload.json` from those artifacts (coverage, phase split, sampling
   plan, approximations). Distinguish stage share, share within the callable and share
   the extracted source can affect; nested scopes, not additive. Collectives,
   unattributed work and unsupported kernels stay in the denominator and inventory.

Default selection is **per-phase Top 5 AND overall GPU-time share strictly >3%, per
operator**: `run_points.py` resolves the launch stacks of every kernel above 0.1% or in a
phase's top 30 (`attribution_candidates.json`), sums kernels under their operator (library
entry @ innermost `nn.Module`, `source_map.py`), and applies the rule to the sums: ranked
per phase before filtering, no backfilling of rank 6. Every member kernel of a selected
operator is selected, and so is the operator's other phase branch (same module class and
family, disjoint phases). By kernel name, GLM-5.2's fused MoE lost its stage-2 and sort
kernels (6.3%) and the indexer its top-k and prefill score. `--select-by-kernel` restores
the per-name rule; explicit user scope overrides either (`run_points.py --select-arg`).
Every selected identity appears in `kernels.json` with its gaps; which kernels become one
UT is decided downstream, never by dropping an identity here.

## 3. Deliver kernels.json

Stop the probe server as soon as the evidence is collected. Write `REPORT.md` with the
sampling plan, achieved coverage per phase, uncovered mass, failed or hidden windows and
the selection. Then the slot's finalizer checks the configured selection scope, the
denominator and the inventory sections (`selection_scope`, `candidate_inventory`,
`coverage_summary` in `kernel_ranking.json`, see [selection coverage](references/selection-coverage.md))
and exports `kernels.json` (`stages/profile/kernels.py`):

- `operators[]`: each selected operator, its member kernels, entry and share;
- `kernels[]`: each selected kernel's `path` (entry callable, entry file and SHA-256,
  launching ops, resolved serving-stack frames), `points` (phase, achieved shape and
  context, represented instances, launches and device time per instance, grid/block,
  host-op operand dims), `weight` (launches over the measured workload, whole-workload
  and per-phase share) and `gaps`;
- `denominator`, sampling gaps, and the SHA-256 of every evidence file it summarizes.

```bash
python3 "$WORKLOAD_ROOT/stages/profile/kernels.py" export --profile-dir "$EVAL_DIR" \
    --params <selection params.json>      # outside the pipeline
python3 "$WORKLOAD_ROOT/stages/profile/kernels.py" verify --kernels "$EVAL_DIR/kernels.json"
```

`verify` proves the summary still describes unchanged evidence. Report profile sampling
coverage and approximations; never report a UT, a task or an E2E result from this stage.
