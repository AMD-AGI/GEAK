---
name: kernel-extract
description: Extract actual GPU kernel source and workload-weighted standalone tests from AgentX replay evidence. Use with serving logs, traces and an accessible deployed GPU environment; covers phase selection, live dispatch, source binding and correctness qualification without timing. Excludes fixed ISL/OSL benchmarking and kernel optimization unless explicitly requested.
---

# AgentX kernel extraction

This is the legacy combined workflow and the shared implementation bundle. New split
scenarios use [kernel-profile](../kernel-profile/SKILL.md) for runtime evidence and
[kernel-ut](../../ut/kernel-ut/SKILL.md) for template-based generation/qualification.
Follow the selected skill's boundary; UT-only invocations do not run the capture stages.

The agent performs the five stages below using this skill's method and the
shared tools bundled with it. Set `SKILL_ROOT` to this directory and `SHARED_TOOLS`
to `$SKILL_ROOT/shared`. Read [setup and inputs](references/setup.md) first.
When run as the pipeline's `profile` stage, also read
[pipeline handoff](references/pipeline-handoff.md) for input paths and outputs.

The objective is a reviewable source-and-test delivery for the user's current
AgentX workload. Preserve the installed stack, accepted overlays, baseline,
TP layout, allocation and deployed graph/compile modes. Do not silently start
optimization or a new full replay. Missing GPU access permits useful CPU work,
but leaves GPU validation explicitly incomplete.

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

This stage is a fixed template. **The inputs you write are `serving.json` (step 1),
the runtime identity for shared multi-config sampling, and the review of
`task_decls.json` (§3).** Everything else is produced by the scripts
below; do not hand-write census, rank-merge, ranking, shape-manifest, source-map or
task-plan scripts -- every run that did produced numbers no other run could compare
with, and spent most of the hour doing it.

1. **Declare the deployment.** `python3 "$SKILL_ROOT/scripts/probe_server.py" init --out
   "$EVAL_DIR/serving.json"`, then copy model, TP, GPUs, flags and environment from the
   baseline's launch exactly. Leave `decode_capture` at `profile_cuda_graph`.
2. **Observe and plan.** Use `observed_workload.py` for each supplied config, then
   the [shared sampling workflow](references/multi-config-sampling.md):
   `multi_config.py plan` → `capture` → `project --merge` → `select`.
   Supply one `--config` for a single workload. Pin its runtime identity and pass
   the scenario's global sampling budget. Compatible configs share observed points;
   projection retains each config's own frequencies and uncovered mass. Lowest/highest
   concurrency do not substitute for the middle configs' dispatch evidence.
3. **Capture and freeze profile evidence.** The capture helper holds each pool's
   GPU-set lease and runs `probe_server.py` / `run_points.py`: prefill on the graph
   server, decode normally on the eager server, per-point census, achieved-shape
   relabeling, ranking and dispatch/source evidence. Resume reuses signed censuses;
   `--redo` requests recapture. Failed or hidden windows stay explicit gaps.
   `compile_watch.py` records startup, warmup and graph-capture compile variants and
   call descriptors as well as timed-window evidence. Profile stops at that evidence;
   it never drafts UT declarations automatically.
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
Every selected identity needs a task or explicit blocker; execution limits may defer tasks
but cannot turn a partial batch into completed extraction.

## 3. Declare complete task boundaries

Read [delivery contract](references/delivery-contract.md). Start this UT step explicitly:

```bash
python3 "$SKILL_ROOT/scripts/draft_task_decls.py" --eval-dir "$EVAL_DIR" \
    --out "$EVAL_DIR/task_decls.json"
```

`task_decls.json` is drafted
from the evidence: one task per selected operator with all its member kernels and its
other phase branch; its `target_callable` is the operator's entry (or the branches'
innermost common frame); its `ut_family` is the family whose `match` patterns fit (the
module class breaks ties); a collective is a task on the deployment's world size
(`required_gpu_count` = tp) whose lane builds the multi-rank harness, and only device
copies are excluded with a stated reason; each phase gets one `joint_phases` row per
distinct set of the task's kernels launched together (two rows in one phase are dispatch
branches). Every decision carries its basis and a `needs_review` note. Review it:

- **Boundary.** One owner per complete math boundary. Keep required companions (a
  dispatch-branch sibling or a mandatory chain member below the threshold) under
  `companions` with a reason. Mandatory chains, including split-K plus reduce, are one
  task; mutually exclusive branches are separate `joint_phases` rows, never demanded in
  every invocation. Whole fused MoE keeps routing and the complete operation.
- **Merging across configs (a multi-config group).** After the initial declaration
  review, follow [the multi-config UT merge procedure](references/multi-config-ut-merge.md)
  before authoring. It first builds the provisional plan consumed by `merge_policy.py`,
  then applies the policy and rebuilds the final plan. Preserve every dispatch branch
  and record each application in `decisions.json`.
- Selected indexer score/top-k is one task covering both served prefill and decode
  branches ([joint indexer UT](references/joint-indexer-ut.md)); collectives follow
  [multi-rank UT](references/collective-ut.md). The [GLM example](references/glm-agentx-example.md)
  only for matching features; it supplies no default shapes or identities.
- **Family and axes.** Confirm `ut_family` (or `custom` with `ut_family_why`), pin the
  family's const `axes`/`attrs` and write `case_axes` per phase from the host-op dims in
  `evidence/shapes` (expressions in `shape` and `context`). `build_task_plan.py --check`
  lists exactly what is still missing.
- Empty each `needs_review`, then:

  ```bash
  python3 "$SKILL_ROOT/scripts/build_task_plan.py" --eval-dir "$EVAL_DIR"
  ```

  It lints the declaration (every selected identity owned once, joint rows consistent
  with where kernels launched, nothing left unreviewed) and writes `task_plan.json`,
  each task's `run/requested_kernel.json` and `run/ut_case_plan.json` (point cases plus
  bucket cases for the bands no point covers, without double-weighting), and scaffolds
  the UT from its family. An existing `unittest.py` is never overwritten without `--force`.
  It also slices the compile evidence per task into `run/server_variants.json` and
  `run/server_calls.json` (structural operands inline; bulk operands as seeded descriptors, so
  no tensor file -- `run/server_calls/*.pt` only for a call recorded before this, or above
  COMPILE_WATCH_INLINE_CAP) and sets `meta.server_preimports`
  (decls `server_preimports`; sglang defaults to `sglang.srt.utils.common`).

The source map's frames prove a call reached a file on this deployment; they do not prove
a binding. Verify the live callable with `$SHARED_TOOLS/seam_trace.py` and
`python3 "$SHARED_TOOLS/kernel_selection.py" --strict-launch` (`--help` for its inputs),
keeping complete raw verdicts. If the declared boundary is wrong, fix `task_decls.json`
with the evidence and rebuild; do not edit a failed verdict into a pass.

Stop the probe server as soon as the evidence is collected. The CPU half of §3–§4
(declaration review, source resolution inside the serving container, scaffold, case
manifest, CPU lint) needs no GPU; run GPU gates only once the server's devices drained.

**Then write the UTs in parallel, not one after another.** Once `build_task_plan.py` has
written the tasks, hand them to the UT wave; do not author them yourself in this session:

```bash
python3 "$SKILL_ROOT/scripts/ut_wave.py" plan  --eval-dir "$EVAL_DIR"        # lanes and GPUs
python3 "$SKILL_ROOT/scripts/ut_wave.py" start --eval-dir "$EVAL_DIR" --runner "$RUNNER" \
    --backend docker --image <the serving image>
python3 "$SKILL_ROOT/scripts/ut_wave.py" wait  --eval-dir "$EVAL_DIR" --timeout 540   # repeat
```

One CPU agent lane per task writes and fixes its own task directory
([UT lane](references/ut-lane.md)); `--author-workers` controls author concurrency.
Default `--gpu-mode on-demand` leases GPUs only for each `ut_wave.py gpu -- <command>`.
The pool is the deployment's physical GPUs (`serving.json` `gpus`); release the probe
server and any outer campaign GPU lock first. Each command leases `required_gpu_count`
cards together (normally 1), drains its workers, then returns them to the pool. CPU
writing/lint holds no GPU. The default `--timing none` qualifies all non-timing
gates with `unittest.py --validation-mode correctness --reuse`; no timing tail runs.
Preserve weighted shapes and timing support for the optimizer. An explicitly requested
performance wave can use `--timing tail --timing-workers N` on disjoint GPU sets after
authoring drains; multi-rank tasks retain an exclusive tail.
`wait` returns 0 when every task is `passed`, `blocked` or `failed`; read `ut_wave/summary.json`
and the lane logs it names, fix nothing in a task yourself, and carry blocked and failed
tasks into the report with their reasons. `start` again resumes an interrupted wave.

## 4. Extract source and build the standalone UT

Read [source extraction](references/source-extraction.md) and
[task authoring](references/task-authoring.md). Default to fixed-seed generated inputs
at shapes/dtypes/layouts established from existing workload evidence. Run one such
baseline case per phase, then expand cases and editable leaves. The UT reads its
frozen case manifest; it never launches serving, profiles, or captures shapes.
Each `tasks/<name>/` contains its own `kernel_src/`, `cases.py`, `meta.json`,
`unittest.py`, baseline manifest/overlay and `run/` evidence. Captured oracles are
optional; preserve minimal structural fixtures only where needed.
Copy the shared harness/leg/overlay helpers into the task; freeze them once authored.

Build each task from its UT contract family: read [UT contracts](ut_contracts/README.md),
create the task with `ut_contracts/scaffold_ut.py new`, and fill only what the family
leaves open (the live `call`, the family's hooks, the case manifest, a replay bundle for
graph phases, negative controls). The family fixes operand distributions, layout facts and
each output's error bound; `unittest.py` runs every gate through `ut_driver.py`. Run the
task's CPU lint (`python3 ut_contract.py <task>`) before any GPU session. Depart from a
family only by an override with a stated reason; for an operator no family covers, write a
contract in the same format (`--family-file`) rather than a free-form harness.

Deliver actual native/JIT/generated computation source for every claimed leaf,
with build resources and version/hash provenance. A copied dispatcher that
imports installed computation is not extraction of those leaves. Prove actual
leaf invocation through the candidate overlay, independently of correct output
or import location. Preserve deployed signatures, layouts, quantization,
routing semantics, cache/page state and complete outputs.

Run every GPU command from its task directory under the lease tools in
[setup](references/setup.md) — `gpu_lock.sh` for one device, `gpu_lock_set.py` for the
whole allocated set outside a wave. Inside an on-demand UT lane use `ut_wave.py gpu --
<command>`; its host broker owns the lease and returns GPUs after worker cleanup.
Never nest the old lock wrappers inside it. `GPU_IDS` is populated only in GPU commands.

The UT must cover baseline/candidate binding, frozen-oracle and random-value
correctness at served shapes, phase-specific graph/compile behavior and input
refresh, each editable leaf's numerical negative control, restored-source
integrity. UT extraction does not measure timing: use the explicit correctness
validation mode, retain every weighted signature, and report performance as unmeasured.
Optimization later measures both timing legs for every positive-mass signature. Compilation
failure is not a successful numerical negative control. Report missing cases
and mass; do not renormalize survivors into full coverage.

The UT must also cover what the server compiled, in the server's import environment:

- **server_preimports.** Both legs import `meta.server_preimports` before the cases. A
  serving framework can rebind library functions at import (sglang replaces
  `triton.next_power_of_2`, a constexpr function, with a plain one); a kernel calling it
  in a jit body is green in a bare UT and fails in the server. The lint rejects an
  sglang/vLLM entry without preimports.
- **server_calls gate.** Every recorded entry call is replayed in both legs: integer and
  small operands (sequence bookkeeping, page tables, scales) exactly as the server passed
  them, bulk payloads from a seed. The candidate must run every call the baseline runs and
  match it. This is where a branch only warmup reaches (one token, no cache) is tested.
- **compile_variants gate.** Each Triton variant the server built for the task's kernels
  must also be built by the baseline legs, and the legs' Triton/torch namespaces must match
  the server's. A missing variant is reported with the fields that differ from the nearest
  UT variant; fix it with a case (the server call is in `run/server_calls.json`) or waive it
  in `meta.ut_contract.compile_variant_waivers` with a reason. Optional arguments in
  generated cases (descale, sinks, None-ness, index dtypes) come from these server calls,
  not from defaults.

Before reporting a verdict, verify deterministic input construction, baseline
self-noise and that numerical bad-source controls actually fail. For generated
cases, obtain reference outputs from the frozen baseline in a separate process
using the same seeds; never use the candidate to generate its own expected output.
Do not create a serving capture merely to obtain a golden file. With persisted
capture oracles, also run `oracle_power.py`; regenerate `structure_only` placeholders
before checking them. See [task authoring](references/task-authoring.md).

Randomize ordinary numerical payloads, not the workload definition. Keep observed
shape/layout/quantization, phase and call weights. Routing, indices, page tables and
sequence metadata must retain valid semantics and the observed work/access pattern:
reuse or reconstruct them from existing metadata where reliable; capture only the
missing structural state. Synthetic structural cases may extend correctness coverage,
but do not claim they reproduce serving performance without evidence. Any necessary
capture uses `CAPTURE_PERSIST_POLICY=structure_only` unless explicitly overridden.

Use `h.measure_legs()` and `h.serving_weighted_speedup()` from the shared
harness (the driver's timing gate does). Set `workload.kind="agentx_trace_replay"`, `weighting="observed"`, and
`serving_weight_model.source="observed_workload"` with an evidence path.
Each case has a unique `sig`, observed `regime` and finite nonnegative
`effective_calls` on one consistent basis. Aggregate as
`sum(baseline_ms * calls) / sum(candidate_ms * calls)`; no fixed-load fallback
or weighted geometric mean. A near-1x unmodified extraction is expected.

## 5. Validate and deliver

Read [source records](references/source-delivery-records.md), freeze the task
snapshot before GPU validation and generate records from actual outcomes.
After real GPU tests, preflight the exact return with the shared delivery gate:

```bash
node "$SHARED_TOOLS/kernel_extraction_delivery.mjs" \
  --kernel "$TASK_DIR/run/requested_kernel.json" \
  --result "$TASK_DIR/run/draft_return.json" --phase extract \
  --out "$TASK_DIR/run/RETURN.json"
```

Use `extract_op` for the whole-op return form defined in the contract. Both
forms require actual source delivery in this skill. The gate reads workflow
schemas and hashed evidence; it does not execute or independently certify GPU
tests. Keep failures visible and continue independent tasks when possible.

Write `REPORT.md` and `results.json` with all selected identities, task/source
locations, actual commands and devices, per-gate status, declared and achieved
coverage, uncovered mass and limitations. Separate inventory completion, task
completion, GPU UT status and delivery consistency. A CPU-only check or a
subset of passing tasks is not full acceptance. Do not report E2E improvement
without a separately authorized and measured E2E result.
