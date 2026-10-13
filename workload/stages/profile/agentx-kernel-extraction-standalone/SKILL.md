---
name: agentx-kernel-extraction-standalone
description: Extract actual GPU kernel source and workload-weighted standalone tests from AgentX replay evidence. Use with serving logs, traces and an accessible deployed GPU environment; covers phase selection, live dispatch, source binding, correctness and timing. Excludes fixed ISL/OSL benchmarking and kernel optimization unless explicitly requested.
---

# AgentX kernel extraction

This skill is self-contained. The agent performs the stages below using the
bundled tools; no external workflow runner, role system or GEAK installation is
required. Set `SKILL_ROOT` to the directory containing this file. All tool paths
are relative to it. Read [setup and inputs](references/setup.md) first.

The objective is a reviewable source-and-test delivery for the user's current
AgentX workload. Preserve the installed stack, accepted overlays, baseline,
TP layout, allocation and deployed graph/compile modes. Do not silently start
optimization or a new full replay. Missing GPU access permits useful CPU work,
but leaves GPU validation explicitly incomplete.

## 1. Establish the run and resume state

Use the completed measured baseline logs, benchmark exports, effective model
and serving configuration, workload specification and compatible raw traces.
Exclude warmup/discarded legs; record exact files, interval, rank/stream scope,
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
[selection coverage](references/selection-coverage.md).
Logs determine case distributions and sampling points; runtime traces determine
actual kernels and costs. Census every material phase and live dispatch family.
Keep request, token, call and device-time mass distinct. Sample common cases,
dispatch boundaries and tails. Target at least 90% of device-time mass per
material phase and disclose achieved coverage, approximations and missing mass.
Coverage targets do not authorize unbounded capture.

Use existing traces first. If needed, design bounded probes with explicit
request/time/byte limits using the user's actual serving launcher; follow
[capture procedure](references/capture.md). A probe's synthetic load does not
replace the original baseline weights. Preserve failed evidence; allow one
recapture per failed point before reporting the gap and continuing elsewhere.

Write `kernel_workload.json` and `kernel_ranking.json`. Distinguish stage share,
share within the callable and share the extracted source can affect; these are
nested scopes, not additive. Keep collectives, unattributed work and unsupported
kernels in the denominator and inventory. `scripts/parse_profile.py` can parse
trace windows; window percentages alone are not whole-workload percentages.

Default selection is **per-phase Top 5 AND overall GPU-time share strictly >3%**.
Rank PREFILL and DECODE separately before filtering; union by live identity,
without backfilling rank 6. Apply the same rule to other material phases in scope.
Explicit user scope overrides this default. Run:

```bash
python3 "$SKILL_ROOT/scripts/select_targets.py" \
  "$EVAL_DIR/kernel_ranking.json" --output "$EVAL_DIR/selection_targets.json"
```

Every selected identity needs a task or explicit blocker. Execution limits may
defer tasks but cannot turn a partial batch into completed extraction.

## 3. Declare complete task boundaries

Read [delivery contract](references/delivery-contract.md) before authoring.
Save `task_plan.json` with one owner per complete math boundary. For each task,
save the original declaration as `run/requested_kernel.json` (called `KERNEL`
in the bundled contract), including exact selected identities, callable
candidates, phases and `source_delivery_required=true`.

Keep required companions even when below the selection threshold. Use
`joint_ut.phases` for mandatory chains, including split-K plus reduce; use
`dispatch_ut.phases` for mutually exclusive selected branches of one whole op.
Never demand mutually exclusive kernels in every invocation. Whole fused MoE
tasks preserve routing and the complete operation, not a synthetic dense GEMM.

For selected indexer score/top-k, deliver one unified task covering both served
prefill and decode branches. Read [joint indexer UT](references/joint-indexer-ut.md).
For collectives read [multi-rank UT](references/collective-ut.md). Consult the
[GLM example](references/glm-agentx-example.md) only for matching features;
it supplies no default shapes, layer multipliers or kernel identities.

Trace live callable candidates using `scripts/seam_trace.py` and run
`scripts/kernel_selection.py --help` for the verifier. Preserve complete raw
verdicts and deeper-candidate evidence. Names alone cannot prove a binding or
that the requested whole chain executes. If the declared boundary is wrong,
repair the saved task plan with evidence before extracting; do not edit a
failed verdict into a pass or rerun an unchanged UT to repair JSON fields.

## 4. Extract source and build the standalone UT

Read [source extraction](references/source-extraction.md) and
[task authoring](references/task-authoring.md). Start by replaying one captured
baseline case per phase; retain it while adding editable leaves and full cases.
Each `tasks/<name>/` contains its own `kernel_src/`, `cases.py`, `meta.json`,
`unittest.py`, baseline manifest/overlay, captured oracles and `run/` evidence.
Copy the bundled harness/leg/overlay helpers into the task; freeze them once authored.

Deliver actual native/JIT/generated computation source for every claimed leaf,
with build resources and version/hash provenance. A copied dispatcher that
imports installed computation is not extraction of those leaves. Prove actual
leaf invocation through the candidate overlay, independently of correct output
or import location. Preserve deployed signatures, layouts, quantization,
captured routing, cache/page state and complete outputs.

Run every GPU command from its task directory under the supplied lease tools:

```bash
cd "$TASK_DIR"
bash "$SKILL_ROOT/scripts/gpu_lock.sh" "$GPU_ID" python3 unittest.py
# For actual multi-rank tests or serving captures, lease the whole allocated set:
python3 "$SKILL_ROOT/scripts/gpu_lock_set.py" "$GPU_SET" -- <foreground-command>
```

Single-lock comma lists are pools that lease one device; set locks hold every
requested device. Reconcile device IDs with physical allocation and share lock
files across participants; see setup. Serialize captures and UTs if isolation
cannot be established. No detached GPU children may outlive the lease.

The UT must cover baseline/candidate binding, frozen-oracle and random-value
correctness at served shapes, phase-specific graph/compile behavior and input
refresh, each editable leaf's numerical negative control, restored-source
integrity, and both timing legs for every positive-mass signature. Compilation
failure is not a successful numerical negative control. Report missing cases
and mass; do not renormalize survivors into full coverage.

Use `h.measure_legs()` and `h.serving_weighted_speedup()` from the bundled
harness. Set `workload.kind="agentx_trace_replay"`, `weighting="observed"`, and
`serving_weight_model.source="observed_workload"` with an evidence path.
Each case has a unique `sig`, observed `regime` and finite nonnegative
`effective_calls` on one consistent basis. Aggregate as
`sum(baseline_ms * calls) / sum(candidate_ms * calls)`; no fixed-load fallback
or weighted geometric mean. A near-1x unmodified extraction is expected.

## 5. Validate and deliver

Read [source records](references/source-delivery-records.md), freeze the task
snapshot before GPU validation and generate records from actual outcomes.
After real GPU tests, preflight the exact return with the bundled local gate:

```bash
node "$SKILL_ROOT/scripts/kernel_extraction_delivery.mjs" \
  --kernel "$TASK_DIR/run/requested_kernel.json" \
  --result "$TASK_DIR/run/draft_return.json" --phase extract \
  --out "$TASK_DIR/run/RETURN.json"
```

Use `extract_op` for the whole-op return form defined in the contract. Both
forms require actual source delivery in this skill. The gate reads bundled
schemas and hashed evidence; it does not execute or independently certify GPU
tests. Keep failures visible and continue independent tasks when possible.

Write `REPORT.md` and `results.json` with all selected identities, task/source
locations, actual commands and devices, per-gate status, declared and achieved
coverage, uncovered mass and limitations. Separate inventory completion, task
completion, GPU UT status and delivery consistency. A CPU-only check or a
subset of passing tasks is not full acceptance. Do not report E2E improvement
without a separately authorized and measured E2E result.
