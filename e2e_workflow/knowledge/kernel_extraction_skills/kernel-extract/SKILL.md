---
name: kernel-extract
description: Select top GPU kernels, extract editable source and build workload-weighted standalone UTs for AgentX trace replay across models and backends. Excludes fixed ISL/OSL and non-AgentX runs.
---

# AgentX kernel extraction

Apply only to AgentX replay in `kernel_extractor` (`select`, `extract` and
`extract_op`), `system_architect` and `op_benchmarker`, including refreshes and
resumes. Legacy profiler artifacts remain usable as evidence. Activation is
independent of roofline and expert-skill toggles.
Fixed ISL/OSL and other non-AgentX runs retain their existing behavior.

Preserve GEAK's role schemas, frozen serving baseline, launcher verification,
correctness, graph/compile mode, TP/GPU contracts and configured benchmark metric.
Use the current accepted stack and actual replay load. Put probes and evidence
under `EVAL_DIR`; this skill does not add backend support or change the load.

The workflow is: configs/logs → phase distributions and trace sampling plan →
measured prefill/decode kernel inventory → scope and scheduling → live
source/dispatch identification → complete case signatures → executable UT with
independent candidate-call proof.
Use other phase names when the workload has them. Configs/logs guide sampling;
runtime traces establish kernel identity and cost.

Read [selection scope and coverage](references/selection-coverage.md) in `select`,
when reconciling prior UT deliveries, or when assessing extraction completeness.
Discovery, execution budgets, per-UT correctness and workload coverage are
separate contracts. A small successful validation batch does not establish that
the kernel inventory or the requested extraction scope is complete.
Read the [delivery and workflow acceptance contract](references/delivery-contract.md)
before declaring task boundaries in `select` and before authoring or returning
an `extract`/`extract_op` result. It defines machine evidence transport, joint
versus per-case dispatch, and a CPU preflight using the workflow's actual gate.

## Entry: baseline logs directly to kernel extraction

After baseline completes, `kernel_extractor` in `PHASE=select` owns discovery
and candidate selection. There is no standalone Profile or Strategize pass.
Read `BASELINE_LOG`, `BASELINE_BENCH_DIR`, `BASELINE_SUMMARY`, effective serving
configuration and model metadata first. Locate the completed measured leg in
the benchmark exports; exclude warmup/discarded legs from the workload census.
If a supplied path differs from the harness layout, resolve it from the baseline
summary and launch/benchmark records. Record the exact files and time interval.
Read `TASK` and any user-supplied UT deliveries before choosing the selection
scope. Their manifests and current handoff notes provide discovery leads, not
current costs, dispatch proof or permission to reuse old validation results.

Use that evidence to identify material phases and likely hot call sites, then
inspect the installed container source to identify shape-dependent dispatch
branches. Reuse compatible traces/captures already on disk. When costs, paths
or operands are missing, take bounded, targeted captures selected from the
baseline distribution. Do not run another full AgentX replay, an outer warmup,
or a generic full-model profile to construct a Top-N. Logs select sampling
points and provide weights; they do not prove GPU time or executed kernel names.

Apply the measurement and selection sections below within this skill. Write
`kernel_workload.json` and `kernel_ranking.json` under `EVIDENCE_DIR`; return
their paths as `profile_workload_json` and `kernel_ranking_json`, together with
`head_candidates`, `kernel_candidates`, `config_directions`, `drop_list` and
`strategy_path`. Use the existing candidate contracts in
`SKILL_DIR/roles/system_architect.md`, including exact `device_kernel`,
`entity_kind`, live seam evidence and backend capabilities. Do not run a separate
architect pass or author UTs in `select`; the workflow invokes `extract_op` or
`extract` for the selected candidates, allowing batch UT authoring. Missing evidence
keeps a candidate in the diagnostic/drop list, with the reason, rather than
giving it invented costs or an executable queue entry. Preserve the complete
inventory and each disposition in `kernel_ranking.json`, including candidates
deferred by budget and material phase leaders below optimization thresholds.
Use the architect's optimization policy for scheduling; it does not limit
discovery or define acceptance for a requested UT extraction/validation task.

After accepted changes, read the current stack's existing benchmark logs under
`EVAL_DIR` and compare them with the baseline and previous evidence links.
Invalidate costs, dispatch identities or weights affected by the change and
refresh only those points. Baseline evidence remains the starting point, not
proof of an unchanged current stack. On resume, reuse compatible saved evidence
and candidates; do not restart a full profile or repeat a completed baseline.

## Establish workload evidence

Read the workload specification, effective launch/model configuration, benchmark
exports, logs, traces and captured operands. Record sources for:

- Arrival and warmup policy, observed batch distribution and active phases.
- Shapes, dtypes, strides, quantization, routing and other dispatch-sensitive state.
- Cache behavior, speculative execution, active layers and parallel sharding.
- Backend/version, device, graph/compile settings and actual dispatch branches.

Apply feature-specific checks only where relevant. Unknown is not false; mark
irrelevant features not applicable. Derive dimensions, call counts and phase
boundaries from this run, including separate target/draft architectures when
present. For an unfamiliar model, use a bounded representative capture to
establish these facts before ranking or extracting.

Read [workload measurement](references/dynamic-workloads.md) when designing
captures or folding a sampled distribution. Consult the optional
[GLM example](../kernel-extract-glm-agentx-example/SKILL.md) only for matching
features; it supplies no default shapes, multipliers or kernel identities.

## Measure and rank within extraction

1. Census cases by phase, live call site and operand/state signature. Reconcile
   effective configuration with observed execution, including graph padding and
   eager fallbacks. A capture table or one profile window is not the distribution.
2. Sample material phases, served dispatch boundaries and tails. Target at least
   90% of measured device-time mass in each material phase; disclose achieved
   coverage and uncovered mass. Keep request, token, call and time weights distinct.
   Record per-point ISL/OSL, concurrency, cache preparation and profile-window
   controls with expected/observed shapes in the workload artifact's sampling plan.
   Coverage is a reporting target, not permission to launch a full replay or
   unbounded capture sweep. Report gaps when targeted probes cannot close them
   within the available budget.
3. Verify phase identity, expected progress/calls and per-rank trace integrity.
   Record each check's applicability and `pass`, `fail` or `unanswerable` result.
   Diagnose missing graph work with a short eager capture if needed; final timing
   uses deployment mode. Preserve failed evidence and allow one recapture per
   failed point, then report the gap.
4. Resolve device kernels to live launchers using correlation, host nesting,
   operands, grids and dispatch evidence. Use `scripts/parse_profile.py`,
   `scripts/seam_trace.py` and `scripts/kernel_selection.py`, or equivalent
   backend probes inside the actual serving container/environment. Inspect the
   installed source and accepted overlay, not an assumed host checkout. Names
   alone cannot identify a seam or establish editability.
5. Fold per-case costs over observed effective calls. Complete counts need no fit;
   sampled models require supported axes, branch-aware fits, residual checks and
   disclosed extrapolation. A symbol absent after a dispatch switch is not a
   zero-cost observation of the logical operation.
6. Keep summed kernel device time, union busy time and wall time separate, with
   consistent rank/stream scope. Preserve collective and unattributed mass.
   Check ledger closure and independently check deployment representativeness
   where evidence permits; neither check substitutes for the other.

Write round-local `kernel_workload.json` with feature evidence, distributions,
call/phase weights, coverage, integrity and limitations, plus
`kernel_ranking.json` with device/seam/case identities and attributed shares.
For each candidate distinguish `share_of_stage` (logical operation),
`share_in_seam` (launches reached by the seam) and `share_editable` (work the
source change can affect). These scopes are nested, not additive. Express
shares in percent with explicit denominators.
Report leading kernels separately for each active phase, their within-phase
shares and their workload-weighted total-device-time shares, with coverage and
estimation limits. A decode-only ranking cannot stand in for the whole workload.

When trace-window `profile_topN.json`/`.md` already exists or a targeted capture
produces it, preserve it and parser `pct_gpu_time` as trace-window evidence;
these files are not prerequisites and must not be fabricated from logs.
Attach `kernel_extraction` to each candidate (and any trace row) with `workload_json`,
`ranking_json`, `profile_pct_gpu_time`, the three shares and
`selection_pct_gpu_time`: the measured share the proposed change can affect,
converted to total GPU device-time percent. Return `profile_workload_json`
pointing to the workload artifact. For deep reprofiles returning `heads`, use
`selection_pct_gpu_time` as each head's `pct_gpu_time`. Retain diagnostic rows
with missing required evidence as ineligible rather than routing them.

## Select candidates and schedule work

The default extraction rule is **per-phase Top 5 AND overall share > 3%**:
rank PREFILL and DECODE separately by their within-phase GPU device time,
take each phase's first five kernels, then retain only kernels whose
`selection_pct_gpu_time` over the entire workload is strictly greater than 3.
Union the two retained sets by live kernel identity. This is not global Top 5,
not Top 5 OR >3%, and not five tasks per phase regardless of share. Other
observed phases use the same rule when in scope; explicit user scope wins.

Inventory the material families in every active phase before applying task
budgets. Reconcile supplied UT deliveries by live call site, phase, operands
and dispatch; record overlap between leaf and joint tasks. Account for every
ranked row, using an explicit residual group for the small tail. A missing
source mapping remains an unresolved inventory entry, not an omitted kernel.

Run `scripts/select_targets.py` from this skill against `kernel_ranking.json`
and write `selection_targets.json` under `EVIDENCE_DIR`; see the selection
reference for its command and input contract. Map every selected identity to
an extraction candidate or a specific blocker. Neither a low optimization
ceiling nor a small execution budget removes a required extraction target.
Unsupported selected kernels remain visible; do not silently skip them and
promote rank 6. The old optimization 5% gate must not filter this target set.

Return all evidenced extraction targets in the candidate queues, without
truncating by task budget; keep other discovered kernels in the inventory.
Optimization headroom may order subsequent optimization, but cannot replace
phase-time ranking for extraction. Keep out-of-scope, unsupported and unresolved
entries in the inventory with distinct reasons.
For a requested extraction/UT campaign, low expected optimization gain is not
an extraction failure; use the requested coverage or count and report any
scheduler limit that prevents it. Do not silently raise budgets or lower the
production optimizer's thresholds. Bound source rewrites by `share_editable`;
establish which launches a backend/table change can affect separately. Carry
`kernel_extraction` into each candidate and set its `pct_gpu_time` to
`selection_pct_gpu_time` for scheduling, thresholds and Amdahl calculations.
An optimization-only caller may schedule a subset under its existing policy;
that subset is not completion of the skill's extraction target set.

Keep distinct phase/dispatch identities. Roofline cannot supply missing workload
weights. E2E projections must account for uncovered work, host/idle time,
communication, overlap and scheduling feedback, using the configured metric and
aggregation. A device-time speedup alone is not an e2e result.

## Turn selected kernels into complete UT tasks

Selection ranks device kernels; delivery follows a complete callable boundary.
Record selected kernel IDs separately from task membership so a fused or joint
task can cover several targets without counting the same work twice. Include
required companion kernels/glue even when those companions fail the selection
threshold: the threshold chooses targets, not fragments of a math contract.
Return one candidate per `ut_task` in exactly one queue, with
`selected_device_kernels` and `extraction_members` retaining each selected
member's identity, served regimes and original `kernel_extraction` cost evidence.
Put a joint task's complete `joint_ut.phases` declaration in that candidate,
not only in a prose plan or the ranking file. A task ID alone does not tell the
caller which branches or member kernels must pass. See the joint reference
for the declaration and validation fields.
This applies to any mandatory chain, including split-K plus reduction, not just
indexers. A multi-kernel implementation alone does not require `joint_chain`:
fused operations with alternative dispatch retain whole-op UTs and per-case
kernel checks as specified in the delivery contract.
Use `dispatch_ut.phases` to declare alternative selected branches when a
whole-op task contains multiple selected kernels. For requested actual source
delivery, set `source_delivery_required=true` and preserve the requested leaf
list; the delivery contract defines the records checked against current files.

For an indexer score/top-k target, deliver **one unified indexer UT task** across
its served prefill and decode branches, following
[joint indexer validation](references/joint-indexer-ut.md). Its scored boundary
is query/key inputs through scores and the deployed selection/transform glue
to final indices. Separate score-only/top-k checks are diagnostics inside this
task, not independent substitutes for joint correctness or chain timing.
Honor a requested wider preamble boundary explicitly. Do not silently import
an old delivery's optimized candidate as the current baseline.
For a staged authoring call, honor `AUTHORING_STAGE`: the replay bootstrap
proves one installed-callable case per declared phase before the full stage
adds editable leaves, complete cases and all GPU gates. Bootstrap readiness is
an intermediate handoff, never a successful full UT.

## Kernel extractor: deliver an executable UT

UTs may be authored and reviewed in batches before GPU validation. Keep each
task's source, cases, baseline and generated build artifacts in its own directory.
Review the live seam, complete case signatures and weights, baseline isolation
and actual candidate execution. A written UT remains unvalidated until its own
GPU tests pass; only passing UTs may enter downstream optimization.

GPU validation may run concurrently only on exclusively leased, disjoint
physical GPUs. Shared or overlapping GPU sets require serial GPU execution;
authoring does not need to wait. For every single-GPU validation, compile,
benchmark or probe, run from the task directory through the existing lock:

```bash
cd "$TASK"
bash "$SKILL_DIR/../kernel_workflow/scripts/gpu_lock.sh" "$GPU_ID" python3 unittest.py
```

Use the same wrapper for other GPU commands, including `op_bench.py`. It holds
the per-device lock for the command lifetime and pins GPU visibility, so jobs
sharing a device wait while jobs on separate devices can run together. Preserve
the allocation and container device mapping: different logical IDs alone do
not prove physical isolation. Do not bypass the lock or launch detached GPU
children that outlive it. A comma list passed to `gpu_lock.sh` is a pool from
which it leases **one** GPU, not a lease of the whole list.

For a standalone multi-rank UT, lease its entire allocated set with
`python3 "$SKILL_DIR/scripts/gpu_lock_set.py" "$UT_GPU_SET" -- <command...>`
from the task directory. This nests the existing single-device locks in a
common order, then exposes the full requested set for the foreground command.
Use the real collective/rank harness and verify every rank; a set lease alone
does not turn a single-GPU UT into a multi-rank correctness test.
For collective targets, read [multi-rank collective UTs](references/collective-ut.md).

Serving/TP captures must retain `SERVING_GPU` and `SERVING_TP` and use the
`bench_e2e.sh` serving lock and lifecycle. A serving lock alone does not exclude
single-GPU UT jobs: reserve the full serving set against other GPU work, or
serialize serving captures and validation when that isolation is unavailable.

GPU validation must exercise correctness, dispatch/binding and deployment
graph/compile checks with real served cases. Record commands, actual locked
device/environment, exit status and per-case results separately for each UT.
Fix and retest failures; if a UT cannot pass, return its failing smoke status
and blocker, and continue with independent UTs. A shared baseline, harness or
evidence defect pauses the affected tasks until corrected. CPU/static checks
never substitute for a passing GPU test.

Read [source extraction and UT validation](references/source-extraction.md).
Resolve the selected candidate's evidence or the current profile's links on
resume. Missing required evidence yields smoke `fail` with an actionable reason.

Build cases from observed operands, state and dispatch. Include material served
phases, boundaries and tails; record missing cases and uncovered mass. Preserve
real routing and compact quantized inputs where required by the math contract.
Keep process/rank-local capture, the oracle and deployment correctness checks.

Use a per-signature baseline reference for actual kernel names, full grids and
calls per phase instance. Verify the candidate's intended execution independently;
optimized code may change launch topology. For source-edit candidates, deliver
`kernel_src/` and prove that its editable branches actually execute.

Keep task-specific inputs/assertions in `cases.py` or support files, preserve the
immutable shared harness and overlay baseline, and record evidence in
`meta.kernel_extraction`. Smoke the generated UT before returning `pass`.
Then preflight the exact serialized return against the original `KERNEL` using
`node "$SKILL_DIR/scripts/kernel_extraction_delivery.mjs"` as described in the
delivery contract. Return complete verdicts or supported task-local file/hash
references; prose aliases and elided gate fields are not machine evidence.
Repair return-only defects using saved evidence without rerunning passing GPU
tests. A preflight pass checks delivery consistency, not UT correctness or
source editability; retain the independent GPU evidence required above.

## Observed weighting contract

Call `h.serving_weighted_speedup(per_case, meta)` with
`workload.kind="agentx_trace_replay"`, `workload.weighting="observed"`,
`workload.serving_weight_model.source="observed_workload"`, and that model's
`evidence_path` pointing to the workload artifact. Each `workload.cases[]` entry has:

| Field | Contract |
|---|---|
| `sig` | Unique stable signature shared by timing cases and measured results; includes relevant phase/shape/state/dispatch distinctions |
| `regime` | Nonempty observed phase label, matching results and `served_regimes` when set |
| `effective_calls` | Finite nonnegative call mass on one common run/iteration basis; fractional mass is valid for interpolation |
| `m` | Actual kernel M for GEMM bucket probes; omit for operations without this axis |

Score `sum(baseline_ms * calls) / sum(optimized_ms * calls)`. Convert time mass
back to calls using matching reference latency when necessary and record the
conversion. Sampling bins choose probes; weights come from the original joint
distribution using measured costs or validated interpolation, not unvalidated
nearest-bin rounding or a weighted geomean.

Every positive-mass served case needs both measured latencies. Missing/duplicate
cases and missing observed models fail; retain measured unchanged cases in the
denominator. Zero-mass correctness/extrapolation cases do not count as measured
coverage. Existing `agentx_log_fold` tasks remain supported. Missing AgentX
weights must not fall back to fixed-load analytic weighting.

## Op benchmarker: preserve the workload

Keep the extracted cases, weights, binding checks and coverage across backend
and author lanes. `served_buckets` may select preliminary GEMM probes; only the
full UT validates other axes, repeated M with different state and non-GEMM work.
Verify runtime engagement and inspect regressions in material phases/tails
under the existing acceptance policy. Table overrides must preserve untargeted
entries and pass dispatch and numerical checks.
