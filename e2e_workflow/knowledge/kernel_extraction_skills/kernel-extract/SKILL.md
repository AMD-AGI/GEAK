---
name: kernel-extract
description: Extract actual GPU kernel source and workload-weighted standalone tests from AgentX replay evidence. Use with serving logs, traces and an accessible deployed GPU environment; covers phase selection, live dispatch, source binding, correctness and timing. Excludes fixed ISL/OSL benchmarking and kernel optimization unless explicitly requested.
---

# AgentX kernel extraction

The agent performs the five stages below using this skill's method and the
repository's shared tools. Set `SKILL_ROOT` to this directory and `SHARED_TOOLS`
to `e2e_workflow/scripts`. Read [setup and inputs](references/setup.md) first.
When invoked by GEAK, also read [workflow handoff](references/workflow-handoff.md)
for input paths and the final return format.

The objective is a reviewable source-and-test delivery for the user's current
AgentX workload. Preserve the installed stack, accepted overlays, baseline,
TP layout, allocation and deployed graph/compile modes. Do not silently start
optimization or a new full replay. Missing GPU access permits useful CPU work,
but leaves GPU validation explicitly incomplete.

For vLLM, first read [vLLM observation and profiling](references/vllm.md).
Its throughput logger and request/profiler APIs differ from the SGLang defaults.

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

Given a completed baseline's **serving log and config**, the path from that log to a
small set of profile points is executable: `observed_workload.py` → `plan_sampling_points.py`
→ `serving_probe.py` (once per point) → `coverage_ledger.py`. Run them in that order with
the commands in [workload measurement](references/dynamic-workloads.md); using them rather
than your own is what makes two runs against one deployment comparable.

These counts come from the log, never from a parametric ISL/OSL model.
`parse_profile.analytic_regime_calls` assumes uniform request lengths and cannot see
prefix-cache hits, accept length or the batch distribution; on an agentic replay it is
wrong by whole multiples. Use it only when no serving log exists at all, and say which
of the two you used.

Use existing traces first. If needed, design bounded probes with explicit
request/time/byte limits using the user's actual serving launcher; follow
[capture procedure](references/capture.md). A probe's synthetic load does not
replace the original baseline weights. Preserve failed evidence; allow one
recapture per failed point before reporting the gap and continuing elsewhere.

Write `kernel_workload.json` and `kernel_ranking.json`. Distinguish stage share,
share within the callable and share the extracted source can affect; these are
nested scopes, not additive. Keep collectives, unattributed work and unsupported
kernels in the denominator and inventory. Window percentages alone are not
whole-workload percentages: `coverage_ledger.py` converts them using the observed
instance counts, under the constraints listed in the reference.

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
in the delivery contract), including exact selected identities, callable
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

Trace live callable candidates using `$SHARED_TOOLS/seam_trace.py` and run
`python3 "$SHARED_TOOLS/kernel_selection.py" --strict-launch --help` for the verifier. Preserve complete raw
verdicts and deeper-candidate evidence. Names alone cannot prove a binding or
that the requested whole chain executes. If the declared boundary is wrong,
repair the saved task plan with evidence before extracting; do not edit a
failed verdict into a pass or rerun an unchanged UT to repair JSON fields.

## 4. Extract source and build the standalone UT

Read [source extraction](references/source-extraction.md) and
[task authoring](references/task-authoring.md). Default to fixed-seed generated inputs
at shapes/dtypes/layouts established from existing workload evidence. Run one such
baseline case per phase, then expand cases and editable leaves. The UT reads its
frozen case manifest; it never launches serving, profiles, or captures shapes.
Each `tasks/<name>/` contains its own `kernel_src/`, `cases.py`, `meta.json`,
`unittest.py`, baseline manifest/overlay and `run/` evidence. Captured oracles are
optional; preserve minimal structural fixtures only where needed.

Write `cases.py` and `meta.json` (declared capabilities, case manifest, tolerance,
observed workload; every `kernel_src/` file is either copied source with its origin in
`kernel_extraction.source_origins` or a binding file you wrote, listed in
`kernel_extraction.binding_files`), then scaffold, check and freeze the UT with the
generator:

```bash
python3 "$SHARED_TOOLS/generate_ut.py" all "$TASK_DIR" --ranking "$RANKING_JSON"
```

`$RANKING_JSON` is the kernel ranking of this extraction: `PREVIOUS_RANKING_JSON` on an
AgentX resume, in a baseline round the `kernel_ranking.json` written in step 2 (the
`ranking_json` of `workflow_tasks.json`). `--ranking` writes its absolute path into
`meta.kernel_extraction.ranking_json` (or fill that field yourself), so give it on the
first `all`, before freeze: a later change of the path is a change of `meta.json` and
voids the UT results. The check refuses a phase in which the kernel has calls in that
ranking without timing and correctness cases (task-authoring.md, "Every phase the
kernel has calls in").

It copies `harness_lib.py`, `leg_runner.py`, `overlay_setup.py` and `ut_runner.py`
unchanged, writes the fixed `unittest.py` (never hand-write or edit it), checks the
hooks, metadata and coverage on the CPU, and freezes the test contract in
`run/ut_contract.json`. Rerun it after any change to `cases.py` or `meta.json`; a
changed test invalidates every earlier result. `python3 unittest.py` then runs the whole
UT: binding, independent baseline references, correctness, timing, weighting and the
timing receipt. A kernel that runs on several GPUs at once (a collective, or a fused
kernel exchanging data with other ranks inside the kernel) is declared in
`capabilities.multi_rank` and follows [multi-rank UT](references/collective-ut.md).

Deliver actual native/JIT/generated computation source for every claimed leaf,
with build resources and version/hash provenance. A copied dispatcher that
imports installed computation is not extraction of those leaves. Prove actual
leaf invocation through the candidate overlay, independently of correct output
or import location. Preserve deployed signatures, layouts, quantization,
routing semantics, cache/page state and complete outputs.

Run every GPU command from its task directory under the lease tools in
[setup](references/setup.md) — `gpu_lock.sh` for one device, `gpu_lock_set.py` for the
whole allocated set. Never run a GPU command outside a lease. A multi-rank UT instead
leases `SERVING_GPU` as one group, bounded by the UT's total time limit; its
`verify_ut.py` and source-mutation runs lease the same way:

```bash
GEAK_GPU_RUN_TIMEOUT=<meta.timeouts.total_s> \
  bash "$SHARED_TOOLS/../../kernel_workflow/scripts/gpu_lock.sh" "group:$SERVING_GPU" -- python3 unittest.py
```

The UT must cover baseline/candidate binding, frozen-oracle and random-value
correctness at served shapes, phase-specific graph/compile behavior and input
refresh, each editable leaf's numerical negative control, restored-source
integrity, and both timing legs for every positive-mass signature. Compilation
failure is not a successful numerical negative control. Report missing cases
and mass; do not renormalize survivors into full coverage.

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

The generated UT times with `h.measure_legs()` and weights with
`h.serving_weighted_speedup()`. Set `workload.kind="agentx_trace_replay"`,
`weighting="observed"`, and `serving_weight_model.source="observed_workload"` with an
evidence path. Each case has a unique `sig`, observed `regime` and finite nonnegative
`effective_calls` on one consistent basis. Aggregate as
`sum(baseline_ms * calls) / sum(candidate_ms * calls)`; no fixed-load fallback
or weighted geometric mean. A near-1x unmodified extraction is expected.

## 5. Validate and deliver

Read [UT validation](references/ut-validation.md) and
[source records](references/source-delivery-records.md), freeze the task
snapshot before GPU validation and generate records from actual outcomes.
Run the UT, the verification of the test itself (`verify_ut.py`) and the per-leaf
source mutations as one chain: one script under ONE lease call, so the GPUs are not
released and queued for again in between.

```bash
cd "$TASK_DIR"
cat > run/gpu_chain.sh <<'EOF'
python3 unittest.py || exit                              # full UT -> run/ut_result.json
python3 "$SHARED_TOOLS/verify_ut.py" --task . || exit    # -> run/verify_ut.json, exit 0 on PASS
for leaf in <leaf> ...; do                               # one reversible mutation per editable leaf
  <apply the mutation of $leaf to its kernel_src/ file, logging what changed>
  python3 unittest.py --mutation "$leaf" > "run/mutation_$leaf.log" 2>&1   # rejected: exit 1
  <restore that file byte for byte>
done
python3 unittest.py --correctness-only > run/restored_check.log 2>&1
grep -Eq '^\[ut\] full UT result [^ ]+: current' run/restored_check.log || python3 unittest.py
EOF
bash "$SHARED_TOOLS/../../kernel_workflow/scripts/gpu_lock.sh" "$GPU_ID" bash run/gpu_chain.sh
# a multi-rank UT: the group lease of SERVING_GPU, its time limit covering the whole chain
GEAK_GPU_RUN_TIMEOUT=<several x meta.timeouts.total_s> \
  bash "$SHARED_TOOLS/../../kernel_workflow/scripts/gpu_lock.sh" "group:$SERVING_GPU" -- bash run/gpu_chain.sh
```

The chain stops at a failing UT or verification; fix it and run the chain again.
Only a `PASS` bound to the current contract and source snapshot is deliverable; the
delivery gate checks it. `INCONCLUSIVE` (the run was stopped from outside: GPU memory,
lock wait, IPC fault, time limit) is not a task defect: run it again. A `FAIL` names the
check or control that failed; fix `cases.py`/`meta.json`, regenerate, rerun the UT and
verify again. The per-leaf source mutations are still required: controls prove the
test is sensitive, mutations prove it executes the extracted source. A mutation run
(`--mutation <leaf>`, correctness only) writes `run/mutations/<leaf>/ut_result.json`,
never the delivered `run/ut_result.json`. After the restore, one `--correctness-only`
run decides: its `full_result.current` is true when the last full UT passed with the
same contract and source snapshot, and that result is delivered as it is; otherwise the
restored task differs from the tested one, the full UT runs again, and `verify_ut.py`
and the mutations, bound to the same hashes, must run again too. The delivery gates
still check the hashes. Keep one full UT to about 10 minutes (`wall_s` in
`run/ut_result.json`): if it takes longer, merge timing cases that fall in the same
deployment bucket (in a graph-deployed phase: the same deployment graph, M padded to
the same captured size with the other case dimensions equal -- see "Graph replay
groups" in task-authoring.md) and drop duplicates; the coverage rules do not change.
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
Only this full form writes `run/RETURN.json` (with `--out`, on pass). A failing full
preflight is a defect to fix; `--prepared` is the workflow's packing check and must not
be used to write `RETURN.json`. `target_callable` in `run/requested_kernel.json`,
`RETURN.json` and `meta.json` is the entry the `kernel_selection.py` verdict certifies,
e.g. `SiluAndMul.forward`, not the class; `candidate_bind` may still bind the class.
When that entry resolves to the same code on both legs (a `forward` inherited from a
base class), the candidate overrides that entry itself so the legs differ there; never
change `target_callable` to pass `legs_differ`.

Write `REPORT.md` and `results.json` with all selected identities, task/source
locations, actual commands and devices, per-gate status, declared and achieved
coverage, uncovered mass and limitations. Separate inventory completion, task
completion, GPU UT status and delivery consistency. A CPU-only check or a
subset of passing tasks is not full acceptance. Do not report E2E improvement
without a separately authorized and measured E2E result.
