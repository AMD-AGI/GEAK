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

**This pipeline is required, not a suggestion.** Sample **multiple points per material
phase** and fit the cost model in [cost attribution](references/dynamic-workloads.md) over
them; one capture is a single point and admits no fit. A phase that carries mass and was
measured at fewer than two points fails selection rather than being ranked. Reusing a
compatible window already captured from the baseline replay counts as one measured point,
not as the census.

One short window can establish *what* a phase executes. It cannot establish *how much of
the workload* that phase is: a 5-second sample that contains one long prefill batch and
thirty decode steps reports a prefill-dominated split for a decode-dominated replay, and
every downstream percentage inherits that error. So derive the PREFILL/DECODE GPU-time
split from the **log's instance counts weighted by the measured per-point cost**, over the
whole measured leg, and record it as `phase_share` in `kernel_ranking.json`:

```json
"phase_share": {
  "basis": "log_instance_counts_x_measured_point_cost",
  "PREFILL": 0.1217, "DECODE": 0.8783,
  "points": [
    {"point_id": "pr_2660", "phase": "PREFILL", "verified": true, "counted": true},
    {"point_id": "pr_32768", "phase": "PREFILL", "verified": "partial",
     "counted": true, "relabelled_to_achieved_shape": true},
    {"point_id": "de_1", "phase": "DECODE", "verified": false, "counted": false,
     "why_excluded": "no DECODE step spans in the capture"}
  ],
  "evidence": {"observed_workload_json": "...", "sampling_points_json": "...",
               "coverage_json": "..."}
}
```

**Only a verified capture may be counted.** `points` carries `serving_probe.py`'s own
verdict per point, and the selector refuses a counted point whose `verified` is `false`:
a capture that holds no step span of the intended phase can hold another phase entirely,
and four such DECODE points once produced a decode mass out of windows that contained no
decode at all. A `"partial"` point is a real measurement of a *different* shape than
planned; count it only after relabelling it to the achieved shape and re-deriving the
instances it represents, and set `relabelled_to_achieved_shape`. Exclude the rest with a
reason, and recapture if a phase then falls below the minimum.

State the window's own span and step count as a limitation whenever the fit rests on a
short capture, and never present a window percentage as a whole-workload percentage.

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

Default selection is **whole-workload GPU-time share strictly >3%**, and nothing else.
The share of a kernel is `Σ_phase phase_share[phase] × share_of_phase_pct[phase]`, so the
threshold is applied once, on the log-weighted split above, across every material phase in
scope. There is no per-phase Top-K: a phase leader can fall below 3% and a rank-20 row can
sit above it, and both outcomes are correct. Within-phase ranks stay in the artifact as
inventory and disclosure, never as a gate. Union by live identity. Explicit user scope
overrides this default. Run:

```bash
python3 "$SKILL_ROOT/scripts/select_targets.py" \
  "$EVAL_DIR/kernel_ranking.json" --output "$EVAL_DIR/selection_targets.json"
```

The selector recomputes each overall share from `phase_share` rather than trusting the
ranking's own `selection_pct_gpu_time`, which it retains as `window_pct_reported` for
comparison. It refuses a missing or window-only `phase_share`, and refuses a phase that
carries mass but was measured at one point. `--allow-window-denominator` exists only to
record an explicitly declared-approximate scope; its output is stamped non-comparable and
that stamp belongs in the delivery. `--top-k` is an optional execution cap — anything it
drops stays in `dropped_by_cap` and remains a selected identity owing a task or blocker.

**Recall fusable chains whose combined share clears the threshold.** A per-kernel
threshold hides the one optimization a chain of small consecutive stages actually admits:
rope-then-cache-write, norm-then-quant, score-then-top-k, gemm1-then-gemm2. No stage
carries the mass, the chain does, and judging the stages one at a time deletes the fusion
candidate before anyone can see it. Declare such chains in `kernel_ranking.json` and the
selector recalls **every** member of a group whose summed share exceeds the threshold:

```json
"fusion_groups": [{
  "id": "qk_rope_cache_norm",
  "members": ["fused_qk_rmsnorm|...", "_fused_qk_rope_cat_and_cache_mla_kernel|..."],
  "adjacency_evidence": {"kind": "observed_consecutive_launches",
                         "trace": "...", "observed_sequences": 30,
                         "same_stream": true, "intervening_kernels": 0},
  "rationale": "consecutive stages of the QK path, fusable into one pass"
}]
```

A group needs at least two members, all present in the ranking, and
`adjacency_evidence.kind` of `observed_consecutive_launches` or `observed_dispatch_chain`.
Grouping inferred from a shared callable or kernel name is rejected, because summing is
only sound for **disjoint** rows and a parent op and its own child kernel overlap
completely — that is the same prohibition as "do not add parent/child shares". A recalled
member keeps its own share and is booked once however many groups name it; the group's
total is never added to the selected mass. Set `--fusion-group-pct-gt` to hold chains to
a different bar than single kernels. Recalled members appear with
`reason=recalled_by_fusion_group` and owe a task or blocker like any other selected
identity; the group id travels with them so the chain stays reviewable as one fusion unit.

**Communication operators are selected but get no UT.** Mark every collective row with
`entity_kind: "collective"` in the ranking. It stays selected, keeps its share and stays
in the denominator — a TP all-reduce holding 27% of device time is a fact the report must
carry — but no UT is authored for it and it is not optimized. The selector puts it in
`ut_excluded` with that reason and counts only the rest in `ut_target_count`.

This is a scope decision, not a deferral. `deferred` means pending work and invites a
retry that is never coming; a collective excluded here needs no 8-rank lane, no budget
and no follow-up. Say so in the delivery, keep its measured mass visible, and do not
renormalize the denominator to hide it. `--ut-exclude-kind` changes the set if a run
genuinely is scoped to optimize communication.

Every other selected identity needs a task or explicit blocker. Execution limits may
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

**PREFILL and DECODE may be delivered as separate tasks.** One callable serving both
phases does not have to become one task: when the two phases take different branches,
operand layouts, graph/compile modes or cases, a per-phase task is the clearer boundary
and is explicitly allowed. Split or join on evidence, not on the callable's name. A split
costs nothing in coverage as long as each task declares `served_regimes` for the phase it
owns, the two together account for every selected identity and required companion of that
callable, and neither silently drops the other phase's mass. Cross-link the siblings in
`run/requested_kernel.json` so the pair is reviewable as one boundary, and keep per-phase
results and weights separate — there is then no single cross-phase aggregate to report,
and none should be invented. A joined task remains correct where one invocation genuinely
spans both phases.

For selected indexer score/top-k, deliver the score and top-k stages of the chain
together, and cover both served prefill and decode branches — as one unified task, or as
a prefill task and a decode task that cross-reference each other. Read
[joint indexer UT](references/joint-indexer-ut.md). What is not optional is the coverage:
a prefill-only indexer delivery leaves the decode branch owing a task or a blocker.
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
baseline case per phase to establish replay — **that is a bootstrap checkpoint, never
the deliverable** — then expand the manifest to span the measured distribution before
any gate is reported. The UT reads its
frozen case manifest; it never launches serving, profiles, or captures shapes.
Each `tasks/<name>/` contains its own `kernel_src/`, `cases.py`, `meta.json`,
`unittest.py`, baseline manifest/overlay and `run/` evidence. Captured oracles are
optional; preserve minimal structural fixtures only where needed.
Copy the shared harness/leg/overlay helpers into the task; freeze them once authored.

**The case manifest covers the measured shape distribution, not the one shape the census
happened to open on.** A decode kernel served at batch sizes 1 through 9 is not tested by
a case at batch 3, and a prefill kernel served from 1 to 32768 new tokens is not tested by
a single 15k instance — yet both pass every correctness gate, because the bootstrap case
and the finished UT are indistinguishable from the outside. Build the manifest from
`plan_sampling_points.py`'s points: each planned point's shape needs a case within the
ledger's shape tolerance, and each case carries the `effective_calls` that point represents,
so the weighting is the observed one. Add dispatch boundaries, graph-padding buckets and
material tails as their own cases; a bucket that changes branch or kernel is a separate
case even when its mass is small.

Emit the frozen manifest as `run/ut_cases.json`
(`[{"sig":…, "phase":"DECODE", "shape":18, "effective_calls":N}, …]`) and grade it:

```bash
python3 "$SKILL_ROOT/scripts/coverage_ledger.py" \
  --points "$EVAL_DIR/sampling_points.json" --ut-cases "$TASK_DIR/run/ut_cases.json" \
  --census <point_id>=<census.json> --target 90 \
  --out "$TASK_DIR/run/coverage.json"
```

Record the result as `shape_coverage` on the task return, with the ledger path, the
target, and per phase its `basis` (`mass` or `instance`), `coverage_pct` and
`meets_target`. A phase below target is allowed and often honest, but it must carry
`shortfall.uncovered_pct` and `shortfall.reason`; the delivery gate rejects a missing or
silently-below-target coverage block. Remember the ledger refuses mass coverage for a
phase measured at one point and reports instance coverage instead — a 100% from one point
is the denominator agreeing with itself, not coverage.

Deliver actual native/JIT/generated computation source for every claimed leaf,
with build resources and version/hash provenance. A copied dispatcher that
imports installed computation is not extraction of those leaves. Prove actual
leaf invocation through the candidate overlay, independently of correct output
or import location. Preserve deployed signatures, layouts, quantization,
routing semantics, cache/page state and complete outputs.

Run every GPU command from its task directory under the lease tools in
[setup](references/setup.md) — `gpu_lock.sh` for one device, `gpu_lock_set.py` for the
whole allocated set. Never run a GPU command outside a lease.

The UT must cover baseline/candidate binding, frozen-oracle and random-value
correctness at served shapes, phase-specific graph/compile behavior and input
refresh, each editable leaf's numerical negative control, restored-source
integrity, both timing legs for every positive-mass signature, and the measured
shape distribution graded by `coverage_ledger.py`. Compilation
failure is not a successful numerical negative control. A green gate report on a
one-shape manifest is not a passing UT; report the achieved coverage with the same
prominence as the pass. Report missing cases
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
capture defaults to `CAPTURE_PERSIST_POLICY=metadata_only`: save the complete
shape/layout histogram and scalar metadata, with no tensor I/O. Explicitly choose
`structure_only` only for indispensable structural fixtures (64 MiB per process,
8 MiB per case by default, checked before copying). Packed integer weights are
numerical payloads and require a format-aware constructor. Do not persist generated
inputs or golden outputs. Compute references per bucket in the frozen baseline
process, compare in memory, and release each batch; the shared helper uses a bounded
anonymous memory buffer, never `_baseline_random.pt`.

Use `h.measure_legs()` and `h.serving_weighted_speedup()` from the shared
harness. Set `workload.kind="agentx_trace_replay"`, `weighting="observed"`, and
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
coverage, uncovered mass and limitations. Report **two** coverage numbers and never let
one stand for the other: selection coverage (how much of the workload's device time the
selected identities carry) and, per task and per phase, shape coverage (how much of that
kernel's own served distribution the frozen case manifest exercises). A delivery can
select 60% of device time and still test one shape of it. Separate inventory completion,
task completion, GPU UT status and delivery consistency. A CPU-only check or a
subset of passing tasks is not full acceptance. Do not report E2E improvement
without a separately authorized and measured E2E result.
