---
name: kernel-profile
description: Profile one AgentX serving deployment's GPU kernels from its measured baseline -- observed workload, bounded trace sampling, census, ranking and target selection -- and deliver profile_handoff.json for kernel-ut. Excludes task boundaries, UT authoring and optimization.
---

# AgentX kernel profile

One deployment, one measured baseline. The agent performs the steps below with this
skill's scripts. Set `SKILL_ROOT` to this directory and `SHARED_TOOLS` to
`e2e_workflow/scripts`. Read [setup](../kernel-ut/references/setup.md) first; when invoked
by GEAK also read [workflow handoff](../kernel-ut/references/workflow-handoff.md).

The objective is a measured account of what this deployment's GPU time is spent on, which
kernels are selected and where each is launched from: `profile_handoff.json`. Preserve the
installed stack, accepted overlays, baseline, TP layout, allocation and deployed
graph/compile modes. Do not silently start a new full replay. Missing GPU access permits
useful CPU work, but leaves capture explicitly incomplete. This skill decides no task
boundary, writes no UT and persists no tensor payloads: [kernel-ut](../kernel-ut/SKILL.md)
consumes the handoff and generates numerical inputs from seeds.

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

Every selected identity is handed to UT with its gaps; which kernels form one UT is
decided there, never by dropping an identity here.

## 3. Deliver the handoff

Stop the probe server as soon as the evidence is collected. Write the profile section of
`REPORT.md`: sampling plan, achieved coverage per phase, uncovered mass, failed or hidden
windows and the selection. Then export and verify the handoff:

```bash
python3 "$SKILL_ROOT/scripts/profile_handoff.py" export --eval-dir "$EVAL_DIR"
python3 "$SKILL_ROOT/scripts/profile_handoff.py" verify --handoff "$EVAL_DIR/profile_handoff.json"
```

`export` refuses a ranking without `selection_scope`, `candidate_inventory` and
`coverage_summary`, or a selected target absent from the ranking. It writes each selected
target's ranking row (shares, phases, shapes, source frames, gaps as recorded) and the
SHA-256 of every summarized artifact; `verify` proves the handoff still describes unchanged
evidence. Report profile coverage and approximations; never report a UT, a task or an E2E
result from this skill.
