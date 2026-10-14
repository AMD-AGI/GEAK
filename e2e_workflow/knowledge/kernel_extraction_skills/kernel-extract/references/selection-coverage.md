# Selection scope and coverage

Read for the internal selection step of `prepare`, prior-delivery reconciliation and completion reporting.
Record whether the request is optimization, UT delivery or skill review, and
its phases/families/count/coverage. Skill review alone does not start a campaign.

## Targets and scheduling

Default: `(rank_in_PREFILL <= 5 OR rank_in_DECODE <= 5) AND overall_pct > 3`.
Rank by within-phase device time, filter by total-workload share, then union by
live kernel identity. Exactly 3% is excluded; do not backfill rank 6. Preserve
both phase obligations for an identity selected in both phases. Explicit user
scope takes precedence; record exclusions such as a non-collective scope before
ranking, without silently changing the overall denominator.

Keep collectives, unsupported kernels and unresolved mappings in the inventory.
A selected kernel that cannot be extracted gets a blocker, not a replacement.
A missing phase map entry means zero only after absence has been established.
Use current measured/folded costs, not historical percentages or expected gain.

Run the deterministic selector:

```bash
python3 "$KERNEL_EXTRACTION_SKILL_DIR/scripts/select_targets.py" \
  "$EVIDENCE_DIR/kernel_ranking.json" --output "$EVIDENCE_DIR/selection_targets.json"
```

`KERNEL_EXTRACTION_SKILL_DIR` names this skill, not the workflow `SKILL_DIR`.
Input `ranking[]` rows need a unique `id` (or unique `name`), finite
`selection_pct_gpu_time` and `share_of_phase_pct` keyed by phase. The helper
breaks exact ties by identity and records retained/excluded Top-5 rows. Its
output is a target plan, not dispatch, source or GPU-validation evidence.

`HEAD_BUDGET`, `BUDGET` and `KERNEL_BUDGET` govern execution, not discovery.
`HEAD_THRESHOLD_PCT` and `MILESTONE_MIN_PCT` govern optimization; they must not
replace the extraction rule. Preserve all evidenced extraction targets in the
delivery, including failed and budget-deferred entries for resume. Only passing
UTs enter optimization queues after the skill finishes. Record requested work
that the caller cannot schedule; do not silently raise budgets or redefine
completion. A low optimization ceiling does not invalidate a UT.

Queue one owner per `ut_task`, in exactly one queue. A grouped owner retains
`selected_device_kernels` and `extraction_members[]` containing each member's
`device_kernel`, `short_name`, `served_regimes` and original `kernel_extraction`.
Declare mandatory companions in `joint_ut.phases[].required_device_kernels`;
if included in `extraction_members`, mark unselected companions with
`kernel_extraction.selected=false`. Alternative branches use `dispatch_ut`.
Follow the [delivery contract](delivery-contract.md); a shared callable or task
name alone does not prove grouping. Never schedule owner and leaves twice or
sum their overlapping shares.

## Evidence artifacts

`kernel_workload.json` records feature/configuration evidence, the measured
interval and rank/stream scope, distributions, effective calls, sampling plan,
coverage, integrity checks and limitations. Keep summed kernel time, union busy
time and wall time separate, including collective and unattributed mass.

Preserve existing trace-window `profile_topN.json`/`.md` and parser percentages;
they are optional evidence, not a ranking synthesized from logs. Candidates
link `kernel_extraction.workload_json` and `ranking_json`, retain original
`profile_pct_gpu_time`, and distinguish `share_of_stage`, `share_in_seam` and
`share_editable`. These nested cost scopes are not additive. Express shares as
percent with explicit denominators. `selection_pct_gpu_time` is the measured
share the proposed change can affect over total workload device time; copy it
to candidate `pct_gpu_time` for routing, including refreshed deep-track heads.

Add to `kernel_ranking.json`:

- `selection_scope`: user purpose/extent and its source, effective budgets and
  thresholds, target-plan path and any scheduling mismatch.
- `candidate_inventory`: stable phase/call-site/device identities, source and
  case evidence, costs/denominators, prior-task matches and overlap relations.
  Keep eligibility (`eligible`, `below_optimization_threshold`, `unsupported`,
  `unresolved`) separate from execution (`queued`, `deferred_budget`, `passed`,
  `failed`, `not_run`). Missing evidence is unknown with a reason, not zero cost.
- `coverage_summary`: accounted/unresolved mass by phase, unassigned rows,
  reconciled/unresolved prior tasks and scheduled/deferred/tested units.

Account for every ranking row, using explicit member lists and mass for a small
tail group. Preserve material phase leaders even below the overall threshold.
A probe budget ending leaves unresolved entries visible. An older saved queue
without an inventory has unknown completeness; reconcile existing evidence.

## Prior deliveries and completion

Read supplied manifests and latest handoff notes even when learned-KB lookup is
off. Match current operation, call site, phase, operands and dispatch; a name
match or absence from one window is insufficient. Record matched, overlapping/
superseded, not observed, or unresolved. Revalidate current costs and dispatch;
old optimized candidates and smoke results are not the serving baseline.

Count implementations, task packages and disjoint work separately. Report four
outcomes: inventory reconciliation, requested-task completion, each UT's tested
case coverage, and e2e validation. Passing a subset does not complete missing
tasks; passing representative or reconstructed inputs does not prove full
input-distribution fidelity. Do not infer e2e gains from device-time speedup
without accounting for uncovered work, communication, overlap and scheduling.
