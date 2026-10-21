# Selection scope and coverage

Use with the [core skill](../SKILL.md) for selection, prior-delivery review and
extraction acceptance. Store the inventory and scope in `kernel_ranking.json` and the task plan.

## Discovery precedes scheduling

Establish whether the user requests optimization, extraction/UT delivery, or a
test of this skill. Record the requested phases, families, count or coverage,
and the source of that scope. A request to audit or fix the skill uses existing
artifacts as evidence; it does not start another extraction campaign.

The selection rule itself is in [the core skill](../SKILL.md) §2; this file covers how to
apply it to an inventory. Precisely: `overall_pct > 3`, where
`overall_pct = Σ_phase phase_share[phase] × share_of_phase_pct[phase]`, `phase_share` is
the PREFILL/DECODE split estimated from the serving log's instance counts weighted by
measured per-point cost, and `share_of_phase_pct` uses that phase's device time under one
consistent rank/stream scope, from current measured costs rather than historical
percentages or expected gain. Exactly 3% is excluded.

There is no per-phase Top-K. Rank within a phase for inventory, disclosure and probe
priority, never as a gate: a phase leader below 3% is excluded and a rank-20 row above 3%
is selected. Dropping the Top-K is what lets a decode-heavy replay select decode work that
a prefill-weighted window had buried, so do not reintroduce a rank cut to shorten the list
— use `--top-k` if execution must be bounded, which keeps the remainder in `dropped_by_cap`
as selected identities still owing a task or blocker.

A declared `fusion_groups[]` entry is evaluated on its **summed** share and recalls every
member when that clears the threshold, so a chain of consecutive sub-threshold stages
survives into extraction as a fusion candidate instead of being deleted stage by stage.
This is the one place member shares may be added, and only because adjacency evidence
establishes that the members are disjoint device-kernel rows. It does not license adding
parent/child shares, inferring a joint seam's editable share from one child, or summing
overlapping members — those remain prohibited below and the selector rejects a grouping
whose evidence is a shared name. A recalled member counts once toward
`selected_overall_pct` no matter how many groups contain it, and the group total is never
itself added. Record the group id on each recalled identity so the chain stays reviewable
as one fusion unit, and keep a group that failed the threshold in the artifact with its
measured total rather than dropping it silently.

Deduplicate a kernel selected in both phases while preserving both phase/case obligations;
distinct live dispatch identities stay separate even if one callable reaches them. A
deduplicated identity may still be delivered as a prefill task and a decode task.

Keep collectives and unsupported kernels in the ranking. A selected but
unextractable kernel needs a blocker; it does not silently disappear. Under a
threshold-only rule it also no longer consumes a slot that a schedulable kernel could
have used, which is part of why the rank cut is gone: a TP collective should
cost the delivery its own mass and nothing more.

Collectives are the one class that is selected without owing a UT. Mark them
`entity_kind: "collective"` and the selector routes them to `ut_excluded`: their mass
stays in the denominator and in the report, no UT is authored, and they are not
optimized. Record them as an explicit scope exclusion rather than a deferral — a
deferral says the work is pending, which for a communication operator out of scope is
simply untrue, and it leaves a phantom obligation in every later coverage statement. Additional targets or an explicit
non-collective scope require recording that changed scope before ranking. Never
renormalize the overall denominator merely because a target cannot be extracted. Joint tasks may
cover multiple selected kernels with proved boundaries, but joint/leaf
overlaps must not inflate ranks, shares or the number of delivered kernels.

`select_targets.py` (run it as shown in SKILL.md §2, after live identities are resolved)
needs a top-level `phase_share` block and `ranking[]` rows with a unique `id` (or unique
`name`) and `share_of_phase_pct` keyed by phase. `selection_pct_gpu_time` is optional and
is echoed as `window_pct_reported`; the threshold is never applied to it, because it is a
capture-window figure. Pass `--phases` with the exact observed phase labels when they
differ from the default `PREFILL DECODE`. The phase map must be complete for observed
activity; absent keys mean zero only after that has been established, and missing/invalid
costs fail selection.

`phase_share` needs `basis`, one nonnegative weight per phase (fractions or percentages
both work; they are normalized) and `points_measured` per phase. The selector refuses a
`basis` naming a single capture window, and refuses a phase that carries mass but was
measured at fewer than `--min-points-per-phase` points (default 2). Both refusals exist
because a one-window split silently decides the whole ranking. `--allow-window-denominator`
waives them for a declared-approximate scope and stamps the policy block accordingly;
carry that stamp into the delivery.

Optional `fusion_groups[]` entries need `id`, at least two `members` naming ranking rows,
and `adjacency_evidence.kind` of `observed_consecutive_launches` or
`observed_dispatch_chain`; `--fusion-group-pct-gt` sets their bar when it differs from the
per-kernel one. The output records per-phase ranks for every observed row, every group
with its summed `group_overall_pct` and verdict, the selected set with its deduplicated
`selected_overall_pct` and `recalled_count`, and `dropped_by_cap`. It is a target plan,
not dispatch, editability or GPU-validation evidence — a recalled chain is a fusion
candidate to investigate, not a proof that the stages can be fused.

Task/time budgets limit execution, not discovery: they never change the selection rule,
remove a required target, or redefine the requested scope. Return all evidenced targets in
priority order and report the remainder the agent could not schedule. A budget-deferred
candidate keeps its identity and evidence for resume rather than counting as rejected, and
a below-threshold entry can leave the optimization queue while staying in the inventory. A
phase leader can fall below the overall threshold; preserve both shares. A small
optimization ceiling does not invalidate a UT. The target count is whatever the threshold
admits — one kernel, or twenty — and a long list is a scheduling problem to report, not a
reason to cut the list back to a rank.

Build the inventory from all available ranked rows, per-phase leaders, current
source/dispatch evidence and any supplied delivery manifests. Group a small
tail explicitly, with its members and mass; do not omit unassigned rows.
Prioritize bounded probes for material unresolved entries. When the probe
budget ends, retain unresolved identities and missing evidence without
inventing eligibility. Record collective, library-only and unattributed work.

The task plan contains UT tasks, while the ranking contains selected kernel identities.
Return one owner per explicit `ut_task`, once in the task plan. For grouped tasks,
retain `selected_device_kernels` and `extraction_members[]` with `device_kernel`,
`short_name`, `served_regimes` and the original `kernel_extraction` evidence.
`selected_device_kernels` contains only identities retained by the deterministic
selector. Required companions belong in the phase's `required_device_kernels`;
if also listed in `extraction_members`, mark their `kernel_extraction.selected`
as `false`. Their delivery does not make them additional selected targets.
Declare joint phase boundaries on the owner as described in
[joint indexer validation](joint-indexer-ut.md). Do not schedule the owner's
leaves again in another task. Do not infer a grouping from shared callable
names, sum overlapping member shares, or replace the owner's attributed cost
with the full logical stage cost without evidence of editability.

## Reconcile previous UT deliveries

When the user supplies a delivery, read its manifest, current README/HANDOFF,
task metadata and relevant source boundaries. Follow superseding handoff notes;
historical smoke logs can refer to obsolete harnesses or optimized candidates.
These user-supplied references are task evidence; they do not require a knowledge-base service.

Match by logical operation, call site, phase, operands and live dispatch. Names
alone are insufficient: a generic GEMM symbol can serve several projections,
and one Python seam can select different decode and prefill kernels. Distinct
shape/call-site tasks must not be merged just because they share a callable.
Record each prior task as matched, overlapping/superseded, not observed on this
stack, or unresolved, with evidence. Absence from one sampled window proves
neither workload absence nor a dead branch.

Count source implementations, task packages and nonoverlapping optimization
units separately. A score-only task, a top-k-only task and their joint task
are three test packages but overlapping work. Joint tasks must retain their
member launches, glue and boundary; keep the existing exact device-kernel
anchors for executable candidates. Do not add parent/child shares or infer a
joint seam's editable share from one child. Revalidate current dispatch and
weights instead of copying an older delivery's percentages or eligibility.

## Inventory and completion evidence

Add these sections to the ranking artifact (alongside the task plan):

- `selection_scope`: requested purpose and extent, its source, effective task
  budgets/thresholds, `selection_targets.json`, and any mismatch between requested
  and schedulable work.
- `candidate_inventory`: stable identities, covered ranking rows, phase/call
  site, known device kernels, source/evidence links, current share and denominator,
  prior-task mapping and overlap/member relationships. Distinguish eligibility
  (`eligible`, `below_optimization_threshold`, `unsupported`, `unresolved`) from
  execution state (`queued`, `deferred_budget`, `passed`, `failed`, `not_run`).
  Missing evidence uses unknown/null values with a reason, never zero cost.
- `coverage_summary`: inventoried and unresolved mass by phase, unassigned
  ranking rows, reconciled/unresolved prior tasks, scheduled/deferred/tested
  units and the requested scope still outstanding. Sum only disjoint work on
  a common denominator; leave a combined share unknown when attribution is missing.

Before returning selection, check that each material ranked row, phase leader
and supplied prior task has a disposition. Account for every other ranking row
through explicit tail groups. Keep the inventory when reusing saved queues;
an older selection with no inventory has unknown completeness. Reconcile it
from existing evidence without repeating completed baseline or GPU tests.

Report four separate outcomes: inventory reconciliation, requested task-set
completion, each UT's tested case coverage, and e2e validation. A batch's
`passed` means only that its executed UTs passed. It does not certify missing
tasks, unseen branches or full workload fidelity. A five-task request with two
passing tasks remains incomplete; an explicitly requested two-task smoke can
be complete while inventory/workload coverage stays partial. For a skill test,
include selection omissions and budget deferrals in the verdict, not just the
passing UT results. Synthetic values or median-context reconstructions remain
declared approximations even when every declared timing signature passed.

<!-- Standalone adaptation, 2026-10-12: local tools and agent-owned stages. -->
