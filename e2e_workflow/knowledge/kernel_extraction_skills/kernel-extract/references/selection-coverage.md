# Selection scope and coverage

Use with the [core skill](../SKILL.md) for selection, prior-delivery review and
extraction acceptance. Store the inventory and scope in `kernel_ranking.json` and the task plan.

## Discovery precedes scheduling

Establish whether the user requests optimization, extraction/UT delivery, or a
test of this skill. Record the requested phases, families, count or coverage,
and the source of that scope. A request to audit or fix the skill uses existing
artifacts as evidence; it does not start another extraction campaign.

The selection rule itself is in [the core skill](../SKILL.md) §2; this file covers how to
apply it to an inventory. Precisely: `(rank_in_PREFILL <= 5 OR rank_in_DECODE <= 5) AND
overall_pct > 3`, where within-phase rank uses that phase's device time and `overall_pct`
uses whole-replay time over total device time across all phases, one consistent rank/stream
scope, current measured costs rather than historical percentages or expected gain. Exactly
3% is excluded. A phase with fewer than five observed kernels contributes those available.
Deduplicate a kernel selected in both phases while preserving both phase/case obligations;
distinct live dispatch identities stay separate even if one callable reaches them.

Keep collectives and unsupported kernels in the ranking. A selected but
unextractable kernel needs a blocker; it does not silently disappear or free
a Top-5 slot. Additional targets or an explicit non-collective scope require
recording that changed scope before ranking. Never renormalize the overall
denominator merely because a target cannot be extracted. Joint tasks may
cover multiple selected kernels with proved boundaries, but joint/leaf
overlaps must not inflate ranks, shares or the number of delivered kernels.

`select_targets.py` (run it as shown in SKILL.md §2, after live identities are resolved)
needs `ranking[]` rows with a unique `id` (or unique `name`), `selection_pct_gpu_time` and
`share_of_phase_pct` keyed by phase. Pass `--phases` with the exact observed phase labels
when they differ from the default `PREFILL DECODE`. The phase map must be complete for
observed activity; absent keys mean zero only after that has been established, and
missing/invalid costs fail selection. It ranks by phase share, breaks exact ties by stable
identity, and records both retained and excluded Top-5 rows. Its output is a target plan,
not dispatch, editability or GPU-validation evidence.

Task/time budgets limit execution, not discovery: they never change the selection rule,
remove a required target, or redefine the requested scope. Return all evidenced targets in
priority order and report the remainder the agent could not schedule. A budget-deferred
candidate keeps its identity and evidence for resume rather than counting as rejected, and
a below-threshold entry can leave the optimization queue while staying in the inventory. A
phase leader can fall below the overall threshold; preserve both shares. A small
optimization ceiling does not invalidate a UT. The final target count can be below five
after the overall-share filter, or above five across the two phase lists.

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
