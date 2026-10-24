# Evolution within the kernel engineering workflow

Use the same mathematical contract, editable boundary, tools, correctness tests
and full-call measurement as the rest of this workflow. The archive changes what
you can develop next; it does not change what counts as a measured improvement.

When planning, inspect EVOLUTION_ARCHIVE and the evidence for relevant entries.
Choose a retained `parent_id` for each direction and optional `donor_ids` whose
ideas or source are useful. Distinguish the best validated program from promising
alternatives and unfinished attempts. Continue strong exploitation of the best
while giving distinct approaches a finite, purposeful opportunity to develop.
Provide an `approach` label describing the computational organization; a label
is a proposed grouping, not proof that two implementations are different. When
useful alternatives exist, include development of one alongside refining the best.
The prompt contains only recent archive entries and the best entry. Search the
full archive file and its referenced evidence for older approaches when planning.
Deep engineers may run alongside other directions in this mode. Keep their total
declared cost within BUDGET_REMAINING, counting DEEP_COST for each deep engineer.
Use the remaining budget to answer concrete engineering questions. Do not abandon
a potentially useful formulation solely because its first implementation is slow.
For exploratory retention, these instructions replace the stock role's rule to
save only above-baseline patches. Correctness and promotion requirements still apply.

Each engineer receives a private copy of the run base with its selected parent's
patch applied. That parent is the starting implementation. Do not reset it to the
current global winner. Donors are references for deliberate combinations: explain
which contribution you intend to preserve and check that it is present in the
executed implementation before interpreting the result.

Investigate like a kernel engineer: understand the complete callable and its state,
locate a measured cost, check what actually executed, state a plausible explanation
and a rival, then choose a measurement or intervention that separates them. Zoom
out to another computational organization when local tuning stops being useful.
Hardware models guide questions; their assumptions and gaps are not measured gains.
Use the existing profiling tools when they resolve an uncertainty, and confirm
performance with the unprofiled full-call evaluator.

Keep `best_patch.diff` cumulative against the private workspace's initial Git
commit, including inherited parent changes. Save the best correct implementation
even if it is slower than the original. Retain unfinished code with its limitation
clearly stated; do not describe it as correct or measured. Preserve raw evidence
paths, useful failed alternatives, corrected premises and the next unresolved
question in the existing engineering outputs. Never edit the archive's base or
another engineer's workspace. Independent verification decides eligibility for
promotion; a missing measurement is not zero performance.
