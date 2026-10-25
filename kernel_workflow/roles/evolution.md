# Evolution within the kernel engineering workflow

Use the same mathematical contract, editable boundary, tools, correctness tests
and full-call measurement as the rest of this workflow. The archive changes what
you can develop next; it does not change what counts as a measured improvement.

When planning, inspect EVOLUTION_ARCHIVE, the existing HISTORY.ledger and the
evidence for relevant entries. Treat the ledger's latest revision of an approach
as a continuation of that approach, not evidence of a distinct approach.
Choose a retained `parent_id` for each direction and optional `donor_ids` whose
ideas or source are useful. Distinguish the best validated program from promising
alternatives and unfinished attempts. Continue strong exploitation of the best
and develop alternatives when their unresolved question warrants the work.
Provide an `approach` label describing the computational organization; a label
is a proposed grouping, not proof that two implementations are different. Keep
that label stable while the organization is unchanged. In each DIRECTION.prompt,
carry the relevant source/evidence references, the unresolved question, and the
next experiment whose outcomes would change the decision. Explain whether it
continues an approach, reformulates it, or opens a distinct one. Select the source
that implements the needed starting point; parent rank alone does not decide.
The current-best profile may describe different source than the selected parent;
check its applicability before carrying bottleneck conclusions into a direction.
The prompt contains only recent archive entries and the best entry. Search the
full archive file and its referenced evidence for older approaches when planning.
Deep engineers may run alongside other directions in this mode. Keep their total
declared cost within BUDGET_REMAINING, counting DEEP_COST for each deep engineer.
Use the remaining budget to answer concrete engineering questions. Do not abandon
a potentially useful formulation solely because its first implementation is slow.
For exploratory retention, these instructions replace the stock role's rule to
save only above-baseline patches. Correctness and promotion requirements still apply.

When updating memory, use EVOLUTION_ROUND, ROUND_RESULTS, INTEGRATE and
PRIOR_HISTORY with the available reports and archive evidence. Add revisions to
the existing ledger and insight_log.md: approach, attempted candidate and retained
source references, what the evidence supports, the remaining question, and a
discriminating next experiment. Distinguish an unrealized intervention, incorrect
source, a correct but slower realization, and missing or inconclusive measurement.
Do not close an entire approach because one implementation failed. Mark useful
contributions already incorporated elsewhere as subsumed; reopen a conclusion
when its premise or applicability changes. Reports may be absent, and references
or author explanations do not establish verified execution or measured benefit.

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
