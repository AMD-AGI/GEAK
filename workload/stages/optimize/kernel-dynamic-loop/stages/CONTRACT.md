# Kernel module call contract

Use workload's [direct-call contract](../../../CONTRACT.md#direct-module-calls).
The attempt handoff supplies paths and limits; do not assume they are shell exports.
Each call records its module and implementation path, question, parent call, input/source
identity, scope, resource/time budget and decisive check in `PLAN.md`. Keep its outputs
and `REPORT.md` under a distinct attempt subdirectory. Reports include status, start/end
times, native evidence and unresolved work. Partial reports are available while a call
runs, so Brain can revise other plans. Missing, invalid and regressed results differ.

## Shared boundaries

Only declared editable sources may change. The original baseline, harness, inputs,
weights, build/source bindings and undeclared companions remain fixed. Use the existing
[source tools](../../../../engine/campaign/source_tree.py) for identity and copies.
Candidate copies are independent. Use separate complete task copies for measurement
where native bindings support relocation; otherwise the canonical task is a serialized
test location. Restore its prior source after a diagnostic or verification call there.
Delivery to the canonical task is always serialized.
A required UT/build-boundary change returns to the parent for a new qualified task.
Task may derive probes outside the frozen parent. Their scope and source mapping remain
explicit; Profile evidence from them does not substitute for full-parent Verify.

The same agent may implement several modules. Independent coding/research can run in
parallel on separate copies; GPU calls are serialized unless the parent permits disjoint
measurement resources. Preserve each task's required device/rank width and any assigned
gang; pool capacity is not an implicit rank count. Resource allocation and release belong
to the parent.

`BUDGET` bounds explored directions per attempt, including parallel alternatives,
not module calls or source-file counts. All calls consume wall time. `MIN_IMPROVE`
governs inner advancement; native PASS, useful measured gains and parent KEEP remain
separate. Preserve useful tested alternatives without claiming their admission.
Use the native full-task aggregate and state each comparison source and scope.

`ATTEMPT_CAP` bounds this method call, including inner tests and handoff. The worker's
outer measurement follows the return; its reservation belongs to the parent. Do not
subtract that reservation again unless the caller explicitly includes it in this window.
`MEASURE_TIMEOUT` is the outer hook's hard limit, not a test-duration estimate. Plan
from observed costs and do not reserve optional tests as mandatory work. Preserve calls
and evidence before changing source or exiting; do not pad time or switch methods.

## Acceptance and ownership

Read the selected module's `README.md` for its criterion; implementations must preserve
it. Native receipts own validity and measurements. Retain raw outputs and generated
receipts before reuse can overwrite them. An internal candidate is provisional until
the parent accepts it; never edit worker status, credit or accepted snapshots.

The parent [Optimize contract](../../README.md) owns cumulative accounting, KEEP,
rollback and final cold verification. An inner delivery does not complete its budget
or establish E2E gain. Performance comparisons with another search method require
controlled GPU experiments; CPU adapter checks do not establish a performance floor.
