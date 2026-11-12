# Kernel module call contract

Use workload's [direct-call contract](../../../CONTRACT.md#direct-module-calls).
The attempt handoff supplies paths and limits; do not assume they are shell exports.
Each call records its module and implementation path, question, parent call, input/source
identity, scope, resource/time budget and decisive check in `PLAN.md`. Keep its outputs
and `REPORT.md` under a distinct attempt subdirectory. Reports include status, start/end
times, native evidence and unresolved work. Missing, invalid and regressed results differ.

## Shared boundaries

Only declared editable sources may change. The original baseline, harness, inputs,
weights, build/source bindings and undeclared companions remain fixed. Use the existing
[source tools](../../../../engine/campaign/source_tree.py) for identity and copies.
Candidate copies are independent; the canonical task is a serialized test/delivery
location. Restore its prior source after a diagnostic or verification call has finished.
A required UT/build-boundary change returns to the parent for a new qualified task.

The same agent may implement several modules. Independent coding/research can run in
parallel on separate copies; GPU calls are serialized unless the parent supplies disjoint
measurement resources. Preserve the complete assigned device/rank set, including for
collectives. Resource allocation and release belong to the parent.

`BUDGET` bounds optimization directions per attempt, including parallel alternatives,
not module calls. All calls consume wall time. `MIN_IMPROVE` is the configured relative
gain for advancing the inner incumbent, not a replacement for native validity or outer
KEEP. Use the native full-task aggregate. Plan within `ATTEMPT_CAP`, including tests
and handoff; check observed test cost before launching measurements. Preserve active
calls and evidence before changing source or exiting; do not pad time or switch methods.

## Acceptance and ownership

Read the selected module's `README.md` for its criterion; implementations must preserve
it. Native receipts own validity and measurements. Retain raw outputs and generated
receipts before reuse can overwrite them. An internal candidate is provisional until
the parent accepts it; never edit worker status, credit or accepted snapshots.

The parent [Optimize contract](../../README.md) owns cumulative accounting, KEEP,
rollback and final cold verification. An inner delivery does not complete its budget
or establish E2E gain. Performance comparisons with another search method require
controlled GPU experiments; CPU adapter checks do not establish a performance floor.
