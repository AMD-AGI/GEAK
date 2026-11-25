# Module contract

A module defines a capability and its input/output/acceptance criteria in `README.md`.
Implementations live in `<implementation>/SKILL.md`; start with one general method and
add another only for a distinct need. A module is not a position in an execution order.
[Brain](../brain/SKILL.md) owns planning and coordination.

## Direct module calls

Each invocation's `PLAN.md` names the question, module and implementation path, input
identities, scope, output directory, constraints, resource/time budget and decisive
evidence. State which request or observation prompted the call, which decision it should
inform and which prerequisites it actually consumes. Name the source of required validation
and the next action each possible result would support. The task and explicit plan govern
scope, including over implementation defaults.

Invoke the skill or tool with those inputs and a separate candidate/output directory.
A source snapshot is not the complete input contract: preserve declared data and fixture
dependencies when copying or isolating a call, including its implementation's workspaces.
A call may use part of a larger implementation, combine tools or return a running handle.
For ongoing work, return its location, progress/blockers and affected input identities
so the controller can respond before completion.

Return native receipts/logs and `REPORT.md` with status, findings, changed artifacts and
uncertainty, including failures. Answer the plan's question: what was tested or ruled out,
what bottleneck/opportunity remains, and which decision or next question/module it supports.
Report actual progress and preparation/search/validation costs where observed; unknown
costs remain unknown. Distinguish a measured negative result from missing evidence or
execution failure; identify reusable evidence, affected work and what would unblock the
missing delivery. Suggestions are evidence for Brain, which chooses the next call.
Keep receipts attributable to their invocation, source and harness; preserve them before
reuse. Reuse evidence whose claim-relevant inputs still match; identify the changed dependency
before repeating it. A new output directory alone does not invalidate a receipt.
Repair presentation from matching evidence, never by inventing missing facts.

Use the validation owner for the claim: a diagnostic observation, qualified UT,
cold-verified kernel and serving comparison are different deliveries. Changing a method
does not change acceptance. Existing finalizers remain authoritative for their full
deliveries; supply their declared inputs and environment.
The [sequential driver protocol](../engine/core/README.md) describes its environment,
`result.json` and unconditional finalizers; these are not prerequisites for every
scoped module call.
