# Task

Derive cheaper, faithful optimization probes under the [shared contract](../CONTRACT.md).
Initial implementation: [native-slice](native-slice/SKILL.md).

**Input:** the immutable parent task, question, source versions and probe budget.
**Output:** a scoped test or native invocation, parent/case/source mapping, coverage
limits and observed or estimated cost.
**Acceptance:** parent files remain unchanged; selected native references, calls,
bindings and measurement semantics are preserved. Results apply only to the declared
scope. Profile may run it for screening; [Verify](../verify/README.md) owns admission.
