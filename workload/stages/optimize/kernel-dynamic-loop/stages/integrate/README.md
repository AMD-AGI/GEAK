# Integrate

Combine candidate changes under the [shared contract](../CONTRACT.md).
Initial implementation: [source-merge](source-merge/SKILL.md).

**Input:** candidate changes and a hypothesis that their effects are complementary.
**Output:** a combined source identity, constituent changes and unresolved interactions.
**Acceptance:** the combination respects the computation/edit boundary and retains its
provenance. It is a new candidate for Verify; component speedups cannot be added together.
