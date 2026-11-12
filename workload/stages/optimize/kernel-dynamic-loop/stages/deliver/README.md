# Deliver

Hand source and evidence to the parent under the [shared contract](../CONTRACT.md).
Initial implementation: [source-handoff](source-handoff/SKILL.md).

**Input:** candidate identities, native evidence, remaining work and the attempt boundary.
**Output:** one selected source in the canonical task and `RUN_DIR/HANDOFF.md` with evidence.
**Acceptance:** frozen companions are preserved and verification status is explicit.
A candidate awaiting the worker's native sweep is pending, not a win. With no candidate
or a blocker, retain the prior accepted source. Parent receipts own KEEP and final acceptance.
