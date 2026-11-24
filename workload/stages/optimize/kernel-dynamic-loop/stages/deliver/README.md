# Deliver

Hand source and evidence to the parent under the [shared contract](../CONTRACT.md).
Initial implementation: [source-handoff](source-handoff/SKILL.md).

**Input:** candidate identities, native evidence, remaining work and the attempt boundary.
**Output:** one selected source in the canonical task and `RUN_DIR/HANDOFF.md` with evidence.
**Acceptance:** frozen companions are preserved; native test status and inner admission
are explicit. Select a source for the parent's next acceptance check, preserving its
verified fallback. A source awaiting that check is not a parent win, even if its inner
test passed. With no useful candidate, retain the accepted source. Parent receipts own KEEP.
