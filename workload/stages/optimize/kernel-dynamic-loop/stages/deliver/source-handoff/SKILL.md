---
name: source-handoff
description: Hand one selected kernel candidate and its verification evidence to the parent worker while preserving pending and blocked outcomes.
---

# Source handoff

Read the [Deliver contract](../README.md).
Drain owned calls within the attempt boundary and preserve their output. Choose the
best source-bound candidate for the parent's next acceptance check, using its declared
comparison and rule. Consider tested small gains as well as the inner incumbent;
failure to advance the inner incumbent does not erase a useful validated alternative.
An unchecked alternative does not displace a ready verified improvement. If none is
ready, a promising unverified source may be submitted explicitly as pending. Retain
the accepted fallback and place only the chosen declared edits in the canonical task.

Write `RUN_DIR/HANDOFF.md` with selected source identity, native test status, comparison
basis and inner admission separately; include useful alternatives, call report links,
unfinished work, costs, blockers and the next useful plan.
The worker's subsequent receipt overrides earlier claims. Hand control back without
marking its campaign complete or repeating a worker-owned sweep.

For a requested route review, reuse the existing
[HTML report module](../../../../../report/html-timeline/SKILL.md).
