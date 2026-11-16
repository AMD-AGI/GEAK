# UT

Define and qualify the computation boundary for an optimization task under the
[invocation contract](../CONTRACT.md#direct-module-calls).

**Input:** selected workload evidence, source/callable identities, planned scope and runtime.
**Output:** independently runnable frozen tasks, editable-source declarations, native
qualification/delivery receipts and the covered or deferred identities.
**Acceptance:** the native task and delivery gates pass for representative inputs,
the actual device width and the declared edit boundary. CPU preflight alone is incomplete.

Implementation: [kernel-extract](kernel-extract/SKILL.md), reusing the vendored extractor.
A boundary, harness or case change produces a new qualified delivery. Preserve the old
version and report which dependent tasks need replacement; unrelated tasks stay valid.
