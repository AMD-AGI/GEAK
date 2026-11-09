# Setup and Baseline calls

Use the [invocation contract](../CONTRACT.md#direct-module-calls). Setup prepares only
what the plan needs: inspect the configured environment, permitted resources and native
harness requirements using the readiness steps of [measure](measure/SKILL.md). Report
available resources and blockers; preparing a machine does not establish a baseline.

For Baseline, [adopt](adopt/SKILL.md) a compatible measured receipt or
[measure](measure/SKILL.md) the named system. Give the call its own output directory
and resolved parameters; run `baseline/finalize.py` in that environment before accepting
the reference. Keep the original receipt and distinguish the reference system from any
candidate. A changed environment requires new matching evidence, not a relabelled run.
