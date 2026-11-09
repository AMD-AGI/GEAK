# Roofline calls

Use the [invocation contract](../CONTRACT.md#direct-module-calls). Select
[kernel-roofline](kernel-roofline/SKILL.md), a configured alternative, or the
[CPU fixture](stub-roofline/SKILL.md) for testing the interface.

Provide the declared task/case evidence and run the implementation's analysis and fold
as described in [the slot contract](README.md). Set `IN` to that explicit input delivery
and `OUT` to a distinct analysis directory, then run `roofline/finalize.py` there.
For a scoped analysis, use a preserved input view with the corresponding task ownership
and workload shares; do not rewrite the parent profile or extrapolate subset coverage.
Unknown bounds remain unknown. The report informs the next plan; it does not force a
kernel search or establish an E2E gain.
