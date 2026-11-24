---
name: native-slice
description: Derive lightweight, source-bound probes from an immutable native UT for candidate screening, without replacing full-task verification.
---

# Native task slice

Read the [Task contract](../README.md) and parent UT's actual call and timing contract.
Choose original cases that discriminate the hypothesis, considering workload cost,
branch/layout regimes and correctness boundaries. Preserve their inputs and dimensions.
Record selection rationale and omitted scope; do not invent a full-workload score.

Use a native case filter or callable helper when available. Otherwise create a small
external runner reusing original generators, references and timers. Keep the complete
candidate computation; isolate a stage only when its native interface and reference
are exposed. Native helpers that write overlays or oracles use private complete task
copies. Native binding checks own the executed source; delivery ancestry is recorded
separately and does not become another eligibility gate.

Write the runner/invocation and manifest in this call's output directory. Include
parent/source identities, selected case IDs, original measurement modes, comparison
source, uncovered scope and cost. Profile can compare candidates on assigned resources.
Scoped results guide the next plan; Verify on the complete parent owns admission.
Report a limit when a faithful probe cannot be constructed.
