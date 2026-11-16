---
name: workload-runtime
description: Prepare or recover the permitted workload runtime and resources for a planned module call.
---

# Runtime preparation

Read the [Setup contract](../README.md). Reuse a compatible environment; otherwise
prepare the runtime specified by the plan using its native setup and resource tools.
Resolve the actual image/source versions, workload inputs, tool entrypoints and resource
availability. Use the selected benchmark's readiness checks when preparing a measurement;
[measure](../../baseline/measure/SKILL.md) describes the bundled benchmark path.

Return the usable handles, observed identities, remaining allocation time and blockers
in `REPORT.md`. Preserve existing work and record what was changed. Preparing or
recovering an environment does not authorize a benchmark or a resource release.
