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
Carry the configured runner's trust-store dependencies into its worker runtime along
with credential handles; the image's trust store may differ from the host's.
After a runner startup failure, test its configured request in the worker runtime
before redispatching the affected batch; a TLS handshake does not establish API availability.
Preserve the failure and choose another suitable runner if preparation remains blocked.
Follow resource selectors through native consumers that remask them; launcher affinity
does not prove physical isolation. Resolve unwritable runtime state through its owner's
supported controls before changing pinned sources.

When asked to watch for replacement capacity, inspect the authorized resource pool until
the plan's deadline or handoff. Report allocation changes promptly; distinguish a queued
job, an allocated node and a usable runtime. Check current occupancy, runtime access and
remaining lease before returning handles; a new job on the same host need not fix it.
Match each candidate to the declared resource scope, including any pool restriction,
using allocation metadata; account ownership or a job label alone may not establish it.

Return the usable handles, observed identities, remaining allocation time and blockers
in `REPORT.md`. Preserve existing work and record what was changed. Preparing or
recovering an environment does not authorize a benchmark or a resource release.
