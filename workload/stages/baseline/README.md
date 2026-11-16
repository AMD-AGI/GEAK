# Baseline

Establish or revalidate a measured reference under the
[invocation contract](../CONTRACT.md#direct-module-calls).
[Setup](../setup/README.md) prepares resources; preparation alone is not a baseline.

**Input:** the named reference system, scenario and compatible measured receipt or runtime.
**Output:** `baseline.json` and supporting native receipts.
**Acceptance:** [finalize.py](finalize.py) owns receipt validity, configured alignment
and drift admission. Run it with this invocation's `OUT` and resolved `PARAMS` before
accepting a reference.

## Implementations

| Skill | Use |
| --- | --- |
| [adopt](adopt/SKILL.md) | Reuse a compatible, trusted measured receipt |
| [measure](measure/SKILL.md) | Measure the named runtime; prove configured reproduction pins |

A changed environment needs matching evidence, not a relabelled receipt.
Keep the reference distinct from candidates, including when the reference is itself optimized.

## Artifact

Both methods produce `baseline.json` with `metric`, `unit`, `throughput`,
`higher_is_better`, `noise_band`, `config`, `measured` and `provenance`.
`reference`, when present, records the named comparison value and ratio.
Provenance names the receipt, its hash, node and measurement mode; pinned reproductions
also retain the alignment checks and observed stack/cache identities.

`noise_band` is configured E2E decision policy, not measured variability or kernel
KEEP policy. [Evaluate](../evaluate/README.md) owns its use.
Comparisons require matching node and measurement mode; a reference from another
environment does not substitute for a fresh control.

The [driver](../../engine/core/README.md) runs `finalize.py` to emit `result.json`
and its measured metrics for either implementation.
