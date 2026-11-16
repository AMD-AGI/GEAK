# Apply back

Integrate a candidate and prove it took effect under the
[invocation contract](../CONTRACT.md#direct-module-calls).

**Input:** verified candidate bytes/configuration, binding metadata and target runtime.
**Output:** integrated runtime identity, loading/effect proof and a reversible handoff.
**Acceptance:** the intended runtime uses the qualified candidate with its identities intact.
Integration alone does not establish an E2E gain.

## Implementations

| Skill | Use |
| --- | --- |
| [overlay-ab](overlay-ab/SKILL.md) | Kernel overlay and loading proof; may compose the serving comparison |
| [system-runtime](system-runtime/SKILL.md) | Configuration, image/build or other system candidate |

For kernel integration, use the source-identity and actual-loading checks in
`overlay-ab`. The selected `kernel_src`, `kernel_src_hashes`, frozen
`task_meta_sha256` and any binding-file hash must match Optimize's cold evidence.
Task metadata supplies the binding, not a replacement working source.

Hand the proven runtime to [Evaluate](../evaluate/README.md) when comparison is requested.
A serving failure returns its native evidence and candidate identity to Brain for
routing to the owning module; do not rewrite frozen tests to make integration pass.

## Driver delivery

The sequential `applyback` stage composes kernel integration and Evaluate.
It consumes `optimize/out/optimized.json`, the qualified Profile task metadata and
`baseline/out/baseline.json` under `RUN_ROOT`. Split scenarios read task metadata from
`$TASKS_DIR` (`task_source_stage: "ut"`); legacy invocations fall back to
`$RUN_ROOT/profile/out/tasks`. With an `adopt-ci-artifacts` baseline, `points.py` selects
every CI point a UT covered and `ci_point_leg.sh` measures each in that point's own CI
configuration (`ci_config.py` alignment, `_shared/ci_pins.py` pins, `_shared/serving_watch.py`)
against its CI number (`reference: ci`); `aggregate_points.py` combines the points.

| Artifact | Purpose |
| --- | --- |
| `overlay/`, `applied.json` | Reversible integration, selected sources and hashes |
| `shadow_check.json` | Proof that the overlay shadows its targets |
| `e2e/bench_summary.json` | Native reference/candidate serving evidence |
| `ab.json`, `result.json` | Admissibility, comparison and stage verdict |

[finalize.py](finalize.py) checks integration and delegates the serving verdict to
Evaluate's criterion. Native validity, binding options, proof fields and benchmark commands
come from the scenario. Allocate the benchmark's full declared resources exclusively;
an E2E measurement cannot share devices with optimization or cold verification.
