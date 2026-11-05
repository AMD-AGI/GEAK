---
name: kernel-profile
description: Profile serving workloads to locate GPU kernel dispatch, shapes, invocation frequencies and deployed source evidence. Supports shared sampling across configs and produces a hashed handoff for separate UT generation.
---

# Kernel dispatch profiling

Own workload observation and runtime evidence. Finish with `profile_handoff.json`;
UT authoring, source extraction into candidate tasks and GPU UT qualification belong
to the separate `kernel-ut` skill. A complete profile handoff does not claim a UT pass.

The pipeline supplies `$IN` (baseline output), `$OUT`, `$PARAMS`, `$CONFIG` and
`$WORKLOAD_ROOT`. Set `TOOLS=$WORKLOAD_ROOT/stages/profile/kernel-extract` and
`PROFILE_SLOT=$WORKLOAD_ROOT/stages/profile`. These are the existing shared tools;
do not follow the combined kernel-extract skill's UT stages. Read its
[setup](../kernel-extract/references/setup.md),
[workload measurement](../kernel-extract/references/dynamic-workloads.md) and
[selection coverage](../kernel-extract/references/selection-coverage.md).
Outside the pipeline set these paths explicitly and supply the same input artifacts.

## Observe and capture

1. Reuse the measured baseline's logs, effective launch settings, image/source/model
   identities, rank scope and compatible captures. Preserve per-config observations;
   client concurrency alone is not an execution shape. Keep `PROGRESS.md` in `$OUT`.
2. Declare `serving.json` from the actual deployment. Run the standard observation and
   point planners. With multiple configs, follow
   [shared sampling](../kernel-extract/references/multi-config-sampling.md):
   `multi_config.py plan` → `capture` → `project --merge` → `select`. This supports one
   config too. Inspect budget gaps before capture; retain them after achieved-shape
   relabeling. Lowest/highest concurrency do not replace middle-only dispatch evidence.
3. For an ordinary single config, run `probe_server.py` / `run_points.py` with
   **`--profile-only`**, under the full allocated GPU-set lease from setup. This collects
   census, kernel shapes, source map and adjacency without creating UT declarations.
   Preserve actual graph/eager modes and failed probe evidence. The shared-pool capture
   helper already passes `--profile-only` and holds the pool lease.
4. Apply the numeric selection policy in `$PARAMS`. Use complete operators, per-config
   then union where applicable; retain companions, collectives, tails and unsupported
   identities in the inventory/denominator. The split multiconfig example uses >3%.
   Keep `selection_scope`, `candidate_inventory` and `coverage_summary` in the ranking,
   as required by the existing coverage contract.

## Locate dispatch and freeze evidence

For every selected identity retain its observed points, achieved shapes, host-op
dimensions/layouts, launch grid/block, source frames and operator entry when available.
Keep recorded compile variants and call descriptors, including warmup-only dispatch,
alongside the census. Shape alone does not identify a compile variant. Locate installed
source/build provenance in the serving environment; missing native generated source or
an ambiguous callable is an explicit gap, not an invented implementation.

Use existing capture/shape/source-map scripts. Ordinary numerical payloads should be
seed descriptors; preserve only necessary structural state. Do not start an extra
capture merely to create a golden `.pt`. Stop the probe server when evidence is complete.

The [handoff contract](../HANDOFF.md) defines the boundary. Set profile parameter
`output_kind: "dispatch_evidence"`, then:

```bash
python3 "$PROFILE_SLOT/finalize.py"
```

It checks selection policy, measured denominators, counted-point census/shape evidence
and observed logs, writes `profile_handoff.json` with file hashes, and emits the profile
stage result. All selected identities are indexed, including unresolved entries with
their gaps. Missing evidence required to describe the profile blocks the handoff;
explicit source/UT-boundary questions remain visible for UT review. Report profile
sampling coverage/approximations separately from UT coverage, which is unset here.

For standalone export/reuse without modifying an existing profile directory:

```bash
python3 "$PROFILE_SLOT/handoff.py" export --profile-dir <evidence-dir> \
  --params <the-profile-selection-params.json> --out <handoff.json>
```

Do not run `build_task_plan.py`, scaffold UTs or start a UT wave in this stage. If UT
generation later requests more evidence, resume profiling and publish a new handoff;
the old handoff and its consumers remain auditable.
