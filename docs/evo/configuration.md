# EVO search and runtime configuration

This reference describes the current branch. EVO is the default optimizer;
the settings below explicitly configure a longer search.
GEAK-evo uses the shared kernel workflow with retained candidate source,
parent selection, independent verification and final source selection.
Correct, slower measurements can be retained without promoting the candidate.

## Explicit optimizer settings

Merge these settings into a qualified task's launch arguments:

```json
{
  "optimizer": "evo",
  "evolution_start": "first_stall",
  "evolution_return_development": true,
  "evolution_import_stall_sources": false,
  "evolution_case_evidence_view": false,
  "evolution_preserve_founder_attempts": false,
  "evolution_evidence_handoff": false,
  "budget": 96,
  "max_no_improve": 96,
  "max_forced_replans": 96,
  "min_improve": 0.02,
  "progress_delta": 0.02,
  "candidate_floor": 1.0,
  "deep_cost": 2
}
```

The small default run has a direction budget of 6 and a stall limit of 2. It
enables development return after EVO activates. Use explicit larger budgets and
stall limits for sustained development; enabling EVO alone does not allocate a
longer run. First-stall activation preserves the current stall count. The optional
flags control distinct evidence and initialization paths; see the
[run guide](stock-workflow.md) and [archive fields](archive.md).

These search settings leave the task's model, knowledge and skill configuration
to the shared GEAK controls; a longer search need not disable those capabilities.

## Sustained development and time allocation

EVO preserves the shared specialist and deep-engineer loops and their role
allowances. Choose search controls together:

- `budget`, `max_no_improve` and `max_forced_replans` govern how much development
  the loop admits. A large direction budget alone does not prevent a stall stop.
- `deep_cost` accounts for a dedicated deep-engineering round. Deep engineers
  retain their own iterative implementation, profiling and measurement loop.
- Model concurrency allows simultaneous agent work. GPU evaluations acquire the
  assigned device/group locks, so choose the number of directions for that pool's
  capacity. Engineering and verification overlap across directions; integration
  waits for the round's pipeline.
- For a direct `kernel_lane.js` run, `preparation_allowance_s` and
  `search_duration_s` can reserve a full search window after preparation. They
  require the process-bounded runtime, an outer `deadline_epoch`, and ordered
  search/final reserves. The dispatcher `kernel_workflow.js` rejects these two
  relative-time arguments. Keep using the enclosing workflow's supported clocks
  for Hyperloom and serving invocations.
- `search_reserve_s`, `final_validation_reserve_s` and `final_cleanup_reserve_s`
  reserve closure phases. Optional `engineer_reserve_s` ends planning and
  engineering earlier to leave time for verification/integration; it is a
  user-declared allocation, so size it from the task's measured costs.
- `delivery_facts: true` exposes observed role-wall and available GPU-wait facts
  to planning and engineering in a process-bounded run. These observations include
  waits and are not a complete evaluation-service cost model.

The aim is to give agents useful development time while completing the numerical
contract and final delivery. Queueing, archive work and evaluation still consume
wall time; inspect the recorded phase/role timelines when adjusting the allocation.
Keep measurement quality and final verification intact when reducing overhead.

## Checkpoint migration

For independently launched, compatible lineages, merge these settings into each
lineage's complete task arguments:

```json
{
  "optimizer": "evo",
  "evolution_start": "search_start",
  "eval_dir": "/absolute/path/private-lineage-a-evaluation",
  "population_lineage": {
    "root": "/absolute/path/shared-population",
    "id": "lineage_a",
    "imports": ["lineage_b"]
  }
}
```

Give each lineage its own fresh evaluation directory, unique ID and assigned
resources. Use a shared population directory and explicitly name allowed peers;
configure the other lineage with its own ID and imports. Import requires a
compatible source base and metric plus the checkpoint's applicable verification
and source checks. Imported programs become archive parents/donors; they do not
automatically replace the local incumbent. Migration shares source and evidence;
the caller still launches and budgets each lineage.

`search_start` is required for this import path: the default `first_stall`
continuation creates a new local archive rooted at its later incumbent.

## Task and resource binding

Keep the task's model/backend, fixture hashes, numerical contract, workload
weights, hardware, device allocation and measurement commands with its launch
configuration. Relative search requires a positive `preparation_allowance_s`,
an absolute `deadline_epoch`, ordered search/final reserves and the bounded
process runtime. Generate absolute clocks through the existing launch path.
Independent complete-operation evaluation has its own declared allowance.

`GEAK_GPU_LOCK_WAIT` sets the GPU acquisition wait in seconds. Record its
effective value with the task arguments. Preserve the declared per-arm budget
and record actual preparation, search and evaluation costs.

The `0.02` promotion setting is a search decision threshold. Candidate selection
uses the task's declared primary objective, numerical contract and verification
result. Retention, promotion and delivery are separate operations.

## Source and measurement records

Pin the runtime and effective arguments for a reproducible run. Retain baseline
and candidate measurements, case identities, correctness results and timing
quality with the exact tested source. Use the complete-operation measurement
for delivery, and the enclosing workload's evaluation for serving integration.
