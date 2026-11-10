# EVO runtime configuration

This configuration reference applies to runtime
`37501053c5c92dcc180087a29a730877043193a2`.
GEAK-EVO uses the shared kernel workflow with retained candidate source,
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
  "deep_cost": 2,
  "use_expert_skills": false,
  "use_learned_kb": false
}
```

Selecting `optimizer: "evo"` alone uses the default direction budget of 6 and
leaves development return off. Set the options explicitly when using the
configuration above. The optional flags control distinct evidence and
initialization paths; see the [run guide](stock-workflow.md) and
[archive fields](archive.md).

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
