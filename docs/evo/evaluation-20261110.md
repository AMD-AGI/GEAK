# Reviewed EVO code and evaluation status — 10 November 2026

The reviewed implementation is `37501053c5c92dcc180087a29a730877043193a2`.
It includes the shared workflow, retained-source archive, independent verification,
and the native regression-evidence repairs in `0deb3413` and `37501053`.
These repairs preserve correct, slower measurements without promoting them.
The code has software validation; a performance advantage over GEAK remains an
experimental question.

## Configuration is part of the design

Selecting `optimizer: "evo"` alone does not reproduce the evaluated configuration.
Its default direction budget is 6 and development return is off. The recent
eight-hour, first-stall comparisons used the following common optimizer settings:

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

The ten-arm engagement summary below also includes two declared single-factor
variants: GATE_EVO00 used `min_improve: 0.0`, and LR ON used
`evolution_import_stall_sources: true`. They did not use the common settings unchanged.

These are optimizer settings to merge into a qualified task's launch arguments,
not a standalone launch file. Preserve its model/backend, task and fixture hashes,
numerical contract, workload weights, hardware, device allocation, and measurement
commands. The timed runs also used `search_duration_s: 28800` and
`final_validation_reserve_s: 2400`. Relative search requires a positive
`preparation_allowance_s`, an absolute `deadline_epoch`, ordered search/final
reserves, and the process-bounded runtime; copying just the search duration is
invalid. Generate fresh absolute clocks through the existing launch path.
The independent full evaluation has its own declared allowance.

The GPU acquisition wait was 1,200 seconds. The reviewed code exposes it through
`GEAK_GPU_LOCK_WAIT`; record the effective value alongside the arguments.
Preserve the declared per-arm budgets when reproducing an assigned comparison;
some historical comparisons allowed GEAK four hours and EVO eight hours.
For a new matched-budget contrast against `optimizer: "greedy"`, keep the same
task, evaluator, model, hardware and budget. Record actual preparation, search
and evaluation time as well as assigned budgets. The `0.02` promotion gate is an internal search rule,
not statistical confidence or a guarantee against regression on a new workload.

## What the measurements establish

The following are individual assigned comparisons, not a pooled superiority test.
Each method's speedup uses its own measured original implementation as baseline.

| Comparison | Recorded result | Interpretation |
| --- | --- | --- |
| P04 | EVO 1.18938×; GEAK 1.04003×; ratio 1.14360 | A descriptive win in one pair. EVO never activated its archive, so the gain cannot be credited to evolution. GEAK's native OOM/unknown remains recorded separately from these full-final timings. |
| M01 | EVO 1.05133×; GEAK 1.04038×; ratio 1.01053 | A small descriptive difference in one pair. Evolution activated but produced no later commit. |
| M02 | EVO 1.76043×; GEAK 1.03618×; ratio 1.69895 | Strong timing nonstationarity makes the reliable comparative effect unknown. Keep the recorded result and its raw evidence. |
| R01 | Return ON/OFF ratio 1.01027 | This tests development return, not EVO versus GEAK. The ON original-versus-original sanity ratio was 4.64766; the reliable effect is unknown. |

Across ten heterogeneous first-stall EVO arms, four never activated evolution,
four activated without an evolutionary commit, and two produced such a commit.
This is an engagement observation. It does not establish why a run was slow,
prove that earlier activation helps, or identify the bytes ultimately delivered.
Archive insertion, parent selection, verified integration, delivered source and
full evaluation must be joined separately.

Twelve preserved candidates missed their scheduled final-evaluation windows.
Any later measurements have separate prospective scopes; they do not replace
the missing registered endpoints. Neither these comparisons nor this release
establish average EVO superiority, a current-recipe AgentX serving gain, or
85% attainment of a calibrated workload roofline.

## Features and their evidence

* **Configured, with limited engagement:** first-stall initialization and development
  return were enabled in all ten arms. Six activated evolution and two reached an
  evolutionary commit. These counts do not separately establish engagement or
  benefit for return-development, parent selection or integration.
* **Optional or unresolved:** stall-source import, case-evidence presentation,
  founder retention and search-start activation. Importing a source is not evidence
  of developing it; these options have no established general performance benefit.
* **Reviewed software changes awaiting full-run qualification:** prepared baseline
  handling, planner-input observation, continuation metadata, native invalidity
  handling, final-validation accounting, the configurable GPU wait, and retention
  of slower native/delivery measurements. Do not transfer performance credit from
  an older experiment runtime to these later commits.

Pin the historical runtime and its recorded effective arguments when reproducing
a historical run; current defaults are not a substitute. Test earlier archive
activation with the existing `evolution_start: "search_start"` option as a separately
declared contrast, retaining non-engaged and failed assigned arms in its readout.
Report original sanity, per-case baseline/candidate times, correctness and timing
quality before aggregate speedups. Timing-quality flags do not authorize trimming
or replacement measurements.
