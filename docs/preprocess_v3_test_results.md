# GEAK v3 preprocess — HIP sweep results

End-to-end validation of the v3 preprocess pipeline (orchestrator + 3
always-on subagents + per-language KB injection + Path-A short-circuit)
against 3 AKA HIP kernels: `ball_query`, `matrix_multiplication`, `silu`.

This document is the human-readable surface; the machine-readable
counterpart is the per-sweep `validation_runs/<timestamp>/summary.json`
that the test runner emits. The runner appends one section per sweep
(smoke → pilot → full) underneath this preamble.

## Methodology

* **Runner**: `scripts/preprocess_v3_test_runner.py` — parent iterates
  the YAML plan and spawns one Python child per scenario × run; child
  invokes the v3 orchestrator in-process (bypassing the `geak` CLI to
  avoid the optimisation loop, which is out of scope for preprocess
  testing).
* **Plan**: `tests/test_plan.yaml` — 3 kernels × 5 scenario kinds; the
  runner derives two additional record types (path_b_coverage and
  cross_language) by piggybacking on `path_b_determinism`'s run1.
* **Oracle**: `scripts/preprocess_v3_oracle.py` — AST-extracts
  `TEST_SHAPES` from each kernel's `scripts/task_runner.py`. The
  integer-prefix of each shape tuple is the signature used for set
  containment checks against the harness-generator's emitted shapes.
* **Per-invocation timeout**: 1800 s (hard wall clock, enforced by the
  parent's `subprocess.run(..., timeout=1800)`).
* **Status legend**: `pass` (every check ok), `warn` (only warning-
  flagged checks failed), `fail` (a non-warning check failed), `error`
  (timeout or child crash before assertions ran).

## Sweeps

(Sweep sections are appended below this line by the runner — newest
first. Each sweep section reports per-scenario status, key wall-clocks,
and any anomalies worth surfacing for v3 follow-up work.)

<!-- SWEEP-INSERTION-POINT -->
