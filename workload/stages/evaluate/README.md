# Evaluate

Compare the declared reference and candidate systems using their native serving receipts.
[Native E2E](native-e2e/SKILL.md) is the initial implementation; another implementation
must satisfy the same workload, quality and comparison contract.

`finalize.py` owns the shared A/B calculation and receipt admission. It consumes the
existing `e2e/bench_summary.json` convention, including an admissible fresh `stock_control`
(the unchanged reference for this comparison), and the configured `native_validation`.
`reference_baseline`, when supplied in `PARAMS`, names the historical reference receipt;
otherwise the standard recipe's `RUN_ROOT/baseline/out/baseline.json` is used. Relative
configured paths resolve from the repository root.

The call writes `ab.json`, `REPORT.md` and `result.json`. Set `OUT`, `PARAMS`, `STAGE=evaluate`
and `STAGE_SKILL=native-e2e` for a direct call. The standard applyback finalizer first
checks kernel integration, then calls this same evaluator with that verified metadata;
it keeps its existing output envelope and does not calculate a second verdict.

`ok` means the comparison is admissible, not that the candidate won or met the task's
performance floor. The controller reads all required objectives and the uncertainty
policy before adoption. Preserve the reference when the candidate is worse or inconclusive.
