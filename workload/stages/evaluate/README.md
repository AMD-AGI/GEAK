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

The E2E decision threshold is `PARAMS.noise_band`, then the historical receipt's
`noise_band`, then `stages.baseline.params.noise_band` in `config/default.json`.
Explicit scenario values remain overrides. This is a policy threshold, separate from
kernel `keep_threshold` and measured repeat variability. When changing policy for an
existing comparison, retain its receipts and old verdict and re-evaluate in a new `OUT`.

The call writes `ab.json`, `REPORT.md` and `result.json`. Set `OUT`, `PARAMS`, `STAGE=evaluate`
and `STAGE_SKILL=native-e2e` for a direct call. The standard applyback finalizer first
checks kernel integration, then calls this same evaluator with that verified metadata;
it keeps its existing output envelope and does not calculate a second verdict.

`ok` means the comparison is admissible, not that the candidate won or met the task's
performance floor. The controller reads all required objectives and the uncertainty
policy before adoption. Preserve the reference when the candidate is worse or inconclusive.

## Reference modes

- **stock control** (default): the A/B denominator is the fresh unchanged arm the benchmark
  measured in the same comparison (`e2e/bench_summary.json` `stock_control`, admissible).
- **`reference: ci`** (multi-config scenarios, which never re-measure a baseline): the
  denominator is the official number of the CI point the leg reproduced
  (`APPLYBACK_BASELINE`, else the baseline stage's `baseline.json`). The leg must show
  `alignment.json` aligned=true (it launched exactly the CI's image, args and env) and
  `pins.json` proven=true (`stages/_shared/ci_pins.py`), at the CI's duration. A stock control,
  when present, is kept beside it as information.

In both modes a leg whose server died (`server_watch.json` from `stages/_shared/serving_watch.py`,
or for older legs a fatal line in its saved `server.log`) or whose measured window is under
98% of the configured one is refused.

