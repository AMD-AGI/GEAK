# Evaluate

Compare the declared reference and candidate systems using their native serving receipts.
[Native E2E](native-e2e/SKILL.md) is the initial implementation; another implementation
must satisfy the same workload, quality and comparison contract.

Declare the decision this comparison supports. A scoped native screen may return matched
selection evidence under the [direct-call contract](../CONTRACT.md#direct-module-calls);
its restricted protocol and coverage are not full task acceptance. Complete comparisons
still use the finalizer below; a screen cannot replace required adoption evidence.
Choose its scope for a decision, not to complete an inherited list of tests. Keep workload,
conditioning and active instrumentation comparable even for a screen; extra repetitions
cannot repair a systematic mismatch.

## Model correctness

Use paired GSM8K final-answer exact match for model correctness. Score the unchanged
reference and the deployed candidate against gold answers on the same test questions.
Token IDs, generated text and conditioning-prompt hashes are diagnostic only. A
throughput-only screen leaves correctness unevaluated.

Use the merged config's `gsm8k` policy or the [workload defaults](../../config/default.json)
for a direct call. Use `screen_questions` for a quality screen and `finalist_questions`
for the adoption decision. Record any task-specific overrides before measuring.
Report correct/total and accuracy for each arm, then compute the signed change in
percentage points as `100 × (candidate_correct − reference_correct) / total`.
Quality passes when this change is at least `−max_drop_pp`. A larger drop is an
accuracy regression. A missing, incomplete or unmatched pair leaves quality unevaluated
and blocks adoption. Carry the quality decision and both receipts alongside the native
performance verdict.

## Performance comparison

Resolve each arm's repeats, required quality and deployment reuse from its native owner.
The finalizer's `repeats` applies to the candidate; a fresh stock control has its own
admission requirements. Do not mirror one arm's count onto another. Reuse compatible
quality evidence under its declared scope; a new client result identity alone requires no
new quality run or server restart. Retain valid slow repeats in the declared statistic;
keep invalid receipts and their native reasons. Cancellation, request errors and normal
drain retain the harness's verdict rather than new admission rules after the fact.
Each measured repeat keeps its own native run identity and raw receipt; reusing a measurement
does not count as an independent repeat.

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
