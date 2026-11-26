---
name: native-e2e
description: Compare two serving systems on native workload performance and paired GSM8K accuracy, retaining the evidence for both decisions.
---

# Native serving comparison

Read the [invocation contract](../../CONTRACT.md#direct-module-calls) and
[Evaluate's artifact interface](../README.md). Inputs are the reference and candidate
identities, integration/effect evidence, workload, quality requirements, budget and
comparison policy. Resolve them before launching measured traffic.
Use native input preparation to resolve required run metadata in the intended client context.

For a scoped screen, follow Evaluate's screening contract and return matched native
observations. The complete receipt and finalization steps below apply to full comparisons.

Use the native harness already selected for the task. Run the reference and candidate
on comparable isolated resources with matching workload and conditioning. Give each arm
its own runtime/cache identity; retain raw receipts and failed or slow runs. Quality and
actual-path checks must cover the candidate's change, including configuration changes.
Do not manufacture kernel artifacts for a system that changed only configuration.

Apply the [Evaluate quality criterion](../README.md#model-correctness) to every full
serving comparison and to any screen that will judge model correctness.
Run the [GSM8K client](../../../../e2e_workflow/scripts/gsm8k_eval.py) once against
each arm's actual serving deployment, outside timed performance traffic. Set `ARM` to
`reference` or `candidate`, `BASE_URL` to that server's OpenAI-compatible `/v1` URL,
`MODEL` to its served model name and `LIMIT` to the sample size for this decision.

```bash
python3 "$GEAK_ROOT/e2e_workflow/scripts/gsm8k_eval.py" \
  --base-url "$BASE_URL" --model "$MODEL" --limit "$LIMIT" --fewshot 5 --seed 0 \
  --out "$OUT/e2e/gsm8k/$ARM.json"
```

Use the same dataset snapshot, client version, few-shot examples, greedy decoding
settings and generation budget on both arms. Record that protocol and the deployment
identity for each receipt. Verify both receipts cover the same requested set of distinct
question IDs and gold answers. The client counts request failures as misses. Check the
serving logs for failed requests before admitting an arm. A completed response without
an extractable answer counts as incorrect.

Write `$OUT/e2e/gsm8k/comparison.json` with both receipt paths, correct/total and
accuracy per arm, signed change in percentage points, tolerance and quality decision.
Return those figures with the performance report. Reuse a scored arm only when its
deployment, question set and evaluation protocol match the current comparison.

Write `e2e/bench_summary.json`: candidate throughput, metric basis, repeats/arm records,
native exit status and an admissible fresh `stock_control` with its provenance. Retain
native P90 interactivity when required; inverse rounded ITL is not its substitute. Point
`native_validation` at the native validity/effect receipt and required values. Keep
historical drift separate from the fresh comparison. A missing prerequisite is blocked.

With the call's environment prepared, run:

```bash
python3 "$WORKLOAD_ROOT/stages/evaluate/finalize.py"
```

Return its receipt and `REPORT.md`, including negative findings and uncertainty. A
configuration or implementation bundle's total effect does not identify the contribution
of each edit. The task's acceptance/floor requirements still govern adoption; a local
kernel speedup or an admissible comparison alone is not sufficient.
