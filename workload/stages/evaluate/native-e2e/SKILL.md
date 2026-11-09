---
name: native-e2e
description: Measure and compare two declared serving systems with the native workload and quality protocol, retaining raw evidence and judging the result through the shared Evaluate finalizer.
---

# Native serving comparison

Read the [invocation contract](../../CONTRACT.md#direct-module-calls) and
[Evaluate's artifact interface](../README.md). Inputs are the reference and candidate
identities, integration/effect evidence, workload, quality requirements, budget and
comparison policy. Resolve them before launching measured traffic.

Use the native harness already selected for the task. Run the reference and candidate
on comparable isolated resources with matching workload and conditioning. Give each arm
its own runtime/cache identity; retain raw receipts and failed or slow runs. Quality and
actual-path checks must cover the candidate's change, including configuration changes.
Do not manufacture kernel artifacts for a system that changed only configuration.

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
