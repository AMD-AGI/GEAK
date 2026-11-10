# GEAK external orchestrator contract

Hyperloom and other callers use `interface/run_e2e.py` to pass a task to GEAK and
receive measured results. The maintained specification is
[the external integration contract](../../interface/run_e2e.md).

Use that specification for:

- [Command and options](../../interface/run_e2e.md#command), including the agent backend.
- [Input and output schemas](../../interface/run_e2e.md): `handoff.json`, `result.json`,
  and the per-kernel `kernel_journey.json` records.
- Baseline and candidate measurement alignment, acceptance rules, retained evidence,
  and recovery when a workflow stops before returning.
- [Failure results](../../interface/run_e2e.md#when-there-is-no-result-to-report):
  the `timeout` and `error` statuses, `error_class` (including `llm_unreachable` and
  `agentx_client_unavailable`) and `phase_reached`.
- Reusing a deployment bundle for workload sweeps.

For workflow arguments and local usage, see the
[API reference](api-reference.md), [Run a workflow](../how-to/run-agent.md) and
[GEAK guide](geak-guide.md).
