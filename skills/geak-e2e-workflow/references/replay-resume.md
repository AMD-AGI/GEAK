# AgentX trace replay and phase resume

Read only the section relevant to the run. Use the checked-out
`e2e_workflow/README.md` and `interface/run_e2e.py` for the latest field contract.

## AgentX is a replay workload

Synthetic ISL/OSL/concurrency describes fixed-shape requests. AgentX's client
owns request mix, arrival pattern, warmup and measurement duration. Declare
it explicitly; do not replace replay with a synthetic benchmark at average lengths.

For the Python handoff use a `workload_spec` object, for example:

```json
{
  "kind": "agentx_trace_replay",
  "metric_basis": "p90_intvty_inferencex",
  "concurrency": 8
}
```

This object belongs at `handoff.workload_spec`, alongside the normal required
model and output fields. The current Python interface forwards this object;
a top-level `workload_kind` shortcut is not a replacement there. A direct
Workflow call can use `workload_kind: "agentx_trace_replay"` or the same
`workload_spec` object.

Preserve an upstream caller's complete corpus, entry count, durations, client,
metric and acceptance contract. For a standalone default, the checked revision
expands the declaration to the canonical corpus/window described in its README;
read those defaults and disclose them before launch. Do not invent a custom
short corpus and claim a canonical result.

Before launch, in the actual container:

```bash
MODEL=/models/Kimi-K3 bash "$GEAK_ROOT/e2e_workflow/scripts/agentx_smoke.sh"
```

Replace the model path. This checks environment/model without starting a
server. `BASE_URL` optionally adds a real short replay against an existing
authorized server. The selected `aiperf` must support `--scenario`, and the
mapper must produce the requested metric. The native Python interface also
checks the selected client; stock aiperf is not enough for this replay.

Default standalone scoring is `p90_intvty_inferencex = 1000 / P90(ITL_ms)` in
tokens/s/user, decode only. Total input+output tokens/s is a guard. Do not
substitute output-only throughput or TTFT-inclusive normalized interactivity.
Preserve an explicitly requested metric and matching environment settings.

Use `observed_isl`/`observed_osl` only as workload targeting information when
known; otherwise GEAK adopts shapes measured from its own baseline. They do
not define replay request lengths. Report shape provenance.

Inner search windows may be shorter than the canonical final measurement.
Inspect `submission_valid` and comparability metadata before labeling a result
canonical. Changing `corpus`, `num_entries` or `duration_s` does not change
the `canonical_*` reference automatically. Allow enough wall time for baseline
and final replay windows, server startup/warmup and final validation.

## Resume a pinned e2e run

1. Locate the exact original handoff, GEAK SHA, `eval_dir`, logs and latest
   `workflow_return.json`. Determine whether the parent/session and GPU workers
   are still running before starting another process.
2. Read the prior return's `state` and on-disk phase/gate artifacts. Preserve
   model, GPU identity, workload, baseline and measurement contract.
3. Create a new handoff file for the continuation. Pin the same `eval_dir`,
   copy the **state object** when available, and choose `phases` from the current
   contract: `setup,profile,config,tune,head,kernel,final`. Do not resume Setup
   over an established run accidentally.
4. For pending final A/B work use `phases: "final"` with the same `eval_dir`.
   Let GEAK reconstruct its accepted/pending changes and finish measurements.
   When state is absent, check the current disk-recovery implementation before
   assuming other phases can reconstruct everything.
5. Dry-run the continuation, give it a new log/result path and sufficient
   timeout, then launch using the same validated runtime. Preserve prior results.

The interface's completed-run recovery can return an already persisted result
without repeating GPU work. Verify which path ran. Neither an old log line nor
a saved candidate patch proves a continuation completed fresh validation.

This is phase continuation, not a generic Claude conversation resume. Do not
invent a `--resume` flag for `run_e2e.py`, or select an experiment by latest mtime.
