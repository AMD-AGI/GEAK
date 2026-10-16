# Run an admitted native task

`geak_evo.command.run_admitted_task` composes the shared search, final selection,
two full native evaluations and checked result export. It is a callable for the
existing native owner. It does not install services or adopt a generic lane task.

```python
result = run_admitted_task(
    owner_config_ref,
    store=store, services=host_services, evaluator=arena_evaluator,
    budget=native_budget, episode=native_evidence.episode,
    output_dir=new_output_directory, max_steps=decision_bound,
)
```

The owner supplies the already-admitted `ResearchStore`, `HostServices`,
`ArenaEvaluator`, the existing native host `Budget`, and its existing
`NativeEvidence.episode` reader. The same ledger is used by search probes and
final measurements. No second budget or lifecycle service is created.

The pinned owner JSON has these fields:

| Field | Value |
| --- | --- |
| `schema` | `geak-evo-admitted-command-v1` |
| `search` | `campaign_id`, `seed_candidate_id`, and the existing `prepare_native_search` keyword arguments, excluding `services` and `clock`. Serialize `profile` with `ModelProfile.as_dict()`. Paths are absolute strings. |
| `baseline_candidate_id` | Registered candidate containing the exact admitted task source. |
| `final_measurements` | `baseline_seconds` and `selected_seconds`, both equal to the existing native operation envelope. |
| `objective` | The common lane's objective mapping, unchanged: `kind` is `geomean` or `time_weighted`; weighted scoring requires a positive finite `weights` value for every case. Workload provenance fields are retained. |

`search.shared_budget` must equal the supplied ledger's configuration and retain
the original session, boot and deadline. Search uses that full original deadline;
this command does not reserve final headroom in advance. The caller must bound
search decisions and evaluations to leave time and native call capacity for
both final evaluations. The existing ledger admits only one operation at a time.
If the remaining budget cannot admit a full final operation, the command writes
an explicit incomplete result with null metrics; it does not extend the clock.

An installed owner entrypoint can call `geak_evo.command.main(argv, store=...,
services=..., evaluator=..., budget=..., episode=...)`. Its arguments are
`--owner-config PATH SHA256 --output-dir NEW_DIRECTORY --max-steps N`.
It needs no provider import loader or model command on the command line.

The command fixes its config and decision bound on first use, records search
intent, then freezes the returned committed revision. It checks that revision
is still current before final work and export. An interrupted search without a
frozen return requires owner reconciliation; retry cannot fund another batch.
Final evaluation IDs are derived after selection. A retry uses the same IDs,
requires existing evaluator request and dispatch intent, and can settle after
the deadline without launching again. Unsettled reservations retain elapsed
time from their original start; already-settled costs remain unchanged.
Recovery after the deadline closes the same operation's accounting. An accrued
operation-envelope overrun remains an incomplete result even after cleanup settles.

Each invocation writes `result.json` to a new output directory. A missing or
cleared recommendation never selects a fallback. Budget refusal, unsettled
work, a failed baseline or changed campaign gives empty `final_patch` and null
metrics. A passing result retains exact source, measurements and provisional
flags through `export_final_result`.

The [native dispatcher adapter](dispatch-adapter.md) converts the common
task/result files around this callable. See [current support](../../geak_evo/README.md#current-support)
for the supported timing formats. The result's `improved` means measured speedup
above 1; the common lane still gates
writeback against `incumbent_speedup * (1 + min_improvement)` and performs its
existing final validation. Exit status 0 is not final framework acceptance.

CPU tests cover actual shared search reaching a pending synthetic provider
result, and synthetic selection followed by the real Arena evaluator and
checked export. They also cover same-ID final recovery, retained costs, failed
baseline replay, budget refusal, stale revisions and interrupted search.
These tests do not qualify a model, device, installed owner or live run.
