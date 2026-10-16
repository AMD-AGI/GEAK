# Shared execution services

`geak_evo.host.prepare_native_search` composes the cleaned search directly from
the existing GEAK host inputs. It replaces construction inside the historical
`coordinator.host`; it does not copy that coordinator's import loader, root
publisher, process launcher or device manager.

The caller passes the admitted `config`, `admission`, `entry` and `collection`
file references, the original clock, a `ModelProfile`, and `HostServices`:

| Service | Existing implementation |
| --- | --- |
| Model transport and process boundary | `planner.model.ModelRuntime` containing the installed qualified transport, route, event and boundary services |
| Workspace preparation | The installed workspace module's `fresh_workspace` |
| Operation publication | The existing owner's `seal(directory, files)` channel |
| Native engineer | The installed host driver and collector, each by file reference |
| Episode evidence | The installed harvest module and `native_episode_material` |

The caller opens the `ResearchStore` and attaches its registered frontier before
preparation. The returned `NativeSearch` contains the real `EvolutionLoop`,
`NativePlanner` and `NativeAssignment`. `search.run(max_steps=...)` runs bounded
planner decisions and returns the saved search snapshot. Continuation, program
leases, cells, islands and allocation advice follow the supplied policy.

The campaign's `host-binding.json` retains the model profile, source references,
templates, read grants, original clock, policy and decision ceiling. Reopening
cannot increase those limits or replace the profile. An unresolved call must be
reconciled through the existing loop/planner/engineer recovery APIs using its
original identity before search can continue. Historical campaigns without this
binding require an explicit migration; this API is for prospective deployments.

`make_native_evaluator(store, config, callbacks=callbacks)` supplies the cleaned
`ArenaEvaluator` to the existing `NativeCallbacks.launch` and `settle` methods.
It uses the native-v2 task and `geak-arena-settlement-v1` response already used by
the native host. It does not adapt them into the separate `NativeEvaluator`
operator-task/v1 protocol. The owner still binds the helper, admission, wait
bound, physical device and lifecycle. Recovery calls `settle(None, ...)` for the
same operation; the evaluator never relaunches that identity.

The packaged engineer uses GEAK's existing Node `createRuntime`, with exactly one
model invocation and no schema retry. Its backend name now comes from the sealed
host model configuration and its model/effort from the pinned route. The existing
Python driver still owns execution and positive settlement.

The deployment must admit the packaged prompt directory in the existing read
policy and authenticate all supplied service implementations before construction.
Python objects are service bindings, not proof of installation custody.
An explicit `prompt_dir` forwards the loop's existing instruction-version choice;
the host binds every selected prompt reference, including the lease planner.
The default remains the packaged prompts. A compatibility pilot may select the
publisher's exact legacy prompt directory, but must report that instruction
version separately and cannot qualify the shortened prompts from that pilot.

CPU checks exercise direct search construction, one real planner/model boundary
using a synthetic process, pending-call retention, unchanged recovery identities,
native callback metrics and the Node engineer entry. They do not qualify model
access or a GPU provider. The common Hyperloom task converter and final measured
lane result remain outside this module: final selection, paired measurement,
export and full-workload validation must still use the existing owner path.

## Kernel result export

`geak_evo.result.export_final_result` replays one campaign's final recommendation
and reads its selected and explicit baseline evaluations from the existing Arena
owner root. The baseline must match admitted task bytes; source, case, runtime
and device identities must agree. The reader takes the existing owner lock and
checks the request's evaluator source identity. The updated evaluator saves
collection receipts before propagating interruptions; its source hash differs
from the frozen version. New installations must admit that exact version.
Saved operations keep their original authenticated runtime.

Numeric export supports published `task-report-v1` graph measurements
with phase cache receipts. It checks the actual report/request/phase hashes,
derives each mean from raw samples, and matches observed timer, replay-state
checks, seeded cache manifests and empty cache paths. Unsupported, missing or
incompatible timing evidence yields null metrics and no final patch.

Generated tasks use [their raw paired reader](evaluation-port.md#generated-paired-results).
The selected evaluation supplies both the external baseline leg and candidate
leg. The separate ORIGINAL evaluation remains a reference; its candidate timing
is never substituted for the selected evaluation's internal baseline.

The supported published `evaluation_contract.validate_report` derives each
scalar as `math.fsum(samples) / benchmark_iterations`. The host's
`published_protocol.validate_phase`, `published_observations` and Arena reader
copy it without another reduction. The exporter uses that same exact formula;
it does not accept differently reduced or caller-supplied means.

The exporter writes `result.json`, selection and measurement references, exact
selected source, and a text patch. Binary patch export is explicitly unsupported.
The optional `objective` argument is the lane's mapping (`kind`, `weights`, and
`workload_spec_path` / `workload` as supplied), retained without changing weight
values or provenance. `time_weighted` requires positive numeric time shares for
every benchmark case and computes `sum(w) / sum(w / paired_speedup)`, matching the
lane. Missing weights or cases are rejected. All-case geomean and arithmetic
ratios remain separate; omitting the objective defaults to geomean. A slower
selected program keeps its source and selected patch, with an empty `final_patch`.

Exporter `improved` means only a primary ratio above one. Before writeback, the
command/lane still requires that ratio to exceed
`incumbent_speedup * (1 + min_improvement)`; it must not treat the exporter label
as satisfying the lane's threshold.

The common result fields remain provisional. An applied clear, missing choice,
rejected measurement or unfinished campaign is not changed into a successful
run. Population authentication, Director validation and full-workload acceptance
remain with their existing callers. The exporter starts no task or model work.
CPU integration tests use synthetic owner collections and mock only the final
selection validator boundary; that validator has its own replay tests.
