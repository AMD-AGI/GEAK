# Evaluation port

External whole-call UTs can also use the existing Arena request/settlement
boundary. The first supported external method is GLM `weight_lib` v2: three
fresh processes, its unchanged minimum-two-process gain gate, and separate
phase folds. See the [ORIGINAL-only MQA task](../../examples/tasks/glm_mqa/README.md).
This route adds no device manager and does not acquire graph or serving
qualification. `external_primary` exports the method's native primary-phase
score; it never substitutes a cross-phase geomean.

The engine retains both existing evaluation contracts. `ArenaEvaluator` records
provisional native task observations; `NativeEvaluator` retains the separate
workload-native correctness, repeated timing and eligibility rules. A result
from one contract does not acquire the authority of the other.

## Source and interfaces

The source is the read-only Oct-15 composition at
`research-collab/claude/geak_evo_cleanup_20261016/oct15_source`, authenticated by
its `SOURCE_PINS.json`. All nine source files used by this port matched those pins before
editing.

| Original module | Engine module | Retained entry points |
| --- | --- | --- |
| `evolution/src/geak_evo/arena_job.py` | `geak_evo.evaluate.jobs` | `load_engineer_job`, `load_arena_job`, `validate_arena_job`, `job_digest` |
| `evolution/src/geak_evo/jobs.py` | `geak_evo.evaluate.jobs_v1` | `load_job`, `validate_job`, `job_digest` |
| `evolution/src/geak_evo/native_arena_contract.py` | `geak_evo.evaluate.native_contract` | `validate_native_job`, exact artifact and fixture declarations |
| `evolution/src/geak_evo/arena_evaluator.py` | `geak_evo.evaluate.evaluator` | `ArenaEvaluator`, `ArenaEvaluationError`, `ArenaEvaluationUnsettled` |
| `evolution/src/geak_evo/native_evaluator.py` | `geak_evo.evaluate.native_evaluator` | `NativeEvaluator`, `NativeEvaluationError`, `NativeEvaluationUnsettled`, paired acquisition and evidence verification |
| `evolution/src/geak_evo/native_job.py` (collection methods) | `geak_evo.evaluate.collection` | `NativeJobCollector.verify_anchor_evidence`, `collect_pair`, `collect` |
| `evolution/src/geak_evo/measurements.py` | `geak_evo.evaluate.timing` | `analyze_measurements`, `assess_roofline`, `format_text`, `format_json`, `MeasurementError` |
| `bridge/native_quantities.py` | `geak_evo.evaluate.quantities` | `NativeQuantities`, `NumericToken`, allocation and exact-token readers |
| `evolution/src/geak_evo/artifact_transport.py` | `geak_evo.evaluate.artifacts` | `pack`, `artifacts_fit`, existing transport limits |

`jobs.load_engineer_job(path, artifact_root=None, expected_digest=None)` keeps
strict dispatch among workload jobs v1, Arena replay v1 and native Arena v2.
Unknown or malformed explicit schemas never fall back. `jobs.job_digest` is
the Arena digest; workload-v1 callers use `jobs_v1.job_digest`. `load_job` and
`validate_job` are also available from `jobs` for callers needing both loaders.

Canonical record bytes use `_util.canonical_json`, preserving the original
Unicode handling. Streaming hashes remain streaming. Existing named exceptions
retain their names and also inherit `EvoError`. Job, native-contract, quantity
and transport sites that originally raised bare `ValueError` still raise that
exact built-in type. Exception names are persisted in error records, so a new
subclass would change saved behavior even when a `ValueError` handler catches it.

## Runtime boundary

Both evaluators require runtime-injected `launch(request, operation_dir)` and
`settle(handle_or_None, request, operation_dir)` callbacks before evaluation.
Missing callbacks raise an explicit error. Launch intent precedes callback
execution, settlement authenticates request/source and raw artifact hashes,
and recovery reconciles the same operation without another launch.

The host's native quickcheck runner remains behind those callbacks. The port
does not replace its execution, device mapping, reset protocol, correctness
judge or positive cleanup with a synthetic implementation. The native-v2 task
contract retains protected numerical, timing, runtime and report declarations.
The host retains responsibility for fixture admission and mounting; contract
validation alone does not claim that native checks or mounts occurred.

`NativeEvaluator` continues to require the exact pinned trusted operator tool,
its descriptor validator and source identity. Candidate code is not imported
by this adapter. It preserves complete-case correctness, complete-call timing,
repeated and paired acquisition, provisional flags and source-specific retained
evidence. Its optional `clock` keyword supplies the shared `SystemClock`
interface; the default still measures monotonic elapsed wall cost.

`NativeQuantities` keeps exact serialized timing tokens, job/source/epoch/case
bindings, raw request/host/plan/report joins, physical device identity and
positive settlement. The existing `disjoint-physical-v1` allocation declaration
is retained. Unknown mapping permits hypotheses but produces no timing facts.
A claim or missing trial cannot supply a replacement measurement.

`NativeJobCollector` retains the per-campaign checks required when wiring
`NativeEvaluator` into the research loop. The host injects its existing store,
evaluator, validated workload job, job identity, epoch, optional paired timing
configuration and loop. Construction verifies the matching store/job/epoch and
installs `verify_anchor_evidence` as the store's verifier. The host supplies
`collector.collect` as the loop's collection callback; the loop or its existing
snapshot interface must be attached before collecting a new direction that
needs the original missing-card record. This ports the collection methods,
not `NativeJobCampaign`'s deployment and runtime constructor.

The collector checks the harvested source again before native dispatch and
after collection. Paired anchor verification retains eligibility, uninterrupted
acquisition, settlement, unique role, exact source, job and epoch checks.
Paired timing analysis receives both the authenticated receipt binding and the
complete job case registry. Unsettled native evaluations propagate their
original exception and result, leaving reconciliation to the existing loop.
The collector's pretty JSON encoding is retained because its bytes already
identify work models and saved feedback in the original implementation.

Source identities remain version-specific. Arena requests include the exact
evaluator source hash, and quantity indexes include the host-supplied producer
reference. Replacing the original modules with this port does not admit an old
operation or quantity index under a new identity. Old-run continuation must
retain its original authenticated runtime, or use an explicitly admitted new
measurement path. This port adds no compatibility shim or guessed source pin.

The prospective Arena evaluator propagates non-`Exception` interruptions from
launch or settlement after saving `arena_evaluation.json`. A dispatcher cutoff
therefore cannot become a failed candidate that permits another search step.
This correction changes the evaluator source hash and requires a new admitted
code closure before installation. It does not change the frozen c08 evaluator
or authorize old-run continuation with the corrected source.

## Verification

The existing 12 job and 26 timing tests pass against both the original and port.
Ten shared synthetic tests also pass against both. They exercise complete
native-v2 loading, unknown-schema rejection, provisional flags and unknown
costs, partial case failure, wrong-source settlement with retained raw evidence,
reconciliation without relaunch, prepared operation identity, trial-bound
quantities, noncomparable fallback, the older evaluator's complete timing
scope and exact original exception types for eleven representative invalid
inputs. The collection port adds the five original paired-collection tests and
four focused tests for source changes before and during collection, forged
epoch/missing-case analysis, and pending native exceptions. Its nine tests pass
on the original and the port. Total: **57 passed on each layout**.

Run the port tests from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests/geak_evo -p 'test_evaluate_*.py'
```

The shared synthetic file also supports `GEAK_EVO_EVALUATION_LAYOUT=original`
with the pinned original `evolution/src` and `bridge` directories on
`PYTHONPATH`. Run it from outside the engine checkout so the original package
is selected. The two existing suites are unchanged apart from their new import
paths.

No GPU, model, native process or live study was used. The tests establish CPU
contract parity; they do not qualify an execution provider. Root owns final
integration, packaging, reciprocal review and commits.

October 16 prospective cleanup: the unused `fixture_admission`,
`streamed_identity` and `planner_inputs` wrappers are retired after checking
repository and current-composition callers. Exact job validation and fixture
hash declarations remain. `NativeEvaluator`, native collection, paired timing,
roofline analysis and both timing report formats remain available. The counts
above describe initial port parity, not qualification of a live provider.


## Generated GEAK tasks

`evaluate.generated_task` validates the existing `geak-arena-generated-task-v1`
contract. Both `load_engineer_job` and `validate_arena_job` accept it. The native-v2
and workload-v1 routes keep their own schemas and timing rules.

`project_generated_task` takes a completed extraction observation, its admitted
host policy, exact artifact rows and parsed metadata. It preserves the observed
case descriptors, source/oracle hashes and candidate binding. It declares the
pinned harness's paired timing:
10 warmups, 50 samples, separate baseline/candidate processes, adaptive pair
repetition, and the ratio of their medians. It does not substitute synchronized
mean timing or assign serving weights. The host must authenticate the extraction,
materialize its protected metadata and validate those bytes before admission.
The editable file list follows the extractor's `kernel_src/` layout; the host
must verify that this directory contains the complete permitted source set.
Declared timer settings do not establish that a later run executed them.

`jobs.declared_call_counts` exposes declared timer invocations per case within
one process or leg through `ArenaEvaluator.declared_call_counts()`. Warmups are
separate from measured invocations: native-v2 declares 10 warmups and 100 direct
calls or graph replays; generated-v1 declares 10 warmups and 50 samples with one
inner call each. Generated outer paired repetitions remain separate. These
counts exclude setup and do not count total kernel dispatches or establish
observed collector coverage. They do not change latency units, statistics or
actual timer/fallback evidence.

The accessor first validates the full existing contract, then returns a fresh
integer mapping. This preserves native validation's acceptance of equivalent
values such as `10.0` and `100.0`; the declaration bytes and digest remain
unchanged. Generated validation still requires integer counts and one inner
call. Other schemas are unsupported by this count API. Host-specific patches,
absolute source references and synthetic integration evidence belong to the
provider review packet; the canonical count tests use portable CPU fixtures.

These adaptive, baseline-first ratios are selection evidence only. Keep them
separate from native-v2 graph timing. An optimizer-only GEAK versus GEAK-EVO
comparison requires the same task, cases, harness and timer in both arms, plus
a separately specified endpoint measurement; this port does not establish them.

Source and build artifacts are hashed by streaming. The metadata needed for
semantic checks stays bounded; oracle bodies have a separate streaming check.
This reader and its fixture mount plan do not launch or resume a task and grant
no execution or measurement authority.
Generated-task execution and final paired measurements still need their runtime
adapter; see [current support](../../geak_evo/README.md#current-support).

Source provenance (under `research-collab/codex/`):

- Validator: `generated_task_streamed_artifacts_source_20261016_v1/source/evolution/src/geak_evo/generated_task_contract.py`, SHA256 `0bfcd5a1c672003994ef79b2dbaa27cd18ab908faea8d1bfbda11de06449acff`.
- Projection: `hyperloom_generated_lane_adoption_source_20261012_v1/source/adopt_generated_task.py:build_contract`, file SHA256 `a1dc2a8d8f0156170eb87e275b119426f606d1100242f1b3a0e7cf330bf4b51b`. The public projection additionally checks observed/copied file identity and supplied policy; completed-extraction authentication remains the host's responsibility.

The eleven imported validator function bodies match that source. The projection
matches its original output for complete input; missing or conflicting input
raises instead of inventing metadata.


## Generated paired results

`evaluate.generated_report.read_generated_report` reads the existing owner
collection's bytes. It imports no task code and starts no work. It recomputes
full-case results from the actual raw harness records: parent request, child
starts/exits, source observations, signatures, correctness, 50 samples per leg,
upper-middle sample reduction, separate replicate medians and adaptive stopping.
Missing records or inconsistent derived scores cannot supply a result.

The command and `result.export_final_result` support this route. For generated
tasks the selected evaluation's **internal baseline/candidate pair** supplies
the ratio. The separate ORIGINAL evaluation remains a checked reference; its
candidate-side timing is not used as the paired baseline. The external baseline
is identified by its protected closure and resolved-file observations. The
admitted editable source ID is retained separately and does not name that
external implementation. `original_paired_reference` exposes ORIGINAL's own
internal pair, full-case geomean and measurement reference; it need not equal 1.0.
When comparing search methods or claiming a gain over the starting program,
report that reference alongside the selected ratio. Similar ratios do not
establish a gain over the starting program, even when both beat the external
baseline. Do not divide two adaptive scores to invent a matched comparison.

Raw timer, graph fallback, cache, compile and dispatch details remain in the
paired score. Export additionally requires matching actual timer/cache settings
across both legs and replicates, and the ORIGINAL reference. This prospective
comparability check is stricter than merely parsing a valid harness report.
A matching wall fallback remains labeled `wall`; it is never graph evidence.
If candidate graph capture falls back to event or wall timing while its baseline
leg retains graph timing, export refuses the comparison rather than combining
those different timers.
Cases and replications are not pooled with native-v2 graph means. All exported
measurement/custody/reward/framework-acceptance flags remain false.

This ports the existing generated observer (`5eb07dd9`) and its producer-shaped
CPU tests, replacing filesystem reads with the owner's retained byte map. Byte
identities and protected metadata are rechecked; the source ID binds recorded
executable modes. The existing owner still checks fixture/alias preparation and
physical settlement. CPU composition and parsing retained K6 producer output
establish reader compatibility, not a new GPU run or workload acceptance.

Retained source provenance under `research-collab/codex/`:

- `hyperloom_generated_task_native_protocol_source_20261012_v1/source/observer/generated_task_observations.py`, SHA256 `5eb07dd9fdd105c95c9615f84c1f233ef60ef23c7187e59da37d200df2e4d98b`.
- `raw_harness_evidence_source_20261012_v1/source/harness_lib.py`, SHA256 `5bb5f4496126a12af85be69dbc9a8393070a66c3ee6b725bdd862c129cfb79a7`.
- `raw_harness_evidence_source_20261012_v1/source/leg_runner.py`, SHA256 `0959dccc59dd0d1bd07963761223b9847280ae4f4921b19f93bb07e47f6b0953`.
