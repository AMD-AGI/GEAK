# Fixed-floor quality stopping

This feature is under qualification.
The repository includes a concrete Docker/bwrap actor builder and a frozen paired scorer adapter.
The host caller must supply its admitted GPU reservation and the protected process runner.
The CPU profile does not grant GPU access.

The host caller must supply `ExecutionBoundary`, `ProcessEvaluator`, and `SelectedArtifacts` objects.
A JSON argument cannot supply these objects.
The caller uses `invoke_kernel(..., quality_stop_controller=controller, settings_profile="isolated")`.
The runner rejects stopping arguments without that controller.
Both `GEAK_LOCAL_HELPERS` and `GEAK_SHARED_TOOL_CACHE` must remain disabled.

## Concrete launch components

`quality_stop_launch.create_native_actor()` creates the actor, shell wrapper, native wrapper, and launch receipts.
Its default boundary is `DockerActorBoundary`.
Its default shell uses bwrap and exposes no GPU devices.
A trusted reservation can supply device arguments, a boundary subclass, and a shell factory.
The builder returns the exact container ID, boundary, manifest path, and native wrapper path.
The returned `ActorLaunch` supports a context manager and an idempotent `close()` method.
It stops only its exact container and verifies closure.
The caller must retain any GPU reservation until this closure completes.
`DockerCommands` also supplies exact-container inspection, process attachment, pending-start cancellation, and closure checks.
These lifecycle methods contain no cluster device policy or provider configuration.

The host retains the SDK client, private signing key, controller state, evaluator records, and protected snapshots.
The outer container runs the native CLI.
Every Bash command enters the inner bwrap boundary.
The SDK profile disables direct file tools.
The native CLI can reach the host recorder, but inner Bash commands cannot reach the network.

The builder requires two separate temporary roots.
`bash_tmp_root` supplies the inner `/tmp` directory.
`native_tmp_root` holds protected native task outputs in the outer container.
Both roots bind at identical host and outer paths.
The native temporary path plus `/claude-<UID>` must contain at most 44 bytes.
The builder pins `TMPDIR` and `CLAUDE_CODE_TMPDIR` to that native root.
It preserves `HOME` and masks the existing home path with a fresh directory.

The boundary binds `options.env["CLAUDE_CONFIG_DIR"]` to the actor's native configuration directory.
That binding aligns the SDK transcript mirror with the native CLI.
Each actor uses its own binding.
The boundary preserves unrelated environment options and the caller's process environment.
The documented composition requires no host `CLAUDE_CONFIG_DIR` setting.

`FrozenPairedEvaluator` implements the frozen GPT-OSS scorer contract.
It checks the source, task, scorer, image, bucket weights, process exit, parity, and runtime closure.
The caller supplies a `ProtectedPairedRunner` for its admitted runtime.
The evaluator allows 600 seconds for each fresh process.
An unknown process outcome prevents later measurement admissions.

The caller composes the native objects in this order:

1. Verify the immutable source, task, scorer, interpreter, and native CLI bindings.
2. Verify the recorder and resource admission.
3. Enter the host reservation.
4. Call `create_native_actor()` with that reservation and its reviewed runtime factories.
5. Create the protected paired runner and `FrozenPairedEvaluator`.
6. Create `SelectedArtifacts` from the trusted startup contract.
7. Create `QualityStopController` with the actor boundary and evaluator.
8. Call `invoke_kernel()` with the controller and isolated settings.
9. Retain the signed closure receipt and selected snapshot.
10. Close every owned container before releasing the reservation.

The caller supplies one integer `deadline_epoch` to both the controller and workflow arguments.
The caller also supplies the canonical `native_session_id` before the SDK prepares the boundary.
The SDK rejects a later session change.
The control arm uses the same profile with `stopping_enabled=False`.

The following composition shows the public object interface.
`actor_options`, `startup`, `runner`, and the other launch values come from verified host configuration.
The caller retains its reservation context around this entire block.

```python
from interface.native_cost_controls.quality_stop_controller import (
    QualityStopController,
    SelectedArtifacts,
)
from interface.native_cost_controls.quality_stop_evaluator import CALLS, FrozenPairedEvaluator
from interface.native_cost_controls.quality_stop_launch import create_native_actor
from interface.run_kernel_native import invoke_kernel

actor = create_native_actor(**actor_options)
evaluator = FrozenPairedEvaluator(
    state_dir=private_root / "paired_records",
    scorer_path=scorer_path,
    task_dir=task_root,
    task_hashes=task_hashes,
    call_weights=CALLS,
    runner=runner,
    environment=runner.scorer_environment(),
)
selected = SelectedArtifacts(
    baseline_commit=None,
    expected_seed_files=startup["expected_seed_files"],
    metadata_files=startup["metadata_files"],
    patch_path=patch_path,
    actor_patch_path=actor_patch_path,
    export_root=export_root,
    actor_export_root=actor_export_root,
    files={"kernel_src/flydsl_hgemm_impl.py": "flydsl_hgemm_impl.py"},
)
controller = QualityStopController(
    state_dir=private_root / "controller",
    candidate_root=candidate_root,
    trial_id=trial_id,
    native_session_id=native_session_id,
    budget=6,
    max_no_improve=2,
    deadline_epoch=deadline_epoch,
    buckets=CALLS,
    boundary=actor.boundary,
    evaluator=evaluator,
    selected_artifacts=selected,
    stopping_enabled=stopping_enabled,
)
arguments = {**workflow_arguments, "deadline_epoch": deadline_epoch}
result = invoke_kernel(
    arguments,
    timeout_s,
    settings_profile="isolated",
    working_directory=workspace_root,
    quality_stop_controller=controller,
)
```

The actor builder tests cover source identity, fresh directories, credentials, cleanup, and reservation ownership.
Native CPU fixtures separately cover seed handoff, boundary decisions, certificate consumption, and final closure.
Neither test class establishes GPU timing quality or provider savings.

## Scientific contract

The fixed floor is `1.05`. The controller allows at most three eligible boundary looks.
Each look contains exactly three fresh paired processes. A started invalid batch consumes its look.
The controller never replaces a process or retries a timing batch.

For process `i`, the score is:

```text
S_i = sum_j(calls_j * reference_ms_i,j) / sum_j(calls_j * candidate_ms_i,j)
X_i = log(S_i)
L = mean(X) - sqrt(1682/59) * sd(X) / sqrt(3)
Stop only when exp(L) > 1.05.
```

The numeric implementation adds a conservative rounding margin to the critical value.
The per-look error allowance is `1/60`. The family allowance is `0.05` across three looks.
These error bounds require valid per-look inference. The declared model assumes independent, identically distributed normal process log-scores.
Random order alone does not establish that model.

The controller freezes the candidate before it draws three independent fair order bits.
The controller persists the full order vector before it starts any measurement.
The controller does not balance or redraw the vector.

The final experiment requires eight separate processes for each selected source.
Those final measurements, the Welch reader, and the cost gate remain separate pre-launch requirements.
A floor certificate cannot replace the final comparison.

## Native control

The lane places a checkpoint after round memory and storage reclaim.
Budget, deadline, and no-improvement rules retain priority.
The lane preserves the existing forced-replan behavior.
The lane rechecks its original deadline after the checkpoint.
A qualifying exit requires remaining native search capacity.
After a positive decision, the lane requests a signed `consume` reply.
This stage performs no timing process.
The host rechecks the clock, source, producer census, and exclusion lease before it signs consumption.
The VM does not read its own wall clock.

The checkpoint uses `agent()` directly. It does not use the `agentT()` timeout or retry wrapper.
The local transport rejects an unsupported checkpoint route without provider fallback.
Scientific requests retain their original bytes.
The local response identifies `model="local-quality-stop-controller-v2"` and origin `quality_stop_controller`.
The transport accepts only the measured `{"type":"ephemeral"}` decoration on exact task blocks or known skill notices.
It removes that decoration from a copied view for matching.
It retains the original request bytes.

`NativeProducerCensus` operates independently of the helper registry.
It joins every native node with journal starts, journal results, and native terminal evidence.
Queued nodes without identities prevent certification.
An `agentT()` return does not prove producer completion.
The census also tracks native tool calls and background tasks.
The current implementation rejects explicit background Bash results that lack a qualified completion adapter.

The registry blocks new native admissions during a checkpoint.
The host boundary also holds continuous OS exclusion through snapshot, measurement, signing, and persistence.
Two observations of idle processes do not replace this continuous lease.

## Source and decision authority

The trusted launch pins the public key and the exact candidate directory.
The model's setup result must name that directory.
`quality_stop_startup.derive_seed_contract()` derives the expected seed from trusted task bytes in private scratch space.
This preparation runs the official materializer and executes no candidate code.
The startup contract requires a regular, hash-pinned `reference_io.pt` file.
It predeclares the exact `.geak/workspace.json` bytes as metadata.
The controller excludes this metadata from executable snapshots.
The general source policy continues to reject symbolic links.

The native lane requests a signed `seed` reply after director setup and before analysis or profiling.
The host freezes the actor and compares the actual seed against the expected source and metadata records.
It then binds the real seed commit to `SelectedArtifacts`.
The initial evaluation directory must remain absent until native setup creates it.

The candidate must contain a clean committed Git tree and a local `.git` directory.
The source census rejects untracked files, symbolic links, hard links, submodules, and changed executable modes.
Only the source paths named by `SelectedArtifacts.files` may change from the trusted seed.

The quality-stop workflow gives both arms the same runtime contract.
Create temporary helpers in a unique directory under the isolated Bash `/tmp`.
Save persistent profile records in the declared `EVAL_DIR`.
Keep helpers and profile output outside every candidate workspace, including ignored directories.
Keep benchmark working directories, interpreters, and source bindings unchanged.
Do not add helper directories to `PATH` or `PYTHONPATH` for benchmark or correctness commands.
Do not replace `rocminfo` or substitute constants for measured device information.
Preserve the exact error when a required runtime import or device query fails.
Report that failure through the role result.
The caller must qualify changed runtime bindings through the actual failing import path before it admits another trial.
Prompt instructions do not replace the strict source census or runtime checks.
The controller checks immutable seed files even when the worktree contains uncommitted changes.
The controller compares raw blobs and index entries. It does not execute Git clean filters through `git status`.
The supported Git layout comes from an isolated `git init` repository.
The controller rejects remotes, includes, alternates, replacement refs, promisor state, shallow state, metadata links, and unknown Git settings.
Git receives a minimal environment with global and system configuration disabled.
The controller preserves the existing `HOME` value.

The evaluator receives the protected snapshot, its SHA256, the process identity, and the saved timing order.
The host evaluator checks process exit, parity, record schema, source identity, and freshness.
The controller supplies workload weights from trusted launch data.
The evaluator cannot substitute different weights in its result.

The host signs an ASCII JSON payload with RSA-2048, PKCS#1 v1.5, and SHA-256.
The native VM verifies the signature before it reads the decision.
The verifier follows RFC 8017 and FIPS 180-4.
The tests compare it with Python `cryptography`, Node's standard cryptography library, and SHA-256 test vectors.
The signer requires the Python `cryptography` package.

The signature binds the exact task, native identities, look, round, candidate hash, deadline, and decision.
The private key remains in host memory outside the actor's process namespace.
The isolation adapter must protect that memory, controller files, native journals, workflow source, and snapshots.
SDK hooks alone do not provide this OS boundary.

The controller records each look before measurement starts.
Duplicate requests cannot start more measurements.
Stored successful replies require current source, authority, census, and deadline checks.
A failed disk write prevents later replay from returning a successful decision.

The lane still runs reports, director checks, knowledge writes, and its ordinary final duties.
The controller then compares the final source with the certified snapshot.
`SelectedArtifacts` pins the seed source, final patch path, and complete map of exported source files.
Each mapping connects a candidate-relative path with an export-relative path.
The configuration names host and actor paths separately.
The live startup profile uses `baseline_commit=None` and trusted `expected_seed_files`.
Focused tests can use a prebound baseline commit instead.
The caller must bind the seed contract to the evaluator's immutable reference before launch.

The initial profile requires exact canonical Git diff bytes from that baseline commit to the certified commit.
The controller disables external diff commands and text conversion.
It rejects binary patches and textually different patches, even when another patch can produce the same source.
This narrow profile requires a separate check of the actual native writer.
The controller does not execute a supplied patch in its trusted host process.

The final task includes the actual report patch path, director patch path, and export directory.
The controller checks each exported file against the certified source and rejects extra exports.
The signed final result binds all patch and export hashes and the successful consumption payload.
The runner also checks native closure and source identity after the full Workflow returns.
It performs these checks inside the live SDK context, before disconnect.
It checks the returned patch path and all selected artifact hashes again.
A changed source or unknown native outcome invalidates the trial.
The host records a signed `native_closed` receipt.
An ordinary control closure still requires the trusted seed and protected source evidence.
The receipt also binds the hash, size, and relative path of `native_census.json` in the private controller directory.
This protected census preserves every native node and its exact initial mirrored task.
It includes the session-store descriptors, full initial entries, completed tools, local emissions, and root identities.
It also includes the complete journal and bridge bytes with their hashes and source descriptors.
The host requires equal producer sets and no active tools or background tasks before it creates this census.
It waits at most ten seconds for pending bridge requests to reach a terminal event.
It seals new bridge sends before it captures the final bytes.
The final reader must join this census with the complete recorder attempt population.

## Native replacement contract

The prospective replacement path supports one narrow SDK interruption sequence.
It uses host-enforced retirement, which does not establish OS-process termination.
Whole-actor exclusion, source checks, GPU checks, final scoring, and owned process closure remain mandatory.

The host first binds the root prompt through `QualityStopSDKClient.query()`.
This path accepts one string prompt and the default query session.
Headerless model requests must retain that exact initial prompt and its qualified native context.
The qualified SDK changes its known system notice from a cached text-block list to a string during continuation.
The host normalizes only that known representation for comparison.
It forwards the original request bytes unchanged.
The SDK can send one initial `HEAD /api/hello` request.
The proxy adds the configured upstream base path before the transport checks this exact logical path.
This request must contain no body, query, fragment, or native identity header.
It must precede every provider forward and the root workflow tool.
The host rechecks this bootstrap contract under the forwarding lock.
Other headerless endpoints remain unsupported.

The host binds root tool hooks to exact IDs and inputs from trusted root `AssistantMessage` events.
The Workflow outcome hook can add its resolved `script` field.
The host accepts that field only when it equals the pinned workflow source bytes.
A hook can wait asynchronously when its root message arrives later.
Every hook identity wait uses at most ten seconds, including a pending replacement.
The SDK receives the matcher setting `timeout=30`.
The CPU qualification does not establish an enforced native deadline for that setting.
An expired identity wait returns an explicit denial and latches failure through the runtime's own ten-second limit.
Every ordinary pre-hook exception also returns explicit denial and latches failure.
This includes failures during tool-ID extraction and worker startup.
The native callback adapter can otherwise replace a callback error with an empty reply that supplies no denial.
Every ordinary post-hook exception latches an unknown outcome, including an error after removal of the active operation.
An unidentified child hook cannot acquire root authority.
Root task inspection does not enter the effectful tool census.
Every child tool, including `TaskOutput`, `TaskList`, and `TaskGet`, remains active until its matching outcome hook.

The host admits a replacement only when all these conditions hold:

1. The old and new nodes use the same phase, index, and verified logical label.
2. The new attempt number equals the old attempt number plus one.
3. Both nodes use the observed `start` or `progress` state.
4. The agents differ, and neither node is a checkpoint agent.
5. Both initial tasks, prompt IDs, sessions, run descriptors, and journal keys match.
6. The protected old transcript ends with the exact SDK interruption control frame.
7. The control frame names the preceding native frame through `parentUuid`.
8. Every recorded old bridge request contains a terminal event.
9. No known old tool or background task remains active.
10. No old `StructuredOutput` operation completed, and no old journal result exists.

The host preserves each raw native label.
The initial label defines the logical label for its phase and index.
The pinned native source emits `logical_label + " (retry N)"` for attempt `N + 1`.
That suffix remains present in progress and terminal events.
The host requires this exact rule and accepts at most six attempts, as the pinned source permits five retries.
It rejects arbitrary suffixes, missing suffixes, and later raw-label changes.
Each retirement proof records `logical_label` and both unchanged raw nodes.
Role checks use the verified logical label, including seed setup and storage reclaim.
The final census preserves raw labels.
Generation joins use verified logical labels.
Every generation and cost join preserves its agent identity.

The transcript reader uses protected directory descriptors and `O_NOFOLLOW`.
It requires stable regular files, the expected owner, one link, complete JSONL, and unchanged earlier bytes.
Ordinary tool text cannot supply an interruption control frame.
Embedded timestamps cannot establish when the host observed the frame.

The host allows at most 600 seconds for a pending replacement proof.
The original trial deadline can shorten this wait.
The host does not reset the SDK stall timer, trial clock, pair deadline, or cost limits.
The ordinary identity wait remains ten seconds.
The successor remains unforwarded until the proof passes.
A timeout latches failure.
A successor can change again while its predecessor remains pending.
The host retains the complete attempt chain and permanently blocks each superseded pending identity from forwarding.
Its waiting model requests close locally with `native_pending_successor_superseded` and both attempt flags set to `false`.
This narrow refusal does not latch failure.
Every ancestor still needs its full retirement proof before the latest successor can proceed.
The host installs the ancestor fences in attempt order under the shared lock.
An intermediate tool admission remains disqualifying.

`NativeProducerCensus.condition` protects retirement, tool admission, and bridge forwarding with one shared lock.
The host records the proof before it permanently fences the old identity and admits the successor.
An old request with only a `started` event still blocks retirement.
A durable ledger failure prevents successor admission.
The bridge route remains unclassified until forwarding admission succeeds.

The host writes `native_retirements.jsonl` in the private controller directory.
Each fence records the old and new nodes, initial hashes, task hash, prompt ID, journal key, and protected transcript bytes.
It also records the exact bridge prefix and the sorted old request IDs.
`old_unforwarded_request_ids` identifies the exact local refusals for superseded pending attempts.
The reader must reject every other two-event bridge refusal.
The proof labels its activity evidence `trusted_host_state_under_shared_admission_lock`.
The empty activity arrays attest to the trusted host state under that lock.
They do not reconstruct the complete hook history.
The signed census and source hashes bind this attestation for the final reader.

The host rejects every later old model request and every later old tool before admission.
It records `denied_model` or `denied_tool` and latches failure.
It does not invent completed tools for denied hooks.
The final reader must reject scientific success after any denial.
The reader must also retain every started attempt, raw journal row, recorder attempt, and charge.
Unknown charges remain unknown.

At checkpoints and final closure, the host rechecks the protected old transcript.
The host rejects any later old journal result.
At a checkpoint, every other current node must have a successful native result.
At final closure, every current node must have a successful native result.
Retired nodes remain a separate population with proof-qualified host fences.
The census retains both populations.
Generation and cost joins include every provider request from either population.
The ledger retains proven local refusals without inventing provider attempts.

## Native failure evidence

The census latches its first failure without resetting that latch.
It writes `native_census_first_failure.json` once when the private state directory permits the write.
The record preserves the leaf error code, trigger identity, current nodes, pending replacements, and known activity.
The record hashes task and tool payloads instead of copying possible credential text.
The writer uses exclusive creation, `O_NOFOLLOW`, mode `0600`, and file and directory synchronization.
It preserves any existing diagnostic file.
A diagnostic write error cannot clear or replace the census failure.

## Recorder contract

The private bridge ledger records local replies and scientific forwards in `controller.state_dir/bridge.jsonl`.
Local records state `origin="quality_stop_controller"`, `upstream_bridge_attempted=false`, and `provider_attempted=false`.
Scientific bridge records keep `provider_attempted` unknown.
Each scientific forward carries a fresh internal `x-geak-quality-request-id` header.
The host recorder must consume and remove this header before provider forwarding.
Only the recorder can establish whether an actual upstream attempt occurred.

A study recorder can add a reversible cache namespace to scientific requests.
Each trial needs its own namespace and first-group cache evidence.
A waiting period alone cannot prove a cold provider cache.
The recorder must retain unknown cost when an attempt lacks usable usage evidence.
Local checkpoint replies must not count as provider requests or billed savings.

## Qualification limits

CPU fixtures test control flow, protocol checks, and signature verification.
They do not measure model quality, GPU speed, or provider savings.
A fixture boundary is not a production isolation adapter.
Production launch requires independent qualification of the OS lease, evaluator, final reader, and price policy.
Missing required price scope must remain unknown.
The study's sampled GPU ownership checks do not prove continuous exclusion of every external GPU user.
The frozen scorer imports candidate Python in its own interpreter.
Its qualification requires exact-source review and the declared candidate file scope.
The scorer does not guarantee resistance to arbitrary candidate monkeypatching inside that interpreter.
