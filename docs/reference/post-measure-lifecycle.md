# Post-measurement callback protocol

`bench_e2e.sh` can run an optional caller-owned callback on the same owned server after all timed rounds and before teardown. GEAK does not choose an evaluator, interpret accuracy, or change promotion thresholds. A successful callback receipt is execution evidence for the caller to validate.

Supported version 1 modes are fresh synthetic `warm_server` and `isolated_server`, with `PROFILE=0`, `REUSE_SERVER=0`, and `BENCH_CLIENT=native|inferencex`. Internal isolated leaves use legacy mode. Direct legacy/profile/capture runs, caller-owned servers, AgentX clients and unverified process groups are unsupported. The endpoint must use literal `http://127.0.0.1:<port>`; hostnames (including `localhost`), wildcard client addresses and IPv6 are unsupported. The server must have a verified PID/start time, lead its own process group, and own the local IPv4 HTTP listener itself or through a process in that group. That listener may bind either `127.0.0.1` or the IPv4 wildcard address.

Before a replay, discover capability without launching a server:

```bash
bash /path/to/bench_e2e.sh --post-measure-capabilities
```

The JSON contains schema `geak.post_measure.v1`, supported modes, artifact names, ownership/endpoint limitations and SHA256 hashes of the actual staged dispatcher, replica runner, summarizer, lifecycle helper and teardown library. Stage `bench_lifecycle.py` alongside the existing scripts; a missing sibling fails before launch. Old bundles without this capability must not be assumed to support the hook.

`interface/run_e2e.py` adds `post_measure_lifecycle` to normalized results, including results recovered from disk. It advertises `status=available` only when the returned `bench_e2e.sh` exactly matches the current product script and all required siblings match. Missing siblings are staged atomically; old or conflicting scripts are never replaced. The metadata records source hashes and the capability command. `replay_support.bench_e2e_fallback=true` covers direct benchmark replay. `replay_support.final_launch_script=true` additionally requires the exact deterministic `tuning/recovery_launch.sh` bridge. Arbitrary workflow-authored launchers remain unverified, even when direct benchmark replay is available. Callers must rediscover capability against the actual staged code before each replay.

Set `GEAK_POST_MEASURE_REQUEST` to an absolute path containing this request:

```json
{
  "schema": "geak.post_measure.v1",
  "request_id": "11111111-1111-4111-8111-111111111111",
  "measurement_epoch": "22222222-2222-4222-8222-222222222222",
  "contract_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "expected_config_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "callback_argv": ["/path/to/python", "/path/to/caller_evaluator.py", "--request", "/path/to/native_request.json"],
  "timeout_sec": 1200
}
```

Replace the example UUIDs with fresh canonical UUIDs and the digests with actual full SHA256 values. `expected_config_sha256` must match the caller's `EFFECTIVE_CONFIG_DIGEST`. This is a declared configuration link; it is not a substitute for checking the observed serving argv, environment, model and deployment. The caller owns the complete semantic evaluation contract identified by `contract_sha256`.

GEAK executes the literal argv vector, appending `--launch-context <absolute path>` and `--output-dir <absolute path>`. There is no shell evaluation. The callback writes an opaque JSON object to `<output-dir>/native_receipt.json` and returns its process status. GEAK validates JSON shape and records its digest, but does not inspect tasks, metric values or policy decisions. Callback stdout/stderr stay in private attempt files and are not relayed into the benchmark's output.

The request must remain unchanged, and every attempt must use a fresh output directory. A copied request snapshot is retained under `post_measure/request.json`. Files are bound by content hashes rather than modification times. JSON objects are limited to 1 MiB; each sealed throughput artifact is limited to 8 MiB. Capability discovery reports both limits. Exceeding a limit makes the callback evidence unavailable.

## Live context and receipts

`post_measure/owner.json` persists the identity already captured by `server_record_identity`: PID, verified process group, `/proc` start ticks, boot ID, protected groups, request/epoch, and a fresh launch nonce. `post_measure/launch_context.json` adds readiness time, mode and replica/attempt, local endpoint, declared config digest, and observed serving argv/environment digests and executable path. A procfs listener observation binds the endpoint port to that owned group. Raw argv and environment values are not written to the context.

`post_measure_receipt.json` has schema `geak.post_measure.receipt.v1` and includes:

- `request_id`, `measurement_epoch`, `contract_sha256`, `request_sha256`, `launch_nonce`;
- `launch_context` and `throughput_artifacts`, each with relative `path`, `sha256`, and `bytes`;
- `status`, `callback_returncode`, `callback_result` (path/hash/size or null), and `callback_cleanup_status`;
- measurement-finished, evaluation-started and evaluation-finished timestamps in Unix nanoseconds.

`status=completed` means callback return code zero, a valid JSON object was produced, bindings remained unchanged, and its recorded process group was cleaned. It does **not** mean accuracy passed. Other statuses include `failed`, `missing_result`, `timed_out`, `cancelled`, `invalid_result_or_context`, `identity_mismatch`, `binding_changed`, `supervisor_failed`, `callback_cleanup_unconfirmed`, and `throughput_restore_failed`. Missing or invalid receipts remain unavailable evidence. The caller must require its exact native task/metric/runtime/contract and normal quality gate; it must not interpret a throughput exit code as a quality pass.

Before evaluation, GEAK seals the measured summary and run bytes under `post_measure/throughput/`, records their hashes in `post_measure/measurement.json`, and retains them in the lifecycle helper. If a callback accidentally modifies or removes the original throughput files, GEAK restores the measured bytes and marks quality `binding_changed`. The isolated scheduler verifies and consumes the sealed copies, including when restoring an original file fails. An invalid seal stops selection without granting another attempt. The callback cannot replace measured throughput or buy a quality retry through an invalid summary. A failure to restore bytes is explicitly reported and cannot provide acceptable quality evidence. These bindings detect accidental changes within the caller's process boundary; they are not a sandbox against a malicious callback running as the same user.

For warm mode there is one callback after all timed rounds. For isolated mode each throughput-valid leaf runs its callback before teardown. The scheduler selects the first throughput-valid attempt regardless of callback outcome. Existing throughput failures may still use their one retry. It writes `post_measure_manifest.json`, schema `geak.post_measure.manifest.v1`, listing the exact selected replica/attempt and throughput/receipt/cleanup artifacts. It does not choose a score or aggregate quality; the caller must apply its fixed policy to all selected launches. No prior score is reused.

## Cancellation and limits

The callback runs in a supervised process group. Its supervisor remains the live group leader even if the callback exits before its children. On completion, failure, timeout or a stop signal, GEAK invokes its existing `server_teardown.sh` against that verified supervisor identity. A caught TERM keeps the supervisor alive until bounded escalation, allowing TERM-ignoring callback children in that group to be cleaned safely. The callback must not move children into unrelated sessions/process groups; escaping workers are outside this initial protocol.

GEAK's INT/TERM handling forwards the signal to the active callback helper or isolated leaf, lets its cleanup finish, then reaches the existing server EXIT teardown. The serving-GPU lock remains held through evaluation and cleanup. `timeout_sec` bounds callback execution; allow additional cleanup time (one-second callback grace plus process reaping and the configured server teardown grace). The caller must also enforce its session/outer replay deadline; cancelling only a waiting Python thread is insufficient.

`post_measure_cleanup.json` reports only the recorded server process group's cleanup, with request/epoch and launch nonce. Its success value is `recorded_group_gone`, not a claim to have discovered all detached workers. A caller whose replay shell cannot finish may request bounded cleanup through the same staged library:

```bash
python3 /path/to/bench_lifecycle.py cleanup \
  --output-dir /path/to/current/attempt \
  --request-id 11111111-1111-4111-8111-111111111111 \
  --measurement-epoch 22222222-2222-4222-8222-222222222222
```

This command requires the matching recorded request/epoch and live start identity; it does not infer ownership from a process name or port. A dead/reused server leader cannot authorize signaling an unrelated group. Existing server-leader-exit/escaped-worker gaps remain explicit unsupported cases; cleanup uncertainty is never repaired by broad process kills. The caller must retain that uncertainty and reject the measurement for promotion.

When the request variable is absent, the existing throughput-only lifecycle is unchanged. The deterministic recovery bridge preserves the caller's current request, output directory and benchmark path across deployment-environment loading, including an absent request. Other final launchers must provide equivalent behavior before a caller can use them for this protocol.
