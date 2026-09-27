# Native workflow cost controls

GEAK offers two optional controls for native Claude SDK execution. Both controls default to disabled.

| Setting | Effect |
| --- | --- |
| `GEAK_SHARED_TOOL_CACHE=1` | Add one five-minute cache marker to a supported shared tool prefix. |
| `GEAK_LOCAL_HELPERS=1` | Return deterministic tool calls for five exact helper sites in a fresh kernel workflow. |

Neither setting changes the model, effort, tool catalog, ToolSearch configuration, or native permission mode.
Neither setting selects a different runtime.

## Supported entrypoints

The cache setting applies to the native SDK path in `interface/run_e2e.py`.
The standalone Node engine and direct CLI fallback retain their existing behavior.
The E2E launcher can select the Node engine from provider configuration. The cache setting does not override that selection.

The helper setting currently supports one fresh kernel workflow through `interface/run_kernel_native.py`.
This thin entrypoint calls the existing persistent SDK runner.
It accepts the public kernel arguments directly because the E2E launcher accepts an E2E handoff instead.
The helper contract supports `optimize` and `author` with the normal single-lane dispatcher.
Bakeoff, resumed lanes, alternative lane scripts, and full E2E helper dispatch retain the original model path.

Create a JSON file with the public kernel arguments:

```json
{
  "kernel_path": "/path/to/kernel",
  "workflow_dir": "/path/to/GEAK/kernel_workflow",
  "exp_root": "/path/to/experiments",
  "mode": "optimize",
  "target_language": "triton",
  "budget": 6,
  "gpu_ids": "0",
  "apply_to_original": "false"
}
```

Create a native working directory outside every Git repository.
Start the Python command from the GEAK checkout:

```bash
GEAK_NATIVE_RUN_DIR="$(mktemp -d)"
SHELL=/bin/bash GEAK_LOCAL_HELPERS=1 GEAK_SHARED_TOOL_CACHE=1 \
  python -m interface.run_kernel_native kernel-args.json result.json \
  --settings-profile isolated --cwd "$GEAK_NATIVE_RUN_DIR" --timeout 7200
```

This entrypoint uses the existing native runner's model and permission settings.
The current runner uses `permission_mode="bypassPermissions"`.
The existing `GEAK_CLAUDE_MODEL`, `GEAK_CLAUDE_EFFORT`, `GEAK_CLAUDE_SETTINGS`, and `GEAK_CLAUDE_BIN` settings remain authoritative.
The controls do not change those settings.
The explicit `--cwd` option sets the native session's working directory.
The current local helper profile excludes Git status context.
If the native directory belongs to a Git repository, the helpers retain the model path.

`--settings-profile isolated` explicitly selects the SDK's `setting_sources=[]` profile.
That profile disables user, project, and local filesystem settings, including their hooks and `CLAUDE.md` files.
Without that explicit profile, the helper control retains the original model path.
Use the same profile, shell, and native working directory for both arms of a cost comparison.
The setting does not override native permissions or managed policy.

## Shared tool caching

The policy learns the shared definitions from the first supported request for the SDK session.
The definitions stay in memory. The policy makes no discovery request.
The request must contain that exact prefix followed by one `StructuredOutput` schema.
The policy inserts this byte sequence at the final shared tool:

```json
,"cache_control":{"type":"ephemeral","ttl":"5m"}
```

The supported layout already contains three five-minute markers in the system and conversation blocks.
All other request bytes stay unchanged.
A changed catalog, unsupported marker layout, compressed body, or malformed request retains its original bytes.
The proxy forwards provider response bytes and usage counters unchanged.

The default transport uses direct HTTP or system TLS.
Custom proxies, client certificates, and unsupported TLS settings retain the original endpoint.
The proxy sends each incoming request upstream once. It adds no retry or redirect policy.
The native client still controls its own retries and redirects.
The proxy opens a new upstream connection for each request, so its network latency can differ.

## Local helpers

| Helper | Native command result | Structured result |
| --- | --- | --- |
| Clock reader | Integer epoch | `{"epoch": ...}` |
| Warm-start resolver | Resolver JSON | The same JSON object |
| Storage reclaim | Exact completion marker | `{"ok": true, "note": "reclaimed"}` |
| Citation writer | Citation count | `{"filed": ...}` |
| Experience writer | Writer JSON | The same JSON object |

The bridge joins the root Workflow call, typed progress, initial session mirror, and exact public source template.
An HTTP request cannot register a helper.
The source contract reads public source and renders selected task expressions with Node.
That renderer runs no workflow or helper command.
The initial request must contain the exact task and only qualified native context blocks.
The qualified envelope permits a fixed date reminder and one known native notice.
Additional policy text or an unknown notice retains the model path before any local emission.
The system envelope also requires the qualified default Opus 4.8 instructions from CLI 2.1.221.
The bridge checks the working directory, shell, platform, and operating system against its local runtime.
It retains the actual environment text when it binds the continuation.
Unknown system instructions retain the model path before local emission.

The bridge returns a synthetic `Bash` tool block.
The native CLI applies its existing permission path and runs the command.
The bridge then projects the actual native result into a `StructuredOutput` tool block.
The native CLI validates that block through its ordinary schema path.
The bridge never returns an `allow` permission decision.
Before it returns structured output, the bridge verifies the prior message prefix and the new assistant and tool-result messages.
The bridge permits only native cache-marker movement and one known notice serialization change.
Extra hook feedback, changed output, or a changed message produces an `UNKNOWN` state with no model fallback.

Each operation has one ledger inside the SDK process lifetime.
A StructuredOutput acknowledgement enters `ACKNOWLEDGED`. That state cannot replay a result.
Replayable completion requires a matching native `done` event and the full result from the native journal.
The bridge binds the journal path to the original Workflow tool result, session, run, and mirrored child identity.
The journal reader rejects changed identities, changed prefixes, malformed records, and conflicting results.
It preserves incomplete records as pending evidence.
The bridge compares full values even when the native progress preview truncates them.
A matching fresh native retry can replay a completed result without another Bash call.
An incomplete, denied, conflicting, or unrecognized result cannot start another command or regain unrestricted model fallback.
Unsupported requests retain the model path before any local tool emission.
After local emission, missing identity or unsupported continuation encoding stops the request.

Local responses use `model="local-deterministic-helper-v1"` and an explicit local-origin header.
Their zero usage counters are synthetic. They do not represent provider measurements.
The proxy preserves all actual provider counters on forwarded responses.

The bridge requires native SDK session mirrors, typed Workflow progress, and child identity headers.
Older or unsupported SDK options retain the original path.
Caller-supplied `can_use_tool` callbacks or native tool lifecycle hooks retain the original helper path.
The bridge cannot prove their final input and outcome order, so it leaves those callbacks unchanged.
The isolated profile does not disable managed hooks.
Managed command rewrites and post-hook feedback stop local continuation when they change the qualified input or messages.
A native stop can leave an operation incomplete. The bridge does not convert that state into success.
The native qualification covers the recorded SDK/CLI and its isolated fixture environment.
It does not establish complete hook equivalence for every managed environment or future SDK release.
An existing session store requires eager flushing. The bridge preserves that store and its optional methods.
The native CLI continues its normal disk transcript writes.
The bridge uses a temporary local ledger and removes it when the client closes.
It does not support resuming local operations across SDK sessions.
The helper driver accepts at most 65,536 bytes of native stdout.
It recognizes one measured Bash loader warning when the exact Bash and `libtinfo` files match their qualified hashes.
The driver retains the complete raw output for continuation checks.
The clock, resolver, citation, and experience projections can remove one exact leading warning.
Changed warnings, repeated warnings, and additional text remain unsupported for those projections.
The driver checks the runtime files again before local continuation.
The storage projection retains its existing completion-marker contract.

## Validation and comparison limits

The tests use synthetic tasks, native event fixtures, and loopback HTTP servers.
They test all five helper projections, unchanged defaults, exact fallback bytes, concurrent streams, and shutdown cancellation.
The combined lifecycle test sends local helpers through SDK hooks and forwards only the scientific fixture request.
Negative tests reject unsupported retry aliases and missing identities after local emission.
The L0 job includes these tests.

These tests establish implementation behavior under their stated conditions.
They do not establish workflow cost savings or kernel quality on current main.
A cost comparison requires matched workflows, model settings, task data, budgets, and evaluation conditions.
