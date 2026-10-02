# Native workflow cost controls

GEAK offers two optional controls for native Claude SDK execution. Both controls default to disabled.

| Setting | Effect |
| --- | --- |
| `GEAK_SHARED_TOOL_CACHE=1` | Add one five-minute cache marker to a supported shared tool prefix. |
| `GEAK_LOCAL_HELPERS=1` | Return deterministic tool calls for five exact helper sites in a fresh kernel workflow. |

Neither setting changes the model, effort, tool catalog, ToolSearch configuration, or native permission mode.
Neither setting selects a different runtime.
The cache policy also preserves SDK hooks, settings, and compaction options.

## Supported entrypoints

The cache setting applies to `interface/run_kernel_native.py` and the native SDK path in `interface/run_e2e.py`.
The standalone Node engine and implicit CLI fallback retain their existing behavior.
The E2E launcher can select the Node engine from provider configuration. The cache setting does not override that selection.
An explicit [process wrapper](#direct-cli-and-parent-launchers) also supports native CLI requests and descendant native sessions.

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

`GEAK_SHARED_TOOL_CACHE=1` enables caching.
`GEAK_SHARED_TOOL_CACHE_POLICY` selects the policy.
An unset policy selects `legacy_shared_tool_prefix`, which preserves the existing behavior.
`registered_native_prefix_v1` requires an explicit catalog and its expected SHA-256 hash.

| Policy | Prefix registration | Trailing tools | Existing cache markers |
| --- | --- | --- | --- |
| `legacy_shared_tool_prefix` | Learn from the first supported request for this SDK session. | Exactly one `StructuredOutput` tool. | Three fixed five-minute markers. |
| `registered_native_prefix_v1` | Read the selected catalog before the SDK client starts. | Any trailing tools, including none. | Zero to three supported five-minute markers outside the tools. |

The legacy policy keeps its learned definitions in memory and makes no discovery request.
Both policies insert this byte sequence at the final shared tool:

```json
,"cache_control":{"type":"ephemeral","ttl":"5m"}
```

All other request bytes stay unchanged.
A changed catalog, unsupported marker layout, compressed body, or malformed request retains its original bytes.
The proxy forwards provider response bytes and usage counters unchanged.

### Explicit native prefix registration

Set all four variables to select the registered policy:

| Variable | Required value |
| --- | --- |
| `GEAK_SHARED_TOOL_CACHE` | `1` |
| `GEAK_SHARED_TOOL_CACHE_POLICY` | `registered_native_prefix_v1` |
| `GEAK_SHARED_TOOL_CATALOG` | An absolute path to the catalog JSON file. |
| `GEAK_SHARED_TOOL_CATALOG_SHA256` | The expected file hash, with 64 lowercase hexadecimal digits. |

The catalog contains one nonempty JSON array of ordered leading tool definitions.
Each definition must describe a client tool with a unique name and an `input_schema` object.
The array must exclude `StructuredOutput` and tool-level `cache_control` fields.
Schema fields named `cache_control` remain schema data.
The catalog contains only the shared prefix, not the complete request.

The SDK adapter verifies the catalog bytes, expected hash, and structure before it starts a supported native session.
It then retains the tool count and fingerprints in memory.
A later file change does not change that session's registration.
A new session reads the file again and verifies the expected hash again.
Missing files, invalid catalogs, unknown policies, and hash mismatches stop supported session startup.
The adapter does not learn registered definitions from any request.
It checks the native session identity before it matches the prefix.

The policy matches the registered definitions in their original tool order.
Trailing tools can change between requests.
For example, a request can omit `StructuredOutput` or use a different `StructuredOutput` schema after the registered prefix.
The policy defines no fixed prefix count, workflow family, or workflow phase.
The policy adds no namespace for experimental groups.

Use the kernel argument file described above.
Replace `<sha256>` with the expected hash from the catalog producer's record.
Start this command from the GEAK checkout:

```bash
GEAK_CLAUDE_BIN=/absolute/path/claude \
GEAK_SHARED_TOOL_CACHE=1 \
GEAK_SHARED_TOOL_CACHE_POLICY=registered_native_prefix_v1 \
GEAK_SHARED_TOOL_CATALOG=/absolute/path/native-catalog.json \
GEAK_SHARED_TOOL_CATALOG_SHA256='<sha256>' \
  python -m interface.run_kernel_native kernel-args.json result.json \
  --settings-profile isolated --timeout 7200
```

This command uses the native runner's current model, effort, settings, and permissions.
Use the same pinned CLI that produced the catalog.
The explicit isolated profile matches the producer's filesystem settings scope.
Use the same four variables with the earlier helper command to combine registered caching and local helpers.
Keep that command's isolated settings profile, Bash shell, and working directory outside Git repositories.
The registered policy does not alter helper eligibility or helper results.
When caching is disabled, local helpers ignore the catalog settings.

The E2E launcher uses the same registration on its native SDK path.
Use the existing [E2E handoff contract](../run_e2e.md) to create `handoff.json`.
The following command explicitly disables automatic runtime selection and clears explicit runtime selectors:

```bash
GEAK_AGENT_AUTO=0 GEAK_AGENT_PROFILE= GEAK_AGENT_BACKEND= GEAK_MODEL= \
GEAK_SHARED_TOOL_CACHE=1 \
GEAK_SHARED_TOOL_CACHE_POLICY=registered_native_prefix_v1 \
GEAK_SHARED_TOOL_CATALOG=/absolute/path/native-catalog.json \
GEAK_SHARED_TOOL_CATALOG_SHA256='<sha256>' \
  python -m interface.run_e2e handoff.json result.json
```

This command requires the native Claude SDK.
If the SDK import fails, the E2E launcher's existing CLI fallback does not apply this cache policy.
Use a catalog that matches this E2E session's settings and tools.
The kernel producer does not qualify a different E2E settings profile.

### Catalog production

`produce_catalog` runs a selected native profile against a local synthetic endpoint.
It exports the requested leading tools and a receipt that records their source.
It requires Linux, `bwrap`, and working user, network, and PID namespaces.
The capture uses a private mount namespace, private devices, and a private temporary directory.
The capture sees the source filesystem as read-only.
The producer stops if it cannot establish isolation.
It never uses the host network as a fallback.

| Profile | Captured request | Source requirements |
| --- | --- | --- |
| `main` | One fresh main SDK session. | The current public native runner. |
| `kernel-child` | One native Workflow child. | The current public runner options and `kernel_workflow/kernel_lane.js`. |
| `kernel-frozen` | One native Workflow child for a frozen comparison. | Explicitly selected kernel and options sources, with expected hashes. |
| `hyperloom-specialist` | One specialist main request from selected Hyperloom source methods. | An external Hyperloom source tree and its pinned source manifest. |

Use `kernel-child` for a catalog from the current public kernel source.
Replace the CLI hash placeholder with its independently recorded pin.
The output directory must not exist, and its parent must exist.
Start this command from the GEAK checkout:

```bash
python -m interface.native_cost_controls.produce_catalog \
  --profile kernel-child \
  --cli /absolute/path/claude \
  --cli-sha256 '<cli-sha256>' \
  --sdk-version 0.2.128 \
  --prefix-count 20 \
  --output /absolute/path/new-kernel-catalog
```

The checked public profile uses SDK `0.2.128`, CLI `2.1.221`, and the `c06b72ee` public runner and kernel sources.
Its child request contains 21 tools, including the final `StructuredOutput` tool.
The example therefore exports 20 leading tools.
The frozen comparison uses different source options and exports 22 leading tools from 23 tools.
The public prefix did not receive a full workflow cost experiment in this change.

`--model` defaults to `claude-opus-4-8`.
For the main and public kernel profiles, `--effort` defaults to the public runner's `ultracode` setting.
The public `ultracode` setting omits an explicit native `--effort` flag.
Pass explicit values when those intended profiles use different model or effort settings.
`--timeout` defaults to 60 seconds and accepts values from 1 to 300 seconds.
The prefix count is an explicit input, not a universal constant.
The kernel profiles require one final `StructuredOutput` tool outside the selected prefix.
`kernel-child` can export a shorter leading prefix with additional tools before `StructuredOutput`.
`kernel-frozen` requires exactly the selected prefix and one `StructuredOutput` tool.

The producer writes `catalog.json` and `receipt.json` into the new directory.
The receipt records the selected profile, CLI hash, SDK sources, runner sources, filters, permissions, and capture hash.
It also records the catalog hash, source checks, synthetic response counts, and profile limits.
The receipt omits absolute source paths, and neither file stores a raw request body.
Use `catalog.json` with `GEAK_SHARED_TOOL_CATALOG`.
Use the receipt's `catalog_sha256` value with `GEAK_SHARED_TOOL_CATALOG_SHA256`.

The capture uses fixed synthetic prompts, isolated settings, empty MCP configuration, and fake credentials.
The local endpoint forwards no request to a provider.
The main profile returns only fixed text.
The kernel profiles permit one exact synthetic Workflow call through copied source functions.
They reject other tool calls and return fixed text to the native child.
The kernel profiles disconnect after the child response without requiring Workflow completion.
The capture does not run kernel search, correctness tests, or GPU benchmarks.
Synthetic usage values do not measure provider cost or cache benefit.

The `main` profile does not establish child catalogs.
The `kernel-child` profile does not establish Hyperloom specialist catalogs.
No profile establishes another source revision, CLI build, tool filter, permission setting, or custom environment without a matching source record.
Use the isolated settings profile when comparing a produced kernel catalog with the native kernel entrypoint.
Custom settings, MCP tools, skills, and managed hooks require a separate source record that matches the intended workflow.

An external kernel source requires both `--kernel-source PATH` and `--kernel-source-sha256 HASH`.
The optional `kernel-frozen` comparison also requires `--kernel-options-source PATH` and `--kernel-options-sha256 HASH`.
That comparison requires an explicit native effort value, such as `--effort xhigh`.
It rejects `ultracode` as an effort value.
The comparison sources can be unavailable outside the original experiment.
Public kernel usage does not require those frozen sources.

#### Hyperloom specialist source

GEAK does not include Hyperloom.
The `hyperloom-specialist` profile requires a reviewed external checkout and a manifest with exactly these five source paths:

```json
{
  "schema": "geak-hyperloom-source-v1",
  "files": {
    "src/hyperloom/orchestrator/specialists/subprocess_.py": "<subprocess-sha256>",
    "src/hyperloom/orchestrator/specialists/runner.py": "<runner-sha256>",
    "src/hyperloom/orchestrator/specialists/leaf.py": "<leaf-sha256>",
    "src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py": "<prompt-builder-sha256>",
    "src/hyperloom/inference_optimizer/cli/executors.py": "<executors-sha256>"
  }
}
```

Replace each hash placeholder with the expected source file hash.
Record the manifest file's SHA-256 hash separately.
Use that hash with `--hyperloom-source-manifest-sha256`:

```bash
python -m interface.native_cost_controls.produce_catalog \
  --profile hyperloom-specialist \
  --cli /absolute/path/claude \
  --cli-sha256 '<cli-sha256>' \
  --sdk-version '<sdk-version>' \
  --prefix-count 18 \
  --hyperloom-source /absolute/path/hyperloom \
  --hyperloom-source-manifest /absolute/path/hyperloom-source.json \
  --hyperloom-source-manifest-sha256 '<manifest-sha256>' \
  --output /absolute/path/new-specialist-catalog
```

This profile executes unchanged selected source methods for the native command and writable directory list.
It also uses the source leaf definitions and tool filter.
It does not execute the factory, actor, runner, leaf agent, or framework directory discovery.
Supply each intended framework directory with a repeated `--hyperloom-framework-root /absolute/path/framework` option.
Each directory must exist.
The producer records hashes of the directory paths and their count, not a hash of all directory contents.

The profile preserves the source default effort and rejects explicit native effort overrides.
It requires 23 tools without `StructuredOutput` and exports their first 18 definitions.
These counts restrict this producer profile, not the registered marker policy.
The counts alone do not establish parity with a measured Hyperloom catalog.
The local endpoint returns fixed terminal text and requests no tool execution.
The receipt identifies this scope as a specialist main request, not a leaf request.

### Source requirements

The catalog producer must establish the source of each definition before registration.
Use a synthetic native capture that records its source and cannot send a paid request.
Use the pinned CLI and source entrypoint that the intended workflow uses.
Use the same tool filters and permissions as the intended workflow.
Export only the ordered leading tool definitions into the catalog file.
Record hashes for the CLI, source, tool filters, synthetic capture, and catalog file.
Record the permission settings with those hashes.
Keep the expected catalog hash separate from the catalog file.
Do not learn a registered prefix from paid requests.
Do not commit extracted request bodies.

`RegisteredToolPrefix` verifies the supplied bytes, digest, and structure.
It does not verify producer provenance or reproduce the synthetic capture.
The producer remains responsible for the source record and its relationship to the intended workflow.
A valid hash alone does not establish that relationship.

### Layout and transport limits

The registered policy preserves request bytes when the prefix is missing, changed, reordered, or shorter than the registration.
It also declines existing tool markers, request-level markers, one-hour markers, and layouts that already contain four markers.
Unsupported content blocks, hidden nested cache markers, and compaction blocks retain the original request bytes.
The policy treats `cache_control` inside tool schemas and tool arguments as data.
Provider output-format fields also retain the original request bytes.
The cache policy does not disable compaction or change its settings.

The SDK adapter preserves the existing path for unsupported SDK options, alternate providers, and resumed or forked sessions.
The adapter checks those paths before it loads the registered catalog.
The default transport uses direct HTTP or system TLS.
The SDK adapter keeps the original endpoint for custom proxies, client certificates, and unsupported TLS settings.
The proxy sends each incoming request upstream once. It adds no retry or redirect policy.
The native client still controls its own retries and redirects.
The proxy opens a new upstream connection for each request, so its network latency can differ.

### Direct CLI and parent launchers

`run_registered` explicitly places the cache proxy around a native CLI command or its parent launcher.
This path supports native CLI descendants that a parent launcher, such as Hyperloom, starts.
Use a fresh local process tree with no pre-existing or remote workers.
For Hyperloom, use a fresh local topology with no prior Ray cluster.
Supply one `--catalog PATH SHA256` pair for each registered prefix.
Place the original command and its arguments after `--`:

```bash
python -m interface.native_cost_controls.run_registered \
  --catalog /absolute/path/specialist-catalog.json '<specialist-catalog-sha256>' \
  --catalog /absolute/path/kernel-catalog.json '<kernel-catalog-sha256>' \
  -- /absolute/path/claude --print 'Return the intended task result.'
```

Replace the command after `--` with the original native CLI command or the parent launcher command.
Retain all original arguments when wrapping an existing command.
Use catalogs that match the CLI, source, filters, and permissions for that command and its native descendants.
The wrapper does not verify those producer relationships from a file hash alone.

The wrapper verifies every catalog before it starts the command.
It tries the longest registered prefix first.
The first matching prefix owns the decision, including a decline.
It never learns a prefix from incoming traffic.
The SDK adapter requires the configured SDK session identity before matching a prefix.
The process wrapper permits multiple native session identities and requires an exact registered prefix for each modified request.
A request with no matching prefix retains its original bytes.

The wrapper preserves the command arguments, model, effort, compaction settings, and permissions.
It changes only `ANTHROPIC_BASE_URL` in the child environment.
The existing `ANTHROPIC_BASE_URL` becomes the upstream endpoint, or the proxy uses `https://api.anthropic.com` when it is absent.
The four SDK cache variables do not enable this path.
The explicit wrapper command enables this path.

The wrapper requires exclusive control of a fresh Linux process.
It requires `/proc`, pidfd support, and Linux child subreaping.
Linux child subreaping transfers orphaned descendants to the wrapper.
Use a noninteractive command that honors `ANTHROPIC_BASE_URL`.
It preserves the command's standard input, standard output, and standard error.
It rejects existing children, additional threads, and a modified `SIGCHLD` handler before command startup.
Missing Linux process controls, unsupported providers, custom proxies, and unsupported TLS settings also stop startup.
The wrapper does not start the command if the proxy cannot start.

The wrapper tracks Linux descendants, including detached sessions and descendants that use a double fork.
It forwards `SIGINT`, `SIGTERM`, and `SIGHUP` to children it directly owns.
When the command exits, the wrapper terminates remaining descendants.
After a two-second grace period, cleanup sends `SIGKILL` to descendants that remain.
The proxy remains active until the wrapper collects every descendant's exit status.
Uninterruptible kernel tasks can delay cleanup.
Pre-existing workers and remote workers remain outside this lifecycle scope.

The wrapper does not create the producer's filesystem or network isolation.
The original command retains its filesystem effects, including Hyperloom's native CLI configuration writes.
Use the container recipe below for a full Hyperloom optimize launch.
Plain host optimize launches remain outside the supported recipe.

#### Hyperloom in a disposable container

Start one fresh disposable container for each run.
Run the wrapper inside that container.
The container owns `~/.claude/config.json`, which Hyperloom's existing preflight can write.
Do not mount the host home or host Claude configuration into the container.
Use the image's private home for the native CLI configuration.
The image's normal user must write its private home and the output directory.
Hyperloom executes its existing preflight.

The container must start a fresh local Ray topology with no reachable existing cluster.
Remove `RAY_ADDRESS` before the wrapper starts.
Do not mount existing Ray state or select remote workers.
This recipe supports `--nodes 1`.
The wrapper's descendant scope does not include Docker's host daemon or workers outside the container.

The image must contain GEAK at `/opt/geak` and the qualified Hyperloom, native CLI, SDK, and framework dependencies.
Use the same CLI and source versions that the catalog records identify.
Supply credentials through the caller's environment or an environment file.
Keep credential values out of commands and receipts.

Mount the model at `/workload/model`.
Start this command from `/opt/geak` inside the prepared container:

```bash
env -u RAY_ADDRESS USER_DATA_PATH=/output \
  python3 -m interface.native_cost_controls.run_registered \
  --catalog /catalogs/specialist/catalog.json '<specialist-catalog-sha256>' \
  --catalog /catalogs/kernel/catalog.json '<kernel-catalog-sha256>' \
  -- python3 -m hyperloom.inference_optimizer.cli optimize \
  --model /workload/model --framework sglang --gpu-type mi355x \
  --tp 1 --nodes 1 --max-hours 2
```

The `--model` path identifies the serving checkpoint.
`USER_DATA_PATH` selects the workspace that receives Hyperloom's session directories.
The example uses SGLang, MI355X GPUs, tensor parallel size 1, and a two-hour budget.
Use the workload arguments and GPU bindings that your intended run requires.
Keep those arguments identical when comparing cached and uncached runs.

The following Docker template supplies explicit resource limits and the workload bindings:

```bash
GEAK_CONTAINER_IMAGE='your-registry/qualified-image@sha256:<image-digest>'
GEAK_CONTAINER_ENV_FILE=/absolute/path/provider.env
GEAK_CONTAINER_CATALOG_DIR=/absolute/path/catalogs
GEAK_CONTAINER_MODEL_DIR=/absolute/path/model
GEAK_CONTAINER_OUTPUT_DIR=/absolute/path/new-output
GEAK_CONTAINER_CPUSET='<cpu-list>'
GEAK_CONTAINER_CPUS='<cpu-quota>'
GEAK_CONTAINER_MEMORY='<memory-limit>'
GEAK_CONTAINER_SHM_SIZE='<shared-memory-size>'
GEAK_CONTAINER_RENDER_NODE='/dev/dri/renderD<node-number>'
GEAK_CONTAINER_ROCR_VISIBLE_DEVICES='<gpu-uuid-or-index>'
GEAK_CONTAINER_HIP_VISIBLE_DEVICES='<hip-visible-index>'
mkdir -- "$GEAK_CONTAINER_OUTPUT_DIR"

docker run --rm --init --network bridge \
  --env-file "$GEAK_CONTAINER_ENV_FILE" \
  --cpuset-cpus "$GEAK_CONTAINER_CPUSET" --cpus "$GEAK_CONTAINER_CPUS" \
  --memory "$GEAK_CONTAINER_MEMORY" --shm-size "$GEAK_CONTAINER_SHM_SIZE" \
  --device /dev/kfd --device "$GEAK_CONTAINER_RENDER_NODE" \
  --env "ROCR_VISIBLE_DEVICES=$GEAK_CONTAINER_ROCR_VISIBLE_DEVICES" \
  --env "HIP_VISIBLE_DEVICES=$GEAK_CONTAINER_HIP_VISIBLE_DEVICES" \
  --mount "type=bind,src=$GEAK_CONTAINER_CATALOG_DIR,dst=/catalogs,readonly" \
  --mount "type=bind,src=$GEAK_CONTAINER_MODEL_DIR,dst=/workload/model,readonly" \
  --mount "type=bind,src=$GEAK_CONTAINER_OUTPUT_DIR,dst=/output" \
  --workdir /opt/geak --entrypoint /usr/bin/env \
  "$GEAK_CONTAINER_IMAGE" \
  -u RAY_ADDRESS USER_DATA_PATH=/output \
  python3 -m interface.native_cost_controls.run_registered \
  --catalog /catalogs/specialist/catalog.json '<specialist-catalog-sha256>' \
  --catalog /catalogs/kernel/catalog.json '<kernel-catalog-sha256>' \
  -- python3 -m hyperloom.inference_optimizer.cli optimize \
  --model /workload/model --framework sglang --gpu-type mi355x \
  --tp 1 --nodes 1 --max-hours 2
```

Replace the image, paths, hashes, workload arguments, and resource placeholders with the caller's qualified inputs.
Select the host CPU indices with `GEAK_CONTAINER_CPUSET`.
Set `GEAK_CONTAINER_CPUS` to the maximum CPU quota, measured in CPU cores.
Set the memory limit and shared-memory size for the intended workload.
Use Docker size values, such as `16g`, for those two limits.
Select one AMD render node with `GEAK_CONTAINER_RENDER_NODE`.
Set `GEAK_CONTAINER_ROCR_VISIBLE_DEVICES` to the GPU UUID or index that matches that render node.
Set `GEAK_CONTAINER_HIP_VISIBLE_DEVICES` to the visible HIP index after ROCR selects the GPU.
Use the same resource limits for cached and uncached comparisons.
The template uses the selected AMD render node and a read-only model mount.
Supply the workload mounts that the intended workflow requires.
Keep native CLI configuration inside the disposable container when selecting those mounts.
The environment file must not select an existing or remote Ray cluster.
The wrapper starts inside the container and retains the proxy until its descendant cleanup completes.
Docker removes the container after exit, while the explicit output mount retains the workflow results.

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

Registered cache tests cover catalog hashes, immutable registration, session gates, trailing tools, nested markers, and unchanged SDK options.
Wrapper tests cover inherited streams, preserved arguments, startup rejection, signals, detached descendants, and proxy cleanup.
A synthetic child tests private configuration writes while a separate host configuration remains unchanged.
That test does not execute Hyperloom's preflight or create a Docker container.

Local native captures used CLI `2.1.221`, SDK `0.2.128`, and explicit source records.
They produced the public kernel prefix of 20 tools and the frozen kernel prefix of 22 tools.
The Hyperloom specialist capture produced its prefix of 18 tools.
Each capture passed the portable policy check with only the marker bytes changed.
The frozen kernel and Hyperloom prefix fingerprints matched the separately reviewed catalogs under those pinned conditions.
The public kernel capture used different source options from the frozen experiment.
These catalog captures do not measure full workflow cost or kernel quality.

Documentation checks cover shell syntax, producer arguments, and the Hyperloom command flags.
Those checks do not execute the container or a full optimize run.

These tests establish implementation behavior under their stated conditions.
They do not establish workflow cost savings or kernel quality on current main.
A cost comparison requires matched workflows, model settings, task data, budgets, and evaluation conditions.
