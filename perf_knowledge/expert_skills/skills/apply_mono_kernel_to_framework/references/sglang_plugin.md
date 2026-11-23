# SGLang: a mono kernel as an `sgl-amd-mono`-style plugin

Status: a **design prior**, and the less developed of the two targets: the delivery phase this is ported from
was designed for vLLM first and never exercised end to end for either engine (see the status note in
[skill.md](../skill.md)). The interface facts below were read from upstream source on 2026-11-23: SGLang `main`
at commit [`38ae3402`](https://github.com/sgl-project/sglang/commit/38ae340221b4b65859c68e85d9e8ffb6b9c6f167) and
the plugin repository at [`4cba5b25`](https://github.com/sgl-project/sgl-amd-mono/commit/4cba5b253e7c2bbe8329d4f423538a6474083514).
They are version-dependent and have already changed once (next section): **the commit you target is the authority**.

## Upstream references (public)
| Link | What it gives you |
|---|---|
| [RFC 43949](https://github.com/sgl-project/sglang/issues/43949) | The overall spec of the mono-kernel framework for SGLang. Its text uses the older flag name |
| [PR 44353](https://github.com/sgl-project/sglang/pull/44353) | The vendor-neutral framework and its one runtime switch for vendor plugins, first under `python/sglang/srt/monokernels/` |
| [PR 45389](https://github.com/sgl-project/sglang/pull/45389) | The refactor that split the plugin contracts from their runtime integration (the layout below) |
| [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) | The AMD plugin that holds kernels, specs and forwards. SGLang pins it by commit (`python/pyproject_other.toml`), so a plugin change reaches users through a pin bump |

## What changed, and what to read first
| Item | On `main`, 2026-11-23 | Before |
|---|---|---|
| Switch | `--enable-amd-monokernel` (field `enable_amd_monokernel` in `python/sglang/srt/arg_groups/fields/exec_.py`), off by default, marked experimental | `--monokernel-backend amd`. That flag is now in `REMOVED_FLAGS` (`python/sglang/srt/server_args.py`): passing it stops the launch with a message naming the replacement |
| Contracts a plugin implements | `python/sglang/kernels/plugin_kernels/amd/spec.py`, which imports nothing from the runtime | `python/sglang/srt/monokernels/spec.py`, now a shim that re-exports them "at their former import path" |
| SGLang's integration | `python/sglang/srt/kernel_plugins/amd/binding.py`: startup checks, weight observation, TP agreement, CUDA-graph coverage, forward dispatch and fallback | Under `python/sglang/srt/monokernels/` as well |
| API version | Module-level `MONOKERNEL_API_VERSION` in `spec.py`. A spec whose `api_version` differs is refused (`find_spec`) | Not checked |

Read those files, and `python/sglang/kernels/plugin_kernels/README.md`, at your commit before anything else.
Record the flag, the import path and the API version you target, and state them in the report. Do not copy a
flag from documentation: on that day the plugin's own README still used the older flag in most of its launch
examples. Read it from the source you run.

## Shape of the deliverable
- **Plugin layout.** Mirror one model in `sgl-amd-mono`: a package holding the kernel, its runtime support
  library, the spec and a forward (`<model>/spec.py` is "spec and forward"; `common/` is the device runtime:
  guarded waits, epochs, peer memory). The patch adds it under the plugin's layout and the model file itself
  stays untouched. Specs register through SGLang's general plugin entry point (`register` in `spec.py`).
- **Whole-forward replacement.** With the switch on, the binding replaces the model's forward for the steps the
  spec covers; every other step runs the model's own forward (`_install_forward`). A kernel that covers one
  layer kind therefore ships a forward that runs the stock modules for every other layer and calls the mono
  kernel for the covered kind.
- **The forward's contract** (`BuiltMonokernels.forward`). It takes the model forward's arguments and returns
  what it returns, writes the KV cache and auxiliary hidden states the standard path writes for every row,
  padded rows included, and is capturable in a CUDA graph: no host sync, shapes fixed by the plan's key.
- **Selection.** By graph key alone (`ShapeKey`: forward mode, batch size, tokens per request; rows = batch size
  × tokens per request). "Every rank holds it alike", and a captured graph replays the decision made at capture.
  The forward mode is `decode` or `target_verify`. This skill delivers pure decode only; a verify step needs a
  kernel whose oracle covered verify steps and a spec that accepts speculative decoding.
- **Capture sizes.** At startup the binding aligns the decode capture batch sizes with the spec's widths unless
  they were set explicitly (`covered_capture_sizes`). After capture it checks the captured graphs themselves,
  logs which steps run the kernels, and refuses a runner none of whose captured decode graphs does
  (`_CaptureCheck`). That log is upstream's own engagement evidence. It does not replace a launch counter.
- **Counters.** Per-rank launches and skips by reason, readable from the server log. The binding had no launch
  counter at that commit, so the counter line of this skill is the plugin's job.

### `MonoSpec` fields
| Field | What to put there |
|---|---|
| `devices` | Per ISA (`gfx942`, `gfx950`) a `DeviceLimits`: `widths` (step rows, ascending), `tp_sizes`, `resident_blocks` (the launch grid, one workgroup per compute unit). Widths are device-dependent through the size of LDS, so the two lists differ. A GPU with fewer compute units than `resident_blocks` is refused |
| `features` | An allow-list. `Feature` enumerates the SGLang features that are "refused unless a spec lists them": LoRA, expert parallelism, speculative decoding, pipeline parallelism, KV offload, torch-compiled decode, partitioned GPUs and others. A refusal-list entry of the kernel that is one of these is expressed by **not** listing it |
| `requires` | `Requirements`: accepted dtypes, decode attention backends and server-argument values, exact model-config fields and environment values. Use it for the refusal-list entries that are not features |
| `config_disable_reasons`, `model_disable_reasons` | Callables that return a list of reasons: before load for what `requires` cannot state, after load for the weights as loaded |
| `numerics` | `Numerics`: kind `BITWISE` or `TOLERANCE`, with bounds named `<launch>.<output>.<metric>`, the metric being `max_abs`, `rel` or `cosine`. A `max_abs` bound of zero means bitwise equality, and `TOLERANCE` needs at least one bound. Put the KV write at zero and activations at the derived relative bound. Claim `BITWISE` only if the layer oracle shows a maximum absolute difference of 0 for every output. A spec that is not `BITWISE` is refused under `--enable-deterministic-inference` |
| `build` | Called once the model is prepared and before KV-cache sizing, with a `BuildContext`: the ISA, the widths the capture plan routes, the wait deadline, the weight copies. JIT every width here |
| `load_tensor` | Optional hook that sees checkpoint tensors before model-specific conversion, to retain copies. If the spec has it and the load path does not pass tensors through it, that is a refusal |
| `api_version` | The `MONOKERNEL_API_VERSION` the spec is written against |

A tolerance is the one the kernel's own oracle derived. Delivery never widens it, and never upgrades it to
`BITWISE` because a smoke happened to match.

### How a refusal is expressed
At that commit an unsupported setup is an error that lists its reasons, never a silent fallback. The flag's
help text: "An unsupported startup configuration raises an error listing the reasons."
- **Before the weights load,** at argument resolution: `handle_monokernel` collects every reason
  (`monokernel_disable_reasons`) and raises one `ValueError`, "... cannot run this setup:" followed by the list.
- **After load, before KV sizing:** `install_monokernel_forward` gathers each rank's reasons over the TP CPU
  group (`_gather_tp`) and raises one `ValueError` that names the rank of each. "Ranks branch together; one
  skipping the collectives below would hang its peers." Ranks that load different specs, or that would route
  different graph keys, are an error too.
- **After capture:** `_CaptureCheck`, above.
- **Not an error:** a step outside the served keys runs the model's own forward, and a draft model with no
  registered spec keeps its normal forward with an info line.
- The binding has no separate per-bind cache stage at that commit: KV-layout conditions belong in `requires`
  and `model_disable_reasons`.

### `SGLANG_READS`
A convention of the plugin, not a field of SGLang's contract: every module of `sgl-amd-mono` that reads SGLang
internals declares them in `SGLANG_READS`, and `test/unit/test_contract.py` checks each declared name against
the installed SGLang, "so an incompatible SGLang change fails on CPU". List there every name the forward or
the spec reads (modules, attributes, attention-backend fields, KV-pool accessors) and run that test before the
smoke: a renamed attribute then fails in seconds on CPU instead of as a wrong pointer inside a kernel.

### Fail-stop
- Every wait is bounded. `deadline_s` in `binding.py` hands the plugin half of `--watchdog-timeout`, capped at
  60 s (`MAX_DEADLINE_S`). A watchdog that would leave less than 5 s (`MIN_DEADLINE_S`) is a startup refusal,
  and no watchdog means 60 s.
- An error is recorded in the sticky error word and broadcast to the peers. The plugin's README: "A failed
  wait stops the launch on every rank, so no token from a failed step is returned."
- Consequence (follows from the rule, not measured): raising the watchdog to survive a slow first compile also
  lengthens how long a rank disagreement takes to surface, up to the cap. Compile at load instead.

## What differs from vLLM
These matter most when the kernel's oracle was frozen against the other engine.

| Topic | SGLang | Consequence | Status |
|---|---|---|---|
| Layer boundary | Where SGLang's all-reduce fusion is active the all-reduce is deferred: a layer's input is the previous layer's unreduced feed-forward partial sum plus the residual, and its output is this layer's unreduced partial sum | A kernel built at vLLM's layer boundary is **not** a drop-in: the SGLang kernel starts with an input all-reduce and its expert stage does no all-reduce. Check which engine the oracle was frozen against, and refuse otherwise | read from source while building layer oracles |
| What is replaced | The model's whole forward for covered steps, not one layer inside the stock layer loop | The plugin's forward owns the layer loop for those steps; in vLLM's layer mode the stock loop stays | read from source |
| KV storage | K and V live in separate token pools, reached through the KV-pool accessors | The kernel's KV addressing follows those accessors; list them in `SGLANG_READS` | read from source |
| Runtime objects | The forward sees SGLang's own batch object (`ForwardBatch`) and attention metadata, inside a `TpModelWorker` | Routing reads the batch object and the graph key, not vLLM's forward context | read from source |
| Native determinism | With the two-stage MoE backend the native path accumulates with atomic adds, so native output is not reproducible bit for bit from run to run | Measure native-versus-native greedy agreement first, and judge on-versus-off and off-is-stock against it, not against 100% | observed while freezing layer oracles |

## Smoke specifics
- Switch on: the deployment's own launch command, at the TP size the kernel was built for, plus the commit's
  switch (`--enable-amd-monokernel` at that commit, `--monokernel-backend amd` on older ones). Switch off: the
  identical command without it. Both installed through the workflow's overlay, never by editing site-packages.
- CUDA graphs stay **on** for the smoke. The mono kernel must run inside the captured decode graphs, which is
  where its speedup comes from; a smoke with graphs off proves nothing about the delivered path.
- Read the build line the binding logs (widths built, build time, memory, weight copies, wait deadline): it
  gives the size of any second weight copy and the deadline actually in force.
- The counter line and the server command are the two places where the acceptance checks branch per engine.
  The other checks (kernel carried unchanged, call consistency, selection table, framework reads) are the same
  as for vLLM.
- Upstream, SGLang pins the plugin by commit, so an upstream delivery is two changes: the plugin commit and
  the pin bump. Smoke the pair, not the plugin commit alone.

The checks themselves are in [smoke_and_acceptance.md](smoke_and_acceptance.md).
