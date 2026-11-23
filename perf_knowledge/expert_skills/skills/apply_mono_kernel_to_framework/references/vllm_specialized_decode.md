# vLLM: a mono kernel as a `specialized_decode` layer

Status: a **design prior**. The delivery phase this is ported from was designed but not exercised end to end
(see the status note in [skill.md](../skill.md)). The interface facts below were read from upstream source at
the public commit [`f09d5e30c`](https://github.com/vllm-project/vllm/commit/f09d5e30c35862c19a17dd1706147d29d1f7c4f4)
and, where stated, on `main` on 2026-11-23. They are version-dependent: **the commit you target is the
authority**. Where it and this file disagree, follow the commit and write the difference in your notes.

## Upstream references (public)
| Link | What it gives you |
|---|---|
| [RFC 60904](https://github.com/vllm-project/vllm/issues/60904) | The overall standard for specialized (mono-kernel) decodes under `vllm/models` |
| [PR 61311](https://github.com/vllm-project/vllm/pull/61311) | A shared `DecodeSpec` / `SpecializedDecode` abstraction. In review when this was written, so the interface may change |
| [PR 62256](https://github.com/vllm-project/vllm/pull/62256) | The authoring guide for ROCm decode mono kernels |
| [PR 63486](https://github.com/vllm-project/vllm/pull/63486) | The CI plan: mono kernels checked stage by stage against float64 references |
| [PR 60397](https://github.com/vllm-project/vllm/pull/60397), [PR 61082](https://github.com/vllm-project/vllm/pull/61082) | DeepSeek-V4.1-Flash examples: one mono decode layer, then every backbone layer |
| [PR 61662](https://github.com/vllm-project/vllm/pull/61662) | A layer-mode example for another model; the closest shape to what this skill delivers |

## Read the commit first
Before writing anything, open the framework source at the exact commit you target and note what it has. At the
pinned commit:
- **The switch.** `KernelConfig.monokernel` in `vllm/config/kernel.py` (lines 299-303), default off, registered as
  `--monokernel` in `vllm/engine/arg_utils.py`. Its docstring ends: "A deployment that cannot run it is refused
  at startup." One package at that commit is switched by the additional-config key `decode_backend` instead; on
  `main` the same docstring says the two are one switch.
- **Packages to copy the shape of.** `vllm/models/<model>/amd/specialized_decode/`: a README (switch, supported
  deployments, steps, fail-stop, layout), `runtime/` (configuration and refusals, `SpecializedDecode.plan()`
  returning a step whose `layer(i)` hands out one layer's kernel, guards, the per-rank runner, peer memory and
  TP agreement) and `mono/` (the kernel source). The model files only call in.
- **An older, flatter form.** `vllm/models/deepseek_v41/amd/mono_decode.py` (`MonoDecodeLayer`).
- **Not there.** No shared decode-abstraction module was found by file name outside the per-model packages, at
  the pinned commit or on `main`; each package defines its own `SpecializedDecode`. Treat PR 61311 as pending.

If your commit has none of these, the hook you add is yours to justify: keep it to a flag and one call site at
model load, and list it separately from the model-owned code.

## Shape of the patch
| Part | Rule |
|---|---|
| Location | The model owns the code: `vllm/models/<model>/amd/specialized_decode/`. No change to the vLLM core beyond what the commit already provides |
| Switch | The commit's opt-in switch, default off (`--monokernel` at the pinned commit). The packages there refuse the first-generation model runner and stock `torch.compile` |
| Mode | Layer mode: `plan()` runs once per step, and the model's layer loop asks the step for the kernel of each layer index (`Step.layer(i)`). Stock modules and parameters are never replaced. Step mode, which replaces the whole forward, is out of scope when you deliver one layer kind |
| `plan()` | Pure decode with a row count in the kernel's widths uses the mono kernel. Prefill, mixed batches, speculative verify and any other width keep the stock path. The decision depends only on metadata every rank holds alike and replays under CUDA graphs exactly as captured. Upstream's `decide_step` (`runtime/routing.py`) returns the decision together with its reason |
| Refusal | Explicit, every reason at once, the same verdict on every rank, never a silent fallback. How it is expressed is the next section |
| Build | JIT every width at model load, **before the KV-cache profiling run**, so that KV sizing sees the kernel's memory: peer buffers, control region, any weight repack. Create the cross-rank peer group over the TP group, using the CPU group of `get_tp_group()` for the object gather |
| Counters | Per rank: `launches`, read from the control region at log intervals and not per step, and `skips{reason}` from `plan()`. Use what the commit has first: at the pinned commit one package keeps `mono_steps` / `stock_steps` on its `SpecializedDecode` and logs launch coverage now and then without a host sync (`output_step` in `runtime/executor.py`: output steps, launches, covered layers each); the older DeepSeek-V4.1 file logs a one-time "active" line and a debug-level skip reason |
| Fail-stop | Use the kernel's sticky error word. On an error the engine stops with the cause. Never emit the token of a failed step (two upstream designs below) |

### How a refusal is expressed
The internal design this skill started from treated a refusal as a log line plus a disabled kernel. Upstream
does the opposite for an explicit opt-in, and upstream wins:
- **A deployment that cannot run fails at startup with an error.** `vllm/config/kernel.py` says so for the
  switch. `MonoDecodeLayer._create` in `vllm/models/deepseek_v41/amd/mono_decode.py` (line 384) raises
  `ValueError("--monokernel ...")`: "an explicit opt-in that cannot run is an error". In the `specialized_decode`
  packages, `specialized_decode_config` (`runtime/spec.py`) joins every entry of `refusal_reasons` into one
  `ValueError`, and a README states that startup fails "with one `ValueError` that lists every unsupported
  setting". The older DeepSeek-V4.1 file raises on the first failing reason only; list them all in new code.
- **Every rank reaches the same verdict first.** The model stage agrees across TP ranks before raising ("Every
  TP rank agrees on each verdict before anything collective runs"); a refusal on one rank alone is a hang.
- **A layer the kernels cannot serve is not an error.** In `MonoDecodeLayer._create` such a layer keeps vLLM's
  path with a one-time info log. That is partial coverage, explicit and identical on every rank.
- **A step the kernels cannot take is not an error.** It runs the stock layers (`MonoDecodeLayer._skip` logs the
  reason once at debug level). This skill additionally counts it by reason.

### The three refusal stages
1. **Configuration, before load.** Pipeline, data and expert parallelism; LoRA; speculative decoding; KV
   connectors; quantization formats other than the loaded one; a TP size or device the kernel was not built
   for; fewer compute units than the kernel's resident blocks (a partitioned GPU); and every entry of the
   layer kind's own refusal list.
2. **Model, after load.** Weight formats, shapes and dtypes as actually loaded, not as the checkpoint stores
   them.
3. **Cache, at each KV-cache bind.** Layout, block size, KV dtype, slot-mapping form.

One upstream README names the same three points: in the model's constructor, when the model state builds the
executor after load, and at every KV-cache bind.

### Two upstream fail-stop designs
- **Device-side assert.** After the layer loop of every forward and before logits and sampling, an asynchronous
  device assert checks that the sticky error word is clear. If it is set the GPU queue traps, so no token is
  returned from the failed step or any later one, with no host synchronization per step. This is the guarantee
  this skill requires.
- **Periodic agreement.** Each step queues a non-blocking copy of the rank's failure word; every 16 steps the
  ranks agree over the TP CPU group and all raise at the same step. Its README states that tokens of the failed
  steps can reach the client before the engine stops. If your commit only offers this form, say so in the report.
- In both, in-kernel waits are bounded (10 s) and every rank stops for the same step: a rank that raised alone
  would leave the others waiting in the next collective.

## The layer glue
- **Weights are read in place.** They are the tensors the framework loaded; nothing is copied or re-quantized
  unless the kernel cannot read the loaded layout (then see the second-weight-copy row below).
- **Same contract as the native layer.** For the same inputs (positions, hidden states, residual, attention
  metadata, KV cache, block table, sequence lengths) the outputs and the KV-cache write equal the native
  layer's within the per-output bound. The two paths share every tensor at the layer boundary, so they can
  interleave freely.
- **Eligibility from metadata only.** Nothing but tensor metadata and configuration feeds the choice of path.
- **Attention inputs.** Read the attention metadata from the forward context (`get_forward_context()`) and
  hand the KV cache, slot mapping, block table and sequence lengths to the kernel. The KV write happens inside
  the kernel, at the framework's slot mapping and in its format.
- **State outside capture.** Per-instance state (peer group handle, weight views, compiled launchers) is
  created at load or on the first eligible call, in either case outside any graph capture.
- **No native compute hidden in the glue.** Calling the stock attention or MLP sub-module from the kernel path,
  and counting the step as a launch, invalidates the A/B.

## vLLM-specific pitfalls
| Pitfall | What happens | Rule | Status |
|---|---|---|---|
| KV cache reallocated after profiling | vLLM sizes the KV cache from a profiling run and binds the real cache afterwards; pointers captured at the first bind read the old cache | Re-read the KV pointers every step, or rebind whenever the cache tensor's identity changes, and rerun the stage-3 check at every bind. Upstream's `bind_caches` rebinds on an identity change and refuses a first bind or rebind during graph capture | observed on the first serving run |
| Mixed graph mode | With `FULL_AND_PIECEWISE` the engine also captures piecewise graphs using small dummy decode batches. They met the kernel's conditions, so the compiled model's piecewise graphs of those sizes were never captured; a later short prefill replayed a missing graph and faulted on every rank | Steps in piecewise mode, capture or replay, run vLLM's own layers. Upstream routes the same way: a piecewise graph is keyed by token count alone and replays mixed steps, so it stays stock | observed, found by a forced-decode test and not by the serving benchmark |
| Collectives around capture | The collective library's watchdog queries events of collectives issued before the capture, which the runtime rejects even in thread-local capture mode | Let the ranks settle briefly after warm-up and agree on capture success across ranks; native-path collectives in the glue need the same care | observed in a multi-rank test harness |
| Watchdog thread | A host thread that reads the error word while a graph is captured in global capture-error mode broke the first start-up | Capture in relaxed mode (`capture_error_mode` of `torch.cuda.graph`) while that thread runs | observed |
| Leaked first runner | An import traceback that referenced a constructor frame kept the first model runner, and its GPU memory, alive | Drop the references on failure | observed |
| Trapped rank does not exit | A kernel that traps on its deadline leaves a process that writes a GPU core dump and stays alive | Set `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` (a ROCm runtime variable) | reported, not re-tested |
| Second weight copy | A kernel that cannot read the framework's layout needs its own copy, which competes with the KV cache. Upstream's DeepSeek-V4.1 kernels do this on gfx942 and read in place on gfx950 | Allocate it before KV sizing and state its size per rank in the report | design |

## Smoke specifics
- Switch on: the deployment's own serve command, at the TP size the kernel was built for, plus the commit's
  switch. Switch off: the identical command without it. Both installed through the workflow's overlay, never
  by editing site-packages.
- The switch-off run must also match the **unpatched** server token for token. That is what shows the patch is
  inert when off.
- A launch counter of 0 with the switch on means `plan()` routed everything away. That is a failed effect
  proof even if the outputs match.
- Keep full CUDA graphs for decode on in both legs. An eager leg measures launch latency, not the kernel.
- Upstream keeps an engine-fault test for one package (`tests/models/<model>/amd_specialized_decode/`,
  `test_engine_fault.py`): it stalls a rank in a served model and checks that the engine stops. Run the
  equivalent before trusting fail-stop.

The checks themselves are in [smoke_and_acceptance.md](smoke_and_acceptance.md).
