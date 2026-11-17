# vLLM: a mono kernel as a `specialized_decode` layer

The vLLM-specific shape of the delivery. The steps that both frameworks share (read the source at the
target commit first, routing from rank-shared data only, build before KV sizing, counters, fail-stop,
smoke) are in [../skill.md](../skill.md#procedure); this page holds the vLLM form of each step and what
is specific to vLLM.

## Upstream references

- [RFC #60904](https://github.com/vllm-project/vllm/issues/60904): the overall spec (open);
- [#61311](https://github.com/vllm-project/vllm/pull/61311): the shared `DecodeSpec` / `SpecializedDecode`
  abstraction, still in review, so the interface may change;
- [#62256](https://github.com/vllm-project/vllm/pull/62256): the authoring guide;
- [#63486](https://github.com/vllm-project/vllm/pull/63486): the CI plan;
- [#60397](https://github.com/vllm-project/vllm/pull/60397) /
  [#61082](https://github.com/vllm-project/vllm/pull/61082): DeepSeek-V4.1 examples;
- [#61662](https://github.com/vllm-project/vllm/pull/61662): GLM layer mode.

Read the vLLM source at the commit you target before writing anything. Use whatever that commit actually
has (`KernelConfig.monokernel`, an existing `specialized_decode/` package of another model,
`vllm/models/common/mono/`). Where this guide and the commit disagree, follow the commit and say so in
your notes.

## Shape of the patch

- **Location.** The model owns the code: `vllm/models/<model>/amd/specialized_decode/`, with no changes to
  the vLLM core beyond what the commit already provides. If the commit has no hook at all, add the
  smallest one possible (a flag and one call site at model load) and list it separately in the patch.
- **Switch.** `--monokernel`, i.e. `KernelConfig.monokernel`, default off. It requires Model Runner V2,
  and models using `@support_torch_compile` are refused.
- **Layer mode.** Each covered layer index is replaced by the mono kernel layer, GLM-style
  `step.layer(idx)`. Step mode (replacing the whole forward) is out of scope: the delivery covers one
  layer kind.
- **`plan()` (routing).** Pure decode with a row count in `WIDTHS` uses the mono kernel. Prefill, mixed
  batches, speculative verify, and any width outside `WIDTHS` keep the stock path. The decision depends
  only on the batch descriptor every rank shares, and replays under CUDA graphs exactly as it was
  captured.
- **Refusal instead of silent fallback**, in three stages, each listing every reason at once:
  1. config, before load: PP, DP, EP, LoRA, speculative decoding, KV connectors, quantization formats
     other than the loaded one, a TP size the kernel was not built for, and the layer kind's `REFUSALS`;
  2. model, after load: weight formats, shapes and dtypes as actually loaded;
  3. cache, at each KV-cache bind: layout, block size, KV dtype, slot mapping form.

  A refusal disables the mono kernel for the run with a log line that names every reason; it is not an
  exception.
- **Build.** JIT every width at model load, before the KV-cache profiling run, so KV sizing sees the mono
  kernel's memory (peer buffers, control region, any weight repack). The `PeerGroup` is created over the TP
  group (`get_tp_group()`'s CPU group for `all_gather_object`).
- **Counters.** Per rank: `launches` (read from the control region at log intervals, not per step) and
  `skips{reason}` from `plan()`. Expose them in a periodic log line, or in the metrics endpoint if the
  commit has one for specialized decode.
- **Fail-stop.** Use the kernel's sticky error word. On a decoded error, abort the engine with the decoded
  cause; never emit the token of a failed step.

## Smoke specifics

- Server: `vllm serve <MODEL_PATH> -tp <TP> --monokernel` (switch on) versus the same command without the
  flag (switch off), installed through the workflow's overlay setup, never by editing site-packages.
- The switch-off run must also match the unpatched server bit for bit, which shows the patch is inert when
  off.
- A launch counter of 0 with the switch on means `plan()` routed everything away. That is a failed effect
  proof, even if outputs match.

## Sources

- Upstream links above are public vLLM issues and pull requests.
