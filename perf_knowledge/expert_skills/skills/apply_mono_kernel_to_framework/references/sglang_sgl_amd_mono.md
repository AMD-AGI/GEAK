# SGLang: a mono kernel as an `sgl-amd-mono`-style plugin

The SGLang-specific shape of the delivery. The steps that both frameworks share (read the source at the
target commit first, routing from rank-shared data only, build before KV sizing, counters, fail-stop,
smoke) are in [../skill.md](../skill.md#procedure); this page holds the SGLang form of each step and what
is specific to SGLang.

## Upstream references

- [RFC #43949](https://github.com/sgl-project/sglang/issues/43949): the overall spec;
- [#44353](https://github.com/sgl-project/sglang/pull/44353): merged. A vendor-neutral framework in
  `python/sglang/srt/monokernels/`, plus 5 files and 18 lines outside it;
- [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono): the AMD plugin that holds kernels, specs
  and forwards. SGLang pins it by commit.

**Read `python/sglang/srt/monokernels/` of the SGLang source at the commit you target first.** It defines
the spec type, the registration hook and the selection key. Use exactly those, and record the
`api_version` you target.

## Shape of the deliverable

- **Switch.** `--monokernel-backend amd`, off by default.
- **Plugin layout.** Mirror one model in `sgl-amd-mono`: a package with the FlyDSL kernel, `mk_runtime`,
  the spec, and a forward. The patch adds it under the plugin's layout. If the framework at the commit only
  offers whole-step `model.forward` replacement, the forward runs stock modules for every layer except the
  covered kind, and calls the mono kernel for that one. The model file itself stays untouched.
- **`MonoSpec` fields:**
  - `devices`: per ISA (`gfx942`, `gfx950`), the widths (`WIDTHS`, which is device-dependent through LDS),
    the supported TP sizes, and the number of resident blocks (`PHYSICAL_CU_COUNT`);
  - `features` / `requires`: the allowed features, and the environment conditions with every `REFUSALS`
    entry expressed as a requirement;
  - `numerics`: `TOLERANCE` with per-output bounds (from `NUMERICS`: the KV write is exact, and
    activations at the derived relative bound). Claim `BITWISE` only if the layer unittest shows
    max_abs = 0 for every output;
  - `build`: JIT every width at load, before KV-cache sizing; `load_tensor`; `api_version`.
- **Selection.** SGLang selects by CUDA-graph key (forward mode, batch size, tokens per request), which is
  identical on every rank and replays as captured. Pure decode at widths in `WIDTHS` only.
- **`SGLANG_READS`.** List every SGLang-internal name the forward or spec reads (modules, attributes,
  attention-backend fields, KV-pool accessors), plus the CPU test that imports the commit and asserts each
  name exists. Run that test before the smoke.
- **Fail-stop.** Every wait is bounded (the kernel's deadline is half the watchdog, clamped to 5–60 s). An
  error is recorded in the sticky word and broadcast to peers, and the step produces no token. Decode the
  error on the host and abort with the cause.
- **Counters.** Per-rank launches and skips by reason, readable from the server log.

## Smoke specifics

- Server: `python3 -m sglang.launch_server --model-path <MODEL_PATH> --tp <TP> --monokernel-backend amd`
  (on) versus without the flag (off), installed through the workflow's overlay setup.
- CUDA graphs stay **on** for the smoke. The mono kernel must run inside the captured decode graphs, which
  is where its speedup comes from.

## Sources

- Upstream links above are public SGLang issues, pull requests and the `sgl-amd-mono` repository.
