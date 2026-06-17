# SGLang Internals — Where Kernels Live & How to Swap Them Without Touching site-packages

This is the map the **Config Tuner** and **Kernel Extractor** use to find the editable surface of a
running sglang server, and how the **e2e Integrator** overlays an optimized kernel reversibly. Seeded
from a sglang 0.5.11 / ROCm setup; verify paths against the installed version at run time
(`python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))"`).

## 1. The three knobs that change which kernel runs (no source edit)
The spec's "backend can be changed from three dimensions": **launch flags**, **env vars**, **source**.
Always try flags+env FIRST (Config Tuner, Tier 0) — they are reversible and reshape the profile.

### Launch flags (`python3 -m sglang.launch_server ...`)
- `--attention-backend {triton, aiter, fa3, ck_tile, torch_native, ...}` and the decode/prefill
  split `--prefill-attention-backend` / `--decode-attention-backend` (version-dependent). Biggest
  lever for attention-heavy serving.
- `--quantization {fp8, awq, gptq, ...}`, `--kv-cache-dtype {auto, fp8_e4m3, ...}`. Largest
  prefill compute lever when the accuracy budget allows.
- `--enable-torch-compile`, `--torch-compile-max-bs N` — fuses elementwise/norm chains.
- `--cuda-graph-max-bs N`, `--disable-cuda-graph` — decode launch-overhead. (HIP graph on ROCm.)
- `--chunked-prefill-size`, `--max-prefill-tokens`, `--schedule-conservativeness` — prefill/decode
  interleave.
- `--tp-size / --dp-size / --ep-size`, `--mem-fraction-static` — parallelism + KV budget.
- `--speculative-algorithm {EAGLE, NEXTN, ...}` + draft model — decode boost (only if the served model
  ships a draft/speculative head; verify against its config, do not assume).
- `--enable-flashinfer-mla`, MoE flags (`--enable-ep-moe`, etc.) when the arch matches.

### Env vars (set before launching the server)
- `SGLANG_TORCH_PROFILER_DIR=<dir>` — turns on the profiler dump target (used by the Profiler).
- aiter toggles: `SGLANG_USE_AITER`, `SGLANG_AITER_MOE`, and friends (grep the installed tree:
  `grep -rEl "os.environ.get\\(.SGLANG_|os.getenv\\(.SGLANG_" $SGLANG_DIR`).
- hipBLASLt / Tensile tuning DB: `HIPBLASLT_TUNING_FILE` / `TENSILE_*`; rocBLAS `ROCBLAS_TENSILE_*`.
  Populating the tuning DB for the exact GEMM shapes is the classic "untuned GEMM falls back to a
  default solution" fix (watch the `not found tuned config ... using default config` warnings).
- `HIP_VISIBLE_DEVICES` — pin GPUs (the bench script already does this).

## 2. Where editable kernels live (the source surface)
- **sglang python kernels** (Triton, custom): `$SGLANG_DIR/srt/layers/` — `attention/`,
  `moe/`, `quantization/`, `activation.py`, `layernorm.py`, `rotary_embedding.py`, and the
  linear-attn / mamba path for hybrid models. These are the `triton`/`fused_custom`/`reduction_norm`
  entries the Profiler flags as `editable`.
- **aiter** (separate package `import aiter`): AMD fused ops. Editable, but usually treated as a
  backend swap target, not a rewrite.
- **library calls** (hipBLASLt/Tensile/rocBLAS GEMM, CK attention): NOT source-editable — they are
  the `library_gemm`/`library_attn` classes → Config Tuner territory only.

To locate the function behind a profiled Triton kernel name: grep the kernel's `short_name`
(snake_case) under `$SGLANG_DIR` — sglang Triton kernels are defined with `@triton.jit` and the
JIT'd function name appears verbatim in the trace.

## 3. The reversible overlay (how an optimized kernel gets back into the server)
**Never edit site-packages.** The overlay is a **manifest-driven `sitecustomize.py`**, built by
`scripts/overlay_setup.py`, that lives on `PYTHONPATH=$EVAL_DIR/overlay:$PYTHONPATH` and applies its
changes at interpreter startup. The overlay dir contains **NO `sglang/` package** — three entry kinds
compound via `_overlay_manifest.json`:

> ⚠️ **Do NOT copy a package subtree.** Copying `sglang/...` (with an `__init__.py` chain) onto an
> earlier PYTHONPATH entry **fully shadows** the install — Python does not merge regular packages
> across path entries, so every sibling submodule vanishes and `import sglang` breaks. This was a real
> bug. The mechanisms below inject only the specific patched module, leaving parents/siblings resolving
> from the install.

### (a) `add-module` — inject a single patched submodule (preferred for a code change)
```bash
python3 scripts/overlay_setup.py add-module \
  --overlay "$EVAL_DIR/overlay/cand_X" \
  --module "sglang.srt.layers.attention.fla.chunk_fwd" \
  --patch  "<final_patch.diff>" --src-file "<installed chunk_fwd.py>"
```
At startup the sitecustomize loads the patched file via `importlib.util.spec_from_file_location`, puts
it in `sys.modules[<dotted>]`, and binds it as an attribute on its parent — so both `import a.b` and
`from a.b import c` see the patch, while `import sglang` and all siblings still resolve from the
install. Verify: `PYTHONPATH=<overlay> python3 scripts/overlay_setup.py check --module <dotted>`.

### (b) `add-rebind` (monkeypatch) — rebind a single function/symbol
```bash
python3 scripts/overlay_setup.py add-rebind \
  --overlay "$EVAL_DIR/overlay/cand_X" \
  --target "sglang.srt.layers.activation:silu_and_mul" \
  --impl-module my_opt --impl-attr fast_silu_and_mul --impl-file my_opt.py
```
At startup: `setattr(import_module(mod), attr, getattr(import_module(impl_module), impl_attr))`.

### (c) `add-capture` — install a shape/I-O capture hook (used by the Kernel Extractor)
Both are reversible — just drop the overlay dir from `PYTHONPATH`. Head-op winners that are pure
**env/flag** (TunableOp CSV, `HIPBLASLT_TUNING_FILE`, `--quantization fp8`) need NO overlay at all —
they go straight into the server launch env/flags.

## 4. Validation hooks
- The server must be WARM before any timed bench (see `e2e_optimization.md` measurement discipline).
- Output parity: run the bench with temp=0/greedy + fixed seed against baseline; a faster wrong
  server is a regression. The e2e Integrator gates on this.
- Re-profile AFTER each accepted change — the bottleneck shifts (e.g. once GEMM is tuned, attention
  or a Triton norm may become the new top entry).

## 5. Reliability notes
- Pin the sglang version + commit in the run log; paths above can move between releases.
- If a flag is unknown to the installed version, `launch_server --help` is the authoritative list —
  the Config Tuner should grep `--help` before sweeping an axis.
- One axis at a time; keep a warm server across a sweep where possible; record throughput median +
  spread so a win is distinguishable from noise.
