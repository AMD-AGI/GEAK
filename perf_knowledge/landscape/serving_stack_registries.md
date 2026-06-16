---
title: How production serving stacks organize kernel layers and select per-op backends
kind: landscape
updated: 2026-06-09
sources:
  - https://github.com/vllm-project/vllm/issues/33163
  - https://docs.vllm.ai/en/latest/design/custom_op/
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
  - https://huggingface.co/docs/kernels/en/layers
  - https://nvidia.github.io/TensorRT-LLM/
  - https://docs.nvidia.com/cuda/cublas/
  - https://deepwiki.com/NVIDIA/cudnn-frontend/2.5-heuristics-and-engine-configuration
---

# How production serving stacks organize kernel layers & select a backend per op per HW

Research scope: study how vLLM, SGLang, TensorRT-LLM, HuggingFace TGI/`kernels`, PyTorch
`torch.library`+Inductor, and the cuDNN/cuBLASLt "registry+heuristic" libraries **organize their
kernel layer** and **select a backend per operator per hardware**, so we can (a) design the
`perf_knowledge/index/sota_registry.yaml` schema to be *consumed by code*, not just read by humans, and
(b) wire it into `e2e_workflow` so the `system_architect` and `op_benchmarker` query it.

## TL;DR

Every mature stack separates the **same three concerns** that perf_knowledge must also separate:

1. **A registry** — a name → implementation table. The key is *(op[, device][, capability][, mode])*;
   the value is a *factory/handle* (a callable that lazily imports + builds the impl). Always
   string-keyed and decorator-registered (`@register_attention_backend("triton")`,
   `@CustomOp.register("rms_norm")`, `register_kernel_mapping(...)`).
2. **A selection policy** — how one entry is chosen. Three tiers, used in combination:
   *static config / priority list* (vLLM `ir_op_priority`, SGLang `--attention-backend`),
   *per-hardware/per-mode table* (HF `kernels` capability ranges, vLLM platform dispatch),
   and *runtime heuristic + autotune-with-cache* (cuBLASLt `…AlgoGetHeuristic`, cuDNN engine
   heuristics, TRT-LLM `Autotuner.choose_one`).
3. **A packaging/distribution layer** — how the kernel binary actually arrives: compiled into the
   engine (vLLM csrc, TRT plugins), pulled from a versioned hub (HF `kernels`), or shipped as a
   vendor op library that the registry dispatches into (aiter, cuDNN/cuBLASLt, MIOpen).

Key structural lessons for perf_knowledge: (i) the registry value should be a *handle/seam* (env var, launch
flag, or import path), not prose; (ii) selection must be keyed on *(op, gen, dtype, regime)* and
return a single **winner** plus ranked alternates, mirroring "heuristic returns a list sorted by
estimated time, index 0 = default"; (iii) record an **autotune-cache pointer** (aiter tuned-gemm
CSV, hipBLASLt offline tune) the way cuBLASLt serializes the chosen `algo`; (iv) carry an **errata /
exclusion** field (cuDNN's `CUDNN_ERRATA_JSON_FILE` blocklist) for known-bad pairs; (v) AMD must be a
first-class key (`rocm`/`gfx94x`), never a fallback.

---

## vLLM — platform dispatch + `CustomOp` registry + the `ir_op_priority` config (the closest analog)

vLLM is the single most useful model because it is converging on *exactly* the per-op-per-hardware
selection table we want.

**Organization.** Three layers:
- **Platform abstraction** (`vllm/platforms/interface.py`, `RocmPlatform` in `platforms/rocm.py`):
  `current_platform` is lazily initialized; each platform implements `get_attn_backend_cls()`,
  `check_and_update_config()`, etc. ROCm detects arch via
  `torch.cuda.get_device_properties().gcnArchName`, enables FP8 on gfx94/gfx95, and uses FNUZ fp8 on
  gfx94 — i.e. **device knowledge is centralized in the platform object**, not scattered in models.
- **`CustomOp` registry** (`vllm/model_executor/custom_op.py`): ops register by name
  `@CustomOp.register("rms_norm")` and implement `forward_native` / `forward_cuda` / `forward_hip` /
  `forward_cpu`. Dispatch: ROCm → `forward_hip()`, falling back to `forward_native()`. Enable/disable
  is driven by `compilation_config.custom_ops` (the special tokens `"all"`/`"none"`, plus per-op
  overrides), and is auto-set to `"none"` when the Inductor torch.compile backend is active (so the
  compiler fuses instead). **This is a name→impl table with a global on/off policy and a torch.compile
  interaction** — directly analogous to a SOTA registry with a "let the compiler do it" mode.
- **Attention backend selection**: `--attention-backend` flag, plus the
  `AttentionBackendEnum.CUSTOM` placeholder that out-of-tree vendors register into
  (`register_backend(backend=CUSTOM, class_path="atom...AiterBackend")`, run with
  `VLLM_ATTENTION_BACKEND=CUSTOM`).

**Selection.** Historically *13 env vars* (`VLLM_ROCM_USE_AITER`, `VLLM_ROCM_USE_AITER_RMSNORM`, …).
The **`--aiter-config` / `--kernel-config` RFC (#33163, Jan 2026)** replaces them with a structured,
validated config. Two designs, both directly relevant:
- **`ir_op_priority`** — *per-operator backend priority*, the schema we should copy:
  ```
  --kernel-config.ir_op_priority.rms_norm=vllm_c
  --kernel-config.ir_op_priority.linear=aiter
  --kernel-config.ir_op_priority.attention=aiter
  ```
  Critically, because "kernels of the same type in AITER can be implemented via CK, ASM, triton,
  gluon, HIP," the values can be *variant-qualified*: `aiter_ck`, `aiter_asm`, `aiter_triton`.
- **`--aiter-config`** — per-feature booleans gated by a top-level `enabled`, also as JSON:
  `--aiter-config '{"enabled": true, "mha": false, "unified_attention": true, "fp8_bmm": true}'`.

**Packaging / AMD.** Kernels are compiled into vLLM (`csrc/`, `csrc/rocm/` paged attention,
QuickReduce), or dispatched into the **aiter** op library, or provided by the **ATOM** out-of-tree
plugin which registers via vLLM's official `register_platform`/`register_model`/`register_backend`
entry points and acts as an "incubation layer" before upstreaming. The Feb-2026 ROCm blog stresses
the *software-layer routing* philosophy: `ROCM_AITER_FA` routes workloads at the software layer so the
same logic spans MI300X→MI325X→MI355X without hardware-specific rewrites.

**License:** Apache-2.0. Real URLs:
[CustomOp design](https://docs.vllm.ai/en/latest/design/custom_op/),
[aiter-config RFC #33163](https://github.com/vllm-project/vllm/issues/33163),
[ROCm attention blog](https://vllm.ai/blog/2026-02-27-rocm-attention-backend),
[ATOM plugin guide](https://rocm.github.io/ATOM/docs/vllm_plugin_backend_guide.html).

## SGLang — decorator registry + automatic per-HW/per-phase defaults

**Organization.** A literal registry: `ATTENTION_BACKENDS = {}` plus a decorator factory
`register_attention_backend("flashinfer")(create_flashinfer_backend)`. Each value is a **factory**
taking a `runner` that *lazily imports + instantiates* the backend class. Selection is a dict lookup
of the string name.

**Selection.** If `--attention-backend` is unset, SGLang **auto-selects by hardware + model arch**:
Hopper→`fa3`, Blackwell→`trtllm_mha`, Ampere/Ada→`flashinfer` else `triton`, **AMD/ROCm→`triton`**
(aiter where available). Per-phase override: `--prefill-attention-backend` /
`--decode-attention-backend`; if they differ SGLang auto-wraps a hybrid dispatcher. Per-hardware and
per-model branching lives *inside the factory*: `create_dsv4_backend` branches on `is_hip()`
(`DeepseekV4HipRadixBackend` vs CUDA); `create_trtllm_mla_backend` asserts MLA-only;
`attn_backend_wrapper` swaps in linear-attention backends (`GDNAttnBackend`, `Mamba2AttnBackend`, …)
for hybrid models and returns a `HybridLinearAttnBackend`. Deprecated names are *aliased*
(`"nsa"`→`"dsa"` with a warning).

**Packaging / AMD.** aiter is the **default** kernel backend on AMD for MoE/GEMM/attention/all-reduce;
SGLang ships `sgl-kernel` and pulls aiter (Triton/CK/ASM impls). If aiter lacks coverage for a new
model, the documented fallback is "switch to Triton." Selection is *runtime detection + name*, not a
static per-shape table.

**License:** Apache-2.0. Real URLs:
[attention backend docs](https://docs.sglang.ai/advanced_features/attention_backend.html),
[attention_registry.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py),
[AITER+SGLang integration blog](https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html).

## NVIDIA TensorRT-LLM — plugins (build-time) + Autotuner (runtime, profiled & cached)

**Organization.** Two parallel mechanisms:
- **TensorRT engine path**: a **plugin** system (e.g. the GEMM plugin injects tuned matmul kernels);
  **multiple optimization profiles** give TensorRT "more chances to pick a better kernel" for
  different input sizes — i.e. shape-bucketed kernel selection baked into the engine at build time.
- **PyTorch-native path (now default)**: a Python **Autotuner**
  (`tensorrt_llm/_torch/autotuner.py`). Custom ops call `tuner.choose_one(...)`, which searches a
  **profiling cache** for `(best_runner_id, best_tactic, min_time)`. Applied to fused MoE and
  NVFP4/FP8 linear ops. Some ops mix heuristic + autotune (FP8 picks CUDA-cores vs cuBLAS by batch
  size). `enable_autotuner` toggles it.

**Selection.** Autotune-with-cache keyed on the op + problem shape, returning the fastest *tactic*.
This is the canonical "**profile candidates once, cache the winner, key by shape**" pattern — the same
shape we want for the aiter/hipBLASLt offline-tune pointers in perf_knowledge.

**Packaging / AMD.** Kernels ship as compiled TRT plugins / CUDA. **No AMD support** (NVIDIA-only) —
record as `na` for AMD, but borrow the *autotuner-cache* design.

**License:** Apache-2.0. Real URLs:
[TRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/),
[autotuner crash issue (cache-key detail)](https://github.com/NVIDIA/TensorRT-LLM/issues/10679).

## HuggingFace `kernels` / Kernel Hub (+ TGI) — a versioned, per-(device,capability,mode) registry

The single most directly copyable *schema* for us.

**Organization.** `kernelize(model, mode=...)` walks modules and swaps `forward` for any layer with a
registered Hub kernel. Mappings are nested dicts registered via `register_kernel_mapping(...)` or the
scoped `use_kernel_mapping(...)` context manager:
```python
{
  "SiluAndMul": {                                  # 1. op / layer name
    "cuda": LayerRepository(repo_id="kernels-community/activation",
                            layer_name="SiluAndMul", version=1),
    "rocm": LayerRepository(...),                  # 2. device key — AMD is first-class
  },
  "MultiHeadAttention": {
    Device(type="cuda", properties=CUDAProperties(min_capability=90,        # 2'. capability range
                                                  max_capability=sys.maxsize)): LayerRepository(...),
    "cuda": { Mode.TRAINING: LayerRepository(...),                          # 3. mode key
              Mode.INFERENCE: LayerRepository(...),
              Mode.FALLBACK: LayerRepository(...) },
  },
}
```
- **Capability-range selection** with a tie-break we should adopt verbatim: ranges are *inclusive*;
  when several match, **the smaller (more specific) capability interval wins** ("more optimized for a
  specific set of GPUs"). This is exactly `gfx942` vs "all CDNA" specificity.
- **Mode fallback chain**: `INFERENCE → INFERENCE|TORCH_COMPILE → TRAINING → … → FALLBACK`. If a
  kernel doesn't support backward or torch.compile, `kernelize()` *falls back to the original layer*
  (or raises with `use_fallback=False`).

**Selection.** Pure **static per-(device,capability,mode) table** — no runtime autotune. The leaf is a
**versioned repository handle** (`repo_id`+`version`/`revision`), and `LockedLayerRepository` pins
exact builds for reproducibility.

**Packaging / AMD.** Kernels are a **first-class Hub repo type**: `get_kernel("kernels-community/...")`
detects Python/torch/CUDA-or-ROCm version and downloads the matching prebuilt binary; ready for
torch.compile; 1.7–2.5× over baseline. AMD via the `"rocm"` device key, same pattern as CUDA. **TGI**
historically baked in flashinfer/flashdecoding and moved to multi-backend (vLLM/TRT-LLM as backends);
note TGI entered *maintenance mode 2025-12-11* — the live successor of its kernel story is `kernels`.

**License:** `kernels` Apache-2.0; community kernels carry their own per-repo licenses. Real URLs:
[kernels layers docs](https://huggingface.co/docs/kernels/en/layers),
[hello-hf-kernels](https://huggingface.co/blog/hello-hf-kernels),
[kernels-community](https://huggingface.co/kernels-community),
[TGI multi-backend](https://huggingface.co/blog/tgi-multi-backend).

## PyTorch `torch.library` + Inductor — opaque custom ops, lowerings, and pattern-fusion

**Organization.** `@torch.library.custom_op("ns::op", mutates_args=...)` registers an op; by default
torch.compile treats it as **opaque** (Inductor runs it as-is). To make an op *participate*, register
a **lowering** (`torch/_inductor/lowering.py`, one lowering per aten op) or a **pattern**
(`register_lowering_pattern` / `register_graph_pattern` / `register_replacement`) so Inductor can fuse
it. A 2025 RFC adds `register_custom_pass(pass, stage)` for ordered custom FX passes.

**Selection.** Not a backend *bake-off* — it's *compiler-driven*: the registry decides whether an op
stays a hand kernel or gets fused/codegen'd. vLLM uses exactly these pattern passes for RMSNorm+quant,
QK-norm+RoPE fusions. The lesson for perf_knowledge: a backend entry needs a `torch_compile` flag (does the
kernel survive compilation / is it better fused?) — mirrors HF `kernels`' `TORCH_COMPILE` mode.

**Packaging / AMD.** Custom ops are user/library C++/CUDA/HIP or Triton; Inductor codegens Triton for
both CUDA and ROCm. **AMD: yes** (Inductor+Triton on ROCm). **License:** BSD-3.
[Custom Operators](https://docs.pytorch.org/docs/stable/compile/programming_model.custom_ops.html),
[Custom Backends](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html).

## cuDNN / cuBLASLt — the canonical "registry + heuristic + autotune-cache + errata" design

The library-level archetype of what a *machine-queryable* selection layer looks like.

- **cuBLASLt**: `cublasLtMatmulAlgoGetHeuristic(...)` returns an array of `cublasLtMatmulAlgo_t`
  **sorted by increasing estimated compute time** — *index 0 = default best guess*. A
  `cublasLtMatmulPreference_t` **filters** candidates (max workspace, alignment, device-utilization).
  Autotune path: request up to ~100 candidates, time each on real data, pick fastest by median. The
  chosen `algo` is **trivially serializable** → cache & reload (same library version). Recommender
  accuracy ~93%.
- **cuDNN v9**: engine heuristics via `CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR` (frontend
  `EngineHeuristics_v8`) with **modes A/B/fallback**; can target deployment **SM count**; engines are
  shipped as *separately dlopen-able libraries* (`cudnn_engines_precompiled`, `cudnn_heuristics`,
  `cudnn_engines_runtime_compiled`). `cudnnFindPlan` = the exhaustive autotune path.
- **Errata filter** (`CUDNN_ERRATA_JSON_FILE`): a JSON **blocklist** of known-bad engines, *version-
  and arch-scoped* (`cudnn_version_start`/`_end`, `-1` = ongoing). **Users can append their own.**

**Selection.** Tiered exactly as perf_knowledge needs: fast heuristic → ranked list → optional autotune →
cache → minus an errata blocklist. **AMD analog:** hipBLASLt (offline tuning), MIOpen
(Find/Immediate), rocBLAS — same heuristic-vs-tune split. **License:** proprietary NVIDIA (design
borrowed, not code). [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/),
[cuDNN heuristics](https://deepwiki.com/NVIDIA/cudnn-frontend/2.5-heuristics-and-engine-configuration).

---

## Comparison table

| stack | op → backend mechanism | per-HW selection | kernel packaging | AMD |
|---|---|---|---|---|
| **vLLM** | `CustomOp.register(name)` + `forward_{cuda,hip,native}`; attn via `--attention-backend`/`CUSTOM`; **`ir_op_priority.<op>=<backend[_variant]>`** | `current_platform` detects gfx94/95, FNUZ; per-op priority + 13→config env migration | compiled `csrc/`+`csrc/rocm/`; aiter op-lib; ATOM out-of-tree plugin | **first-class** (gfx942/gfx950, aiter, ATOM) |
| **SGLang** | `ATTENTION_BACKENDS` dict + `@register_attention_backend(name)` → lazy factory | auto by HW+arch (Hopper=fa3, AMD=triton/aiter); per-phase prefill/decode; `is_hip()` inside factory | `sgl-kernel`; aiter (triton/ck/asm) default on AMD | **first-class** (aiter default, triton fallback) |
| **TRT-LLM** | TRT plugins (build) + `Autotuner.choose_one` (runtime) | optimization-profile shape buckets; autotune cache `(runner,tactic,time)` | compiled TRT plugins / CUDA | **na** (NVIDIA-only) |
| **HF `kernels`** | nested dict `name→device→[Mode]→Repository`; `register_kernel_mapping`/`kernelize` | `Device`+`CUDAProperties(min/max_capability)`, smallest interval wins; `Mode` fallback chain | **versioned Hub repos** (`get_kernel`, prebuilt per torch/CUDA-ROCm), `Locked*` pins | **first-class** (`"rocm"` device key, same pattern) |
| **PyTorch/Inductor** | `torch.library.custom_op`; `register_lowering`/`register_*_pattern`/`register_custom_pass` | compiler-driven (fuse vs opaque); torch.compile mode | user C++/CUDA/HIP + Triton codegen | **yes** (Inductor+Triton on ROCm) |
| **cuDNN / cuBLASLt** | heuristic API returns ranked algo/engine list; index 0 default | `…AlgoGetHeuristic` (sorted by est. time) + preference filter; cuDNN modes A/B + SM-count; errata blocklist | dlopen engine libs; serializable `algo` cache | **na** (analogs: hipBLASLt, MIOpen, rocBLAS) |

---

## What we borrow

### (a) `perf_knowledge/index/sota_registry.yaml` schema (bump to **v3**)

The current v2 entry — `{operator, backend, status, gens, dtypes, regimes, card, sources}` — is a
*human catalog*. It lacks the three things every stack above has: a **selection key→winner**, a
**dispatch seam**, and an **autotune/errata** layer. Add, per card frontmatter (so `_gen_registry.py`
stays the single source of truth):

1. **A resolved `select` block** (the cuBLASLt/HF-kernels "ranked list, index 0 = default" idea).
   Make the registry answer *one* question — "best backend for (op, gen, dtype, regime)?" — by adding
   to the registry a generated `selection:` section keyed by `(operator, gen, regime, dtype_class)`
   whose value is `winner: <backend[/variant]>` + `alternates: [...]` ordered by status then perf.
   Mirror vLLM's variant-qualified values: allow `aiter/ck`, `aiter/asm`, `aiter/triton` (add a
   `variant:` field to cards), since "aiter" alone is ambiguous exactly as the RFC notes.
2. **A `dispatch:` / rebind-seam field per entry** — the *handle*, not prose. Borrow vLLM's seams
   directly: `{ env: VLLM_ROCM_USE_AITER_RMSNORM=1, flag: "--attention-backend triton",
   ir_op_priority: "rms_norm=aiter_ck", import_path: "aiter.ops.rmsnorm:rms_norm" }`. This is what
   makes the KB *consumable by code* — the agent can apply the change, not just read about it. (The
   sota cards already have an "integration (rebind seam)" section per `conventions.md`; promote it to
   structured frontmatter.)
3. **Specificity-ranked HW keys** (HF `kernels` rule): when two entries match, the **narrower `gens`
   set wins** (gfx950-specific beats all-CDNA). Encode this as a `specificity` the generator computes
   from `len(gens)`, used as the selection tie-break.
4. **An `autotune:` pointer** (cuBLASLt serialized-algo / TRT autotune cache / aiter tuned DB):
   `{ tuner: hipblaslt_offline | aiter_tuned_gemm | gradlib, cache: "aiter/configs/tuned_fmoe.csv",
   key: "M,N,K,dtype" }`. Tells the op_benchmarker *whether a tune exists and where the cache lives*
   before it re-tunes.
5. **A `torch_compile:` flag** (PyTorch/HF mode): `survives | prefer_fused | opaque` — whether the
   kernel should be used standalone or left to Inductor/vLLM-IR fusion. Prevents recommending a
   hand kernel that the compiler would beat by fusing.
6. **An `exclusions:` / errata list** (cuDNN errata JSON): per-entry known-bad `(gen, dtype, shape,
   rocm_ver)` tuples with a reason — so the agent never re-proposes a combo that already failed
   numerics or crashed. Version-scoped like cuDNN's `cudnn_version_start/_end`.

Net: v3 entry = v2 fields **+** `variant, dispatch{env,flag,ir_op_priority,import_path},
autotune{tuner,cache,key}, torch_compile, exclusions[], perf[]` — and a generator-built top-level
`selection:` map `(op,gen,dtype_class,regime) → {winner, alternates}`.

### (b) Wiring into `e2e_workflow` (architect + op_benchmarker query it)

- **`system_architect:strategize`** — when routing Top-N kernels into config/kernel/host tracks,
  query `selection[(op, detected_gen, dtype, regime)]`. If `winner.dispatch.env`/`.flag`/
  `.ir_op_priority` exists, route to the **config track first** (cheap lever, no source edit) — this
  *is* vLLM's `ir_op_priority` / `--aiter-config` story and SGLang's `--attention-backend` auto-select.
  Only route to the kernel-authoring track when no dispatch seam wins.
- **`op_benchmarker:bakeoff`** — "DISCOVER existing impls, tune cheap levers, DECIDE author_plan"
  maps 1:1 to the cuBLASLt/TRT flow: enumerate `alternates` for the cell as the **candidate set**;
  before tuning, check `autotune.cache` (skip re-tuning if a tuned DB already covers the shape);
  honor `exclusions` to prune known-bad candidates (cuDNN errata pattern); write the measured winner
  back as a `perf[]` entry (perf_knowledge perf-number format) so the registry *learns* — exactly TRT-LLM's
  "profile once, cache the winner" loop, persisted in the existing `backend_playbook.md`.
- **Feedback loop** — `architect:update_experience` already appends to `backend_playbook.md`; have it
  also emit a card-frontmatter patch (`perf[]`, `status`, new `exclusions`) and re-run
  `_gen_registry.py`, so the registry is the durable, machine-queryable mirror of what the team
  learned. This makes perf_knowledge the "ATOM incubation layer": validated wins get promoted into the SOTA
  table.
- **AMD-first keying** — detected `gfx` from preflight is the primary selection key; never treat AMD
  as a fallback (the mistake vLLM/SGLang explicitly corrected by making aiter first-class).

## Sources

- vLLM CustomOp design — https://docs.vllm.ai/en/latest/design/custom_op/
- vLLM aiter-config / ir_op_priority RFC #33163 — https://github.com/vllm-project/vllm/issues/33163
- vLLM unified attention auto-selection RFC #21805 — https://github.com/vllm-project/vllm/issues/21805
- vLLM ROCm attention blog (software-layer routing) — https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- vLLM ATOM out-of-tree plugin guide — https://rocm.github.io/ATOM/docs/vllm_plugin_backend_guide.html
- vLLM ATOM RFC #33478 — https://github.com/vllm-project/vllm/issues/33478
- SGLang attention_registry.py — https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
- SGLang attention backend docs — https://docs.sglang.ai/advanced_features/attention_backend.html
- AITER + SGLang integration (ROCm blog) — https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
- TensorRT-LLM docs — https://nvidia.github.io/TensorRT-LLM/
- TensorRT-LLM autotuner cache-key (issue #10679) — https://github.com/NVIDIA/TensorRT-LLM/issues/10679
- HF kernels — layers/mapping schema — https://huggingface.co/docs/kernels/en/layers
- HF Kernel Hub intro — https://huggingface.co/blog/hello-hf-kernels
- HF kernels-community — https://huggingface.co/kernels-community
- HF TGI multi-backend — https://huggingface.co/blog/tgi-multi-backend
- PyTorch custom operators (Inductor) — https://docs.pytorch.org/docs/stable/compile/programming_model.custom_ops.html
- PyTorch custom backends — https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html
- cuBLAS / cuBLASLt heuristics — https://docs.nvidia.com/cuda/cublas/
- cuDNN frontend heuristics & engine config — https://deepwiki.com/NVIDIA/cudnn-frontend/2.5-heuristics-and-engine-configuration
