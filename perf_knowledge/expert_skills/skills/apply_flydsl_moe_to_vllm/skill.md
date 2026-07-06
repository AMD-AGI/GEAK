---
id: apply_flydsl_moe_to_vllm
title: Apply a FlyDSL int4-W4A16 fused-MoE rewrite back into live vLLM e2e (no OOM, keep the speedup)
kind: expert_skill
authors:
- geak
scope: e2e
match:
  operator: fused_moe_grouped_gemm
  arch_class:
  - '*'
  gens:
  - gfx942
  dtypes:
  - int4_w4a16
  regimes:
  - prefill
  - decode
  from_backend: triton
  to_backend: flydsl
  profile_signature:
    op_name_regex: fused_moe_kernel_gptq_awq|invoke_fused_moe_wna16
    min_pct_gpu: 10.0
expects:
  e2e_delta_min_pct: 1.0
  parity: required
validation:
  status: validated
  last_verified: '2026-06-25'
  gpu: gfx942/MI300X
  model: Kimi-K2.6-int4-W4A16
  measured:
    isolated: ''
    e2e_pct: 32.6
    parity: pass
  artifact: perf_knowledge/expert_skills/skills/apply_flydsl_moe_to_vllm
role: advisory_prior
supersedes: []
---

## When to use
A validated FlyDSL rewrite of an **int4 W4A16 (GPTQ/AWQ, no-zp) fused-MoE grouped GEMM** exists (see the
sibling kernel skill `flydsl_rewrite_quantized_moe`, isolated ~3.6x) and must be landed **end-to-end in a
live vLLM server** on MI300X (gfx942) TP4 — i.e. the isolated win has to survive serving. Use when the
apply hits **HIP OOM / `Engine core initialization failed` during `determine_available_memory`**, or when
an A/B shows the new kernel slower than Triton e2e.

## Mechanism
A FlyDSL grouped-GEMM that is fast in isolation fails its first e2e apply in two ways, each with a fix:
- **HIP OOM at startup** — the converted weights/scales are kept *beside* the originals, or re-materialized
  *per forward* (`[E,N,K]` int16 unpack), doubling/spiking MoE memory. Caused by a runtime-path conversion
  whose cache key includes the **activation `data_ptr()`** (reallocated every step → cache misses forever →
  re-converts every forward). Fix: convert **once at load time, in place, same-byte**, keyed by the **new
  weight `data_ptr()` only**. **GOTCHA (verified 2026-06-26, the dominant real hog): re-home BOTH the
  weight AND the scale param** — `layer.w*_weight_packed.data = w_flat` AND `layer.w*_weight_scale.data =
  s_flat`. Re-homing only the weight (a common miss) leaves the original `[E,N,G]` bf16 scale alive on the
  layer while the converted scale is also cached → **scales DUPLICATED ≈ +246 MiB/layer × N layers ≈
  +14.5 GiB** → Available KV collapses (e.g. 5.96 GiB vs 17.16 needed for 262144) → `Engine core
  initialization failed`. This — NOT the `repeat_interleave` A2 pre-gather — was the binding KV constraint
  on Kimi-K2.6 (measured: repeat_interleave cost only ~0.23 GiB; the scale leak cost ~14.5 GiB).
- **Eager A/B is slower (−22…−34%) despite isolated 3.6x** — the shim issues more launches/layer than
  Triton's single fused op; eager per-launch host latency dominates. Fix: run with HIP graphs (drop
  `--enforce-eager`); capture replays the sequence at zero launch cost and the faster GEMMs surface.

The golden rule: **convert weights ONCE at load time; key all caches by the NEW weight `data_ptr()`, never
the activation pointer.** The runtime shim then only LAUNCHES (capture-safe: routing `blocks =
sorted_expert_ids.numel()` is a host shape read, aiter GPU sort + FlyDSL `cexe` are capturable, exes
JIT-compiled at warmup before capture). **The graph/inductor path needs NO custom op.**

## Procedure
The integration shim is **vendored in this skill dir**: `flydsl_moe_shim.py` (functions
`convert_layer_inplace`, `flydsl_fused_experts_impl`). Apply it via two **env-gated** edits to the
installed vLLM (off by default ⇒ byte-identical stock; back up the two files first so revert = restore).

1. **Load-time conversion** — `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
   → `CompressedTensorsWNA16MoEMethod.process_weights_after_loading`, AFTER the existing repack to
   `[E,N,K//2]` uint8 + scale transposes, append:
   ```python
   import os as _os
   if _os.environ.get("VLLM_USE_FLYDSL_MOE") == "1":
       import sys as _sys
       _d = _os.environ["FLYDSL_SHIM_DIR"]
       if _d not in _sys.path:
           _sys.path.insert(0, _d)
       import flydsl_moe_shim
       flydsl_moe_shim.convert_layer_inplace(layer)   # in-place same-byte; frees originals
   ```
2. **Runtime route** — `vllm/model_executor/layers/fused_moe/fused_moe.py` → TOP of `fused_experts_impl`
   (before the constraint checks), insert:
   ```python
   import os as _os
   if (_os.environ.get("VLLM_USE_FLYDSL_MOE") == "1"
           and use_int4_w4a16 and w1_zp is None and w2_zp is None):
       import sys as _sys
       _d = _os.environ["FLYDSL_SHIM_DIR"]
       if _d not in _sys.path:
           _sys.path.insert(0, _d)
       import flydsl_moe_shim
       return flydsl_moe_shim.flydsl_fused_experts_impl(
           hidden_states, w1, w2, topk_weights, topk_ids, inplace,
           activation=activation, apply_router_weight_on_input=apply_router_weight_on_input,
           global_num_experts=global_num_experts, expert_map=expert_map,
           w1_scale=w1_scale, w2_scale=w2_scale)   # shim absorbs extras via **_ignored
   ```
3. **Launch (graph mode — NO `--enforce-eager`)**. Point `FLYDSL_SHIM_DIR` at THIS skill dir (the vendored
   shim) and `FLYDSL_ROOT` at a FlyDSL checkout whose kernels AND `build-fly` bindings are the SAME tree:
   ```bash
   export FLYDSL_SHIM_DIR=<this_skill_dir>          # contains the vendored flydsl_moe_shim.py
   export FLYDSL_ROOT=<flydsl_checkout>             # kernels + build-fly MUST be one tree (see pitfalls)
   export PYTHONPATH=$FLYDSL_ROOT:$FLYDSL_ROOT/build-fly/python_packages
   export VLLM_USE_FLYDSL_MOE=1
   export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 VLLM_ROCM_USE_AITER_RMSNORM=0
   vllm serve <model> -tp 4 --gpu-memory-utilization 0.95 --max-model-len 32768 \
     --trust-remote-code --tool-call-parser kimi_k2 --enable-auto-tool-choice \
     --reasoning-parser kimi_k2 --no-enable-prefix-caching --mm-encoder-tp-mode data
   ```
4. **Verify startup (in order, no error)**: weight load → (silent in-place convert) → `Available KV cache`
   (no OOM) → `torch.compile took ...` → `Capturing CUDA graphs ... finished` → `Application startup
   complete`. Smoke a completion; confirm **0** `shim failed` / `weights not converted` lines.
5. **Gate**: same-session A/B baseline (Triton, `VLLM_USE_FLYDSL_MOE` unset) vs FlyDSL, BOTH graph, identical
   `vllm bench serve` (ISL/OSL/conc), plus GSM8K within noise. Quote same-session ratios only.

## Knobs & pitfalls
- **Obtain & PIN FlyDSL — this skill OWNS the build via its bundled `ensure_flydsl.sh` (single source of
  truth).** The shim needs both `flydsl` and `kernels.moe_gemm_2stage` importable from ONE consistent tree.
  Do NOT hand-roll clone/build steps here and do NOT hardcode a machine-specific checkout path — just run
  the skill's own script as step 0:
  ```bash
  bash "$FLYDSL_SHIM_DIR/ensure_flydsl.sh"                 # this skill's dir; version-gated (>=MIN_VERSION reuse,
                                                           # else clone+build PIN into container-internal /opt/flydsl),
                                                           # flock-guarded, applies hip-cmake + patchelf fixes, writes flydsl_env.sh
  source "${FLYDSL_ROOT:-/opt/flydsl/FlyDSL}/flydsl_env.sh"
  python3 -c "import flydsl, kernels.moe_gemm_2stage as k; print(flydsl.__file__, k.__file__)"   # must resolve under ONE tree
  ```
  `ensure_flydsl.sh` reuses any ambient flydsl whose `__version__ >= MIN_VERSION` (never overwrites a newer
  one / never pip-installs system-wide), else builds the PIN into an isolated `/opt/flydsl` — so it is safe
  to run on shared boxes. Only if it exits non-zero is flydsl genuinely unavailable.
  **Pin pitfall:** a stale `FLYDSL_ROOT` from a DIFFERENT/older tree hijacking the bindings while kernels load
  from another fails kernel compile with `Dynamic int_tuple leaf must be an i32 or i64 value, got: <unknown
  type>` (eager) / `... got: gl$v` (graph) — a **version mismatch, NOT a torch.compile incompatibility** (do not
  wrap the MoE as a custom op). Always confirm
  `python3 -c "import flydsl, kernels.moe_gemm_2stage as k; print(flydsl.__file__, k.__file__)"` resolve under
  the SAME tree before launch.
- **KV headroom (CORRECTED 2026-06-26)**: in-place convert is memory-neutral ONLY IF both weight and
  scale params are re-homed (see the Mechanism GOTCHA). Once the scale-duplication leak is fixed, FlyDSL
  starts and serves at the FULL fair config — `mem 0.9`, NO `--max-model-len` cap (262144) — with
  **Available KV 20.13 GiB / 307,566 tokens** (vs Triton 23.1 GiB; was 5.96 GiB → OOM before the fix),
  and beats Triton **+77% e2e at equal config** (257→455 tok/s, cc64; TTFT −45%, TPOT −44%, cosine
  0.99998). The runtime `repeat_interleave` A2 pre-gather is a SECONDARY cost (~0.23 GiB; removing it via
  the stage-1 `compile_moe_gemm1` compact-input/sorted-row in-kernel gather is a clean optional follow-up,
  not what unblocks startup). Capping `--max-model-len 32768` is now only a workaround for the
  *un-fixed* shim or for squeezing extra concurrency; it is NOT required once scales are re-homed.
  Residual ~3 GiB vs Triton is convert-time allocator fragmentation (does not block startup); reuse
  scratch + gemm2 `accumulate=True` (stage-2 out `[M,hidden]` not `[M*top_k,hidden]`) to recover it.
- **Scales bf16 end-to-end** (`scale_is_bf16=True`, packed `(E,G//2,N,2)`); wrong layout → cosine ≈ 0.48.
- In-place convert is **destructive to the Triton layout** → no Triton fallback once converted; gate
  correctness offline + GSM8K and verify **0 fallbacks**.
- **Offline diag must use TP-sharded per-GPU dims, not full dims.** `flyc.compile` on a flat weight with
  >2³¹ elements overflows the shape codec → `struct.error: 'i' format`. Live TP4 is fine (inter-dim is
  sharded so each rank fits); only standalone validators hit this if they build the FULL (unsharded)
  tensor. Mirror the live per-rank shape (N=moe_intermediate//TP) in any offline numeric/graph-capture check.

## Do-no-harm notes
- Keep everything env-gated (`VLLM_USE_FLYDSL_MOE=1`); off ⇒ byte-identical stock, so the patched install
  never regresses other runs. Revert = restore the two files (keep `.flydsl_bak` backups).
- **Never measure/deploy in eager** — eager hides the speedup behind launch latency and reads as a
  regression. Graphs are required for the win.
- Compare against a STOCK baseline at the SAME serving config; do not stack on an already-patched stack.

## Sources
- Vendored implementation (this dir): `flydsl_moe_shim.py` — the exact validated shim
  (`convert_layer_inplace` load-time in-place convert + `flydsl_fused_experts_impl` launch-only runtime).
  This skill is self-contained: the two vLLM edits above + this shim + a FlyDSL checkout are all that's
  needed; no external eval-dir is required to reproduce.
- Measured (2026-06-25, vLLM 0.19.0, Kimi-K2.6 int4-W4A16, MI300X TP4): same-session GRAPH A/B,
  ISL8192/OSL1024/conc64/192prompts — baseline (Triton) **300.51 tok/s** → FlyDSL **398.35 tok/s**
  (**+32.6%**), decode TPOT 168.6→61.3 ms (2.75x), 0 fallbacks, smoke coherent. Independently reproduced
  earlier at +32–34% (392 vs 297 tok/s) with GSM8K parity 0.915 and graph-replay cosine 1.0.
- Sibling kernel-scope skill (produces the rewritten kernel this skill lands): `flydsl_rewrite_quantized_moe`.
- Companion learned card: `e2e_workflow/knowledge/learned/flydsl-moe-applyback-gfx942.md` +
  method `e2e_workflow/knowledge/learned/method-cudagraph-safe-integration.md`.
