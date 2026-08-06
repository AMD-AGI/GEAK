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
  last_verified: '2026-07-26'
  gpu: gfx942/MI300X
  model: Kimi-K2.6-int4-W4A16
  measured:
    isolated: '2.0-2.32x'
    e2e_pct: 46.3
    e2e_pct_equal_config: 77.0
    e2e_pct_best_full_stack: 143.2
    parity: pass
    note: 'BEST (2026-07-26, Kimi-K2.6 int4-W4A16, MI300X TP4): full-stack same-session warm A/B, TRUE bare baseline 393.3 -> 956.3 tok/s = 2.43x (+143.2%), parity pass, TPOT 158.9->64.7 ms (decode 2.45x), TTFT 4427->2673. Stack = cfg0 (VLLM_ROCM_USE_AITER=1 + VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 + aiter-MLA, which UPGRADES the decode path to a FULL cuda-graph; +2.68%) THEN the flydsl authored int4-W4A16 2-stage MoE (h0 alone ref 399.53->956.66 = +139.45%, iso 2.32x, non-overlapping, parity pass) THEN decode tile_n=64 refinement (+3.34%). KEY ENABLER: the decode-FULL cuda-graph (via aiter MLA) is what lets the flydsl decode gain fully surface; without it the win is much smaller. This does NOT conflict with the RUN5 numbers below — those are flydsl-marginal (+46.3% same-session) / equal-config (+77%); +143.2% is the whole accepted stack vs the bare baseline. RUN5 (07-20): flydsl-marginal +46.3% same-session (1.63x), +77% equal-config. Startup no-OOM re-verified 2026-07-23 at mem 0.9 / full 262144 (Available KV 25.25 GiB, capture 51/51+35/35, no hang). Older capped +32.6% was the pre-accumulate=True shim.'
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
- **Graph capture DEADLOCKS under TP>1 — server never becomes healthy (`e2e_delta=null`, hung on a capture
  batch, 0 forwards served)** — this is the #1 reason a kernel wins isolated yet scores nothing e2e. The
  shim JIT-compiles its FlyDSL exes lazily on first call, keyed by `(weight.data_ptr, stage, M)`. vLLM
  captures the decode path into CUDA graphs over ~86 distinct M-bucket descriptors (PIECEWISE + FULL); ANY
  `(layer, M)` not already compiled compiles **inside** the captured region → a blocking op under capture →
  TP ranks desync → c10d collective timeout, capture frozen (observed hung at PIECEWISE 2/51 and at FULL
  25/35 on Kimi-K2.6 TP4). The isolated unittest CANNOT catch this (single process, no capture) — only the
  e2e gate does, so a shim can pass isolated at high speedup and still be undeployable. Fix: **precompile
  EVERY capture M-bucket BEFORE capture** via the vendored `flydsl_capture_precompile.py` (Procedure step
  2b) — an exhaustive eager warmup that JITs all exes OUTSIDE the capture stream, on all TP ranks in
  lockstep. VERIFIED 2026-07-20 same-box controlled A/B: a 2.51x isolated shim WITHOUT the seam deadlocked
  (FULL 25/35, never healthy); the SAME shim WITH the seam precompiled 86/86 outside capture, captured
  51/51 + 35/35, reached `Application startup complete`, and served requests. The seam is env-gated
  (`VLLM_USE_FLYDSL_MOE=1`) so it only affects THIS candidate's server — zero impact on any other kernel.

The golden rule: **convert weights ONCE at load time; key all caches by the NEW weight `data_ptr()`, never
the activation pointer; and PRECOMPILE every capture M-bucket BEFORE capture (never rely on lazy
first-call JIT inside the captured region).** The runtime shim then only LAUNCHES (capture-safe: routing
`blocks = sorted_expert_ids.numel()` is a host shape read, aiter GPU sort + FlyDSL `cexe` are capturable,
all exes already compiled+cached by the step-2b warmup before capture). **The graph/inductor path needs NO
custom op.**

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
2b. **Precompile-before-capture seam — MANDATORY for TP>1 graph capture (skip it and the server
   deadlocks at capture; see Mechanism).** The vendored `flydsl_capture_precompile.py` (in THIS skill dir)
   wraps `GPUModelRunner.capture_model` to run an exhaustive eager warmup over every capture descriptor
   BEFORE the real capture, so all FlyDSL exes JIT+cache OUTSIDE the capture stream. Add ONE env-gated edit
   at the END of `vllm/v1/worker/gpu_model_runner.py` (same `FLYDSL_SHIM_DIR` as steps 1–2):
   ```python
   import os as _os
   if _os.environ.get("VLLM_USE_FLYDSL_MOE") == "1":
       import sys as _sys
       _d = _os.environ["FLYDSL_SHIM_DIR"]
       if _d not in _sys.path:
           _sys.path.insert(0, _d)
       import flydsl_capture_precompile          # vendored in this skill dir
       flydsl_capture_precompile.install_capture_precompile(_sys.modules[__name__])
   ```
   The seam is env-gated (`VLLM_USE_FLYDSL_MOE=1`) and self-disables otherwise, so it is scoped to THIS
   candidate's server ONLY and cannot affect any other kernel/candidate. It skips FULL-mode descriptors on
   purpose (the same M set is covered by PIECEWISE warmup; a raw FULL `_dummy_run` dirties vLLM's overlapped
   shared-experts buffer) and resets those buffers before capture — do not remove that logic.
3. **Launch (graph mode — NO `--enforce-eager`)**. Point `FLYDSL_SHIM_DIR` at THIS skill dir (the vendored
   shim) and `FLYDSL_ROOT` at a FlyDSL checkout whose kernels AND `build-fly` bindings are the SAME tree:
   ```bash
   export FLYDSL_SHIM_DIR=<this_skill_dir>          # contains the vendored flydsl_moe_shim.py
   export FLYDSL_ROOT=<flydsl_checkout>             # kernels + build-fly MUST be one tree (see pitfalls)
   export PYTHONPATH=$FLYDSL_ROOT:$FLYDSL_ROOT/build-fly/python_packages
   export VLLM_USE_FLYDSL_MOE=1
   export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 VLLM_ROCM_USE_AITER_RMSNORM=0
   # full config, NO --max-model-len cap (verified no-OOM 2026-07-23 at mem 0.9, full 262144):
   #   scale re-home + stage-2 accumulate=True make the convert memory-neutral, so the old
   #   `--max-model-len 32768` cap is NOT needed (see Knobs). Add it back only to squeeze concurrency.
   vllm serve <model> -tp 4 --gpu-memory-utilization 0.9 \
     --trust-remote-code --tool-call-parser kimi_k2 --enable-auto-tool-choice \
     --reasoning-parser kimi_k2 --no-enable-prefix-caching --mm-encoder-tp-mode data
   ```
4. **Verify startup (in order, no error)**: weight load → (silent in-place convert) → `Available KV cache`
   (no OOM) → `[flydsl-precompile] BEGIN exhaustive eager warmup ...` → `[flydsl-precompile] DONE eager
   warmup (N/N); all FlyDSL exes compiled+cached outside capture` (from step 2b — one per TP rank) →
   `Capturing CUDA graphs ... 100%` (must reach 100%, not hang) → `Application startup complete`. Smoke a
   completion; confirm **0** `shim failed` / `weights not converted` lines. If capture hangs (no `DONE
   eager warmup` line, or the capture bar freezes with repeated `No available shared memory broadcast block
   found in 60 seconds`), step 2b was not wired — do NOT report a number; fix the seam, do not reject.
5. **Gate**: same-session A/B baseline (Triton, `VLLM_USE_FLYDSL_MOE` unset) vs FlyDSL, BOTH graph, identical
   `vllm bench serve` (ISL/OSL/conc), plus GSM8K within noise. Quote same-session ratios only.
   - **GATE ON THE LIVE SERVING REGIME — pick the acceptance metric from the actual ISL/OSL; do NOT assume
     decode (regime-adaptive, verified 2026-08-04).** The isolated op-bench M-weighting MUST mirror the served
     shapes, and the same-session A/B must improve the metric the *dominant* regime is bound by:
     - **decode-dominated serving** (short ISL, e.g. 1024/1024/64) → primary signal is a **TPOT drop** (best
       run 07-26: 158.9 -> 64.7 ms, decode 2.45x). Here a candidate whose isolated bake-off is weighted toward
       large-M prefill (observed: M8192 = 72% of the metric) can be REAL 1.83x isolated yet REGRESS decode
       serving by **-15.95% e2e**, because its tuned tiles lose to vLLM's default at the live decode M-buckets
       → `rejected`.
     - **prefill-heavy serving** (long ISL, e.g. 8192/1024/64) → primary signal is a **TTFT drop and/or
       output-throughput gain** (TPOT is secondary here); the prefill M-buckets dominate e2e. Validated on
       ISL8192: 300.51 -> 398.35 tok/s (+32.6%) and the +79.95% ISL8192 run. Do NOT reject a real prefill win
       merely because TPOT did not move — that would be the wrong metric for this regime.
     - **balanced / unknown** → require a same-session throughput gain AND no regression in EITHER TTFT or TPOT.
     Universal rule: the isolated bake-off M-weighting must match the served ISL/OSL, and never stack an e2e
     regression. The shim already buckets tiles by M (prefill M>=512 swept / mid M>64 / decode M<=64 tile_n=64),
     so the decode tile_n=64 refinement is scoped to decode and never perturbs the prefill bucket.
   - **Sit the flydsl MoE on a decode-FULL cuda-graph config stack.** The big decode gain only surfaces when
     the decode path is a FULL cuda-graph. On Kimi-K2.6 that upgrade came from the aiter MLA backend (enabled
     by `VLLM_ROCM_USE_AITER=1`), which flips decode capture from PIECEWISE-only to FULL; pair with
     `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4`. The +143.2% best run = this cfg0 stack THEN the flydsl kernel.
   - **Note on ceiling vs AMD's published +162%:** AMD's blog (+162% / TPOT -69% / TTFT -65%, Kimi-K2.5, gfx942)
     came from an INSTRUCTION-LEVEL hand-authored FLIR kernel (hand-placed mfma sched / direct-to-LDS / LDS
     swizzle / software pipeline). This skill only LANDS a rewrite (apply mechanics); reproducing the full
     ceiling needs the sibling `flydsl_rewrite_quantized_moe` to author a kernel that wins DECODE, not just a
     per-M tile reuse. gfx942 caveat: async-copy/direct-to-LDS is gfx950-default (off on gfx942), LDS budget
     65536 B — so gfx942 needs arch-specific hand-tuning, not a stock-primitive swap.

## Knobs & pitfalls
- **Obtain & PIN FlyDSL — DELEGATE to the `ensure_flydsl` skill (the single source of truth for the
  build).** The shim needs both `flydsl` and `kernels.moe_gemm_2stage` importable from ONE consistent tree.
  Do NOT hand-roll clone/build steps here and do NOT hardcode a machine-specific checkout path — run the
  `ensure_flydsl` skill as step 0:
  ```bash
  bash perf_knowledge/expert_skills/skills/ensure_flydsl/ensure_flydsl.sh   # version-gated (>=MIN_VERSION reuse,
                                                           # else clone+build PIN into container-internal /opt/flydsl),
                                                           # flock-guarded, applies hip-cmake + patchelf fixes, writes flydsl_env.sh
  source "${FLYDSL_ROOT:-/opt/flydsl/FlyDSL}/flydsl_env.sh"
  python3 -c "import flydsl, kernels.moe_gemm_2stage as k; print(flydsl.__file__, k.__file__)"   # must resolve under ONE tree
  ```
  The `ensure_flydsl` skill reuses any ambient flydsl whose `__version__ >= MIN_VERSION` (never overwrites
  a newer one / never pip-installs system-wide), else builds the PIN into an isolated `/opt/flydsl` — safe
  on shared boxes. Only if it exits non-zero is flydsl genuinely unavailable.
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
  Residual ~3 GiB vs Triton is convert-time allocator fragmentation (does not block startup).
- **Candidate memory acceptance (MANDATORY before integrate — scale re-home is necessary but NOT
  sufficient):** the candidate kernel's stage-2 output MUST be `[M,hidden]` (gemm2 `accumulate=True` /
  `compile_moe_gemm2_ex(mode=REDUCE)`, top-k moe_sum folded into the in-kernel accumulate). A candidate
  that ships the expanded `[M·top_k,hidden]` output (`accumulate=False` + host moe_sum + A2 pre-gather)
  allocates ~940 MiB at M=8192 (top_k=8, hidden=7168) as a **prefill-warmup transient** — even with scales
  re-homed it does NOT fit at mem-frac 0.95 and the server never becomes healthy (observed 2026-07-16:
  +896 MiB requested / 626 MiB free → e2e OOM REJECT, 0 tok/s, despite 1.85× isolated + parity PASS). Do
  NOT accept an isolated-only win; reject `mem_footprint_starves_kv` unless the stage-2 output is `[M,hidden]`.
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
- Vendored implementation (this dir — self-contained, no external files needed):
  - `flydsl_moe_shim.py` — the exact validated shim (`convert_layer_inplace` load-time in-place convert +
    `flydsl_fused_experts_impl` launch-only runtime).
  - `flydsl_capture_precompile.py` — the capture-precompile seam (`install_capture_precompile`) that wraps
    `GPUModelRunner.capture_model` with the pre-capture eager warmup (Procedure step 2b).
  This skill is self-contained: the three env-gated vLLM edits above (steps 1, 2, 2b) + these two vendored
  files + a FlyDSL checkout are all that is needed; no external eval-dir, learned card, or other skill dir
  is required to reproduce.
- Measured (2026-06-25, vLLM 0.19.0, Kimi-K2.6 int4-W4A16, MI300X TP4): same-session GRAPH A/B,
  ISL8192/OSL1024/conc64/192prompts — baseline (Triton) **300.51 tok/s** → FlyDSL **398.35 tok/s**
  (**+32.6%**), decode TPOT 168.6→61.3 ms (2.75x), 0 fallbacks, smoke coherent. Independently reproduced
  earlier at +32–34% (392 vs 297 tok/s) with GSM8K parity 0.915 and graph-replay cosine 1.0.
- Capture-safety verified (2026-07-20, vLLM 0.21.0, Kimi-K2.6 int4-W4A16, MI300X TP4): controlled same-box
  A/B on one 2.51x-isolated shim — WITHOUT step 2b the server deadlocked at CUDA-graph capture (FULL 25/35,
  `Application startup complete` never reached); WITH step 2b it precompiled 86/86 descriptors outside
  capture, captured 51/51 PIECEWISE + 35/35 FULL, reached `Application startup complete`, and served. Step
  2b is the difference between "isolated win, undeployable" and "healthy server".
- Sibling kernel-scope skill (produces the rewritten kernel this skill lands): `flydsl_rewrite_quantized_moe`.
