# Head-Kernel Playbook — GEMM & Attention Backend Bake-off + Tuning (persistent)

The **Op Benchmarker** owns this file. It is the experience library for the *highest-pct_gpu_time*
kernels — dense GEMM and attention — which are usually **library calls** (hipBLASLt / CK) and were
historically skipped by the kernel squad because they are not source-editable. They are NOT
un-optimizable: a fixed-shape library GEMM is one of the most tunable things on the chip. Optimize it
by changing **which implementation runs**, **how that implementation is tuned**, and (for editable
backends) **the kernel code itself** — cheapest first.

> Why this exists: e2e is Amdahl-dominated. A GEMM at ~78% of GPU time only needs **1.15x** to give
> ~+10% e2e. That is a far better bet than a 1.3x on a 2% kernel. Spend budget on the head first.

> **On sglang/gfx942, dense-GEMM Tier-B = aiter's per-shape DB; Tier-C = an authored Triton GEMM.**
> The live dense-GEMM path is aiter `tuned_gemm.py` (seam `aiter.tuned_gemm:gemm_a16w16`). Tune it via
> `AITER_TUNE_GEMM=1` capture → `gradlib/gemm_tuner.py` → deploy `AITER_CONFIG_GEMM_BF16`, and verify
> engagement with `AITER_LOG_TUNED_CONFIG=1`. Full recipe: **`gemm_tuning/aiter_gemm_tuning.md`**. (TunableOp /
> `HIPBLASLT_TUNING_FILE` hook the PyTorch dispatch, which this live path does not use — so they are not
> the GEMM lever here; tune aiter / author Triton instead.)

## The ladder (cheapest-first; each rung gated by the immutable oracle + e2e Amdahl + parity)

| Tier | what changes | source edit? | parity | mechanism |
|---|---|---|---|---|
| **A** backend select | which impl computes the op | no | safe* | run each backend on the op unittest, pick fastest-correct |
| **B** per-backend tune | autotune *within* the chosen backend | no | safe* | hipBLASLt solution sweep / TunableOp / CK instance / Triton autotune |
| **C** code rewrite | the kernel source | **yes** (triton/hip/ck only) | safe* | recurse `kernel_workflow` on the op task dir |
| **D** quantization | dtype of the op | no (flag) | **breaks** | `--quantization fp8`, kv-cache fp8 → **accuracy gate**, not byte parity |

`*` "safe" = same math/dtype, so *expected* near-identical — but NOT guaranteed byte-identical
(bf16 reduction order differs across backends and can flip a borderline argmax). **Always re-check
e2e parity**; if it fails on a non-quant change, flag it for an accuracy eval (see below).

---

## Backend menu — DENSE GEMM (C = A·Bᵀ [+bias] [+act], MI300X / gfx942)

- **hipBLASLt / Tensile** — sglang default. Strong *when the shape is in its tuning DB*; otherwise
  falls back to a generic solution (watch `not found tuned config ... using default config`).
- **rocBLAS** — alternate library; sometimes wins skinny/odd shapes hipBLASLt mis-tunes.
- **PyTorch TunableOp** — runtime auto-tuner that benchmarks rocBLAS+hipBLASLt candidates per shape and
  caches the winner to a CSV. Pure env, parity-safe, the **easiest first move**.
- **aiter GEMM** — AMD fused GEMM (+epilogue). Often wins decode/skinny + fuses bias/act; lost to
  default hipBLASLt on the Qwen3.5-27B prefill shapes in the 2026-06-04 run (see Learned).
- **CK / ck_tile GEMM** — Composable Kernel; tunable by instance (block sizes, pipeline, MFMA).
- **Triton matmul** — editable; autotunable; the path to Tier C code rewrites (split-K, persistent,
  epilogue fusion). Worth it when fusion (bias+act, or GEMM+norm) collapses a neighbor kernel.
- **FlyDSL GEMM** — aiter's Python kernel DSL with instruction-level control (`flydsl_hgemm` bf16/fp16;
  `flydsl_preshuffle_gemm_a8` for fp8/A4W4). **The SOTA author backend for dense/quantized GEMM on
  gfx942/950** (AMD's choice for Kimi-K2.5 fused-MoE: vendor-reported up to +162% throughput). TWO levers:
  (1) **env** — FlyDSL is a backend aiter's per-shape DB tune races (`libtype=flydsl`); a normal
  `AITER_TUNE_GEMM` capture → gradlib → `AITER_CONFIG_GEMM_BF16` deploy auto-selects it where it wins,
  no extra code, verify with `AITER_LOG_TUNED_CONFIG=1` (`libtype is flydsl`). (2) **Tier-C author** —
  `target_language=flydsl`; baseline reuses the vendor DSL's existing GEMM primitive (don't hand-write
  layout/MFMA), optimize loop tunes the knobs. JIT (no build); requires `is_flydsl_available()`.
  *Authoring principle for a block-scaled quant GEMM:* collapse the per-block-scale K-loop into ONE
  full-K fused call — fold the block-scale into a per-channel operand scale, cache the static-weight
  requant+preshuffle once (it's a pure function of the weights), requant activations per-token. In this
  regime the bound is often **kernel-dispatches per call**, not raw FLOPs, so minimizing dispatch count
  is the win. SOTA card: `perf_knowledge/operators/dense_gemm/backends/flydsl.md`.

> **Don't stop at the first authored kernel — the win is in the optimize loop.** A from-scratch DSL
> author baseline is typically **~parity** with the existing tuned library kernel (it hasn't reached the
> hardware's low-precision cores yet; a naive version often re-materializes a dequant/requant pass and is
> memory-bound, even marginally *slower*). The whole speedup comes from optimizing it. For a
> (block-)quantized GEMM the optimization chain, in order, is:
> 1. **Kill the memory-bound dequant/requant materialization** — fold the scales and emit ONE fused
>    low-precision MFMA GEMM. This exposes the actual compute core (the big jump).
> 2. **Re-profile.** The new bottleneck is usually the **per-token requant prologue** → fuse it
>    (e.g. torch.compile), gated to large-M where its host floor pays off (decode prologue is cheap).
> 3. **Per-shape tile/pipeline tuning** of the fused core (tile_n, cshuffle, lds, splitK…).
> 4. **Integrate** the orthogonal wins into one kernel.
> Re-profile after EACH lever — the bottleneck shifts every time, and the next lever is whatever just
> became dominant. Report the single-kernel result as **library-baseline / first-author / optimized**
> (re-measure the author baseline, don't assume it equals the library) so the gain is attributed honestly.
>
> **The decode regime is the e2e-critical one — optimize for it, do not regress it.** Steady-state
> serving throughput is decode/TPOT-bound (skinny-M GEMM at M ≈ running batch ≈ conc), even though
> prefill GEMMs rank higher by GPU-time. The unittest MUST contain a decode-M case (the extractor
> enforces this); your isolated geomean must win — or at minimum not regress — that case, or the e2e
> gate will reject the kernel even at a large prefill speedup (observed: prefill-only iso 1.39× → e2e
> −9%). Decode-killers to avoid (each tanks small-M while looking fine on prefill):
> - **Per-call weight transpose / requant / preshuffle materialization.** Weights are static — do the
>   `w.T` / `w_scale.T` / fp8-requant / preshuffle ONCE and cache it (keyed by weight id; cache size ≥
>   num_layers or it thrashes). A per-call full-weight transpose dwarfs the tiny decode GEMM.
> - **Tall `BLOCK_M` tiles** (e.g. 256) on M ≈ 64 — mostly wasted lanes; use small/auto BLOCK_M and
>   raise `split_k` for decode.
> - **JIT dispatch / recompile per call** on the decode path — precompile / make the kernel M-agnostic
>   so it's captured once into the decode CUDA graph (eager re-dispatch per step destroys TPOT).
>
> **CUDA-graph capture safety — THE #1 e2e-integration killer (a kernel can win isolated yet never run
> live).** sglang captures the decode path into a CUDA graph. Two things HANG capture (server never
> becomes healthy, 0 forwards, gate rejects for no-engagement — observed run #4/#5: isolated 2.7× but
> 0 live forwards):
> - **Any host sync in the hot path** — `.item()`, `.cpu()`, `.tolist()`, `.sum().item()`,
>   `torch.cuda.synchronize()`, a Python `if` on a GPU scalar. The classic offender is a per-call
>   **weight-fingerprint** (`w_scale.sum().item()`) used to key a weight cache. Inside graph capture a
>   host sync deadlocks. FIX: key the weight cache by **`weight.data_ptr()`** (a pure host int, weights
>   are persistent) and do all fingerprint/prep work ONCE at warmup, never per call. Keep a sync-free
>   fast path for the live decode call.
> - **JIT/compile or dynamic allocation during capture.** Compile + register the kernel for ALL decode
>   M-buckets at WARMUP, before capture (an `flydsl_overlay_precompile(weight, weight_scale, m_buckets)`
>   hook the integrator calls once, pre-capture). The unittest's loose-tol fp32 oracle won't catch a
>   capture hang — only the e2e gate does, and only if the seam is capture-safe.
> Author the kernel so its STEADY-STATE call (the second call onward, post-warmup) does zero host syncs
> and zero compiles. See e2e_integrator.md "CUDA-graph-safe overlay" for the seam side.
>
> **Persistent memory footprint is a HARD integration constraint (a fast kernel can still lose e2e by
> starving the KV pool).** The weight-prep cache is held for ALL layers simultaneously (cache size ≥
> num_layers, else it thrashes → −55%). If the kernel re-materializes weights as **bf16** (raw +
> preshuffled, every layer) the cache balloons to tens of GB (observed: 92.6GB across 64 layers × 4
> projections → forced mem-fraction-static to 0.45 → KV-cache starved → usable e2e −9% even though the
> GEMM was +24% at equal memory). At fixed concurrency the KV budget is a dominant throughput lever, so
> the integrated kernel MUST fit at the SAME mem-fraction the accepted config uses. Two ways to stay
> small: (1) **fused fp8** — kill the bf16 dequant entirely (the optimize-loop step 1) and cache only
> COMPACT fp8/preshuffled weights (~the model's own fp8 size); this is faster AND smaller. (2) If a seam
> serves many (N,K) (e.g. fp8_utils blockscale GEMM = up/gate + down + qkv + o), route ONLY the tuned
> target (N,K) to the authored kernel and pass other shapes through to stock — bounds the cache to one
> projection family and avoids regressing untuned shapes. Prefer (1); use (2) to bound footprint further.

### Tier-B tuning knobs per GEMM backend
- **hipBLASLt**: enumerate solution indices for the exact (M,N,K,dtype,transpose,bias) and pin the
  best via `HIPBLASLT_TUNING_FILE=<file>` (offline `hipblaslt-bench`, or the hipBLASLt ext-op API).
  Also `TENSILE_*` / rocBLAS `ROCBLAS_TENSILE_*`.
- **PyTorch TunableOp**: `PYTORCH_TUNABLEOP_ENABLED=1` `PYTORCH_TUNABLEOP_TUNING=1`
  `PYTORCH_TUNABLEOP_FILENAME=<csv>` → run a warmup pass to populate, then ship with `TUNING=0`.
- **Triton matmul**: autotune over `BLOCK_M/BLOCK_N/BLOCK_K`, `GROUP_M`, `num_warps∈{4,8}`,
  `num_stages∈{1,2}`, `matrix_instr_nonkdim∈{16,32}` (MFMA), `waves_per_eu`, `kpack`; `SPLIT_K` for
  small-M decode. Bake the winning config dict into the kernel.
- **CK**: pick the instance/config (tile, pipeline v1/v2, padded vs not).
- **FlyDSL**: per-shape knobs `tile_m/tile_n/tile_k`, `split_k` (raise for decode small-M),
  `block_m_warps/block_n_warps`, `b_preshuffle` (+ `shuffle_weight` once, for large-N prefill),
  `b_to_lds`, `waves_per_eu`, `stages`, `async_copy`. aiter's tuned DB picks these per shape; for an
  authored kernel the optimize loop sweeps them. fp32 accumulate; fp8 via `flydsl_preshuffle_gemm_a8`.

---

## Backend menu — ATTENTION (prefill paged + decode paged)

- **CK / ck_tile** (`FmhaBatchPrefillWithPagedKVCache`) — strong paged attention on MI300X; tunable by
  instance. Library (Tier A/B/D only).
- **aiter attention** (`aiter_attn`) — the sglang ROCm default in this image. Library.
- **Triton FA** (`--attention-backend triton`) — editable; autotunable; gave **+5.2% e2e** on the
  hybrid Qwen3.5-27B (but NOT byte-identical — see Learned). Tier A/B/C.
- **fa3 / flashinfer-mla** — version/arch dependent.

### Tier-B tuning knobs for attention
- Server flag swap: `--attention-backend {triton,aiter,ck,fa3}`, and the split
  `--prefill-attention-backend` / `--decode-attention-backend` (version-dependent).
- `--page-size`, cuda/HIP-graph batch sizes (decode launch overhead).
- Triton FA: autotune `BLOCK_M/BLOCK_N`, `num_warps`, `num_stages`, `waves_per_eu`.
- Tier D: `--kv-cache-dtype fp8_e4m3` (memory + bandwidth; accuracy gate).

---

## Class → ranked plan (priors; the op unittest is the judge)

| op | regime | Tier A order | Tier B | Tier C (if editable wins) | Tier D |
|---|---|---|---|---|---|
| dense GEMM | prefill (large M) | aiter(DB, races hipBLASLt/asm/**flydsl**) → hipBLASLt → CK → flydsl | aiter per-shape DB tune | **flydsl** author → Triton author (epilogue fuse / split-K) | fp8 |
| dense GEMM | decode (M=batch) | aiter(DB, +flydsl split-K) → hipBLASLt → flydsl | aiter per-shape DB tune | **flydsl**/Triton split-K author | fp8 |
| dense GEMM (fp8 a8w8 blockscale, sglang) | prefill/decode | **CK skill `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md` (MANDATED this eval)** — capture (M,N,K) → aiter CK tuner → fp8_utils Triton→CK switch + `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` | CK tuner (`gemm_a8w8_blockscale_tune.py --libtype both --mp <all GPUs>`) | — (Triton-overlay / aiter-bf16 levers FORBIDDEN here; they bypass the skill) | — (already fp8) |
| paged attention | prefill | CK → triton → aiter | instance / FA autotune | triton FA rewrite | kv fp8 |
| paged attention | decode | aiter → triton → CK | page-size / FA autotune | triton FA rewrite | kv fp8 |

## How the Op Benchmarker uses this
1. Read the op task dir (`op_kind`, shapes, dtype, `math_contract`) + this file's ranked plan.
2. **Tier A**: run `scripts/op_bench.py` to bench every available backend against the immutable oracle;
   keep only correct ones; record ms + speedup.
3. **Tier B**: autotune each *promising* backend (cap the search budget); re-bench.
4. **Tier C**: if the best correct backend is triton/hip/ck, hand the op task dir to the recursive
   `kernel_workflow` for code-level optimization (it already enforces the immutable unittest).
5. Emit the winner = (backend, winner_kind ∈ {env, flag, patch}, tuning_artifact|code_patch,
   isolated_speedup). The e2e Integrator turns that into an overlay/config and runs the Amdahl gate.
6. **CURATE** `knowledge/learned/` after the run (read INDEX → merge/insert ≥★★ / archive
   contradicted), per `knowledge/learned/README.md`. Do NOT append run narratives to this file.

## Parity / accuracy gate (read before accepting a head-kernel win)
- Same-dtype backend swap or tuning → expect near-identical, but **verify e2e greedy/temp=0 parity**.
  If it diverges (real cross-backend bf16 argmax flip), do NOT auto-accept on throughput alone:
  run a small task-accuracy probe (e.g. gsm8k / a translation set) and accept only if quality holds.
- Any quantization (Tier D) → byte parity is expected to fail by design → **always** the accuracy gate.

## Learned experience → `knowledge/learned/`
Per-head/per-shape optimization findings are NOT appended here anymore. They live as distilled,
evidence-cited cards in **`knowledge/learned/`**, read via **`knowledge/learned/INDEX.md`** (grouped by
reuse key `kernel_class · gfx`). Open only the cards matching the current run's `(kernel_class, gfx,
regime)`; rank bets by `EV = Amdahl_ceiling × confidence`; honor each card's `dead-end:` lines.
