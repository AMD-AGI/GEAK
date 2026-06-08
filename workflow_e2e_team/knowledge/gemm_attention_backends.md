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
> engagement with `AITER_LOG_TUNED_CONFIG=1`. Full recipe: **`aiter_gemm_tuning.md`**. (TunableOp /
> `HIPBLASLT_TUNING_FILE` hook the PyTorch dispatch, which this live path does not use — so they are not
> the GEMM lever here; tune aiter / author Triton instead.)

## The ladder (cheapest-first; each rung gated by the immutable oracle + e2e Amdahl + parity)

| Tier | what changes | source edit? | parity | mechanism |
|---|---|---|---|---|
| **A** backend select | which impl computes the op | no | safe* | run each backend on the op unittest, pick fastest-correct |
| **B** per-backend tune | autotune *within* the chosen backend | no | safe* | hipBLASLt solution sweep / TunableOp / CK instance / Triton autotune |
| **C** code rewrite | the kernel source | **yes** (triton/hip/ck only) | safe* | recurse `team_workflow` on the op task dir |
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
| dense GEMM | prefill (large M) | TunableOp → hipBLASLt(tuned DB) → CK → Triton → aiter | per-backend autotune | Triton: epilogue fuse / split-K / persistent | fp8 |
| dense GEMM | decode (M=batch) | aiter → TunableOp → hipBLASLt → Triton(split-K) | per-backend autotune | Triton split-K | fp8 |
| paged attention | prefill | CK → triton → aiter | instance / FA autotune | triton FA rewrite | kv fp8 |
| paged attention | decode | aiter → triton → CK | page-size / FA autotune | triton FA rewrite | kv fp8 |

## How the Op Benchmarker uses this
1. Read the op task dir (`op_kind`, shapes, dtype, `math_contract`) + this file's ranked plan.
2. **Tier A**: run `scripts/op_bench.py` to bench every available backend against the immutable oracle;
   keep only correct ones; record ms + speedup.
3. **Tier B**: autotune each *promising* backend (cap the search budget); re-bench.
4. **Tier C**: if the best correct backend is triton/hip/ck, hand the op task dir to the recursive
   `team_workflow` for code-level optimization (it already enforces the immutable unittest).
5. Emit the winner = (backend, winner_kind ∈ {env, flag, patch}, tuning_artifact|code_patch,
   isolated_speedup). The e2e Integrator turns that into an overlay/config and runs the Amdahl gate.
6. **Record results in Learned below** (newest first), with model, shape, dtype, gfx, measured ms.

## Parity / accuracy gate (read before accepting a head-kernel win)
- Same-dtype backend swap or tuning → expect near-identical, but **verify e2e greedy/temp=0 parity**.
  If it diverges (real cross-backend bf16 argmax flip), do NOT auto-accept on throughput alone:
  run a small task-accuracy probe (e.g. gsm8k / a translation set) and accept only if quality holds.
- Any quantization (Tier D) → byte parity is expected to fail by design → **always** the accuracy gate.

## Learned — POSITIVE methods & routing only (append after each run; newest first)
<!-- format: - [YYYY-MM-DD | model | model_class | gfx | op | shape/dtype] METHOD: the lever + how to
     apply + how to verify. POSITIVE PRIORS ONLY — no "dead-end/rejected/skip/no_win". Record mechanism
     facts as positive routing. Full per-run results (incl. nulls) go in the eval-dir final_report.md.
     (model_class, gfx, shape/dtype) is the cross-model REUSE KEY. -->

- [2026-06-07c | Qwen-Qwen3.5-27B | hybrid_linear_attention_gated_delta | gfx942 | dense GEMM prefill up/gate (K=5120, N∈{14336,16384,34816}) + down/qkv (N=5120, K∈{17408,6144}), M∈{16040→16128,16369→16384,15360,1024 padded buckets} | bf16 TN — **live bias=False**] METHOD: re-confirmed the aiter per-shape DB tune as the live-path dense-GEMM Tier-B lever (PCT_GPU_TIME=78.97%). Tier-A bake-off (op_bench): hipblaslt 9.75ms (correct, default), aiter entrypoint UNAVAILABLE on this build (62 entrypoints tried, all failed — `moe_cktile2stages_gemm2_ck` missing arg; the op_bench aiter probe can't reach the GEMM seam, but the LIVE `aiter.tuned_gemm.tgemm.mm` path works fine), triton 14.13ms = RETIRED placeholder stub (not a real impl → author-needed). Capture (235 live shapes via `AITER_TUNE_GEMM=1`) re-proved **228/234 bias=False** (only 6 tiny vision shapes bias=True) — meta.json declared bias=True (oracle) but the live serving path is bias=False; ALWAYS capture from server. Tuned 38/78 padded buckets (gradlib `gemm_tuner.py --indtype bf16 --mp 8`, FLOPs-DESC order so dominant large-M prefill landed first; partial DB = no regression, uncovered → default). N=34816 M=16384→triton sol=0 auto; all other dominant rows→hipblaslt; every err_ratio=0.0 (parity-safe). Engagement PROVEN in fresh process: all 8 dominant bias=False shapes hit "is tuned on cu_num = 304" via padded_M (16040→16128, 16369→16384). Contention-robust min-based 2-round same-session A/B (tgemm.mm(a,b,None)): geomean **1.029x isolated**, wins N=34816 (1.047x), alt-M N=16384 (1.052x), down K=17408 (1.056x), down K=6144 M=15360 (1.058x); dominant 16040×14336/16384 flat (already optimal) — matches the 2026-06-07/07b priors (1.032/1.015x). At ~79% GPU → ~+2.3% e2e ceiling; deploy winner_kind=env `AITER_CONFIG_GEMM_BF16=<csv> AITER_LOG_TUNED_CONFIG=1`. ALSO emit author_plan triton (route=author; no editable triton GEMM — op_bench stub is placeholder; CK absent on image, HIP not requested) → let e2e gate pick best of {tuned, authored}. REUSE KEY (gfx942, dense bf16 TN GEMM, bias=False live): capture-bias-from-server + FLOPs-DESC + padded_M recipe transfers; CSV solidx build-locked (re-tune on aiter/ROCm upgrade).

- [2026-06-07b | Qwen-Qwen3.5-27B | hybrid_linear_attention_gated_delta | gfx942 | dense GEMM prefill up/gate (K=5120, N∈{14336,16384,34816}) + down/qkv (N=5120, K∈{17408,6144}), M∈{15360–16384 buckets, 1024} | bf16 TN — **live bias=False**] METHOD: ⚠️ THE LIVE CAPTURE PROVED bias=False, NOT bias=True. The op task meta.json declared bias=True (oracle), but the `AITER_TUNE_GEMM=1` warm capture (ISL/OSL/conc=1024/1024/64, 234 real shapes) showed 228/234 dense GEMMs are issued with **bias=False** (bias added separately); only 6 tiny vision shapes are bias=True. The prior 2026-06-07 tuned CSV used bias=True → would have 0 engagement on the live bias=False path. So ALWAYS capture bias from the live server; never inherit it from meta.json. Recipe: capture → bucket-reduce M via `aiter.ops.gemm_op_common.get_padded_m(M,N,K,0)` (234→78 unique padded buckets) → **sort input by M·N·K FLOPs DESC** so gradlib (which processes input order) tunes the GPU-time-dominant large-M prefill shapes FIRST (incremental writes → dominant coverage lands within budget; gradlib otherwise tunes M-ascending = decode-first = worst ROI). `gemm_tuner.py --indtype bf16 --mp 8`; ~2-3min/big-prefill shape racing ~1365 hipBLASLt+asm+triton+skinny solutions. All 5 dominant prefill families got FULL large-M coverage (25 rows): N=34816 M=16384→**triton sol=0 auto**, all others→hipblaslt; every row err_ratio=0.0 (parity-safe). Engagement PROVEN in a fresh process with `AITER_CONFIG_GEMM_BF16=<csv> AITER_LOG_TUNED_CONFIG=1` calling tgemm.mm(a,b,**None**) — 6/8 dominant bias=False shapes hit "is tuned on cu_num = 304" via padded_M (M=16040→16128, M=16369→16384). Contention-robust min-based same-session A/B (3 fresh-process rounds, box shared w/ tuner): geomean **1.015x isolated**, wins concentrated N=34816 (1.039x), alt-M N=16384 (1.049x), down K=17408 (1.051x); dominant 16040×14336/16384 flat (already optimal) — matches the 2026-06-07 prior (1.032x). At ~79% GPU time → ~+1.2-2.4% e2e ceiling, Integrator gates stacked. Deploy as winner_kind=env. REUSE KEY (gfx942, dense bf16 TN GEMM, **bias=False live**): capture-bias-from-server + FLOPs-DESC tuning order + padded_M bucketing recipe transfers; CSV solidx build-locked (re-tune on aiter/ROCm upgrade).

- [2026-06-07 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | dense GEMM prefill up/gate (K=5120, N∈{14336,16384,34816}, M∈{16040,16369,1024}) | bf16 TN +bias] METHOD: aiter per-shape DB tune is the live-path Tier-B lever and IS a real isolated win on the up/gate head. Live seam = `aiter.tuned_gemm.tgemm.mm`/`gemm_a16w16`; default DB has NO config for these exact (M,N,K) (logs "not found tuned config ... using torch solution:0") so untuned aiter≈hipblaslt. Captured the 8 real shapes → `gradlib/gemm_tuner.py --indtype bf16 --mp 8` (~25min, racing ~1365 hipBLASLt + asm solutions/shape; small-M 1024 buckets are fast, big N=34816 prefill ~minutes each; writes incrementally). All 8 tuned to **hipblaslt** solutions, err_ratio=0.0 (numerically validated bf16 algo swap → parity-safe). Same-session interleaved A/B (3x): untuned geomean 2.957ms → tuned 2.866ms = **1.032x isolated**, win concentrated on N=34816 (9.72→9.29, 9.97→9.38 ≈1.05–1.06x) and alt-M N=16384 (4.57→4.34); dominant 16040×14336 already near-optimal (flat). Deploy env `AITER_CONFIG_GEMM_BF16=<csv> AITER_LOG_TUNED_CONFIG=1`; engagement PROVEN ("is tuned on cu_num = 304 ... libtype is hipblaslt" on every shape). At ~54% GPU time, 1.032x isolated → ~+1.7% e2e ceiling (Integrator gates stacked). ALSO emit author_plan triton (route=author; no editable triton GEMM for this op — op_bench triton stub is a placeholder, 5.56ms ≫ 2.87ms) — let the e2e gate pick best of {tuned, authored}. REUSE KEY (gfx942, K=5120 up/gate prefill bf16 TN +bias): the tuned CSV + capture recipe transfers; re-tune on any aiter/ROCm upgrade (solidx build-locked).

- [2026-06-05 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | dense GEMM prefill up/gate (K=5120, N∈{14336,16384,34816}) + down/qkv (N=5120, K∈{17408,6144}) | M~1k–16k bf16 TN +bias] METHOD: optimize the dense-GEMM head via (Tier-B) the aiter per-shape DB and (Tier-C) an authored Triton GEMM, both gated e2e. The live path is aiter `tuned_gemm.py::gemm_a16w16` (rebind seam for an authored kernel). Tier-B: `AITER_TUNE_GEMM=1` capture → `gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>` (gradlib races hipBLASLt/asm/triton/skinny solutions per shape) → deploy `AITER_CONFIG_GEMM_BF16=<csv>`; verify with `AITER_LOG_TUNED_CONFIG=1` (`is tuned on cu_num` hits >0). Tier-C: author+optimize a Triton GEMM via team_workflow on the op task dir. Cover the full shape set (down-proj K=17408, qkv K=6144, lm_head, decode M-buckets), one shared CSV. Parity-safe (hipBLASLt algorithm swap; tuner `err_ratio<0.05`). Gate STACKED on the current accepted config with the tight interleaved A/B (0.5% band). REUSE KEY (gfx942, K=5120 up/gate & N=5120 down/qkv prefill bf16 TN +bias): the aiter tuned CSV + authored-triton approach transfers to same-class prefill GEMM shapes.

- [2026-06-05 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | full-attention prefill | mixed] METHOD: `--attention-backend triton` is a cheap real win (+~5% e2e) and exposes an editable Triton `_fwd_kernel` for later head/kernel work. Try it first (Config Tuner); parity holds.
