---
key: fp8_a8w8_blockscale dense GEMM · gfx942 · sglang Triton live path + vLLM CK live path
type: lever
confidence: ★★★
effect: **BEST MEASURED e2e = +19.35% (vLLM/gfx942, CK live seam replaced by an AUTHORED 2-lane Triton GEMM rebound at `aiter:gemm_a8w8_blockscale`; iso 1.6955×, disjoint A/B, gsm8k-clean)**; CK-tuned env lever ≈1.008× null on a vLLM CK seam; older Triton config-JSON overlay iso ~1.06–1.16× prefill (DEPRECATED)
confirms: 21
last_seen: 2026-08-14
status: ACTIVE (vLLM gfx942) · Triton-config-JSON sub-lever DEPRECATED
---
## 20th confirm — **the head-track WINNER: author a 2-lane Triton fp8 blockscale GEMM** (Qwen3-14B-FP8, vLLM 0.21.0 TP=1, gfx942/304CU, ISL/OSL 1024/1024 conc=64) — 2026-08-13
- lever: the CK `kernel_gemm_xdl_cshuffle_v3` head (62.1% GPU, non-editable) is beaten outright by an
  AUTHORED Triton replacement rebound at the same seam → **4022.4 → 4800.5 tok/s = +19.35% e2e**
  (3+3 repeats, disjoint: cand_min 4732.6 > ref_max 4030.3; TPOT 15.53→13.20 ms; gsm8k 52/60 both legs).
  Post-win profile: CK gone; total GPU 17.00→13.66 ms/decode-step, the GEMM seam 10.57→7.32 ms (−31%).
- apply: PYTHONPATH-only overlay (`sitecustomize` rebinds the module attr `aiter:gemm_a8w8_blockscale`;
  vLLM's op body does a LAZY `from aiter import ...`, so the rebind engages every call and survives
  PIECEWISE cuda-graph capture). site-packages untouched, fully reversible.
- the two lanes (pure-integer config tables — no autotune, no host sync → capture-safe):
  · **decode M≤64**: ONE masked m-tile (BLOCK_M=16/32/64 by M) + FINE BLOCK_N to fill 304 CUs
    (seed's BM64/BN128 gives only 40 CTAs = 13% fill); + deterministic split-K (fp32 partial + reduce).
    Measured table (N,K)→(BLOCK_N,SPLIT_K,warps,waves,stages,cache_mod): qkv (7168,5120)=(128,5,8,0,3,.cg),
    o (5120,5120)=(128,5,8,0,3,.cg), gate_up (34816,5120)=(128,**1**,8,0,2,.cg), down (5120,17408)=(128,**8**,2,0,2,.cg);
    M≤16 GEMV corner → SPLIT_K=8 (1.4–2.3× over the M=64 configs).
  · **prefill M≥512**: dense kernel + a `SCALAR_BS` fast path (BLOCK_N ≤ GROUP_N=128 ⇒ the block scale is a
    scalar per n-tile ⇒ `acc += dot*(a_s*b_s)` instead of a per-element rescale).
- caution (**the big one — cost 2 rejected A/Bs**): on this SAME head, three sibling candidates with
  *comparable* isolated wins split violently at e2e — c0 1.6955×→**+19.35%**, c1 1.6797×→+13.65%,
  a c1 re-author 1.6488×→**−32.51%**, routed aiter-bpreshuffle 1.5346×→**−32.54%** (TPOT 15.5→23.3 ms in
  both losers). Isolated serving-weighted speedup on captured oracle shapes does NOT rank e2e; **A/B every
  sibling** (see [[method-iso-e2e-gap-decode-tpot]]).
- caution: routed CK-bpreshuffle also costs a 12.3 GiB duplicate shuffled-weight copy (KV pool 154.4→142.1 GiB);
  not binding at conc=64 here, but it was NOT the reason it lost — decode TPOT was.
- caution (**audit what the win ADDED — but fix it INSIDE the GEMM, not as a standalone target**,
  2026-08-14): this decode split-K table creates a new `_reduce_kernel` (fp32 partial reduction) worth
  **5.3% GPU**, self-inflicted byte traffic the roofline model does not score as removable. Attacking it as
  its OWN extracted op yielded **zero isolated speedup** (it is already a pure streaming reduce at the HBM
  roof) — so the lever is to remove the round trip in the GEMM (atomic/single-pass or persistent-CTA
  accumulation), or to re-pick SPLIT_K against the LIVE padded cuda-graph M buckets instead of the oracle's
  M∈{1,64}. Always re-profile after acceptance and audit the additions, then route them to the producer.
- source: /wekafs/test_results/Qwen3_14B_20260813/e2e_Qwen3-14B-FP8_20260813_031549_2866199_26474/
  overlay/cand_kernel_gemm_xdl_cshuffle_v3/{integrate_result.json,integrate_result_c1_triton*.json,
  integrate_result_c2_aiter.json,overlay_c0_triton/}; profile/round_head/ (+ the `_reduce_kernel` null)
---
> ✅ **18th confirm — CK-tuned≈CK-default is now MEASURED, not inferred (2026-08-11, Qwen3-14B-FP8
> **vLLM** 0.21.0 TP=1, gfx942/MI300X cu_num=304, VLLM_ROCM_USE_AITER=1 so the live seam is ALREADY CK).**
> Ran the mandated CK tuner over 24 (M,N,K) (4 NK families x M in {16,32,64,128,1024,2048}), `--libtype both
> --mp 1`: all 24 winners `libtype=ck`, errRatio 0.0. Deployed `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>` and
> verified **20/20 live buckets HIT** ("is tuned on cu_num = 304"; padded_M maps 737->1024, 1169/1324->2048,
> 1->16). Cross-process A/B (the CSV binds at IMPORT, so tuned vs default CANNOT be A/B'd in one process):
> **serving-weighted 1.0076x**, per-bucket span 0.98-1.04, prefill gate_up 0.982x. So on a vLLM CK live path
> the mandated Tier-B lever is a genuine NULL — run it (it is mandated and cheap) but budget nothing on it.
> ⚡ **The real win came from the OPPOSITE direction: vLLM's stock Triton block-scaled GEMM (EvoK
> `gemm/block_scaled/fp8_e4m3/impls/triton_2`, a tuned descendant of `w8a8_triton_block_scaled_mm`) BEATS
> tuned CK on this rig** — prefill 1.65-2.25x on all four families, decode M=1 1.04-1.42x. Note this
> INVERTS the sglang-rig finding above (there CK beat untuned Triton at small M and lost at large M); the
> direction of the CK-vs-Triton inequality is a property of WHICH Triton (untuned stock vs tuned) and of the
> M range, so always bake off both legs per bucket rather than porting a prior verdict.
> ✅ **Deployable winner = M/family-ROUTED EvoK+CK dispatcher** (CK kept only for 8<M<=256 with N,K<=8192
> i.e. the qkv/o decode buckets where CK is marginally ahead; everything else → EvoK Triton):
> serving-weighted **1.3529x** (geomean 1.4691, correctness PASS rel_err<=0.0075 @ tol 0.06), vs 1.3073x for
> unrouted EvoK — routing exists purely to erase two decode regressions (qkv 0.88x, o 0.93x). Must be a
> `torch.library.custom_op` + `register_fake`: the oracle wraps both legs in `torch.compile`, and the raw
> impl breaks dynamo via `get_w8a8_block_fp8_configs` → `torch.cuda.get_device_name`
> (`Unsupported('torch.* op returned non-Tensor')` → silent `optimized_ms: null`). Router must be pure
> integer branching on (M,N,K) to stay graph-capture safe.
> 🔧 **Image defect worth knowing:** the CK tuner's JIT build fails on this ROCm 7.2.2 image with
> `thrust/system/hip/config.h: fatal error: 'cub/detail/detect_cuda_runtime.cuh' file not found` (in-place-
> hipified rocThrust, pulled in via `torch/headeronly/util/complex.h`). Fix = `CPATH=<dir>` holding empty
> `cub/{util_debug,util_namespace,version}.cuh` + `cub/detail/detect_cuda_runtime.cuh` + a self-contained
> `thrust/complex.h`, then delete stale `aiter/jit/build/module_*_tune` dirs and `lock_module_*` before retry.
> ✅ **17th confirm (2026-07-06, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, e2e_cycle0).**
> CK skill end to end on the 12 live (M,N,K) (4 NK families × {1,64,16384}), `--libtype both --mp 1`:
> ALL 12 winners `libtype=ck`, errRatio 0.0. §9.1 scale-layout check (M=64, all 4 families): CK wants
> `transpose_scale=False` (rel-err ~1e-5; True→~0.19 = the catastrophic layout bug). In-process CK-vs-untuned-Triton
> A/B (correct non-transposed scale): **decode M=1 4.7–6.2×, M=64 1.37–6.0× FASTER** (up_gate weakest at
> 1.37×/M64; qkv/o ~6×); **prefill M=16384 0.56× (SLOWER on every family)** → M-routed overlay (M≤256→CK)
> is mandatory, reconfirmed. Share-weighted decode speedup ~1.87× (M64) / ~4.1× (M1). CK also FIXES the
> known untuned-Triton small-M/large-K down-proj defect (CK rel_err ~true-math; Triton wrong at M=1,K=17408).
> Engagement verified ("is tuned on cu_num = 304"). Shipped `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE)
> + M-routed fp8_utils CK overlay. e2e gate pending (Integrator).
> ✅ **16th confirm (2026-06-25, Qwen3-14B-FP8 TP=2, gfx942/MI300X cu_num=304, e2e_qwen3_14b_fp8_..._3515_8954).**
> Re-ran the CK skill end to end on the exact 16 live (M,N,K) (4 NK families × {1,16,2048,13645}), `--libtype
> both --mp 2`: ALL 16 winners `libtype=ck`, errRatio 0.0. Dominant head M=13645,N=5120,K=8704 CK
> kernelId=0 = 1.469 ms (matches the 1.456 ms / ~1.78× kernel prior). Engagement reconfirmed
> ("is tuned on cu_num = 304"); faithful block-scale dequant correctness rel_err 0.0017 (tol 0.06). Op_bench
> bake-off reconfirmed bpreshuffle drop-in is WRONG (rel_err 42.4). Shipped the M-ROUTED overlay (M≤256→CK,
> else Triton) per the banner below. NOTE: one stale JIT baton lock from a prematurely-killed tuner run
> stalled the build for ~25 min with 0 compiler procs — clearing `aiter/jit/build/lock_module_*` +
> `build/module_*_tune` and restarting (per skill §2/pitfall) fixed it; watch for this.
> 🔑 **REGIME SPLIT is the deployable refinement (2026-06-25, Qwen3-14B-FP8 TP=2, gfx942, conc=16).**
> Re-ran the CK skill end to end (tuned 16 live (M,N,K) = 4 NK families × {1,16,2048,13645}, `--libtype both
> --mp 2`, all winners `libtype=ck` errRatio 0.0, engagement confirmed via "is tuned on cu_num=304"). The
> PRODUCTION custom-op `aiter.gemm_a8w8_blockscale` (what fp8_utils calls post-switch), measured eager with
> CUDA events on this box, is **regime-split, not uniformly faster**:
>   · decode/skinny-M (M≤256): CK ~**3.4–4.0× FASTER** than the untuned Triton block-scale default
>     (Triton is ~0.11–0.14 ms flat for any small M; CK ~0.03 ms). Crossover at M≈256↔512.
>   · prefill/large-M (M≥512): CK ~**0.5–0.66× (1.5–2× SLOWER)** than Triton on EVERY family (down/gate_up/
>     qkv/o), measured both device-only (CUDA events) and wall, plain & transposed x_scale layouts identical.
>     This contradicts the tuner's own ~1.46–1.49 ms "kernel" `us` column — the production
>     `gemm_a8w8_blockscale_ck` lookup path runs ~4.85 ms eager on M=13645,N=5120,K=8704. Trust the
>     production-op A/B, not the tuner CSV `us`, for the prefill verdict.
> ✅ **Deployable winner = M-ROUTED overlay (NOT a wholesale Triton→CK import swap).** A bare import swap
> routes prefill to CK too → regresses the GPU-time-heavy prefill. Instead the fp8_utils overlay rebinds
> `triton_gemm_a8w8_blockscale` to a dispatcher: `M≤256 → CK (tuned), else → stock Triton`
> (env `SGLANG_CK_BLOCKSCALE_M_MAX`, default 256). Verified correct (err~0.003, tol 0.06): decode hits CK
> 0.0355 ms while prefill passes through to Triton at parity. Captures the e2e-critical decode/TPOT win
> (steady-state M≈conc=16) with ZERO prefill regression. `winner_kind=env` (apply_env
> `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`) + `code_patch=<fp8_utils M-routed CK overlay>` +
> `tuning_artifact=<tuned.csv>`. Still gate at e2e (HIP-graph capture must keep the small-M CK path live).
---
> ✅ **CK-skill result (this eval, Qwen3-14B-FP8 TP=2, gfx942/MI300X):** ran the MANDATED CK playbook end
> to end. Tuned all 20 live (M,N,K) (4 NK families × {1,16,2048,13645,16385}) with `gemm_a8w8_blockscale_tune.py
> --libtype both --mp 2`; ALL winners are `libtype=ck`, errRatio 0.0. Dominant head (M=13645,N=8704,K=5120)
> CK kernelId=0 = **1.456 ms vs untuned Triton 2.60 ms = 1.78× at the kernel level** (tuner & raw `gemm_a8w8_blockscale_tune`
> agree exactly). Engagement verified: `AITER_LOG_TUNED_CONFIG=1` prints "is tuned on cu_num=304".
> ⚠️ **CONDITIONED caution for the Integrator e2e gate:** the *production* custom-op `aiter.gemm_a8w8_blockscale`
> (what fp8_utils calls after the Triton→CK switch) measured **4.85 ms eager** on the same shape — ~3.4 ms
> per-call Python/dispatch wrapper overhead ON TOP of the 1.46 ms kernel (steady-state over 500 iters, NOT a
> one-time cost). So in EAGER it is ~1.9× SLOWER than Triton despite the faster kernel. This overhead is the
> kind HIP-graph / torch.compile capture amortizes (the live serving path captures decode & wraps prefill),
> which is exactly why the e2e gate — not an isolated eager bench — is the judge. The win is REAL at the
> kernel; verify the wrapper overhead is absorbed under graph capture at e2e before crediting it.
---
> ⛔ **DEPRECATED for the current eval.** The Triton config-JSON overlay below keeps the slow Triton
> blockscale seam live and **bypasses the user's CK skill**. For an sglang fp8 a8w8 blockscale head the
> MANDATED lever is the CK playbook `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`
> (capture live (M,N,K) → aiter CK tuner `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` →
> fp8_utils Triton→CK switch overlay + `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`). The correct
> baseline for the speedup is the **UNTUNED Triton default** the stack runs by default — NOT a CK-default
> heuristic. The "CK env yields ~1.00×" caution below was measured CK-default-vs-CK-tuned on a vLLM CK
> live path; it does NOT describe CK-tuned-vs-Triton-untuned on this sglang rig. Do not let this card
> steer the bake-off to the Triton overlay / flydsl / aiter-bf16 levers — follow the CK skill.

# fp8 a8w8 blockscale GEMM → per-(N,K) M-bucketed Triton config-JSON overlay  (DEPRECATED — see banner)
- lever: the live seam is `aiter.ops.triton.gemm_a8w8_blockscale`. Stock has NO per-(N,K) config for
  the model's large shapes → it uses a generic tile = headroom. Win = drop a tuned
  `gfx942-GEMM-A8W8_BLOCKSCALE-N=<N>-K=<K>.json` into `aiter/ops/triton/configs/gemm/`
  (winner_kind=**patch**; `AITER_TRITON_CONFIGS_PATH` is `__file__`-fixed, not env-overridable).
- apply: **M-bucket the config** — tile shape depends on (N,K):
  · wide-N up/gate (N=34816,K=5120): prefill BM=256/BN=128/GROUP_M=4/nw=8.
  · K-heavy/narrow-N down (N=5120,K=17408): prefill **BM=128/BN=256**/GM=4/nw=8 (widen BN, keep BM).
  · square-ish qkv/o (N=5120,K=6144): prefill BM=256/BN=128/GM=4/nw=8 (small clean win).
  · decode (M≤1024) MUST stay generic BM=128. Integrator must rebind BOTH `sglang...fp8_utils` globals
    `triton_gemm_a8w8_blockscale` + `gemm_a8w8_blockscale_bpreshuffle`.
- verify: honest in-process `config=` kwarg A/B, same synth fp8 operands held fixed, interleaved
  min-of-N; confirm engagement via live `_get_config(M,N,K)` (returns a (dict,use_persistent) tuple → [0]).
- caution: a FLAT overlay (BM=256 for all M) tanks decode 0.6–0.7× — decode MUST stay generic.
- caution: BN=256 + BM=256 together = LDS spill (0.29×) — widen only one dim.
- caution: on the **vLLM CK live path** (not Triton) this overlay does NOT apply — live is CK
  xdl-cshuffle; the lever there is env `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>`, but it yields ~1.00×
  (CK default heuristic already picks the optimal `256x128x128 intrawave_v3`). ALWAYS check which live
  path (CK vs Triton) is engaged BEFORE choosing the lever.
- caution: **backend availability is a PROVISIONING gate, not a no-win.** The mandated CK lever needs the
  aiter CK tuner (`csrc/ck_gemm_a8w8_blockscale/`); the FlyDSL alternative needs aiter's `aiter/ops/flydsl/`
  wrapper AND the top-level `flydsl` pip pkg. If `env_report.absent_backends` lists either, record the
  two-part remedy (flydsl: `pip install 'flydsl>=0.1.5'` AND a flydsl-enabled `amd_aiter` build that ships
  `aiter/ops/flydsl/` — pip flydsl ALONE is insufficient; `aiter.ops.flydsl` stays ModuleNotFoundError) and
  fall back to an available lever — never silently drop the head. See `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`.
- caution: the aiter `gemm_a8w8_blockscale_bpreshuffle` path benches ~1.5× faster than the plain
  Triton blockscale kernel BUT is WRONG as a naive drop-in (op_bench Qwen3-14B-FP8: rel_err 43.6 vs the
  blockscale baseline's 0.0075) — it needs weights preshuffled first. It is the large-M prefill lever,
  not a free swap; only use via the preshuffle seam (`aiter:gemm_a8w8_blockscale_bpreshuffle` + a once
  `shuffle_weight`), never by rebinding the live blockscale call to it directly.
- source: exp/e2e_*Qwen3.5-27B-FP8*/ runs 06-08 … 06-15 (11 re-confirms); + exp/e2e_qwen3_14b_fp8_20260624
  (13th: Qwen3-14B-FP8 TP=2, 4 per-GPU families. In-process config= A/B, fixed synth fp8 operands, min-of-50.
  Prefill M={13645,16385}: up/gate N=17408,K=5120 BM256/BN128/GM4/nw8 = 1.16×; qkv N=3584,K=5120 same = 1.15×;
  down N=5120,K=8704 BM256/BN128/GM1/nw8 = 1.09×; o N=5120,K=2560 BM256/BN128/GM4/nw8 = 1.09×; prefill geomean 1.11×.
  All correct (fp8 tol 0.06). BM128_BN256 LOST on every family here (default-class), reconfirming "widen BM not BN"
  for these K=5120-ish shapes. Decode kept generic via M_LEQ_1024 key. Overlay engagement re-verified live via
  get_gemm_config (BM 128→256 on prefill). aiter bpreshuffle benched 4.63 vs 6.48 ms (1.4×) but rel_err 41.8 = NOT a drop-in.)
  (12th: re-confirmed live seam + "not found tuned config, will use default config" headroom on up/gate
  N=17408,K=5120; aiter bf16 DB tune is the WRONG lever here — fp8 path is the Triton seam, not aiter.tuned_gemm)

## 19th confirm — Qwen3-14B-FP8, **vLLM 0.21.0 TP=1 gfx942**, live seam = aiter CK (62.14% GPU) — 2026-08-13
**bpreshuffle IS a drop-in AFTER all three layout fixes, and it BEATS EvoK triton_2 here.** The earlier
"rel_err 41.8/43.6 = not a drop-in" verdicts were an incomplete port: the call needs BOTH
(1) `shuffle_weight(w, layout=(16,16))` on the fp8 weight AND (2) a **K-major x_scale**
`xs.transpose(0,1).contiguous().view(*xs.shape)`. With both, rel_err = 0.00744 (== CK baseline).
- live seam / how to ship: rebind the module attribute `aiter.gemm_a8w8_blockscale` (vLLM's custom-op body
  does a LAZY `from aiter import ...`, so a rebind engages every call and is torch.compile/graph safe).
- op_bench M=2048 gate_up (N=34816,K=5120): CK 2.954 ms → **bpreshuffle 1.105 ms = 2.68×**;
  EvoK triton_2 1.163, aiter Triton 1.545, vLLM stock Triton 1.560. **EvoK LOST here** (opposite of the
  18th confirm's Triton-live rig) — because the live baseline is CK, not untuned Triton. Always re-measure.
- M/family router beats any single kernel: bp wins gate_up (N>8192) from M≥8, down (K>8192) from M≥128,
  qkv/o only M≥512; CK keeps small-M decode (the extra x_scale-transpose launch dominates there).
  Immutable oracle, serving-weighted: **routed_bp_ck 1.5346×** (geomean 1.5237; qkv 1.335 / o 1.267 /
  gate_up 1.823 / down 1.379) vs ck_bpreshuffle-everywhere 1.68 device-only-screen but worse at decode.
- **caution (new, cost me a FAIL):** caching the shuffled weight keyed on `w.data_ptr()` ALONE is unsound —
  a freed weight's address is recycled and the cache serves a stale permutation (oracle random-draw
  `down_M2048 rel_err 69.2`). Pin a STRONG ref to the source tensor in the cache entry.
- **caution (HBM):** the cache is a SECOND full fp8 weight copy (~13 GB over 40 layers × 4 projections) →
  memory gate. Ideal deployment shuffles in `process_weights_after_loading` (REPLACE, not duplicate);
  zero-extra-HBM fallback = routed EvoK+CK (screen 1.44 weighted).
- **Tier-B CK tune re-null (2nd rig):** `gemm_a8w8_blockscale_tune.py --preshuffle --libtype both` built its
  JIT tune modules fine (the ROCm 7.2.2 cub/thrust defect did NOT reproduce with the CPATH shim) but returned
  `errRatio 1.0 / us -1 / kernelName None` for ALL 20 shapes → `no valid candidate found` → **no tuned CSV**.
  Tuner-side correctness check, not a kernel problem. Budget nothing on the CK tuner on gfx942 vLLM rigs.
