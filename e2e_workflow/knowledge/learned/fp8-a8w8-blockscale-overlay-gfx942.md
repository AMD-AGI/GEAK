---
key: fp8_a8w8_blockscale dense GEMM · gfx942 · sglang Triton live path
type: lever
confidence: ★★★
effect: iso ~1.06–1.16× prefill (Triton-overlay, DEPRECATED); CK-tuned KERNEL ~1.78× vs untuned Triton on the M=13645 head (kernel-level)
confirms: 17
last_seen: 2026-07-06
status: DEPRECATED-FOR-THIS-EVAL
---
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
