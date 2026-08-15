# Backend Playbook — Which Backend Suits Which Kernel (persistent experience)

This is the **experience library** the System Architect owns. It maps a kernel CLASS to a ranked list
of backends worth trying, and it GROWS: after every run, append confirmed POSITIVE methods/routing to the
"Learned" section (with model, shape regime, and the measured result). Seeded from MI300X (gfx942)
experience; treat the seed as priors, not gospel — the unittest is the judge.

## Backend menu (what each is good at on MI300X)
- **aiter** — AMD's fused op library (GEMM, rmsnorm+quant, MoE, some attention). **On sglang/gfx942 it
  IS the live dense-GEMM dispatcher** (`tuned_gemm.py` → hipBLASLt `Cijk_*`/asm/triton/skinny). Tune
  its per-shape DB (`bf16_tuned_gemm.csv`) — this is THE GEMM lever (see `gemm_tuning/aiter_gemm_tuning.md`).
  Also fuses norm+quant; often wins skinny/decode GEMM.
- **hipBLASLt / Tensile** — the kernels aiter actually executes for dense GEMM. NOT separately tunable
  via `HIPBLASLT_TUNING_FILE` on this stack (aiter bypasses the PyTorch/hipBLASLt C dispatch for its
  tuned shapes). The "not found tuned config" warnings = aiter shapes you haven't tuned yet = target list.
- **CK / ck_tile (Composable Kernel)** — attention (FmhaBatchPrefill/paged), some GEMM. Best paged
  attention on MI300X today; tunable via instance selection.
- **Triton** — custom/novel kernels (mamba/gated-delta linear attn, fused norms, activations,
  bespoke fusions). Fastest to iterate; good for memory-bound and fusion. The kernel squad's home.
- **HIP / raw** — when you need warp-cooperative control Triton can't express, or to hand-fuse.
- **asm (MFMA intrinsics / hand asm)** — last 10–20% on a proven-hot compute-bound kernel; high cost,
  only for a kernel that is large pct_gpu_time and already backend-chosen.

## Class → ranked backends (priors)
| kernel class | try in this order | notes |
|---|---|---|
| dense GEMM (prefill, large M) | **tune aiter `bf16_tuned_gemm.csv`** (capture→gradlib→`AITER_CONFIG_GEMM_BF16`) | confirmed +1.22% (partial) on hybrid-dense gfx942; NOT TunableOp/HIPBLASLT_TUNING_FILE — see `gemm_tuning/aiter_gemm_tuning.md` |
| skinny GEMM (decode, M=batch) | tune aiter DB (decode M-buckets) → skinny default | aiter dispatches skinny kernels; tune M=16/32/48/64 buckets |
| paged/prefill attention | CK(ck_tile) → aiter → triton FA | `--attention-backend` swap is free to try |
| mamba / gated-delta linear attn | triton (tune) → HIP | almost always Triton; tune tiling/scan |
| rmsnorm (+quant/residual) | aiter fused → triton fused | fuse with neighbor add/quant |
| rope / qk-norm | triton fused → aiter | fold into attention pre-step |
| activation (silu/gelu + mul) | fused act_and_mul (aiter/triton) | collapse into the producing GEMM epilogue if possible |
| elementwise/fill/cast/copy | fuse away (host_runtime) / cuda-graph | usually shouldn't be its own kernel |

## How to use this in a run
1. Architect reads the Profiler Top-N classification + shapes.
2. For `library_*` kernels → hand to Config Tuner with the ranked swaps above (no source edit).
3. For editable kernels → hand to Extractor + kernel squad; pass the ranked backends as the
   squad's "candidate backends" so it compares them via the (immutable) unittest.
4. **CURATE** `knowledge/learned/` after the run (read INDEX → merge/insert ≥★★ / archive
   contradicted), per `knowledge/learned/README.md`.

## Roofline-prior calibration (predicted vs actual — one line per measured direction)
Recorded so the `target_eff` priors / byte models in `analysis_skills/roofline/SKILL.md` self-correct.
- 2026-08-13 · gfx942 · vLLM Qwen3-14B-FP8 TP=1 conc=64 · **dense fp8 blockscale GEMM** (CK
  `kernel_gemm_xdl_cshuffle_v3`, 62.14% GPU): predicted `roofline_pct 0.27`, `attainable 3.34×`,
  `expected_e2e +41.5%` (confidence **low**, bound_type latency) → measured **iso 1.6955×, e2e +19.35%**.
  Prior **~2× OPTIMISTIC in magnitude but CORRECT in rank** (it was the #1 underperforming head and the
  authored-Triton route paid). Direction of error: `attainable_speedup` derived from a latency-bound
  (non-HBM) roof over-credits headroom — treat low-confidence latency-bound `attainable` as an upper bound.
- 2026-08-13 · same rig · **`kernel_unified_attention_3d`** (18.22→21.69% GPU): predicted
  `roofline_pct 0.517` vs the 0.50 paged-attn target, `attainable 1.0×`, `expected_e2e 0.0` = *saturated*.
  Not yet measured (extraction failed) — **unfalsified, do not upgrade its confidence.** Its SHARE rose
  only because the GEMM shrank (absolute 3.10→2.96 ms/decode-step), which the prior predicted correctly.
- 2026-08-13 · same rig · post-win head `_decode_gemm_kernel` 48.3% @ `roofline_pct 0.42`,
  `attainable 2.14×`, `+25.7%` (low conf) — pending. Its `_reduce_kernel` (5.3%) is a *self-inflicted*
  split-K round trip, i.e. a byte-reduction target the roofline model does not model as removable.
- 2026-08-14 · same rig · **`_reduce_kernel`** (5.3% GPU, the split-K fp32 partial reduction): treated as a
  standalone extraction → **measured iso 1.00× (no headroom)**. It is a pure streaming reduce already at the
  HBM roof, so the roofline view was RIGHT that nothing is recoverable *in place*; the model's blind spot is
  that the whole kernel is *removable by its producer*. Lesson for the prior: a self-inflicted round-trip
  should be scored against the PRODUCER's byte budget, never as its own tuning target.
- 2026-08-14 · same rig · **quant-prologue cluster** (`_fused_rms_fp8_group_quant` +
  `_act_mul_and_dynamic_fp8_group_quant` + 2 aiter HIP quant kernels, 10.2% GPU): authored iso **1.3889×**,
  but the one-dispatch fusion was live-UNSAFE (deferred launch vs. immediate consumer) so the DEPLOYED iso
  was **1.196×** → Amdahl ceiling only **+1.71%**, measured e2e **+0.59% with overlapping distributions**
  (session ref spread 5.8%). Calibration: Amdahl was accurate (0.59% ≤ 1.71%); the error mode to avoid is
  budgeting from the AUTHORED iso instead of the DEPLOYABLE one. Under ~10% pct_gpu_time, screen with
  `pct_gpu × (1 − 1/S_deployable)` and expect carry-forward, not a solo gate pass.

## Learned experience → `knowledge/learned/`
Confirmed routing/method findings are NOT appended here anymore. They live as distilled, evidence-cited
cards in **`knowledge/learned/`**, read via **`knowledge/learned/INDEX.md`** (grouped by reuse key
`kernel_class · gfx`). Open only the cards matching the current run's `(model_class, gfx, regime)`;
rank by `EV = Amdahl_ceiling × confidence`; honor each card's `dead-end:` lines.
