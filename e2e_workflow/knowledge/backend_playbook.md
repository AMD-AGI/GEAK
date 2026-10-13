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

## Roofline prior calibration (predicted vs measured — one line per direction)
- 2026-08-19 · gfx950 vLLM mxfp4 grouped fused-MoE (`_matmul_ogs...swiglu`, gpt-oss-120b, decode): roofline
  predicted `attainable_speedup=1.0`, `expected_e2e_gain_pct=0.0` (memory-bound, `roofline_pct` 0.95–1.0,
  headroom `saturated`, confidence **low**). MEASURED e2e **+26.9%** via a whole-file Triton rewrite. The
  device-time byte/FLOP roofline model was WRONG for this seam — it cannot see the win, which is
  host launch-overhead / decode-seam collapse, not a byte reduction. Correct behavior held (confidence was
  low → not ranked on, head not dropped). Lesson: for a fused MoE dispatcher at decode, do NOT trust a
  `saturated`/`attainable=1.0` roofline verdict to size the opportunity; the launch-overhead win is invisible
  to it. This is the exact "measured EXCEEDS predicted attainable" failure mode — flag loudly.
- 2026-10-01 · gfx942 sglang fp8 Qwen3-14B dense MLP (`_e04_kernel`, silu+group-quant fusion, decode,
  10.76% head): roofline predicted `attainable_speedup=7.264×`, `expected_e2e_gain_pct=9.28%` (bound
  **latency**, `roofline_pct` 0.121 → "far from ceiling", headroom `underperforming`, confidence **low**).
  MEASURED via a 2D-tiled shape-adaptive tile micro-tune: isolated **1.656×**, e2e **+2.32%**. Here the
  model was OPTIMISTIC the other way — a latency-bound `attainable` of 7.26× over-predicted what a tile
  tune recovers by ~4.4×. Correct behavior held (confidence low → not ranked on). Lesson: a latency-bound
  `attainable_speedup` at low confidence is a weak upper bound on BOTH sides; size a micro-tune from the
  `pct_gpu_time` Amdahl ceiling (here 4.45% max → realized ~half), not from the roofline attainable.
- 2026-10-01 · gfx942 sglang fp8 Qwen3-14B PREFILL fp8 a8w8 block-scale dense GEMM
  (`_gemm_a8w8_blockscale_seed_kernel`, c0 authored-Triton seam, ~84% head): roofline predicted
  `attainable_speedup=4.978×`, `expected_e2e_gain_pct=67.2%` (bound **latency**, confidence **low**).
  MEASURED: a deepest-seam RE-capture + RE-author of the already-authored Triton kernel produced a
  BYTE-COPY candidate (`weighted_speedup=0.9926`, iso ~1.0×) → e2e **dead-end**, i.e. NO further win beyond
  the already-banked +6.55% authored-Triton swap. Third confirm that a latency-bound, low-confidence
  roofline `attainable` grossly over-predicts (4.978× vs ~1.0× realized). Correct behavior held (conf low →
  not ranked on; the head is NOT dropped — at 84% it STAYS the top target, but the only remaining lever is
  a genuinely NEW kernel idea (flydsl SOTA fp8 GEMM / a real split-K rewrite), NOT another re-capture of the
  same seam, which is exhausted at this budget).

## How to use this in a run
1. Architect reads the Profiler Top-N classification + shapes.
2. For `library_*` kernels → hand to Config Tuner with the ranked swaps above (no source edit).
3. For editable kernels → hand to Extractor + kernel squad; pass the ranked backends as the
   squad's "candidate backends" so it compares them via the (immutable) unittest.
4. **CURATE** `knowledge/learned/` after the run (read INDEX → merge/insert ≥★★ / archive
   contradicted), per `knowledge/learned/README.md`.

## Learned experience → `knowledge/learned/`
Confirmed routing/method findings are NOT appended here anymore. They live as distilled, evidence-cited
cards in **`knowledge/learned/`**, read via **`knowledge/learned/INDEX.md`** (grouped by reuse key
`kernel_class · gfx`). Open only the cards matching the current run's `(model_class, gfx, regime)`;
rank by `EV = Amdahl_ceiling × confidence`; honor each card's `dead-end:` lines.
