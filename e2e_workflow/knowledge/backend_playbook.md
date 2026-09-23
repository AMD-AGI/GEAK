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

## gfx1151 / RDNA3.5 (Strix Halo APU) — a DIFFERENT menu, not a derated one

Everything above is MI300X/MI350X experience. On gfx1151 most of that menu is **absent**, and one
entry is **inverted**. Do not derate the CDNA priors — replace them.

| backend | gfx942/950 | **gfx1151** |
|---|---|---|
| aiter C++/CK GEMM + per-shape DB (`AITER_CONFIG_GEMM_BF16`) | **THE lever** | **absent** |
| CK / ck_tile | best paged attention | **`ckProfiler` not on the image** |
| hipBLASLt offline Tensile tune | available | **`hipblaslt-bench` not on the image** |
| FlyDSL | SOTA author backend | CDNA-only |
| fp8 / MXFP4 / block-scaled | core lever | **no hardware path at all** |
| PyTorch TunableOp | *not* the lever (aiter bypasses the torch dispatch) | **IS a lever — `[measured]` +10.93% e2e fresh-server, per-shape BLAS table** |
| aiter **Triton** `gemm_a16w16` | `use_aiter_triton_gemm()` gates it **OFF** | **engages and wins — `[measured]` +13.85% e2e** |
| Triton (authored) | one option among many | the main authoring path |

**Class → backend on gfx1151 (measured priors):**

| kernel class | try in this order | evidence |
|---|---|---|
| dense GEMM (decode/skinny, the e2e-critical one) | reroute `rocm_unquantized_gemm_impl` → aiter-Triton `gemm_a16w16` + gfx1151 tile table → TunableOp per-shape table | `[measured]` +13.85% / +10.93% e2e — recipe: `gemm_tuning/gfx1151_gemm_tuning.md` |
| dense GEMM (large square, prefill) | **leave it on the vendor path** | `[measured]` vendor wins 3 of 4 GEMM regimes here |
| `lm_head` | skip | `[measured]` already at the 233 GB/s DRAM wall |
| fused row ops (softmax, add+rmsnorm) | author Triton, collapse dispatches | `[measured]` 2.21× / 1.81×, reaching 99.4–99.8% of the measured DRAM wall |
| attention decode | author Triton, split KV for residency | `[measured]` GQA 2.92× |

Three structural cautions specific to this part:
- **Never force a BLAS library globally.** `[measured]` per-shape mixing **+14.3%** vs global
  hipBLASLt **−8.2%** — a 22-point spread. The win is the per-shape table, nothing else.
- **`torch` defaults to rocBLAS here** (`preferred_blas_library()` → `Cublas`), so a `torch.mm`
  baseline says nothing about hipBLASLt.
- **Isolated × does not predict e2e at decode.** `[measured]` 1.31× isolated → +13.85% e2e; the win
  is launch/dispatch. Gate on e2e in both directions.

Hardware model: `perf_knowledge/hardware/rdna35_gfx1151/`. Roofline denominators and the **third
(32 MB MALL) roof**: `analysis_skills/roofline/peaks.md` + `SKILL.md` §3 step 5 — scoring a
MALL-resident kernel against the DRAM roof is wrong by 3.4× and silently voids its verdict.

## Roofline prior calibration (predicted vs measured — one line per direction)
- 2026-08-19 · gfx950 vLLM mxfp4 grouped fused-MoE (`_matmul_ogs...swiglu`, gpt-oss-120b, decode): roofline
  predicted `attainable_speedup=1.0`, `expected_e2e_gain_pct=0.0` (memory-bound, `roofline_pct` 0.95–1.0,
  headroom `saturated`, confidence **low**). MEASURED e2e **+26.9%** via a whole-file Triton rewrite. The
  device-time byte/FLOP roofline model was WRONG for this seam — it cannot see the win, which is
  host launch-overhead / decode-seam collapse, not a byte reduction. Correct behavior held (confidence was
  low → not ranked on, head not dropped). Lesson: for a fused MoE dispatcher at decode, do NOT trust a
  `saturated`/`attainable=1.0` roofline verdict to size the opportunity; the launch-overhead win is invisible
  to it. This is the exact "measured EXCEEDS predicted attainable" failure mode — flag loudly.

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
