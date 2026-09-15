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

- 2026-09-08 · gfx950 sglang+aiter Triton `kernel_unified_attention_3d` (gpt-oss-120b, ALL_DECODE, 5.01% head):
  roofline predicted `attainable_speedup=1.0`, `expected_e2e_gain_pct=0.0` (`roofline_pct` 0.567,
  `bound_type=latency`, headroom **saturated**, confidence **low**). MEASURED isolated **1.2336x**
  byte-exact, e2e +0.402% (kernel-only A/B, gate=stack). **Measured EXCEEDS predicted attainable — flag
  loudly**, and it is the SAME failure mode as the 2026-08-19 mxfp4 MoE line above: the win was a
  DISPATCH COLLAPSE (split-K NS 4→1 removes a whole `reduce_segments` launch + an fp32 segment
  round-trip), which a device-time byte/FLOP model structurally cannot see. Rule forming across two
  confirms: when `bound_type` is `latency`/launch-shaped, treat `saturated` as "no BYTE headroom", never
  as "no headroom" — the Amdahl e2e SIZING, by contrast, was accurate (predicted ~+0.3-1.0% at a 5% head,
  measured +0.402%), so keep ranking on pct_gpu_time and use roofline only to pick the MECHANISM.

- 2026-09-09 · gfx950 vLLM fp8 quant/norm prologue cluster (`_act_mul_and_dynamic_fp8_group_quant`,
  Qwen3-14B-FP8 TP1, decode-bound, 5.0% row / 13.63% cluster): roofline predicted `roofline_pct` 0.089,
  `attainable_speedup=1.0`, `expected_e2e_gain_pct=0.0`, headroom **unknown**, confidence **low**.
  MEASURED iso **1.741x** geomean and e2e **+3.966%**. **Measured EXCEEDS predicted attainable — flag
  loudly**, third confirm of the same failure mode (see the two lines above). Note the distinct
  signature here: `roofline_pct` was very LOW (8.9%) yet the model still returned `attainable=1.0` with
  `headroom_class=unknown` — i.e. the model declined to make a prediction and the pipeline read that as
  "no headroom". Treat `unknown`/`attainable=1.0` at low `roofline_pct` as UNMODELED, never as
  saturated. Correct behavior held (confidence low → not ranked on, candidate not dropped).
  · The Amdahl sizing, by contrast, was EXACT: 13.63% at the conservative serving-weighted 1.3847x
  bounds the gain at +3.94% and the measurement landed on +3.97%. Keep sizing on
  `pct_gpu_time x serving-weighted iso`; use roofline only to pick the mechanism.
  · Second calibration from the same round, this one at the KERNEL level: three kernels were rewritten
  and only the one whose per-launch cost was ABOVE the box's ~5.6 us dispatch floor improved
  (8.12 -> 5.64 us); the two already at the floor returned -2.1% / +2.2%. Add a floor check to the
  screen — a small kernel at the dispatch floor has no recoverable time regardless of its %GPU, and
  the only lever left for it is removing launches.

- 2026-09-09 · gfx950 sglang+aiter native-mxfp4 MoE (gpt-oss-120b TP1, decode-dominated), round-1
  re-profile of the accepted stack. The roofline artifact was **stage A / confidence low on every entry**,
  so per doctrine it was displayed and NOT ranked on — correct behavior, but note what it emitted:
  `attainable_speedup=1.0` / `expected_e2e_gain_pct=0.0` for BOTH ck_tile MoeFlatmm heads (37.73% and
  20.98%, `roofline_pct` 1.0, `headroom_class=unknown`) and `saturated`/1.0 for unified_attention_3d
  (5.37%, `roofline_pct` 0.794). A stage-A prior that returns `expected_e2e_gain_pct=0.0` for 64% of GPU
  time carries ZERO ranking information — `ranking_by_expected_gain` degenerated to a copy of
  `ranking_by_pct`. Treat stage A as "not a prior at all" and do not spend a phase reading it.
  · The direction actually measured this round (the `elementwise_overhead` cluster, 5.2% combined) was
  **absent from the roofline entries entirely** — the artifact only models head-threshold rows — so there
  was no prediction to score. Fourth data point for the same rule: the model is silent or `unknown`
  exactly where the launch/dispatch-shaped wins live. Hand-computed effective bandwidth on that cluster
  (153 GB/s on a 737 KB copy, 138 GB/s on a 393 KB fill, ~3% of HBM roofline) was what actually settled
  it, and it settled it as "no BYTE headroom, launch-latency-bound" — the same signature the three lines
  above say to route to launch-removal, not to a rewrite.
  · Amdahl sizing was again the reliable half: the pad-attributable share was bounded at ~3.3% of GPU
  time with an e2e ceiling well under 1%, which is why the cluster was dropped from the kernel layer
  without spending a server boot or an oracle capture.

- 2026-09-10 · gfx950 sglang+aiter, gpt-oss-120b TP1 decode, round-2 milestone —
  `unified_attention_3d` (5.37% GPU). Roofline (stage A, confidence low) predicted
  `attainable_speedup=1.0` / `headroom_class=saturated` / `expected_e2e_gain_pct=0.0`; the isolated
  measurement was **1.1226x byte-exact**. That is the SECOND time this seam beat its own saturated
  verdict (1.234x previously) — record it loudly: a device-time byte/FLOP model cannot see a
  dispatch-collapse win, so a `saturated` label must never retire a high-%GPU seam.
  · The e2e half of the prior looked right (+0.186% measured vs 0.0 predicted) but for the WRONG reason
  and must not be scored as a hit: the candidate never executed (host fast path unreachable at HIP-graph
  capture), so the leg measured stock against stock. An `expected_e2e_gain_pct` can only be calibrated
  against an ENGAGEMENT-PROVEN leg; check engagement before writing a calibration line.
  · Amdahl remained the honest bound: ceiling +0.586% (5.37% x 1.1226x), measured +0.186%, noise band
  0.5% — the ceiling correctly said "stack-only at best" before the budget was spent.

- 2026-09-11 · gfx950 vLLM+aiter, Qwen3-14B-FP8 TP1 (isl/osl/conc 1024/1024/64), round-1 milestone —
  the direction that PAID (`_act_mul_and_dynamic_fp8_group_quant_kernel` launcher re-geometry, 4.96% GPU
  at round_head) had NO roofline entry at all: the stage-A artifact only models rows at/above the 5%
  head threshold, so the winning candidate sat just under the bar and was unmodeled. **Fifth consecutive
  data point for the same rule — the prior is silent exactly where the launch/geometry-shaped wins
  live.** Score: no prediction, so nothing to calibrate; Amdahl did the work instead (4.96% x 2.063x
  => ceiling +2.53%, measured +1.6005% e2e, inside it and above the 0.5% band).
  · Every modeled row this round carried `confidence: low` except `kernel_unified_attention_3d`
  (medium, `roofline_pct` 0.70, `saturated`) — and the re-profile agrees with that one: hbm_util 0.702
  against a 0.50 target, i.e. byte reduction or nothing. Ranking on the low-confidence rows would have
  been actively misleading here: `attainable_speedup` 2.93-8.35x with `expected_e2e_gain_pct` up to
  11.6% on CK rows whose measured bake-offs closed at 1.00x in a previous round of this same run.
  Direction of the error is consistent and one-sided: **stage-A `attainable_speedup` on library CK/GEMM
  rows is grossly OPTIMISTIC (it reads a low `roofline_pct` as recoverable headroom), while it is
  SILENT/`saturated` on the seams that actually moved.** Keep using it to flag saturation, not to rank.

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
