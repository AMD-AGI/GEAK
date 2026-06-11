# Backend Playbook — Which Backend Suits Which Kernel (persistent experience)

This is the **experience library** the System Architect owns. It maps a kernel CLASS to a ranked list
of backends worth trying, and it GROWS: after every run, append confirmed POSITIVE methods/routing to the
"Learned" section (with model, shape regime, and the measured result). Seeded from MI300X (gfx942)
experience; treat the seed as priors, not gospel — the unittest is the judge.

## Backend menu (what each is good at on MI300X)
- **aiter** — AMD's fused op library (GEMM, rmsnorm+quant, MoE, some attention). **On sglang/gfx942 it
  IS the live dense-GEMM dispatcher** (`tuned_gemm.py` → hipBLASLt `Cijk_*`/asm/triton/skinny). Tune
  its per-shape DB (`bf16_tuned_gemm.csv`) — this is THE GEMM lever (see `aiter_gemm_tuning.md`).
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
| dense GEMM (prefill, large M) | **tune aiter `bf16_tuned_gemm.csv`** (capture→gradlib→`AITER_CONFIG_GEMM_BF16`) | confirmed +1.22% (partial) on hybrid-dense gfx942; NOT TunableOp/HIPBLASLT_TUNING_FILE — see `aiter_gemm_tuning.md` |
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
4. Record results below.

## Learned — POSITIVE methods & routing only (append after each run; newest first)
<!-- NEW ENTRIES GO DIRECTLY BELOW THIS LINE (newest first) -->

- [2026-06-09 | Qwen-Qwen3.5-27B-FP8 | hybrid_linear_attention_dense | gfx942 | gated-delta recompute_w_u_fwd (FLA wy_fast) | varlen prefill-dominated isl/osl/conc=1024/1024/64] **METHOD: optimize `recompute_w_u_fwd` with BK=BV=128 tiling; the real win lives in LARGE-PREFILL cases (T~15-16k → 1.13-1.16x isolated), so this is a STACK-and-compound target on the FLA cluster.** Extract seam = the wy_fast module under `sglang.srt.layers.attention.fla`; the optimized callable rebinds via the overlay (`[overlay] injected ...wy_fast`). **ENGAGEMENT VERIFICATION (positive method, reusable for any FLA/mamba overlay): instrument the candidate kernel with a one-shot stderr banner and grep the server log for `[overlay-mark] <kernel> OPTIMIZED kernel CALLED` — this PROVES the optimized kernel ran on the live serving path (both bench and parity legs) rather than inferring engagement from an e2e wiggle.** Parity held (12/12 greedy temp=0 seed=0 prompts byte-exact ref vs cand). Amdahl: at pct_gpu_time=1.39% the e2e ceiling is ~0.17%, and the tight 2-launch A/B (7 repeats/leg, TP=1, single GPU, back-to-back) measured +0.216% (cand_med 899.940 vs ref_med 897.997) — non-negative, no regression, but below the 0.5% band with overlapping distributions, exactly as Amdahl predicts. Route as carry-forward so it compounds with siblings; the Director's final COMBINED gate decides the headline. The pre-dispatch screen still applies: `pct_gpu × (1 − 1/plausible_iso) < NOISE_BAND_PCT` ⇒ stack-only, don't expect a solo pass.

- [2026-06-09 | Qwen-Qwen3.5-27B-FP8 | hybrid_linear_attention_dense | gfx942 | fp8 a8w8 blockscale dense GEMM (up/gate N=34816, K=5120) | prefill-dominated] **ROUTING: on the FP8 variant the live dense-GEMM dispatcher is the fp8 a8w8 BLOCKSCALE path (`_gemm_a8w8_blockscale_kernel`), and it is THE head lever — ~82% summed GPU time, with ranks 1-3 alone = 38.9% at the three large-N=34816 up/gate prefill shapes.** This is the FP8 analogue of the bf16 `tuned_gemm` head; spend the head/config budget here first. **The rebind seam to drive a candidate GEMM onto the live path is `aiter.ops.triton.gemm_a8w8_blockscale` (overlay rebinds it).** REQUIREMENT learned this run: any candidate GEMM placed on this seam MUST be HIP/CUDA-graph-CAPTURABLE — sglang captures decode batches into CUDA graphs, so a kernel that only runs eager (e.g. a FlyDSL/preshuffle compiler that faults under graph capture on the first decode batch) cannot be measured e2e even when its isolated speedup is large and real. To even bench a non-capturable candidate you must serve with `--disable-cuda-graph`. Prefer a capturable Triton/aiter-DB candidate so it engages under normal graph capture.

- [2026-06-09 | Qwen-Qwen3.5-27B-FP8 | hybrid_linear_attention_dense | gfx942 | full-attention prefill surface after --attention-backend triton | mixed] **ROUTING/MECHANISM: after the cfg `--attention-backend triton` win the bottleneck STRUCTURE is unchanged (fp8 blockscale GEMM still ~82%) but full-attention moves from the non-editable CK paged kernel (ck_tile FmhaBatchPrefillWithPagedKVCache) onto the EDITABLE sglang Triton `_fwd_kernel` (extend/prefill_attention.py, ~0.87% gpu).** So the attention-backend swap not only buys the small scheduling win, it also exposes a source-editable full-attention surface for the kernel track. Confirm the swap is a small-Amdahl/scheduling effect, not a GPU-time redistribution (total GPU time stayed flat ~6111→6124 ms), so order-of-bets is unchanged: the blockscale GEMM head remains the primary lever, the editable Triton FLA/attention cluster is the compound/stack track.

- [2026-06-07 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | Amdahl budgeting for the editable-Triton cluster (gated-delta/FLA/mamba) | prefill-dominated isl/osl/conc=1024/1024/64] **ROUTING: spend the head/config budget on dense GEMM (~81% gpu) first; treat the editable-Triton FLA/mamba kernels as a STACK-and-compound track, not a standalone-e2e-win track.** Measured isolated speedups in this cluster are real and large (chunk_gated_delta_rule_fwd_h 1.18x, chunk_fwd_kernel_o 1.14x, causal_conv1d 1.10x) and engagement is provable via the overlay banner `[overlay] injected module sglang.srt.layers.attention.{fla,mamba}.<mod>` (4 hits/worker) plus the `[OVERLAY_ENGAGED]` marker. But each sits at only ~1-3% gpu time, so the e2e ceiling = `pct_gpu × (1 - 1/iso_speedup)` is ~0.1-0.45% — below the 0.5% noise band by construction in this prefill regime where ~80% gpu is dense GEMM. **Quick-screen rule before dispatching any of these: if `pct_gpu × (1 - 1/plausible_iso) < NOISE_BAND_PCT`, route it as carry-forward-only (stack all siblings, let the Director's final COMBINED gate vs TRUE baseline decide) rather than expecting it to pass a solo gate.** These are still the best EDITABLE targets on this arch; the way to make them count is to combine the whole cluster (chunk_h + chunk_o + conv1d + recompute_w_u + gating) and measure the sum.

- [2026-06-07 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | sglang e2e A/B harness | any] **MEASUREMENT METHOD: pin PORT for every sglang leg and run a tight INTERLEAVED A/B (REF, CAND, REF, CAND...) on a single GPU, then gate on BOTH delta_med>band AND non-overlapping distributions (cand_min>ref_max).** sglang derives `grpc_port = port + 10000` and hard-rejects >65535, so the bench auto port allocator crashes any launch where PORT>55535 (e.g. 59339->69339); always pin PORT to a low value (31337/31537 used). The ~0.5% noise band is real: across clean legs ref/cand medians at ~1551 tok/s overlap routinely, so a delta under the band with overlapping [min,max] is a NULL, not a win — require separation, not just a positive median. Add grpc-port-flake retries (saw 3) into the leg budget.
<!-- format: - [YYYY-MM-DD | model | model_class | gfx | kernel_class | shape_regime] METHOD: how to
     optimize this well (the lever + how to apply + how to verify).
     POSITIVE PRIORS ONLY. Do NOT record "dead-end / rejected / skip X / doesn't work" here — a null
     this run may just mean it wasn't optimized well, and a blocklist wrongly stops future runs from
     trying. Record mechanism facts as positive ROUTING ("optimize X via Y"), never as "X failed".
     Per-run results incl. what didn't move e2e live in the eval-dir final_report.md, not here.
     The (model_class, gfx, shape_regime) tuple is the REUSE KEY for cross-model transfer. -->

- [2026-06-05 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | dense GEMM (up/gate K=5120, down/qkv N=5120) | prefill+decode] **GEMM lever = aiter's per-shape DB + an authored Triton kernel, in the head track.** The live dense-GEMM path is aiter `tuned_gemm.py` (executing hipBLASLt `Cijk_*`), with a clean rebind seam at `aiter.tuned_gemm:gemm_a16w16`. To tune: capture real shapes with `AITER_TUNE_GEMM=1` → `gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>` → deploy `AITER_CONFIG_GEMM_BF16=<tuned.csv>` (no package edit); **verify engagement with `AITER_LOG_TUNED_CONFIG=1` (`is tuned on cu_num` hits >0).** Also author+optimize a Triton GEMM via team_workflow on the same op task dir and let the e2e gate pick the best. Parity-safe (hipBLASLt algorithm swap; tuner `err_ratio<0.05`). Full recipe: `knowledge/aiter_gemm_tuning.md`. Gate every GEMM change STACKED on the current accepted config with the tight interleaved A/B (small real wins count at the 0.5% band). **CONFIRMED 2026-06-08: a bias-correct full-coverage aiter tune (via the `AITER_TUNE_GEMM=1` capture) WINS +2.23% e2e** (1548.9 → 1583.5 tok/s, non-overlapping 5-repeat A/B, 246 `is tuned on cu_num` hits) STACKED on `--attention-backend triton` → ~+6% cumulative vs baseline (1492.7 → ~1583.5). The capture's correct `bias=False`/full coverage is what makes it both ENGAGE and WIN; an earlier ~0/−0.59% reading came from a bias-mismatched/partial tune and is superseded.

- [2026-06-05 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | full-attention prefill (--attention-backend) | mixed] **`--attention-backend triton` is a cheap real win (+~5% e2e) on hybrid-dense gfx942 sglang** (1546.7 → ~1623 tok/s same-session). It also moves the 16 full-attention prefill layers onto an EDITABLE Triton `_fwd_kernel` (extend/prefill_attention.py), exposing an attention surface for the head/kernel tracks. Try it first; parity holds (greedy temp=0; benign bf16 tie-break only).

- [2026-06-05 | Qwen-Qwen3.5-27B | hybrid_linear_attention_dense | gfx942 | gated-delta linear-attn (FLA chunk_gated_delta_rule_fwd) | varlen prefill+decode] **Editable Triton cluster ~9% GPU — optimize via team_workflow (the kernel squad).** It reaches large isolated speedups; to make them count e2e, ensure the win MECHANISM survives varlen serving (buffers/shapes vary per call — prefer kernel-level tiling/scan/dtype wins over per-call CUDA-graph caches keyed on buffer pointers, and watch HIP memory under varlen). A strong editable-kernel target on this arch.

- (seed) [* | dense | gfx942 | dense_gemm | any] On sglang the dense-GEMM live path is aiter `tuned_gemm` → tune the aiter DB (above) and/or author a Triton GEMM; verify engagement via `AITER_LOG_TUNED_CONFIG=1`. Watch the "not found tuned config" log lines — they list the exact untuned shapes to target.
