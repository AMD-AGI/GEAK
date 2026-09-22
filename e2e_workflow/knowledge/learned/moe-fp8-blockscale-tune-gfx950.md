---
key: fp8_w8a8 block-scale fused-MoE grouped GEMM · gfx950 · vLLM
type: lever
confidence: ★★
confirms: 4
effect: TWO different seams depending on whether AITER MoE is on. (A) AITER ON (`VLLM_ROCM_USE_AITER[_MOE]=1`, seam `aiter.fused_moe:fused_moe`): the aiter asm 1-stage kernel runs UNTUNED (`[fused_moe] using 1stage default` for every token tier — the shipped `tuned_fmoe.csv` has ZERO `per_1x128`/cu_num=256 rows), so a per-token-tier row in an `AITER_CONFIG_FMOE` CSV picking a bigger asm tile is a free env win: iso 1.11–1.31×/bucket, serving-weighted 1.155× (Qwen3.5-122B-A10B-FP8 TP2, 25.7% head → +3.44% Amdahl ceiling), ZERO HBM, e2e pending. (B) AITER OFF (Triton seam): per-shape Triton config tune (winner_kind=env, ZERO HBM) → iso 1.02–1.16× per M-bucket, serving-weighted ~1.03× (decode M64 1.026×, prefill M8192 1.041×). Same VLLM_TUNED_CONFIG_FOLDER mechanism as the int4/bf16 MoE cards, dtype segment = fp8_w8a8 + block_shape. An authored Triton rewrite of `fused_experts_impl` (Tier-C) beat the env-tune bake-off (iso 1.034×). e2e transfer did NOT clear the noise band at the FINAL gate: Director same-session A/B = +0.16% (1.0016×), ranges OVERLAP → validated_no_win, byte-exact parity 12/12. The lever ENGAGES (decode-bucket rebind fired on both TP workers) but at a 21.4% head with iso ~1.03× the Amdahl ceiling (~0.6%) is inside serving noise. 4th confirm SIZES THE SEAM SWAP ITSELF: flipping A↔B (Triton MoE → aiter fused asm fmoe) leaves the GEMM cost EXACTLY unchanged (one fused launch per decode layer costs what the two Triton launches did; 97.3% of the HBM roofline in BOTH legs) — 100% of the MoE-side e2e win is the peripheral quant/routing CHAIN the fused kernel absorbs (**−77% GPU time** over six kernels), so size an aiter-MoE swap by the launch chain it deletes, never by expected GEMM headroom.
last_seen: 2026-08-23
---
# fp8_w8a8 block-scale fused-MoE → the memory-free vLLM config-tune lever (fp8 analog of int4/bf16 cards)

## (A) when AITER MoE is ON: tune the aiter fmoe DB, not the Triton config folder
- FIRST establish which seam is live. `VLLM_ROCM_USE_AITER[_MOE]=1` routes `torch.ops.vllm.rocm_aiter_fused_moe`
  → `aiter.fused_moe:fused_moe`; the Triton `VLLM_TUNED_CONFIG_FOLDER` lever (section B) is then DEAD CODE.
  Probe: `grep '\[fused_moe\] using .*stage' server.log`. `using 1stage default for (gfx950, 256, <token>, ...)`
  = an UNTUNED asm kernel picked by heuristic = the headroom. `using 1stage (kernelName1='...')` = a tuned row hit.
- lever: `AITER_CONFIG_FMOE=<csv>` (aiter reads it via `AITER_CONFIGS.AITER_CONFIG_FMOE_FILE`). Lookup is an
  EXACT match on `_INDEX_COLS` = (gfx, cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype,
  q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1), where `token = get_padded_M(M)` = nextPow2 below
  32768 → write ONE ROW PER POW2 TIER you actually serve. Build the CSV by APPENDING your rows to the merged
  `/tmp/aiter_configs/tuned_fmoe.csv` so other models' shipped rows are not lost (env replaces, not merges).
- candidate space = the asm manifest `<get_asm_dir()>/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_silu.csv`; filter
  `smf==0` and `inter_dim % subGU_n == 0`, set `block_m = subGU_m`, `run_1stage=1`, `xbf16=0`, `flat=0`.
- measured on E=256/N=512/K=3072/topk=8/silu/fp8-block[128,128]/gfx950-256CU (vs the live default):
  token 1 → keep DEFAULT (2stage; no row); token 64 → `vs_pf2_silu_16x128` bm=16 **1.12×**;
  token 128/256 → default already best (skip); token 512/1024/2048 → `vs_ps_silu_64x256` bm=64 1.07–1.11×;
  token 4096 → **1.31×**; 8192 → **1.29×**; 16384 → 1.22×. Big tile wins as soon as M ≥ 512; the small-tile
  16x128 that wins decode M=64 REGRESSES prefill hard (0.60× at 8192) → the win is per-tier, never one row.
- `blockscaleBf16` (`xbf16=1`) twins of the same tiles are 1.00–1.03× at best and FAIL the correctness gate
  (max_rel_err 0.95–1.49 vs 1.0) → lossy upcast path, reject on parity, not on speed.
- caution (also verify): `aiter.jit.core.AITER_CONFIGS.get_config_file` is `@functools.lru_cache`d, so the FIRST
  resolution in a process STICKS. Any in-process A/B that flips `AITER_CONFIG_FMOE` between candidates must call
  `AITER_CONFIGS.get_config_file.cache_clear()`, `aiter.fused_moe.cfg_2stages = None` and
  `get_2stage_cfgs.cache_clear()` — otherwise every candidate silently re-runs the default and all bake-off ms
  come out identical (a false "no headroom"). A fresh server process is unaffected.
- caution (also verify): the stock offline tuner `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` CANNOT tune
  the asm 1-stage path on this image — every 1stage task dies in `mp_tuner.work_group` (`tuple(data[k] for k in
  args[0])`, mp_tuner.py:229) because the task passes INTEGER data indices while `generate_data_1stage` returns a
  NAME-keyed dict → "Critical error in work_group: 0" ×N then "no valid candidate found". Don't debug it: race the
  manifest kernels directly THROUGH the live seam with a one-row CSV per candidate (that also measures exactly what
  deployment gets). Also: no CK-2stage candidates are reachable this way without valid `kernelName1/2` strings, and
  the shipped CSV has none for per_1x128/cu_num=256.

## (B) when the Triton seam is live: the `VLLM_TUNED_CONFIG_FOLDER` config tune

- path: same as `moe-bf16-tune` / `moe-int4-w4a16-tune` but for fp8 block-quant. (1) check whether a
  tuned config ships for `(E,N,device,fp8_w8a8,block_shape)` — vLLM ships NONE for unseen fp8-blockscale
  MoE shapes on gfx950/MI355 (verified 0 configs match `*MI355*fp8_w8a8*block_shape*`), so the expert
  grouped-GEMM falls back to the slow default tile (`Using default MoE config`). (2) Sweep per M-bucket
  against `fused_experts`+`override_config` with fp8 weights + block scales, parity rel<1e-2. (3) Deploy
  `VLLM_TUNED_CONFIG_FOLDER`, pair with `--max-num-batched-tokens ≈2·ISL` (clamp 8192..32768).
- lookup filename: `get_config_file_name(E, N, "fp8_w8a8", [128,128])` →
  `E=<E>,N=<N>,device_name=<dev>,dtype=fp8_w8a8,block_shape=[128,128].json`. N = moe_intermediate//TP.
- fp8-SPECIFIC constraint (differs from bf16/int4 sweep): the block-scale kernel requires
  `BLOCK_SIZE_K % block_k == 0` and `BLOCK_SIZE_N % block_n == 0` → pin BLOCK_SIZE_K=128, BLOCK_SIZE_N∈{128,256}.
  Build the quant_config via `fp8_w8a8_moe_quant_config(w1_scale,w2_scale,block_shape=[128,128])`; weights
  float8_e4m3fn, scales float32 shaped [E,ceil(2N/128),ceil(K/128)] / [E,ceil(K/128),ceil(N/128)].
- expected gain: iso 1.02–1.16× per bucket (mid buckets M128/256 biggest at ~1.15×; decode M64 1.026×,
  prefill M8192 1.041×), serving-weighted ~1.03×. ZERO extra HBM → sails the mem_footprint gate.
  Naive Amdahl ceiling ~0.6% at a 21.4% head, BUT decode is graph-hidden/under-counted (profiled decode
  share 0.00 → floored 0.30) so measured e2e may exceed it (cf. bf16 card +7.01% > +3.37% ceiling).
- also: editable in-tree Triton MoE present (`kernel_src/fused_moe/fused_moe.py`, hot `fused_moe_kernel`
  @triton.jit + `fused_experts_impl`) → Tier-C `route=rewrite`; flydsl grouped-MoE primitives import on
  this gfx950 image (`aiter.ops.flydsl.flydsl_moe_stage1/stage2`, is_flydsl_available=True) → `route=author`.
  aiter ALSO ships a native fp8 block-scale fused MoE (`aiter.fmoe_fp8_blockscale_g1u1`) reachable via the
  vLLM `VLLM_ROCM_USE_AITER[_MOE]` backend swap — a separate Tier-A candidate the Integrator can A/B
  (mutually exclusive with the Triton config env, since aiter MoE bypasses the Triton seam).
- caution (also verify flydsl viability before routing it): for fp8-[128,128]-block MoE the aiter FlyDSL
  path did NOT compile on this gfx950 image — the high-level `flydsl_moe_stage1/2` wrapper accepts only
  b_dtype∈{fp4,fp8 MXFP8 per-32 e8m0, bf16xint4} and raises ValueError on bf16xbf16, and the
  precision-preserving fallback (dequant fp8-block→bf16/fp16 then `compile_moe_gemm1/2`) hits an internal
  DSL compiler bug (`UnboundLocalError 'a0'` in the non-int4 prefetch pipeline, `moe_gemm_2stage.py`,
  reproduced across tiles). So `is_flydsl_available=True` (imports OK) is NOT sufficient — verify the
  actual dtype path compiles; for [128,128]-block fp8 MoE prefer the Triton rewrite. flydsl would need a
  fixed non-int4 `moe_gemm_2stage` build or an MXFP8 requant path validated to the tol.
- caution (also verify): a milestone interleaved A/B here showed +1.36% NON-overlapping byte-exact for
  the authored Triton rewrite, but the Director SAME-SESSION A/B collapsed it to +0.16% (overlapping
  ranges) = validated_no_win. On a decode-bound serving run always trust the Director same-session A/B
  over the milestone A/B: a milestone win at a ~20% head with iso ~1.03× can be entirely serving noise
  once re-measured against a fresh same-session baseline (base median rose from 2405.9 warm-start to
  2451.5 same-session — most of the apparent gain was baseline drift). Byte-exact parity is NOT evidence
  of a throughput win.
- 🆕 confirm (2026-08-23, SAME model/box, TP2 gfx950 vLLM 0.26, ISL/OSL 1024/1024 conc 64) — **what the
  A→B seam swap is actually worth, from a post-swap trace, and it is not the GEMM.** After
  `VLLM_ROCM_USE_AITER=1` landed as the accepted config, a full-trace name scan shows the ENTIRE Triton
  MoE chain absent (`fused_moe_kernel`, `per_token_group_quant_8bit_kernel`,
  `silu_and_mul_per_block_quant_kernel`, `moe_align_block_size_kernel`,
  `count_and_sort_expert_tokens_kernel`, `moe_sum_vec_kernel` — all gone), and the accounting splits:
  · **peripheral chain −77% GPU time**, absorbed into two aiter kernels (`dynamic_per_group_scaled_quant`
    + `topkGatingSoftmax`) = the whole MoE-side saving;
  · **grouped GEMM itself flat (1.000×)**: aiter's fused asm fmoe does a decode layer in ONE launch for
    exactly what the two Triton launches it replaced cost together. Both legs sit at **97.3% of the HBM
    roofline**, so there was never GEMM headroom to recover on either seam.
  Consequences worth carrying: (a) section (A)'s `AITER_CONFIG_FMOE` tile tune is tuning a kernel already
  at the memory wall — its iso 1.11–1.31×/bucket must be re-read as tile/occupancy recovery at specific
  token tiers, not roofline headroom, and its e2e ceiling should be sized conservatively; (b) the fmoe
  head's %GPU ROSE 21.6% → 25.5% while its absolute cost was flat — pure denominator illusion, read
  per-launch us; (c) the post-swap #2 head is aiter TP all-reduce at 23.9% (comm swapped RCCL →
  `reduce_scatter_cross_device_store`, flat at the interconnect roof), i.e. a CONFIG lever, not a rewrite target.
- source: exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-23 (4th confirm: post-AITER-swap trace decomposition
  of the MoE seam — chain −77% GPU time, GEMM flat at 97.3% of the HBM roof; round bench +21.3%
  driven by the chain + the dense GEMM, not by the MoE GEMM).
- source: exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-19..08-20 (E=256, N=512, K=3072, topk=8, silu, fp8
  block[128,128], vLLM 0.26.0, TP=2, gfx950/MI355 OAM, MoE 21.4% GPU; no shipped config → default
  fallback; iso per-bucket 1.015–1.158×, authored iso 1.034×; Director same-session +0.16% (1.0016×),
  validated_no_win, byte-exact 12/12).
  driver: `config/tune_moe_fp8_blockscale.py`; tuned artifact under
  `config/moe_tuned_fp8/E=...,N=...,dtype=fp8_w8a8,block_shape=[128,128].json`.
- source (A, aiter seam): exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-20 warm-start run (same box/model as above but
  with AITER MoE ON, MoE head 25.7% GPU). driver `config/fmoe_tune/bakeoff_fmoe_cfg.py` (races asm-manifest
  kernels through the live seam on the frozen captured routing), artifact
  `config/fmoe_tune/tuned_fmoe_qwen35_122b.csv`, engagement re-verified per tier
  (`using 1stage (kernelName1='..._vs_ps_silu_64x256E')`). iso serving-weighted 1.155×; e2e NOT yet gated.
