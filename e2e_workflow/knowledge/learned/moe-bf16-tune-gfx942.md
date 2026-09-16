---
key: bf16 fused-MoE grouped GEMM · gfx942+gfx950 · vLLM AND sglang
type: lever
confidence: ★★★
confirms: 6
effect: TWO independent levers on the same op — (A) per-shape Triton config tune (winner_kind=env, ZERO HBM) → iso 1.01–1.66× per M-bucket, serving-weighted 1.10–1.40×; e2e VERIFIED +7.01% (Mixtral-8x7B gfx950 TP8, Director-validated, byte-exact) — above the +3.37% Amdahl ceiling once bundled with `--max-num-batched-tokens 8192`. PORTS TO SGLANG unchanged (env is `SGLANG_MOE_CONFIG_DIR`, same `E=<E>,N=<N>,device_name=<dev>.json` name under a `configs/triton_<ver>/` subdir): serving-weighted 1.239x byte-exact on Gemma-4-26B-A4B TP2 gfx950. The lever RE-FIRES on an already-tuned config: a second pass adding BLOCK_SIZE_K=256 + the gfx launch knobs (`waves_per_eu`, `matrix_instr_nonkdim`) bought a further serving-weighted 1.043× / 1.06–1.14× on the small-M buckets, byte-exact. (B) the Tier-C WHOLE-FILE Triton overlay of the modular fused-MoE module is a SEPARATE, stackable win: +6.15% e2e byte-exact on Mixtral-8x7B gfx950 vLLM TP8 at a 35% head — and the gain came from the peripheral launch chain the same file owns, NOT from the head GEMM (per-launch time unchanged in situ).
last_seen: 2026-08-21
---
# bf16 fused-MoE grouped GEMM → the memory-free config-tune lever (analog of the int4 card)

- path: (1) check whether a tuned config ships for this `(E,N,device)` — vLLM ships none for unseen
  bf16 MoE shapes on gfx942/gfx950, so the expert grouped-GEMM falls back to a slow default tile
  (server log: `Using default MoE config`). (2) Sweep per M-bucket and write the JSON. (3) Deploy via
  `VLLM_TUNED_CONFIG_FOLDER`, paired with `--max-num-batched-tokens ≈2·ISL` (clamp 8192..32768).
  Try this FIRST on bf16 MoE models, not just int4 — it is the same mechanism with `dtype=None`, so
  the lookup filename is `E=<E>,N=<N>,device_name=<dev>.json` (no `dtype=` segment).
- expected gain: iso 1.01–1.66× per bucket (large-M prefill buckets are the big ones — M=8192 hit
  1.658× with BM256/BN256/BK128/w8), serving-weighted 1.10–1.40×, banking +7.01% e2e at ~35% MoE head
  share. ZERO extra HBM, so it sails the mem_footprint gate.
- apply: adapt `SKILL_DIR/knowledge/gemm_tuning/moe_int4_tuning.md` to bf16 — dense weights
  w1[E,2N,K]/w2[E,K,N] (no scales/quant_config), the model's own activation (GELU_TANH for Gemma-4,
  the default SILU for Mixtral), sweep against `fused_experts`+`override_config` (parity rel<1e-2).
  On vLLM 0.26.0 `fused_experts` has no `inplace` kwarg — drop it. Tile-only, so parity holds by
  construction. **N is per-TP-rank (`moe_intermediate//TP`)** — re-derive from model config + serving TP.
- verify: `get_config_file_name(E,N,None,None)` gives the target filename; confirm nothing ships for
  this device first. Engagement: REF log prints `Using default MoE config`, CAND prints
  `Using configuration from …E=<E>,N=<N>,…json` with zero default-config lines.
- caution: **drop any bucket whose tuned tile is not >1.0×** and keep vLLM's default there — a
  regressing tile in the JSON slows that bucket (Gemma-4 M=1024 came out 0.98× and was dropped). The
  per-bucket subprocess sweep is the trustworthy iso number; a merged single-process run inflates the
  default baseline and understates the win. Do not treat the serving-weighted Amdahl ceiling as a cap
  — the measured +7.01% exceeded +3.37% because device-time under-counted the MoE; trust the
  byte-exact e2e A/B, and keep the batched-tokens flag paired with the config folder on re-deploy.
- **the config JSON is not just tiles — it is splatted into the Triton launch.** vLLM's
  `invoke_fused_moe_triton_kernel` does `fused_moe_kernel[grid](..., **config)`, so ANY extra key in
  the JSON reaches the compiler: `waves_per_eu`, `matrix_instr_nonkdim` (and `num_warps/num_stages`).
  A first-pass sweep that only varies BM/BN/BK/GROUP_M/warps/stages therefore leaves a second, cheaper
  win on the table — **an ALREADY-ACCEPTED tuned config is still tunable**, and re-tuning it is a
  legitimately NEW candidate (not a re-proposal of the live one). Also widen BLOCK_SIZE_K to 256: on
  gfx950/MI355 that alone carried most of the second-pass win. `kpack` is DEAD on gfx950 (the Triton
  AMD backend warns and force-overrides it to 1) — drop it from the grid.
- when re-tuning on top of a live config, make the ACCEPTED config the per-bucket baseline (not vLLM's
  default), and re-verify the winners in a FRESH process with fresh input draws before shipping: the
  sweep's own bests are ~1 noise-band optimistic and 3 of 8 buckets flipped to <1.0× on re-measure
  (identical-config control pairs read 0.991–1.005×, i.e. a ±0.8% floor at these per-launch sizes).
- also: an editable in-tree Triton MoE exists (`kernel_src/fused_moe.py`, seam = fused_moe/grouped_gemm
  dispatcher) → Tier-C `route=rewrite`. Tier-A fused-backend swaps LOSE to a tuned Triton on bf16
  Mixtral/gfx950: vLLM `AiterExperts` (and raw `aiter.fused_moe`, which logs `using 2stage default` —
  aiter ships no tuned_fmoe row for this shape either) came in serving-weighted 0.92× (wins only the
  M=64 bucket at 1.08×), and aiter's FlyDSL MoE is quant-only — `flydsl_moe_stage1` raises
  `Unsupported stage1 dtype combination: a_dtype=bf16, b_dtype=bf16`, so a flydsl author lane for a
  bf16 fused MoE has no seed to reuse (it imports fine — the gap is capability, not presence).
- **lever B — the Tier-C whole-file overlay, and WHERE ITS WIN ACTUALLY COMES FROM.** On vLLM 0.26 the
  live bf16 MoE path is the MODULAR one (`UnquantizedFusedMoEMethod` → `TritonExperts.apply` →
  `invoke_fused_moe_triton_kernel`); the legacy `fused_experts_impl` records ZERO calls, so hook the
  MODULE, not that function. Bind as a whole-file module swap of the fused_moe module (it registers a
  custom op at import, so a side-by-side second copy fails to re-define it). Measured +6.15% e2e,
  byte-exact — but the reprofile shows the head `fused_moe_kernel` per-launch time UNCHANGED (decode
  per-launch time identical before and after, HBM utilisation 0.64 of roof both times): the embedded per-(shape,M-bucket)
  tile table and cache-modifier hints bought nothing at the memory roof. What actually moved was the
  PERIPHERAL chain inside the same file — the C++ `moe_align_block_size_small_batch_expert` (3.2% GPU)
  was replaced by a Triton parallel align kernel (2.0%), and total GPU time in the window fell 4%. So on
  an already-tuned bf16 MoE, size the whole-file overlay by the LAUNCH CHAIN it can collapse, not by the
  grouped-GEMM's roofline headroom (see [[editable-triton-cluster-amdahl]]).
- **sglang port (same lever, different env).** `fused_moe_triton_config.get_moe_configs` reads
  `$SGLANG_MOE_CONFIG_DIR/configs/triton_<X_Y_Z>/E=<E>,N=<N>,device_name=<dev>.json` (N is
  `w2.shape[2]` = the PER-RANK intermediate; the version subdir is mandatory and the dir REPLACES the
  shipped tree, so build the overlay by symlinking the shipped `configs/` and adding only your file).
  Nothing ships for an unseen `(E,N,MI355X)` → both grouped GEMMs run `get_default_config`
  ("Using default MoE kernel config"). Bench it in-process by binding a UT candidate that applies the
  same nearest-key rule under `triton_utils.override_config` — the env is process-global and would
  otherwise move BOTH legs of the A/B.
- **TUNE A FULL M LADDER, not just the captured buckets.** The lookup is nearest-key
  (`min(keys, key=|k-M|)`), so a table with only {1,64,8192,16384} sends every live M in ~128..4128 to
  the 64-row's tiny tile — a silent regression on chunked-prefill/mixed batches. Cover
  {1,8,64,128,256,512,1024,2048,4096,8192,16384}. Buckets the capture has no record for are tuned by
  SLICING the first m token rows out of a real prefill capture (real routing rows, real activations —
  never synthesized routing).
- **the separate `_down.json` second table is a DEAD END here** (sglang only; vLLM has no analog).
  gemm2 can carry its own tile (only BLOCK_SIZE_M is forced to match the up config, one shared sort),
  but an independent 30-point down sweep bought 1.0098x at M=64 and 1.0005x/0.993x at M=16384/8192 —
  inside noise. Ship the up table alone; when `_down.json` is absent sglang reuses the up config for
  gemm2 (`down_config or config`), which is exactly what the override-based A/B measures.
- **caution (aiter fused-MoE cannot bind a non-128-multiple intermediate).** Tier-A swap to
  `aiter.fused_moe.fused_moe` (QuantType.No, ActivationType.Gelu, gate_mode=interleave) at
  E=128/N=352/K=2816 dies in `ck_moe_stage1_fwd` with `RuntimeError: wrong! device_gemm with the
  specified compilation parameters does not support this GEMM problem` — the CK MoE instances have no
  tile for inter_dim=352. Also re-confirms flydsl MoE is quant-only
  (`flydsl_moe_stage1(a_dtype=bf16,b_dtype=bf16) -> ValueError Unsupported stage1 dtype combination`),
  so on a bf16 MoE the ONLY Tier-C author language is triton (route=rewrite, the in-tree
  `fused_experts_impl` is editable).
- **where the tune does NOT reach: the dominant decode bucket.** On Gemma-4-26B-A4B TP2 the
  serving weight is 53% decode M=64 / 47% prefill M=16384, and M=64 tops out at 1.03x (best tile is
  barely better than the default) while prefill hits 1.57-1.62x and M=1 1.30x. So the config tune
  banks the prefill half and leaves the decode half for the Tier-C rewrite — size the author lane's
  target on the decode bucket, not on the headline weighted number.
- source (sglang): 2026-08-21 Gemma-4-26B-A4B-it (E=128, N=352 per-rank, K=2816, topk=8, gelu gated +
  gate/up INTERLEAVED, sglang 0.5.17, TP=2, gfx950/MI355X, MoE 16.27% GPU, ISL 8192/OSL 1024/conc 64):
  iso 1.2959x (M=1) / 1.0300x (M=64) / 1.5715x (M=8192) / 1.5992x (M=16384), serving-weighted 1.2390x,
  ALL byte-exact (max_rel_err 0.0 on 6 eager + 18 random-value draws + 4 graph-replay cases) through
  the immutable UT; Amdahl ceiling +3.24% e2e; engagement verified ("Using MoE kernel config from
  <overlay>"); e2e gate pending. drivers: `EVAL_DIR/config/tune_moe_bf16_sglang.py`,
  `tune_moe_down_sglang.py`, `build_moe_config_overlay.py`, `moe_tuned_candidate.py`,
  `moe_aiter_candidate.py`.
- source: 2026-07-05 Gemma-4-26B-A4B (E=128,N=704,K=2816, topk=8, gelu_tanh, vLLM 0.21.0, TP=1,
  gfx942, MoE 40.66% GPU; iso M=1 1.88× … M=8192 1.09×, e2e gate pending);
  2026-08-13 + 2026-08-17 Mixtral-8x7B-Instruct-v0.1 (E=8,N=1792,K=4096, topk=2, silu, vLLM 0.26.0,
  TP=8, gfx950/MI355 OAM, MoE 35.21%/38.55% GPU; no shipped MI355 config → default fallback; e2e
  Director-validated_win **+7.0%**, byte-exact 12/12);
  2026-08-20 same model/box, WARM-STARTED on that accepted config (MoE still 33.26% GPU after it):
  second-pass knob tune iso 1.0615/1.1224/1.1375/1.1166× at M=16/32/64/128, 1.015× at M=1024,
  serving-weighted 1.0432× (geomean 1.054), all buckets byte-exact, Amdahl ceiling +1.40% e2e; e2e gate
  pending. drivers: `EVAL_DIR/config/tune_moe_bf16.py`, `EVAL_DIR/config/tune_moe_bf16_v2.py`
  (+ `verify_moe_tuned_v2.py`, `moe_fused_bakeoff.py` for the fused-backend Tier-A).
- source (lever B): exp/e2e_*Mixtral-8x7B*/ 2026-08-21 (E=8, N=1792 per-rank, K=4096, topk=2, silu,
  vLLM 0.26.0, TP=8, gfx950, ISL/OSL 1024/1024 conc 64, head 35.01% GPU): whole-file Triton overlay,
  iso 1.231× through the immutable UT, e2e **+6.15%** with strictly disjoint
  steady-state repeats, under the +7.04% Amdahl ceiling, byte-identical greedy output 12/12; overlay
  engagement proven by the injection banner in all 10 server processes.
