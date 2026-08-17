---
key: bf16 fused-MoE grouped GEMM · gfx942+gfx950 · vLLM
type: lever
confidence: ★★★
confirms: 3
effect: per-shape Triton config tune (winner_kind=env, ZERO HBM) → iso 1.01–1.66× per M-bucket, serving-weighted 1.10–1.40×; e2e VERIFIED +7.01% (Mixtral-8x7B gfx950 TP8, Director-validated, byte-exact) — above the +3.37% Amdahl ceiling once bundled with `--max-num-batched-tokens 8192`.
last_seen: 2026-08-17
---
# bf16 fused-MoE grouped GEMM → the memory-free vLLM config-tune lever (analog of the int4 card)

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
- also: an editable in-tree Triton MoE exists (`kernel_src/fused_moe.py`, seam = fused_moe/grouped_gemm
  dispatcher) → Tier-C `route=rewrite`; flydsl `flydsl_moe_stage1/stage2` DOES import on the gfx950
  image (unlike the mxfp4 case) → `route=author`. Emit both and let the e2e gate pick.
- source: 2026-07-05 Gemma-4-26B-A4B (E=128,N=704,K=2816, topk=8, gelu_tanh, vLLM 0.21.0, TP=1,
  gfx942, MoE 40.66% GPU; iso M=1 1.88× … M=8192 1.09×, e2e gate pending);
  2026-08-13 + 2026-08-17 Mixtral-8x7B-Instruct-v0.1 (E=8,N=1792,K=4096, topk=2, silu, vLLM 0.26.0,
  TP=8, gfx950/MI355 OAM, MoE 35.21%/38.55% GPU; no shipped MI355 config → default fallback; e2e
  Director-validated_win 10753→11508 tok/s, byte-exact 12/12).
  driver: `EVAL_DIR/config/tune_moe_bf16.py`.
