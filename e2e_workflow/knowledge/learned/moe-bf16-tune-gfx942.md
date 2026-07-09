---
key: bf16 fused-MoE grouped GEMM · gfx942/MI300X · vLLM
type: lever
confidence: ★★
effect: per-shape Triton config tune (winner_kind=env, ZERO HBM) → isolated 1.06-1.25× per M-bucket (decode M32-128 ~1.2×, prefill M6202 ~1.06-1.11×, M1 up to 1.88×); e2e gate PENDING Integrator
last_seen: 2026-07-05
---
# bf16 fused-MoE grouped GEMM head → the memory-free vLLM config-tune lever (analog of the int4 card)
- lever (try FIRST on a bf16 MoE model too, not just int4): vLLM ships NO tuned Triton config for an
  unseen `(E,N)` bf16 fused-MoE shape on gfx942, so the expert grouped-GEMM falls back to a SLOW default
  tile (server log: `Using default MoE config`). Tuning that one config is a memory-free e2e lever exactly
  like the int4 case — the same `VLLM_TUNED_CONFIG_FOLDER` mechanism, just `dtype=None` (dense bf16) so the
  lookup filename is `E=<E>,N=<N>,device_name=<dev>.json` (NO `dtype=` segment).
- apply: adapt `SKILL_DIR/knowledge/gemm_tuning/moe_int4_tuning.md` to bf16 — dense bf16 weights
  w1[E,2N,K]/w2[E,K,N] (no scales/quant_config), `activation=MoEActivation.GELU_TANH` for a gelu_tanh MoE,
  sweep per M-bucket against `fused_experts`+`override_config` (parity rel<1e-2), write the JSON, deploy
  `winner_kind=env VLLM_TUNED_CONFIG_FOLDER=<dir>` + `--max-num-batched-tokens ≈2·ISL` (clamp 8192..32768).
  Tile-only → parity holds by construction, ZERO extra HBM (sails the mem_footprint gate).
- verify: `get_config_file_name(E,N,None,None)` gives the target filename; confirm the pre-tune shape has
  no shipped config for this device (only NVIDIA/fp8 `E=128,N=704` existed on the Gemma-4 run). Engagement:
  REF server.log prints `Using default MoE config`; CAND prints `Using configuration from …E=<E>,N=<N>,…json`.
- caution: **drop any bucket whose tuned tile is not >1.0× (keep vLLM's default there).** On Gemma-4
  M=1024 came out 0.98× in the sweep and was dropped; a regressing tile in the JSON would slow that bucket.
- caution: same JIT-warm-baseline optimism as the int4 card — the per-bucket subprocess sweep is the
  trustworthy iso number; a merged single-process run inflates the default baseline (understates the win).
- caution: **N is per-TP-rank (moe_intermediate//TP)** — re-derive from model config + serving TP; on the
  Gemma-4 run TP=1 so N=moe_intermediate=704 directly.
- source: 2026-07-05 Gemma-4-26B-A4B (gemma4_text MoE, E=128, N=704, K=2816, topk=8, gelu_tanh), vLLM
  0.21.0, TP=1, gfx942/MI300X, ISL/OSL=1024, pct_gpu_time(moe)=40.66. Fresh per-bucket subprocess sweep:
  M=1 1.88× / M=32 1.25× / M=64 1.19× / M=128 1.19× / M=6202(prefill) 1.06-1.11× / M=8192 1.09×, all
  rel_err<1e-2, ZERO HBM. driver: EVAL_DIR/config/tune_moe_bf16.py. e2e gate pending Integrator.
- also: editable in-tree Triton MoE exists (triton_moe.py) → Tier-C `route=rewrite`; flydsl available
  (0.1.4) → `route=author`. Both emitted in author_plan to let the e2e gate pick best of {tuned, authored}.
