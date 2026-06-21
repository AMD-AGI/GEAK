---
key: int4_w4a16 fused-MoE grouped GEMM · gfx942/MI300X · vLLM
type: lever
confidence: ★★★
effect: per-shape Triton config tune (winner_kind=env, ZERO HBM) → +16% e2e VERIFIED (iso ~1.59×); 6 re-confirms on Kimi-K2.6 (decode-dominated isl/osl/conc=8192/1024/64)
last_seen: 2026-06-21
---
# int4 W4A16 fused-MoE grouped GEMM head → the memory-free vLLM config-tune lever
- lever (try FIRST on an int4 MoE model): vLLM ships NO tuned Triton config for an unseen
  `(E,N,int4_w4a16)` shape, so the expert grouped-GEMM (`fused_moe_kernel_gptq_awq`) — often the single
  biggest GPU-time chunk (~50–57% on Kimi-K2.6, per-rank E=384, N=moe_intermediate//TP=256, K=7168,
  topk=8, group_size=32) — falls back to a SLOW default tile. Tuning that one config is the #1 e2e lever.
- apply: follow `SKILL_DIR/knowledge/moe_int4_tuning.md` — derive the per-rank shape from model config
  (+TP), sweep per M-bucket against the faithful `fused_experts` int4_w4a16 path (`override_config`,
  parity rel<1e-2), write `E=…,N=…,dtype=int4_w4a16.json`, deploy `winner_kind=env`
  `VLLM_TUNED_CONFIG_FOLDER=<dir>` paired with `--max-num-batched-tokens ≈2·ISL` (clamp 8192..32768).
  Tile/scheduling only → parity holds by construction, ZERO extra HBM.
- verify: grep the server log — REF leg prints `Using default MoE config` (tuned-loaded 0); CAND leg
  prints `Using configuration from .../E=384,…,dtype=int4_w4a16.json` (tuned-loaded 1). Then confirm the
  e2e gate (non-overlapping A/B), not just isolated.
- caution: **prefer this over an fp8/quant REWRITE of the same op.** An fp8-fold caches a 2nd fp8 weight
  copy and, at memory parity, OOMs at KV-cache init (op-level ~1.5× but e2e-undeployable — the Integrator
  rejects it `mem_footprint_starves_kv`). Only pursue fp8 author route when `ENABLE_FP8=true` AND it
  passes the memory-footprint gate.
- caution: **confirm the run's baseline did NOT already bank the config** — if baseline server.log already
  shows `Using configuration from …int4_w4a16.json`, the lever is consumed (no fresh e2e win this round).
- source: GEAK/e2e_workflow/knowledge/moe_int4_tuning.md (+16.4% e2e, 514.55→598.98 tok/s, GSM8K 0.965→0.973).
- source: perf_knowledge/case_studies/by_model/kimi_k2.6_int4_moe_mi300x.md; 2026-06-21 A/B ref med=461.3 vs
  cand med=535.4 = +16.05% e2e non-overlapping (cand_min>ref_max), Director-verified iso 1.59× (6 re-confirms).
