---
key: int4_w4a16 fused-MoE grouped GEMM · gfx942/MI300X · vLLM
type: lever
confidence: ★★★
effect: per-shape Triton config tune (winner_kind=env, ZERO HBM) → +11-18% e2e VERIFIED (iso ~1.25-1.6×); 10 re-confirms on Kimi-K2.6 (TP=8 N=256 and TP=4 N=512; isl/osl/conc=8192/1024/64)
last_seen: 2026-06-23
---
# int4 W4A16 fused-MoE grouped GEMM head → the memory-free vLLM config-tune lever
- lever (try FIRST on an int4 MoE model): vLLM ships NO tuned Triton config for an unseen
  `(E,N,int4_w4a16)` shape, so the expert grouped-GEMM (`fused_moe_kernel_gptq_awq`) — often the single
  biggest GPU-time chunk (~50–57% on Kimi-K2.6, per-rank E=384, N=moe_intermediate//TP=256, K=7168,
  topk=8, group_size=32) — falls back to a SLOW default tile. Tuning that one config is the #1 e2e lever.
- apply: follow `SKILL_DIR/knowledge/gemm_tuning/moe_int4_tuning.md` — derive the per-rank shape from model config
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
- caution: **measure each bucket in its OWN subprocess (matches the live server) — the in-process
  immutable unittest under-reports.** Running all M-buckets in one process lets Triton's JIT cache make
  the default-tile `run(None)` baseline warm/fast (the unittest reported geomean ~1.01× iso, M=8192/16384
  ~1.0×). Isolated re-measure of the SAME tuned cfg = M=8192 1.62×, M=16384 1.51× (default slow-fallback
  7.54/14.84ms → tuned 4.67/9.82ms, rel_err=0). Trust the per-bucket subprocess sweep, not the merged run.
- caution: **confirm the run's baseline did NOT already bank the config** — if baseline server.log already
  shows `Using configuration from …int4_w4a16.json`, the lever is consumed (no fresh e2e win this round).
- source: GEAK/e2e_workflow/knowledge/gemm_tuning/moe_int4_tuning.md (+16.4% e2e, 514.55→598.98 tok/s, GSM8K 0.965→0.973).
- source: perf_knowledge/case_studies/by_model/kimi_k2.6_int4_moe_mi300x.md; 2026-06-21 A/B ref med=461.3 vs
  cand med=535.4 = +16.05% e2e non-overlapping (cand_min>ref_max), Director-verified iso 1.59× (6 re-confirms).
- source: 2026-06-22 re-derive on Kimi-K2.6 TP=8 (eval e2e_..._20260621T232601Z): same shape E=384/N=256/
  K=7168/gs32/topk8, baseline 501.7 tok/s + 'Using default MoE config' (lever unconsumed); fresh sweep iso
  M=8192 1.615× / M=4096 1.521× / M=16384 1.512×, decode M≤64 no-regress (1.04–1.33×). 7th confirm (e2e gate pending Integrator).
- source: 2026-06-22 re-derive on Kimi-K2.6 TP=8 (eval e2e_..._20260622T155005Z): same shape, baseline
  server.log 'Using default MoE config' (lever unconsumed); fresh per-bucket subprocess sweep iso M=8192
  1.606× / M=4096 1.538× / M=16384 1.479× / M=2048 1.072×, decode no-regress M=1 1.061× / M=64 1.204×,
  all buckets rel_err=0. Oracle engagement re-verified: unittest.py loads tuned JSON ('Using configuration
  from …int4_w4a16.json', tuned=true, pass=true). **e2e gate PASSED 2026-06-23 (10th confirm): same-session
  interleaved A/B REF 437.9 → CAND 486.2 tok/s = +11.33%, non-overlapping (cand_min 486.2 > ref_max 437.9),
  parity pass, engagement banner in cand server.log. Lower e2e delta than prior runs here because baseline was
  faster (436 vs ~500 tok/s) and integrate iso re-measured 1.2528× (subprocess ~1.6× @M8192).**
- source: 2026-06-22 re-derive on Kimi-K2.6 at **TP=4** (eval e2e_..._20260622_160143; lookup N=512 since
  moe_intermediate=2048//4). Per-bucket subprocess sweep iso M=16384 1.713× / M=8192 1.668× / M=4096 1.599×,
  decode no-regress M=1 1.111× / M=64 1.095×, all rel_err pass. **e2e gate PASSED (Integrator cfg0): +17.66%
  e2e, 257.7→303.2 tok/s, non-overlapping (cand_min 302.5 > baseline_max 260.2), engagement verified
  ('Using configuration from …E=384,N=512,…int4_w4a16.json'; 'Using default MoE config' gone), KV 21.97 vs
  23.1 GiB no starvation @ mem=0.9.** 9th confirm; first TP=4/N=512 e2e-verified data point.
- caution: **N is per-TP-rank (moe_intermediate//TP).** TP=8→N=256, TP=4→N=512; the lookup filename and the
  whole sweep change with TP. Re-derive from model config + the run's serving TP, never reuse a TP=8 JSON at TP=4.
- source: 2026-06-23 Kimi-K2.6 TP=4 (eval e2e_..._20260623_092549): **lever already CONSUMED in the accepted
  baseline** — baseline_flags seeded the prior TP=4/N=512 moe_tuned dir as VLLM_TUNED_CONFIG_FOLDER + mnbt=16384,
  server.log shows 'Using configuration from …E=384,N=512,…int4_w4a16.json' and ZERO 'Using default MoE config'.
  So the env config-tune gives NO fresh e2e win this round (no have_winner). For the dominant int4-MoE head
  (pct_gpu_time=52) the next lever is the **FlyDSL author route** (is_flydsl_available True, flydsl 0.2.2;
  reuse aiter/ops/flydsl moe_gemm_2stage / a4w4 grouped primitives) handed to kernel_workflow to beat the
  already-tuned config. op_bench grouped probe: flydsl correct rel_err 0.0037, ck correct, aiter grouped
  entrypoint signature-failed (per-backend no-win, harness OK). Confirms the 'verify baseline didn't already
  bank the config' caution as a real, recurring gate.
