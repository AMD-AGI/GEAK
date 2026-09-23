---
# --- discovery header ---
name: dense-gemm-bf16-gfx1151-vllm
description: gfx1151/RDNA vLLM dense bf16 GEMM → route ALL five decode Linears to aiter Triton gemm_a16w16 (split-K, gfx1151-tuned table): KB overlay for qkv/o/down + a tuned-table extension for gate_up/lm_head; +24.37% e2e Director-verified full stack
keywords: [dense-gemm, bf16, gfx1151, rdna, strix-halo, vllm, aiter-triton, gemm-a16w16, split-k, rocm-unquantized-gemm, decode, kb-replay, tuned-gemm-table, gate-up-proj, lm-head]
kernels: [_gemm_a16_w16_kernel, gemm_a16w16, rocm_unquantized_gemm_impl]
platforms: [gfx1151]
kernel_class: dense_gemm
regime: decode
# --- classification + evidence ---
key: dense bf16 GEMM · gfx1151 (RDNA3.5 / Strix Halo iGPU, unified memory) · vLLM
type: routing
confidence: ★★★
effect: FULL-STACK +24.37% e2e Director-validated (133.894→163.401 tok/s, 1.2437×, non-overlapping 3-repeat same-session A/B, final spread 0.73%; validated_win; serving-greedy parity pass 10/12 identical). Two stacked levers on the SAME aiter gemm_a16w16 gfx1151 route: (1) KB-recovered overlay for qkv/o/down = +13.85% (o_proj alone +13.588% attributed); (2) a gfx1151-tuned-table extension adding gate_up (iso 1.195×) + lm_head (iso 1.053×) = +9.338% incremental on top (tuning-phase A/B 152.172→166.382). All 5 heads engaged in the final server. Isolated × only 1.05–1.31× — the win is decode launch/dispatch, not FLOPs.
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
lifecycle: active
last_seen: 2026-09-22
---
# gfx1151 vLLM dense bf16 GEMM → aiter Triton gemm_a16w16 (the route CDNA gates OFF)
- lever: on gfx1151 the live vLLM dense bf16 GEMM seam is
  `vllm...layers.utils:rocm_unquantized_gemm_impl` (`aten::linear` → a rocm/rocBLAS unquantized
  kernel). Reroute it to aiter's Triton `gemm_a16w16` (`aiter.ops.triton` `_gemm_a16_w16_kernel`,
  split-K) backed by a gfx1151-tuned per-shape tile table. Route ALL FIVE decode Linears — the
  overlay covers qkv_proj / o_proj / down_proj (decode M=8; biggest single head o_proj N=2560 K=4096
  at +13.588% attributed e2e), and a **data-only tuned-table extension** adds the two families the
  prior run had left on the Tensile path: gate_up_proj (N=19456 K=2560, the largest per-layer
  projection ×36 layers) and lm_head/vocab (N=151936 K=2560, the single biggest weight).
- apply: two stackable pieces on the same route. (1) OVERLAY — bind `rocm_unquantized_gemm_impl` onto
  `gemm_a16w16` for the (N,K) families + ship the gfx1151 tile table; pure overlay, CUDA-graph-safe,
  byte-exact, **replays straight from the e2e KB overlay artifact** for this exact identity
  (qwen3-4b-instruct-2507 · vllm · bf16 · tp1 · this workload point). (2) TUNED-TABLE EXTENSION —
  add the gate_up/lm_head rows to the aiter gemm_a16w16 M≤16 decode routing table and deploy it as a
  data-only artifact via `GEAK_TUNED_GEMM_TABLE=<tuned_gemm_table.json>` (ZERO code edit) layered on
  the unchanged overlay; idempotent deploy. This lever cannot ride the PYTHONPATH overlay, so a
  finalize bundle must run its deploy step (table copy + cache invalidation) before launch or it
  reaches the shipped server as a silent no-op.
- verify: one ENGAGED banner per rerouted Linear + live hit counters (this run all five heads
  engaged: qkv/o/down 100/100/100, gate_up 288 hits, lm_head 1000+) and CUDA-graph-safe in the
  server log; then confirm with a same-session interleaved A/B (non-overlap gate) + serving-greedy
  parity. A tuned table that never binds fails silently, so the hit counters are the gate, not the
  file's presence — check gate_up/lm_head specifically since they are the shapes prior runs dropped.
- caution: gfx1151 is RDNA3.5 (Strix Halo, Radeon 8060S iGPU) with **unified/shared system memory,
  NO discrete HBM** — do NOT carry CDNA gfx942/gfx950 tuning priors or HBM-bandwidth roofline
  assumptions across; the tile table is arch-specific and must be re-tuned per gfx. Note this is the
  same aiter-triton `gemm_a16w16` route that `use_aiter_triton_gemm()` gates OFF on gfx942/gfx950
  vLLM (see aiter-bf16-tuned-gemm-gfx942.md) — on gfx1151 it engages and wins, so re-check the live
  dispatch per arch rather than assuming it's dead. Also verify the e2e gate, not the isolated ×:
  ~1.31× iso → +13.85% e2e because the win is decode launch/dispatch, not GEMM FLOPs.
- caution (isolated-bench measurement trap, cost the prior run this win): a loop-timed isolated GEMM
  bench that reuses ONE weight tensor reads it out of Infinity Cache, not LPDDR5X — this run first
  saw qkv 3.33× (642 GB/s, impossible above the ~256 GB/s pin) and had earlier EXCLUDED gate_up on a
  0.998× read from the same cache-warm F.linear baseline. Rotate through many distinct weight buffers
  (~1.5 GB) so every call streams a cold weight like real decode (36 distinct layers); only then are
  the numbers physical (~224–231 GB/s) and gate_up shows its real 1.195×. Do not size a decode GEMM
  from a cache-resident bench.
- caution (image provisioning, not a no-win): `hipblaslt-bench` and `ckProfiler` were absent on this
  image, so the offline hipBLASLt/Tensile tune and the CK-instance sweep rungs were unavailable — the
  aiter-triton route was the reachable lever, not necessarily the only fast one.
- source: exp/e2e_*Qwen3-4B-Instruct-2507*/ 2026-09-22 (Director-validated full-stack same-session
  A/B 133.894→163.401 tok/s, 1.2437×, validated_win, parity pass; KB-recovered overlay replayed +
  gate_up/lm_head tuned-table extension re-gated on gfx1151); cross-ref
  aiter-bf16-tuned-gemm-gfx942.md for the CDNA vLLM `rocm_unquantized_gemm_impl` seam.
