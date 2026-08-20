---
key: dense_gemm_bf16 · gfx950 · sglang decode+prefill
type: lever
confidence: ★★
confirms: 2
effect: backend swap = NO win (iso 1.0×, hipBLASLt is already the fastest). The aiter per-shape DB tune IS a real lever on sglang (live seam = `aiter.tuned_gemm:gemm_a16w16`) and engages; ZERO extra HBM; e2e transfer pending (prior gfx942·sglang = +2.23%).
last_seen: 2026-08-13
---
# bf16 dense GEMM on sglang/gfx950 — backend swap dead-ends, the aiter DB tune engages

- path: (1) Tier-A swap is a dead-end — op_bench on gate_up decode: hipblaslt 0.048 ms « triton-stub
  0.083 « ck 0.101 « flydsl 0.154 « aiter 0.400. The live default already wins. (2) On sglang the live
  path IS `aiter.tuned_gemm:gemm_a16w16`, so the aiter per-shape DB tune (`AITER_CONFIG_GEMM_BF16`) is
  a real lever — contrast vLLM/gfx950, where the live seam is `rocm_unquantized_gemm_impl`→hipBLASLt,
  aiter regresses, and the same DB tune is NOT a lever. (3) Check shipped coverage, then tune only the
  uncovered shapes.
- expected gain: strictly coverage-gated, and gfx950 shipped coverage is UNEVEN — for Qwen3-8B,
  gate_up (N=24576,K=4096) and down (N=4096,K=12288) have 0 shipped buckets (real headroom, tune these
  first), while o_proj (4096,4096)=19 buckets and qkv (6144,4096)=10 are already shipped → near-noise,
  skip them. Expect low single-digit e2e (the gfx942·sglang analog banked +2.23%).
- apply: capture live shapes from a warm server with `EXTRA_ENV="AITER_TUNE_GEMM=1"` (do NOT set
  SGLANG_GRPC_PORT — this wheel has no gRPC ext and it crashes startup; shapes are captured at
  CUDA-graph warmup regardless). Bucket-reduce with `aiter.tuned_gemm.get_padded_m(M,N,K,0)`. Tune with
  gradlib `gemm_tuner.py --input_file <csv> --tuned_file <out> --mp <ngpus>`; this build BUGS on
  `--indtype bf16` (raises `KeyError` on the bf16 torch dtype inside `GemmTuner.pre_process`) — OMIT it and let it
  read the CSV `dtype` column. Deploy as a COLON-MERGE list
  `AITER_CONFIG_GEMM_BF16=<base_configs>:<model_configs/*>:<mine>` — a single path REPLACES aiter's
  default merge and drops shipped qkv/o_proj coverage.
- verify: `AITER_LOG_TUNED_CONFIG=1` → `grep 'is tuned on cu_num' server.log` (the log prints `N:24576`,
  not `N=`). Confirmed engaged: gate_up and down both resolve to the tuned `libtype hipblaslt` rows,
  154 tuned hits, shipped coverage preserved; err_ratio 0.0 per row → parity-safe algo swap.
- caution: single-GPU gradlib is SLOW on wide/large-M shapes (~56 s/shape racing 2084 hipBLASLt
  solutions, prohibitive for M≥2048 N=24576) — bucket-reduce hard and prefer decode buckets. A partial
  DB never regresses (uncovered shapes fall back). `lm_head` (N=151936) is not captured through this
  seam. If sglang is ABSENT from the image the DB tune is uncapturable (it needs a warm server) and the
  e2e gate is un-runnable — then the only remaining lever is Tier-C author against the hipBLASLt bar.
- source: exp/e2e_*Qwen3-8B*/ 2026-08-13 ×2, gfx950 (capture + gradlib + colon-merge deploy; second run
  covered attn qkv+o only, reconfirming the Tier-A dead-end at iso 1.0×);
  recipe `gemm_tuning/aiter_gemm_tuning.md`.
