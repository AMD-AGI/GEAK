---
name: dense-gemm-bf16-gfx950-sglang
description: bf16 dense GEMM on sglang/gfx950 - backend swap is a dead-end, the aiter per-shape DB tune is the lever, and hot-tuned rows must be re-gated COLD.
key: dense_gemm_bf16 · gfx950 · sglang decode+prefill
keywords: [aiter, tuned_gemm, gemm_a16w16, AITER_CONFIG_GEMM_BF16, gradlib, gemm_tuner, hipblaslt, flydsl, colon-merge, cold-cache, get_padded_m, shipped-coverage, split-k, decode-roofline]
kernels: [aiter.tuned_gemm:gemm_a16w16, Cijk_Alik_Bljk_BBS_BH_*, hgemm_*]
platforms: [gfx950, gfx942]
kernel_class: dense_gemm
regime: decode+prefill
type: lever
confidence: ★★
confirms: 6
effect: backend swap = NO win (iso 1.0×, hipBLASLt already fastest, 4 confirms). The aiter per-shape DB tune is the only env lever on sglang (live seam `aiter.tuned_gemm:gemm_a16w16`) and engages hard (1180 tuned hits vs 0), ZERO extra HBM — but the HOT-tuned CSV must be re-gated COLD or it regresses decode, and on a SMALL, hipBLASLt-native shape family the cold gate can eat the whole win. Cold-gated: 1.05× serving-wtd (Qwen3.5-397B) but exactly 1.00× on Llama-3.1-8B TP1 AND on Qwen3-8B TP1 (every kept row landed on a ramp M bucket, both served buckets regressed); gfx942·sglang analog banked +2.23% e2e. NOTE (aiter d9e5ef7c): `gradlib/gemm_tuner.py` is now hipblaslt-ONLY — the multi-backend (asm/opus/flydsl/triton/skinny/torch) race moved to `csrc/gemm_a16w16/gemm_tuner.py --with-hipblaslt`.
3rd NEGATIVE (same Qwen3-8B box, the SKINNY down/qkv/o family): the tuner DOES elect `libtype=flydsl` split-K at decode here — but every pick LOSES COLD by ~0.71x, and 2 of the 3 (N,K) decode buckets were ALREADY covered by a shipped `model_configs/*_bf16_tuned_gemm.csv` from an unrelated model. Check shipped coverage per (M,N,K) before tuning, and cold-gate flydsl split-K picks especially.
lifecycle: active
last_seen: 2026-08-22
---
# bf16 dense GEMM on sglang/gfx950 — backend swap dead-ends, the aiter DB tune engages

- path: (1) Tier-A swap is a dead-end — 3rd confirm. Qwen3.5-397B-MXFP4 attn/linear-attn projections,
  served-weighted device ms: hipblaslt 0.041 « triton-stub 0.084 « ck 0.102 « aiter-fused 0.348;
  flydsl raised `Invalid tiling configuration` at M=8192 (its default tile table has no entry for the
  prefill bucket) so it could not be scored. The live default already wins. (2) On sglang the live path
  IS `aiter.tuned_gemm:gemm_a16w16`, so the aiter per-shape DB tune (`AITER_CONFIG_GEMM_BF16`) is the
  real lever — contrast vLLM/gfx950 where the live seam is `rocm_unquantized_gemm_impl`→hipBLASLt and
  the same tune is NOT a lever. (3) Check shipped coverage first, then tune only the uncovered shapes.
- **capture for free from the BASELINE server log — no AITER_TUNE_GEMM capture run needed.** When
  coverage is zero, aiter already prints every live call as
  `shape is M:..,N:..,K:.. dtype=.. bias=.., not found tuned config in ..., will use default config!`.
  Parse those lines out of the existing `baseline/server.log`: it gives the exact runtime lookup key
  (true `bias`, true M ladder incl. the CUDA-graph decode buckets) with ZERO extra server launches.
  Then bucket-reduce with `aiter.tuned_gemm.get_padded_m(M,N,K,0)` (262 raw → 137 padded rows here).
- ⚠️ **NEW, the big one: gradlib tunes HOT — re-gate every tuned row COLD or it regresses decode.**
  `gemm_tuner.py` scores solutions with a back-to-back cache-resident loop, but the live server reads
  these weights COLD (per-layer weight set is GBs, past the 256 MB MALL). On this run **34 of 104
  hot-tuned rows LOST cold**, down to 0.68×, including the head decode row (64,4608,4096): hot 1.18×,
  cold 0.91×. Even gradlib's own `--compare --update_improved --min_improvement_pct` gate does not
  catch it (its verification pass is hot too). Fix: after tuning, re-time each kept row default-vs-tuned
  with the L2/MALL flushed before every sample and drop anything under ~1.02× (driver:
  `$EVAL_DIR/config/cold_gate.py`). Cold-gating turned a 1.019× serving-weighted candidate (with decode
  regressions) into a clean 1.05× with none.
- ⚠️ **4th confirm + the NEGATIVE case that bounds this lever: Llama-3.1-8B-Instruct TP1 gfx950 sglang
  0.5.17 (2026-08-21), head `mlp.gate_up_proj` N=28672 K=4096, 8.37% GPU / 18.24% family.** Tier-A
  reconfirmed dead (served-wtd device ms: hipblaslt **0.149** « triton-stub 0.224 « aiter-fused 0.638;
  flydsl again raised the SAME `Invalid tiling configuration ... m=8192` — its default tile table has no
  prefill entry, so env-flydsl is unreachable for a bf16 prefill bucket and only the AUTHOR route can use
  it). Tier-B: log-derived capture (181 lines → 78 padded rows via `get_padded_m`, zero AITER_TUNE_GEMM
  run), 27 prioritized rows tuned with gradlib on ONE GPU (~55 s/decode row, ~4 min/prefill row, all 27
  won by `libtype=hipblaslt`, `err_ratio 0.0`). **Cold re-gate (2 independent repeats, tight agreement)
  killed it at the SERVED buckets**: the head decode `M=64,N=28672` is **0.955–0.966×** and the head
  prefill `M=8192,N=28672` **0.988–0.991×**, and `M=64,N=4096,K=14336` (down_proj) is a brutal
  **0.66×** — while the ONLY reproducible wins sit on ramp/mid-M buckets the conc=64 steady state never
  visits (M=512·down 1.49–1.51×, M=256·down 1.22×, M=128·gate_up 1.15–1.16×, M=8960 mixed-prefill
  1.06–1.08×). Serving-weighted head effect after cold-gating = **1.00×** ⇒ no direct_light candidate.
  RULE OF THUMB: when the shipped hipBLASLt default already picks the same Tensile kernel the tuner
  re-elects (here `Cijk_..._MT128x64x128` for the head decode row), the DB tune has no headroom — check
  the tuner's `kernelName` against the profile's kernel name BEFORE spending an hour of gradlib.
- ⚠️ **5th confirm, 2nd NEGATIVE — Qwen3-8B TP1 gfx950 sglang 0.5.17 (2026-08-21), head `mlp.gate_up_proj`
  N=24576 K=4096, 7.53% GPU / 17.61% family (gate_up+down+qkv+o_proj, all `bias=False`, ZERO shipped
  coverage: 200 `not found tuned config` lines / 0 `is tuned` in baseline/server.log).** Tier-A dead a
  4th time (served-wtd device ms: hipblaslt/torch-default **0.410** « triton-stub 0.468 « aiter-fused
  1.723; flydsl raised the SAME `Invalid tiling configuration ... m=8192 n=24576 k=4096` — 3rd repro, so
  env-flydsl is unreachable for ANY bf16 prefill bucket and only the AUTHOR route can use FlyDSL).
  Tier-B with the NEW multi-backend tuner (`csrc/gemm_a16w16/gemm_tuner.py --with-hipblaslt --libtype all`,
  which races 2084 hipblaslt + ~1109 flydsl + 58 opus + asm/skinny/torch per shape): **the tuner re-elected
  the DEFAULT `libtype=torch,solidx=0` for the served decode bucket (M=64) — i.e. it found nothing better —
  and its prefill pick (`hipblaslt solidx=438241`, `MT256x256x64`, fastest HOT pick) cold-gates
  to 0.973× and 0.913× in two independent re-times.** Serving-weighted (0.7 decode / 0.3 prefill) = 1.00×
  ⇒ no direct_light candidate, again. The rule of thumb generalizes: **when aiter's own default is `torch`
  (native hipBLASLt heuristic) on a hipBLASLt-native (N,K) with K=4096, the DB tune has no cold headroom.**
- ⚠️ **6th confirm, 3rd NEGATIVE — SAME Qwen3-8B TP1 gfx950 sglang 0.5.17 box (2026-08-22), but the
  SKINNY h2 family** `down_proj N=4096 K=12288` + `qkv N=6144 K=4096` + `o_proj N=4096 K=4096`
  (9.29% GPU combined, all `bias=False`). Three NEW facts the earlier confirms did not have:
  1. **Tier-A dead a 5th time, and now with 2 reproducible repeats.** Served-weighted cold device ms
     (op_bench, decode weight ~0.94): hipblaslt/torch-default **0.0751–0.0757** « triton-stub 0.25–0.28
     « ck 0.33–0.34 « aiter-fused 0.41–0.48; flydsl again `Invalid tiling configuration ... m=8192
     n=4096 k=12288` (**4th repro** — env-flydsl is unreachable for ANY bf16 prefill bucket).
     ⚠️ First bake-off attempt ran while a foreign tenant held GPU2 at 100% and produced a FAKE
     `triton 2.77× / ceiling 6.31%` (the hipblaslt decode leg was inflated ~21× by the contention). Always check
     `rocm-smi` GPU% and require two agreeing repeats before believing an op_bench head number.
  2. **CHECK SHIPPED COVERAGE PER (M,N,K), not per model.** The prior "zero shipped coverage" note was
     only true for the h1 head (N=24576). Here `model_configs/qwen3_5_397b_bf16_tuned_gemm.csv` — a
     DIFFERENT model's shipped table that auto-merges when the env is unset — already covers
     `64,4096,4096` and `64,6144,4096` with `libtype=flydsl`, so the live decode default for o_proj/qkv
     IS tuned flydsl. Symptom when you tune them anyway: the merge **raises** `Found N duplicate shape
     entries` and rewrites your own CSV down to the uncovered rows.
  3. **The tuner elects flydsl split-K at decode, and it is a HOT-ONLY win.** For the uncovered
     `down_proj` rows it picked `flydsl split_k4 t64x64x128` (fastest HOT pick at M=64) — but the cold
     re-gate through the live seam is **0.71×** (default vs tuned, 3 passes)
     ⇒ kept 0/2, no direct_light candidate. `err_ratio` on those split-K rows is 0.015–0.017 (vs 0.005
     for split_k1), so a flydsl split-K row is also a parity risk, not just a perf one.
  **What IS left is Tier-C.** Live-seam COLD decode roofline at M=64 on MI355X, as a fraction of the HBM roof:
  down **~28%**, qkv **~14%**, o_proj **~11%**; prefill M=8192 is compute-bound and sits near the
  achievable FLOPS roof. Note the
  two worst decode buckets are the ones ALREADY on tuned flydsl ⇒ the DSL's tuned registry has run out
  of gas on skinny K=4096, which is why the authored (Triton-first) route — not another DB tune — is
  the lever for this family.
- ⏱ **COST TRAP with the multi-backend tuner on ONE GPU — budget it or you get nothing.** Its final
  verification pass groups ~2550 candidates/shape; **8 shapes (20409 tasks) did NOT finish in 90 min** at
  the default `--iters 101 --warmup 5`, and the CSV is written only at the END (no incremental output ⇒ a
  timeout loses the whole run). Cut to `--warmup 2 --iters 20` (~5× faster; the cold re-gate is the real
  judge anyway) and tune **≤2 shapes per invocation** — 2 shapes then complete in ~8 min end-to-end.
- apply: tune with gradlib `gemm_tuner.py --input_file <csv> --tuned_file <out> --mp <ngpus>
  --compare --update_improved --min_improvement_pct 2`; do NOT pass `--indtype` (this build overwrites
  the CSV dtype column / has KeyError'd) — the CSV already carries `torch.bfloat16`. ~2 shapes/min on 4
  GPUs for N≈5120,K=4096. Deploy as a COLON-MERGE
  `AITER_CONFIG_GEMM_BF16=<configs/bf16_tuned_gemm.csv>:<model_configs/*bf16_tuned_gemm*.csv>:<mine>`
  — when the env is UNSET aiter auto-merges `model_configs/`, so a bare single path silently drops all
  shipped coverage. Check for duplicate lookup keys across the merged files first: the merge RAISES and
  rewrites the package CSVs in place if it finds any.
- verify: `AITER_LOG_TUNED_CONFIG=1` → `grep -c 'is tuned on cu_num' server.log` (the log prints `N:4608`,
  not `N=`). Here 0 hits / 1036 misses before → **1180 hits / 92 misses** after, libtype breakdown
  hipblaslt 828 / **flydsl 340** / asm 8 / skinny 4. FlyDSL's env reachability path is real: the DB tune
  selected `libtype=flydsl` for a whole family (N=4096,K=2048) with no authoring. `err_ratio 0.0` per
  row → parity-safe algorithm swap.
- caution: single-GPU gradlib is slow (~2000 hipBLASLt solutions/shape) — bucket-reduce hard, `--mp
  <all GPUs>`. A partial DB never regresses (uncovered shapes fall back). Sizing: even with strong
  per-shape wins the SERVING-weighted number is what counts — decode-only was 1.10× and prefill-only
  1.03×, netting 1.05× at meta's 0.7/0.3 mix, i.e. ~+0.9% e2e at an 18.9% family share (near the noise
  band; ship it because it is free and stackable, but do not size a round around it). The head decode
  shape runs at only ~15% of the HBM roof — that residual is a Tier-C authoring target, not a
  tuning one.
- source: exp/e2e_*Qwen3-8B*/ 2026-08-13 ×2, gfx950; Qwen3.5-397B-A17B-MXFP4 (sglang 0.5.17, ROCm 7.2,
  TP4) 2026-08-21 (log-derived capture + gradlib + cold re-gate + colon-merge deploy, engagement
  verified on a live warm server; e2e gate pending); Llama-3.1-8B-Instruct (sglang 0.5.17, ROCm 7.2, TP1,
  gfx950) 2026-08-21 — the NEGATIVE confirm above (cold-gated to 1.00× at the served buckets, no
  direct_light candidate; artifacts in `<eval>/config/llama31_8b_bf16_tuned_gemm*.csv`).
  Recipe `gemm_tuning/aiter_gemm_tuning.md`.
  Qwen3-8B-Instruct (sglang 0.5.17, ROCm 7.2, TP1, gfx950, MI355X) 2026-08-21 — the 5th confirm /
  2nd negative above; artifacts in `<eval>/config/gemm_tune/` (untuned_head.csv, qwen3_8b_bf16_tuned_head.csv,
  cold_gate.py + *.report.json).
  Qwen3-8B-Instruct SAME box 2026-08-22 — the 6th confirm / 3rd negative (h2 skinny family); artifacts
  `<eval>/config/gemm_tune/{tune_h2.sh,h2_untuned_[ab].csv,h2_tuned_[ab].csv,h2_tuned_all.csv,
  h2_tuned_coldkept.csv.report.json,roofline_h2.py,h2_roofline.json}` and
  `<eval>/kernels/authored_decode_splitk_dense_linear_task/opbench_result_run[12].json`.
