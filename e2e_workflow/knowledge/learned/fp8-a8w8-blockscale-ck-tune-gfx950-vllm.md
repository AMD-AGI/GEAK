---
key: fp8 a8w8 blockscale GEMM · gfx950 · vLLM prefill+decode
type: lever
confidence: ★★★
effect: e2e +16.1% / +65.69% / +18.86% / +48.87% on four models; iso 1.86–3.17× serving-weighted. Gain is gated by SHIPPED aiter table coverage — ~0 when the head shapes already bind tuned (measured 1.03× on a saturated model). Probe coverage before budgeting. AFTER the tune, Tier A/B is spent but Tier-C authoring is NOT: authored Triton beat the tuned CK seam for +3.54% / +3.95% e2e on the two decode-touching heads.
last_seen: 2026-09-10
---
# gfx950 vLLM fp8 a8w8 blockscale — per-shape CK tune DB (no overlay needed)

- path: (1) probe the live seam + coverage — `AITER_LOG_TUNED_CONFIG=1`, count `use default` vs
  `is tuned`; (2) if AITER is OFF the live baseline is UNTUNED Triton `_w8a8_triton_block_scaled_mm`,
  so add the swap `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1` (Triton→CK) — this is the
  dominant term, ~1.5× on its own; (3) tune the `use default` head shapes and deploy the DB — this
  recovers the prefill regression that default-CK introduces. No fp8_utils overlay and no code_patch:
  on vLLM the tuned CSV alone binds (unlike the sglang seam).
- expected gain: read it off step 1. All shapes `is tuned` → ~1.0×, skip the lever. Zero coverage →
  the full 1.86–3.17× serving-weighted, e2e-transferring at +16% to +65% depending on head share
  (67.96% GPU → +65.69%; 19.78% → +18.86%). Partial coverage → scale by the uncovered fraction.
  A broad aiter-linear swap can EXCEED the single-head Amdahl ceiling, since it re-routes every fp8
  blockscale linear rather than one kernel.
- apply: `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --libtype both --mp <all GPUs>`
  over the captured head (M,N,K) → tuned CSV, deployed as
  `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>`. Pre-merge your rows OVER the shipped table into one
  file — a bare path REPLACES aiter's merge and drops shipped coverage. ZERO extra HBM.
- verify: `is tuned on cu_num` for every head shape, errRatio 0. Parity rel_err ~2e-4 vs the fp32
  dequant oracle (<< TOL 0.05).
- caution: gains are prefill-driven (M=4096 ~3.3×; decode M1/M64 ~1.14×), so a decode-bound mix banks
  less than the serving-weighted figure — check the e2e gate, and extend the tune to small-M decode.
  Also verify parity on EVERY deploy: a re-run of the +18.86% lever read +16.79% but failed the
  integrate gate on output_corruption. Quote the +65.69% as a free-GPU number — a contended re-run of
  the same lever landed 1.067×. The +16.1% (Qwen3.5-27B) is the weakest of the three: parity n/a under
  the fp8 accuracy gate, tuned-shape binds unreproduced, and a server-flag bundle confounds it.
  If ckProfiler is absent the CK *author* lane is unavailable, but the tune DB still applies.
- confirm (2026-08-19, Qwen3-14B-FP8 TP1, gfx950/MI355, head 67.3% GPU): swap-only aiter-linear
  Triton->CK (VLLM_ROCM_USE_AITER[_LINEAR]=1), UNTUNED CK, measured on the immutable unittest
  (`aiter.gemm_a8w8_blockscale`, non-transposed scale, parity rel~7e-3 << TOL 0.05): serving-weighted
  1.513x, geomean 1.28x. Regime split as the card warns — decode M1/M64 1.8-2.3x, prefill M571 REGRESSES
  0.66-0.84x (untuned CK). The per-shape CK tune (recovers prefill -> card's 1.86-3.17x) was NOT runnable
  here: the offline tuner `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` + ckProfiler are
  ABSENT from this image's aiter wheel (no csrc dir); foreign-checkout tuners exist on NFS but version-
  mismatch the installed aiter -> unsafe. So shipped swap-only; e2e prefill-regression risk left to the
  Integrator gate + operator-provisioned tuner.
- confirm (2026-09-07, Qwen3-14B-FP8 TP1, gfx950/MI355 cu_num=256, vLLM 0.27.1 + aiter 0.1.19,
  head 23.13% GPU post-tune): the lever WORKED and the tuner gap the 2026-08-19 confirm reported is
  CLOSED on this image — `aiter_meta/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py
  --libtype ck -k --mp 4` ran in 188 s for 61 rows and shipped as a `configs/model_configs/*.csv`
  DROP-IN (aiter colon-MERGES model_configs over configs/, so nothing vendored is edited and shipped
  coverage is preserved — prefer this to `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<path>`, which REPLACES).
  Shipped gfx950/cu_num=256 coverage for all four head (N,K) was ZERO. e2e +48.87% (3874->5768 tok/s,
  interleaved isolated-server A/B, arms disjoint, gsm8k 0.880 vs 0.875). `-k` matters: split-K wins
  every M<=128 row on the K=17408/5120-narrow families; the K=5120 wide families (qkv 7168, gate_up
  34816) tune to splitK=0.
  · **Tier-A bake-off AFTER the tune (the state a head-kernel lane inherits): the tuned CK seam is the
    winner and iso speedup is 1.00x — the lever is spent, do not re-ship it.** Served-weighted (M=64
    decode + M=1024 prefill, cold-cache CUDA-event), gate_up N=34816/qkv N=7168 K=5120:
    tuned-CK-live 0.1014/0.0338 ms · untuned CK default 0.1466/0.0382 (so the tune = 1.45x/1.13x,
    all of it prefill: M=1024 0.311 vs 0.912 ms) · **cktile default 0.3273/0.1438 = 3.2x/4.3x SLOWER**
    (its *tune* harness `module_gemm_a8w8_blockscale_cktile_tune` does not build in budget; the runtime
    `.so` does — so cktile is measurable-but-unwinnable-untuned, and remains the one unexplored rung)
    · **aiter Triton blockscale 0.1632/0.0862 = 1.61x/2.55x slower than tuned CK** (this is the
    Tier-C rewrite bar, not a candidate) · bpreshuffle is NOT reachable by weight-prep alone:
    `shuffle_weight(W, layout=(16,16))` + `gemm_a8w8_blockscale_bpreshuffle_ck` runs FAST (0.086 ms
    decode) but rel_err 7-8.4 = wrong, and loses at prefill anyway.
  · **flydsl fp8 blockscale does not exist on gfx950** even though `is_flydsl_available()` is True and
    `flydsl` 0.2.4 + `aiter/ops/flydsl/` are both present: the only block-scaled flydsl wrapper is
    `blockscale_bpreshuffle_gemm_gfx1250.py` (RDNA/gfx1250). The gfx950 flydsl GEMM seeds are
    `flydsl_hgemm` (bf16) and `flydsl_preshuffle_gemm_a8` (per-token/per-channel a8, tunable
    tile_m/n/k + lds_stage + xcd_swizzle) — usable as an AUTHOR seed only if the lane writes the
    128x128 block-scale epilogue itself. So "flydsl first for a GEMM head" does NOT hold for fp8
    BLOCK-scale on gfx950; the editable Triton rewrite is the higher-ROI first author lane.
  · caution: after the tune, `__amd_rocclr_fillBufferAligned` appears at 4.0% GPU / 4800 launches —
    exactly the split-K (AtomicAdd) rows the tuned table now selects zeroing their accumulator. It is
    real cost, it is already inside the op-level device timing (so the tuned rows still win). A
    Set-first / splitK=0 row looked like the lever against it — **the 2026-09-08 confirm below MEASURED
    that and it LOSES (0.75x)**; the fill is the price of a win, not recoverable headroom.
  · source: exp/e2e_*Qwen3-14B-FP8*/ 2026-09-07 (tuning/tuning_report.md,
    config/op_bakeoff/wide_bakeoff.json)
- confirm (2026-09-08, SAME run as the 2026-09-07 confirm, second head `h2` = the AtomicAdd split-K CK
  instance + its `__amd_rocclr_fillBufferAligned`, 19.02% GPU = 15.05% gemm + 3.97% fill; o_proj
  N=5120/K=5120 and down_proj N=5120/K=17408, M in {1,64,1024}): **the fillBuffer is NOT recoverable
  headroom — the atomic rows are the measured optimum WITH their zero-init priced in.** Benched through
  the deep seam `fn(XQ,WQ,xs,ws,Out,splitK,kernelName)` with a fresh `torch.empty` Out per call, so the
  fill sits inside the timed window; serving-weighted vs the live tuned row: forcing the SAME tuned
  instance to `splitK=0` (non-atomic, no fill) = **0.75x**, the tuner's large-M `splitK=0` instance =
  0.38x, CK default heuristic = 0.65x, aiter Triton blockscale = **0.32x** (3.1x slower — the Tier-C
  rewrite bar on this head, steeper than the 1.6-2.6x seen on h0), cktile default = **0.13x** (7.5x
  slower, so cktile is now measured-and-losing on a second head), bpreshuffle wrong again (rel_err 37-43
  at M>=64). Tier A/B is CLOSED at iso **1.00x**; the split-K-vs-fill trade is already optimal, and the
  card's earlier "lever against it is a Set-first split-K row" line is now REFUTED by measurement.
  · caution (harness, cost ~1 h): the extractor's capture-promote step OVERWROTE the built `meta.json`
    with the raw `capture_meta.json` (no `workload`/`target_callable` keys), which silently breaks the
    immutable `unittest.py` AFTER its own smoke passed. Symptom: `meta.json` and `capture_meta.json`
    identical in size+mtime. Re-run the task's own `build_meta.py` to restore it (deterministic; the
    oracle sha256 still matched the capture record, so provenance was intact).
- confirm (2026-09-08, SAME run, THIRD head `h1` = the CK **MPadding / GemmSpecialization=1** instance,
  15.53% GPU, 880 launches, PREFILL-only, gate_up N=34816 + qkv N=7168 K=5120 at the 128-UNALIGNED chunk
  M in {588, 1109}): **Tier A AND Tier B are both CLOSED at iso 1.00x — and the new fact is WHY Tier B is
  closed.** The deployed table is tuned on pow2 M only, so an unaligned M never hits an exact row; but
  `get_CKGEMM_config` retries the key through `get_padded_m(M,N,K,gl)` for gl in {None,0,1} (588 -> 608 ->
  1024, 1109 -> 1152 -> 2048), so the pow2 row IS bound. Tuning the four EXACT-M rows anyway
  (`--libtype ck -k --mp 1`, 45 s) returned **kernelId=0 splitK=0 for all four — byte-identical to the
  padded-M row already live**, i.e. an unaligned prefill M has NO separate tuning headroom on this table.
  Do not spend a lane re-tuning unaligned M buckets; check `get_padded_m` coverage first.
  · prefill bake-off vs the live tuned CK (summed over both families, cold-cache CUDA-event, M=588+1109):
  tuned-CK-live **0.3578 ms** · aiter Triton blockscale 0.7666 = **2.14x slower** (the Tier-C bar at
  prefill, vs 1.6-2.6x at h0 decode and 3.1x at h2) · untuned CK default heuristic 0.9525 = 2.66x slower
  (so the CK tune is worth 2.66x on THIS head — the biggest per-head tune multiple of the three) ·
  cktile default 1.3051 = **3.65x slower** (third head on which cktile is measured-and-losing untuned) ·
  bpreshuffle under `shuffle_weight` wrong again (rel_err 8.8-9.5) · flydsl absent for fp8 BLOCKSCALE.
  · Amdahl caveat specific to this head: it is PREFILL-only, so its gain lands on TTFT, not on the
  measured output tok/s — size any Tier-C bid against the run's 4.36% server-restart noise floor.
- confirm (2026-09-09, SAME run, **Tier-C after the tune**): the card's own prediction — "the editable
  Triton rewrite is the higher-ROI first author lane" — HELD, and it is the lever that remained once
  Tier A/B closed at iso 1.00x on all three heads. Authored Triton kernels bound at the aiter seams
  (`aiter:gemm_a8w8_blockscale` for the M<=128 & N>=16384 decode family, one level deeper at
  `aiter.ops.gemm_op_a8w8:gemm_a8w8_blockscale_ck` for the o_proj/down_proj families, so the two
  COMPOSE) measured iso 1.314x / 1.452x and transferred to e2e **+3.54%** and **+3.95%**, both
  non-overlapping and under their Amdahl ceilings. Route the deeper seam SECOND so the outer overlay
  hands its fall-through into it, and restrict each authored path to the validated (N,K) families with
  a host-side int predicate (no host sync → CUDA-graph safe); everything else falls through to the
  stock entry captured before the rebind.
  · REGIME CAVEAT (measured, not inferred): the same authoring recipe on the PREFILL-ONLY MPadding head
  **regressed −2.30% e2e** (iso only 1.049x, Amdahl ceiling +0.73%, TTFT 772→808 ms) — authored Triton
  loses to tuned CK at large-M prefill on this op. Screen on the head's regime and its isolated
  multiple: an iso win inside the lane's own no-op floor is not worth an e2e round.
  · source: exp/e2e_*Qwen3-14B-FP8*/ 2026-09-08..09 (`overlay/cand_*/integrate_result.json`).
- confirm (2026-09-10, Qwen3-14B-FP8 TP1, gfx950/MI355 cu_num=256, vLLM 0.27.1, NEW run, head h0 =
  the Set/splitK=0 decode CK instance, gate_up N=34816 + qkv N=7168 K=5120, 23.15% GPU): the whole card
  REPRODUCED end-to-end on a fresh run. Shipped gfx950/cu_num=256 coverage was again exactly ZERO (327
  `will use default config`, 0 `is tuned`); the 40-row CK tune (M pow2 16..8192 x 4 (N,K), `--libtype ck
  -k`, ~10 min) shipped as the `configs/model_configs/*.csv` DROP-IN and banked **+18.84% e2e**
  (5200.5 -> 6180.5 tok/s, 4 interleaved legs, arms disjoint, gsm8k 0.870 vs 0.875, TTFT -40%).
  · **Tier A is then CLOSED at iso 1.00x for the FOURTH time** — the post-tune bake-off found nothing
  faster than the live tuned CK. Served-weighted over the task's own 6 cases (M in {1,64,1024} x both
  families, cold-L2 CUDA-event): tuned-CK-live **0.1458 ms** · aiter Triton blockscale 0.2969 =
  **0.49x** · cktile default 0.5625 = **0.26x** (fourth head on which cktile is measured-and-losing) ·
  bpreshuffle rel_err 38.1 (wrong again, fifth head) · **`flatmm_a8w8_blockscale_asm` is a process-KILLER
  for this head — it hard-aborts the interpreter with "flatmm a8w8 blockscale asm only support Half
  output now!" on a bf16-out GEMM; guard it out of any driver rather than try/except (an abort is not
  catchable)** · flydsl fp8 BLOCKSCALE re-confirmed gfx1250-only by SOURCE this time
  (`gemm_a8w8_blockscale_bpreshuffle_flydsl` raises `RuntimeError(... only supported on gfx1250)`
  outright), so flydsl stays a structural non-candidate for the block-scale epilogue on gfx950.
  · **NEW, and the Tier-C signal: aiter's Triton blockscale BEATS tuned CK on exactly one of the six
  cases — the WIDE skinny-M one** (gate_up M=1, N=34816: 42.5 vs 67.5 us = **1.59x**), while losing
  everywhere else including the narrow M=1 (qkv N=7168: 35.1 vs 17.8 us = 0.51x). That is the measured
  crossover, and it lands precisely inside the card's earlier "M<=128 & N>=16384 decode family"
  restriction on the authored-Triton overlay — so that restriction is a measured boundary, not a
  heuristic, and the +3.54% Tier-C route is re-confirmed without re-authoring. A Tier-C bid here must
  carry the host-side (N,K)+M predicate and fall through to stock CK, or it loses the other five cases.
  · **caution (accuracy, new): the tuned CK splitK=2 row selected at M=1 measures rel_err 0.070-0.078
  against the fp32 dequant oracle**, ~10x the 0.0065-0.0075 that Triton/cktile return on the identical
  draw and ABOVE the task tol 0.02 — while every M>=64 row is 0.0073. It did not move gsm8k (0.870 vs
  0.875), so it is not a blocker, but a single-row split-K accumulation is the noisiest thing this table
  selects: check M=1 parity explicitly when a downstream gate is tight.
  · source: exp/e2e_Qwen3-14B-FP8_20260910_133807_993_9138/ (tuning/tuning_report.md,
  config/op_bakeoff/wide_bakeoff.json, kernels/ck_a8w8_blockscale_gemm_E0_Set_*/opbench_result.json)
- source: exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-12; exp/e2e_*Qwen3-14B-FP8*/ 2026-08-13
  (Director-validated_win TP1, head 67.96% GPU); exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-13
  (Director-validated_win TP2, head 19.78% GPU); exp/e2e_*DeepSeek-V4-Flash-0731*/ 2026-08-13
  (saturated shipped tables → iso 1.03×, the zero-headroom case); exp/e2e_*DeepSeek-V4-Pro*/ 2026-08-16
  (run-level 1.26× Director-validated, but both attributed heads reversed to dead_end
  (implausible_speedup) at review — not counted as a confirm here).
  Recipe: `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`; tuned CSVs under `config/ck_tune/`.
