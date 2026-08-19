---
key: fp8 a8w8 blockscale GEMM · gfx950 · vLLM prefill+decode
type: lever
confidence: ★★★
effect: e2e +16.1% / +65.69% / +18.86% on three models; iso 1.86–3.17× serving-weighted. Gain is gated by SHIPPED aiter table coverage — ~0 when the head shapes already bind tuned (measured 1.03× on a saturated model). Probe coverage before budgeting.
last_seen: 2026-08-16
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
- source: exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-12; exp/e2e_*Qwen3-14B-FP8*/ 2026-08-13
  (Director-validated_win TP1, head 67.96% GPU); exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-13
  (Director-validated_win TP2, head 19.78% GPU); exp/e2e_*DeepSeek-V4-Flash-0731*/ 2026-08-13
  (saturated shipped tables → iso 1.03×, the zero-headroom case); exp/e2e_*DeepSeek-V4-Pro*/ 2026-08-16
  (run-level 1.26× Director-validated, but both attributed heads reversed to dead_end
  (implausible_speedup) at review — not counted as a confirm here).
  Recipe: `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`; tuned CSVs under `config/ck_tune/`.
