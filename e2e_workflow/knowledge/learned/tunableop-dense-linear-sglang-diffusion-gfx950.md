---
key: library_gemm · gfx950 · sglang-diffusion bf16 dense DiT linear (torch F.linear live path)
type: lever
confidence: ★★★
last_seen: 2026-08-11 (2 frameworks: sglang-diffusion FLUX.1-dev bf16; sglang-LLM Qwen3-14B fp8 per-tensor, MI350X)
source: e2e_FLUX-1-dev_20260729_132053 · h0_hipblaslt_dense_bf16_gemm (39% GPU time)
---

> ✅ **2nd confirm, different framework (2026-08-11, sglang-LLM Qwen3-14B `--quantization fp8`, MI350X).**
> The routing rule held again: that model's dense fp8 linear is **per-TENSOR**, so sglang calls
> `torch._scaled_mm` directly and TunableOp engaged (aiter's tuned_gemm never sees it). Prune-first was
> also reconfirmed and is now the *dominant* effect: the full CSV regressed the decode buckets 0.53–0.56x
> under cold-flush timing while TunableOp's own hot numbers claimed a 1.5x win — 178 MB of weights fits
> the 256 MB Infinity Cache. Pruned to the winning M=1024 row: 1.28x prefill, 1.00x decode (HIP-graph
> replay, no lookup tax). New caution learned: TunableOp keys on the **exact M**, so a win measured at a
> synthetic prefill bucket can bind to nothing online — check the live keys before banking it.
> (source: exp/e2e_Qwen3-14B_20260811_151017_324708_1735, h1 mlp_gate_up_proj_fp8_gemm)

lever: **PyTorch TunableOp per-shape hipBLASLt solution tune, with the regressing shapes PRUNED out
of the deploy CSV.** `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_FILENAME=<dir>/x.csv` (torch inserts the device ordinal → ship `x0..x{TP-1}.csv`).

**Routing rule this card exists for:** the standing advice "never TunableOp on sglang" is scoped to
sglang-**LLM**, whose dense GEMMs go through `aiter.tuned_gemm.gemm_a16w16` and bypass the torch
dispatch. **sglang-diffusion is the opposite case** — `multimodal_gen/runtime/layers/linear.py`
`UnquantizedLinearMethod.apply` calls `torch.nn.functional.linear` directly, so the torch/hipBLASLt
dispatch *is* the live path and TunableOp engages. Read the live `apply()` before choosing the lever;
the framework, not the vendor, decides which tuner binds. Conversely the aiter DB tune
(`AITER_TUNE_GEMM`/`AITER_CONFIG_GEMM_BF16`) binds to NOTHING here without a linear.py overlay, and
routing through `gemm_a16w16` costs ~8% in Python dispatch before it can win anything back.

effect: mass-weighted (per-image launch-count weighted, 9 live DiT-linear families) **1.026x isolated**;
best families 1.14x (M1024·K15360·N3072 single-stream proj_out) and 1.06x (M512·K3072·N3072).
Amdahl ceiling ~1.0% e2e at 39% GPU time. ZERO extra HBM, env-only, fully reversible.

caution: **also verify per-shape, then prune.** TunableOp's internal timing is HOT and un-flushed, so it
happily "wins" a shape that is slower under a cold-L2 measurement — the skinny M=1 modulation linears
regressed 0.56x that way and dragged the full-coverage CSV to 0.99x (a net LOSS). Measure every tuned
shape against default with a cold-flush paired A/B and keep only the winners; unlisted shapes fall back
to default, so a pruned CSV can only help. Also: enabling TunableOp adds a per-call lookup that taxes
uncovered GEMMs ~1-2% in isolated device time.

also-verified-null: aiter `gemm_a16w16` untuned 0.92x, aiter Triton `gemm_a16w16` 0.44x, and
`flydsl_hgemm` 0.78-0.90x even after a 72-point tile/split_k sweep on pre-shuffled weights — on gfx950
hipBLASLt/Tensile is strong at these mid-size bf16 shapes, so the Tier-C author lanes (flydsl/triton)
start 1.3-2.3x behind the bar. Cheap engagement proof without a full A/B:
`PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` on one short run, then diff the recorded live keys against the
tuned CSV keys (they are byte-identical strings, e.g. `tn_3072_1024_15360_ld_15360_15360_3072`).
