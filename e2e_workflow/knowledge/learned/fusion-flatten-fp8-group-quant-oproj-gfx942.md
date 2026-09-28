---
key: kernel fusion (reshape+quant) · gfx942 · sglang MLA fp8 decode
type: lever
confidence: ★★★
effect: +1.606% e2e output tok/s MARGINAL on top of an already-accepted fusion, non-overlapping (169.979 -> 172.709 tok/s); TPOT -1.624%; gsm8k flat 0.940. Kernel: one standalone quant call/layer removed, replaced by a fused flatten+quant at 4.94 us/layer. REPRODUCED 2026-08-30 on a different container and a different underlying stack: **+2.142% output tok/s marginal, non-overlapping (215.421 < 219.277)**, TPOT -2.52%.
confirms: 2
last_seen: 2026-08-30
---
# fold the o_proj input reshape into its fp8 group-quant (MLA attention output)
- lever: MLA's attention output is `(B, H, D)` and o_proj wants `(B, H*D)` fp8 + per-128-group scale.
  The stock path does `x.flatten(1,2)` and then a separate `dynamic_per_group_scaled_quant`. aiter has
  `fused_flatten_fp8_group_quant(x, group_size=128, dtype_quant=fp8)` which does both. Cheap, exact,
  and it fires on EVERY layer — unlike MLP-side quant fusions, which on DSR1 only reach the 3 dense
  layers of 61. Prefer the seam with the widest layer coverage.
- apply: rebind `forward_absorb_core` in
  `sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla`; the helper returns the
  `(fp8, scale)` TUPLE and hands it straight to o_proj. That works because `Fp8LinearMethod.apply`
  (fp8.py:781) already accepts a tuple for block_quant and routes it to
  `aiter_w8a8_block_fp8_linear(input_scale=...)`; on gfx942 `use_triton=True`, so the scale must be
  row-major `(M, K//128)` fp32. This "(fp8, scale) tuple" seam is PREBUILT and arch-independent — it
  is the cheapest place to land a quant fusion in sglang fp8.
- verify: `_fused_flatten_fp8_group_quant_kernel` must appear in the decode trace at ~1 call per layer
  (61/step on DSR1), and `dynamic_per_group_scaled_quant_kernel` must lose exactly one call per layer
  at unchanged us/call. Log the fused-vs-fallback path per rank: 8/8 ranks fused, 0 fallbacks.
- caution: the fused kernel is barely cheaper than the quant call it replaces (+4.94 vs -5.11
  us/layer), so a single overlaid trace will NOT predict this rung's e2e win — the trace under-predicts
  it by ~30x while over-predicting the collective-fusion rung. **Also verify with a marginal A/B leg**;
  trace arithmetic is sound only for the fusion set in aggregate (it explained 93% of the combined
  -0.685 ms/token ITL), not per rung. Also verify fp8 dtype: this box is `float8_e4m3fnuz`, and several
  aiter entry points branch to a gfx95-only path on `float8_e4m3fn`.
- source: 2026-08-29 (COMBINED_ATTRIBUTION.md §4.1, §5 e07)
- source: 2026-08-30 (05_FUSION_APPLYBACK.md e09; overlay fusion/fusion_overlays/dsr1/e09_oproj_flatten_quant)
