---
key: kernel fusion (collective+norm+quant) · gfx942 · sglang fp8 decode TP8
type: lever
confidence: ★★★
effect: reproduced on 2 independent runs. Decode kernel time -20.0% over the fusion set (51.96 -> 41.55 us/layer); launches 245 -> 121/iter (-50.6%). e2e transfer, ISOLATED at last: **+1.583% output tok/s, non-overlapping** (167.330 -> 169.979 tok/s) — i.e. this rung ALONE is worth ~1.6%, not the +11.80% of the stack it previously shipped inside. 3rd confirm 2026-08-30: **+2.014% output tok/s marginal, non-overlapping (220.853 < 224.145)**, TPOT -2.23%, gsm8k 0.935 -> 0.945.
confirms: 3
last_seen: 2026-08-30
---
# aiter fused AllReduce+add+RMSNorm+per-group-quant — and the flag that lies about being on
- lever: on TP decode, the AllReduce → add → RMSNorm → per-group-quant chain after each
  attention/MLP block is one of the largest non-donor runs in the layer. aiter ships it fused
  (`custom_fused_ar_rms_per_group_quant` / `fused_allreduce_rmsnorm_quant_per_group`); sglang has the
  flag. Worth trying on any TP>1 decode-bound serving run.
- apply: `--enable-aiter-allreduce-fusion` (sglang `server_args.py`). On DSR1/v0.5.12 the flag was
  advertised but not actually set — the baseline printed "Enable Aiter AllReduce Fusion" while the
  args dump had `enable_aiter_allreduce_fusion=False`. Check the dump, not the banner. There are TWO
  gates on the same flag: `apply_aiter_all_reduce_fusion` and
  `LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer` — neutralise both, or the seam is
  half-live. Seam for an overlay: `self.input_layernorm.forward_with_allreduce_fusion` in `prepare_attn`.
- verify: in the trace, `cross_device_reduce_1stage` + `add_rmsnorm_quant_kernel` collapse into
  `allreduce_fusion_kernel_1stage<...>` (2026-08-26: 15.376+9.727 -> 18.216 us/layer-step;
  2026-08-30: 16.28+9.72 -> 20.78 — the same numbers twice). The stronger check is that the standalone
  `dynamic_per_group_scaled_quant_kernel` loses exactly ONE call per layer at unchanged us/call.
  ENGAGED is not execution — log the path actually taken; the fused helper prints its banner even if it
  falls back on every call.
- caution: also verify the ops you predicted would NOT change (`add_rmsnorm_quant` was pre-registered
  as "already enabled, should not move" and was in fact absorbed). And **also verify the PREFILL
  shape separately**: the fused collective carries a byte guard (64 MiB) and at ISL=8192/TP8 the
  prefill collectives are 101.9 MiB, so the fusion is decode-only at that shape and TTFT does not move
  (-0.53%, not significant) — the decode win is real, the prefill win is not there to be had.
- source: 2026-08-26 round1 (apply/FINAL_RESULT.md §4 patch 02)
- source: 2026-08-29 (COMBINED_ATTRIBUTION.md §4.1, §5 e02)
- 2026-08-30, two things that decide whether this rung can be reached at all:
  - `ca_comm` must be `aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce` — the class
    `dispatch_custom_allreduce()` returns on ROCm. **sglang's own `CustomAllreduce` defines no fused
    method at all**, so on a build that selects it (`_use_amd_deterministic_impl()`, i.e.
    `SGLANG_USE_1STAGE_ALLREDUCE=1` or deterministic inference) `GroupCoordinator.fused_allreduce_rmsnorm`
    returns None and `--enable-aiter-allreduce-fusion` silently degrades to a plain all-reduce plus
    norm. Check the class, not the flag.
  - the flag is not optional even for the QUANT variant reached by an overlay: it is what makes the
    previous layer stamp `_sglang_needs_allreduce_fusion` on its output
    (deepseek_v2.py:1982 <- `should_fuse_mlp_allreduce_with_next_layer`). With the flag off, the
    overlay's branch is never entered and its counters read **0 fused / 0 fallback while the ENGAGED
    banner still prints on all 8 ranks**.
  - caution: phase-tag the counters. At ISL 8192 the prefill call is ~117 MiB and aiter
    `should_custom_ar` declines above 64 MiB, so a correct decode-only landing looks like
    `decode 3600/0, prefill 180/4740`. Untagged it reads as a 19% fallback rate and gets misjudged.
- source: 2026-08-30 (05_FUSION_APPLYBACK.md e04; overlay fusion/fusion_overlays/dsr1/e04_ar_norm_quant)
