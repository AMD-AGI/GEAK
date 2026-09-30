---
key: kernel fusion (router GEMM + topk softmax + shared-expert gate) · gfx942 · sglang Qwen3.5 MoE decode, cudagraph
type: lever
confidence: ★★
effect: +6.046% e2e output tok/s, non-overlapping (ref 191.037 [189.853..191.269] n=6 vs cand 202.588 [202.222..202.832] n=4, interleaved warm_server, TP8, ISL8192/OSL1024/CONC4). TPOT -5.73%, ITL -5.97%, TTFT -3.52%. gsm8k 0.930 -> 0.930 (n=200). Unit-side 1.363x, in-adapter graph 39.6 -> 23.9 us/layer. The e2e gain (~1.1 ms/step over 60 layers) is larger than the per-layer kernel saving predicts, because 6 launches collapse to 2 in a launch-bound decode step.
confirms: 1
last_seen: 2026-09-28
---
# fold the router GEMM, topk softmax and shared-expert sigmoid gate into ONE GEMM + ONE aiter.topk_softmax
- lever: Qwen2/3.5-MoE decode runs gate GEMM -> topk_softmax -> shared_expert_gate GEMM [H->1] -> sigmoid -> fp32 copy ->
  fused_append_shared_experts_with_weights, all per layer. `aiter.topk_softmax(w, ids, tei, logits, renorm, num_shared_experts, "sigmoid")`
  already scores the appended shared slot, so concatenate [gate; shared_gate; 0-pad to a multiple of 8 rows], run one F.linear, and
  hand topk_softmax the [M, E+S] slice. The kernel takes strided ids, so write routed ids into ids_buf[:, :K] of a persistent
  [cap, K+S] int32 buffer whose shared columns are PRE-FILLED with E (no fill kernel).
- apply: rebind `sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts` and return
  `self.experts(h, StandardTopKOutput(w, ids, logits[:, :E]))`. Build lazily on the first non-capturing call. Re-point
  gate/shared_expert_gate `.weight.data` at views of the fused weight so no duplicate is resident. Never free an old id buffer.
- verify: `[overlay-router_shared_topk] ENGAGED rank=0..7` with gemm=[M,H]x[H,E+S+pad] printed, `FIRST_FUSED_CALL capturing=False` per
  rank, and 0 FALLBACK. In the trace, fused_append_shared_experts and the [H->1] GEMM + sigmoid must reach n=0.
- caution: only valid when the topk config is plain softmax (no grouped/bias/custom routing, `num_fused_shared_experts==0` in the topk
  config because the block appends itself), `_use_aiter` is on, the runner is not triton_kernels/flashinfer, deepep is off, and the gate
  is unquantized bf16. Also verify that the expert-distribution recorder and return-routed-experts are off (they read the split path).
  Also verify the pad: an unaligned 513-row weight breaks the 16B vector loads of topk_softmax.
- source: exp/e2e_*Qwen3.5-397B-A17B-FP8*/ 2026-09-28 (05_FUSION_APPLYBACK.md e02; overlay fusion/fusion_overlays/Qwen3.5-397B-A17B-FP8/router_shared_topk)
