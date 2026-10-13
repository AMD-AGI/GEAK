---
key: kernel fusion (router GEMM + topk softmax + shared-expert gate) · gfx942 · sglang Qwen3.5 MoE decode, cudagraph
type: lever
confidence: ★★★
effect: THREE non-overlapping e2e confirms on this exact model. (1) 2026-09-28: +6.046% e2e output tok/s (ref 191.037 n=6 vs cand 202.588 n=4), TPOT -5.73%, gsm8k 0.930->0.930 (n=200). (2) 2026-10-10 (e04): +6.721% e2e, non-overlapping at 2 pairs (ref=[277.99,277.17] median 277.58 vs cand=[296.71,295.76] median 296.24, spread 0.32%, clear win >=10x spread), TPOT -6.43%, TTFT -2.18%, 40/40 req, engaged x8 ranks UNDER cuda graph, 0 sched exc; accuracy PASS at full-1319 gsm8k (base 0.3814 cand 0.4246, no regression; n=200 was inconclusive from greedy-concurrency noise). (3) 2026-10-12 (e02, stacked on an accepted GDN in_proj dual-GEMM e01): +7.598% e2e, non-overlapping at 3 pairs (ref=[293.859,293.683,292.422] median 293.683 vs cand=[315.583,315.996,316.541] median 315.996, spread 0.489%, clear win >=10x spread), TPOT 12.81->11.87ms -7.34%, TTFT ~504->~493ms -2.3%, engaged x8 ranks UNDER cuda graph (capturing=True at M=4, pre-capture built all 60 blocks), 0 fallback/guard-miss; accuracy PASS via unit-equivalence (calibration mode unit_equivalence_only; routed ids+weights bit-exact, shared weight bf16 tol<=0.00192), informational gsm8k n=200 0.46->0.465. Unit 2.85-3.13x, routed ids+weights bit-identical to the stock aiter chain. The e2e gain is larger than per-layer kernel arithmetic because 6 launches collapse to 2 per layer x60 in a launch-bound decode step.
confirms: 3
last_seen: 2026-10-12
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
  Also verify the pad: an unaligned 513-row weight breaks the 16B vector loads of topk_softmax (pad W to a
  multiple of 8 rows -> 520; pass gating=logits[:, :E+S]=513 so the vectorized load lands in the pad, not OOB).
- caution (★★★ CUDA-GRAPH, load-bearing — cost a full A/B this run): this sglang build captures the decode
  graphs (bs 1,2,4) at init BEFORE any eager forward, so a fused weight built lazily on the first CALL lands
  AFTER capture and the captured decode graphs freeze on the STOCK method -> the fusion engages ONLY in eager
  prefill and is INERT in decode (the smoke run showed ENGAGED gemm=[80,...] capturing=False, i.e. post-capture
  prefill only). FIX: also wrap `sglang.srt.model_executor.cuda_graph_runner.CudaGraphRunner.capture` and build
  every SparseMoeBlock's fused weight+buffers FIRST (walk `self.model_runner.model.modules()`), so capture
  traces the fused path. VERIFY `pre-capture build: N SparseMoeBlock(s) fused before graph capture` precedes
  `Capture cuda graph begin`, and that ENGAGED prints `capturing=True` at the decode shape (gemm=[4,...]).
- caution (aiter.topk_softmax semantics on THIS build): `topk = topk_indices.size(-1)` (pass ids[:, :K]=K routed,
  NOT K+S), gating must be [M, E+S] with the shared logit in column E, and it writes routed ids into [:, :K] +
  the shared sigmoid weight into topk_weights[:, K] but NEVER writes the shared id -> pre-fill the shared column
  of a persistent ids buffer with the base id E. Harness trap: the shipped gsm8k_leg template posts to
  `$BASE_URL/chat/completions`; sglang's OpenAI endpoint is at `/v1` -> pass `--base-url $BASE_URL/v1` or every
  request 404s to pred=None (scores a silent 0.0 that the gate "passes" trivially).
- source: exp/e2e_*Qwen3.5-397B-A17B-FP8*/ 2026-09-28 (router_shared_topk); exp/Qwen35/20261010_test_e2e_qwen35/geak/e2e_Qwen3.5-397B-A17B-FP8_20261010T095738731057_55bdd857Z 2026-10-10 (05_FUSION_APPLYBACK.md e04; overlay fusion/fusion_overlays/Qwen3.5-397B-A17B-FP8/e04_router_shared_topk); exp/Qwen35/20261010_test_e2e_qwen35/geak/e2e_Qwen3.5-397B-A17B-FP8_20261011T164752267404_4c70519dZ 2026-10-12 (05_FUSION_APPLYBACK.md e02; overlay fusion/fusion_overlays/Qwen3.5-397B-A17B-FP8/e02_router_shared_topk)
