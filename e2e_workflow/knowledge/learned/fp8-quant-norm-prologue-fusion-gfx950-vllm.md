---
name: fp8-quant-norm-prologue-fusion-gfx950-vllm
description: Fuse AND re-geometry the fp8 dynamic-group-quant / rms-add / act-mul prologue ring around a blockscale GEMM stack — +3.97% then a further +1.60% e2e from the SAME kernel; only members above the launch floor pay.
keywords: [fusion, fp8, group-quant, dynamic-quant, rmsnorm, act-mul, silu, elementwise-overhead, dispatch-floor, byte-reduction, module-overlay, decode-bound, launch-geometry, num-warps, waves-per-eu, block-size-n, launcher-config-table, cudagraph-capture]
kernels: [_act_mul_and_dynamic_fp8_group_quant_kernel, _fused_rms_fp8_group_quant_kernel, dynamic_per_group_scaled_quant, scaled_quant_kernel, aiter.ops.triton.activation]
platforms: [gfx950]
kernel_class: method
regime: decode
key: fp8 blockscale dynamic-group-quant + rms-add + act-mul prologue/epilogue cluster around a dense fp8 GEMM stack · gfx950 · vLLM (aiter linear on)
type: lever
confidence: ★★★
confirms: 2
effect: TWO levers, both e2e-gated, on the SAME ring. (1) fusion: iso 1.741x geomean over a 13.63%-GPU cluster -> +3.966% e2e (4 replicas, non-overlapping, gsm8k-neutral). (2) launch-geometry retune of the surviving act-mul+quant kernel AFTER that fusion pass: iso 2.063x geomean, in-situ 2.08x (34.7 -> 16.7 ms at an IDENTICAL 2560 calls), 4.96% -> 2.45% GPU -> a further +1.6005% e2e, non-overlapping, gsm8k -0.67pp (inside spread). CORRECTION to this card's earlier 'transfer is COMPLETE, do not expect more from a second pass': that held only for the FUSION lever; a DIFFERENT lever on the same kernel paid again.
lifecycle: active
last_seen: 2026-09-11
---
# gfx950 vLLM — fuse the fp8 quant/norm cluster that surrounds a blockscale GEMM head
- lever: once the fp8 GEMM heads are tuned/authored, the next mass is the ring of small quant/norm
  kernels feeding them (`act_mul + dynamic fp8 group quant`, `rms_add + fp8 group quant`, standalone
  `scaled_quant`). Rewrite them as fused Triton kernels that make ONE pass over the activation instead
  of an activation round-trip per stage. Cluster-sized, not kernel-sized: bid for the whole ring.
- apply: ship as a MODULE-replacement overlay of the framework's aiter-op shim (vLLM: inject
  `vllm._aiter_ops` on `PYTHONPATH`) rather than a rebind of individual callables — one module patch
  covers every call site, and the API server, EngineCore and worker each log the injection so
  engagement is countable. Warm the Triton JIT at module import so nothing compiles inside CUDA-graph
  capture. Memory-neutral (KV pool within 0.2%); it does not buy throughput with KV budget.
- lever 2 (after fusing): the surviving wide-activation kernel is usually still launched with ONE
  static geometry for every M. Replace its launcher with a small per-M config table (block width /
  num_warps / waves_per_eu), keeping the numerics fixed (quant block size == group size). Here prefill
  M took BLOCK_SIZE_N=1024/num_warps=2 and decode M<=128 took BLOCK_SIZE_N=512/num_warps=1/
  waves_per_eu=4, for 2.08x in situ on an unchanged launch count. A kernel at the dispatch floor is
  launch-bound; a kernel well ABOVE it is geometry-bound, and geometry is cheaper than a rewrite.
- verify: (1) a static GPU probe at the served decode M BEFORE the timed A/B, asserting each fused
  kernel actually launches and that any by-design fall-through (e.g. rms_add deferring to upstream at
  prefill) is intentional; (2) N injection banners in the candidate server log and ZERO in the
  reference; (3) e2e where the workload lives — this cluster is decode-side, so the signal is TPOT
  (9.37 → 9.07 ms) more than TTFT.
- verify (capture): on a graph-replayed decode path, count the launcher's config lines emitted DURING
  the `Capturing CUDA graphs` window in the candidate server log and ZERO in the reference — that is
  what proves the CAPTURED graph recorded the new geometry, not just that the module was injected
  (see [[method-verify-engagement]]).
- caution: **screen each member against the box's per-launch DISPATCH FLOOR before authoring it.** Here
  only 1 of the 3 rewritten kernels moved: the act-mul+quant kernel over the wide (34816) activation
  fell 8.12 → 5.64 us (−30.5%), while the two 5120-wide kernels were already at ~5.6 us — the floor —
  and came back −2.1% / +2.2%, i.e. noise. A kernel already at the floor is launch-count-bound: no
  rewrite can win, only removing launches can. Also verify the ACCURACY gate rather than byte parity
  (see `method-e2e-ab-harness.md`): the fp8 baseline is not byte-reproducible across server launches.
- source: exp/e2e_*Qwen3-14B-FP8*/ 2026-09-11 (ROUND=1, vLLM TP1, isl/osl/conc 1024/1024/64; launch-
  geometry retune, overlay `cand_fp8_quant_norm_prologue_cluster`, profile `round_head` -> `round_1`;
  ref 6369.3 -> cand 6471.2 tok/s).
- source: exp/e2e_*Qwen3-14B-FP8*/ 2026-09-09 (ROUND=1, TP1, isl/osl/conc 1024/1024/64,
  overlay `cand_k0fq*`, profile `round_head` → `round_1`); technique card in
  `kernel_workflow/knowledge/learned/`.
