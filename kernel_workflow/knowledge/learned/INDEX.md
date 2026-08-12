# Learned — index of distilled kernel_workflow experience cards

<!-- GENERATED FILE — do not hand-edit. Regenerate with:
       python3 kernel_workflow/scripts/kb.py --kb-dir kernel_workflow/knowledge/learned index
     Every line below is derived from one card's discovery frontmatter. To change a line, edit the
     card's `description`/`keywords`/`confidence` and regenerate. -->

Open the cards matching your run as **additional, advisory priors** — they only ADD candidate levers to
try, never remove any and never replace measurement. The frozen-baseline isolated A/B + oracle parity is
always the judge (see `README.md`). **Cap: <=40 card lines.** Confidence (a hint strength, not
authority): ★ noise/unverified · ★★ single non-overlap or >=2 consistent · ★★★ >=2 non-overlap.

Effects are **ratios or percent deltas only, never wall-clock or absolute throughput** — those vary box
to box and stay in the run's `EVAL_DIR` (see `README.md` -> "Content rules").

**How to use this file: READ it, then open the 0–3 cards that look relevant.** Each line carries the
card's own description, the kernel symbols it was measured on, and its keywords — enough to judge
relevance without opening anything. Match on *meaning*, not on an exact string: a card written for
`split-k on skinny-M GEMM` is worth opening for a tall-K GEMM too. If nothing matches, that is a real
answer — plan cold, exactly as this workflow does without any KB.

## attention
- [gfx950 · memory-bound] Store the paged KV cache as e4m3 fp8 and dequant to bf16 in registers: halves KV HBM traffic on a scatter-bound attention op, ~1.9x on heavy shapes ★★ — (fp8-kv-storage-with-bf16-in-register-dequant-on-scatter-boun-attention-gfx950-memory-bound.md)
  - kernels: paged_attention_large · kw: paged-attention, fp8-kv-cache, hbm-bound, kv-cache-quant, memory-bound, gfx950, long-context
- [gfx950 · memory-bound] On attention already at the paged-scatter HBM roofline, compute-precision, occupancy and launch-fusion each returned ~1.00x or worse; only traffic moved ★★ — (traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound.md)
  - kernels: paged_attention_large · kw: paged-attention, hbm-bound, anti-pattern, fp8-mfma, occupancy, launch-overhead, mxfp4, memory-bound, gfx950

## attention_decode
- [gfx950 · decode] Collapse host dispatch first, then per-grid-density launch tuning, mask hoisting and per-regime constexpr clones: ~1.58x geomean on paged decode attention ★★ — (dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: launch-overhead, host-dispatch, decode, constexpr-promotion, paged-attention, triton, launch-tuning
- [gfx950 · decode] Clearing the .cg non-temporal hint on once-read KV tile loads in paged attention decode: bit-identical, +8.2% geomean, carried by bandwidth-bound cases. ★★ — (drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, paged-attention, cache-modifier, kv-cache, memory-bound, triton
- [gfx950 · decode] On latency-floored paged attention decode at ~1 WG/CU, cache-modifier, sw-prefetch, loop-split and graph replay all measured <=1.00x. ★★ — (four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: anti-pattern, cache-modifier, software-prefetch, graph-replay, launch-overhead, attention-decode, paged-kv, latency-bound, gfx950
- [gfx950 · decode] With a lean launcher and an MFMA-optimal tile, decode graph capture, manual SW pipelining and occupancy recovery all measured at or below 1.00x. ★★ — (four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, launch-overhead, graph-capture, software-pipelining, occupancy, wave-quantization, anti-pattern
- [gfx950 · decode] Launch-meta is the primary lever on latency-bound paged grouped-attention decode; unpinning waves_per_eu beats a pinned hint, numerically exact. ★★ — (launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: launch-meta, num-stages, waves-per-eu, occupancy, attention-decode, paged-kv, latency-bound, gfx950
- [gfx950 · decode] Cache-locality swizzle and host/graph-capture levers both returned ~1.00x on a decode attention op that was neither bandwidth- nor dispatch-bound. ★★ — (positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: attention-decode, paged-decode, launch-overhead, l2-locality, xcd-swizzle, cuda-graph, anti-pattern, gfx950
- [gfx950 · decode] Attention decode under a worst-element max_rel gate: split-KV and fp8 KV close on numerics, not speed; split count and scale granularity do not help. ★★ — (reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, split-kv, flash-decode, fp8-kv, numerics, oracle-parity, anti-pattern
- [gfx950 · decode] Cap the parallelism split at one workgroup per CU and make pipeline depth a function of launched WGs: 1.63x geomean on paged split-KV decode attention. ★★ — (split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: paged-decode, attention-decode, split-kv, num-stages, waves-per-eu, launch-shape, occupancy, gfx950, triton

## dense_gemm
- [gfx950 · both] Closed axis: on a latency-bound fp8 GEMM with zero bank conflicts, hand-scheduled double buffer, mfma reshape, split-k and graphs all return <=1.00x ★★ — (a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: anti-pattern, latency-bound, double-buffer, split-k, cuda-graph, mfma, dense-gemm, fp8, triton, gfx950
- [gfx950 · both] Bitcast a non-ISA fp8 operand type to the native one so real MFMA issues instead of per-element emulation: ~12x alone on block-scaled fp8 GEMM ★★ — (bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: fp8, mfma, dtype-bitcast, dense-gemm, emulation-fallback, block-scale, triton, gfx950
- [gfx950 · compute-bound] fp16 dense GEMM on gfx950: cut num_stages to 1, coarsen M in-body, then rewrite in Gluon with a big-BM register-staged wide-K MFMA loop — 2.66x ★★ — (gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, gluon, mfma, register-staging, lds-tiling, num-stages, m-coarsening, gfx950, fp16, compute-bound
- [gfx950 · compute-bound] gfx950 dense GEMM: with waves-per-eu pinned at 2 by the backend, four occupancy-raising directions returned ~1.00x or worse — spend the round elsewhere ★★ — (occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, occupancy, waves-per-eu, num-warps, ping-pong, mfma, gfx950, compute-bound, anti-pattern
- [gfx950 · compute-bound] For mid-size bf16 dense GEMM on gfx950, five independent code generators all land below the shipped vendor solution; that axis is closed, not underexplored. ★★ — (once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, bf16, gfx950, vendor-library, codegen, anti-pattern, split-k, occupancy, tile-geometry, roofline
- [gfx950 · compute-bound] Own the launch/dispatch layer of a frozen bf16 dense GEMM, then race Triton vs hand HIP vs tuned vendor library per shape: 4.05x geomean. ★★ — (own-the-dispatch-layer-then-race-backends-behind-it-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, bf16, gfx950, dispatch-shim, backend-routing, vendor-library, hipblaslt, launch-config, argmin-dispatch, codegen
- [gfx950 · compute-bound] Size exposed host residue first: with kernel time far above per-call host cost, a 25.6% host cut and graph replay each bought 0 on a bf16 dense GEMM. ★★ — (size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, gfx950, launch-overhead, hip-graph, host-dispatch, anti-pattern, dispatch-shim, bf16
- [gfx950 · compute-bound] Once an fp16 dense GEMM sits at ~2/3 of its MFMA roof, six host/layout/algorithm directions each returned ~1.00x; the residual gap is a backend lowering gap ★★ — (the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, roofline, mfma, split-k, launch-overhead, hip-graph, xcd-swizzle, convert-layout, gfx950, compute-bound, anti-pattern

## linear_attention
- [gfx950 · prefill] Varlen chunked linear-attention: audit the baseline grid first — most of a 28.9x headline was a B-fold redundant grid; 2.45x when deduped both sides ★★ — (audit-the-baseline-launch-grid-before-believing-a-large-head-linear-attention-gfx950-prefill.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: linear-attention, varlen, grid-dedup, launch-overhead, frozen-baseline, harness-artifact, gfx950
- [gfx950 · memory-bound] Once a kernel sits at its store roofline, occupancy lift, persistent grid-stride, finer store-skip granularity and graph replay all returned ~1.00x or regressed ★★ — (axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: anti-pattern, closed-axis, occupancy, persistent-kernel, grid-stride, graph-replay, launch-overhead, store-bandwidth, memory-bound, gfx950, linear-attention
- [gfx950 · launch-bound] A caller grid whose kernel guards to the diagonal still dispatches ~98% empty workgroups; a host shim collapsing that dim was the largest single win ★★ — (collapse-a-redundant-launch-grid-instead-of-guarding-inside--linear-attention-gfx950-launch-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: launch-overhead, grid-collapse, host-shim, empty-workgroups, varlen, linear-attention, gfx950, triton
- [gfx950 · prefill] gfx950 Triton small-grid ops: graph capture, config/occupancy ladders, cache policy and k-loop restructuring all returned ~1.00x or worse ★★ — (host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: anti-pattern, hip-graph, occupancy, config-sweep, cache-modifier, triton-pipeliner, launch-overhead, gfx950
- [gfx950 · prefill] Chunked linear-attention gfx950: hoist scales out of the contraction, share the Gram matrix across the GQA group, write-through the stores — ~1.65x stacked ★★ — (shorten-the-load-to-dot-chain-before-chasing-bytes-linear-attention-gfx950-prefill.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: linear-attention, dependency-chain, gqa-head-sharing, cache-modifier, loop-hoisting, mfma-tiling, gfx950
- [gfx950 · memory-bound] Non-temporal '.cs' cache_modifier on write-once output stores lifts store bandwidth on gfx950 store-bound kernels; a cache-policy win, not vectorization ★★ — (streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: cache-modifier, non-temporal-store, store-bandwidth, memory-bound, linear-attention, gfx950, roofline, triton

## memory_movement
- [gfx950 · both] Dispatch-bound Triton scatter/fill: cached direct-launch object + re-grid + write-through store gives ~2.53x, flat across 32x the work ★★ — (collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, host-runtime, dispatch-bound, triton, memory-movement, grid-fill, cache-modifier, gfx950
- [gfx950 · both] On a dispatch-bound tiny op, graph replay, side streams, launch-arg slimming and harness-constant attacks all returned <=1.00x on gfx950 ★★ — (four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both.md)
  - kernels: write_req_to_token_pool_triton · kw: anti-pattern, closed-axis, hip-graph, graph-replay, dispatch-bound, host-runtime, launch-overhead, memory-movement, measurement-floor, gfx950
- [gfx950 · launch-bound] Closed axis: on a dispatch-bound tiny copy, four GPU-side and three extra host-submit directions all returned ~1.00x; only the raw launch path moved ★★ — (gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: anti-pattern, launch-overhead, dispatch-bound, tiny-kernel, memory-movement, block-size, num-warps, host-submit
- [gfx950 · launch-bound] Dispatch-bound tiny index-scatter: raw ctypes hipModuleLaunchKernel with pre-packed params replaces the Triton Python launcher, ~2.6x per-case ★★ — (raw-driver-launch-for-a-dispatch-bound-copy-op-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, dispatch-bound, host-submit, tiny-kernel, memory-movement, hip-graph, ctypes

## moe_grouped_gemm
- [gfx950 · mixed] Weight-side fp4 pays; activation-side narrowing closed three ways - parity floor, missing fp6 lowering, iid synthetic inputs with no outliers ★★ — (activation-narrowing-is-gated-by-parity-and-by-the-benchmark-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, grouped-gemm, fp4, fp6, weight-quantization, parity-gate, anti-pattern, dot-scaled
- [gfx950 · compute-bound] Dequant-VALU/latency-bound int4 MoE grouped GEMM on gfx950: eight knob and rewrite axes all measured flat or negative (~1.00x) - low-prior directions ★★ — (axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_int4_w4a16 · kw: moe, grouped-gemm, int4, weight-only-quant, w4a16, anti-pattern, closed-axis, split-k, num-warps, mfma-nonkdim, cuda-graph, vgpr-pressure, compute-bound
- [gfx950 · mixed] At occupancy 1 with a dequant->MFMA dependency chain, pipeline-depth and occupancy levers return ~1.00x or regress; a regressing double-buffer is the tell ★★ — (diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, grouped-gemm, fp4, software-prefetch, num-stages, occupancy, l2-residency, dep-chain, anti-pattern
- [gfx950 · mixed] Store the MoE grouped-GEMM weight operand as e2m1 fp4 consumed natively by MFMA, then nonkdim=16 + XCD de-interleave: ~42x isolated on gfx950 ★★ — (narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, grouped-gemm, fp4, dot-scaled, mfma-nonkdim16, xcd-partitioning, l2-residency, weight-quantization
- [gfx950 · prefill] Split an int4 W4A16 MoE grouped GEMM into per-n-width Triton entries picked by a host launcher shim; the shim then owns each arm's launch constants. ★★ — (one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe-grouped-gemm, w4a16, int4-dequant, split-entry, launch-config, waves-per-eu, num-warps, triton, gfx950
- [gfx950 · compute-bound] Per-M-bucket host-side launch-config retune on int4 W4A16 MoE grouped GEMM: 3.33x weighted, per-case 2.58-3.89x, kernel body byte-identical ★★ — (per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_int4_w4a16 · kw: moe, grouped-gemm, int4, weight-only-quant, w4a16, launch-config, host-tuning, m-bucket, num-warps, block-size-k, compute-bound
- [gfx950 · prefill] Counter-guided directions (bank conflicts, VALU, barriers, occupancy, empty CTAs, traffic) returned ~1.00x or worse; time a deletion control first. ★★ — (price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe-grouped-gemm, w4a16, int4-dequant, occupancy, lds-tiling, counter-falsification, anti-pattern, launch-config, gfx950
- [gfx950 · prefill] Amortize int4 dequant by reusing one dequantised weight tile across several row-blocks; widen the dot COUNT along M, not the tile extent: ~+24% twice. ★★ — (share-the-dequantised-weight-tile-across-row-blocks-widen-m--moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe-grouped-gemm, w4a16, int4-dequant, fusion-width, lds-tiling, triton, gfx950

## moe_router_topk
- [gfx950 · both] Tiny dispatch-bound Triton op: memoize compile, bake launch opts, call the C launch entry directly - ~1.96x geomean, largest on the smallest case ★★ — (bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both.md)
  - kernels: _topk_forward · kw: launch-overhead, host-runtime, dispatch-bound, triton, moe-router, topk, memoization, gfx950
- [gfx950 · both] Device-side rewrites of a small MoE router top-k (selection topk, pack, BLOCK_M, whole-op rewrite) all returned ~1.00x on gfx950; the win is host-side ★★ — (the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both.md)
  - kernels: _topk_forward · kw: anti-pattern, moe-router, topk, dispatch-bound, static-isa-screen, launch-overhead, triton, gfx950

## quantize_cast
- [gfx950 · both] On a VALU-bound quant cast, a bit-exact reciprocal + FMA replacing per-element division cut VALU/wave 1216->768 for 1.16x, with format constants folded in ★★ — (divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both.md)
  - kernels: _per_token_group_quant_fp8 · kw: quantize-cast, valu-bound, reciprocal, division, bit-exact, fp8, gated-lever, gfx950
- [gfx950 · memory-bound] Above ~60% of nameplate HBM, six bandwidth directions all returned ~1.00x on an fp8 quant cast; the store already lowered to one 128-bit instruction ★★ — (near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: memory-bound, quantize-cast, fp8, closed-axis, cache-modifier, num-warps, tiling, store-vectorization, assembly-inspection
- [gfx950 · memory-bound] Export a launcher object with the runner's __getitem__(grid) shape to re-tile a frozen num_warps=1 launch: 2.29x geomean, bit-exact, on memory-bound fp8 quant ★★ — (reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-config, wrapper-relaunch, quantize-cast, fp8, memory-bound, num-warps, tiling, bit-exact, cache-modifier
- [gfx950 · both] Non-OCP fp8 output makes the compiler emulate the cast in software; native packed convert + bitcast cuts VALU/wave 852->338 on a quant cast ★★ — (software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both.md)
  - kernels: _per_token_group_quant_fp8 · kw: fp8, quantize-cast, dtype-emulation, valu-bound, native-convert, bitcast, bit-exact, gfx950

## quantized_gemm
- [gfx950 · mixed] Arg-plan replay beats device-graph capture at low dispatch counts (13.5% of geomean); its free extra dispatch funds a host restage of a scale operand ★★ — (arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: launch-overhead, host-runtime, graph-replay, quantized-gemm, scale-operand, block-scale, cache-line
- [gfx950 · compute-bound] Block-scaled fp8 GEMM, gfx950: hw-cvt upcast + rank-1 scale collapse + 2-deep dot overlap lift a dequant-bound inner loop ~1.53x over a tuned seed ★★ — (collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: fp8, block-scale, dequant, quantized-gemm, mfma, ilp, unroll, triton
- [gfx950 · compute-bound] Fold and hoist block scales out of the fp8 GEMM K-loop until the inner loop is a plain non-scaled MFMA: 20.2x per-case stacked on gfx950 ★★ — (de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound.md)
  - kernels: gemm_a8w8_blockscale · kw: fp8, block-scale, quantized-gemm, mfma, dequant-hoist, k-loop, l2-swizzle, hip-graph, gfx950
- [gfx950 · compute-bound] Above a tuned block-scaled fp8 GEMM on gfx950 five axes returned ~1.00x: occupancy raise, Gluon/HIP ping-pong, host graph capture, body microtune, tile shrink ★★ — (five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: anti-pattern, occupancy, ilp, quantized-gemm, fp8, block-scale, launch-overhead, tile-size, gfx950
- [gfx950 · compute-bound] Legacy-flavour fp8 operands get silently emulated in fp16 on CDNA4; a zero-copy bit reinterpretation to the native fp8 type engages the matrix core, ~7.8x ★★ — (reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: fp8, mfma, quantized-gemm, bit-reinterpret, emulation-fallback, isa-census, block-scale
- [gfx950 · small-batch] Tiny-M block-scaled fp8 GEMM, gfx950: split-K=2 doubles grid fill with a fused reduce; deeper split-K and a narrower N tile both lose ★★ — (split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: split-k, grid-fill, skinny-m, quantized-gemm, fp8, block-scale, tile-size, triton
- [gfx950 · compute-bound] Once the fp8 GEMM K-loop is scale-free, the latency-hiding axes (num_stages, bigger tiles, nonkdim 32, VGPR shave, LDS bypass) all return <=1.0x on gfx950 ★★ — (the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound.md)
  - kernels: gemm_a8w8_blockscale · kw: fp8, quantized-gemm, mfma, occupancy, lds-tiling, num-stages, closed-axis, gfx950
- [gfx950 · compute-bound] Occupancy, geometry, split-K, barriers, VALU count, epilogue and a Gluon rewrite all returned <=1.00x; price the library floor vs a scale-free floor first ★★ — (where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: anti-pattern, closed-axis, roofline, split-k, occupancy, quantized-gemm, isa-census, block-scale

## topk_router
- [gfx950 · both] Tiny router select whose wall time is flat across a 32x row spread is host-marshaling floored: cached launch closure + steady-state gives 1.9-2.3x per case. ★★ — (dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both.md)
  - kernels: _topk_forward · kw: launch-overhead, host-runtime, dispatch-bound, triton, small-batch, top-k, moe-router, register-math
- [gfx950 · both] Graph capture around one tiny launch replays ~2x slower than a direct launch at both host layers — a closed axis for dispatch-bound single-kernel ops. ★★ — (graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both.md)
  - kernels: _topk_forward · kw: launch-overhead, host-runtime, dispatch-bound, graph-capture, triton, small-batch, moe-router, anti-pattern

## method
- [gfx950 · n/a] On a power-capped GPU, delete-the-work oracles and batched timers lie: per-launch event pairs and entropy-preserving probes cut in-batch control spread ~15x ★★ — (a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: ab-protocol, measurement, power-cap, oracle, triton, gfx950, anti-pattern, bit-exact
- [gfx950 · n/a] On gfx950/CDNA4 occupancy divides one summed ArchVGPR+AGPR pool, so an AGPR-accumulator occupancy escape on an fp32-accum MFMA GEMM cannot exist ★★ — (cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: occupancy, agpr, vgpr, mfma, accumulator, raw-hip, anti-pattern, gfx950, grouped-gemm
- [gfx950 · n/a] Force a rebuild and re-measure the head in-session: a stale resident binary and file-disjoint stacking each produced multi-round phantom results ★★ — (force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a.md)
  - kw: measurement-rig, ab-methodology, stale-binary, stacking, noise-floor, counter-guided, gfx950
- [gfx950 · decode] Hand-count traffic and time a math-stripped read-only twin before staffing a memory round: it closed the largest open lane here at zero patch. ★★ — (hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: roofline, profiler-error, read-only-twin, memory-bound, anti-pattern, attention-decode, method
- [gfx950 · n/a] Census the ISA per region (prologue / main loop / epilogue) on CPU before tuning the hot loop: the two unexamined regions carried the wins the loop refused ★★ — (per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a.md)
  - kw: isa-census, profiling-method, prologue, epilogue, hot-loop, cpu-locate-gpu-price, serialisation, composable-kernel
- [gfx950 · n/a] In a JIT-built frozen C++ vendor stack, header edits may not rebuild, gitignored run dirs hide the diff, and the auto improvement flag false-negatived wins ★★ — (prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a.md)
  - kw: method, jit-rebuild, composable-kernel, verification, false-negative, frozen-baseline, moe, grouped-gemm
- [gfx950 · n/a] A lever shelved at ~1.01x can pay 1.84x once a bigger fix relieves register pressure: re-measure shelved partials on top of each new incumbent ★★ — (re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: method, register-pressure, bottleneck-shift, dense-gemm, fp8, block-scale, triton, gfx950
- [gfx950 · compute-bound] On a partitioned GPU, percent-of-peak against full-chip nameplate under-reads by the CU ratio; rescale to the exposed CU count before calling the gap headroom ★★ — (scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound.md)
  - kw: roofline, percent-of-peak, partition, measurement, mfma, gfx950

## keyword vocabulary (generated — REUSE these before coining a new term)
gfx950(37) · anti-pattern(25) · launch-overhead(21) · triton(18) · occupancy(14) · fp8(13) · mfma(11) · block-scale(9) · dense-gemm(9) · cache-modifier(8) · dispatch-bound(8) · memory-bound(8) · quantized-gemm(8) · grouped-gemm(7) · num-warps(7) · closed-axis(6) · hip-graph(6) · host-runtime(6) · moe(6) · roofline(6) · split-k(6) · attention-decode(5) · compute-bound(5) · launch-config(5) · linear-attention(5) · num-stages(5) · w4a16(5) · bit-exact(4) · decode(4) · graph-replay(4) · lds-tiling(4) · memory-movement(4) · moe-router(4) · paged-attention(4) · quantize-cast(4) · waves-per-eu(4) · attention(3) · bf16(3) · cuda-graph(3) · fp4(3) · int4-dequant(3) · isa-census(3) · latency-bound(3) · method(3) · moe-grouped-gemm(3) · codegen(2) · composable-kernel(2) · dispatch-shim(2) · dot-scaled(2) · emulation-fallback(2) · frozen-baseline(2) · graph-capture(2) · grid-fill(2) · hbm-bound(2) · host-dispatch(2) · host-submit(2) · ilp(2) · int4(2) · l2-residency(2) · measurement(2) · paged-decode(2) · paged-kv(2) · small-batch(2) · software-prefetch(2) · split-kv(2) · store-bandwidth(2) · tile-size(2) · tiling(2) · tiny-kernel(2) · topk(2) · valu-bound(2) · varlen(2) · vendor-library(2) · weight-only-quant(2) · weight-quantization(2) · xcd-swizzle(2) · ab-methodology · ab-protocol · accumulator · agpr · argmin-dispatch · assembly-inspection · backend-routing · bit-reinterpret · bitcast · block-size · block-size-k · bottleneck-shift · cache-line · config-sweep · constexpr-promotion · convert-layout · counter-falsification · counter-guided · cpu-locate-gpu-price · ctypes · dep-chain · dependency-chain · dequant · dequant-hoist · division · double-buffer · dtype-bitcast · dtype-emulation · empty-workgroups · epilogue · false-negative · flash-decode · fp16 · fp6 · fp8-kv · fp8-kv-cache · fp8-mfma · fusion-width · gated-lever · gluon · gqa-head-sharing · grid-collapse · grid-dedup · grid-stride · harness-artifact · hipblaslt · host-shim · host-tuning · hot-loop · jit-rebuild · k-loop · kv-cache · kv-cache-quant · l2-locality · l2-swizzle · launch-meta · launch-shape · launch-tuning · long-context · loop-hoisting · m-bucket · m-coarsening · measurement-floor · measurement-rig · memoization · mfma-nonkdim · mfma-nonkdim16 · mfma-tiling · mxfp4 · native-convert · noise-floor · non-temporal-store · numerics · oracle · oracle-parity · parity-gate · partition · percent-of-peak · persistent-kernel · ping-pong · power-cap · profiler-error · profiling-method · prologue · raw-hip · read-only-twin · reciprocal · register-math · register-pressure · register-staging · scale-operand · serialisation · skinny-m · software-pipelining · split-entry · stacking · stale-binary · static-isa-screen · store-vectorization · tile-geometry · top-k · triton-pipeliner · unroll · verification · vgpr · vgpr-pressure · wave-quantization · wrapper-relaunch · xcd-partitioning

> ⚠ **Near-duplicate keywords** — same concept, different spelling. Pick one, edit the
> cards, regenerate:
> - topk / top-k
