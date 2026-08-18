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
- [gfx950 · prefill] Sparse top-k MLA prefill on gfx950: pack heads per workgroup to delete gather amplification, then strip inner-loop VALU and decouple softmax - 3.08x geomean ★★ — (pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill.md)
  - kernels: _sparse_attn_prefill_ragged_kernel · kw: attention, triton, gfx950, tile-geometry, grid-occupancy, online-softmax, valu-bound, latency-bound, mfma-tiling, occupancy
- [gfx950 · memory-bound] On attention already at the paged-scatter HBM roofline, compute-precision, occupancy and launch-fusion each returned ~1.00x or worse; only traffic moved ★★ — (traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound.md)
  - kernels: paged_attention_large · kw: paged-attention, hbm-bound, anti-pattern, fp8-mfma, occupancy, launch-overhead, mxfp4, memory-bound, gfx950

## attention_decode
- [gfx950 · decode] KV-split parallelism on decode attention fails the elementwise oracle for reduction reorder, not kernel error; a single-partition control shows which it is ★★ — (a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: split-kv, flash-decoding, oracle-parity, reduction-order, decode, anti-pattern, paged-attention
- [gfx950 · decode] Non-temporal / cache-policy hints move a decode kernel only on read-once traffic no restructure can delete, and only when the working set exceeds cache. ★★ — (ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode.md)
  - kernels: paged_attention_ll4mi_QKV_mfma16_kernel · kw: decode, paged-attention, non-temporal-loads, cache-modifier, kv-cache, memory-bound, size-gating
- [gfx950 · decode] HIP paged decode attention on gfx950: per-tensor NT policy, then global-to-LDS DMA bought for prefetch depth, then a transposing LDS read: 1.35x weighted ★★ — (buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
  - kernels: aiter_paged_attention_ragged · kw: gfx950, attention-decode, paged-attention, paged-kv, decode, lds-staging, lds-tiling, prefetch, non-temporal-loads, cache-modifier, bank-conflict, launch-bounds, occupancy, raw-hip, memory-bound, isa-diff
- [gfx950 · decode] One workgroup walks every KV partition of a sequence: the split-K round trip and its reduce dispatch both vanish; 1.27x then 1.43x on two lanes. ★★ — (collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
  - kernels: paged_attention_ll4mi_QKV_mfma16_kernel, paged_attention_ll4mi_reduce_kernel · kw: decode, paged-attention, split-kv, dispatch-collapse, online-softmax, grid-gating, hip, partition, stacking, register-pressure
- [gfx950 · decode] Paged-KV attention decode on gfx950: host alloc/scale hoist first, then fp8 KV storage with bf16 in-register dequant, then occupancy + NT loads; 3.18x stacked. ★★ — (decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode.md)
  - kw: attention-decode, paged-kv, fp8-kv, host-overhead, non-temporal-loads, occupancy, launch-bounds, gfx950
- [gfx950 · decode] Memoize the whole per-call Python/ctypes prologue of a JIT decode-attention wrapper: cumulative 1.00x->3.15x isolated, concentrated at small batch. ★★ — (decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode.md)
  - kernels: paged_attention_ll4mi_QKV_mfma16_kernel · kw: decode, paged-attention, launch-overhead, host-wrapper, small-batch, jit, memoization
- [gfx950 · decode] Constant-byte KV tile, exact-fill the decode grid, then re-sweep launch meta on top: 1.71x and 1.84x geomean on two split-KV decode-attention lanes. ★★ — (derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_3d, kernel_unified_attention_2d, unified_attention · kw: launch-shape, split-kv, tile-size, occupancy, cu-underfill, empty-workgroups, grid-occupancy, constexpr-promotion, waves-per-eu, num-stages, sliding-window, roofline, hardware-counters, attention-decode, paged-attention, decode, cache-modifier, triton, gfx950
- [gfx950 · decode] Collapse host dispatch first, then per-grid-density launch tuning, mask hoisting and per-regime constexpr clones: ~1.58x geomean on paged decode attention ★★ — (dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: launch-overhead, host-dispatch, decode, constexpr-promotion, paged-attention, triton, launch-tuning
- [gfx950 · decode] Clearing the .cg non-temporal hint on once-read KV tile loads in paged attention decode: bit-identical, +8.2% geomean, carried by bandwidth-bound cases. ★★ — (drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, paged-attention, cache-modifier, kv-cache, memory-bound, triton
- [gfx950 · decode] On latency-floored paged attention decode at ~1 WG/CU, cache-modifier, sw-prefetch, loop-split and graph replay all measured <=1.00x. ★★ — (four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: anti-pattern, cache-modifier, software-prefetch, graph-replay, launch-overhead, attention-decode, paged-kv, latency-bound, gfx950
- [gfx950 · decode] With a lean launcher and an MFMA-optimal tile, decode graph capture, manual SW pipelining and occupancy recovery all measured at or below 1.00x. ★★ — (four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, launch-overhead, graph-capture, software-pipelining, occupancy, wave-quantization, anti-pattern
- [gfx950 · decode] Closed axis: on a co-resident decode attention kernel near its BW ceiling, WG geometry / occupancy / load-width tuning went 0 for 7 arms, ~1.00x or worse. ★★ — (geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode.md)
  - kernels: paged_attention_ll4mi_QKV_mfma16_kernel · kw: decode, paged-attention, wg-geometry, occupancy, co-residency, anti-pattern, prefetch, isa-diff
- [gfx950 · both] Read-once paged KV on gfx950: the shipped 16-byte non-temporal helper drops the nt bit; one 128-bit vector builtin load restores it for ~1.07x, all cases up. ★★ — (get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both.md)
  - kernels: paged_attention_ragged · kw: attention-decode, paged-kv, non-temporal-loads, cache-modifier, kv-cache, memory-bound, isa-inspection, gfx950
- [gfx950 · both] Near the achievable HBM ceiling, paged attention LDS-conflict/wait counters and occupancy are not levers: four body directions returned ~1.00x or regressed. ★★ — (high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)
  - kernels: paged_attention_ragged · kw: anti-pattern, closed-axis, roofline, hardware-counters, lds, bank-conflict, occupancy, waves-per-eu, split-kv, attention-decode, paged-kv, gfx950
- [gfx950 · decode] Launch-meta is the primary lever on latency-bound paged grouped-attention decode; unpinning waves_per_eu beats a pinned hint, numerically exact. ★★ — (launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: launch-meta, num-stages, waves-per-eu, occupancy, attention-decode, paged-kv, latency-bound, gfx950
- [gfx950 · decode] Closed axis: graph capture at launcher and wrapper level both lose on a launch-floored decode kernel - the replay floor exceeds the whole op. ★★ — (measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode.md)
  - kernels: paged_attention_ll4mi_QKV_mfma16_kernel · kw: decode, paged-attention, launch-overhead, graph-capture, small-batch, dispatch-collapse, anti-pattern
- [gfx950 · decode] Fuse the split-KV reduce into the attention kernel behind a padded arrival counter with arch-cheap release bits: +9.5%, then +3.0% of epilogue protocol tuning. ★★ — (one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)
  - kernels: paged_attention_ll4mi_QKV_mfma16_kernel, paged_attention_ll4mi_reduce_kernel · kw: decode, paged-attention, split-kv, dispatch-collapse, cross-workgroup, arrival-counter, coherence
- [gfx950 · decode] Cache-locality swizzle and host/graph-capture levers both returned ~1.00x on a decode attention op that was neither bandwidth- nor dispatch-bound. ★★ — (positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: attention-decode, paged-decode, launch-overhead, l2-locality, xcd-swizzle, cuda-graph, anti-pattern, gfx950
- [gfx950 · both] GPU-bound paged attention: wrapper-level HIP-graph capture and host marshalling memoization both land at or below 1.00x - the launch/host axis is closed. ★★ — (price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both.md)
  - kernels: paged_attention_ragged · kw: anti-pattern, closed-axis, launch-overhead, hip-graph, graph-replay, host-runtime, attention-decode, paged-kv, gfx950
- [gfx950 · decode] Closed axis: at ~99.8% of the DRAM roof, decode-attention reduce fusion, grid right-sizing, occupancy and cache-residency steering all return ~1.00x ★★ — (price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)
  - kernels: aiter_paged_attention_ragged · kw: gfx950, attention-decode, paged-attention, decode, anti-pattern, closed-axis, kernel-fusion, dispatch-collapse, occupancy, grid-occupancy, empty-workgroups, l2-residency, roofline, oracle, memory-bound
- [gfx950 · decode] Attention decode under a worst-element max_rel gate: split-KV and fp8 KV close on numerics, not speed; split count and scale granularity do not help. ★★ — (reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, split-kv, flash-decode, fp8-kv, numerics, oracle-parity, anti-pattern
- [gfx950 · decode] Fusing the split-KV reduce into the attention epilogue is a closed axis on gfx950: the cross-block fence L2-serializes the grid for a ~7x regression. ★★ — (split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode.md)
  - kw: attention-decode, paged-kv, split-kv, kernel-fusion, threadfence, dispatch-overhead, anti-pattern, gfx950
- [gfx950 · decode] Cap the parallelism split at one workgroup per CU and make pipeline depth a function of launched WGs: 1.63x geomean on paged split-KV decode attention. ★★ — (split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: paged-decode, attention-decode, split-kv, num-stages, waves-per-eu, launch-shape, occupancy, gfx950, triton
- [gfx950 · decode] Once host time sits far under GPU time, more launch-path work on decode attention returns 1.00x or worse — graph replay measured 0.65-0.76x of eager ★★ — (the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: launch-overhead, host-dispatch, cuda-graph, dispatch-floor, decode, anti-pattern, paged-attention

## composable
- [gfx950 · both] Composable TileLang pre/GEMM/post chain: add a hidden-dim block axis to the token-only grid, hoist the k-loop bounds guard out of the GEMM; 1.88x stacked ★★ — (fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
  - kernels: mhc_post_tilelang_kernel, mhc_pre_big_fuse_tilelang_kernel, hc_prenorm_gemm_block_m_v2_tilelang_kernel · kw: cu-underfill, grid-occupancy, loop-hoisting, tile-geometry, kernel-fusion, anti-pattern, oracle-parity, measurement-discipline, gfx950

## dense_gemm
- [gfx950 · both] Closed axis: on a latency-bound fp8 GEMM with zero bank conflicts, hand-scheduled double buffer, mfma reshape, split-k and graphs all return <=1.00x ★★ — (a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: anti-pattern, latency-bound, double-buffer, split-k, cuda-graph, mfma, dense-gemm, fp8, triton, gfx950
- [gfx950 · both] Bitcast a non-ISA fp8 operand type to the native one so real MFMA issues instead of per-element emulation: ~12x alone on block-scaled fp8 GEMM ★★ — (bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: fp8, mfma, dtype-bitcast, dense-gemm, emulation-fallback, block-scale, triton, gfx950
- [gfx950 · decode] On a launch-floored decode GEMV, host-side alloc reuse plus a lock-free last-hit shortcut gave ~1.08-1.10x while every device-side direction returned ~1.00x. ★★ — (cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode.md)
  - kernels: wvSplitK_hf_sml_ · kw: launch-overhead, dispatch-floor, host-runtime, decode, gemv, caching, timing-drift
- [gfx950 · compute-bound] fp16 dense GEMM on gfx950: cut num_stages to 1, coarsen M in-body, then rewrite in Gluon with a big-BM register-staged wide-K MFMA loop — 2.66x ★★ — (gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, gluon, mfma, register-staging, lds-tiling, num-stages, m-coarsening, gfx950, fp16, compute-bound
- [gfx950 · decode] Capturing one tiny decode dispatch into a HIP graph replays ~2x slower than eager; a Python signature-cache wrapper on the timed path also net-regresses. ★★ — (graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode.md)
  - kernels: wvSplitK_hf_sml_ · kw: hip-graph, launch-overhead, dispatch-floor, decode, gemv, anti-pattern, wrapper-overhead
- [gfx950 · decode] When the harness wall floors above device time, a correct roofline rewrite of a tiny decode GEMV scores ~1.00x; measure the launch floor first. ★★ — (measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode.md)
  - kernels: wvSplitK_hf_sml_ · kw: dispatch-floor, launch-overhead, decode, skinny-m, gemv, roofline, cu-underfill, anti-pattern
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
- [gfx950 · both] Remap a CK block-scale MoE grouped GEMM pipeline to 32x32 MFMA (+CShuffle epilogue, +A-LDS pad): 1.47x isolated, all three token cases ★★ — (32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both.md)
  - kernels: moe_stage1 · kw: moe, grouped-gemm, mfma, block-scale, composable-kernel, cshuffle-epilogue, lds-padding, fp8
- [gfx950 · mixed] With K trips pinned by the operator contract and full-rate MFMA already emitted, inner-loop levers all returned ~1.00x; the wins sat in mapping and routing. ★★ — (a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fmoe_fp8_blockscale_g1u1 · kw: anti-pattern, closed-axis, instruction-schedule, double-buffering, atomic-combine, moe, grouped-gemm, fp8-blockscale
- [gfx950 · mixed] Weight-side fp4 pays; activation-side narrowing closed three ways - parity floor, missing fp6 lowering, iid synthetic inputs with no outliers ★★ — (activation-narrowing-is-gated-by-parity-and-by-the-benchmark-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, grouped-gemm, fp4, fp6, weight-quantization, parity-gate, anti-pattern, dot-scaled
- [gfx950 · both] Amortize (do not shrink) int4 weight dequant in a MoE grouped GEMM: one dequantized B tile per several M blocks, tails split not padded; 2.61x geomean ★★ — (amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: int4, w4a16, moe-grouped-gemm, dequant, amortization, triton, gfx950, bit-exact, tile-geometry
- [gfx950 · compute-bound] Dequant-VALU/latency-bound int4 MoE grouped GEMM on gfx950: eight knob and rewrite axes all measured flat or negative (~1.00x) - low-prior directions ★★ — (axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_int4_w4a16 · kw: moe, grouped-gemm, int4, weight-only-quant, w4a16, anti-pattern, closed-axis, split-k, num-warps, mfma-nonkdim, cuda-graph, vgpr-pressure, compute-bound
- [gfx950 · both] On a register-bound int4 MoE grouped GEMM, VALU cuts, occupancy, tile geometry, split-K, grid compaction and a hand-written port all return ~1.00x or worse ★★ — (axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: int4, w4a16, moe-grouped-gemm, dequant, occupancy, tile-geometry, split-k, anti-pattern, gfx950, triton
- [gfx950 · both] fp8 block-scale MoE grouped GEMM: pipeline-version row + scale-metadata staging (gather packing, LDS slabs, batched prologue fill) compounded to 1.84x ★★ — (block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
  - kernels: kernel_moe_gemm, ck_moe_stage1_gemm · kw: moe, grouped-gemm, block-scale, fp8, lds-staging, pipeline-version, xcd-swizzle, epilogue, prologue, composable-kernel
- [gfx950 · small-batch] Closed axis: split-K/KBatch on a dep-stall-bound grouped MoE GEMM whose grid already spans ~20 block-waves regressed monotonically (0.85x) ★★ — (count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch.md)
  - kernels: moe_stage1 · kw: moe, grouped-gemm, split-k, grid-occupancy, dep-stall, anti-pattern, composable-kernel
- [gfx950 · both] Tune the per-M-bucket launch config first, then buy the tail by deleting satellite dispatches into grids that already exist: 1.26x on bf16 Triton fused-MoE ★★ — (delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel, moe_align_block_size · kw: moe, grouped-gemm, dispatch-collapse, kernel-fusion, launch-overhead, m-bucket, prologue, bf16, triton, gfx950
- [gfx950 · mixed] Four dequant/feed-path directions all returned ~1.00x or worse on an occupancy-2 VGPR-floored int4 MoE GEMM: that axis is closed on gfx950 ★★ — (dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe, grouped-gemm, int4, dequant, num-stages, cache-modifier, raw-hip, occupancy, anti-pattern, gfx950
- [gfx950 · prefill] fp8 block-scale MoE grouped GEMM, gfx950: derive the consumer tile in-file and renegotiate the A-scale contract per-row -> ~4.95x geomean. ★★ — (derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel · kw: moe-grouped-gemm, fp8-block-scale, gfx950, super-tile, occupancy, quantization-contract, xcd-remap, async-copy, paired-ab-rig
- [gfx950 · mixed] At occupancy 1 with a dequant->MFMA dependency chain, pipeline-depth and occupancy levers return ~1.00x or regress; a regressing double-buffer is the tell ★★ — (diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, grouped-gemm, fp4, software-prefetch, num-stages, occupancy, l2-residency, dep-chain, anti-pattern
- [gfx950 · small-batch] Closed axis: HIP-graph capture/replay at the wrapper layer lost on every case (1.006-1.029x slower) when host enqueue is already async-hidden ★★ — (graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch.md)
  - kernels: moe_stage1 · kw: launch-overhead, hip-graph, host-runtime, anti-pattern, moe, grouped-gemm
- [gfx950 · prefill] gfx950 grouped GEMM: graph capture, persistent/split-N, GSMxXCD sweeps and a vendor dense-GEMM swap all priced at ~1.00x or a loss. ★★ — (host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel · kw: moe-grouped-gemm, gfx950, launch-overhead, hip-graph, xcd-remap, persistent-kernel, aiter, anti-pattern, paired-ab-rig
- [gfx950 · prefill] gfx950 fp8 grouped GEMM at MFMA/load interlock: cutting real work from a co-resident LDS/VALU pipe returned ~1.00x or worse, three ways. ★★ — (instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel · kw: moe-grouped-gemm, fp8-block-scale, gfx950, mfma-interlock, anti-pattern, occupancy, paired-ab-rig
- [gfx950 · mixed] Chunk-interleaving the linear workgroup id inverts the hardware XCD round-robin and restores weight reuse inside each XCD's L2 slice. ★★ — (invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fmoe_fp8_blockscale_g1u1 · kw: xcd-swizzle, l2-reuse, workgroup-mapping, bucket-routing, moe, grouped-gemm, fp8-blockscale
- [gfx950 · both] Widen MFMA 16x16->32x32 on a CK block-scale MoE grouped GEMM (with matching host weight re-preshuffle) and pad A-LDS: ~1.25x, bit-exact. ★★ — (mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
  - kernels: moe_stage2 · kw: mfma, lds-padding, bank-conflict, moe, block-scale, composable-kernel, fp8, preshuffle, grouped-gemm
- [gfx950 · mixed] Occupancy, spill, LDS capacity, atomic write amplification and code size all return ~1.00x here; wins live in issue, dependency structure and LDS bank phase ★★ — (nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_stage2 · kw: anti-pattern, occupancy, register-spill, lds-capacity, atomics, code-size, counter-guided, moe-grouped-gemm, fp8-blockscale, gfx950
- [gfx950 · mixed] Store the MoE grouped-GEMM weight operand as e2m1 fp4 consumed natively by MFMA, then nonkdim=16 + XCD de-interleave: ~42x isolated on gfx950 ★★ — (narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, grouped-gemm, fp4, dot-scaled, mfma-nonkdim16, xcd-partitioning, l2-residency, weight-quantization
- [gfx950 · prefill] Split an int4 W4A16 MoE grouped GEMM into per-n-width Triton entries picked by a host launcher shim; the shim then owns each arm's launch constants. ★★ — (one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe-grouped-gemm, w4a16, int4-dequant, split-entry, launch-config, waves-per-eu, num-warps, triton, gfx950
- [gfx950 · mixed] Epilogue LDS row stride at a multiple of the 32-bank period: one pad element collapses the conflict counter, ~+3.3% on fp8 blockscale grouped MoE GEMM ★★ — (pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_stage2 · kw: lds-bank-conflict, lds-tiling, epilogue, cshuffle, counter-guided, moe-grouped-gemm, fp8-blockscale, gfx950
- [gfx950 · mixed] Per-bucket BLOCK_M/num_warps/nonkdim plus byte-once dual-nibble int4 dequant stacks to ~4.6x on int4-weight MoE grouped GEMM, gfx950 ★★ — (per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe, grouped-gemm, int4, dequant, block-m, num-warps, mfma-nonkdim, per-bucket-tuning, occupancy, gfx950
- [gfx950 · compute-bound] Per-M-bucket host-side launch-config retune on int4 W4A16 MoE grouped GEMM: 3.33x weighted, per-case 2.58-3.89x, kernel body byte-identical ★★ — (per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_int4_w4a16 · kw: moe, grouped-gemm, int4, weight-only-quant, w4a16, launch-config, host-tuning, m-bucket, num-warps, block-size-k, compute-bound
- [gfx950 · mixed] Re-route the down-proj stage to the narrow-N V1 pipeline (32x32 MFMA) and shrink the CShuffle M-cluster to 1 XDL/wave: ~1.29x on fp8 block-scaled MoE GEMM ★★ — (pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_gemm_fp8_blockscale · kw: moe, grouped-gemm, fp8-blockscale, composable-kernel, mfma, cshuffle, pipeline-variant, tile-shape, compute-bound
- [gfx950 · prefill] Counter-guided directions (bank conflicts, VALU, barriers, occupancy, empty CTAs, traffic) returned ~1.00x or worse; time a deletion control first. ★★ — (price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe-grouped-gemm, w4a16, int4-dequant, occupancy, lds-tiling, counter-falsification, anti-pattern, launch-config, gfx950
- [gfx950 · mixed] Shape-conditional wins that read as noise on a 3-case geomean cancel when shipped globally; routed per token bucket they add exactly. ★★ — (route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fmoe_fp8_blockscale_g1u1 · kw: bucket-routing, host-runtime, moe, grouped-gemm, tile-selection, sub-variance, fp8-blockscale
- [gfx950 · prefill] Amortize int4 dequant by reusing one dequantised weight tile across several row-blocks; widen the dot COUNT along M, not the tile extent: ~+24% twice. ★★ — (share-the-dequantised-weight-tile-across-row-blocks-widen-m--moe-grouped-gemm-gfx950-prefill.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe-grouped-gemm, w4a16, int4-dequant, fusion-width, lds-tiling, triton, gfx950
- [gfx950 · mixed] Frozen vendor fp8 block-scaled MoE GEMM at ~0.37 of roof: LDS padding, epilogue rewrite, vector width, HIP-graph and fp4 weights all returned ~1.00x ★★ — (where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_gemm_fp8_blockscale · kw: moe, grouped-gemm, fp8-blockscale, composable-kernel, anti-pattern, closed-axis, hip-graph, lds-padding, occupancy, compute-bound

## moe_router_topk
- [gfx950 · both] Tiny dispatch-bound Triton op: memoize compile, bake launch opts, call the C launch entry directly - ~1.96x geomean, largest on the smallest case ★★ — (bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both.md)
  - kernels: _topk_forward · kw: launch-overhead, host-runtime, dispatch-bound, triton, moe-router, topk, memoization, gfx950
- [gfx950 · both] Device-side rewrites of a small MoE router top-k (selection topk, pack, BLOCK_M, whole-op rewrite) all returned ~1.00x on gfx950; the win is host-side ★★ — (the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both.md)
  - kernels: _topk_forward · kw: anti-pattern, moe-router, topk, dispatch-bound, static-isa-screen, launch-overhead, triton, gfx950

## quantize_cast
- [gfx950 · both] On a VALU-bound quant cast, a bit-exact reciprocal + FMA replacing per-element division cut VALU/wave 1216->768 for 1.16x, with format constants folded in ★★ — (divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both.md)
  - kernels: _per_token_group_quant_fp8 · kw: quantize-cast, valu-bound, reciprocal, division, bit-exact, fp8, gated-lever, gfx950
- [gfx950 · both] Fuse a 3-dispatch dynamic per-tensor quant into one kernel behind a 2-round-trip tag-slot grid barrier: 1.73x weighted, every case up. ★★ — (fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
  - kernels: dynamic_per_tensor_quant, fused_dynamic_per_tensor_quant_kernel · kw: quantize-cast, fp8, dispatch-collapse, kernel-fusion, cross-workgroup, arrival-counter, coherence, raw-hip, latency-bound, grid-occupancy, profiler-error, paired-ab-rig, gfx950
- [gfx950 · memory-bound] Above ~60% of nameplate HBM, six bandwidth directions all returned ~1.00x on an fp8 quant cast; the store already lowered to one 128-bit instruction ★★ — (near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: memory-bound, quantize-cast, fp8, closed-axis, cache-modifier, num-warps, tiling, store-vectorization, assembly-inspection
- [gfx950 · memory-bound] Export a launcher object with the runner's __getitem__(grid) shape to re-tile a frozen num_warps=1 launch: 2.29x geomean, bit-exact, on memory-bound fp8 quant ★★ — (reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-config, wrapper-relaunch, quantize-cast, fp8, memory-bound, num-warps, tiling, bit-exact, cache-modifier
- [gfx950 · both] Non-OCP fp8 output makes the compiler emulate the cast in software; native packed convert + bitcast cuts VALU/wave 852->338 on a quant cast ★★ — (software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both.md)
  - kernels: _per_token_group_quant_fp8 · kw: fp8, quantize-cast, dtype-emulation, valu-bound, native-convert, bitcast, bit-exact, gfx950

## quantized_gemm
- [gfx950 · mixed] Arg-plan replay beats device-graph capture at low dispatch counts (13.5% of geomean); its free extra dispatch funds a host restage of a scale operand ★★ — (arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: launch-overhead, host-runtime, graph-replay, quantized-gemm, scale-operand, block-scale, cache-line
- [gfx950 · compute-bound] Five directions returned ~1.00x or worse on a parity-gated fp8 GEMM: occupancy, manual SW pipelining, B-side LDS reshape, graph replay, reduction reassociation. ★★ — (axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: fp8, block-scaled-gemm, bit-exact-gate, anti-pattern, occupancy, software-pipelining, lds-staging, gfx950
- [gfx950 · compute-bound] Block-scaled fp8-FNUZ GEMM on gfx950: replacing the emitted per-element FNUZ upcast with a bit-exact packed integer re-encode dominates the win. ★★ — (bit-exact-integer-re-encode-of-the-fp8-fnuz-upcast-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: fp8, fnuz, dequant, block-scaled, bit-exact, packed-valu, quantized-gemm, gfx950, triton
- [gfx950 · compute-bound] Block-scaled fp8 GEMM, gfx950: hw-cvt upcast + rank-1 scale collapse + 2-deep dot overlap lift a dequant-bound inner loop ~1.53x over a tuned seed ★★ — (collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: fp8, block-scale, dequant, quantized-gemm, mfma, ilp, unroll, triton
- [gfx950 · compute-bound] Fold the fp8 format-recovery constant into the scaled-convert scale operand and lift sign from the byte's sign bit: one convert per pair, shorter dep chain. ★★ — (collapse-the-fp8-dequant-chain-into-one-scaled-convert-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: fp8, block-scaled-gemm, bit-exact-gate, critical-path, dequant, packed-loads, lds-staging, gfx950
- [gfx950 · compute-bound] Fold and hoist block scales out of the fp8 GEMM K-loop until the inner loop is a plain non-scaled MFMA: 20.2x per-case stacked on gfx950 ★★ — (de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound.md)
  - kernels: gemm_a8w8_blockscale · kw: fp8, block-scale, quantized-gemm, mfma, dequant-hoist, k-loop, l2-swizzle, hip-graph, gfx950
- [gfx950 · compute-bound] Above a tuned block-scaled fp8 GEMM on gfx950 five axes returned ~1.00x: occupancy raise, Gluon/HIP ping-pong, host graph capture, body microtune, tile shrink ★★ — (five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: anti-pattern, occupancy, ilp, quantized-gemm, fp8, block-scale, launch-overhead, tile-size, gfx950
- [gfx950 · compute-bound] Legacy-flavour fp8 operands get silently emulated in fp16 on CDNA4; a zero-copy bit reinterpretation to the native fp8 type engages the matrix core, ~7.8x ★★ — (reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: fp8, mfma, quantized-gemm, bit-reinterpret, emulation-fallback, isa-census, block-scale
- [gfx950 · small-batch] Tiny-M block-scaled fp8 GEMM, gfx950: split-K=2 doubles grid fill with a fused reduce; deeper split-K and a narrower N tile both lose ★★ — (split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: split-k, grid-fill, skinny-m, quantized-gemm, fp8, block-scale, tile-size, triton
- [gfx950 · compute-bound] Under a bit-exact accumulation gate, coarsen the inner K sub-tile (fewer, wider dots, one linear fp32 accumulator) — pipeline-balance win, parity untouched. ★★ — (sub-k-coarsening-regroup-the-same-reduction-order-into-fewer-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: fp8, block-scaled-gemm, bit-exact-gate, critical-path, sub-k-coarsening, mfma, gfx950, num-stages
- [gfx950 · compute-bound] Once the fp8 GEMM K-loop is scale-free, the latency-hiding axes (num_stages, bigger tiles, nonkdim 32, VGPR shave, LDS bypass) all return <=1.0x on gfx950 ★★ — (the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound.md)
  - kernels: gemm_a8w8_blockscale · kw: fp8, quantized-gemm, mfma, occupancy, lds-tiling, num-stages, closed-axis, gfx950
- [gfx950 · compute-bound] Once the dequant chain is gone from a compute-bound fp8 GEMM on gfx950, LDS, barrier, occupancy and host-launch directions all return ~1.00x. ★★ — (the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: closed-axis, lds-tiling, occupancy, launch-overhead, hip-graph, quantized-gemm, gfx950, triton, measurement-discipline
- [gfx950 · compute-bound] Occupancy, geometry, split-K, barriers, VALU count, epilogue and a Gluon rewrite all returned <=1.00x; price the library floor vs a scale-free floor first ★★ — (where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: anti-pattern, closed-axis, roofline, split-k, occupancy, quantized-gemm, isa-census, block-scale

## topk_router
- [gfx950 · both] Tiny router select whose wall time is flat across a 32x row spread is host-marshaling floored: cached launch closure + steady-state gives 1.9-2.3x per case. ★★ — (dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both.md)
  - kernels: _topk_forward · kw: launch-overhead, host-runtime, dispatch-bound, triton, small-batch, top-k, moe-router, register-math
- [gfx950 · both] Graph capture around one tiny launch replays ~2x slower than a direct launch at both host layers — a closed axis for dispatch-bound single-kernel ops. ★★ — (graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both.md)
  - kernels: _topk_forward · kw: launch-overhead, host-runtime, dispatch-bound, graph-capture, triton, small-batch, moe-router, anti-pattern

## method
- [gfx950 · n/a] A/B in the grader's own case mix and session: a heavy-only subset showed +2.5% that was +0.3% when graded, and launch-floor cases swing ~20% across sessions ★★ — (a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a.md)
  - kw: measurement, noise-floor, ab-methodology, launch-overhead, thermal-drift, anti-pattern, paged-attention
- [gfx950 · n/a] On a power-capped GPU, delete-the-work oracles and batched timers lie: per-launch event pairs and entropy-preserving probes cut in-batch control spread ~15x ★★ — (a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: ab-protocol, measurement, power-cap, oracle, triton, gfx950, anti-pattern, bit-exact
- [gfx950 · launch-bound] A tiny case that will not move can be floored by the measurement bracket itself, not by the GPU; graph capture and persistent grids both scored ~1.00x there ★★ — (a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-bound, launch-overhead, closed-axis, small-batch, hip-graph, persistent-grid, quantize-cast, measurement
- [gfx950 · n/a] On gfx950/CDNA4 occupancy divides one summed ArchVGPR+AGPR pool, so an AGPR-accumulator occupancy escape on an fp32-accum MFMA GEMM cannot exist ★★ — (cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: occupancy, agpr, vgpr, mfma, accumulator, raw-hip, anti-pattern, gfx950, grouped-gemm
- [gfx950 · n/a] Force a rebuild and re-measure the head in-session: a stale resident binary and file-disjoint stacking each produced multi-round phantom results ★★ — (force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a.md)
  - kw: measurement-rig, ab-methodology, stale-binary, stacking, noise-floor, counter-guided, gfx950
- [gfx950 · launch-bound] Bimodal box throttling rejected a bit-exact tiny-kernel win five times; gate on a median of >=10 samples or an interleaved paired A/B instead ★★ — (gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: measurement-noise, ab-methodology, tiny-kernel, dispatch-bound, launch-overhead, frozen-baseline
- [gfx950 · decode] Hand-count traffic and time a math-stripped read-only twin before staffing a memory round: it closed the largest open lane here at zero patch. ★★ — (hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: roofline, profiler-error, read-only-twin, memory-bound, anti-pattern, attention-decode, method
- [gfx950 · both] For tiny ops, measure with the harness protocol plus position rotation and an A/A null: replay probes overstate ~4x and slot bias fakes ~14% ★★ — (measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both.md)
  - kernels: write_req_to_token_pool_triton · kw: method, measurement, ab-methodology, graph-replay, dispatch-bound, noise-floor, measurement-floor, negative-control, gfx950
- [gfx950 · n/a] Census the ISA per region (prologue / main loop / epilogue) on CPU before tuning the hot loop: the two unexamined regions carried the wins the loop refused ★★ — (per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a.md)
  - kw: isa-census, profiling-method, prologue, epilogue, hot-loop, cpu-locate-gpu-price, serialisation, composable-kernel
- [gfx950 · n/a] Certify or decline a sub-2% kernel effect: per-rep paired geomean plus a two-sided identity-work control and an acceptance bar fixed before measuring ★★ — (per-rep-geomean-plus-a-two-sided-negative-control-method-gfx950-n-a.md)
  - kernels: _topk_forward · kw: method, ab-harness, negative-control, sign-consistency, small-effect, dispatch-bound, gfx950
- [gfx950 · n/a] In a JIT-built frozen C++ vendor stack, header edits may not rebuild, gitignored run dirs hide the diff, and the auto improvement flag false-negatived wins ★★ — (prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a.md)
  - kw: method, jit-rebuild, composable-kernel, verification, false-negative, frozen-baseline, moe, grouped-gemm
- [gfx950 · n/a] A lever shelved at ~1.01x can pay 1.84x once a bigger fix relieves register pressure: re-measure shelved partials on top of each new incumbent ★★ — (re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: method, register-pressure, bottleneck-shift, dense-gemm, fp8, block-scale, triton, gfx950
- [gfx950 · compute-bound] After a decode/layout lever lands, re-price the knobs already dead-listed and sweep the config as a tuple: two thirds of one round win was resurrected knobs. ★★ — (re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: config-sweep, dead-list, tile-shape, num-stages, split-k, quantized-gemm, gfx950, triton, search-strategy
- [gfx950 · compute-bound] On a partitioned GPU, percent-of-peak against full-chip nameplate under-reads by the CU ratio; rescale to the exposed CU count before calling the gap headroom ★★ — (scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound.md)
  - kw: roofline, percent-of-peak, partition, measurement, mfma, gfx950
- [gfx950 · decode] A period-2 per-case timing split can be the caching allocator moving a large buffer, not a shape effect: the timing follows the pointer, not the shape ★★ — (test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
  - kernels: aiter_paged_attention_ragged · kw: measurement, harness-artifact, ab-methodology, measurement-discipline, frozen-baseline, kv-cache, attention-decode, decode, anti-pattern, gfx950, noise-floor
- [gfx950 · both] After one prebind win, four more host/launch directions returned ~1.00x on a short elementwise op; exhaustion test: submit cost vs smallest case GPU time ★★ — (the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-overhead, host-runtime, dispatch-cache, hip-graph, anti-pattern, measurement-floor, quantize-cast, gfx950

## keyword vocabulary (generated — REUSE these before coining a new term)
gfx950(68) · anti-pattern(49) · launch-overhead(36) · occupancy(30) · triton(26) · fp8(21) · grouped-gemm(20) · decode(19) · moe(19) · paged-attention(16) · mfma(15) · attention-decode(14) · hip-graph(14) · cache-modifier(13) · closed-axis(13) · block-scale(12) · memory-bound(12) · dispatch-bound(11) · host-runtime(11) · quantized-gemm(11) · moe-grouped-gemm(10) · roofline(10) · dense-gemm(9) · num-stages(9) · split-k(9) · composable-kernel(8) · num-warps(8) · paged-kv(8) · split-kv(8) · compute-bound(7) · dequant(7) · fp8-blockscale(7) · lds-tiling(7) · quantize-cast(7) · w4a16(7) · bit-exact(6) · dispatch-collapse(6) · graph-replay(6) · grid-occupancy(6) · int4(6) · measurement(6) · waves-per-eu(6) · ab-methodology(5) · kernel-fusion(5) · latency-bound(5) · launch-config(5) · linear-attention(5) · method(5) · small-batch(5) · tile-geometry(5) · attention(4) · bf16(4) · cuda-graph(4) · dispatch-floor(4) · frozen-baseline(4) · kv-cache(4) · lds-staging(4) · memory-movement(4) · moe-router(4) · noise-floor(4) · non-temporal-loads(4) · paired-ab-rig(4) · raw-hip(4) · xcd-swizzle(4) · bank-conflict(3) · bit-exact-gate(3) · block-scaled-gemm(3) · counter-guided(3) · cu-underfill(3) · empty-workgroups(3) · epilogue(3) · fp4(3) · gemv(3) · graph-capture(3) · host-dispatch(3) · int4-dequant(3) · isa-census(3) · l2-residency(3) · lds-padding(3) · measurement-discipline(3) · measurement-floor(3) · oracle-parity(3) · prologue(3) · tile-size(3) · tiny-kernel(3) · valu-bound(3) · arrival-counter(2) · bucket-routing(2) · codegen(2) · coherence(2) · config-sweep(2) · constexpr-promotion(2) · critical-path(2) · cross-workgroup(2) · cshuffle(2) · dispatch-shim(2) · dot-scaled(2) · emulation-fallback(2) · fp8-block-scale(2) · fp8-kv(2) · grid-fill(2) · hardware-counters(2) · harness-artifact(2) · hbm-bound(2) · host-submit(2) · ilp(2) · isa-diff(2) · launch-bounds(2) · launch-shape(2) · loop-hoisting(2) · m-bucket(2) · memoization(2) · mfma-nonkdim(2) · mfma-tiling(2) · negative-control(2) · online-softmax(2) · oracle(2) · paged-decode(2) · partition(2) · persistent-kernel(2) · prefetch(2) · profiler-error(2) · register-pressure(2) · skinny-m(2) · software-pipelining(2) · software-prefetch(2) · stacking(2) · store-bandwidth(2) · tile-shape(2) · tiling(2) · topk(2) · varlen(2) · vendor-library(2) · weight-only-quant(2) · weight-quantization(2) · xcd-remap(2) · ab-harness · ab-protocol · accumulator · agpr · aiter · amortization · argmin-dispatch · assembly-inspection · async-copy · atomic-combine · atomics · backend-routing · bit-reinterpret · bitcast · block-m · block-scaled · block-size · block-size-k · bottleneck-shift · cache-line · caching · co-residency · code-size · convert-layout · counter-falsification · cpu-locate-gpu-price · cshuffle-epilogue · ctypes · dead-list · dep-chain · dep-stall · dependency-chain · dequant-hoist · dispatch-cache · dispatch-overhead · division · double-buffer · double-buffering · dtype-bitcast · dtype-emulation · false-negative · flash-decode · flash-decoding · fnuz · fp16 · fp6 · fp8-kv-cache · fp8-mfma · fusion-width · gated-lever · gluon · gqa-head-sharing · grid-collapse · grid-dedup · grid-gating · grid-stride · hip · hipblaslt · host-overhead · host-shim · host-tuning · host-wrapper · hot-loop · instruction-schedule · isa-inspection · jit · jit-rebuild · k-loop · kv-cache-quant · l2-locality · l2-reuse · l2-swizzle · launch-bound · launch-meta · launch-tuning · lds · lds-bank-conflict · lds-capacity · long-context · m-coarsening · measurement-noise · measurement-rig · mfma-interlock · mfma-nonkdim16 · mxfp4 · native-convert · non-temporal-store · numerics · packed-loads · packed-valu · parity-gate · per-bucket-tuning · percent-of-peak · persistent-grid · ping-pong · pipeline-variant · pipeline-version · power-cap · preshuffle · profiling-method · quantization-contract · read-only-twin · reciprocal · reduction-order · register-math · register-spill · register-staging · scale-operand · search-strategy · serialisation · sign-consistency · size-gating · sliding-window · small-effect · split-entry · stale-binary · static-isa-screen · store-vectorization · sub-k-coarsening · sub-variance · super-tile · thermal-drift · threadfence · tile-selection · timing-drift · top-k · triton-pipeliner · unroll · verification · vgpr · vgpr-pressure · wave-quantization · wg-geometry · workgroup-mapping · wrapper-overhead · wrapper-relaunch · xcd-partitioning

> ⚠ **Near-duplicate keywords** — same concept, different spelling. Pick one, edit the
> cards, regenerate:
> - fp8-blockscale / fp8-block-scale
> - launch-bounds / launch-bound
> - topk / top-k
