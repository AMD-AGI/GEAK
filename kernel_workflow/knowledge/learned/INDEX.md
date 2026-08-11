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
- [gfx950 · mixed] Read-once paged-KV attention: one __builtin_nontemporal_load on a native 128-bit vector emits a real nt dwordx4; the shipped nt helper drops nt on gfx950. ★★ — (genuine-non-temporal-128-bit-kv-loads-when-kv-has-zero-l2-re-attention-gfx950-mixed.md)
  - kernels: paged_attention_ragged · kw: non-temporal, nontemporal-load, kv-cache, l2-reuse, vectorized-load, memory-bound, attention, gfx950, isa-check
- [gfx950 · mixed] GPU-bound op: real host-side savings and a correct wrapper HIP-graph capture measured 1.009x and 0.986x; a below-noise delta cannot be gated into a win. ★★ — (host-marshalling-and-graph-capture-buy-nothing-when-the-op-i-attention-gfx950-mixed.md)
  - kernels: paged_attention_ragged · kw: launch-overhead, hip-graph, graph-capture, host-runtime, gpu-bound, harness-noise, per-shape-gate, attention, gfx950, anti-pattern
- [gfx950 · decode] Null the kernel body (deliberately wrong) to price a whole direction in one measurement: predicted the 0% outcome of optimizing a decode dispatch ★★ — (null-the-kernel-body-before-optimizing-it-attention-gfx950-decode.md)
  - kernels: paged_attention_decode · kw: control-experiment, measurement-method, dispatch-floor, launch-overhead, decode, null-baseline
- [gfx950 · decode] Re-linearise the workgroup id so consecutive workgroups walk one request's contiguous KV: +3.1% on attention decode, scaling with workgroup count ★★ — (re-linearise-the-workgroup-id-in-the-prologue-for-address-lo-attention-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: pid-remap, l2-locality, xcd, address-locality, decode, isa-check, bijection, interleaved-ab
- [gfx950 · mixed] Memory-bound label + ~20% roofline headroom did not mean attackable BW: VALU:VMEM 11.6:1, and LDS/occupancy attacks all regressed. Classify before planning. ★★ — (unclaimed-roofline-headroom-on-a-memory-bound-attention-body-attention-gfx950-mixed.md)
  - kernels: paged_attention_ragged · kw: roofline, memory-bound, lds-bank-conflict, occupancy, valu-vmem-ratio, attention, gfx950, profiling, anti-pattern

## attention_decode
- [gfx950 · decode] Anti-pattern: when a decode case's grid is far under CU count, graph capture regresses it and occupancy-hint tuning buys ~1%; the geomean target is unreachable. ★★ — (a-dispatch-floored-small-case-caps-the-weighted-geomean-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: launch-overhead, hip-graph, waves-per-eu, occupancy, attention-decode, anti-pattern, small-grid, decode
- [gfx950 · decode] Split-KV/flash-decode and fp8 KV both die on a worst-element max_rel gate when the golden bakes a bf16 cast inside the V dot; cosine stays ~1.0 and hides it ★★ — (a-worst-element-parity-gate-closes-kv-reassociation-and-lowe-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, split-kv, flash-decode, fp8-kv, numerics, oracle-parity, dead-end
- [gfx950 · decode] fp8 KV storage + in-register bf16 dequant on decode attention, stacked with occupancy-4 bounds and non-temporal loads: 1.14x on top of the host-optimized state ★★ — (fp8-kv-storage-with-in-register-bf16-dequant-then-re-tune-oc-attention-decode-gfx950-decode.md)
  - kernels: paged_attention_decode · kw: decode, attention, fp8, kv-cache, occupancy, non-temporal-loads, dequant
- [gfx950 · decode] Caching wrapper scratch allocations and hoisting scale prep out of the per-call path on paged decode attention: 2.08x alone, the campaign's largest single lever ★★ — (host-wrapper-allocation-cache-scale-hoist-on-decode-attentio-attention-decode-gfx950-decode.md)
  - kernels: paged_attention_decode · kw: decode, attention, launch-overhead, host-runtime, allocation-cache, dispatch-bound
- [gfx950 · decode] Launch metaparameters (num_warps/num_stages/nonkdim) carry ~1.21x of a 1.23x total on paged grouped decode attention, gfx950/Triton; body rewrites add little. ★★ — (launch-metaparameters-carry-a-latency-bound-paged-decode-att-attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: launch-meta, num-stages, num-warps, software-pipelining, attention-decode, paged-kv, occupancy, triton
- [gfx950 · decode] At an attention decode's ISA and launcher floor, manual SW-pipelining and graph capture both regress; probe occupancy and host share first ★★ — (prove-the-floor-before-restructuring-ilp-or-wrapping-the-lau-attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, software-pipelining, num-stages, launch-overhead, graph-capture, occupancy, dead-end
- [gfx950 · decode] Dropping the '.cg' non-temporal modifier on once-read KV loads in paged attention: +8.2% geomean, bit-identical, on memory-bound cases only ★★ — (streaming-kv-loads-the-non-temporal-cache-modifier-can-cost--attention-decode-gfx950-decode.md)
  - kernels: kernel_unified_attention_2d · kw: attention, decode, paged-kv, cache-modifier, memory-bound, streaming-loads, triton
- [gfx950 · decode] Anti-pattern: on latency-bound paged decode attention already issuing 128-bit KV loads, cache hints, prefetch and loop-split all measured null-to-negative. ★★ — (the-kv-load-region-of-a-latency-bound-decode-attention-is-a--attention-decode-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: cache-modifier, software-prefetch, vectorization, latency-bound, attention-decode, paged-kv, anti-pattern, memory-bound

## dense_gemm
- [gfx950 · decode] Allocation reuse plus a lock-free last-hit shortcut was the only paying lever on a dispatch-floored dense linear; further host trimming measured zero. ★★ — (cache-the-host-hit-path-first-the-rest-of-the-host-path-is-f-dense-gemm-gfx950-decode.md)
  - kernels: wvSplitK · kw: host-runtime, launch-overhead, dispatch-floor, caching, decode, skinny-m, measurement-drift
- [gfx950 · decode] On tiny skinny-GEMV decode ops the host dispatch floor sits above GPU time: a correct 1.61x GPU-side roofline rewrite moved the scored wall 1.02x. ★★ — (device-side-wins-are-invisible-under-a-host-dispatch-floor-dense-gemm-gfx950-decode.md)
  - kernels: wvSplitK · kw: launch-overhead, dispatch-floor, skinny-m, gemv, decode, roofline, memory-latency
- [gfx950 · decode] Graph-capturing one tiny kernel to beat the launch floor: replay measured ~2x slower than eager, and the signature-cache wrapper alone nets 0.88-0.96x. ★★ — (graph-replay-of-a-one-kernel-graph-can-cost-more-than-the-la-dense-gemm-gfx950-decode.md)
  - kernels: wvSplitK · kw: hip-graph, launch-overhead, dispatch-floor, decode, skinny-m, host-runtime
- [gfx950 · both] On a scale-free fp8 MFMA loop, deeper pipelining, bigger tiles, wider mfma non-K dim and VGPR shaving all measured <=1.0x: a closed axis ★★ — (latency-hiding-knobs-are-a-closed-axis-once-an-fp8-mfma-loop-dense-gemm-gfx950-both.md)
  - kernels: gemm_a8w8_blockscale, _gemm_a8w8_blockscale_kernel · kw: fp8, dense-gemm, mfma, occupancy, lds-tiling, num-stages, vgpr-pressure, gfx950
- [gfx950 · prefill] The shared-memory layout round-trip in a Triton GEMM is a coalescing win: removing it is 2.5x slower, and the async-copy replacement does not lower. ★★ — (lds-round-trip-is-a-coalescing-win-not-overhead-dense-gemm-gfx950-prefill.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, convert-layout, lds-tiling, async-copy, software-pipeline, coalescing, roofline, anti-pattern, gfx950
- [gfx950 · prefill] Frozen BLOCK_M is not a wall: in-body M-coarsening then a register-staged wide-K MFMA rewrite gives ~2.66x weighted on fp16 dense GEMM (gfx950). ★★ — (m-coarsen-then-register-staged-wide-k-mfma-fp16-gemm-dense-gemm-gfx950-prefill.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, fp16, m-coarsening, mfma, register-staging, wide-k, lds-tiling, num-stages, gfx950, skinny-m
- [gfx950 · prefill] On compute-bound fp16 GEMM at gfx950 the occupancy axis is closed: every occ-raising/lowering variant lost, some catastrophically. ★★ — (occupancy-axis-closed-under-frozen-waves-per-eu-dense-gemm-gfx950-prefill.md)
  - kernels: _gemm_a16_w16_kernel · kw: dense-gemm, occupancy, waves-per-eu, num-warps, ping-pong, register-double-buffer, anti-pattern, gfx950, fp16
- [gfx950 · compute-bound] Price a vendor GEMM on the exact shapes before funding kernel-source rounds: 3.71x geomean over the capture, ~+25% over six rounds of in-language tuning ★★ — (price-the-vendor-library-on-the-exact-shapes-before-funding--dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: vendor-library, measurement-method, dense-gemm, interleaved-ab, compute-bound, isa-check
- [gfx950 · compute-bound] Screen codegen knobs that share a hardware budget jointly: two each filed dead alone measured +8.9%/+5.8% together, carrying the round 2.98x -> 3.19x geomean ★★ — (screen-resource-coupled-codegen-knobs-jointly-not-one-at-a-t-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: config-sweep, lds, vgpr, pipeline-stages, async-copy, tile-shape, occupancy, measurement-method, dense-gemm, compute-bound
- [gfx950 · compute-bound] Settle the clock with untimed real work before the timed window: +5.1% geomean on a dense GEMM with byte-identical device code ★★ — (settle-the-clock-controller-with-untimed-real-work-before-th-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: measurement-method, control-experiment, dense-gemm, compute-bound, interleaved-ab, harness-artifact, counters
- [gfx950 · compute-bound] Sweep tile aspect at fixed area under the accumulator-per-lane wall before growing area: +7.5% on all cases atop a 2.64x launch-config retune ★★ — (tile-aspect-at-fixed-area-under-a-hard-accumulator-per-lane--dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: tile-shape, tile-aspect, launch-config, vgpr, dense-gemm, compute-bound, interleaved-ab, occupancy
- [gfx950 · both] Fold + hoist all block-scale arithmetic out of the fp8 GEMM K-loop so the inner loop is a native unscaled MFMA loop: ~20x isolated on gfx950 ★★ — (turn-a-block-scaled-fp8-gemm-inner-loop-into-an-unscaled-mfm-dense-gemm-gfx950-both.md)
  - kernels: gemm_a8w8_blockscale, _gemm_a8w8_blockscale_kernel · kw: fp8, block-scale, dense-gemm, mfma, scale-hoist, l2-reorder, hip-graph, gfx950

## linear_attention
- [gfx950 · launch-bound] Audit the autotune config list for a missing axis (MFMA instruction size) before tuning inside it: up to 1.18x over the converged autotuner ★★ — (audit-the-autotune-config-list-for-a-missing-axis-before-tun-attention-gfx950-launch-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: autotune, config-sweep, mfma, occupancy, vgpr, launch-bound, interleaved-ab
- [gfx950 · prefill] Host-shim collapse of a redundant batch dim in the launch grid kills ~98% empty workgroups on chunked linear attention: ~2.8x on the largest case ★★ — (collapse-a-redundant-launch-grid-dimension-in-a-host-shim-linear-attention-gfx950-prefill.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: launch-overhead, grid-collapse, host-shim, linear-attention, varlen, empty-workgroups
- [gfx950 · launch-bound] Report every gain in a production-grid column beside the harness column: it cut a 22.68x harness geomean to 2.09x and reclassified 4 of 13 directions ★★ — (measure-a-production-grid-column-beside-the-harness-column-attention-gfx950-launch-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: measurement-method, control-experiment, harness-artifact, production-grid, launch-bound, launch-overhead, kernel-cache
- [gfx950 · mixed] Price a memory-bound round against a standalone ruler at the kernel's exact read+write byte count: re-priced 1.4x of funded headroom to a measured 1.18x ★★ — (price-a-memory-bound-round-against-a-standalone-ruler-re-mea-linear-attention-gfx950-mixed.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: roofline, measurement-method, memory-bound, control-experiment, dispatch-floor, interleaved-ab, linear-attention
- [gfx950 · prefill] Write-combining cache modifier on write-once fp32 output stores bypasses L2 write-allocate on gfx950: +6.4% on the store-bound case, bit-identical ★★ — (streaming-non-temporal-stores-for-a-write-once-output-linear-attention-gfx950-prefill.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: non-temporal-store, cache-modifier, store-bandwidth, l2-write-allocate, linear-attention, memory-bound
- [gfx950 · mixed] Sweep waves-per-EU as a zero-edit register dial on linear attention: falsified a funded occupancy direction fast, then shipped as a +2% pin after a rewrite ★★ — (use-the-waves-per-eu-request-as-a-zero-edit-register-dial-an-linear-attention-gfx950-mixed.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: waves-per-eu, vgpr, occupancy, isa-check, config-sweep, linear-attention, measurement-method
- [gfx950 · prefill] At an HBM-store-bandwidth roof, occupancy lift, persistent/fewer workgroups, finer store-skip granularity and graph replay all measured null or negative ★★ — (what-stops-paying-once-the-store-pipe-is-the-roof-linear-attention-gfx950-prefill.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: anti-pattern, store-bandwidth, occupancy, persistent-kernel, grid-stride, graph-replay, launch-overhead, measurement-methodology

## memory_movement
- [gfx950 · launch-bound] Replace the graded symbol with a cached direct-launcher closure when the per-call time is host dispatch: 2.24x alone, 2.35x cumulative, uniform across batch ★★ — (bypass-the-triton-dispatch-path-when-the-metric-is-host-laun-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, dispatch-floor, launch-bound, measurement-method, control-experiment, graph-capture, memory-movement, interleaved-ab
- [gfx950 · launch-bound] Grade launch-bound work against a cheapest-packet control, not the baseline: called the graded outcome 3/3 times on a dispatch-floor-bound copy kernel ★★ — (grade-launch-bound-work-against-a-cheapest-packet-control-no-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, dispatch-floor, control-experiment, measurement-method, launch-bound, interleaved-ab
- [gfx950 · launch-bound] Once a raw driver launch is in place, graph capture, doorbell/persistent kernels and native submit shims all measure neutral-or-worse on tiny ops. ★★ — (host-submit-axis-closed-below-raw-launch-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, host-dispatch, hip-graph, persistent-kernel, latency-bound, memory-movement, triton, closed-axis
- [gfx950 · launch-bound] Dispatch-bound tiny memory-movement Triton kernels: replacing the Python launch wrapper with a raw ctypes driver module-launch gives ~2.6x. ★★ — (raw-driver-module-launch-dispatch-bound-copy-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, host-dispatch, latency-bound, memory-movement, scatter, triton, ctypes, small-batch

## moe_grouped_gemm
- [gfx950 · small-batch] Anti-pattern: split-K/KBatch on a grouped MoE GEMM whose block grid already exceeds CU count regresses monotonically; dep-stall is not grid underfill ★★ — (a-56-dep-wait-on-a-grouped-moe-gemm-is-not-evidence-of-an-un-moe-grouped-gemm-gfx950-small-batch.md)
  - kernels: moe_stage1 · kw: split-k, moe, grouped-gemm, occupancy, dep-stall, grid-fill, anti-pattern, gfx950
- [gfx950 · both] On CDNA4 the VGPR and AGPR files share one budget, so a large fp32 accumulator pins occupancy at 2; four occupancy-escape rewrites all measured non-positive. ★★ — (accumulator-set-occupancy-floor-on-a-unified-vgpr-agpr-pool--moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: occupancy, vgpr, agpr, accumulator, raw-hip, warp-specialization, register-pressure, moe, grouped-gemm, gfx950
- [gfx950 · both] Anti-pattern: once weights are fp4, shrinking the activation operand is closed three ways — parity floor, no fp6 lowering, no outliers to protect. ★★ — (activation-operand-shrink-precision-floor-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel · kw: fp4, fp6, mixed-precision, parity-gate, outliers, activation-quant, dot-scaled, moe, grouped-gemm
- [gfx950 · both] On a register-tight int4 GEMM at occupancy 2, cheaper dequant math, load cache modifiers and compiler auto-pipelining all measured neutral to clearly negative. ★★ — (at-an-occupancy-floor-dequant-op-count-load-cache-hints-and--moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: dequant, cache-modifier, num-stages, software-pipeline, lds, int4, grouped-gemm, noise-floor, gfx950
- [gfx950 · mixed] Bitcast fp8 dot operands to the part's native OCP e4m3 dialect to delete a per-element software cast: 15.4x geomean on a fused-MoE grouped GEMM ★★ — (check-whether-the-fp8-operand-dialect-is-the-one-this-part-e-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: fp8, dtype-dialect, mfma, valu-emulation, isa-check, moe, dequant
- [gfx950 · mixed] Anti-pattern: downgrading B to fp4/mxfp4 on a gfx950 f8f6f4-MFMA GEMM cuts zero op-count and cannot help a latency-bound kernel ★★ — (fp4-weight-storage-buys-nothing-on-an-f8f6f4-mfma-that-is-la-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_stage1 · kw: mxfp4, fp4, low-precision, mfma, moe, grouped-gemm, anti-pattern, gfx950
- [gfx950 · both] Store the streamed MoE expert-weight operand as fp4 consumed natively by scaled MFMA: 42x isolated on a weight-streaming grouped GEMM, gfx950. ★★ — (fp4-weight-storage-moe-grouped-gemm-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel · kw: fp4, dot-scaled, mfma, moe, grouped-gemm, weight-streaming, hbm-bound, nonkdim, xcd-swizzle
- [gfx950 · mixed] Freeze one operand's pointer at a time to upper-bound every memory-side direction: capped the whole memory system at ~10% on the largest MoE GEMM case ★★ — (freeze-one-operand-s-pointer-at-a-time-to-put-an-upper-bound-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: control-experiment, measurement-method, l2-locality, operand-reuse, moe, counters
- [gfx950 · both] On a dequant-latency-bound int4 MoE grouped GEMM at the host-config optimum, seven in-kernel axes each re-measured as no-gain or a regression. ★★ — (in-kernel-axes-closed-on-dequant-latency-bound-moe-gemm-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_int4_w4a16 · kw: moe-grouped-gemm, int4-dequant, anti-pattern, closed-axis, split-k, num-warps, matrix-instr-nonkdim, num-stages, cuda-graph
- [gfx950 · mixed] Merge G quant groups into ONE dot instead of one dot per group on a packed-weight MoE GEMM: 2.24x vs a 1.94x incumbent (+15%), 2.92x geomean end state ★★ — (merge-several-quant-groups-into-one-dot-instead-of-one-dot-p-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: dequant, quantization-group, mfma, moe, isa-check, interleaved-ab, tile-shape
- [gfx950 · mixed] On CK fp8 block-scale grouped-MoE GEMM (gfx950), remap the pipeline to MFMA 32x32 + CShuffle epilogue: ~1.47x isolated, biggest on large cases ★★ — (mfma-32-remap-cshuffle-epilogue-on-a-ck-block-scale-grouped--moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_stage1 · kw: moe, grouped-gemm, fp8-blockscale, composable-kernel, mfma, cshuffle, lds-padding, gfx950
- [gfx950 · both] Anti-pattern: at occupancy 1 with a reused operand far larger than L2, occupancy, pipeline-depth and L2-swizzle knobs are all negative or ~1% ceilings. ★★ — (occupancy-pipeline-knobs-closed-when-dep-chain-bound-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel · kw: occupancy, software-pipelining, double-buffer, prefetch, num-stages, l2-residency, xcd-swizzle, dep-chain, moe, grouped-gemm
- [gfx950 · large-batch] Pad the launch grid to a multiple of NUM_XCD so the XCD pid remap engages: +3.3% geomean on an already-retiled MoE grouped GEMM, L2 hit 65%->85% ★★ — (pad-the-grid-to-a-multiple-of-num-xcd-so-the-pid-swizzle-fir-moe-grouped-gemm-gfx950-large-batch.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: pid-remap, l2-locality, xcd, grid-geometry, moe, large-batch, isa-check, interleaved-ab
- [gfx950 · both] Per-bucket BLOCK_M/num_warps plus byte-once dual-nibble int4 dequant lifts a weight-quantized MoE grouped GEMM ~4.6x, empirical roofline 0.10 to 0.60. ★★ — (per-bucket-block-m-plus-byte-once-dual-nibble-int4-dequant-o-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: moe, int4, weight-only-quant, grouped-gemm, block-m, num-warps, mfma-nonkdim, dequant, software-pipeline, gfx950
- [gfx950 · both] Per-M-bucket host launch-config retune on int4-weight MoE grouped GEMM: 3.33x isolated with the JIT body left byte-identical. ★★ — (per-m-bucket-host-launch-config-int4-moe-gemm-moe-grouped-gemm-gfx950-both.md)
  - kernels: fused_moe_int4_w4a16 · kw: moe-grouped-gemm, int4-dequant, w4a16, launch-config, host-tuning, per-m-bucket, block-size, group-size-m
- [gfx950 · mixed] Pre-lay a strided per-iteration operand in a prologue kernel in the consumer's reading order: +11.5% geomean on an already-converged MoE grouped GEMM ★★ — (pre-lay-a-strided-per-iteration-operand-in-a-prologue-kernel-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, operand-reuse, l2-locality, address-locality, prologue-kernel, measurement-method
- [gfx950 · mixed] Price an inferred padding or grid-quantization tax with a block-count sweep before funding a round: both inferred taxes were fictions and the lane returned 0% ★★ — (price-an-inferred-padding-or-grid-quantization-tax-with-a-bl-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: grid-geometry, control-experiment, measurement-method, moe, tile-shape, memory-bound, launch-config
- [gfx950 · mixed] Size each shared-memory buffer from the instantiated template variant, not the generic max: 1.29x on an MoE grouped GEMM by doubling resident workgroups per CU ★★ — (size-shared-memory-from-the-instantiated-variant-not-the-gen-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_stage1 · kw: lds, occupancy, vgpr, moe, template-instantiation, isa-check, code-object, dead-allocation
- [gfx950 · large-batch] Merge adjacent same-group row-blocks into a double-height tile to amortise weight dequant: ~1.23x on large-batch MoE shapes, slower on the smallest ★ — (merge-adjacent-same-group-row-blocks-to-amortise-weight-dequ-moe-grouped-gemm-gfx950-large-batch.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: dequant, tile-shape, moe, operand-reuse, vgpr, large-batch, interleaved-ab

## moe_router_topk
- [gfx950 · decode] Anti-pattern: graph capture/replay around a single tiny kernel is slower than a direct launch; replay dispatch alone exceeds the whole launch it replaces. ★★ — (graph-capture-does-not-pay-for-one-small-dispatch-moe-router-topk-gfx950-decode.md)
  - kernels: _topk_forward · kw: graph-capture, launch-overhead, dispatch-bound, anti-pattern, triton, decode
- [gfx950 · decode] On a tiny dispatch-floored Triton router, collapsing the host launch path (cached compiled-kernel closure + trusted steady state) carries most of the win. ★★ — (host-launch-path-collapse-on-a-dispatch-floored-router-moe-router-topk-gfx950-decode.md)
  - kernels: _topk_forward · kw: launch-overhead, host-runtime, dispatch-bound, triton, topk-router, decode
- [gfx950 · decode] Anti-pattern: at one full occupancy wave, tail store-layout round-trips and serial reduction depth are hidden behind the load path; removing them buys ~0. ★★ — (the-tail-is-free-once-the-grid-is-one-full-occupancy-wave-moe-router-topk-gfx950-decode.md)
  - kernels: _topk_forward · kw: convert-layout, lds, occupancy, memory-bound, anti-pattern, reduction, triton

## quantize_cast
- [gfx950 · memory-bound] Past ~63% of nameplate HBM on a 3x-traffic elementwise cast, every bandwidth knob measured zero: warps, tile width, cache modifiers, store vectorization. ★★ — (memory-lever-axis-closed-near-achievable-bw-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: hbm-ceiling, memory-bound, closed-axis, store-vectorization, cache-modifier, num-warps, quantize-cast, assembly-inspection
- [gfx950 · mixed] Census the ISA for emulated narrow-dtype convert and divide in a quantize/cast kernel: 3.75x geomean, 4.76x/4.00x on the memory-bound streaming cases ★★ — (narrow-dtype-convert-and-divide-can-lower-to-emulation-censu-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: fp8, dtype-dialect, isa-check, valu-emulation, quant, inline-asm, memory-bound, launch-bound
- [gfx950 · mixed] Run a footprint control before believing a per-case bandwidth number: it showed one quant case cache-served and the largest at ~98% of achievable DRAM BW ★★ — (price-cache-residency-with-a-footprint-control-before-believ-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: roofline, measurement-method, control-experiment, l2-locality, quant, memory-bound, counters, harness-artifact
- [gfx950 · mixed] Export a launcher OBJECT so the frozen num_warps=1/one-program-per-row launch can be re-tiled: 1.15x plateau -> 2.3x on memory-bound fp8 quant-cast. ★★ — (reinterpret-frozen-launch-via-wrapper-object-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-config, wrapper-relaunch, harness-seam, quantize-cast, fp8, memory-bound, tiling, num-warps
- [gfx950 · launch-bound] A tiny-shape case can be floored by the measurement bracket itself; graph replay and persistent grids buy pure throughput but ~0 scored, and cost the big cases. ★★ — (small-case-floor-is-the-timing-bracket-quantize-cast-gfx950-launch-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-overhead, closed-axis, measurement-floor, hip-graph, persistent-kernel, small-batch, quantize-cast
- [gfx950 · mixed] Memoize the compiled kernel and call its low-level entry directly on a launch-bound quant/cast shape: 2.77x there, ~1.0x once the case is bandwidth-bound ★★ — (the-generic-triton-launch-path-is-not-the-floor-when-the-sma-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-overhead, dispatch-floor, measurement-method, launch-bound, kernel-cache, quant
- [gfx950 · mixed] Port a winning mechanism into the sibling shape-specialised instances: cross-instance ports paid +1.60-3.95% while stacking three orthogonal patches paid +0.31% ★★ — (when-an-operator-ships-several-shape-specialised-instances-g-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: cross-instance-port, measurement-method, interleaved-ab, quant, noise-band, control-experiment

## quantized_gemm
- [gfx950 · compute-bound] Block-scaled fp8 GEMM pinned near the latency-bound floor: hardware fnuz->OCP cvt upcast plus fused split-K reduce lifts it to compute-bound ★★ — (fp8-hardware-cvt-upcast-is-the-wall-under-a-block-scaled-fp8-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: fp8, block-scale, dequant, split-k, upcast, triton, compute-bound, quantized-gemm
- [gfx950 · compute-bound] Issue/dep-chain-bound Triton GEMM on gfx950: manual K-block pipelining, occupancy raising, B re-layout and host graph replay all lost - 5 lanes, 0 wins ★★ — (occupancy-and-manual-pipelining-dead-on-issue-bound-gemm-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: occupancy, software-pipelining, num-stages, vgpr-pressure, launch-overhead, hip-graph, lds-tiling, quantized-gemm, gfx950, anti-pattern
- [gfx950 · compute-bound] At 2 waves/SIMD an ILP-bound block-scaled fp8 GEMM loses from any occupancy push; grid/tail, hand-scheduling, graph capture, microtune all zero ★★ — (occupancy-push-and-four-neighbouring-axes-are-closed-on-an-i-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: occupancy, vgpr, ilp, spill, hip-graph, grid-tail, anti-pattern, quantized-gemm
- [gfx950 · mixed] fp8 block-scaled GEMM, gfx950: fold the scale-format fixup into the native scaled-cvt operand, pack 4 fp8 per dword, coarsen sub-K - 10.66x bit-exact ★★ — (order-preserving-critical-path-levers-fp8-blockscale-gemm-quantized-gemm-gfx950-mixed.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: fp8, block-scale, quantized-gemm, dep-chain, sub-k-coarsening, scaled-cvt, lds-tiling, bit-exact, gfx950, critical-path
- [gfx950 · mixed] Under a near-bit-exact parity gate the fp32 MFMA K-reduction order is structural: ILP splits, Kahan and native fp8 MFMA all fail parity, not perf ★★ — (parity-gate-freezes-fp32-reduction-order-quantized-gemm-gfx950-mixed.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: bit-exact, parity-gate, reduction-order, reassociation, mfma, fp8, quantized-gemm, gfx950, anti-pattern
- [gfx950 · compute-bound] Size a dequant-in-loop K loop in operand bytes and register/LDS budget, not instructions: instruction count correlated NEGATIVELY with time on four ladders ★★ — (price-a-dequant-in-loop-k-loop-in-operand-bytes-and-register-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: dequant, quantized-gemm, vgpr, lds, occupancy, counters, measurement-method, control-experiment, compute-bound
- [gfx950 · compute-bound] Probe the correctness gate with a one-ulp re-association before funding any reduction-reordering lane: it retired four rounds aimed at a ~1.6x roof ★★ — (probe-the-correctness-gate-with-a-one-ulp-re-association-bef-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel, _w8a8_triton_block_scaled_mm · kw: correctness-gate, measurement-method, dtype-dialect, mfma, quantized-gemm, roofline, control-experiment
- [gfx950 · compute-bound] Per-1x128 dequant that is uniform across the N tile collapses to a rank-1 per-row FMA; unrolling K by 2 overlaps the second MFMA with the dequant VALU work ★★ — (rank-1-scale-collapse-plus-2-deep-overlap-unroll-hides-dequa-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: dequant, block-scale, mfma, ilp, unroll, block-pingpong, fp8, quantized-gemm
- [gfx950 · compute-bound] Register-prefetch the SMALL per-iteration operands one k-block ahead as the body's last VMEM: +3.9-7.0% on the large-M cases of a block-scaled GEMM ★★ — (register-prefetch-the-small-per-iteration-operands-one-k-blo-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel, _w8a8_triton_block_scaled_mm · kw: prefetch, vgpr, isa-check, operand-reuse, compute-bound, quant, dequant, mfma

## topk_routing
- [gfx950 · launch-bound] Collapse Triton's Python launch path before tuning the body of a small launch-bound top-k op: 1.58x standalone, 2.19x on the smallest grid ★★ — (collapse-the-python-launch-path-before-tuning-the-body-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: launch-overhead, launch-bound, dispatch-floor, kernel-cache, topk, measurement-method, interleaved-ab, graph-capture
- [gfx950 · launch-bound] Restate small-k selection over a chunk axis instead of Triton's distributed axis: device time -30 to -33% on every case at zero occupancy cost ★★ — (move-small-k-selection-onto-a-chunk-axis-off-the-distributed-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: topk, cross-lane, isa-check, vgpr, occupancy, launch-bound, interleaved-ab, control-experiment
- [gfx950 · launch-bound] Probe the per-call floor with the real armed launcher, not an empty jitted kernel: bounded all remaining headroom at +7.1% and closed the run ★★ — (probe-the-floor-with-the-real-armed-launcher-and-re-price-pe-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: launch-overhead, dispatch-floor, launch-bound, measurement-method, topk, control-experiment, interleaved-ab
- [gfx950 · launch-bound] Reshape a tile into the layout's own factorisation instead of fighting convert_layout: -11.5% instructions, +7.2% on the one device-bound top-k case ★★ — (reshape-a-tile-into-the-layout-s-own-factorisation-instead-o-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: cross-lane, lds, topk, launch-bound, isa-check, tile-shape, workgroup-size

## method
- [gfx950 · launch-bound] On sub-microsecond kernels a bimodal box throttle makes a single-shot gate reject a real win repeatedly; gate on medians or paired same-session A/B. ★★ — (bimodal-throttle-defeats-single-shot-gate-method-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: measurement-noise, throttle, ab-methodology, latency-bound, launch-overhead, frozen-baseline, small-batch
- [gfx950 · n/a] Three cheap probes separate issue-throughput from dependency-latency/VGPR ceilings in a quantized-dequant GEMM inner loop before spending a round. ★★ — (count-vs-latency-bound-discriminator-method-gfx950-n-a.md)
  - kernels: fused_moe_int4_w4a16 · kw: instrument, roofline, valu-bound, dependency-latency, occupancy, launch-overhead, vgpr-pressure, int4-dequant
- [gfx950 · n/a] A JIT-compiled vendor extension can silently re-run the old binary after a header edit, and the harness improvement flag can be false on winning rounds. ★★ — (jit-staleness-and-false-negative-improved-flag-method-gfx950-n-a.md)
  - kw: instrument, jit, build-staleness, verification, false-negative, measurement-hygiene, gfx950
- [gfx950 · n/a] Instrument: a heavy-only subset reported +2.5% where the grader's full mix showed +0.3%; launch-floor cases drift ~20% across sessions and cannot be won ★★ — (measure-in-the-grader-s-own-case-mix-sub-noise-floor-cases-c-method-gfx950-n-a.md)
  - kw: measurement, noise-floor, ab-testing, launch-overhead, decode, gfx950, dead-end
- [gfx950 · n/a] Read VGPR/LDS from the .kd descriptor in the loaded JIT object to get true occupancy; the profiler counter under-reported registers by about half ★★ — (method-gfx950-n-a.md)
  - kw: instrument, occupancy, vgpr, register-bound, profiler, jit, elf
- [gfx950 · n/a] Roofline read against full-chip nameplate showed 33% of peak; against the visible CU partition it was ~72%, i.e. the headroom being chased did not exist ★★ — (score-against-realizable-partition-peak-not-full-chip-namepl-method-gfx950-n-a.md)
  - kw: roofline, nameplate-peak, partition, measurement-methodology, budget-saturation, gfx950

## keyword vocabulary (generated — REUSE these before coining a new term)
launch-overhead(25) · occupancy(23) · gfx950(20) · measurement-method(19) · moe(16) · anti-pattern(15) · interleaved-ab(15) · control-experiment(14) · decode(14) · mfma(13) · isa-check(12) · memory-bound(12) · vgpr(12) · dequant(10) · dispatch-floor(10) · fp8(10) · launch-bound(10) · dense-gemm(9) · grouped-gemm(9) · attention(8) · hip-graph(8) · num-stages(8) · quantized-gemm(8) · roofline(8) · triton(8) · compute-bound(7) · lds(6) · num-warps(6) · tile-shape(6) · cache-modifier(5) · graph-capture(5) · host-runtime(5) · l2-locality(5) · lds-tiling(5) · quant(5) · block-scale(4) · closed-axis(4) · counters(4) · latency-bound(4) · launch-config(4) · linear-attention(4) · operand-reuse(4) · skinny-m(4) · software-pipelining(4) · topk(4) · attention-decode(3) · config-sweep(3) · dead-end(3) · dispatch-bound(3) · dtype-dialect(3) · fp4(3) · harness-artifact(3) · instrument(3) · int4-dequant(3) · kernel-cache(3) · memory-movement(3) · paged-kv(3) · persistent-kernel(3) · quantize-cast(3) · small-batch(3) · software-pipeline(3) · split-k(3) · vgpr-pressure(3) · waves-per-eu(3) · address-locality(2) · async-copy(2) · bit-exact(2) · convert-layout(2) · cross-lane(2) · dep-chain(2) · dot-scaled(2) · fp16(2) · grid-geometry(2) · host-dispatch(2) · ilp(2) · int4(2) · jit(2) · kv-cache(2) · large-batch(2) · measurement-methodology(2) · moe-grouped-gemm(2) · noise-floor(2) · parity-gate(2) · pid-remap(2) · prefetch(2) · store-bandwidth(2) · valu-emulation(2) · xcd(2) · xcd-swizzle(2) · ab-methodology · ab-testing · accumulator · activation-quant · agpr · allocation-cache · assembly-inspection · autotune · bijection · block-m · block-pingpong · block-size · budget-saturation · build-staleness · caching · coalescing · code-object · composable-kernel · correctness-gate · critical-path · cross-instance-port · cshuffle · ctypes · cuda-graph · dead-allocation · dep-stall · dependency-latency · double-buffer · elf · empty-workgroups · false-negative · flash-decode · fp6 · fp8-blockscale · fp8-kv · frozen-baseline · gemv · gpu-bound · graph-replay · grid-collapse · grid-fill · grid-stride · grid-tail · group-size-m · harness-noise · harness-seam · hbm-bound · hbm-ceiling · host-shim · host-tuning · inline-asm · l2-reorder · l2-residency · l2-reuse · l2-write-allocate · launch-meta · lds-bank-conflict · lds-padding · low-precision · m-coarsening · matrix-instr-nonkdim · measurement · measurement-drift · measurement-floor · measurement-hygiene · measurement-noise · memory-latency · mfma-nonkdim · mixed-precision · mxfp4 · nameplate-peak · noise-band · non-temporal · non-temporal-loads · non-temporal-store · nonkdim · nontemporal-load · null-baseline · numerics · oracle-parity · outliers · partition · per-m-bucket · per-shape-gate · ping-pong · pipeline-stages · production-grid · profiler · profiling · prologue-kernel · quantization-group · raw-hip · reassociation · reduction · reduction-order · register-bound · register-double-buffer · register-pressure · register-staging · scale-hoist · scaled-cvt · scatter · small-grid · software-prefetch · spill · split-kv · store-vectorization · streaming-loads · sub-k-coarsening · template-instantiation · throttle · tile-aspect · tiling · topk-router · unroll · upcast · valu-bound · valu-vmem-ratio · varlen · vectorization · vectorized-load · vendor-library · verification · w4a16 · warp-specialization · weight-only-quant · weight-streaming · wide-k · workgroup-size · wrapper-relaunch

> ⚠ **Near-duplicate keywords** — same concept, different spelling. Pick one, edit the
> cards, regenerate:
> - non-temporal-loads / nontemporal-load

> ⚠ **Over the per-class cap of 8:** dense_gemm (12), moe_grouped_gemm (19), quantized_gemm (9). Archive the lowest `confidence × freshness × standing` card in that class
> (★★★ is never auto-evicted), then regenerate.
