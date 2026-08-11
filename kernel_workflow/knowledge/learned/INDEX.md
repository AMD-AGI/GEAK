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
- [gfx950 · decode] Read the VALU-to-VMEM ratio before funding a streaming rewrite off a high percent-of-bandwidth reading: the rewrite returned 1.0x, as did two adjacent ones ★★ — (a-high-percent-of-bandwidth-reading-can-be-a-compute-tail-ar-attention-gfx950-decode.md)
  - kernels: paged_attention_decode · kw: roofline, counters, measurement-method, decode, attention, occupancy, vgpr, dequant, operand-reuse
- [gfx950 · memory-bound] Price the noise before funding a small-case recovery: a size-gate budgeted at 1.37x returned 0.988x, and a baseline-only control moved as much as the candidate ★★ — (cases-sitting-at-the-launch-noise-floor-cannot-be-recovered--attention-gfx950-memory-bound.md)
  - kernels: paged_attention_decode · kw: measurement-method, control-experiment, interleaved-ab, dispatch-floor, harness-artifact, attention, memory-bound, launch-overhead
- [gfx950 · memory-bound] Grade a non-temporal load helper at the ISA level before crediting it: +6.6% geomean on a read-once KV stream, all 9 attention cases up ★★ — (check-that-the-non-temporal-bit-survives-to-the-isa-before-c-attention-gfx950-memory-bound.md)
  - kernels: paged_attention_decode · kw: cache-policy, streaming-operand, isa-check, memory-bound, attention, decode, l2-locality
- [gfx950 · decode] Price graph capture instead of assuming it: on decode attention it left the launch-floored small case tied and taxed the two larger cases ~18-21% ★★ — (graph-capture-does-not-buy-back-a-small-grid-launch-floor-an-attention-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: graph-capture, launch-overhead, dispatch-floor, decode, attention, measurement-method, control-experiment
- [gfx950 · memory-bound] Price the host path before capturing: with a lean launcher already below graph-launch cost, HIP-graph capture regressed -49.6%/-40.2%/-29.0% at batch 2/32/64 ★★ — (graph-capture-is-a-regression-once-the-launcher-is-already-b-attention-gfx950-memory-bound.md)
  - kernels: paged_attention_decode · kw: graph-capture, launch-overhead, dispatch-floor, measurement-method, control-experiment, attention, memory-bound, interleaved-ab
- [gfx950 · decode] Treat a storage-dtype narrowing as invalidating the surrounding tuning: re-swept non-temporal and occupancy knobs added 1.057x and 1.019x on top of 1.053x ★★ — (narrowing-operand-storage-re-opens-the-non-temporal-and-occu-attention-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: dtype-dialect, non-temporal, occupancy, vgpr, decode, l2-locality, operand-reuse, memory-bound
- [gfx950 · decode] Null the kernel body (deliberately wrong) to price a whole direction in one measurement: predicted the 0% outcome of optimizing a decode dispatch ★★ — (null-the-kernel-body-before-optimizing-it-attention-gfx950-decode.md)
  - kernels: paged_attention_decode · kw: control-experiment, measurement-method, dispatch-floor, launch-overhead, decode, null-baseline
- [gfx950 · decode] When ISA already shows max-width loads and the pipeliner overlaps them, close the load axis: cache hints and manual prefetch cost 3-20% here ★★ — (once-the-streaming-loads-are-already-max-width-and-the-pipel-attention-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: pipeline-stages, l2-locality, isa-check, decode, attention, memory-bound, counters, roofline, operand-reuse
- [gfx950 · memory-bound] LDS-wait and bank-conflict counters read as headroom but were overlapped: all three edits aimed at them lost (-5.5%/-6.5%, ~1%, ~4.5% on the large case) ★★ — (price-lds-wait-and-bank-conflict-counters-against-the-achiev-attention-gfx950-memory-bound.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: counters, lds, occupancy, vgpr, roofline, measurement-method, attention, memory-bound, isa-check
- [gfx950 · decode] Price the cross-block fence before fusing a split reduction into its producer: the fence cost ~20-30x the decode dispatch it removes, a 7x regression ★★ — (price-the-cross-block-fence-before-fusing-a-split-reduction--attention-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: fence, dispatch-floor, launch-overhead, decode, attention, control-experiment
- [gfx950 · decode] Re-linearise the workgroup id so consecutive workgroups walk one request's contiguous KV: +3.1% on attention decode, scaling with workgroup count ★★ — (re-linearise-the-workgroup-id-in-the-prologue-for-address-lo-attention-gfx950-decode.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: pid-remap, l2-locality, xcd, address-locality, decode, isa-check, bijection, interleaved-ab
- [gfx950 · memory-bound] Re-test an inherited cross-variant DEAD precision label on this variant's own dispatch: narrowing KV storage took the run from 1.0615x to 1.2993x ★★ — (re-test-an-inherited-cross-variant-dead-precision-label-agai-attention-gfx950-memory-bound.md)
  - kernels: paged_attention_decode · kw: fp8, dtype-dialect, memory-bound, attention, decode, measurement-method
- [gfx950 · memory-bound] Read .vgpr_count out of the code object before opening an occupancy round: with zero spill the occupancy pragmas are no-ops — 5 of 6 directions closed ★★ — (read-the-register-count-out-of-the-code-object-before-spendi-attention-gfx950-memory-bound.md)
  - kernels: _fwd_grouped_kernel_stage1, paged_attention_decode · kw: vgpr, code-object, occupancy, isa-check, waves-per-eu, attention, prefetch, tile-shape
- [gfx950 · memory-bound] Read where the reference rounds before funding reassociation: split-KV diverged at max_rel 1.357 and fp8 KV blew worst-element error 36-125x past the gate ★★ — (reproduce-the-reference-s-in-dot-narrow-precision-cast-befor-attention-gfx950-memory-bound.md)
  - kernels: _fwd_grouped_kernel_stage1 · kw: numerics-gate, reassociation, fp8, dtype-dialect, measurement-method, attention, memory-bound, operand-reuse
- [gfx950 · decode] Size the re-measurement band before believing an end-game delta and quote the median pass, not the max: the band here was 3.5% around a 1.225x median ★★ — (size-the-re-measurement-band-before-believing-an-end-game-de-attention-gfx950-decode.md)
  - kernels: paged_attention_decode · kw: measurement-method, interleaved-ab, env-switch, control-experiment, decode, attention
- [gfx950 · memory-bound] Strip an inherited non-temporal / L1-bypass cache modifier from once-read streaming loads: +8.2% geomean, bit-identical, +15% on the mid batch case ★★ — (strip-the-non-temporal-cache-modifier-from-once-read-streami-attention-gfx950-memory-bound.md)
  - kernels: paged_attention_decode · kw: cache-policy, streaming-operand, memory-bound, attention, decode, isa-check

## dense_gemm
- [gfx950 · compute-bound] Price a published peak-efficiency GEMM schedule against your shape and a backend-pinned waves-per-EU first: four directions on this axis all returned 1.00x ★★ — (a-frozen-waves-per-eu-cap-closes-the-ping-pong-and-occupancy-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: occupancy, waves-per-eu, dense-gemm, compute-bound, mfma, pipeline-stages, lds-tiling, tile-shape, vgpr
- [gfx950 · memory-bound] Measure graph replay against eager before building on it: for a one-kernel capture replay was ~2x SLOWER, and the shim alone net-regressed to 0.88-0.96x ★★ — (a-single-kernel-graph-capture-can-be-a-higher-launch-floor-t-dense-gemm-gfx950-memory-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: graph-capture, launch-overhead, dispatch-floor, measurement-method, interleaved-ab, control-experiment, dense-gemm, memory-bound
- [gfx950 · compute-bound] Shrink pipeline depth to free LDS, then coarsen tile rows in-body under a frozen launch config: 2.11x from the depth cut alone, 2.66x geomean stacked ★★ — (buy-lds-back-from-the-pipeline-depth-then-spend-it-on-in-bod-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: pipeline-stages, lds-tiling, coarsening, tile-shape, vgpr, occupancy, dense-gemm, compute-bound, launch-config
- [gfx950 · compute-bound] Grep the built artifact for a patch marker before recording a negative: 6+ of 44 passes on a dense GEMM scored code that never reached the build ★★ — (confirm-the-edit-reached-the-built-artifact-before-recording-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: measurement-method, control-experiment, isa-check, code-object, config-sweep, dense-gemm, compute-bound
- [gfx950 · memory-bound] Credit a gain only from a same-session interleaved A/B: a cross-session baseline inflated a real ~1.09x into a reported 1.31x on a tiny dense GEMM ★★ — (credit-a-cumulative-gain-only-from-a-same-session-a-b-never--dense-gemm-gfx950-memory-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: measurement-method, interleaved-ab, clock-drift, noise-band, dense-gemm, memory-bound, dispatch-floor
- [gfx950 · memory-bound] Price the host dispatch floor against device time before funding device work: a 1.61x device win on a tiny dense GEMM moved the scored wall only 1.018x ★★ — (price-the-host-dispatch-floor-against-device-time-before-fun-dense-gemm-gfx950-memory-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: dispatch-floor, launch-overhead, host-launch, launch-bound, occupancy, vgpr, lds, measurement-method, control-experiment, dense-gemm
- [gfx950 · compute-bound] Price a vendor GEMM on the exact shapes before funding kernel-source rounds: 3.71x geomean over the capture, ~+25% over six rounds of in-language tuning ★★ — (price-the-vendor-library-on-the-exact-shapes-before-funding--dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: vendor-library, measurement-method, dense-gemm, interleaved-ab, compute-bound, isa-check
- [gfx950 · compute-bound] Screen codegen knobs that share a hardware budget jointly: two each filed dead alone measured +8.9%/+5.8% together, carrying the round 2.98x -> 3.19x geomean ★★ — (screen-resource-coupled-codegen-knobs-jointly-not-one-at-a-t-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: config-sweep, lds, vgpr, pipeline-stages, async-copy, tile-shape, occupancy, measurement-method, dense-gemm, compute-bound
- [gfx950 · compute-bound] Settle the clock with untimed real work before the timed window: +5.1% geomean on a dense GEMM with byte-identical device code ★★ — (settle-the-clock-controller-with-untimed-real-work-before-th-dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: measurement-method, control-experiment, dense-gemm, compute-bound, interleaved-ab, harness-artifact, counters
- [gfx950 · compute-bound] Sweep tile aspect at fixed area under the accumulator-per-lane wall before growing area: +7.5% on all cases atop a 2.64x launch-config retune ★★ — (tile-aspect-at-fixed-area-under-a-hard-accumulator-per-lane--dense-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a16_w16_kernel · kw: tile-shape, tile-aspect, launch-config, vgpr, dense-gemm, compute-bound, interleaved-ab, occupancy

## linear_attention
- [gfx950 · launch-bound] Audit the autotune config list for a missing axis (MFMA instruction size) before tuning inside it: up to 1.18x over the converged autotuner ★★ — (audit-the-autotune-config-list-for-a-missing-axis-before-tun-attention-gfx950-launch-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: autotune, config-sweep, mfma, occupancy, vgpr, launch-bound, interleaved-ab
- [gfx950 · memory-bound] Match the gate's timing shape to the scoring metric: a back-to-back gate silently rejected graph replay, and forcing it on paid exactly 1.0x ★★ — (batched-back-to-back-timing-hides-the-launch-gap-that-a-per--linear-attention-gfx950-memory-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: graph-capture, launch-overhead, dispatch-floor, measurement-method, interleaved-ab, control-experiment, linear-attention, memory-bound
- [gfx950 · memory-bound] Mark write-once output stores non-temporal when the store pipe is the roofline: +6.4% bit-identical on the largest case, cumulative 15.09 -> 15.36x ★★ — (mark-write-once-output-stores-non-temporal-when-the-store-pi-linear-attention-gfx950-memory-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: non-temporal, l2-locality, isa-check, memory-bound, linear-attention, roofline, interleaved-ab
- [gfx950 · launch-bound] Report every gain in a production-grid column beside the harness column: it cut a 22.68x harness geomean to 2.09x and reclassified 4 of 13 directions ★★ — (measure-a-production-grid-column-beside-the-harness-column-attention-gfx950-launch-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: measurement-method, control-experiment, harness-artifact, production-grid, launch-bound, launch-overhead, kernel-cache
- [gfx950 · mixed] Price a memory-bound round against a standalone ruler at the kernel's exact read+write byte count: re-priced 1.4x of funded headroom to a measured 1.18x ★★ — (price-a-memory-bound-round-against-a-standalone-ruler-re-mea-linear-attention-gfx950-mixed.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: roofline, measurement-method, memory-bound, control-experiment, dispatch-floor, interleaved-ab, linear-attention
- [gfx950 · memory-bound] Stop shrinking stored bytes once a finer skip mask breaks wide stores: the coarse quadrant skip won -10.8% on the largest case, every finer mask lost ★★ — (stop-shrinking-stored-bytes-once-the-skip-granularity-turns--linear-attention-gfx950-memory-bound.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: linear-attention, memory-bound, coarsening, tile-shape, measurement-method, control-experiment, interleaved-ab
- [gfx950 · mixed] Sweep waves-per-EU as a zero-edit register dial on linear attention: falsified a funded occupancy direction fast, then shipped as a +2% pin after a rewrite ★★ — (use-the-waves-per-eu-request-as-a-zero-edit-register-dial-an-linear-attention-gfx950-mixed.md)
  - kernels: chunk_scaled_dot_kkt_fwd_kernel · kw: waves-per-eu, vgpr, occupancy, isa-check, config-sweep, linear-attention, measurement-method

## memory_movement
- [gfx950 · launch-bound] Replace the graded symbol with a cached direct-launcher closure when the per-call time is host dispatch: 2.24x alone, 2.35x cumulative, uniform across batch ★★ — (bypass-the-triton-dispatch-path-when-the-metric-is-host-laun-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, dispatch-floor, launch-bound, measurement-method, control-experiment, graph-capture, memory-movement, interleaved-ab
- [gfx950 · launch-bound] Grade a tiny win on the per-case signature its mechanism predicts plus a median: five straight false rejections became a confirmed 3.00x ★★ — (grade-a-sub-microsecond-win-on-a-mechanism-signature-and-a-m-unmatched-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: measurement-method, interleaved-ab, noise-band, clock-drift, launch-bound, dispatch-floor, control-experiment
- [gfx950 · launch-bound] Grade launch-bound work against a cheapest-packet control, not the baseline: called the graded outcome 3/3 times on a dispatch-floor-bound copy kernel ★★ — (grade-launch-bound-work-against-a-cheapest-packet-control-no-memory-movement-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, dispatch-floor, control-experiment, measurement-method, launch-bound, interleaved-ab
- [gfx950 · launch-bound] Submit a tiny single-dispatch copy through the raw driver module-launch instead of graph capture: 2.613x cumulative against 1.322x for capture/replay ★★ — (submit-a-single-dispatch-copy-through-the-raw-driver-module--unmatched-gfx950-launch-bound.md)
  - kernels: write_req_to_token_pool_triton · kw: launch-overhead, dispatch-floor, host-launch, launch-bound, graph-capture, kernel-cache, memory-movement, measurement-method

## moe_grouped_gemm
- [gfx950 · compute-bound] Re-measure an imported launch-knob rule at your own tile: every rule ported from a sibling MoE kernel lost, one of them up to 6.5x worse ★★ — (a-launch-knob-rule-imported-from-a-sibling-kernel-is-not-a-p-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel · kw: launch-config, config-sweep, moe, tile-shape, mfma, vgpr, compute-bound, measurement-method
- [gfx950 · compute-bound] When VGPR and AGPR share one pool, moving an fp32 accumulator cannot buy a wave: the escape round returned 0 and a hand-written HIP rewrite measured 1.005x ★★ — (an-fp32-accumulator-that-pins-occupancy-at-two-waves-is-not--moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel · kw: occupancy, vgpr, mfma, moe, dequant, tile-shape, compute-bound, operand-reuse
- [gfx950 · mixed] Bitcast fp8 dot operands to the part's native OCP e4m3 dialect to delete a per-element software cast: 15.4x geomean on a fused-MoE grouped GEMM ★★ — (check-whether-the-fp8-operand-dialect-is-the-one-this-part-e-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: fp8, dtype-dialect, mfma, valu-emulation, isa-check, moe, dequant
- [gfx950 · mixed] Count block-waves before funding split-K on a token-sorted MoE grouped GEMM: the grid was already ~20 waves deep and split-K measured 0.85x, not 1.62x ★★ — (count-the-block-waves-before-spending-a-round-on-split-k-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: split-k, grid-geometry, occupancy, moe, control-experiment, measurement-method
- [gfx950 · compute-bound] Separate issue-bound from latency-bound before a bit-trick dequant rewrite: on an int4 MoE GEMM a -6% VALU issue count ran ~7% slower ★★ — (cutting-dequant-valu-op-count-buys-nothing-when-the-wall-is--moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: dequant, moe, vgpr, occupancy, mfma, counters, compute-bound, operand-reuse, dtype-dialect
- [gfx950 · mixed] Freeze one operand's pointer at a time to upper-bound every memory-side direction: capped the whole memory system at ~10% on the largest MoE GEMM case ★★ — (freeze-one-operand-s-pointer-at-a-time-to-put-an-upper-bound-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: control-experiment, measurement-method, l2-locality, operand-reuse, moe, counters
- [gfx950 · mixed] Merge G quant groups into ONE dot instead of one dot per group on a packed-weight MoE GEMM: 2.24x vs a 1.94x incumbent (+15%), 2.92x geomean end state ★★ — (merge-several-quant-groups-into-one-dot-instead-of-one-dot-p-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: dequant, quantization-group, mfma, moe, isa-check, interleaved-ab, tile-shape
- [gfx950 · mixed] Register N candidate instances in one build and pick by env var: a 7-config sweep fits in one round and found the run's largest lever (up to +11.4%) ★★ — (one-build-n-selectable-instances-turn-a-config-sweep-into-an-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_gemm_fp8_blockscale · kw: config-sweep, env-switch, build-cost, measurement-method, interleaved-ab, moe, workgroup-size
- [gfx950 · large-batch] Pad the launch grid to a multiple of NUM_XCD so the XCD pid remap engages: +3.3% geomean on an already-retiled MoE grouped GEMM, L2 hit 65%->85% ★★ — (pad-the-grid-to-a-multiple-of-num-xcd-so-the-pid-swizzle-fir-moe-grouped-gemm-gfx950-large-batch.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: pid-remap, l2-locality, xcd, grid-geometry, moe, large-batch, isa-check, interleaved-ab
- [gfx950 · mixed] Pre-lay a strided per-iteration operand in a prologue kernel in the consumer's reading order: +11.5% geomean on an already-converged MoE grouped GEMM ★★ — (pre-lay-a-strided-per-iteration-operand-in-a-prologue-kernel-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: moe, operand-reuse, l2-locality, address-locality, prologue-kernel, measurement-method
- [gfx950 · mixed] Price an inferred padding or grid-quantization tax with a block-count sweep before funding a round: both inferred taxes were fictions and the lane returned 0% ★★ — (price-an-inferred-padding-or-grid-quantization-tax-with-a-bl-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: fused_moe_kernel · kw: grid-geometry, control-experiment, measurement-method, moe, tile-shape, memory-bound, launch-config
- [gfx950 · compute-bound] Re-store the dominant streamed weight operand in fp4 the MFMA reads natively: 31.35x -> 39.95x cumulative in one direction, 42.24x with the wider-K instruction ★★ — (re-store-the-streamed-weight-operand-in-the-narrowest-dtype--moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel · kw: dequant, mfma, moe, operand-reuse, fp8, tile-shape, compute-bound, roofline, dtype-dialect
- [gfx950 · compute-bound] A regressing double-buffer diagnoses a dependency chain, not exposed load latency: one -14% arm closed the whole prefetch/pipeline axis on a MoE GEMM ★★ — (separate-exposed-load-latency-from-a-dependency-chain-before-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: prefetch, pipeline-stages, num-stages, occupancy, moe, dequant, control-experiment
- [gfx950 · mixed] Size each shared-memory buffer from the instantiated template variant, not the generic max: 1.29x on an MoE grouped GEMM by doubling resident workgroups per CU ★★ — (size-shared-memory-from-the-instantiated-variant-not-the-gen-moe-grouped-gemm-gfx950-mixed.md)
  - kernels: moe_stage1 · kw: lds, occupancy, vgpr, moe, template-instantiation, isa-check, code-object, dead-allocation
- [gfx950 · compute-bound] Size the reused operand set against L2 first: on a weight-streaming MoE GEMM whose reuse set is ~120x L2, the whole locality family ceilings at ~1% ★★ — (size-the-reused-operand-set-against-l2-before-pricing-any-lo-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: l2-locality, xcd, pid-remap, operand-reuse, moe, dequant, control-experiment, compute-bound
- [gfx950 · compute-bound] Specialise tile height per batch bucket when packed-weight dequant reloads per tile: a 1.80x step took the verified MoE GEMM geomean from 2.23x to 4.00x ★★ — (specialise-tile-height-per-batch-bucket-when-packed-weight-d-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: tile-shape, dequant, moe, occupancy, vgpr, config-sweep, mfma, prefetch
- [gfx950 · compute-bound] Bucket cases by M and fit the host launch config per bucket before touching a dequant-bound body: 3.26x on its own, and no body rewrite beat it ★★ — (split-the-launch-config-per-m-bucket-on-the-host-before-rewr-moe-grouped-gemm-gfx950-compute-bound.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: launch-config, config-sweep, moe, dequant, tile-shape, compute-bound, roofline, large-batch
- [gfx950 · large-batch] Merge adjacent same-group row-blocks into a double-height tile to amortise weight dequant: ~1.23x on large-batch MoE shapes, slower on the smallest ★ — (merge-adjacent-same-group-row-blocks-to-amortise-weight-dequ-moe-grouped-gemm-gfx950-large-batch.md)
  - kernels: fused_moe_kernel_gptq_awq · kw: dequant, tile-shape, moe, operand-reuse, vgpr, large-batch, interleaved-ab

## quantize_cast
- [gfx950 · memory-bound] Read the ISA for the widest store before a vectorization round on a streaming fp8 quantize: two directions produced no patch against 2.55x/2.40x budgets ★★ — (disassemble-the-store-before-funding-a-vectorization-round-o-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: isa-check, quant, fp8, memory-bound, roofline, occupancy, vgpr, tile-shape, measurement-method
- [gfx950 · mixed] Census the ISA for emulated narrow-dtype convert and divide in a quantize/cast kernel: 3.75x geomean, 4.76x/4.00x on the memory-bound streaming cases ★★ — (narrow-dtype-convert-and-divide-can-lower-to-emulation-censu-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: fp8, dtype-dialect, isa-check, valu-emulation, quant, inline-asm, memory-bound, launch-bound
- [gfx950 · mixed] Run a footprint control before believing a per-case bandwidth number: it showed one quant case cache-served and the largest at ~98% of achievable DRAM BW ★★ — (price-cache-residency-with-a-footprint-control-before-believ-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: roofline, measurement-method, control-experiment, l2-locality, quant, memory-bound, counters, harness-artifact
- [gfx950 · memory-bound] Re-export the launched symbol as a subscriptable launcher and relaunch with your own tile and warp count: 2.32x, against ~1.15x from all in-body work ★★ — (re-export-the-launched-symbol-as-a-subscriptable-launcher-ob-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-config, launch-overhead, kernel-cache, quant, memory-bound, tile-shape, roofline, env-switch, config-sweep
- [gfx950 · memory-bound] Attribute a small case's flat per-call floor to the harness bracket first: graph capture bought ~12% real throughput there and scored exactly zero ★★ — (separate-the-scored-per-call-bracket-from-device-time-before-quantize-cast-gfx950-memory-bound.md)
  - kernels: _per_token_group_quant_fp8 · kw: graph-capture, dispatch-floor, measurement-method, launch-overhead, quant, memory-bound, harness-artifact
- [gfx950 · mixed] Memoize the compiled kernel and call its low-level entry directly on a launch-bound quant/cast shape: 2.77x there, ~1.0x once the case is bandwidth-bound ★★ — (the-generic-triton-launch-path-is-not-the-floor-when-the-sma-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: launch-overhead, dispatch-floor, measurement-method, launch-bound, kernel-cache, quant
- [gfx950 · mixed] Port a winning mechanism into the sibling shape-specialised instances: cross-instance ports paid +1.60-3.95% while stacking three orthogonal patches paid +0.31% ★★ — (when-an-operator-ships-several-shape-specialised-instances-g-quantize-cast-gfx950-mixed.md)
  - kernels: _per_token_group_quant_fp8 · kw: cross-instance-port, measurement-method, interleaved-ab, quant, noise-band, control-experiment

## quantized_gemm
- [gfx950 · compute-bound] Price a manual loop restructure in VGPRs against the next occupancy step, not in spills: two spill-free hoists on a quantized GEMM lost ~10-20% ★★ — (a-spill-free-restructure-can-still-lose-the-win-by-stepping--quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: vgpr, occupancy, pipeline-stages, quantized-gemm, isa-check, compute-bound, operand-reuse, lds-tiling
- [gfx950 · compute-bound] Rescale peak by the CU partition the box exposes before funding a headroom chase: an apparent 33% of peak was ~72% on the large fp8 GEMM cases ★★ — (discount-the-nameplate-relative-roofline-against-the-cu-part-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: roofline, compute-bound, fp8, quant, measurement-method, harness-artifact
- [gfx950 · compute-bound] Empty the K loop of scale arithmetic on a block-scaled fp8 GEMM: three folds/hoists compound +10%, +9.35%, +10.6% to a 20.20x banked geomean ★★ — (empty-the-k-loop-of-scale-arithmetic-entirely-fold-the-co-al-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: quantized-gemm, fp8, quantization-group, dequant, mfma, operand-reuse, roofline, compute-bound
- [gfx950 · compute-bound] Size a dequant-in-loop K loop in operand bytes and register/LDS budget, not instructions: instruction count correlated NEGATIVELY with time on four ladders ★★ — (price-a-dequant-in-loop-k-loop-in-operand-bytes-and-register-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel · kw: dequant, quantized-gemm, vgpr, lds, occupancy, counters, measurement-method, control-experiment, compute-bound
- [gfx950 · compute-bound] Probe the correctness gate with a one-ulp re-association before funding any reduction-reordering lane: it retired four rounds aimed at a ~1.6x roof ★★ — (probe-the-correctness-gate-with-a-one-ulp-re-association-bef-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel, _w8a8_triton_block_scaled_mm · kw: correctness-gate, measurement-method, dtype-dialect, mfma, quantized-gemm, roofline, control-experiment
- [gfx950 · compute-bound] Register-prefetch the SMALL per-iteration operands one k-block ahead as the body's last VMEM: +3.9-7.0% on the large-M cases of a block-scaled GEMM ★★ — (register-prefetch-the-small-per-iteration-operands-one-k-blo-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel, _w8a8_triton_block_scaled_mm · kw: prefetch, vgpr, isa-check, operand-reuse, compute-bound, quant, dequant, mfma
- [gfx950 · compute-bound] Regroup the k reduction into fewer wider dots instead of reassociating it: cumulative geomean 9.17x -> 10.66x at max_rel=0 on all three cases ★★ — (regroup-the-k-reduction-into-fewer-wider-dots-instead-of-rea-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _w8a8_triton_block_scaled_mm · kw: mfma, quantized-gemm, vgpr, occupancy, tile-shape, compute-bound, operand-reuse, correctness-gate, config-sweep
- [gfx950 · compute-bound] Tile/stage/occupancy closes as one axis under a big fp32 accumulator: a 256x64 tile fell to 0.64x, a wider MFMA non-k dim to 0.916x, num_stages=3 no build ★★ — (stop-widening-tiles-and-stages-once-the-accumulator-owns-the-quantized-gemm-gfx950-compute-bound.md)
  - kernels: _gemm_a8w8_blockscale_kernel, _w8a8_triton_block_scaled_mm · kw: tile-shape, occupancy, vgpr, lds, mfma, pipeline-stages, compute-bound, fp8, config-sweep

## topk_routing
- [gfx950 · launch-bound] Collapse Triton's Python launch path before tuning the body of a small launch-bound top-k op: 1.58x standalone, 2.19x on the smallest grid ★★ — (collapse-the-python-launch-path-before-tuning-the-body-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: launch-overhead, launch-bound, dispatch-floor, kernel-cache, topk, measurement-method, interleaved-ab, graph-capture
- [gfx950 · launch-bound] Layout round-trips hidden in the tail behind the critical path are free to keep: deleting all of them moved a launch-bound top-k case inside jitter ★★ — (layout-round-trips-that-sit-in-the-tail-behind-the-critical--topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: lds, cross-lane, topk, launch-bound, control-experiment, isa-check
- [gfx950 · launch-bound] Restate small-k selection over a chunk axis instead of Triton's distributed axis: device time -30 to -33% on every case at zero occupancy cost ★★ — (move-small-k-selection-onto-a-chunk-axis-off-the-distributed-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: topk, cross-lane, isa-check, vgpr, occupancy, launch-bound, interleaved-ab, control-experiment
- [gfx950 · launch-bound] Probe the per-call floor with the real armed launcher, not an empty jitted kernel: bounded all remaining headroom at +7.1% and closed the run ★★ — (probe-the-floor-with-the-real-armed-launcher-and-re-price-pe-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: launch-overhead, dispatch-floor, launch-bound, measurement-method, topk, control-experiment, interleaved-ab
- [gfx950 · launch-bound] A flat baseline wall across a 32x shape sweep is a dispatch floor, not the roofline label: the host lane paid 2.29x on the smallest case, 1.0x on the biggest ★★ — (read-a-flat-baseline-wall-across-a-wide-shape-sweep-as-a-dis-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: dispatch-floor, launch-overhead, host-launch, launch-bound, roofline, topk, measurement-method, control-experiment
- [gfx950 · launch-bound] Reshape a tile into the layout's own factorisation instead of fighting convert_layout: -11.5% instructions, +7.2% on the one device-bound top-k case ★★ — (reshape-a-tile-into-the-layout-s-own-factorisation-instead-o-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: cross-lane, lds, topk, launch-bound, isa-check, tile-shape, workgroup-size
- [gfx950 · launch-bound] Depth reduction on a top-k selection that already fills the device measured 1.008x, while partition-parallel selection on the same code paid 1.185x ★★ — (shortening-a-serial-reduction-chain-buys-nothing-once-the-gr-topk-routing-gfx950-launch-bound.md)
  - kernels: _topk_forward · kw: topk, occupancy, launch-bound, measurement-method, control-experiment, cross-lane, vgpr, dispatch-floor

## method
- [gfx950 · launch-bound] Past the raw driver launch, submit rewrites close the axis: a native shim measured 2.61x vs 2.613x and a doorbell kernel was a clean true-negative ★★ — (past-the-raw-driver-submit-the-remaining-host-submit-ideas-a-unmatched-gfx950-launch-bound.md)
  - kw: launch-overhead, dispatch-floor, launch-bound, graph-capture, measurement-method, control-experiment, interleaved-ab

## keyword vocabulary (generated — REUSE these before coining a new term)
measurement-method(44) · control-experiment(32) · interleaved-ab(26) · vgpr(25) · occupancy(24) · dispatch-floor(21) · isa-check(21) · memory-bound(21) · compute-bound(20) · launch-overhead(19) · tile-shape(19) · moe(18) · launch-bound(17) · operand-reuse(15) · dequant(14) · mfma(14) · roofline(14) · attention(13) · config-sweep(11) · decode(11) · dense-gemm(10) · l2-locality(10) · fp8(9) · graph-capture(9) · quant(9) · counters(8) · dtype-dialect(8) · lds(8) · pipeline-stages(7) · topk(7) · harness-artifact(6) · launch-config(6) · kernel-cache(5) · linear-attention(5) · quantized-gemm(5) · cross-lane(4) · prefetch(4) · code-object(3) · env-switch(3) · grid-geometry(3) · host-launch(3) · large-batch(3) · lds-tiling(3) · noise-band(3) · pid-remap(3) · waves-per-eu(3) · xcd(3) · address-locality(2) · cache-policy(2) · clock-drift(2) · coarsening(2) · correctness-gate(2) · memory-movement(2) · non-temporal(2) · quantization-group(2) · streaming-operand(2) · valu-emulation(2) · workgroup-size(2) · async-copy · autotune · bijection · build-cost · cross-instance-port · dead-allocation · fence · inline-asm · null-baseline · num-stages · numerics-gate · production-grid · prologue-kernel · reassociation · split-k · template-instantiation · tile-aspect · vendor-library

> ⚠ **Over the per-class cap of 8:** attention (16), dense_gemm (10), moe_grouped_gemm (18). Archive the lowest `confidence × freshness × standing` card in that class
> (★★★ is never auto-evicted), then regenerate.
