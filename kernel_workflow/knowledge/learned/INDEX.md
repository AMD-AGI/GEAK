

<!-- CARDS:BEGIN -->
## attention · gfx950 · decode
- ★★ null-the-kernel-body-before-optimizing-it-attention-gfx950-decode — An instrument, not a speedup: gutting a small dispatch's whole body to a trivial copy (deliberately wrong output) measured 4.00 us min vs… [cited 0 / blind 1 / lost 0 / attempts 1]
- ★★ re-linearise-the-workgroup-id-in-the-prologue-for-address-lo-attention-gfx950-decode — director-verified 1.86x geomean end state (1.82 / 2.11 / 1.67 at batch 2 / 32 / 64, decode q_len=1); this lever's own A/B was +3.1% and s… [cited 0 / blind 1 / lost 0 / attempts 3]
- ★★ return-the-tile-loop-to-the-backend-async-copy-pipeliner-attention-gfx950-decode — director-verified 1.86x geomean end state (1.82 / 2.11 / 1.67 at batch 2 / 32 / 64, decode q_len=1); this lever's own A/B was +11.0% over… [cited 0 / blind 1 / lost 0 / attempts 5]
- ★★ size-the-host-launch-path-before-the-device-on-decode-shapes-attention-gfx950-decode — 1.51x geomean director-verified; per-case 1.63x on the smallest decode batch, 1.52x mid, 1.39x on the largest — the ratio decays as devic… [cited 0 / blind 1 / lost 0 / attempts 3]

## dense gemm · gfx950 · compute-bound
- ★★ price-the-vendor-library-on-the-exact-shapes-before-funding--dense-gemm-gfx950-compute-bound — director-verified 3.71x geomean over the untuned capture (per-case 3.00x on the small-M shape M~2K, 4.18x and 4.09x on the large-M shapes… [cited 1 / blind 0 / lost 0 / attempts 1]
- ★★ tile-aspect-at-fixed-area-under-a-hard-accumulator-per-lane--dense-gemm-gfx950-compute-bound — launch-config plus pid-order retuning alone carried a captured GEMM from 1.00x to 2.64x with a byte-identical kernel body, and a fixed-ar… [cited 1 / blind 0 / lost 0 / attempts 4]

## linear attention · gfx950 · launch-bound
- ★★ audit-the-autotune-config-list-for-a-missing-axis-before-tun-attention-gfx950-launch-bound — adding the MFMA instruction-size axis (with a wider K-tile) to a config list that omitted it beat the converged autotuner by 1.18x on the… [cited 0 / blind 1 / lost 0 / attempts 3]
- ★★ measure-a-production-grid-column-beside-the-harness-column-attention-gfx950-launch-bound — verified geomean 22.68x on the harness grid vs 2.09x re-measured on the production grid; the gap tracked the over-launched dimension exac… [cited 0 / blind 1 / lost 0 / attempts 13]

## memory movement · gfx950 · launch-bound
- ★★ bypass-the-triton-dispatch-path-when-the-metric-is-host-laun-memory-movement-gfx950-launch-bound — 2.24x on its own and 2.35x cumulative (director-verified) on a tiny paged copy whose per-call time profiled as ~100% host dispatch; it he… [cited 0 / blind 1 / lost 0 / attempts 6]
- ★★ grade-launch-bound-work-against-a-cheapest-packet-control-no-memory-movement-gfx950-launch-bound — Called the graded outcome 3/3 times: an empty kernel measured 18.8 us vs 16.1 us for the real one before the host path was collapsed and… [cited 0 / blind 1 / lost 0 / attempts 3]

## moe grouped gemm · gfx950 · large-batch
- ★★ merge-adjacent-same-group-row-blocks-to-amortise-weight-dequ-moe-grouped-gemm-gfx950-large-batch — Director-verified per-case 1.39x on the smallest shape and 2.03-2.16x on the two large shapes (geomean 1.83x) for the patch it anchors. W… [cited 0 / blind 1 / lost 0 / attempts 6]
- ★★ pad-the-grid-to-a-multiple-of-num-xcd-so-the-pid-swizzle-fir-moe-grouped-gemm-gfx950-large-batch — Standalone A/B on one compiled binary: +2.5% on the smallest case, +6.0% and +4.4% on the two larger cases (L2 hit 64.7 -> 85.1%, L2 miss… [cited 0 / blind 1 / lost 0 / attempts 3]

## moe grouped gemm · gfx950 · mixed
- ★★ a-frozen-launch-config-still-leaves-the-program-to-tile-map--moe-grouped-gemm-gfx950-mixed — two fusion steps took the verified geomean 17.2x -> 46.5x (+30%, then +107%); per-case at that point the two large-batch cases sat at 57-… [cited 0 / blind 1 / lost 0 / attempts 6]
- ★★ check-whether-the-fp8-operand-dialect-is-the-one-this-part-e-moe-grouped-gemm-gfx950-mixed — 15.4x geomean from one edit when the operands were in the non-native fp8 dialect; it held on every case, and the round's integrated build… [cited 0 / blind 1 / lost 0 / attempts 1]
- ★★ one-build-n-selectable-instances-turn-a-config-sweep-into-an-moe-grouped-gemm-gfx950-mixed — Three separate multi-config instance sweeps returned nothing across three rounds while paying one build per config; registering candidate… [cited 0 / blind 1 / lost 0 / attempts 6]
- ★★ renumber-workgroups-for-weight-slice-contiguity-live-prefix--moe-grouped-gemm-gfx950-mixed — director-verified 1.49x geomean end state, per-case 1.38x / 1.57x / 1.54x from the smallest to the largest token count; this axis contrib… [cited 0 / blind 1 / lost 0 / attempts 4]
- ★★ size-shared-memory-from-the-instantiated-variant-not-the-gen-moe-grouped-gemm-gfx950-mixed — director-verified 1.49x geomean end state; this lever ALONE measured 1.29x, per-case 1.21x at the smallest token count and 1.34x / 1.34x… [cited 0 / blind 1 / lost 0 / attempts 2]
- ★★ size-the-workgroup-per-stage-the-low-k-iteration-stage-wants-moe-grouped-gemm-gfx950-mixed — Director-verified whole stack 1.22x at the smallest scored batch, 1.27x mid, 1.29x at the largest (geomean 1.26); this lever alone, paire… [cited 0 / blind 1 / lost 0 / attempts 12]

## quantize / cast · gfx950 · mixed
- ★★ narrow-dtype-convert-and-divide-can-lower-to-emulation-censu-quantize-cast-gfx950-mixed — Full stack director-verified 3.75x geomean: 2.77x on the small launch-bound case, 4.76x and 4.00x on the two large memory-bound streaming… [cited 0 / blind 1 / lost 0 / attempts 5]
- ★★ the-generic-triton-launch-path-is-not-the-floor-when-the-sma-quantize-cast-gfx950-mixed — Director-verified per-case: 2.77x on the small launch-bound case vs 4.76x and 4.00x on the large memory-bound cases (3.75x geomean). The… [cited 0 / blind 1 / lost 0 / attempts 4]

## quantized gemm · gfx950 · compute-bound
- ★★ fnuz-fp8-operands-can-silently-disable-the-native-matrix-cor-quantized-gemm-gfx950-compute-bound — 9.30 / 9.14 / 9.21x standalone at M=2k / 32k / 64k, and the precondition for the run's 23.45x director-verified geomean (25.6x at the sma… [cited 0 / blind 1 / lost 0 / attempts 2]
- ★★ on-a-latency-bound-gemm-source-statement-order-is-a-tunable--quantized-gemm-gfx950-compute-bound — +3.6% at M=64k from moving one load statement; +5.5% / +3.5% / +3.6% at M=2k / 32k / 64k from a whole-body sweep the next round; +1.0% mo… [cited 0 / blind 1 / lost 0 / attempts 4]
- ★★ hand-write-the-fp8-upcast-as-packed-dword-swar-quantized-gemm-gfx950-compute-bound — the hand-written bit-trick upcast alone measured 5.92x over the stock kernel, and moving the same bit arithmetic to packed dwords a furth… [cited 0 / blind 1 / lost 0 / attempts 6]
- ★★ perf-only-config-sweeps-elect-silently-wrong-configs-quantized-gemm-gfx950-compute-bound — a correctness gate on the sweep caught 3 tile configs and 3 launch-knob configs that ran fast and computed wrong output, among them the s… [cited 0 / blind 1 / lost 0 / attempts 4]

## topk / routing · gfx950 · launch-bound
- ★★ collapse-the-python-launch-path-before-tuning-the-body-topk-routing-gfx950-launch-bound — 1.58x geomean standalone, and per-case it tracks how launch-bound the case is: 2.19x on the smallest grid (~64 CTAs, device time fully hi… [cited 0 / blind 1 / lost 0 / attempts 4]
- ★★ move-small-k-selection-onto-a-chunk-axis-off-the-distributed-topk-routing-gfx950-launch-bound — Device time -30 to -33% on every case (min 3280/4560/7160 -> 2200/3200/4840 ns) at zero occupancy cost; per-case at the wall it paid only… [cited 0 / blind 1 / lost 0 / attempts 4]
<!-- CARDS:END -->
