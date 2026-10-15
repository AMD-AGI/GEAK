---
title: Run recurrence — base rates distilled from the learned card trees
kind: reference
generated_by: index/_gen_run_recurrence.py
---

# Run recurrence — what keeps holding across runs

**GENERATED. Never hand-edit.** Regenerate with `python3 perf_knowledge/index/_gen_run_recurrence.py`; CI runs it with `--check`.

A learned card is one run's conclusion. This file is the part no single card can hold: for each optimization axis, on how many *distinct kernels* it paid, and on how many it was already closed. Rows are counts and links only — the claims stay in the cards, because the generalization is the one thing no run measured.

Read it as a **prior about where to look first**, never as a verdict. [`README.md`](../README.md) applies unchanged: this base proposes candidates, the box decides.

Corpus: **166 active cards** across kernel=137 (137 with keywords), e2e=29 (8 with keywords) · **150 axes published** (>= 3 distinct kernels), 153 below threshold.

## Cross-operator axes

An axis reaches this table at >= 3 distinct kernels and >= 3 distinct kernel classes: measured widely enough that it is a statement about the hardware rather than about one operator.

| axis | paid on | closed on | how to measure | kernels | classes | cards |
|---|---|---|---|---|---|---|
| `launch-overhead` | 15 | 17 | 3 | 27 | 12 | 41 |
| `occupancy` | 9 | 18 | 0 | 22 | 8 | 33 |
| `grouped-gemm` | 10 | 6 | 2 | 20 | 3 | 22 |
| `moe` | 10 | 6 | 2 | 20 | 3 | 21 |
| `roofline` | 4 | 11 | 2 | 17 | 7 | 14 |
| `kernel-fusion` | 13 | 3 | 0 | 15 | 5 | 8 |
| `cache-modifier` | 10 | 7 | 1 | 14 | 5 | 17 |
| `hip-graph` | 5 | 12 | 0 | 14 | 8 | 16 |
| `split-k` | 10 | 5 | 1 | 14 | 5 | 12 |
| `dispatch-collapse` | 12 | 2 | 0 | 13 | 4 | 8 |
| `dispatch-floor` | 4 | 6 | 0 | 12 | 5 | 8 |
| `paged-attention` | 10 | 6 | 2 | 12 | 4 | 24 |
| `grid-occupancy` | 9 | 2 | 0 | 11 | 5 | 7 |
| `mfma` | 7 | 6 | 1 | 11 | 5 | 16 |
| `graph-replay` | 4 | 5 | 2 | 10 | 7 | 9 |
| `host-dispatch` | 1 | 2 | 0 | 10 | 3 | 4 |
| `host-runtime` | 5 | 6 | 1 | 10 | 10 | 13 |
| `num-stages` | 6 | 4 | 1 | 10 | 6 | 10 |
| `cross-workgroup` | 6 | 2 | 0 | 8 | 3 | 4 |
| `raw-hip` | 7 | 1 | 0 | 8 | 4 | 7 |
| `size-gating` | 6 | 0 | 2 | 8 | 4 | 5 |
| `stacking` | 5 | 0 | 3 | 8 | 3 | 5 |
| `cu-underfill` | 6 | 1 | 0 | 7 | 3 | 3 |
| `measurement-discipline` | 3 | 2 | 2 | 7 | 3 | 5 |
| `num-warps` | 5 | 4 | 0 | 7 | 5 | 9 |
| `tile-geometry` | 6 | 2 | 0 | 7 | 4 | 6 |
| `waves-per-eu` | 5 | 2 | 0 | 7 | 3 | 6 |
| `block-scale` | 6 | 1 | 1 | 6 | 4 | 12 |
| `launch-config` | 5 | 1 | 1 | 6 | 5 | 7 |
| `lds-tiling` | 4 | 3 | 0 | 6 | 4 | 7 |
| `xcd-swizzle` | 3 | 3 | 0 | 6 | 3 | 5 |
| `lds-staging` | 4 | 2 | 0 | 5 | 3 | 5 |
| `non-temporal-store` | 5 | 0 | 0 | 5 | 3 | 3 |
| `software-pipelining` | 1 | 4 | 0 | 5 | 4 | 4 |
| `bit-exact` | 4 | 0 | 1 | 4 | 4 | 7 |
| `cuda-graph` | 0 | 4 | 0 | 4 | 3 | 4 |
| `paired-ab-rig` | 3 | 1 | 1 | 4 | 3 | 5 |
| `software-prefetch` | 0 | 4 | 0 | 4 | 3 | 4 |
| `unroll` | 4 | 0 | 0 | 4 | 3 | 3 |
| `xcd-remap` | 3 | 1 | 1 | 4 | 3 | 4 |
| `persistent-kernel` | 0 | 3 | 0 | 3 | 3 | 3 |
| `topk` | 3 | 1 | 0 | 3 | 3 | 4 |
| `valu-bound` | 3 | 0 | 0 | 3 | 3 | 4 |
| `vgpr` | 1 | 2 | 0 | 3 | 3 | 3 |

`paid` / `closed` count DISTINCT KERNELS, and they overlap on purpose: the same axis can pay on one kernel and be shut on another, which is the base rate. Where `closed` leads, the axis is not useless — it is the one to price before funding a round on it.

## Per-operator axes

Concentrated (>= 80% of the axis's kernels in one class) and therefore knowledge about that operator. `operator` is the [`taxonomy.md`](taxonomy.md) id, so the row points at a directory that exists.

| operator | axis | paid on | closed on | kernels | cards |
|---|---|---|---|---|---|
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `1stage` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `AITER_CONFIG_FMOE` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `e8m0` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `fused_moe` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `MI355X` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `per_1x32` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `quark` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `tuned_fmoe` | 0 | 0 | 4 | 1 |
| (unmapped: MoE grouped GEMM (fused dispatcher)) | `xbf16` | 0 | 0 | 4 | 1 |
| (unmapped: attention-decode) | `decode attention` | 5 | 0 | 5 | 1 |
| (unmapped: attention-decode) | `hybrid SWA` | 5 | 0 | 5 | 1 |
| (unmapped: attention-decode) | `launcher overlay` | 5 | 0 | 5 | 1 |
| (unmapped: attention-decode) | `num_kv_splits` | 5 | 0 | 5 | 1 |
| (unmapped: attention-decode) | `partition_size` | 4 | 0 | 5 | 2 |
| (unmapped: attention-decode) | `triton_backend` | 5 | 0 | 5 | 1 |
| (unmapped: attention-decode) | `window_num_kv_splits` | 5 | 0 | 5 | 1 |
| (unmapped: attention-decode) | `byte-exact env win` | 4 | 0 | 4 | 1 |
| (unmapped: attention-decode) | `cpp_itfs JIT` | 4 | 0 | 4 | 1 |
| (unmapped: attention-decode) | `QKV_VERSION` | 4 | 0 | 4 | 1 |
| (unmapped: attention-decode) | `ragged kv_indptr` | 4 | 0 | 4 | 1 |
| (unmapped: attention-decode) | `block_size 1` | 0 | 0 | 3 | 1 |
| (unmapped: attention-decode) | `kv-fp8` | 0 | 0 | 3 | 1 |
| (unmapped: attention-decode) | `ll4mi` | 0 | 0 | 3 | 1 |
| (unmapped: attention-decode) | `mfma16` | 0 | 0 | 3 | 1 |
| (unmapped: attention-decode) | `paged_attention_ragged` | 0 | 0 | 3 | 1 |
| (unmapped: attention-decode) | `ragged` | 0 | 0 | 3 | 1 |
| (unmapped: method) | `carry-forward` | 0 | 0 | 8 | 1 |
| (unmapped: method) | `cluster` | 0 | 0 | 8 | 1 |
| (unmapped: method) | `extraction-seam` | 0 | 0 | 8 | 1 |
| (unmapped: method) | `measurement` | 0 | 2 | 8 | 8 |
| (unmapped: method) | `moe-routing` | 0 | 0 | 8 | 1 |
| (unmapped: method) | `noise-band` | 0 | 0 | 8 | 1 |
| (unmapped: method) | `frozen-baseline` | 0 | 1 | 6 | 6 |
| (unmapped: method) | `ab-methodology` | 0 | 1 | 5 | 6 |
| (unmapped: method) | `noise-floor` | 0 | 1 | 5 | 5 |
| (unmapped: method) | `negative-control` | 0 | 0 | 4 | 4 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `paged-kv` | 5 | 3 | 6 | 10 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `empty-workgroups` | 4 | 1 | 5 | 4 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `kv-cache` | 4 | 1 | 5 | 5 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `launch-shape` | 5 | 0 | 5 | 3 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `register-pressure` | 3 | 1 | 5 | 4 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `hardware-counters` | 3 | 1 | 4 | 2 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `constexpr-promotion` | 3 | 0 | 3 | 2 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `fp8-kv` | 3 | 1 | 3 | 3 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `host-wrapper` | 3 | 0 | 3 | 2 |
| [`attention_decode_paged`](../operators/attention_decode_paged/) | `isa-diff` | 1 | 2 | 3 | 3 |
| [`dense_gemm`](../operators/dense_gemm/) | `accuracy-gate` | 6 | 0 | 6 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `authored-triton` | 6 | 0 | 6 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `cudagraph-safe` | 6 | 0 | 6 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `lazy-import` | 6 | 0 | 6 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `rebind-seam` | 6 | 0 | 6 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `sitecustomize-overlay` | 6 | 0 | 6 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `tier-c-author` | 6 | 0 | 6 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `AITER_CONFIG_GEMM_BF16` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `cold-cache` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `colon-merge` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `decode-roofline` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `gemm_a16w16` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `gemm_tuner` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `get_padded_m` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `gradlib` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `shipped-coverage` | 3 | 0 | 3 | 1 |
| [`dense_gemm`](../operators/dense_gemm/) | `tuned_gemm` | 3 | 0 | 3 | 1 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `byte-exact-parity` | 0 | 0 | 5 | 1 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `prologue` | 4 | 0 | 5 | 3 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `pythonpath-shadow` | 0 | 0 | 5 | 1 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `routing-metadata` | 0 | 0 | 5 | 1 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `warm-start-replay` | 0 | 0 | 5 | 1 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `lds-padding` | 2 | 1 | 3 | 3 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `m-bucket` | 3 | 0 | 3 | 3 |
| [`fused_moe_grouped_gemm`](../operators/fused_moe_grouped_gemm/) | `mfma-nonkdim` | 1 | 2 | 3 | 3 |

Kernel classes with cards but no operator id, left unfiled rather than filed plausibly: `MoE grouped GEMM (fused dispatcher)` (1 cards), `attention-decode` (3 cards), `composable` (1 cards), `fused_norm_gemm` (3 cards), `memory_movement` (4 cards), `method` (22 cards), `other` (21 cards). `method` and `composable` have no operator by construction — they are about how to measure, and a method row filed under an operator would be read as being about that operator. `other` is the parser's fallback for a card with no `kernel_class` at all, which is every e2e card. Anything else here wants a line in `CLASS_TO_OPERATOR`.

## By authoring language

The lane that lets a FlyDSL run see FlyDSL evidence instead of Triton evidence. It is fed by the card `language:` field, which is deliberately NOT backfilled onto older cards — a guessed language is worse than an absent one. Which write paths require it is not uniform; the breakdown below says which.

| language | cards | axes |
|---|---|---|
| `flydsl` | 1 | — |

## By architecture

Axes whose verdict differs across `gfx` targets — the rows worth carrying into [`hardware/`](../hardware/). Needs one axis measured on two platforms.

| axis | platforms | cards |
|---|---|---|
| `AITER_CONFIG_GEMM_BF16` | gfx942, gfx950 | 1 |
| `amdahl` | gfx942, gfx950 | 2 |
| `byte-exact env win` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17 | 1 |
| `carry-forward` | gfx942, gfx950 | 1 |
| `cluster` | gfx942, gfx950 | 1 |
| `cold-cache` | gfx942, gfx950 | 1 |
| `colon-merge` | gfx942, gfx950 | 1 |
| `cpp_itfs JIT` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17 | 1 |
| `decode attention` | ROCm 7.2, gfx950/MI355X, sglang 0.5.17 | 1 |
| `decode-roofline` | gfx942, gfx950 | 1 |
| `extraction-seam` | gfx942, gfx950 | 1 |
| `flash-decoding` | ROCm 7.2, gfx950, gfx950/MI355X, sglang 0.5.17 | 3 |
| `gemm_a16w16` | gfx942, gfx950 | 1 |
| `gemm_tuner` | gfx942, gfx950 | 1 |
| `get_padded_m` | gfx942, gfx950 | 1 |
| `gradlib` | gfx942, gfx950 | 1 |
| `grouped-gemm` | gfx950, gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4 | 22 |
| `host-dispatch` | gfx942, gfx950 | 4 |
| `hybrid SWA` | ROCm 7.2, gfx950/MI355X, sglang 0.5.17 | 1 |
| `launcher overlay` | ROCm 7.2, gfx950/MI355X, sglang 0.5.17 | 1 |
| `moe` | gfx950, gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4 | 21 |
| `moe-routing` | gfx942, gfx950 | 1 |
| `noise-band` | gfx942, gfx950 | 1 |
| `num_kv_splits` | ROCm 7.2, gfx950/MI355X, sglang 0.5.17 | 1 |
| `paged-attention` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950/MI355X, sglang 0.5.17 | 24 |
| `partition_size` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950/MI355X, sglang 0.5.17 | 2 |
| `QKV_VERSION` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17 | 1 |
| `ragged kv_indptr` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17 | 1 |
| `sglang` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4, gfx950/MI355X, sglang 0.5.17 | 4 |
| `shipped-coverage` | gfx942, gfx950 | 1 |
| `sliding-window` | ROCm 7.2, gfx950, gfx950/MI355X, sglang 0.5.17 | 2 |
| `split-k` | gfx942, gfx950 | 12 |
| `split-kv` | ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950/MI355X, sglang 0.5.17 | 11 |
| `triton_backend` | ROCm 7.2, gfx950/MI355X, sglang 0.5.17 | 1 |
| `tuned_gemm` | gfx942, gfx950 | 1 |
| `whole-file-overlay` | gfx942, gfx950 | 2 |
| `window_num_kv_splits` | ROCm 7.2, gfx950/MI355X, sglang 0.5.17 | 1 |

## Below threshold

Seen, but on fewer than 3 distinct kernels. Listed so the pipeline is visible: an axis absent from this file entirely has never been tried, which is a different thing from tried twice.

- **2 kernel(s)**: `ab-protocol` · `bottleneck-shift` · `config-sweep` · `counter-guided` · `cshuffle` · `dependency-chain` · `dispatch-bound` · `dtype-emulation` · `gated-lever` · `grid-fill` · `grid-gating` · `harness-artifact` · `instruction-schedule` · `isa-census` · `measurement-floor` · `measurement-rig` · `memoization` · `mfma-tiling` · `packed-valu` · `pipeline-version` · `prefetch` · `skinny-m` · `small-effect` · `static-isa-screen` · `tile-shape` · `verification` · `w4a16` · `workgroup-mapping`
- **1 kernel(s)**: `ab-harness` · `accumulator` · `agpr` · `amortization` · `api-surface` · `argmin-dispatch` · `assembly-inspection` · `async-copy` · `atomic-combine` · `backend-routing` · `bit-exact-gate` · `bit-reinterpret` · `bitcast` · `block-m` · `block-scaled` · `block-scaled-gemm` · `block-size-k` · `bucket-routing` · `cache-line` · `caching` · `campaign-brief` · `co-residency` · `code-size` · `convert-layout` · `counter-falsification` · `cpu-locate-gpu-price` · `critical-path` · `cshuffle-epilogue` · `ctypes` · `custom-op` · `dep-chain` · `dep-stall` · `dequant-hoist` · `dispatch-cache` · `dispatch-overhead` · `dispatch-shim` · `division` · `dot-scaled` · `double-buffer` · `dtype-bitcast` · `emulation-fallback` · `false-negative` · `flash-decode` · `fnuz` · `fp8-kv-cache` · `fp8-mfma` · `fusion-width` · `gemv` · `gqa-head-sharing` · `grid-collapse` · `grid-dedup` · `harness` · `hbm-bound` · `host-overhead` · `host-shim` · `host-submit` · `host-tuning` · `hot-loop` · `ilp` · `int4-dequant` · `jit` · `jit-rebuild` · `k-loop` · `l2-reuse` · `l2-swizzle` · `launch-meta` · `launch-tuning` · `lds` · `lds-bank-conflict` · `lds-capacity` · `m-coarsening` · `measurement-noise` · `mfma-interlock` · `mfma-nonkdim16` · `module-aliasing` · `moe-router` · `native-convert` · `numerics` · `packed-loads` · `paged-decode` · `parity-gate` · `per-bucket-tuning` · `percent-of-peak` · `persistent-grid` · `ping-pong` · `pipeline-variant` · `power-cap` · `preshuffle` · `profiling-method` · `quantization-contract` · `read-only-twin` · `reciprocal` · `reduction-order` · `register-math` · `register-spill` · `register-staging` · `scale-operand` · `search-strategy` · `serialisation` · `sign-consistency` · `split-entry` · `stale-binary` · `store-bandwidth` · `store-vectorization` · `sub-k-coarsening` · `sub-variance` · `super-tile` · `thermal-drift` · `threadfence` · `tile-selection` · `timing-drift` · `tiny-kernel` · `torch-library` · `triton-pipeliner` · `tuning-knobs` · `upstream-diff` · `varlen` · `vendor-library` · `vendored-source` · `weight-only-quant` · `weight-quantization` · `wg-geometry` · `wrapper-overhead` · `wrapper-relaunch` · `xcd-partitioning`

## Evidence

Every published axis with the cards behind it, so a row can be audited back to the run that measured it. `paid` / `closed` / `how to measure` is the card's `type:`.

One card per line on purpose: this file is regenerated on every run that writes a card and CI diffs it, so a single new card should produce a one-line diff rather than rewrite a paragraph.

### `launch-overhead`
27 distinct kernels · 12 classes · scope `general` · platforms gfx950

**paid** — 15 kernels, 12 cards:
- [arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed.md)
- [bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both](../../kernel_workflow/knowledge/learned/bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both.md)
- [cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode.md)
- [collapse-a-redundant-launch-grid-instead-of-guarding-inside--linear-attention-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-a-redundant-launch-grid-instead-of-guarding-inside--linear-attention-gfx950-launch-bound.md)
- [collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both.md)
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both.md)
- [decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode.md)
- [delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
- [dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode.md)
- [dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both](../../kernel_workflow/knowledge/learned/dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both.md)
- [raw-driver-launch-for-a-dispatch-bound-copy-op-memory-movement-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/raw-driver-launch-for-a-dispatch-bound-copy-op-memory-movement-gfx950-launch-bound.md)
**closed** — 17 kernels, 25 cards:
- [a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound.md)
- [axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)
- [axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound.md)
- [axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both.md)
- [five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
- [four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)
- [four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both](../../kernel_workflow/knowledge/learned/four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both.md)
- [gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound.md)
- [graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both](../../kernel_workflow/knowledge/learned/graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both.md)
- [graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode.md)
- [graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch.md)
- [host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill.md)
- [host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill.md)
- [measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode.md)
- [measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode.md)
- [positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode.md)
- [price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both.md)
- [size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound.md)
- [the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both](../../kernel_workflow/knowledge/learned/the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both.md)
- [the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both](../../kernel_workflow/knowledge/learned/the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both.md)
- [the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound.md)
- [the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode.md)
- [traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound.md)
**how to measure** — 3 kernels, 3 cards:
- [a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a.md)
- [audit-the-baseline-launch-grid-before-believing-a-large-head-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/audit-the-baseline-launch-grid-before-believing-a-large-head-linear-attention-gfx950-prefill.md)
- [gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound.md)
**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)

### `occupancy`
22 distinct kernels · 8 classes · scope `general` · platforms gfx950

**paid** — 9 kernels, 8 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode.md)
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill.md)
- [launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode.md)
- [pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill.md)
- [per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed.md)
- [split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)
**closed** — 18 kernels, 25 cards:
- [axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode.md)
- [axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both.md)
- [axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound.md)
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
- [cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
- [dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)
- [diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
- [five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)
- [geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode.md)
- [high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)
- [host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill.md)
- [instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill.md)
- [nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed.md)
- [occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound.md)
- [once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound.md)
- [price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill.md)
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)
- [six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill.md)
- [the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound.md)
- [the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound.md)
- [traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound.md)
- [where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)
- [where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound.md)

### `grouped-gemm` (also spelled `grouped GEMM`)
20 distinct kernels · 3 classes · scope `general` · platforms gfx950, gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**paid** — 10 kernels, 9 cards:
- [32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both.md)
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
- [invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed.md)
- [mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
- [narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed.md)
- [per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed.md)
- [per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound.md)
- [pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 6 kernels, 9 cards:
- [a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed.md)
- [activation-narrowing-is-gated-by-parity-and-by-the-benchmark-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/activation-narrowing-is-gated-by-parity-and-by-the-benchmark-moe-grouped-gemm-gfx950-mixed.md)
- [axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
- [cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
- [count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch.md)
- [dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)
- [diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
- [graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch.md)
- [where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)
**how to measure** — 2 kernels, 2 cards:
- [prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a.md)
- [route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed.md)
**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)
**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `moe` (also spelled `MoE`)
20 distinct kernels · 3 classes · scope `general` · platforms gfx950, gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**paid** — 10 kernels, 9 cards:
- [32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both.md)
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
- [invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed.md)
- [mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
- [narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed.md)
- [per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed.md)
- [per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound.md)
- [pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 6 kernels, 8 cards:
- [a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed.md)
- [activation-narrowing-is-gated-by-parity-and-by-the-benchmark-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/activation-narrowing-is-gated-by-parity-and-by-the-benchmark-moe-grouped-gemm-gfx950-mixed.md)
- [axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
- [count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch.md)
- [dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)
- [diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
- [graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch.md)
- [where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)
**how to measure** — 2 kernels, 2 cards:
- [prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a.md)
- [route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed.md)
**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)
**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `roofline`
17 distinct kernels · 7 classes · scope `general` · platforms gfx950

**paid** — 4 kernels, 2 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound.md)
**closed** — 11 kernels, 10 cards:
- [axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode.md)
- [axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both.md)
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
- [high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)
- [instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode.md)
- [measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode.md)
- [once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound.md)
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)
- [the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound.md)
**how to measure** — 2 kernels, 2 cards:
- [hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode](../../kernel_workflow/knowledge/learned/hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode.md)
- [scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound.md)

### `kernel-fusion`
15 distinct kernels · 5 classes · scope `general` · platforms gfx950

**paid** — 13 kernels, 5 cards:
- [collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both.md)
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
- [fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both](../../kernel_workflow/knowledge/learned/fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
**closed** — 3 kernels, 3 cards:
- [axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both.md)
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)
- [split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode.md)

### `amdahl`
14 distinct kernels · 2 classes · scope `narrow` · platforms gfx942, gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)
**backend choice** — 8 kernels, 1 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)

### `cache-modifier`
14 distinct kernels · 5 classes · scope `general` · platforms gfx950

**paid** — 10 kernels, 10 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both.md)
- [choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both.md)
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
- [get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both.md)
- [reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
- [shorten-the-load-to-dot-chain-before-chasing-bytes-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/shorten-the-load-to-dot-chain-before-chasing-bytes-linear-attention-gfx950-prefill.md)
- [streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound.md)
**closed** — 7 kernels, 6 cards:
- [axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)
- [dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)
- [four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
- [host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill.md)
- [near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound.md)
- [only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode.md)
**how to measure** — 1 kernels, 1 cards:
- [ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode.md)

### `hip-graph`
14 distinct kernels · 8 classes · scope `general` · platforms gfx950

**paid** — 5 kernels, 3 cards:
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound.md)
- [raw-driver-launch-for-a-dispatch-bound-copy-op-memory-movement-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/raw-driver-launch-for-a-dispatch-bound-copy-op-memory-movement-gfx950-launch-bound.md)
**closed** — 12 kernels, 13 cards:
- [a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound.md)
- [axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)
- [four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both](../../kernel_workflow/knowledge/learned/four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both.md)
- [graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode.md)
- [graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch.md)
- [host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-and-knob-axes-that-measured-closed-on-a-one-node-launch-linear-attention-gfx950-prefill.md)
- [host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill.md)
- [price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both.md)
- [size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound.md)
- [the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both](../../kernel_workflow/knowledge/learned/the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both.md)
- [the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound.md)
- [where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)

### `sglang`
14 distinct kernels · 2 classes · scope `narrow` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4, gfx950/MI355X, sglang 0.5.17

**paid** — 9 kernels, 2 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)
**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)
**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `split-k`
14 distinct kernels · 5 classes · scope `general` · platforms gfx942, gfx950

**paid** — 10 kernels, 4 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)
- [split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch.md)
- [unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)
**closed** — 5 kernels, 7 cards:
- [a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both.md)
- [axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
- [axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both.md)
- [count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch.md)
- [once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound.md)
- [the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound.md)
**how to measure** — 1 kernels, 1 cards:
- [re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound.md)

### `split-kv`
14 distinct kernels · 2 classes · scope `narrow` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950/MI355X, sglang 0.5.17

**paid** — 13 kernels, 7 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)
- [collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode.md)
- [one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)
- [split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)
**closed** — 3 kernels, 4 cards:
- [a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode.md)
- [high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)
- [reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode.md)
- [split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode.md)

### `dispatch-collapse`
13 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 12 kernels, 6 cards:
- [collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both.md)
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
- [delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
- [one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)
**closed** — 2 kernels, 2 cards:
- [measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode.md)
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)

### `whole-file-overlay`
13 distinct kernels · 2 classes · scope `narrow` · platforms gfx942, gfx950

**backend choice** — 13 kernels, 2 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)

### `dispatch-floor`
12 distinct kernels · 5 classes · scope `general` · platforms gfx950

**paid** — 4 kernels, 2 cards:
- [cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode.md)
- [collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both.md)
**closed** — 6 kernels, 5 cards:
- [axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
- [graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/graph-capture-of-a-single-tiny-dispatch-can-be-a-higher-floo-dense-gemm-gfx950-decode.md)
- [measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode.md)
- [the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode.md)
**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)

### `paged-attention` (also spelled `paged attention`)
12 distinct kernels · 4 classes · scope `general` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950/MI355X, sglang 0.5.17

**paid** — 10 kernels, 13 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode.md)
- [collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
- [decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode.md)
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode.md)
- [drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
- [enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode.md)
- [fp8-kv-storage-with-bf16-in-register-dequant-on-scatter-boun-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/fp8-kv-storage-with-bf16-in-register-dequant-on-scatter-boun-attention-gfx950-memory-bound.md)
- [one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)
- [unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)
**closed** — 6 kernels, 8 cards:
- [a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode.md)
- [axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode.md)
- [geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode.md)
- [measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode.md)
- [only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode.md)
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)
- [the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode.md)
- [traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/traffic-is-the-only-live-axis-once-attention-is-scatter-boun-attention-gfx950-memory-bound.md)
**how to measure** — 2 kernels, 2 cards:
- [a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a.md)
- [ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode.md)
**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `grid-occupancy`
11 distinct kernels · 5 classes · scope `general` · platforms gfx950

**paid** — 9 kernels, 5 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both](../../kernel_workflow/knowledge/learned/fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
- [pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill.md)
- [unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)
**closed** — 2 kernels, 2 cards:
- [count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch.md)
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)

### `mfma`
11 distinct kernels · 5 classes · scope `general` · platforms gfx950

**paid** — 7 kernels, 9 cards:
- [32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both.md)
- [bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both.md)
- [collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound.md)
- [gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
- [pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed.md)
- [reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound.md)
- [sub-k-coarsening-regroup-the-same-reduction-order-into-fewer-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/sub-k-coarsening-regroup-the-same-reduction-order-into-fewer-quantized-gemm-gfx950-compute-bound.md)
**closed** — 6 kernels, 6 cards:
- [a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both.md)
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
- [cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
- [occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound.md)
- [the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound.md)
**how to measure** — 1 kernels, 1 cards:
- [scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound.md)

### `fp8-blockscale` (also spelled `fp8-block-scale`)
10 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 10 kernels, 5 cards:
- [derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill.md)
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)
- [invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed.md)
- [pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed.md)
- [pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 4 kernels, 4 cards:
- [a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed.md)
- [instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill.md)
- [nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed.md)
- [where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)
**how to measure** — 1 kernels, 1 cards:
- [route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed.md)

### `graph-replay`
10 distinct kernels · 7 classes · scope `general` · platforms gfx950

**paid** — 4 kernels, 2 cards:
- [arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed.md)
- [collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both.md)
**closed** — 5 kernels, 5 cards:
- [axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound.md)
- [four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
- [four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both](../../kernel_workflow/knowledge/learned/four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both.md)
- [price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both.md)
- [six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill.md)
**how to measure** — 2 kernels, 2 cards:
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)
- [measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both](../../kernel_workflow/knowledge/learned/measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both.md)

### `host-dispatch`
10 distinct kernels · 3 classes · scope `general` · platforms gfx942, gfx950

**paid** — 1 kernels, 1 cards:
- [dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode.md)
**closed** — 2 kernels, 2 cards:
- [size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound.md)
- [the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode.md)
**backend choice** — 8 kernels, 1 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)

### `host-runtime`
10 distinct kernels · 10 classes · scope `general` · platforms gfx950

**paid** — 5 kernels, 6 cards:
- [arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed.md)
- [bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both](../../kernel_workflow/knowledge/learned/bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both.md)
- [cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/cache-the-per-call-host-work-when-the-host-owns-a-large-shar-dense-gemm-gfx950-decode.md)
- [collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-host-launch-path-first-on-a-dispatch-bound-scat-memory-movement-gfx950-both.md)
- [delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
- [dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both](../../kernel_workflow/knowledge/learned/dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both.md)
**closed** — 6 kernels, 6 cards:
- [four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both](../../kernel_workflow/knowledge/learned/four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both.md)
- [graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both](../../kernel_workflow/knowledge/learned/graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both.md)
- [graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch.md)
- [price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both.md)
- [six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill.md)
- [the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both](../../kernel_workflow/knowledge/learned/the-host-lane-pays-once-the-exhaustion-test-is-submit-cost-v-method-gfx950-both.md)
**how to measure** — 1 kernels, 1 cards:
- [route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/route-discarded-sub-noise-knobs-per-shape-instead-of-shippin-moe-grouped-gemm-gfx950-mixed.md)

### `num-stages`
10 distinct kernels · 6 classes · scope `general` · platforms gfx950

**paid** — 6 kernels, 5 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode.md)
- [split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)
- [sub-k-coarsening-regroup-the-same-reduction-order-into-fewer-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/sub-k-coarsening-regroup-the-same-reduction-order-into-fewer-quantized-gemm-gfx950-compute-bound.md)
**closed** — 4 kernels, 4 cards:
- [dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)
- [diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
- [six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill.md)
- [the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound.md)
**how to measure** — 1 kernels, 1 cards:
- [re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound.md)

### `carry-forward`
8 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**backend choice** — 8 kernels, 1 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)

### `cluster`
8 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**backend choice** — 8 kernels, 1 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)

### `cross-workgroup`
8 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 6 kernels, 3 cards:
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
- [one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 2 kernels, 1 cards:
- [axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)

### `extraction-seam`
8 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**backend choice** — 8 kernels, 1 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)

### `flash-decoding`
8 distinct kernels · 2 classes · scope `narrow` · platforms ROCm 7.2, gfx950, gfx950/MI355X, sglang 0.5.17

**paid** — 7 kernels, 2 cards:
- [enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode.md)
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)
**closed** — 1 kernels, 1 cards:
- [a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode.md)

### `measurement`
8 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**closed** — 2 kernels, 2 cards:
- [a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound.md)
- [test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode](../../kernel_workflow/knowledge/learned/test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
**how to measure** — 6 kernels, 6 cards:
- [a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a.md)
- [a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a.md)
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)
- [measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both](../../kernel_workflow/knowledge/learned/measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both.md)
- [scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound.md)
- [scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a.md)

### `moe-routing`
8 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**backend choice** — 8 kernels, 1 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)

### `noise-band`
8 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**backend choice** — 8 kernels, 1 cards:
- [editable-triton-cluster-amdahl](../../e2e_workflow/knowledge/learned/editable-triton-cluster-amdahl.md)

### `raw-hip`
8 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 7 kernels, 5 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode.md)
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
**closed** — 1 kernels, 2 cards:
- [cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
- [dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)

### `size-gating`
8 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 6 kernels, 3 cards:
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill.md)
- [drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
**how to measure** — 2 kernels, 2 cards:
- [ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode.md)
- [scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a.md)

### `sliding-window` (also spelled `sliding window`)
8 distinct kernels · 2 classes · scope `narrow` · platforms ROCm 7.2, gfx950, gfx950/MI355X, sglang 0.5.17

**paid** — 8 kernels, 2 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)

### `stacking`
8 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 5 kernels, 2 cards:
- [collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/collapse-the-dispatch-chain-inside-each-shape-regime-arm-fused-norm-gemm-gfx950-both.md)
- [collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
**how to measure** — 3 kernels, 3 cards:
- [a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a.md)
- [force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a](../../kernel_workflow/knowledge/learned/force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a.md)
- [scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a.md)

### `composable-kernel`
7 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 5 kernels, 4 cards:
- [32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both.md)
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
- [pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 2 kernels, 2 cards:
- [count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch.md)
- [where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)
**how to measure** — 2 kernels, 2 cards:
- [per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a.md)
- [prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a.md)

### `cu-underfill`
7 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 6 kernels, 2 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both](../../kernel_workflow/knowledge/learned/fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
**closed** — 1 kernels, 1 cards:
- [measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode.md)

### `measurement-discipline`
7 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 3 kernels, 1 cards:
- [fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both](../../kernel_workflow/knowledge/learned/fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
**closed** — 2 kernels, 2 cards:
- [test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode](../../kernel_workflow/knowledge/learned/test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
- [the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound.md)
**how to measure** — 2 kernels, 2 cards:
- [a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a.md)
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)

### `num-warps`
7 distinct kernels · 5 classes · scope `general` · platforms gfx950

**paid** — 5 kernels, 5 cards:
- [one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill.md)
- [per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed.md)
- [per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound.md)
- [reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 4 kernels, 4 cards:
- [axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
- [gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound.md)
- [near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound.md)
- [occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound.md)

### `tile-geometry`
7 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 6 kernels, 4 cards:
- [amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both.md)
- [fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both](../../kernel_workflow/knowledge/learned/fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
- [pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 2 kernels, 2 cards:
- [axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both.md)
- [once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound.md)

### `waves-per-eu`
7 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 5 kernels, 4 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode.md)
- [one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill.md)
- [split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)
**closed** — 2 kernels, 2 cards:
- [high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)
- [occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/occupancy-axis-closes-when-the-backend-pins-waves-per-eu-dense-gemm-gfx950-compute-bound.md)

### `accuracy-gate`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)

### `authored-triton`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)

### `block-scale`
6 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 6 kernels, 9 cards:
- [32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both.md)
- [arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/arg-plan-replay-beats-graph-replay-at-low-dispatch-counts-an-quantized-gemm-gfx950-mixed.md)
- [bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/bitcast-the-fp8-flavour-the-matrix-pipe-actually-has-dense-gemm-gfx950-both.md)
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound.md)
- [mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
- [reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/reinterpret-legacy-fp8-bits-to-the-arch-native-fp8-type-to-r-quantized-gemm-gfx950-compute-bound.md)
- [split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch.md)
**closed** — 1 kernels, 2 cards:
- [five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/where-the-headroom-is-not-and-the-two-floors-that-tell-you-s-quantized-gemm-gfx950-compute-bound.md)
**how to measure** — 1 kernels, 1 cards:
- [re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a.md)

### `cudagraph-safe`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)

### `frozen-baseline`
6 distinct kernels · 2 classes · scope `operator` · platforms gfx950

**closed** — 1 kernels, 1 cards:
- [test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode](../../kernel_workflow/knowledge/learned/test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
**how to measure** — 5 kernels, 5 cards:
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)
- [audit-the-baseline-launch-grid-before-believing-a-large-head-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/audit-the-baseline-launch-grid-before-believing-a-large-head-linear-attention-gfx950-prefill.md)
- [gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound.md)
- [prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/prove-the-edit-built-and-prove-the-win-separately-from-the-h-method-gfx950-n-a.md)
- [scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a.md)

### `launch-config`
6 distinct kernels · 5 classes · scope `general` · platforms gfx950

**paid** — 5 kernels, 5 cards:
- [one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill.md)
- [own-the-dispatch-layer-then-race-backends-behind-it-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/own-the-dispatch-layer-then-race-backends-behind-it-dense-gemm-gfx950-compute-bound.md)
- [per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound.md)
- [reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
- [unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)
**closed** — 1 kernels, 1 cards:
- [price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill.md)
**how to measure** — 1 kernels, 1 cards:
- [scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a.md)

### `lazy-import`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)

### `lds-tiling`
6 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 4 kernels, 4 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound.md)
- [pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed.md)
- [share-the-dequantised-weight-tile-across-row-blocks-widen-m--moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/share-the-dequantised-weight-tile-across-row-blocks-widen-m--moe-grouped-gemm-gfx950-prefill.md)
**closed** — 3 kernels, 3 cards:
- [price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill.md)
- [the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound.md)
- [the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-residual-axes-on-a-decoded-fp8-gemm-are-already-closed-quantized-gemm-gfx950-compute-bound.md)

### `non-temporal-loads`
6 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 4 kernels, 4 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode.md)
- [get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both.md)
**closed** — 2 kernels, 1 cards:
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
**how to measure** — 1 kernels, 1 cards:
- [ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode.md)

### `paged-kv`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 5 kernels, 6 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode.md)
- [decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode.md)
- [get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both.md)
- [launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/launch-meta-first-on-latency-floored-paged-decode-and-let-th-attention-decode-gfx950-decode.md)
**closed** — 3 kernels, 4 cards:
- [four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
- [high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)
- [price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both.md)
- [split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-kv-decode-the-two-dispatch-shape-is-welded-budget-the--attention-decode-gfx950-decode.md)

### `rebind-seam`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)

### `sitecustomize-overlay`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)

### `tier-c-author`
6 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 6 kernels, 1 cards:
- [fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm](../../e2e_workflow/knowledge/learned/fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md)

### `xcd-swizzle`
6 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/invert-the-xcd-round-robin-with-a-chunk-interleaved-workgrou-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 3 kernels, 3 cards:
- [axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both.md)
- [positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode.md)
- [the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/the-in-source-ceiling-of-an-mfma-bound-dense-gemm-dense-gemm-gfx950-compute-bound.md)

### `ab-methodology`
5 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**closed** — 1 kernels, 1 cards:
- [test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode](../../kernel_workflow/knowledge/learned/test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
**how to measure** — 4 kernels, 5 cards:
- [a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a.md)
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)
- [force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a](../../kernel_workflow/knowledge/learned/force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a.md)
- [gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/gate-a-tiny-kernel-win-on-a-median-or-a-paired-a-b-method-gfx950-launch-bound.md)
- [measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both](../../kernel_workflow/knowledge/learned/measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both.md)

### `byte-exact-parity`
5 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)

### `decode attention`
5 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, gfx950/MI355X, sglang 0.5.17

**paid** — 5 kernels, 1 cards:
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)

### `empty-workgroups`
5 distinct kernels · 2 classes · scope `operator` · platforms gfx950

**paid** — 4 kernels, 3 cards:
- [collapse-a-redundant-launch-grid-instead-of-guarding-inside--linear-attention-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-a-redundant-launch-grid-instead-of-guarding-inside--linear-attention-gfx950-launch-bound.md)
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)
**closed** — 1 kernels, 1 cards:
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)

### `hybrid SWA`
5 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, gfx950/MI355X, sglang 0.5.17

**paid** — 5 kernels, 1 cards:
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)

### `kv-cache`
5 distinct kernels · 2 classes · scope `operator` · platforms gfx950

**paid** — 4 kernels, 3 cards:
- [choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
- [get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both.md)
**closed** — 1 kernels, 1 cards:
- [test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode](../../kernel_workflow/knowledge/learned/test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
**how to measure** — 1 kernels, 1 cards:
- [ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/ask-whether-the-traffic-is-removable-before-you-tune-how-it--attention-decode-gfx950-decode.md)

### `launcher overlay`
5 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, gfx950/MI355X, sglang 0.5.17

**paid** — 5 kernels, 1 cards:
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)

### `launch-shape`
5 distinct kernels · 2 classes · scope `operator` · platforms gfx950

**paid** — 5 kernels, 3 cards:
- [delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md)

### `lds-staging`
5 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 4 kernels, 3 cards:
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [collapse-the-fp8-dequant-chain-into-one-scaled-convert-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/collapse-the-fp8-dequant-chain-into-one-scaled-convert-quantized-gemm-gfx950-compute-bound.md)
**closed** — 2 kernels, 2 cards:
- [axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode.md)

### `noise-floor`
5 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**closed** — 1 kernels, 1 cards:
- [test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode](../../kernel_workflow/knowledge/learned/test-the-allocator-before-designing-a-kernel-fix-for-a-perio-method-gfx950-decode.md)
**how to measure** — 4 kernels, 4 cards:
- [a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-in-the-graded-case-mix-and-price-a-direction-against-the-method-gfx950-n-a.md)
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)
- [force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a](../../kernel_workflow/knowledge/learned/force-the-rebuild-pair-the-blocks-dump-the-registers-before--method-gfx950-n-a.md)
- [measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both](../../kernel_workflow/knowledge/learned/measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both.md)

### `non-temporal-store`
5 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 5 kernels, 3 cards:
- [cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both.md)
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/streaming-non-temporal-store-for-write-once-output-linear-attention-gfx950-memory-bound.md)

### `num_kv_splits`
5 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, gfx950/MI355X, sglang 0.5.17

**paid** — 5 kernels, 1 cards:
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)

### `partition_size`
5 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950, gfx950/MI355X, sglang 0.5.17

**paid** — 4 kernels, 1 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)
**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `prologue`
5 distinct kernels · 2 classes · scope `operator` · platforms gfx950

**paid** — 4 kernels, 2 cards:
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
**how to measure** — 1 kernels, 1 cards:
- [per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a.md)

### `pythonpath-shadow`
5 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)

### `register-pressure`
5 distinct kernels · 2 classes · scope `operator` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
- [unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)
**closed** — 1 kernels, 1 cards:
- [instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode.md)
**how to measure** — 1 kernels, 1 cards:
- [re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a.md)

### `routing-metadata`
5 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)

### `software-pipelining`
5 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 1 kernels, 1 cards:
- [delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
**closed** — 4 kernels, 3 cards:
- [axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-parity-gated-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
- [four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)

### `triton_backend`
5 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, gfx950/MI355X, sglang 0.5.17

**paid** — 5 kernels, 1 cards:
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)

### `warm-start-replay`
5 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 5 kernels, 1 cards:
- [moe-mxfp4-grouped-authored-gfx950-vllm](../../e2e_workflow/knowledge/learned/moe-mxfp4-grouped-authored-gfx950-vllm.md)

### `window_num_kv_splits`
5 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, gfx950/MI355X, sglang 0.5.17

**paid** — 5 kernels, 1 cards:
- [sglang triton decode-attention split-KV over-splitting (hybrid SWA)](../../e2e_workflow/knowledge/learned/decode-attn-splitkv-triton-gfx950-sglang.md)

### `1stage`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `AITER_CONFIG_FMOE`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `arrival-counter`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 4 kernels, 2 cards:
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
- [one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)

### `atomics`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**closed** — 4 kernels, 3 cards:
- [axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)
- [axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both.md)
- [nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed.md)

### `bit-exact`
4 distinct kernels · 4 classes · scope `general` · platforms gfx950

**paid** — 4 kernels, 6 cards:
- [amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both.md)
- [bit-exact-integer-re-encode-of-the-fp8-fnuz-upcast-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/bit-exact-integer-re-encode-of-the-fp8-fnuz-upcast-quantized-gemm-gfx950-compute-bound.md)
- [cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both.md)
- [divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both.md)
- [reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
- [software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both.md)
**how to measure** — 1 kernels, 1 cards:
- [a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a.md)

### `byte-exact env win`
4 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17

**paid** — 4 kernels, 1 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)

### `coherence`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 4 kernels, 2 cards:
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
- [one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode.md)

### `cpp_itfs JIT`
4 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17

**paid** — 4 kernels, 1 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)

### `cuda-graph`
4 distinct kernels · 3 classes · scope `general` · platforms gfx950

**closed** — 4 kernels, 4 cards:
- [a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/a-hand-written-loop-has-to-out-schedule-not-out-structure-th-dense-gemm-gfx950-both.md)
- [axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
- [positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode.md)
- [the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/the-residual-launch-axis-on-decode-attention-closes-once-hos-attention-decode-gfx950-decode.md)

### `e8m0`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `epilogue`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both.md)
- [pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/pad-the-epilogue-lds-row-stride-off-the-32-bank-period-moe-grouped-gemm-gfx950-mixed.md)
**how to measure** — 1 kernels, 1 cards:
- [per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/per-region-isa-census-before-hot-loop-tuning-locate-on-cpu-p-method-gfx950-n-a.md)

### `fused_moe`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `grid-stride`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/collapse-the-graph-nodes-first-then-shape-gate-a-single-work-quantize-cast-gfx950-launch-bound.md)
- [delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
**closed** — 1 kernels, 1 cards:
- [axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound.md)

### `hardware-counters`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 3 kernels, 1 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
**closed** — 1 kernels, 1 cards:
- [high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)

### `isa-inspection`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 3 cards:
- [choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
- [get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/get-the-nt-bit-onto-kv-loads-by-loading-one-native-128-bit-v-attention-decode-gfx950-both.md)
**closed** — 2 kernels, 1 cards:
- [axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode.md)

### `l2-locality`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-co-resident-sequence-set-to-break-the-kv-addres-attention-decode-gfx950-decode.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 1 kernels, 1 cards:
- [positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode.md)

### `l2-residency`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 2 kernels, 3 cards:
- [cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both.md)
- [choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/choose-the-non-temporal-hint-per-operand-not-per-kernel-attention-decode-gfx950-decode.md)
- [narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 4 kernels, 3 cards:
- [axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode.md)
- [diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)

### `loop-hoisting`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 4 kernels, 2 cards:
- [fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both](../../kernel_workflow/knowledge/learned/fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
- [shorten-the-load-to-dot-chain-before-chasing-bytes-linear-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/shorten-the-load-to-dot-chain-before-chasing-bytes-linear-attention-gfx950-prefill.md)

### `MI355X`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `negative-control`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**how to measure** — 4 kernels, 4 cards:
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)
- [measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both](../../kernel_workflow/knowledge/learned/measure-a-tiny-op-with-the-harness-s-own-protocol-rotated-ag-method-gfx950-both.md)
- [per-rep-geomean-plus-a-two-sided-negative-control-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/per-rep-geomean-plus-a-two-sided-negative-control-method-gfx950-n-a.md)
- [scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/scope-a-closure-a-gate-and-a-stack-by-case-regime-method-gfx950-n-a.md)

### `online-softmax`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 4 kernels, 3 cards:
- [collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
- [pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)

### `oracle-parity`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 1 cards:
- [fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both](../../kernel_workflow/knowledge/learned/fill-the-cus-with-a-hidden-dim-block-axis-then-hoist-the-k-l-composable-gfx950-both.md)
**closed** — 1 kernels, 2 cards:
- [a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode.md)
- [reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode.md)

### `paired-ab-rig`
4 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill.md)
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
**closed** — 1 kernels, 2 cards:
- [host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill.md)
- [instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/instruction-cuts-on-a-co-resident-pipe-do-not-convert-moe-grouped-gemm-gfx950-prefill.md)
**how to measure** — 1 kernels, 1 cards:
- [a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-fixed-order-a-a-control-measures-order-bias-and-then-hides-method-gfx950-n-a.md)

### `per_1x32`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `QKV_VERSION`
4 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17

**paid** — 4 kernels, 1 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)

### `quark`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `ragged kv_indptr`
4 distinct kernels · 1 classes · scope `operator` · platforms ROCm 7.2, aiter (cpp_itfs JIT), gfx950/MI355X, sglang 0.5.17

**paid** — 4 kernels, 1 cards:
- [aiter paged_attention_ragged — QKV_VERSION=EXPERIMENTAL is a free, byte-exact op-level env win](../../e2e_workflow/knowledge/learned/paged-attn-ragged-qkvversion-gfx950-sglang.md)

### `software-prefetch`
4 distinct kernels · 3 classes · scope `general` · platforms gfx950

**closed** — 4 kernels, 4 cards:
- [diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed.md)
- [four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-axes-that-stayed-closed-on-a-latency-floored-paged-deco-attention-decode-gfx950-decode.md)
- [only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode.md)
- [six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill.md)

### `tile-size`
4 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 4 kernels, 3 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch](../../kernel_workflow/knowledge/learned/split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch.md)
- [unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/unclamp-the-kv-tile-from-the-page-size-then-de-rate-it-by-he-attention-decode-gfx950-decode.md)
**closed** — 1 kernels, 1 cards:
- [five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/five-closed-axes-above-an-ilp-bound-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)

### `tuned_fmoe`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `unroll`
4 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 4 kernels, 3 cards:
- [collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)

### `xbf16`
4 distinct kernels · 1 classes · scope `operator` · platforms gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4

**unclassified** — 4 kernels, 1 cards:
- [mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier](../../e2e_workflow/knowledge/learned/moe-mxfp4-fused-aiter-fmoe-gfx950-sglang.md)

### `xcd-remap`
4 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/derive-the-tile-then-renegotiate-the-scale-contract-moe-grouped-gemm-gfx950-prefill.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 1 kernels, 1 cards:
- [host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill.md)
**how to measure** — 1 kernels, 1 cards:
- [a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a.md)

### `AITER_CONFIG_GEMM_BF16`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `bank-conflict`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 2 kernels, 2 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
- [mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
**closed** — 1 kernels, 1 cards:
- [high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both](../../kernel_workflow/knowledge/learned/high-lds-wait-counters-next-to-a-high-roofline-fraction-can--attention-decode-gfx950-both.md)

### `block-size`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**closed** — 3 kernels, 2 cards:
- [axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-a-quant-cast-graph-sits-at-two-no-quantize-cast-gfx950-launch-bound.md)
- [gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound](../../kernel_workflow/knowledge/learned/gpu-side-knobs-are-a-closed-axis-once-submit-dominates-memory-movement-gfx950-launch-bound.md)

### `block_size 1`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `codegen`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 1 kernels, 1 cards:
- [own-the-dispatch-layer-then-race-backends-behind-it-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/own-the-dispatch-layer-then-race-backends-behind-it-dense-gemm-gfx950-compute-bound.md)
**closed** — 3 kernels, 2 cards:
- [axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/axes-that-close-once-decode-attention-sits-on-its-read-roof-attention-decode-gfx950-decode.md)
- [once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/once-it-routes-to-tuned-vendor-assembly-out-generating-it-is-dense-gemm-gfx950-compute-bound.md)

### `cold-cache`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `colon-merge`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `constexpr-promotion`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/derive-the-split-kv-decode-launch-shape-from-a-constant-byte-attention-decode-gfx950-decode.md)
- [dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/dispatch-collapse-first-then-per-regime-specialisation-on-la-attention-decode-gfx950-decode.md)

### `dead-list`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 1 kernels, 1 cards:
- [cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill.md)
**how to measure** — 2 kernels, 2 cards:
- [a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-closure-is-conditional-on-the-body-that-measured-it-method-gfx950-n-a.md)
- [re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/re-price-the-dead-list-when-the-operating-point-moves-method-gfx950-compute-bound.md)

### `decode-roofline`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `dequant`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 5 cards:
- [amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/amortize-int4-dequant-across-m-blocks-instead-of-shrinking-i-moe-grouped-gemm-gfx950-both.md)
- [bit-exact-integer-re-encode-of-the-fp8-fnuz-upcast-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/bit-exact-integer-re-encode-of-the-fp8-fnuz-upcast-quantized-gemm-gfx950-compute-bound.md)
- [collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound.md)
- [collapse-the-fp8-dequant-chain-into-one-scaled-convert-quantized-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/collapse-the-fp8-dequant-chain-into-one-scaled-convert-quantized-gemm-gfx950-compute-bound.md)
- [per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 1 kernels, 2 cards:
- [axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both.md)
- [dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/dequant-op-count-is-off-the-critical-path-once-the-gemm-is-o-moe-grouped-gemm-gfx950-mixed.md)

### `double-buffering`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**closed** — 3 kernels, 2 cards:
- [a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/a-contract-fixed-short-k-loop-closes-the-inner-loop-axes-moe-grouped-gemm-gfx950-mixed.md)
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)

### `fp8-kv`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/decode-attention-pay-the-host-tax-first-then-halve-kv-bytes--attention-decode-gfx950-decode.md)
- [drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/drop-the-non-temporal-cache-hint-on-once-read-kv-streams-attention-decode-gfx950-decode.md)
**closed** — 1 kernels, 1 cards:
- [reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/reproduce-the-golden-s-own-rounding-before-costing-a-kv-reas-attention-decode-gfx950-decode.md)

### `gemm_a16w16`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `gemm_tuner`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `get_padded_m`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `gradlib`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `graph-capture`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**closed** — 3 kernels, 3 cards:
- [four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)
- [graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both](../../kernel_workflow/knowledge/learned/graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both.md)
- [measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode.md)

### `host-wrapper`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/decode-attention-the-python-ctypes-prologue-is-the-first-thr-attention-decode-gfx950-decode.md)
- [enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode.md)

### `isa-diff`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 1 kernels, 1 cards:
- [buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/buy-prefetch-depth-with-a-global-to-lds-dma-on-bandwidth-bou-attention-decode-gfx950-decode.md)
**closed** — 2 kernels, 2 cards:
- [geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/geometry-occupancy-and-load-width-are-a-spent-axis-here-attention-decode-gfx950-decode.md)
- [only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode.md)

### `kv-fp8`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `lds-padding`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 2 kernels, 2 cards:
- [32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both.md)
- [mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/mfma-32-plus-a-lds-pad-on-a-frozen-gridwise-ck-block-scale-m-moe-grouped-gemm-gfx950-both.md)
**closed** — 1 kernels, 1 cards:
- [where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/where-a-native-mfma-block-scaled-moe-gemm-has-no-headroom-le-moe-grouped-gemm-gfx950-mixed.md)

### `ll4mi`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `long-context`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/enable-the-source-s-own-dormant-split-kv-path-before-authori-attention-decode-gfx950-decode.md)
- [fp8-kv-storage-with-bf16-in-register-dequant-on-scatter-boun-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/fp8-kv-storage-with-bf16-in-register-dequant-on-scatter-boun-attention-gfx950-memory-bound.md)

### `m-bucket`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 3 kernels, 3 cards:
- [cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/cache-policy-is-a-per-buffer-per-bucket-decision-on-a-bf16-f-moe-grouped-gemm-gfx950-both.md)
- [delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-satellite-dispatches-once-both-moe-gemms-sit-at-t-moe-grouped-gemm-gfx950-both.md)
- [per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/per-m-bucket-launch-config-on-an-int4-weight-only-grouped-ge-moe-grouped-gemm-gfx950-compute-bound.md)

### `mfma16`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `mfma-nonkdim`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**paid** — 1 kernels, 1 cards:
- [per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed](../../kernel_workflow/knowledge/learned/per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed.md)
**closed** — 2 kernels, 2 cards:
- [axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
- [axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-bf16-fused-moe-with-a-decode-we-moe-grouped-gemm-gfx950-both.md)

### `oracle`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**closed** — 1 kernels, 1 cards:
- [price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/price-the-residual-before-funding-fusion-or-geometry-work-at-attention-decode-gfx950-decode.md)
**how to measure** — 2 kernels, 2 cards:
- [a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/a-b-protocol-and-oracle-confounds-on-a-power-capped-gpu-method-gfx950-n-a.md)
- [the-baseline-seam-can-be-aliased-to-the-candidate-by-a-share-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/the-baseline-seam-can-be-aliased-to-the-candidate-by-a-share-method-gfx950-n-a.md)

### `paged_attention_ragged`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `partition`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 2 kernels, 1 cards:
- [collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/collapse-the-partition-grid-instead-of-optimizing-the-round--attention-decode-gfx950-decode.md)
**how to measure** — 1 kernels, 1 cards:
- [scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/scale-percent-of-peak-to-the-cus-the-box-actually-exposes-method-gfx950-compute-bound.md)

### `persistent-kernel`
3 distinct kernels · 3 classes · scope `general` · platforms gfx950

**closed** — 3 kernels, 3 cards:
- [axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/axes-that-stay-closed-once-the-store-pipe-is-saturated-linear-attention-gfx950-memory-bound.md)
- [host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/host-dispatch-and-backend-swap-closed-on-a-saturated-grouped-moe-grouped-gemm-gfx950-prefill.md)
- [instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode.md)

### `profiler-error`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 2 kernels, 1 cards:
- [fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/fuse-the-quant-passes-behind-a-tag-slot-grid-barrier-quantize-cast-gfx950-both.md)
**how to measure** — 1 kernels, 1 cards:
- [hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode](../../kernel_workflow/knowledge/learned/hand-count-the-bytes-and-build-a-read-only-twin-before-staff-method-gfx950-decode.md)

### `ragged`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx950

**backend choice** — 3 kernels, 1 cards:
- [sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)](../../e2e_workflow/knowledge/learned/paged-attn-ragged-aiter-gfx950-sglang.md)

### `shipped-coverage`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `tiling`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 3 kernels, 2 cards:
- [reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/reinterpret-a-frozen-launch-through-an-exported-wrapper-obje-quantize-cast-gfx950-memory-bound.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 1 kernels, 1 cards:
- [near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound](../../kernel_workflow/knowledge/learned/near-the-practical-hbm-ceiling-the-bandwidth-knobs-are-a-clo-quantize-cast-gfx950-memory-bound.md)

### `topk` (also spelled `top-k`)
3 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 3 kernels, 3 cards:
- [bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both](../../kernel_workflow/knowledge/learned/bypass-the-jit-launcher-for-a-dispatch-bound-triton-op-moe-router-topk-gfx950-both.md)
- [dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both](../../kernel_workflow/knowledge/learned/dispatch-floored-router-select-spend-the-budget-on-the-host--topk-router-gfx950-both.md)
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 1 kernels, 1 cards:
- [the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both](../../kernel_workflow/knowledge/learned/the-device-lane-on-a-small-router-top-k-is-close-to-closed-moe-router-topk-gfx950-both.md)

### `tuned_gemm`
3 distinct kernels · 1 classes · scope `operator` · platforms gfx942, gfx950

**paid** — 3 kernels, 1 cards:
- [dense-gemm-bf16-gfx950-sglang](../../e2e_workflow/knowledge/learned/dense-gemm-bf16-gfx950-sglang.md)

### `valu-bound`
3 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 3 kernels, 4 cards:
- [cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill](../../kernel_workflow/knowledge/learned/cut-valu-on-the-prefill-arm-with-native-casts-and-packed-dot-fused-norm-gemm-gfx950-prefill.md)
- [divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/divide-by-the-group-scale-is-a-correctly-rounded-reciprocal--quantize-cast-gfx950-both.md)
- [pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/pack-heads-per-workgroup-then-strip-the-inner-loop-attention-gfx950-prefill.md)
- [software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/software-emulated-fp8-cast-find-it-by-differential-recompile-quantize-cast-gfx950-both.md)

### `vgpr`
3 distinct kernels · 3 classes · scope `general` · platforms gfx950

**paid** — 1 kernels, 1 cards:
- [delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both](../../kernel_workflow/knowledge/learned/delete-the-in-kernel-bounds-guard-from-the-host-before-decla-quantize-cast-gfx950-both.md)
**closed** — 2 kernels, 2 cards:
- [cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a](../../kernel_workflow/knowledge/learned/cdna4-sums-archvgpr-and-agpr-for-occupancy-method-gfx950-n-a.md)
- [instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/instructions-and-registers-are-not-currency-at-near-total-me-attention-decode-gfx950-decode.md)

### `vgpr-pressure`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**paid** — 2 kernels, 1 cards:
- [retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/retile-to-one-program-per-query-then-delete-every-per-trip-r-attention-gfx950-prefill.md)
**closed** — 2 kernels, 2 cards:
- [axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound](../../kernel_workflow/knowledge/learned/axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound.md)
- [six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill](../../kernel_workflow/knowledge/learned/six-axes-that-stayed-closed-on-a-graph-replay-timed-sparse-p-attention-gfx950-prefill.md)

### `wave-quantization`
3 distinct kernels · 2 classes · scope `narrow` · platforms gfx950

**closed** — 3 kernels, 2 cards:
- [axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both](../../kernel_workflow/knowledge/learned/axes-that-stayed-closed-on-a-roof-bound-fused-norm-gemm-path-fused-norm-gemm-gfx950-both.md)
- [four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode](../../kernel_workflow/knowledge/learned/four-host-and-compute-directions-that-a-latency-floored-deco-attention-decode-gfx950-decode.md)

## Sources
- Generated from the learned card trees: `kernel_workflow/knowledge/learned/` and `e2e_workflow/knowledge/learned/` (schema and evidence rules: their `README.md`).
- Card evidence is a frozen-baseline isolated A/B plus oracle parity (kernel tree) or the e2e Director's A/B (e2e tree). No number is copied here; each row links the cards.
- Axis vocabulary is filtered against [`taxonomy.md`](taxonomy.md) and the card header fields, so scope tags (`gfx950`, `decode`, `fp8`) do not appear as axes.
