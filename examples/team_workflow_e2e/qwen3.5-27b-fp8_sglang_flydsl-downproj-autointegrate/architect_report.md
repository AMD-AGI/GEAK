# Architect Report — Qwen3.5-27B-FP8 e2e Optimization (sglang / MI300X gfx942)

## Headline
- **Baseline:** 931.593 tok/s (TP=1, single GPU, mem-fraction 0.85, default flags)
- **Final:** 1559.934 tok/s
- **Speedup: 1.674x (+67.4% e2e output throughput)**
- TPOT: 64.056 ms -> 37.187 ms median.

## Accepted stack (in order)
1. **Config — `--attention-backend triton`** (cfg0): +2.24% e2e (931.6 -> 952.5 tok/s), non-overlapping A/B, parity pass. Moves the 16 full-attn prefill layers + mamba/linear-attn onto the editable Triton path. Accepted and carried as the reference config for all subsequent A/B.
2. **Head kernel — FlyDSL fused fp8 a8w8 blockscale Triton core (down-proj N=5120,K=17408)** (h1): isolated 2.432x; e2e +60.09% in a same-session 2-launch A/B on the `--attention-backend triton` reference (ref_med 953.139 -> cand_med 1525.857), non-overlapping, parity pass, engaged inside sglang's captured decode CUDA graph. The seam (`sitecustomize` lazy meta-path bind, no nested graph capture) rebinds the single live blockscale GEMM call site, so the fused fp8 core serves **all** blockscale GEMM shapes (down + up/gate + qkv/o) on the decode path — that whole-path win is what produces the +60%.

## Rejected / dropped (kept for the record)
- **cfg1 `--kv-cache-dtype fp8_e4m3`** — skipped: workload is prefill-compute-bound at conc=64 (KV pool ~13% used, queue=0); zero headroom + lossy. Correctly not stacked.
- **h0 up/gate GEMM (N=34816,K=5120), 57.2% gpu** — Triton per-(N,K) config overlay, isolated 1.10x, but **e2e -0.257%** (regression). The +1.10x lands only on prefill tiles (BM=256); the throughput-critical decode regime stays on generic BM=128, so Amdahl gives no e2e conversion. Do-no-harm: not stacked.
- **h2 qkv/o GEMM (N=5120,K=6144), 6.0% gpu** — isolated 1.4296x, but the entire win is a wrapper-level nested `torch.cuda.CUDAGraph` capture/replay that collapses the decode launch floor; that nested capture is **illegal inside sglang's decode graph capture** (it crashed the identical pattern). The capture-safe bare fp8 core is already deployed via the accepted h1 seam for this shape -> no remaining delta (e2e 0%). Rejected: cuda_graph_capture_unsafe.

## Remaining headroom (from the post-integration profile, round_head)
After the FlyDSL core engages, `_fused_blockscale_kernel` is 78.9% of GPU time — the fp8 GEMM is still the head, now on the FlyDSL core. Further compute gains would need a faster fp8 a8w8 blockscale core (the 21% MFMA-peak ceiling on these tiles is intrinsic per MEMORY), not config tuning. The editable tail (gated-delta/FLA cluster: chunk_gated_delta_rule_fwd_h 3.0%, chunk_fwd_kernel_o 2.0%, recompute_w_u 1.6%, causal_conv1d 1.2%) and host-overhead fusion (act_and_mul 2.1%, dynamic_per_group_scaled_quant 2.0%) are each sub-noise solo but could stack-compound. No single remaining lever clears the e2e noise band alone.

## Caveat
Box drift was severe in the multi-launch profile rounds (round_head spread 63%, round_config 46%). Trust ONLY same-session tight A/B deltas (which all accepted/rejected decisions used), never cross-round medians.

Full timeline: `final_report.md` (same directory).
