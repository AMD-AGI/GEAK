# Kernel case studies

Per-kernel evidence, grouped by operator domain. Each links the patterns it used.

## attention
- [MLA prefill flash-attention forward (_attn_fwd, head_dim_qk=192)](/cases/mla-prefill.md) — **1.214x geomean**. MLA prefill flash-attention forward kernel sped up 1.214x geomean by retuning the launch config (smaller BLOCK_M, 2-stage pipelining, fewer warps) plus an empty output buffer.
- [kernel_unified_attention_2d (Triton decode paged attention)](/cases/unified-attention-2d.md) — **1.58x honest (campaign20 single-pass graph) / ~1.00x (KernelForge body-only run)**. Triton decode paged-attention whose only real win is a host-side single-pass CUDA-graph replay (1.58x honest); all in-kernel body edits were <1% wall-clock noise because the kernel is at a hard launch/latency floor.
- [paged_attention (vLLM decode, single-pass flash)](/cases/paged-attention-vllm-singlepass.md) — **1.18x geomean**. vLLM paged-attention decode rewritten as a single-pass flash kernel with online softmax that writes final_out directly, eliminating the second reduce-kernel launch for 1.18x geomean.
- [paged_attention_decode (MiniMax-M2.5 decode paged attention)](/cases/paged-attention-decode.md) — **4.39x geomean (campaign20); 1.51x (spare measurement)**. MiniMax decode paged-attention sped up by host-side routing from the CK/HIP paged_attention_v1 op to the prebuilt bf16 ASM decode kernel, up to 4.39x geomean.
- [paged_attention_ragged (AITER vLLM-style paged decode attention)](/cases/paged-attention-ragged.md) — **~1.05-1.10x on dominant captured cases; flat on perf cases**. AITER paged decode attention on gfx942 won ~1.05-1.10x by flipping K loads non-temporal (NT_KV_LOAD) in the GOLDEN kernel, exploiting block_size=1 single-touch KV streaming.

## moe
- [_fwd_grouped_kernel_stage1 (grouped GEMM stage-1)](/cases/fwd-grouped-stage1.md) — **1.18x geomean**. Launch-bound Triton grouped GEMM stage-1 sped up 1.18x geomean by adding a host-side HIP-graph capture/replay launcher (compute backend unchanged).
- [_topk_forward (Triton MoE router top-k)](/cases/topk-forward.md) — **1.90x (graph-replay, overfit-flagged) / ~1.02x (transferable)**. A launch-overhead-bound Triton MoE router top-k whose only transferable win is host-side do_not_specialize on pointer args (~1.02x); a graph-replay run measured 1.90x but is benchmark-overfit.
- [aiter CK ck_moe_stage1 block-scale grouped GEMM (gate/up + SiLU)](/cases/moe-stage1.md) — **1.08-1.11x geomean (1.077x integrable host-side patch; 1.114x in-kernel .cuh override)**. aiter CK block-scale MoE stage-1 GEMM sped up by host-side occupancy/padding levers (V3->V1 pipeline + block_m=16 for sparse routing), bit-exact, 1.08-1.11x geomean on gfx942.
- [chunk_scaled_dot_kkt_fwd_kernel (linear-attention KKT forward)](/cases/chunk-scaled-dot-kkt.md) — **~1.5x honest geomean (per-shape 1.16 / 1.12 / 1.55x); 17.39x RETRACTED**. Triton linear-attention KKT-forward kernel sped up ~1.5x by hoisting beta out of the dot (fp32 post-scale), pinning a single MFMA config to dodge tune-on-c2 mis-picks, and HIP-graph replay for the small shape.
- [ck_moe_stage2_gemm (MoE stage-2 down-proj CK block-scale GEMM)](/cases/moe-stage2.md) — **1.31x geomean (per-shape 1.25-1.33x)**. An occupancy-bound MoE down-proj CK block-scale GEMM sped up 1.31x geomean by host-side block_m routing away from the high-LDS V3 pipeline toward V1 (2 blocks/CU) instances, with no C++ rebuild and bit-exact output.
- [fused_moe_int4_w4a16 (int4 W4A16 fused-MoE GEMM, vLLM Triton)](/cases/fused-moe-int4-w4a16.md) — **5.19x geomean**. vLLM int4 W4A16 fused-MoE Triton GEMM made 5.19x faster by loading each packed int4 byte once and unpacking both nibbles in-register, plus scale/zp group-dedup broadcast.
- [fused_moe_kernel (vLLM fp8 w8a8 block-scale MoE GEMM)](/cases/fused-moe-fp8-blockscale.md) — **1.36x geomean**. Skinny-M fp8 block-scale MoE GEMM on MI300X sped up 1.36x geomean by hoisting the always-true K-mask out of the load loop plus L2/XCD pid swizzling.
- [fused_moe_kernel_gptq_awq (int4 W4A16 AWQ/GPTQ MoE GEMM)](/cases/fused-moe-gptq-awq.md) — **1.63x geomean (campaign20 launch-config); ~1.05-1.21x (kernel-body passes)**. An int4 W4A16 fused-MoE GEMM whose biggest win came from launch-config tuning (num_warps=1/nonkdim=16/kpack=2) for 1.63x; kernel-body hoisting of per-iter int divides added ~1.05-1.21x in a separate pass.
- [moe_gemm_fp8_blockscale (MiniMax-M2.5 fp8 block-scale fused MoE)](/cases/moe-gemm-fp8-blockscale.md) — **1.19x geomean (per-shape 1.04-1.40x)**. A fused fp8 block-scale MoE GEMM sped up 1.19x geomean by swapping the 1-stage ASM dispatch for aiter's 2-stage CK path with per-regime block_m and NT-load off at token=1024.

## gemm
- [_gemm_a16_w16_kernel (bf16 GEMM)](/cases/gemm-a16-w16.md) — **1.5x geomean (per-shape 1.38 / 1.58 / 1.59x)**. A bf16 Triton GEMM on MI300X sped up ~1.5x by in-kernel super-grouping of pid order (GROUP_M=8 coupled with remap_xcd) for L2 reuse, plus dropping a forced .cg on the B load.
- [gemm_a8w8_blockscale (MiniMax-M2.5 fp8 attn-proj GEMM)](/cases/gemm-a8w8-blockscale.md) — **1.82x (campaign20 graph); 1.37x (KernelForge fast-kernel dispatch); 1.23x (CK-vs-ASM dispatch)**. fp8 a8w8 block-scaled attn-projection GEMM on MI300X sped up via host-side HIP/CUDA-graph replay plus per-shape kernel-family dispatch (CK vs ASM), up to 1.82x, bit-exact.

## kv-cache
- [write_req_to_token_pool_triton (SGLang KV pool index write)](/cases/write-req-to-token-pool.md) — **2.05x geomean**. A launch-overhead-bound tiny Triton op whose 2.05x geomean win came entirely from host-side dispatch-path cuts (do_not_specialize + thin cached launcher + skip launch_metadata), not GPU-internal levers.

## quant
- [_per_token_group_quant_fp8 (per-token-group fp8 quantization)](/cases/per-token-group-quant-fp8.md) — **2.90x geomean**. Memory-bound per-token-group fp8 quantization kernel rewritten in-place (Triton kernel body), reaching 2.90x geomean on gfx942.
