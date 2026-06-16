---
kind: landscape
updated: 2026-06-09
hardware: [MI300X (gfx942, CDNA3), MI325X (gfx942, CDNA3), MI350X/MI355X (gfx950, CDNA4)]
scope: per-operator current SOTA kernel + best backend on AMD Instinct, 2025-2026 evidence
sourcing: numbers tagged vendor/third-party + hardware + version; see ## Sources
---

# AMD Instinct SOTA kernels per operator (2025-2026)

## TL;DR

- **GEMM is essentially solved on CDNA4 at the HIP/C++ and Triton/Gluon level.** AMD's own CDNA4 series shows a HIP/C++ 8-wave ping-pong FP8 GEMM hitting **3204 TFLOPS** (M=N=K=8192) on MI355X — *surpassing* hipBLASLt (3130) without assembly. Gluon goes further: FP16 **1489 TFLOPS @ 98.75% MFMA eff**, BF8 **3257 TFLOPS @ 99.72%**, MXFP4 **5255 TFLOPS @ 92.41%**. The 8-wave ping-pong and 4-wave interleave scheduling patterns both originate from **HipKittens** (Stanford Hazy Research, arXiv 2511.08083, Nov 2025).
- **HipKittens (HK) is the new academic SOTA tile-DSL for AMD**, matching or beating AITER hand-assembly on attention forward (1.0-2.1x vs AITER) and GEMM, and winning 1.8-2.5x on GQA/d=64/memory-bound kernels where no tuned assembly exists. Key finding: **wave specialization underperforms on CDNA3/CDNA4** (only ~80% of peak BF16 GEMM) because AMD's static register allocation starves producers — so AMD needs ping-pong/interleave, not the NVIDIA producer-consumer model.
- **AITER remains the default production backend** for attention (MLA/MHA), MoE, norm/quant in vLLM and SGLang. AITER MHA (`pa_fwd_asm`) and MLA (`mla_decode_fwd`) assembly kernels are the prefill/decode workhorses: ROCM_AITER_FA gives **2.7-4.4x** TPS vs legacy ROCM_ATTN; AITER MLA gives **1.2-1.5x** vs Triton MLA across MI300X/MI325X/MI355X.
- **MoE moved to MXFP4 + comm/compute fusion.** AITER's new **FlyDSL** (MLIR-backed Python DSL) is now a competitive A4W4 FusedMoE path (1.6x latency cut @ concurrency 512), with **MoRI** providing in-kernel cross-GPU dispatch/combine. MoRI quantized all-to-all (FP4 dispatch + FP8 combine) gives **2.56x round-trip bandwidth reduction**; MI355X+MoRI SGLang beats B200 SGLang by **1.25x tok/s/GPU** at iso-latency.
- **Collectives: 3-way adaptive dispatch is SOTA.** vLLM/SGLang auto-select among Custom AllReduce (small <~2MB), **QuickReduce** (mid/large, up to 3x vs RCCL with INT4/FP4 inline compression, MI300-only), and RCCL (largest). QuickReduce primarily lifts TTFT, not TPOT.
- **MLPerf v6.0 (the latest round): MI355X reached parity-to-win vs NVIDIA B300/B200** — Llama2-70B at 92/93/**104%** (offline/server/interactive) of B300; gpt-oss-120b up to **115%** server. Llama2-70B improved **4.4-4.8x** over v5.1, driven by FP4. NVIDIA still leads raw per-GPU throughput on reasoning (GB300 DeepSeek-R1).

---

## 1. GEMM / scaled-FP8 / MXFP4 GEMM

**Best on MI355X (CDNA4, gfx950):**
- **Gluon (Triton-based) — near-peak ceilings (AMD-measured, ROCm 7.0, gfx950-tutorial Triton):** FP16 (a16w16) **1489 TFLOPS, 98.75% MFMA eff** (4096×4096×8192); BF8 (a8w8) **3257 TFLOPS, 99.72%** (4096×4096×16384); MXFP4 (a4w4) **5255 TFLOPS, 92.41%** (4096×4096×32768). Naive FP16 baseline was 520 TFLOPS @ 25% → ~3x to peak. These are the practical low-precision ceilings. MXFP4 uses CDNA4 native scaled-MFMA `v_mfma_scale_f32_16x16x128_f8f6f4`.
- **HIP/C++ 8-wave ping-pong FP8 (AMD-measured, ROCm 7.1.0, MI355X):** M=N=K=4096 → **2680 TFLOPS** (~97% of hipBLASLt 2750); M=N=K=8192 → **3204 TFLOPS** (*beats* hipBLASLt 3130). Built without assembly. 4-wave interleave (one wave/SIMD, full 512-VGPR budget, 128×128 tile) is the robustness/perf successor — no #pragma unroll tuning, consistent across ROCm releases.
- **HipKittens (Hazy, Nov 2025, MI355X):** BF16 GEMM 256×256 no-wave-spec **1610 TFLOPS** (M=N=K=8192) vs TK-on-B200 1538 / CUTLASS-B200 1570; FP8 8-wave **3222 TFLOPS** (48 LoC), 4-wave **3327 TFLOPS** (183 LoC). XCD/L2-aware cache schedules add ~3-19% at large M (e.g. 900→1068 TFLOPS @ 14592). HK GEMM beats Triton 1.3-3.0x.
- **hipBLASLt** is still the no-tune default and the bar everyone targets (~2750 @ 4096, ~3130 @ 8192 FP8 MI355X). QuickTune/TensileLite offline tuning (AMD Quark team) gives one-click model-shape tuning.

**Best on MI300X (CDNA3, gfx942):** hipBLASLt + AITER block-scale GEMM (up to 2x boost on DeepSeek shapes) remain production default; HipKittens validated on gfx942 but headline numbers are CDNA4. Real-world sustained util on MI300X is ~45% of peak (third-party), vs ~93% for H100/B200 — software/clock-scaling gap, narrowing.

URLs: Gluon tutorial, cdna4-gemm-kernels, 4wave-fp8gemm, HipKittens arXiv (see Sources).

## 2. Attention — prefill / decode / MLA

**Backend ranking (vLLM ROCm, AMD-measured, Feb 2026):**
- **MHA:** ROCM_AITER_FA (recommended/auto) > ROCM_AITER_UNIFIED_ATTN (~within 5%) > TRITON_ATTN > ROCM_ATTN. AITER MHA auto-dispatches to **CK or assembly** kernels. ROCM_AITER_FA vs ROCM_ATTN TPS: MI300X **3.82x/2.65x**, MI325X **4.36x/3.12x**, MI355X **3.61x/2.88x** (64/128 req); TPOT 2.8-4.6x faster.
- **MLA:** ROCM_AITER_MLA ≈ ROCM_AITER_TRITON_MLA (share assembly decode kernel) > TRITON_MLA. AITER MLA vs Triton MLA TPS: MI300X 1.33x/1.24x, MI325X 1.41x/1.24x, MI355X **1.52x/1.35x**; TPOT 1.2-1.6x. On gfx942 the Triton-MLA variant edges +2-3% TPS; on gfx950 ROCM_AITER_MLA wins (uses AITER asm MHA prefill, best TTFT).
- **Prefill (compute-bound):** AITER MHA `flash_attn_varlen_func` (CDNA matrix cores). **Decode (memory-bound):** AITER asm `pa_fwd_asm` / `mla_decode_fwd`.
- **Original MLA layer wins (AMD, AITER v0.1.4 / ROCm 6.4):** matrix-absorption + AITER → up to **2x** faster inference; DeepSeek-R1 full-stack AITER kernels: MLA decode **up to 17x**, MHA prefill **up to 14x** vs baseline.

**HipKittens attention (Hazy, Nov 2025, MI355X):** forward beats AITER **1.0-2.1x**, CK 1.0-1.4x, Triton 1.2-4.5x, PyTorch SDPA 1.3-4.5x — in ~500 LoC, beating hand-asm AITER on average. Backward: GQA non-causal 8-wave 1.8x / 4-wave **2.3x** over baselines (AITER GQA bwd only hit 30% of SoTA, PyTorch SDPA 24%). MHA non-causal bwd seq=4096: HK pinned-reg **1024 TFLOPS** ≈ AITER 1018; seq=8192 AITER 1169 > HK 1091.
- **TileLang** reached FlashMLA parity with AITER hand-asm on MI300X (third-party milestone). **Mojo** MHA fwd only ~50% peak (430 TFLOPS) on MI355X with bank conflicts.

**Best on MI300X:** AITER asm MLA/MHA (production). **Best on MI355X:** AITER asm (production) / HipKittens (forward, academic SOTA).

## 3. MoE / fused grouped-GEMM

- **AITER FusedMoE** is the production grouped-GEMM MoE backend (CK / asm-hipModule / Triton variants). On DeepSeek-V3 MI300X: block-scale fused MoE **up to 3x** boost. MoE align&sort redesign gave **10x** on the sort step (AMD).
- **FlyDSL (AITER, MLIR-backed Python DSL)** is the new competitive **A4W4 / MXFP4 MoE** path on MI355X, replacing inflexible CK templates: **1.6x latency reduction** @ concurrency 512. Optional, CK fallback. Kimi-K2.5 on MI300X with FlyDSL + SGLang/AITER: **-65% TTFT, -69% TPOT, +162% throughput**, no accuracy loss.
- **MoRI integration** fuses cross-GPU EP dispatch/combine into the FusedMoE kernel (conceptual analog of DeepGEMM Mega-MoE, which is NVIDIA-only). MoRI quantized all-to-all (FP4 dispatch + FP8 combine): **2.56x** round-trip bandwidth reduction (28672→11200 B/token). MoRI-EP combine kernel (EP8, BF16, 4096 tok, hidden 7168): fp8_blockwise **~736 µs** vs BF16 ref ~907 µs; adaptive InterNodeV1LL gives 1.52x dispatch / 1.82x combine at ≤256 tok/rank.
- MXFP4 GEMMs ≈ **62%** of Llama2-70B e2e cost; MI355X 2.7x over MI325X on Llama2-70B offline (FP4 vs FP8).

**Best on MI300X:** AITER FusedMoE (CK/asm) + FlyDSL for MXFP4. **Best on MI355X:** AITER FlyDSL A4W4 + MoRI EP.

## 4. Norm / Act / RoPE / Quant fused kernels

- **AITER** supplies the fused primitives (CK/ASM/Triton/Gluon/HIP variants): fused-add RMSNorm, RMSNorm+FP8-quant, SiLU+FP8-quant, RoPE+KV-cache, QKNorm+RoPE+KV-set, AllReduce+residual+RMSNorm+quant. `VLLM_ROCM_USE_AITER=1` required to enable GEMM/RMSNorm/MoE AITER even when attention backend is overridden.
- **vLLM torch.compile/Inductor passes** orchestrate fusion: RMSNorm+quant and SiLU+quant passes (`rocm_aiter_fusion`), ActivationFusionPass **+8% throughput**; ROCm-only RoPE+KV-cache fusion (O1+, auto) and AITER-Triton-GEMM padding fusion (GPT-OSS hidden=2880). Note: Inductor-compiled torch-op quant can now auto-fuse some patterns, making SiLU+quant/RMSNorm+quant passes obsolete *except* custom-op cases (attention, collectives, sub-byte quant).
- **PTPC-FP8** (per-token-per-channel) fusion: **up to 2.5x** vs naive on MI300X. SGLang RMSNorm+FP8-dynamic-quant fusion: 1-6% e2e latency, 1-2% throughput.
- AITER v0.1.12: blockwise sparse Sage Attention + fused gated RMSNorm+group-quant; MI355X tuned configs for Kimi-K2.5 / DeepSeek-V3.

**Best (both gens):** AITER fused kernels + vLLM Inductor fusion passes. HipKittens beats AITER/PyTorch 1.1-2.2x on memory-bound layernorm/rotary (MI355X).

## 5. Collectives (RCCL / QuickReduce / MoRI / custom all-reduce)

- **3-way adaptive dispatch (vLLM/SGLang, SOTA):** Custom AllReduce (CR) lowest latency <~512KB-2MB; **QuickReduce** (MK1, two-shot + INT4/INT6/INT8/FP4 inline compression) wins mid/large — **up to 3x vs RCCL**, up to 2.25x on 2x/4x MI300X; RCCL for largest. Crossover ~1MB @ TP2, ~4MB @ TP8. QR lifts **TTFT not TPOT** (decode comm volume tiny). MI300-only; `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`. FP4 added on MI355 (≈ INT4 perf/accuracy).
- **RCCL** (ROCm 7.2): native 4-NIC topology, rail-aligned patterns, NCCL 2.28 backports. **rocSHMEM** GDA backend removes CPU from critical path. **QuickReduce/QuickReduce-FP4** used as MLPerf v6.0 TP comm.
- **MoRI-IO** KV-cache transfer beats Mooncake ~10% (DeepSeek-R1 671B FP8: 34886 vs 31685 tok/s).

**Best on MI300X:** CR + QuickReduce(INT4) + RCCL adaptive. **Best on MI355X:** + QuickReduce-FP4, MoRI EP for distributed MoE.

## MLPerf / head-to-head vs NVIDIA

- **MLPerf Inference v6.0 (latest, AMD blog):** MI355X 1-node Llama2-70B offline 103,480 / server 100,282 / interactive 73,608 tok/s; gpt-oss-120b 95,004 / 82,136. **vs NVIDIA B300**: 92% offline, 93% server, **104% interactive** (AMD wins); gpt-oss-120b up to **115%** server / 111% offline vs OEM B200. 11-node Llama2-70B >1M tok/s, 97-98% scaling. **4.4-4.8x over v5.1** (FP4). Stack: vLLM + AITER + Quark + GEAK + QuickReduce.
- **MLPerf v5.1 (Sep 2025):** MI355X debut (first MXFP4 results, 6-week tune window). Llama2-70B offline 8-chip: MI355X 93,045 vs B200 65,770 vs H200 31,383 (3rd-party derived; precision/chip-count differ). 64-chip 648,248 tok/s. MI355X 3.4x over MI300X (4-node); FP4 2.7x over MI325X FP8. MI325X competitive vs H200 (same gen).
- **NVIDIA still leads raw reasoning per-GPU:** GB300 NVL72 +45% DeepSeek-R1 vs GB200, ~5x vs Hopper; CoreWeave GB300 6,005 tok/s/GPU DeepSeek-R1. AMD closing fast.
- DeepSeek-R1 SGLang on MI300X: AMD claims 2-5x throughput at iso-latency, 75% better throughput / 60% lower latency vs HGX H200; Moreh >21k tok/s on 8x MI300X.

---

## SOTA-by-operator table

| Operator | Best backend on MI300X (gfx942) | Best on MI350X/MI355X (gfx950) | Source / evidence |
|---|---|---|---|
| Dense GEMM (BF16/FP16) | hipBLASLt / AITER (tuned) | Gluon 1489 TFLOPS FP16 @98.75%; HipKittens 1610 BF16 | Gluon blog; HK arXiv |
| Scaled-FP8/BF8 GEMM | hipBLASLt + AITER block-scale (2x) | HIP/C++ 8-wave **3204** (>hipBLASLt); Gluon BF8 **3257**; HK 4-wave **3327** | cdna4-gemm; Gluon; HK |
| MXFP4 GEMM | AITER (emerging) | **Gluon 5255 TFLOPS @92.41%** (native scaled-MFMA) | Gluon blog |
| GEMM offline tuning | hipBLASLt QuickTune/TensileLite | same | hipBLASLt tuning blogs |
| Attention prefill (MHA) | AITER MHA asm/CK (ROCM_AITER_FA) | AITER MHA asm; **HipKittens fwd (academic)** | vLLM ROCm blog; HK |
| Attention decode (paged) | AITER `pa_fwd_asm` (3.8x vs ROCM_ATTN) | AITER `pa_fwd_asm` (3.6x) | vLLM ROCm blog |
| MLA decode | AITER `mla_decode_fwd` / Triton-MLA (+2-3%) | **ROCM_AITER_MLA** (best TTFT) | vLLM ROCm blog |
| MLA prefill | AITER asm MHA | AITER asm MHA | vLLM ROCm blog |
| GQA/MQA bwd, d=64 | AITER (gaps) | **HipKittens 1.8-2.3x** over baselines | HK arXiv |
| Fused MoE grouped-GEMM | AITER CK/asm (3x); FlyDSL MXFP4 | **AITER FlyDSL A4W4** (1.6x) + MoRI EP | LMSYS MoRI; Kimi-K2.5 blog |
| MoE EP dispatch/combine | AITER / mori | **MoRI** (2.56x BW, in-kernel fusion) | LMSYS MoRI blog |
| RMSNorm/LayerNorm (+quant) | AITER fused + vLLM Inductor | AITER + HK (1.1-2.2x mem-bound) | vLLM fusions; HK |
| SiLU/act+quant | AITER + ActivationFusionPass (+8%) | same | vLLM fusions |
| RoPE (+KV) | AITER RoPE+KV fusion | same | vLLM fusions |
| FP8 quant | AITER / PTPC-FP8 (2.5x) | AITER | PTPC-FP8 blog |
| MXFP4/FP4 quant | AITER / Quark | AITER + native HW | ROCm 7.0; Quark |
| All-reduce | CR <2MB → QuickReduce-INT4 → RCCL | + QuickReduce-FP4 | QuickReduce blogs |
| Fused all-reduce+RMSNorm | AITER / SGLang fused | same | SGLang AMD |

## Updates to push into our cards/matrix

- **`operators/dense_gemm/backends/`** — add **Gluon** as a 🟢 SOTA backend (currently absent from matrix); FP16 1489/BF8 3257/MXFP4 5255 ceilings. Tag tilelang/triton row with Gluon-class numbers. Add HipKittens BF16 1610 reference.
- **`operators/scaled_quant_gemm/backends/`** + **`quant_fp4_mxfp`** — refresh with HIP/C++ 8-wave ping-pong 3204 TFLOPS (>hipBLASLt) and 4-wave interleave; note scheduling origin = HipKittens. Confirm hipBLASLt 2750/3130 FP8 bar.
- **Add a new backend card `hipkittens` (HK)** across dense_gemm, scaled_quant_gemm, attention_prefill_fmha, gqa_mqa_attention, rmsnorm/rope — academic SOTA, beats AITER asm on fwd/GQA-bwd/mem-bound. Note wave-specialization-fails-on-CDNA finding for optimization cards.
- **`operators/attention_prefill_fmha` / `attention_decode_paged` / `mla_attention`** — update AITER cells with concrete vLLM speedups (FA 2.7-4.4x; MLA 1.2-1.5x) and the gfx942-vs-gfx950 MLA dispatch nuance (Triton-MLA +2-3% on gfx942; AITER-MLA wins on gfx950).
- **`operators/fused_moe_grouped_gemm` / `grouped_gemm_moe`** — add **FlyDSL** as 🟢 for MXFP4/A4W4 (it's in matrix for grouped_gemm_moe; ensure fused_moe_grouped_gemm reflects FlyDSL 1.6x + MoRI in-kernel fusion).
- **`operators/moe_dispatch_combine` / `all_to_all_dispatch_combine`** — refresh **mori** card with MoRI quantized A2A 2.56x BW, InterNodeV1LL adaptive numbers.
- **`operators/allreduce/backends/`** — add **QuickReduce** as a distinct 🟢 backend (up to 3x vs RCCL, two-shot+inline compression, MI300-only, FP4 on MI355); document 3-way adaptive dispatch + TTFT-only benefit in tuning.md.
- **Norm/act/rope cards** — add vLLM Inductor fusion-pass nuance (some passes now obsolete via torch-op quant) and PTPC-FP8 2.5x; AITER fused gated RMSNorm+group-quant (v0.1.12).
- **`hardware/` cards** — add MI355X MLPerf v6.0 parity-to-win vs B300 (104% interactive Llama2-70B; 115% gpt-oss server) and 4.4-4.8x gen-over-gen; note ~45% sustained util gap on MI300X (third-party) is narrowing.
- **`index/sota_matrix.md`** — Gluon column/cells missing entirely; add HipKittens column; FlyDSL is already present, verify MoRI/QuickReduce rows.

## Sources

- HipKittens: Fast and Furious AMD Kernels — arXiv 2511.08083 (Hazy Research, Nov 2025): https://arxiv.org/abs/2511.08083 ; HTML https://arxiv.org/html/2511.08083v1 ; blog https://hazyresearch.stanford.edu/blog/2025-11-09-hk ; code https://github.com/HazyResearch/HipKittens
- From Naive to Near-Peak: GEMM Kernels with Gluon (MI350/MI355, ROCm 7.0) — https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- FP8 GEMM Optimization on AMD CDNA4 (8-wave ping-pong, ROCm 7.1, MI355X) — https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- Deep Dive Into 4-Wave Interleave FP8 GEMM — https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
- Matrix Core Programming on AMD CDNA3 and CDNA4 — https://rocm.blogs.amd.com/ (Sep 30 2025)
- GEMM Tuning within hipBLASLt Part 1/2 + Day-0 Offline Tuning + TensileLite Advanced Guide — rocm.blogs.amd.com (Sep-Nov 2025, Apr 2026)
- Beyond Porting: vLLM attention backends on AMD ROCm (Feb 27 2026) — https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- AITER-Enabled MLA Layer Inference on MI300X (AITER v0.1.4, ROCm 6.4) — https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- Supercharge DeepSeek-R1 on MI300X (MLA 17x decode / 14x prefill) — https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html
- ROCm/aiter releases (v0.1.12) — https://github.com/ROCm/aiter/releases
- Win on TCO: MI355X SGLang + MoRI (LMSYS, May 28 2026) — https://www.lmsys.org/blog/2026-05-28-mori/ ; AMD https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html
- Accelerating Kimi-K2.5 on MI300X: FlyDSL FusedMoE — https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- Revolutionizing MoE: 10x Align & Sort — https://www.amd.com/en/blogs/2025/revolutionizing-mixture-of-experts-performance-10.html
- DeepGEMM (NVIDIA reference, Mega-MoE concept) — https://github.com/deepseek-ai/DeepGEMM
- QuickReduce: Up to 3x Faster All-reduce for vLLM/SGLang — https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html ; FP4 on MI355 — https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html ; vLLM PR #19744
- vLLM fusion (torch.compile) passes — https://docs.vllm.ai/en/latest/design/fusions/ ; ROCm vLLM perf opt — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- PTPC-FP8 on ROCm (2.5x) — https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html
- ROCm 7.0 release (FP4/FP6/FP8, AITER, Quark) — https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.0-blog/README.html
- ROCm 7.2 (hipBLASLt swizzle, RCCL 4-NIC/NCCL2.28, rocSHMEM GDA, DeepEP) — https://rocm.blogs.amd.com/software-tools-optimization/rocm7.2/README.html
- MLPerf Inference v6.0 (MI355X vs B300/B200) — https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html
- Technical Dive into AMD MLPerf Inference v5.1 — https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html ; MLCommons https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/
- High-Accuracy MXFP4/MXFP6 Mixed-Precision on AMD GPUs (Oct 29 2025) + MXFP4 online rotation — rocm.blogs.amd.com
- TileLang FlashMLA parity with AITER asm (MI300X) — https://github.com/tile-ai/tilelang
- Third-party MI300X util analysis (~45% sustained) — researchgate / emergentmind MI300X perf analysis
