# Quantization Strategy for Inference — AMD Instinct MI300X (CDNA3 / gfx942)

> Scope: AMD only. Target MI300X (gfx942, CDNA3); gfx950 (CDNA4, MI350/MI355) notes inline. This file answers **when** and **how** to quantize for LLM inference on MI300X: the format decision (bf16/fp16 → fp8 → fp4), what to quantize (weights / activations / KV cache), the scaling-granularity tradeoffs, calibration, and an accuracy-gate procedure so you never ship a silent quality regression.
>
> **The MI300X-specific catch:** memory savings are easy; *true compute acceleration* requires **native low-precision MFMA**. CDNA3 has native **FP8 (E4M3/E5M2) MFMA** → real FP8 GEMM speedup. **FP4 (MXFP4) native matmul is a CDNA4 (MI350/MI355) feature**; on MI300X many FP4 paths **dequantize to BF16/FP16** before the MFMA — you get the HBM savings but not FP4 throughput. Always confirm whether your path runs native or dequant.

---

## Hardware support matrix (MI300X)

| Format | Native MFMA on MI300X? | Memory vs bf16 | Throughput vs bf16 | Status |
|---|---|---|---|---|
| **bf16 / fp16** | yes | 1× | 1× | baseline |
| **FP8 E4M3 (W8A8)** | **yes** (hipBLASLt FP8 GEMM + AITER) | ~0.5× | up to ~2× | production-ready |
| **INT8 (W8A8)** | yes (MFMA i8) | ~0.5× | ~2× | SmoothQuant path (CK) |
| **MXFP4 (OCP, block-32, E8M0)** | **dequant-to-bf16 commonly** (native FP4 = CDNA4) | ~0.25× | mainly bandwidth win on MI300X | docs target MI350/355; gfx94x HW path functional but often upcasts |
| **NVFP4 (block-16, 2-level)** | no native | ~0.25× | bandwidth win only on MI300X | modelopt path; kernel maturity varies |

> FP8 on MI300X is the workhorse: real GEMM acceleration via hipBLASLt FP8 + the AITER attention/MoE kernels. FP4 on MI300X is primarily a **capacity/bandwidth** play unless your kernel truly issues FP4 MFMA (rare on gfx942).

---

## Decision ladder: how far to quantize

```
START: bf16/fp16 baseline. Establish accuracy reference (task evals, not just PPL).

1. Memory pressure (model won't fit / KV cache too big)?
   ├─ Quantize WEIGHTS first → FP8 weight-only (W8A16). Cheapest accuracy hit, big memory win.
   └─ KV cache dominating at long context? → FP8 KV cache (huge decode bandwidth win).

2. Need GEMM throughput (compute-bound prefill / high QPS)?
   ├─ FP8 W8A8 (weights + activations).  Native FP8 MFMA → ~2× GEMM.
   │     • per-tensor (simplest, vLLM default) → check accuracy
   │     • per-token act + per-channel weight (ptpc_fp8) → same speed, better accuracy
   └─ Still memory-bound after FP8? → consider FP4 (MXFP4) for capacity, BUT:
         • on MI300X validate it's native, not dequant (else only bandwidth win)
         • accuracy hit is real → must pass accuracy gate

3. Mixed recipe (best accuracy/throughput balance):
   FP8 (or MXFP4) weights + activations, FP8 per-tensor KV cache, sensitive layers (lm_head,
   first/last block, router) kept higher precision.

ALWAYS: run the accuracy gate (below). If it fails, back off one rung.
```

---

## What to quantize, in priority order

1. **Weights** (W-only, e.g. FP8/W8A16): largest static memory, lowest accuracy risk. Best first step for "won't fit."
2. **KV cache** (FP8): decode is HBM-bound on KV reads; FP8 KV ~halves that traffic and doubles effective context capacity. Big throughput win at long context, modest accuracy risk.
3. **Activations** (→ W8A8): unlocks native FP8 GEMM (~2×) but activations have outliers → needs per-token or SmoothQuant. Higher risk than weight-only.
4. **Down to FP4**: only when memory/capacity demands it and the accuracy gate passes; treat as last rung.

**Keep in higher precision (don't quantize first):** `lm_head` / embedding output, the router/gate in MoE, first and last transformer block, and any layer your sensitivity scan flags (large activation kurtosis).

---

## Scaling granularity tradeoffs

| Granularity | Scope of one scale | Accuracy | Cost | When |
|---|---|---|---|---|
| **Per-tensor** | whole tensor | lowest | cheapest (1 scalar) | FP8 default; well-behaved tensors |
| **Per-channel (weight)** | one col/row | better | small | weights (static, free to compute offline) |
| **Per-token (activation)** | one row (token) | better, handles per-token dynamic range | dynamic reduce per token | activations; `ptpc_fp8` pairs per-token-act + per-channel-weight |
| **Block / microscaling (MX)** | block of 32 (MXFP4) or group | best for 4-bit | scale storage + block reduce | FP4/MX; E8M0 power-of-2 scale |

- **MXFP4 (OCP):** block size **32**, single shared **E8M0** (8-bit, exponent-only) scale per block — power-of-2 scale = multiply via bit-shift. Stored as `float8_e8m0fnu`.
- **NVFP4:** block size **16**, **two-level** scale (per-block FP8 E4M3 + per-tensor FP32) → fractional scaling, more precise than MXFP4's power-of-2, at higher overhead.
- Granularity choices are **orthogonal and composable**: e.g. per-token activation scales with block-wise weight scales; online (dynamic) vs delayed (calibrated) scaling is a separate axis.

**Rule:** start per-tensor for FP8; if the accuracy gate fails, escalate to per-token-act/per-channel-weight (`ptpc_fp8`) — same FP8 speed, better accuracy — before abandoning FP8. For 4-bit, block/microscaling is mandatory.

---

## Calibration

- **Dynamic (online) quantization:** scales computed at runtime from the actual tensor (per-tensor or per-token max). No calibration data. vLLM's default FP8 path. Robust, slight runtime cost for the reduction.
- **Static (offline / delayed) quantization:** scales precomputed from a calibration set (a few hundred representative prompts). Needed for GPTQ/AWQ-style weight quant and for static activation scales. Lower runtime cost, requires representative data.
- **SmoothQuant** (for W8A8 INT8/FP8): migrate activation outliers into weights via a per-channel smoothing factor so both quantize cleanly. ROCm has a CK INT8 SmoothQuant path for MI300X.

### Tooling
- **AMD Quark** — the native ROCm quantization toolkit. Produces FP8 W8A8, MXFP4, INT8 W8A8 checkpoints with explicit OCP specs (`OCP_MXFP4Spec`, `FP4PerGroupSpec`, `FP8E4M3PerTensorSpec`) and native vLLM integration (format auto-inferred from `config.json`). Supports mixed recipes (e.g. MXFP4 W/A + FP8 per-tensor KV cache, `e8m0` scale).
- **LLM Compressor 0.9.0** — adds experimental MXFP4 weight quant + scale calibration (pending vLLM validation).
- **vLLM / SGLang** consume these directly. SGLang quant list includes `w8a8_fp8`, `quark`, `mxfp4`, `modelopt_fp4`, `petit_nvfp4`; offline (pre-quantized weights) and online (dynamic scales) both supported. Pre-quantized models (e.g. gpt-oss in MXFP4) serve directly with no extra steps.

---

## Throughput vs accuracy intuition

- **bf16 → FP8 (W8A8):** ~2× model memory reduction, up to ~1.6× end-to-end throughput on MI300X (vLLM ROCm), accuracy typically within noise on standard task evals **if** scaling granularity is right.
- **FP8 → FP4:** ~2× further memory reduction. On MI300X the *compute* gain often does **not** materialize (dequant-to-bf16); end-to-end win exists only when bandwidth savings exceed the upcast overhead. Accuracy hit is real and must be measured. (On CDNA4/MI355 native FP4 MFMA gives true throughput gains.)
- Watch for **dequant-fallback paths** (Petit on MI300X dequantizes FP4→BF16; some Marlin-FP4 paths similarly) — they give memory savings without FP4 GEMM throughput.

---

## Accuracy-gate procedure (do not skip)

Byte-parity with bf16 **breaks** the moment you quantize — so validate behaviorally, not bit-exactly.

```
1. REFERENCE: run the bf16/fp16 model on a fixed eval suite. Record:
   - Task accuracy: a real downstream suite (e.g. lm-eval-harness: MMLU, GSM8K, HumanEval,
     plus your domain evals). NOT perplexity alone — PPL can look fine while task acc drops.
   - A fixed set of generation prompts (greedy, fixed seed) for qualitative diff.

2. CANDIDATE: run the quantized model on the identical suite, identical decoding params.

3. GATES (tune thresholds to your product):
   - Task accuracy delta within budget (e.g. ≤ 0.5–1.0 pt absolute on MMLU/GSM8K).
   - No catastrophic per-task collapse (check each task, not just the average).
   - KV-cache quant: verify long-context tasks specifically (needle-in-haystack / long-doc QA)
     — KV quant errors compound over sequence length.
   - Numerical sanity: max-abs / relative error on a few layer outputs vs bf16 (sniff test).

4. IF FAIL:
   - escalate scaling granularity (per-tensor → per-token-act/per-channel-weight)
   - exclude sensitive layers (lm_head, router, first/last block) from quant
   - add/improve calibration data, or use SmoothQuant for W8A8
   - back off one rung on the decision ladder (FP4 → FP8, W8A8 → W8A16)

5. REGRESSION-LOCK: store the eval scores with the checkpoint + quant recipe + ROCm version.
   Re-run the gate on any recipe, kernel, or ROCm change.
```

---

## Enabling on MI300X (vLLM / SGLang)

```bash
# FP8 (Quark or native), with AITER fused kernels + hipBLASLt FP8 GEMM:
export VLLM_ROCM_USE_AITER=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
vllm serve <fp8-or-quark-model>          # quant format auto-inferred from config.json
# (per-token-act per-channel-weight FP8: select the ptpc_fp8 / quark recipe at quant time)

# SGLang:
python -m sglang.launch_server --model <model> --quantization w8a8_fp8   # or quark / mxfp4
```

For MXFP4 on MI300X (gfx94x): the HW path is functional but the vLLM capability check historically gated it to gfx950; verify your build accepts gfx94x and confirm whether the kernel runs native FP4 or dequantizes. Pre-quantized MXFP4 models (gpt-oss) can serve directly.

---

## Quick "which first" cheat sheet

| Symptom | First move |
|---|---|
| Model won't fit in 192 GB | FP8 weight-only |
| Long-context decode HBM-bound | FP8 KV cache |
| Prefill compute-bound, high QPS | FP8 W8A8 (per-token-act if accuracy slips) |
| Still capacity-bound after FP8 | FP4/MXFP4 — but verify native vs dequant + pass gate |
| Accuracy regressed after FP8 | escalate granularity → exclude sensitive layers → SmoothQuant |

---

## Sources

- vLLM FP8 W8A8 quantization (per-tensor default, dynamic): <https://docs.vllm.ai/en/latest/features/quantization/fp8/>
- vLLM V1 / ROCm inference optimization (FP4/FP8 HW accel on MI300X, AITER, up to 1.6× throughput): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html>
- AMD Quark MXFP4 quantization for vLLM (OCP specs, E8M0, mixed recipes): <https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm.html>
- SGLang quantization (offline/online, w8a8_fp8, quark, mxfp4, ptpc_fp8): <https://sgl-project.github.io/advanced_features/quantization.html>
- SmoothQuant INT8 GEMM on MI300X via Composable Kernel: <https://rocm.blogs.amd.com/software-tools-optimization/ck-int8-gemm-sq/README.html>
- MXFP4 support on gfx94x (community enablement / dequant caveats): <https://github.com/sgl-project/sglang/discussions/13611>
- LLM Compressor 0.9.0 MXFP4 support: <https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more>
