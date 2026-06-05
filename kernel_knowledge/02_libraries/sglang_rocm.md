# SGLang on ROCm / MI300X (CDNA3, gfx942) — Backends, Env, Kernel Dispatch

> Scope: how to launch, tune, and **reason about which kernel actually runs** when serving LLMs with
> SGLang on AMD Instinct MI300X. AMD-only. Verified against `sgl-project/sglang` main (2026-06) and
> ROCm 7.x. CUDA mentioned only as contrast.
>
> Mental model: on MI300X almost every hot kernel (attention, GEMM, MoE, RMSNorm, all-reduce) has
> **2-4 candidate implementations** — AITER (asm/CK/hipBLASLt), Triton, CK directly, or a torch-native
> fallback. The job of an optimizer is to pick the fastest *correct* one per shape regime (prefill vs
> decode) via flags + env. SGLang is Amdahl-dominated by GEMM (often ~70-80% GPU time on dense models)
> then attention then MoE — spend budget in that order.

---

## 1. Images & version matrix (use a prebuilt image; do NOT pip-install AITER yourself)

| Component | Recommended | Notes |
|---|---|---|
| Docker image | `lmsysorg/sglang:*-rocm*` (Docker Hub) or `rocm/sglang-staging:latest` | bundles AITER + tuned hipBLASLt + CK; the *only* sane way to get a matched AITER |
| Build file | `sglang/docker/rocm.Dockerfile` | has MI300X_VF (virtualized) handling; symlinks skipped in image build |
| ROCm | 7.0+ (7.2.x stable as of 2026-06; 7.13 preview exists) | MI300X fully supported since ROCm 7.0.0 (Sep 2025) |
| PyTorch | 2.7+ (ROCm build) | `TORCH_BLAS_PREFER_HIPBLASLT=1` to force hipBLASLt over hipBLAS |
| AITER | the version baked in the image | **never** `pip install aiter` ad-hoc — ABI must match the image |

Build from source (inside a ROCm container):
```bash
# sgl-kernel (HIP custom ops)
python sgl-kernel/setup_rocm.py install
# python package
pip install -e "python[all_hip]"
```

Canonical launch (DeepSeek-R1, TP=8, FP8):
```bash
docker run -d -it --ipc=host --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --group-add render --security-opt seccomp=unconfined \
  -v /home:/workspace rocm/sglang-staging:latest

HSA_NO_SCRATCH_RECLAIM=1 SGLANG_USE_AITER=1 \
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 --tp 8 --quant fp8 \
  --trust-remote-code --host 0.0.0.0 --port 30000
```
`HSA_NO_SCRATCH_RECLAIM=1` is near-mandatory on MI300X — without it the HSA runtime reclaims scratch
between dispatches and you eat large idle gaps / occasional hangs under load.

---

## 2. The `SGLANG_USE_AITER` switch and its kernel family

`SGLANG_USE_AITER=1` is the **master gate** for AMD's tuned-kernel library (AI Tensor Engine for ROCm,
`ROCm/aiter`). It is on by default in recent ROCm images but set it explicitly. When on, SGLang routes
attention, MoE, GEMM, RMSNorm, activation, quant, and some collectives through AITER, which itself does
**runtime dispatch** across CK, hand-written **assembly (ASM)**, Triton, and hipBLAS depending on shape.

AMD-reported MI300X gains (marketing, treat as ceiling): AITER **MLA decode ~17x**, **MHA prefill ~14x**
vs untuned baselines. If `SGLANG_USE_AITER=1` is set but the `aiter` wheel is missing you get
`ImportError: aiter is required when SGLANG_USE_AITER is set to True` — that is the image mismatch tell.

### SGLang AITER / ROCm env vars (verbatim from `python/sglang/srt/`)

| Env var | What it gates | When to touch |
|---|---|---|
| `SGLANG_USE_AITER` | master AITER switch | keep `1` on MI300X |
| `SGLANG_USE_AITER_AR` | AITER **all-reduce** (custom AR) | TP runs; `1` usually wins small-msg AR (see rccl_comm.md) |
| `SGLANG_USE_AITER_AG` | AITER **all-gather** | TP/seq-parallel |
| `SGLANG_USE_AITER_MOE_GU_ITLV` | gate-up interleaved fused-MoE layout | MoE models; layout-dependent perf |
| `SGLANG_USE_AITER_FP8_PER_TOKEN` | per-token FP8 activation quant | FP8 dynamic-act models |
| `SGLANG_USE_AITER_UNIFIED_ATTN` | AITER unified (prefill+decode) Triton attn | alt to split prefill/decode kernels |
| `SGLANG_AITER_FP8_PREFILL_ATTN` | FP8 prefill attention path | FP8 + long prefill |
| `SGLANG_AITER_MLA_PERSIST` | persistent-kernel MLA decode | DeepSeek MLA decode latency |
| `SGLANG_AITER_UNIFIED_VERIFY` | unified attn for spec-decode verify | speculative decoding |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | fused MLA decode + RoPE kernel | DeepSeek decode |
| `SGLANG_ROCM_USE_MULTI_STREAM` | multi HIP-stream overlap | overlap comm/compute |
| `SGLANG_ROCM_DISABLE_LINEARQUANT` | disable ROCm linear-quant fast path | debugging accuracy regressions |
| `SGLANG_OPT_USE_AITER_SILU_MUL` | AITER fused SiLU·mul activation | MoE/MLP activation |
| `SGLANG_OPT_USE_AITER_MHC_PRE/POST` | AITER fused pre/post matmul-head-comm | model-specific fusion |
| `SGLANG_OPT_USE_AITER_INDEXER` | AITER indexer (DSA / NSA sparse attn) | DeepSeek-V3.2 sparse |
| `SGLANG_MOE_PADDING` | pad MoE weights (align GEMM) | MoE; usually `1` |
| `SGLANG_MOE_CONFIG_DIR` | dir of tuned fused-MoE configs (json) | point at your tuned configs |
| `SGLANG_MOE_NVFP4_DISPATCH` | NVFP4 MoE dispatch path | nvfp4 MoE |
| `TORCH_BLAS_PREFER_HIPBLASLT` | torch GEMM → hipBLASLt | keep `1` |
| `HSA_NO_SCRATCH_RECLAIM` | stop HSA scratch reclaim | keep `1` on MI300X |
| `HIP_FORCE_DEV_KERNARG` | device kernarg (lower launch latency) | keep `1` (default in image) |

---

## 3. Attention backends — flag → kernel map

Selected via `--attention-backend <name>` (and the version-dependent split
`--prefill-attention-backend` / `--decode-attention-backend`). If unset, SGLang auto-selects from
hardware + model arch. Registry: **`python/sglang/srt/layers/attention/attention_registry.py`**
(`@register_attention_backend("<name>")`).

| `--attention-backend` | Backend file (`.../layers/attention/`) | Kernel under the hood | MI300X fit |
|---|---|---|---|
| `aiter` | `aiter_backend.py` (`AiterAttnBackend`) | AITER `mha_batch_prefill`, `mla_decode_fwd`/`mla_prefill_fwd`, paged ragged KV | **default & fastest** for supported models (DeepSeek MLA, Llama MHA) |
| `triton` | `triton_backend.py` + `triton_ops/` | Triton FlashAttention (editable, autotunable) | best **fallback** + the path to Tier-C code rewrites |
| `aiter` + `SGLANG_USE_AITER_UNIFIED_ATTN=1` | `triton_ops/aiter_unified_attention.py` | AITER unified Triton kernel (one kernel does chunked prefill + decode) | good when split kernels add launch overhead |
| `wave` | `wave_backend.py` | AMD **Wave** DSL attention | experimental MI300X path |
| `fa3` | `flashattention_backend.py` | FlashAttention-3 | **NVIDIA Hopper default**; FA3 vision needs CUTLASS/TMA → NOT a real MI300X path, use `triton` vision attn instead |
| `flashmla` / `flashinfer_mla` / `cutlass_mla` / `trtllm_mla` | resp. backend files | MLA variants, mostly CUDA-leaning | avoid on MI300X unless verified |
| `dsa` / `nsa` / `dsv4` | `dsa_backend.py`, `nsa_backend.py`, `deepseek_v4_backend*.py` | DeepSeek sparse attention (indexer + flashMLA); uses `SGLANG_OPT_USE_AITER_INDEXER` | DeepSeek-V3.2 / V4; AITER ROCm coverage still maturing |
| `torch_native` / `flex_attention` | `torch_native_backend.py`, `torch_flex_backend.py` | reference torch / FlexAttention | correctness reference only |

Practical ranking on MI300X (the op-unittest is the judge — see backend bake-off doctrine):
- **MHA model (Llama/Qwen dense), decode:** `aiter` → `triton`
- **MHA model, prefill:** `aiter` → `triton` (try unified-attn for launch-bound small batches)
- **MLA model (DeepSeek), decode:** `aiter` (+`SGLANG_ROCM_FUSED_DECODE_MLA=1`, `SGLANG_AITER_MLA_PERSIST=1`) → `triton`
- **MLA model, prefill:** `aiter` → `triton`
- **New/unsupported arch where AITER CK errors** (`device_gemm does not support this GEMM problem`,
  see issue #16025): fall back to `--attention-backend triton`.

`AiterAttnBackend` imports: `from aiter import (...)`, `from aiter.mla import mla_decode_fwd,
mla_prefill_fwd`, `from aiter.ops.triton.attention.unified_attention import unified_attention` — that
import block (`aiter_backend.py` lines ~37-49) is the literal dispatch surface to read when debugging
"which AITER kernel ran".

---

## 4. GEMM dispatch (the head kernel — ~70-80% GPU time on dense models)

SGLang dense linears go through PyTorch → hipBLASLt (preferred via `TORCH_BLAS_PREFER_HIPBLASLT=1`) or,
when AITER linear is on, through AITER GEMM (CK/ASM/hipBLAS). ROCm-specific linear helpers live in
**`python/sglang/srt/layers/rocm_linear_utils.py`**.

| Mechanism | Source edit? | How to engage |
|---|---|---|
| hipBLASLt tuned DB | no | default; watch log `not found tuned config ... using default config` (means generic solution → slow) |
| hipBLASLt solution pin | no | offline `hipblaslt-bench` per (M,N,K,dtype,trans,bias) → `HIPBLASLT_TUNING_FILE=<file>` |
| PyTorch TunableOp | no | `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=<csv>`; warm once, then ship with `TUNING=0` — parity-safe, easiest first move |
| AITER GEMM | no (flag) | engaged with AITER on; often wins skinny/decode + fuses bias/act |
| Triton matmul | **yes** | editable; autotune `BLOCK_M/N/K`, `GROUP_M`, `num_warps∈{4,8}`, `num_stages∈{1,2}`, `matrix_instr_nonkdim∈{16,32}` (MFMA), `waves_per_eu`, `kpack`, `SPLIT_K` for small-M decode |
| FP8 GEMM | flag | `--quantization fp8` → fnuz e4m3 on MI300X (see §6) |

Block-scaled FP8 GEMM and per-token quant paths are in `rocm_linear_utils.py` and the quant layer
(`layers/quantization/`). Decode GEMM is **skinny** (M = batch): AITER skinny-GEMM / TunableOp /
Triton split-K are the candidates; prefill GEMM is large-M and hipBLASLt-tuned-DB usually wins.

---

## 5. MoE dispatch (DeepSeek / Mixtral / Qwen-MoE)

| Path | File | Notes |
|---|---|---|
| AITER fused MoE | `layers/moe/moe_runner/aiter.py`, `layers/moe/rocm_moe_utils.py` | CK 2-stage MoE (`ck_moe_stage1/2`) + ASM; gated by AITER; `SGLANG_USE_AITER_MOE_GU_ITLV` controls gate-up interleave |
| Triton fused MoE | `layers/moe/fused_moe_triton/` (`layer.py`, `triton_kernels_moe.py`) | editable; out-of-box but needs tuning to match CK; config json via `SGLANG_MOE_CONFIG_DIR` |
| MXFP4 MoE | `layers/quantization/rocm_mxfp4_utils.py`, `fused_moe_triton/mxfp4_moe_sm120_triton.py` | **MXFP4 requires CDNA3/CDNA4 + `SGLANG_USE_AITER=1`** |
| `quark_int4fp8_moe` | quark quant | AMD-only online MoE quant for CDNA3/CDNA4 |

Known CK MoE pitfall (issue #16025): GLM-4.7-FP8 + CUDA/HIP-graph capture crashes in `ck_moe_stage2`
(`device_gemm does not support this GEMM problem`). Workarounds: disable graph capture for that model,
or force Triton fused-MoE. CK is hand-written asm → not rapidly tunable for novel shapes; Triton MoE is
the iterate-able path.

---

## 6. Quantization flags (and the MI300X FP8 *fnuz* gotcha)

Supported on AMD: **FP8, MXFP4, AWQ, W8A8, GPTQ, compressed-tensors, Quark, `petit_nvfp4`**.
Not supported (NVIDIA-only): `awq_marlin`, `gptq_marlin`, `gguf`, `modelopt_fp8/fp4`.

| Flag / env | Effect |
|---|---|
| `--quantization fp8` | online FP8 (via AITER or Triton) |
| `--kv-cache-dtype fp8_e4m3` | FP8 KV cache (bandwidth + capacity); **accuracy gate** |
| `--quantization w8a8_int8` / `w8a8_fp8` | INT8 / FP8 W8A8 |
| MXFP4 (model is pre-quant) | needs CDNA3/CDNA4 + `SGLANG_USE_AITER=1` |
| `petit_nvfp4` | `pip install petit-kernel`; NVFP4 on MI300X, no `--quantization` flag for pre-quant models |
| `quark_int4fp8_moe` | AMD-only online MoE quant |

**FP8 dialect — the #1 MI300X correctness trap.** MI300X (gfx942/CDNA3) implements **fnuz** FP8
(finite, no -0, no inf), while MI325/MI350/MI355 (CDNA4) use OCP-standard FP8. fnuz vs OCP share bit
layout but differ in exponent bias by **one** → a byte read in the wrong dialect comes back off by
**exactly 2x**. Code that only distinguishes e4m3/e5m2 (not fnuz/OCP) silently corrupts. Always validate
FP8 accuracy (gsm8k or a small eval) on MI300X; do not assume a config that works on MI350 ports down.

---

## 7. HIP graph (CUDA graph) & scheduling flags

| Flag | Effect / MI300X note |
|---|---|
| `--cuda-graph-max-bs N` | max batch captured into HIP graph (decode launch-overhead killer) |
| `--cuda-graph-bs <list>` | explicit captured batch sizes |
| `--disable-cuda-graph` | turn off capture — use to dodge the CK MoE capture crash (issue #16025) |
| `--enable-torch-compile` / `--torch-compile-max-bs` | torch.compile graph capture |
| `--chunked-prefill-size N` | chunk long prefills; trades TTFT vs throughput; key for mixed traffic |
| `--max-running-requests`, `--max-total-tokens`, `--mem-fraction-static` | scheduler / KV pool sizing |
| `--schedule-policy {lpm,fcfs,...}` / `--schedule-conservativeness` | batch admission |
| `--enable-mixed-chunk` | mix chunked-prefill with decode in one batch |
| `--page-size N` | KV page size (attention kernel granularity) |

HIP graphs matter most for **decode** on MI300X: per-dispatch launch cost is real, and decode is a
stream of tiny kernels. Capture the batch sizes you actually serve. Combine with
`HIP_FORCE_DEV_KERNARG=1` and `GPU_MAX_HW_QUEUES=2`.

---

## 8. Parallelism flags

| Flag | Use |
|---|---|
| `--tp N` | tensor parallel; stay ≤8 (one xGMI island) on a node — see rccl_comm.md |
| `--dp N` / `--enable-dp-attention` | data parallel (+ DP attention for MLA) |
| `--ep-size N` / `--enable-ep-moe` | expert parallel for MoE (all-to-all dispatch/combine) |
| `--pp N` | pipeline parallel; for >8 GPU / multi-node |
| `--enable-deepep-moe` | DeepEP all-to-all MoE comm kernels |

For MoE at scale use **DP attention + EP MoE**; the all-to-all becomes the comm bottleneck — tune RCCL
(`SGLANG_USE_AITER_AR/AG`, `NCCL_MIN_NCHANNELS`) per rccl_comm.md.

---

## 9. Optimizer playbook (cheapest → most invasive)

1. **Config/env sweep first (parity-safe):** `SGLANG_USE_AITER=1`, `HSA_NO_SCRATCH_RECLAIM=1`,
   `TORCH_BLAS_PREFER_HIPBLASLT=1`, TunableOp warm pass, `--attention-backend {aiter,triton}` bake-off,
   HIP-graph batch sizes, `--chunked-prefill-size`. Measure e2e tok/s; Amdahl-gate each change.
2. **Per-backend tune:** hipBLASLt solution pin / TunableOp CSV; tuned fused-MoE config json
   (`SGLANG_MOE_CONFIG_DIR`); AITER MLA persist/fused-decode toggles.
3. **Code rewrite (Tier C):** only if the fastest *correct* backend is Triton/CK-editable — rewrite the
   Triton attention/GEMM/MoE kernel (autotune MFMA `matrix_instr_nonkdim`, `waves_per_eu`, split-K).
4. **Quantize (Tier D, breaks byte parity):** FP8 / MXFP4 / FP8 KV — always behind an accuracy gate,
   and re-check the **fnuz** dialect on MI300X.

> Parity caveat: even a same-dtype backend swap can flip a borderline bf16 argmax (reduction order
> differs across AITER/Triton/CK). Re-check greedy/temp=0 e2e parity; if it diverges, run a task-accuracy
> probe before accepting on throughput alone.

---

## Sources
- SGLang AMD GPU docs: https://github.com/sgl-project/sglang/blob/main/docs/platforms/amd_gpu.md
- SGLang attention backend docs: https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md
- SGLang ROCm Dockerfile: https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile
- SGLang attention registry (source): https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
- AITER repo (ROCm/aiter): https://github.com/ROCm/aiter
- AITER ROCm blog: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- GLM-4.7-FP8 AITER CK MoE crash (issue #16025): https://github.com/sgl-project/sglang/issues/16025
- ROCm SGLang benchmark doc: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang.html
- Step-3 on AMD Instinct (FA3 vision → Triton): https://rocm.blogs.amd.com/artificial-intelligence/step3-model/README.html
- Kimi-K2.5 fused-MoE on MI300X: https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
