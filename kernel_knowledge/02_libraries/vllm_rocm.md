# vLLM on ROCm / MI300X (CDNA3, gfx942) — Backends, Env, Kernel Dispatch

> Scope: launching, tuning, and reasoning about **which kernel runs** when serving LLMs with vLLM **V1**
> on AMD Instinct MI300X. AMD-only. Verified against `vllm-project/vllm` main (2026-06) + ROCm 7.x.
>
> vLLM on ROCm has two kernel worlds: (1) vLLM's **own hand-written HIP custom ops** in
> `csrc/rocm/` (custom PagedAttention, skinny GEMMs), and (2) **AITER** (`ROCm/aiter`) tuned kernels
> wired in through `vllm/_aiter_ops.py`. The optimizer picks per shape via `--attention-backend` enums
> and the `VLLM_ROCM_USE_AITER*` env hierarchy.

---

## 1. Images & version matrix

| Item | Value / note |
|---|---|
| **Official image (use this)** | `vllm/vllm-openai-rocm` (Docker Hub, upstream) — official since ~Jan 20 2026 |
| **Deprecated** | `rocm/vllm`, `rocm/vllm-dev` — **deprecated Jan 2026**, migrate to upstream |
| AMD-published preview tag (example) | `rocm/vllm:rocm7.13.0_gfx950-...` (gfx950 = MI355; pick a gfx942 tag for MI300X) |
| Build file | `docker/Dockerfile.rocm` (default ROCm 7.0; `BASE_IMAGE=rocm/vllm-dev:base`) |
| ROCm | 7.0+ (MI300X fully supported since 7.0.0); 7.2.x stable; wheels: `pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm7.0` |
| Engine | **V1 only** for new work (V0 is gone); all guidance below is V1 |

Canonical launch (Llama-3.3-70B MHA, AITER, high concurrency):
```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 \
TORCH_BLAS_PREFER_HIPBLASLT=1 SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 \
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 --attention-backend ROCM_AITER_FA
```
DeepSeek MLA: `VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-R1-0528 --tensor-parallel-size 8
--attention-backend ROCM_AITER_MLA`.

---

## 2. The `VLLM_ROCM_USE_AITER` hierarchy (`vllm/envs.py`)

`VLLM_ROCM_USE_AITER` is the **master switch (default `0`/False)**. Every other `VLLM_ROCM_USE_AITER_*`
flag is gated by it. Verbatim from `vllm/envs.py` (defaults as of 2026-06 main):

| Env var | Default | Gates kernel |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | **0** | master switch (turn this on first) |
| `VLLM_ROCM_USE_AITER_LINEAR` | 1 | AITER quant ops + GEMM for linear layers |
| `VLLM_ROCM_USE_AITER_MOE` | 1 | AITER fused-MoE kernels |
| `VLLM_ROCM_USE_AITER_RMSNORM` | 1 | AITER RMSNorm (+ fused add/quant variants) |
| `VLLM_ROCM_USE_AITER_MLA` | 1 | Multi-head Latent Attention (DeepSeek) |
| `VLLM_ROCM_USE_AITER_MHA` | 1 | Multi-Head Attention; set `0` → Triton/ROCM_ATTN path |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | 0 | AITER paged-attention decode |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | 0 | AITER unified attn; only active when MHA off |
| `VLLM_ROCM_USE_AITER_FP8BMM` | 1 | FP8 batched matmul |
| `VLLM_ROCM_USE_AITER_FP4BMM` | 1 | FP4 batched matmul — **CRASHES MI300X**, see §6 |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | 0 | ASM FP4 GEMM for MXFP4 (MI350/355 only) |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | 1 | AITER Triton GEMM path |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | 0 | AITER Triton RoPE |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` | 0 | fused shared experts (incompatible w/ MoRI) |
| `VLLM_ROCM_USE_SKINNY_GEMM` | 1 | vLLM's own skinny GEMM (`csrc/rocm/skinny_gemms.cu`) for small batch |
| `VLLM_ROCM_FP8_PADDING` | 1 | pad FP8 linear weights to 256B |
| `VLLM_ROCM_MOE_PADDING` | 1 | pad MoE weights |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | 0 | affects ROCM_AITER_FA only; `0` low-concurrency, `1` high (≥32) |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | 1 | vLLM custom paged-attn decode (used when ROCM_ATTN selected) |

> Note: `VLLM_USE_TRITON_FLASH_ATTN` and `VLLM_USE_ROCM_FP8_FLASH_ATTN` were **V0-era** vars (appear in
> old AMD DeepSeek reference configs). In current **V1** main they are gone — selection is via the
> `--attention-backend` enum + the AITER hierarchy. If you copy an old `VLLM_USE_TRITON_FLASH_ATTN=0`
> recipe it is silently ignored on V1.

---

## 3. Attention backends — `--attention-backend` enum → kernel

Dispatch lives in **`vllm/platforms/rocm.py`** (`get_attn_backend_cls`, gfx9 selection list ~L382-408).
Backend classes are in **`vllm/v1/attention/backends/`**. The candidate list vLLM builds on gfx942 is
(in priority order): ROCM_ATTN → ROCM_AITER_UNIFIED_ATTN → TRITON_ATTN, with AITER MLA/MHA inserted
when `rocm_aiter_ops.is_mla_enabled()/is_mha_enabled()`.

| `--attention-backend` | File (`v1/attention/backends/`) | Kernel | MI300X fit |
|---|---|---|---|
| `ROCM_AITER_FA` | `rocm_aiter_fa.py` (`AiterFlashAttention*`) | AITER flash-attn (prefill+decode metadata, KV shuffle/gather) via `vllm._aiter_ops.rocm_aiter_ops` | **default for MHA models** |
| `ROCM_AITER_MLA` | `mla/rocm_aiter_mla.py` | AITER MLA decode (`_rocm_aiter_mla_decode_fwd`) | **default for DeepSeek MLA** |
| `ROCM_AITER_MLA_SPARSE` | `mla/rocm_aiter_mla_sparse.py` | AITER sparse MLA (DSA) | DeepSeek-V3.2 |
| `ROCM_AITER_TRITON_MLA` | (triton MLA) | AITER Triton MLA | MLA fallback |
| `ROCM_AITER_UNIFIED_ATTN` | `rocm_aiter_unified_attn.py` | AITER unified Triton (chunked prefill + decode in one kernel) | launch-bound small batch |
| `ROCM_ATTN` | `rocm_attn.py` | **vLLM custom HIP paged-attn** (`csrc/rocm/attention.cu`) + `VLLM_ROCM_CUSTOM_PAGED_ATTN` | strong decode; no AITER needed |
| `TRITON_ATTN` | `triton_attn.py` | Triton unified attention (editable) | universal fallback |
| `TRITON_MLA` | `mla/...` | Triton MLA | MLA fallback |

vLLM's **own** PagedAttention HIP kernels (`csrc/rocm/attention.cu`) — these are *vLLM-authored*, not
AITER — are the `ROCM_ATTN` path. The kernel names to grep in a profile:
`paged_attention_ll4mi_QKV_mfma16_kernel` (MFMA-16 main path),
`paged_attention_ll4mi_QKV_mfma4_kernel` (MFMA-4 small-head path),
`paged_attention_ll4mi_reduce_kernel` (cross-block softmax reduce). Bound by `__launch_bounds__` and
templated on `BLOCK_SIZE`, KV dtype, FP8 KV. Exposed via `csrc/rocm/torch_bindings.cpp`
(`rocm_ops.def("paged_attention", ...)`).

Practical ranking (op-unittest decides):
- **MHA decode:** `ROCM_AITER_FA` → `ROCM_ATTN` (custom HIP) → `TRITON_ATTN`
- **MHA prefill:** `ROCM_AITER_FA` → `TRITON_ATTN`
- **MLA decode (DeepSeek):** `ROCM_AITER_MLA` → `ROCM_AITER_TRITON_MLA` → `TRITON_MLA`
- **AITER missing/erroring for a new shape:** `TRITON_ATTN` (several× slower than tuned, but correct)

---

## 4. GEMM / linear dispatch

| Path | Source | Engage with |
|---|---|---|
| hipBLASLt (torch) | PyTorch ROCm | `TORCH_BLAS_PREFER_HIPBLASLT=1` (prefer hipBLASLt over hipBLAS) |
| **vLLM custom skinny GEMM** | `csrc/rocm/skinny_gemms.cu` → ops `LLMM1`, `wvSplitK`, `wvSplitKrc`, `wvSplitKQ` (fp8) | `VLLM_ROCM_USE_SKINNY_GEMM=1` (default); wins **decode** (M=batch) skinny shapes |
| AITER linear / GEMM | `vllm/_aiter_ops.py` (`is_linear_fp8_enabled`, `_rocm_aiter_w8a8_gemm`), `model_executor/kernels/linear/scaled_mm/rocm.py` | `VLLM_ROCM_USE_AITER_LINEAR=1` |
| AITER GEMM tuned configs | csv via `_load_gemm_tuned_configs` / `_check_kernel_tuned(N,K,dtype,csv)` | drop tuned csv to pin per-(N,K) kernel |
| PyTorch TunableOp | env | `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=<csv>`; warm then `TUNING=0` |
| `q_gemm_rdna3*` | `csrc/rocm/q_gemm_rdna3.cu` | RDNA3 quantized GEMM (not MI300X) |

`csrc/rocm/torch_bindings.cpp` is the authoritative list of vLLM's ROCm custom ops:
`LLMM1`, `wvSplitK`, `wvSplitKrc`, `wvSplitKQ`, `paged_attention`. These are the kernels you can edit
(HIP source) when a Tier-C rewrite is justified.

---

## 5. MoE & quant op paths

| Op | Source | Notes |
|---|---|---|
| AITER fused MoE | `model_executor/layers/fused_moe/experts/rocm_aiter_moe.py`, `_aiter_ops._rocm_aiter_fused_moe_impl`, `_rocm_aiter_asm_moe_tkw1_impl` | `VLLM_ROCM_USE_AITER_MOE=1`; ASM + CK |
| topk / grouped-topk | `_aiter_ops._rocm_aiter_topk_softmax/sigmoid/biased_grouped_topk` | AITER routing kernels |
| RMSNorm (+fused) | AITER via `VLLM_ROCM_USE_AITER_RMSNORM`; `check_aiter_fused_qk_rmsnorm()` | fused add+rmsnorm+quant |
| Fusion passes | `vllm/compilation/passes/fusion/rocm_aiter_fusion.py` | torch.compile fusion of AITER ops (rms+quant, etc.) |
| scaled_mm | `model_executor/kernels/linear/scaled_mm/rocm.py` | FP8 scaled matmul |

---

## 6. Quantization (FP8 / FP4 / MXFP4) + the MI300X traps

vLLM supports FP4/FP8 W/A quant with HW accel on MI300X/MI325X/MI350X/MI355X (2-4× memory, up to ~1.6×
throughput). Launch via `--quantization {fp8,...}`, `--kv-cache-dtype fp8`.

**Trap 1 — FP4 crashes MI300X (default-on regression).** `VLLM_ROCM_USE_AITER_FP4BMM` defaults to `1`
but **MI300X (gfx942) has no FP4 HW**; vLLM tries it anyway and crashes (issue #34641, introduced
~Jan 15 2026). **Workaround:** `VLLM_ROCM_USE_AITER_FP4BMM=0`. FP4 paths (`FP4_ASM_GEMM`) are CDNA4
(MI350/355) only.

**Trap 2 — FP8 *fnuz* dialect.** MI300X uses **fnuz** FP8 (no -0/inf, exponent bias off-by-one vs
OCP). Many vLLM FP8 paths distinguish e4m3/e5m2 but not fnuz/OCP → a byte read in the wrong dialect is
off by exactly 2×. This is *why* early DeepSeek-on-AMD FP8 bring-up failed specifically on MI300X.
`VLLM_ROCM_FP8_PADDING=1` (pad to 256B) is required for the fast FP8 linear path.

**Trap 3 — AITER MLA accuracy.** `VLLM_ROCM_USE_AITER_MLA=1` caused full gsm8k accuracy loss with
Kimi-K2 DP2TP4 (aiter issue #1455). Always run an accuracy eval when enabling AITER MLA.

**Trap 4 — AITER coverage gaps.** AITER tuning targets CDNA4 first; on gfx942 a missing shape falls
back to **generic Triton** which can be several× slower than a tuned kernel. Set `AITER_ONLINE_TUNE=1`
to retry on `RuntimeError: wrong! device_gemm`.

---

## 7. HIP graph / scheduling / parallelism flags

| Flag | Effect |
|---|---|
| `--enforce-eager` | disable HIP-graph capture (debug / dodge capture crashes) |
| `--compilation-config` / `-O` | torch.compile level + custom_ops list (AITER fusion lives here) |
| `--max-num-batched-tokens`, `--max-num-seqs` | scheduler / chunked-prefill sizing |
| `--gpu-memory-utilization` | KV pool size |
| `--kv-cache-dtype fp8` | FP8 KV cache (accuracy gate; fnuz on MI300X) |
| `--tensor-parallel-size N` | TP — stay ≤8 (one xGMI island) |
| `--data-parallel-size N` + `--enable-expert-parallel` | DP attn + EP MoE for large MoE |
| `--disable-nccl-for-dp-synchronization` | DP sync without RCCL |
| `--block-size 1` | required for DeepSeek-V3.2 DSA |

Recommended ROCm env block (MI300X):
```bash
TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_FORCE_DEV_KERNARG=1 SAFETENSORS_FAST_GPU=1 \
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4BMM=0 \
NCCL_MIN_NCHANNELS=112   # multi-GPU only; see rccl_comm.md
```

DP + EP for MLA/MoE (high concurrency):
```bash
VLLM_ALL2ALL_BACKEND="allgather_reducescatter" \
vllm serve deepseek-ai/DeepSeek-R1 --data-parallel-size 8 \
  --enable-expert-parallel --disable-nccl-for-dp-synchronization
```

### Quick Reduce (custom all-reduce, ROCm)
| Env | Default | Note |
|---|---|---|
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | `NONE` | `{NONE,FP,INT8,INT6,INT4}` — quantized AR for bandwidth |
| `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` | 1 | cast for the reduce |
| `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | ~2048 | size cap to use quick-reduce |

---

## 8. Optimizer playbook (cheapest → invasive)

1. **Env sweep (parity-safe):** `VLLM_ROCM_USE_AITER=1` + `FP4BMM=0`, `TORCH_BLAS_PREFER_HIPBLASLT=1`,
   `--attention-backend {ROCM_AITER_FA, ROCM_ATTN, TRITON_ATTN}` bake-off, TunableOp warm pass,
   `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` (1 if concurrency ≥32). Amdahl-gate each.
2. **Per-backend tune:** AITER GEMM tuned-csv, hipBLASLt solution pin, tuned fused-MoE configs.
3. **Tier-C rewrite:** edit `csrc/rocm/skinny_gemms.cu` / `attention.cu` or the Triton attn/MoE kernel
   if it is the fastest correct candidate (autotune MFMA `matrix_instr_nonkdim`, `waves_per_eu`, split-K).
4. **Tier-D quant:** FP8 / MXFP4 / FP8 KV behind an **accuracy gate** + fnuz re-check on MI300X.

> Always re-validate greedy/temp=0 e2e parity after a backend swap (bf16 reduction order differs
> across AITER/Triton/custom-HIP). Quant always needs the accuracy gate by design.

---

## Sources
- vLLM env vars (source): https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
- vLLM ROCm platform dispatch (source): https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
- vLLM ROCm custom HIP kernels: https://github.com/vllm-project/vllm/tree/main/csrc/rocm
- vLLM `_aiter_ops` (AITER wiring): https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py
- vLLM V1 performance optimization (ROCm docs): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- vLLM 0.9.x + ROCm blog: https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html
- ROCm first-class in vLLM (FP8/FP4/MXFP4): https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html
- FP4BMM MI300X crash (issue #34641): https://github.com/vllm-project/vllm/issues/34641
- AITER MLA accuracy loss Kimi DP2TP4 (aiter issue #1455): https://github.com/ROCm/aiter/issues/1455
- DeepSeek on MI300X FP8 fnuz writeup: https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
- Prebuilt ROCm vLLM image guide (AMD): https://www.amd.com/en/developer/resources/technical-articles/how-to-use-prebuilt-amd-rocm-vllm-docker-image-with-amd-instinct-mi300x-accelerators.html
