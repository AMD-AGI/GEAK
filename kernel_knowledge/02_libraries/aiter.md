# AITER — AMD AI Tensor Engine for ROCm (MI300X / CDNA3 / gfx942)

> AMD-only. Targets MI300X / MI325X (gfx942, CDNA3) primarily; many ops also tuned for MI350X / MI355X (gfx950, CDNA4). All FP8 on gfx942 is **FNUZ** (E4M3FNUZ / E5M2FNUZ), not OCP E4M3.

AITER (`ROCm/aiter`) is AMD's high-performance, pre-tuned kernel library for LLM inference and training. It is the rough analog of *cuBLAS + cuDNN + FlashAttention + TransformerEngine combined* for AMD Instinct. It is the **default kernel backend** for LLM inference on AMD in both vLLM (V1) and SGLang. AITER is a *dispatcher* over multiple kernel backends — it does not write everything itself; it picks the fastest of Triton / Composable Kernel (CK) / hand-tuned assembly / HIP for a given op + shape + dtype.

This file covers: what AITER ships, how to call ops from Python, how it dispatches internally, the fusion ops, and when AITER beats raw hipBLASLt / CK.

---

## 1. Why AITER exists / when it beats the alternatives

| Situation | Use AITER | Use hipBLASLt / rocBLAS | Use raw CK / Triton |
|---|---|---|---|
| Plain dense GEMM (bf16/fp16) | `tgemm.mm` wraps & autotunes hipBLASLt+rocBLAS; convenient | direct call if you manage tuning yourself | only if you need a custom epilogue |
| FP8 / INT8 a8w8 quantized GEMM | **yes** — `gemm_a8w8` fuses dequant/scale | possible via `torch._scaled_mm` but no fusion | only for novel scaling |
| Fused MoE (sorting + grouped GEMM + act) | **yes** — `fused_moe`, up to **3x** vs unfused | no (no MoE primitive) | CK has MoE but AITER wraps + tunes it |
| MLA decode (DeepSeek-style) | **yes** — `mla_decode_fwd`, up to **17x** vs naive | no | Triton MLA is several x slower than AITER asm |
| MHA prefill (flash) | **yes** — up to **14x** | no | CK FMHA directly if you need custom mask |
| RMSNorm + quant fusion, RoPE+KV write | **yes** — single fused kernel | no | hand-write |
| Novel op not in catalog | no | no | write Triton/CK yourself |

Key reasons AITER wins: (1) **fusion** — it folds dequant, scaling, bias, activation, residual-add and KV-cache writes into one kernel; (2) **asm kernels** — hand-tuned MFMA assembly for the hot DeepSeek/Llama/Mixtral shapes that neither hipBLASLt nor generic Triton reach; (3) **shape-tuned config tables** baked in `aiter/configs/` for gfx942 and gfx950.

Real measured end-to-end: DeepSeek-V3/R1 on 8x MI300X went from ~6485 tok/s to ~13704 tok/s (>2x) after AITER integration in vLLM/SGLang.

---

## 2. Install & build model

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop          # editable; builds JIT scaffolding
# keep your own Triton instead of the pinned one:
AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop
```

AITER uses a **JIT compilation** model: most C++/HIP/asm kernels live as sources and are compiled on first use into `aiter/jit/` (and ahead-of-time blobs under `aot/`). The first call to a given op pays a one-time compile cost; subsequent calls hit the cached `.so`. Set `AITER_LOG_MORE=1` to see what is being built/dispatched.

Hardware support matrix (as of v0.1.15, May 2026):

| GPU | Arch | Status |
|---|---|---|
| MI300X | gfx942 / CDNA3 | Fully supported (primary target) |
| MI325X | gfx942 / CDNA3 | Fully supported |
| MI350X | gfx950 / CDNA4 | Supported |
| MI355X | gfx950 / CDNA4 | Supported |

> gfx942 coverage caveat: for the very newest models (e.g. DeepSeek-V4) some specialized paths (paged MQA logits, sparse MLA prefill/decode) only exist on gfx950; on gfx942 these fall back to generic Triton, which is several x slower. Always verify a tuned AITER path exists for *your* shape before assuming the speedup.

---

## 3. Package layout — where ops live

Top-level `aiter/` modules:

| File | Provides |
|---|---|
| `aiter/__init__.py` | Flat re-exports: `aiter.rms_norm`, `aiter.layer_norm`, `aiter.rope_fwd`, `aiter.flash_attn_func`, … |
| `tuned_gemm.py` | `tgemm.mm(...)` — autotuned dense GEMM wrapper (hipBLASLt/rocBLAS + tuned configs) |
| `fused_moe.py` | External `fused_moe(...)` API — picks MoE kernel by quant method |
| `fused_moe_bf16_asm.py` | asm fast path for bf16 fused MoE |
| `fused_moe_dp_shared_expert.py` | DP + shared-expert MoE variant |
| `mla.py` | `mla_decode_fwd(...)`, MLA prefill helpers |
| `paged_attn.py` | Paged attention (decode) |
| `rotary_embedding.py` | RoPE helpers |
| `int4_utils.py` | INT4 (a4w4) packing helpers |

`aiter/ops/` — the actual op modules (each binds JIT/asm/CK/Triton kernels):

| File | Op family |
|---|---|
| `gemm_op_a8w8.py` | A8W8 (int8/fp8) GEMM, per-tensor & per-token/channel scaled |
| `gemm_op_a4w4.py` | A4W4 (4-bit weight) GEMM |
| `gemm_op_a16w16.py` | bf16/fp16 GEMM |
| `gemm_op_common.py` | shared GEMM dispatch logic |
| `batched_gemm_op_a8w8.py`, `batched_gemm_op_bf16.py` | batched GEMM |
| `moe_op.py`, `moe_sorting.py`, `moe_sorting_opus.py` | MoE grouped GEMM + token sorting |
| `mha.py`, `mhc.py` | multi-head attention (flash) prefill/decode |
| `attention.py` | paged / decode attention entry |
| `pa_sparse_prefill_opus.py` | sparse prefill (Opus C++) |
| `rmsnorm.py`, `norm.py`, `groupnorm.py` | normalization |
| `gated_rmsnorm_fp8_group_quant.py`, `fused_qk_rmsnorm_group_quant.py` | **fused norm+quant** |
| `fused_qk_norm_rope_cache_quant.py`, `fused_qk_norm_mrope_cache_quant.py` | **fused QK-norm + RoPE + KV-cache write + quant** |
| `rope.py`, `pos_encoding.py` | rotary / positional |
| `activation.py` | SiLU/GELU/etc. |
| `quant.py` | dynamic/static quant helpers (fp8/int8, per-token/block) |
| `cache.py` | KV-cache reshape/write |
| `topk.py`, `topk_plain.py`, `sample.py`, `sampling.py` | router top-k / sampling |
| `communication.py`, `custom_all_reduce.py`, `quick_all_reduce.py` | RCCL-bypass collectives (all-reduce/gather/reduce-scatter) |
| `deepgemm.py` | DeepGEMM-style fine-grained FP8 block GEMM path |
| `causal_conv1d.py`, `chunk_gated_delta_rule_fwd_h.py` | SSM / linear-attention ops |
| `shuffle.py`, `trans_ragged_layout.py` | layout/swizzle helpers |

`aiter/ops/` backend subdirs: `triton/` (Triton kernels), `flydsl/` (FlyDSL mixed-precision MoE, e.g. A4W4), `opus/` (single-header C++ `opus.hpp` HIP kernels), `torch_ref/` (reference impls for correctness tests).

---

## 4. Backend dispatch — how AITER picks Triton vs CK vs ASM vs HIP

AITER is a **multi-backend dispatcher**. For a given op + (M,N,K) + dtype + arch it selects from:

1. **ASM (hand-tuned MFMA assembly)** — fastest path for the curated hot shapes (DeepSeek MLA decode, fmoe, block-scale GEMM). Selected when a tuned config matches the shape/arch in `aiter/configs/`.
2. **Composable Kernel (CK)** — broad, robust instance coverage; the default fallback for GEMM/grouped-GEMM/FMHA/MoE when no asm path matches. FlyDSL MoE falls back to CK when FlyDSL isn't installed.
3. **Triton** — portable path (e.g. `aiter.ops.triton.mla_decode`), used when neither asm nor a compiled CK instance covers the shape.
4. **HIP/Opus C++** — single-header `opus.hpp` templated HIP for fused glue kernels (sorting, sparse prefill).

Selection is driven by per-op config tables. Examples of config dispatch knobs:

| Mechanism | Effect |
|---|---|
| `aiter/configs/*.csv` tuned tables | maps (arch, dtype, M/N/K bucket) → best instance / tile / num_stages |
| `AITER_USE_SYSTEM_TRITON=1` | use your Triton, not the pinned one |
| `AITER_LOG_MORE=1` | print dispatch/JIT decisions |
| FlyDSL present? | A4W4 MoE uses FlyDSL; else CK |
| arch == gfx950 | LDS-aware `num_stages` selection, new asm fmoe kernels |

> There is no single "force CK / force asm" master env var across all ops; backend choice is per-op and shape-driven. To force a specific path, you usually call the backend-specific symbol directly (e.g. `aiter.ops.triton.mla_decode(...)` vs `aiter.mla.mla_decode_fwd(...)`).

---

## 5. Python API — concrete call examples

### 5.1 Dense GEMM (`tgemm.mm`) — drop-in Linear

```python
import torch
from aiter.tuned_gemm import tgemm

class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features).cuda())
        self.bias   = torch.nn.Parameter(torch.randn(out_features).cuda())

    def forward(self, x):
        # tgemm.mm(input, weight, bias, scale_a, scale_b)
        return tgemm.mm(x.cuda(), self.weight, self.bias, None, None)
```

`tgemm` autotunes across hipBLASLt and rocBLAS solutions and caches the winner per shape (analogous to PyTorch TunableOp, but inside AITER).

### 5.2 Quantized GEMM (A8W8, FP8/INT8)

```python
import aiter
# x_q: int8/fp8 activations [M,K]; w_q: quantized weights [N,K]
# x_scale: per-token [M,1]; w_scale: per-channel [1,N]
out = aiter.gemm_a8w8(x_q, w_q, x_scale, w_scale, bias=None, dtype=torch.bfloat16)
```

Dequant + scaling + bias are fused into the GEMM epilogue. FP8 on gfx942 uses E4M3FNUZ.

### 5.3 Fused MoE

```python
from aiter.fused_moe import fused_moe
# hidden_states: [num_tokens, hidden]
# w1: [num_experts, 2*inter, hidden]  (gate+up)   w2: [num_experts, hidden, inter] (down)
# topk_weights/topk_ids from router top-k
out = fused_moe(
    hidden_states, w1, w2,
    topk_weights, topk_ids,
    inplace=True,
    # quant args route to the right kernel:
    #   fp8 block-scale -> block-scaled fused MoE (asm/CK)
    #   a4w4            -> FlyDSL (or CK fallback)
)
```

`fused_moe` inspects the quantization method and auto-selects the MoE kernel (bf16 asm / fp8 block-scale / A4W4 FlyDSL / CK). It internally does token sorting (`moe_sorting`), grouped GEMM, activation, and weighted combine in a fused pipeline → up to **3x** vs an unfused stack on MI300X.

### 5.4 MLA decode (DeepSeek MLA)

```python
from aiter.mla import mla_decode_fwd
import torch

batch_size, kv_seqlen, q_seqlen = 128, 4096, 1
num_heads, kv_lora_rank, qk_rope_head_dim = 128, 512, 64
q_head_dim = 128

qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
kv_indptr[1:] = torch.cumsum(torch.full((batch_size,), kv_seqlen, dtype=torch.int, device="cuda"), 0)
qo_indptr[1:] = torch.cumsum(torch.full((batch_size,), q_seqlen,  dtype=torch.int, device="cuda"), 0)
kv_indices = torch.randint(0, 2_097_152, (kv_indptr[-1].item(),), dtype=torch.int, device="cuda")
kv_last_page_lens = torch.ones(batch_size, dtype=torch.int, device="cuda")

q = torch.randn((batch_size * q_seqlen, num_heads, kv_lora_rank + qk_rope_head_dim),
                dtype=torch.bfloat16, device="cuda")
kv_buffer = torch.randn((2_097_152, 1, 1, kv_lora_rank + qk_rope_head_dim),  # page_size=1, num_heads_kv=1
                        dtype=torch.bfloat16, device="cuda")
o = torch.empty((batch_size * q_seqlen, num_heads, kv_lora_rank),
                dtype=torch.bfloat16, device="cuda")

mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
               kv_last_page_lens, max_seqlen_q=1,
               sm_scale=1.0 / (q_head_dim ** 0.5))
```

**Signature:**

```python
def mla_decode_fwd(
    q,                 # [B, num_heads, kv_lora_rank + qk_rope_dim]
    kv_buffer,         # [num_pages, page_size, num_heads_kv(=1), qk_head_dim]
    o,                 # output [B, num_heads, kv_lora_rank]
    qo_indptr,         # [B+1]
    kv_indptr,         # [B+1]
    kv_indices,        # [kv_indptr[-1]]
    kv_last_page_lens, # [B]
    max_seqlen_q,
    sm_scale=None,     # default 1/sqrt(qk_head_dim)
    logit_cap=0.0,     # WIP, ignore
    num_kv_splits=None # auto heuristic, ignore
):
```

In decode, `num_heads_kv == 1` and `page_size == 1` uses the original (unpaged) representation. Up to **17x** vs naive decode.

### 5.5 MHA prefill (flash)

```python
import aiter
out = aiter.flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
```

### 5.6 Norm / RoPE / fused ops

```python
import aiter
y = aiter.rms_norm(x, weight, eps=1e-6)
y = aiter.layernorm2d_with_add_asm(x, residual, weight, bias, eps)   # fused residual-add + LN
q2, k2 = aiter.rope_fwd(q, k, cos, sin)
```

Heavier fusions are exposed via `aiter.ops.*`:
- `aiter.ops.gated_rmsnorm_fp8_group_quant` — RMSNorm → gating → FP8 group quant in one kernel.
- `aiter.ops.fused_qk_norm_rope_cache_quant` — QK-norm + RoPE + KV-cache write + quant fused (the attention pre-processing chain in one launch).

---

## 6. Framework integration (how vLLM / SGLang turn AITER on)

**vLLM (V1):** single master switch

```bash
export VLLM_ROCM_USE_AITER=1          # enables AITER for GEMM, RMSNorm, MoE, attention
# still required even when forcing an attention backend:
vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8 --trust-remote-code
```

Finer-grained vLLM flags (DeepSeek-V3 reference run):

```bash
VLLM_MLA_DISABLE=0 VLLM_USE_TRITON_FLASH_ATTN=0 VLLM_USE_ROCM_FP8_FLASH_ATTN=0 \
VLLM_FP8_PADDING=1 VLLM_USE_AITER_MOE=1 VLLM_USE_AITER_BLOCK_GEMM=1 \
VLLM_USE_AITER_MLA=0 vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8 --trust-remote-code
```

> `--attention-backend` overrides only the attention kernel; `VLLM_ROCM_USE_AITER=1` is still needed for GEMM/RMSNorm/MoE.
> `VLLM_ROCM_USE_AITER_HIP_ONLINE_TUNING=1` enables runtime hipBLASLt online tuning inside the AITER GEMM wrapper.

**SGLang (default on ROCm Docker):**

```bash
SGLANG_ROCM_AITER_BLOCK_MOE=1 CK_BLOCK_GEMM=1 RCCL_MSCCL_ENABLE=0 \
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

SGLang's `apply_fp8_linear` routes AMD FP8 through the AITER path → `torch._scaled_mm` (hipBLASLt-backed); checkpoints are re-quantized to E4M3FNUZ for gfx942.

---

## 7. Op catalog (quick reference)

| Category | Op(s) | Backends | Notes / speedup |
|---|---|---|---|
| Dense GEMM | `tgemm.mm`, `gemm_a16w16` | hipBLASLt/rocBLAS/CK | autotuned wrapper |
| Quant GEMM | `gemm_a8w8`, `gemm_a4w4`, `deepgemm` | asm/CK/FlyDSL | fused scale+bias; FP8-FNUZ |
| Block-scale GEMM | block-scale GEMM | asm/CK | up to **2x** |
| Batched GEMM | `batched_gemm_a8w8`, `batched_gemm_bf16` | CK | |
| Fused MoE | `fused_moe`, block-scale fused MoE, dp_shared_expert | asm/FlyDSL/CK | up to **3x**; auto-select by quant |
| MoE support | `moe_sorting`, `topk` | HIP/Opus | token routing |
| MLA | `mla_decode_fwd`, `ops.triton.mla_decode`, MLA prefill | asm/Triton | decode up to **17x** |
| MHA | `flash_attn_func`, paged/decode attention | asm/CK/Triton | prefill up to **14x** |
| Sparse attn | `pa_sparse_prefill_opus`, sparse MLA (gfx950) | Opus/asm | newest models |
| Norm | `rms_norm`, `layer_norm`, `layernorm2d_with_add_asm`, `groupnorm` | asm/HIP | residual-add fusion |
| Norm+quant fused | `gated_rmsnorm_fp8_group_quant`, `fused_qk_rmsnorm_group_quant` | HIP/asm | one kernel |
| RoPE | `rope_fwd/bwd`, `fused_qk_norm_rope_cache_quant` | HIP/asm | KV-write fusion |
| Activation | `activation` (SiLU/GELU) | HIP | |
| Quant helpers | `quant` (per-token/block fp8/int8), `int4_utils` | HIP | |
| Sampling | `sample`, `sampling`, `topk` | HIP/Triton | |
| Collectives | `custom_all_reduce`, `quick_all_reduce`, all-gather/reduce-scatter | HIP/Iris | RCCL bypass |
| SSM/linear-attn | `causal_conv1d`, `chunk_gated_delta_rule_fwd_h` | Triton | |

---

## 8. Testing / verifying a path exists for your shape

Per-op tests double as usage examples and as a way to confirm a tuned path:

```bash
python3 op_tests/test_gemm_a8w8.py
python3 op_tests/test_moe.py
python3 op_tests/test_mla.py
python3 op_tests/test_mha.py
python3 op_tests/test_rmsnorm2d.py
ls op_tests/test_*.py          # full list
```

Workflow tip for serving: run a short warmup with `AITER_LOG_MORE=1`, confirm hot GEMM/MoE/attention shapes hit asm/CK (not Triton fallback), and check `aiter/configs/` has a matching tuned config for gfx942. If a hot shape falls back to Triton, either add a tuned config or pad shapes to a covered bucket.

---

## Sources
- AITER repository (README, `aiter/` and `aiter/ops/` source dirs): https://github.com/ROCm/aiter
- AITER releases / changelog (v0.1.12–v0.1.15, 2026): https://github.com/ROCm/aiter/releases
- AMD ROCm Blog, "AITER: AI Tensor Engine For ROCm": https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- AMD ROCm Blog, "AITER-Enabled MLA Layer Inference on AMD Instinct MI300X": https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- AMD AI Developer Hub, "MLA decoding kernel of the AITER library" (`mla_decode_fwd` signature/example): https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html
- ROCm vLLM V1 performance optimization (VLLM_ROCM_USE_AITER): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
