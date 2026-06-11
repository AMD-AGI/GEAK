---
title: rmsnorm on aiter — SOTA card
kind: sota_card
operator: rmsnorm
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-09
sources:
  - ROCm/aiter@a6bb4993:aiter/ops/rmsnorm.py
  - ROCm/aiter@a6bb4993:aiter/ops/triton/normalization/rmsnorm.py
  - ROCm/aiter@a6bb4993:csrc/kernels/rmsnorm_kernels.cu
  - https://github.com/vllm-project/vllm/pull/14959
  - https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html
  - https://docs.vllm.ai/en/latest/design/fusions/
  - https://github.com/ROCm/aiter/releases
---

# rmsnorm × aiter

## TL;DR
On sglang/vLLM with `VLLM_ROCM_USE_AITER=1`, **aiter is the live RMSNorm path** and the SOTA choice on
gfx942/gfx950. It ships *three* implementation tiers behind one Python dispatcher — hand-tuned **CK**
(`rmsnorm2d_fwd_ck`), an **asm/HIP** module (`module_rmsnorm` / `module_rmsnorm_quant`), and a portable
**Triton** impl (`ops/triton/normalization/rmsnorm.py`). RMSNorm is **bandwidth-bound** (one row read,
one row write, a tiny fp32 reduction), so the only levers that matter are 128-bit loads, an LDS-resident
reduction, and **fusion** (don't pay a second HBM round-trip). On the serving path you almost never call
plain `rms_norm`; you call a fused variant (`_with_add`, `_with_dynamicquant`). Choose this card's
backend unless you need a fused shape the library lacks → then [triton.md](triton.md).

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| `rms_norm` / `rmsnorm2d_fwd` (asm + CK dispatch) | `aiter/ops/rmsnorm.py` (`module_rmsnorm`, `rmsnorm2d_fwd_ck`) | gfx942/950, bf16/fp16 | bandwidth-bound; achievable floor ≈ `(read+write bytes) / ~3.5 TB/s` effective HBM @ MI300X | standalone norm (rare on serving) |
| `rmsnorm2d_fwd_with_add` (asm `add_rmsnorm` / CK) | same (`module_rmsnorm_quant`, `_with_add_ck`) | gfx942/950, bf16/fp16 | live form via `VLLM_ROCM_USE_AITER_RMSNORM=1` (vLLM #14959) | block input / post-attn residual norm (**the live form**) |
| `rmsnorm2d_fwd_with_dynamicquant` / `add_rmsnorm_quant` / `gated_rmsnorm_fp8_group_quant` | `aiter/ops/rmsnorm.py` (`module_rmsnorm_quant`), `gated_rmsnorm_fp8_group_quant.py` | gfx942/950, fp8 out | part of SGLang's Qwen3 norm+quant fusion; **+1–6% e2e** (label-sourced, #18466) | norm feeding fp8 GEMM → [[fused_norm_quant]] |
| Triton fallback `aiter.ops.triton...rms_norm` | `aiter/ops/triton/normalization/rmsnorm.py` | gfx942/950, bf16/fp16/fp8 | persistent grid `min(rows,num_sms)`; two-pass blocked when `N > 65536/elt` | no asm/CK tune for the shape / portability |

### What the SOTA kernel actually does (on-box C++)
The asm/HIP forward (`csrc/kernels/rmsnorm_kernels.cu`) is the textbook bandwidth-bound recipe:
**128-bit vectorized loads** (`vec8_t<scalar_t>` = 8×bf16/fp16 = 16 B), **fp32 accumulate**, and a single
**LDS block-reduce** for `Σx²`:

```cpp
// ROCm/aiter@a6bb4993:csrc/kernels/rmsnorm_kernels.cu
__shared__ float s_variance;
vec8_t<scalar_t> v8_variance = {0,0,0,0,0,0,0,0};
for (int idx = threadIdx.x; idx < hidden_size/8; idx += blockDim.x) {
  vec8_t<scalar_t> x = ...;      // 128-bit global_load_dwordx4
  v8_variance += x * x;          // accumulated in fp32 inside vec8_t
}
float v8_variance_sum = v8_variance.sum();
using BlockReduce = hipcub::BlockReduce<float, 1024>;        // LDS reduction
float variance = BlockReduce(reduceStore).Reduce(v8_variance_sum, hipcub::Sum{}, blockDim.x);
if (threadIdx.x == 0) s_variance = rsqrtf(variance / hidden_size + epsilon);
__syncthreads();
// second sweep: out = (x * s_variance) * weight  (vectorized store)
```
One block per token row, `Σx²` reduced once in LDS (no global atomics), `rsqrtf` once per row, then a
vectorized normalize+store. The whole op is two HBM streams; everything else is free.

### Python dispatch (the three-tier seam)
The Python entrypoints in `aiter/ops/rmsnorm.py` route **by hidden dim**, not by a global flag:
```python
# ROCm/aiter@a6bb4993:aiter/ops/rmsnorm.py
def rmsnorm2d_fwd(input, weight, epsilon, use_model_sensitive_rmsnorm=0):
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        out = rmsnorm2d_fwd_ck(input, weight, epsilon, use_model_sensitive_rmsnorm)  # CK tier
    else:
        out = torch.empty_like(input)
        rmsnorm(out, input, weight, epsilon)                                          # asm tier
    return out
```
So for the common LLM hidden dims (2048–8192) the **asm `module_rmsnorm_quant`** path runs;
**CK** (`rmsnorm2d_fwd_ck`) only takes over for `N > 8192` or when `use_model_sensitive_rmsnorm` is set
(a numerics knob — see below). Triton is the portable fallback used when neither asm nor CK has a tune.

## Config space / knobs
| knob | where | default / range | effect |
|---|---|---|---|
| `VLLM_ROCM_USE_AITER` | env | 0 | master gate; off ⇒ native vLLM kernel, none of this engages |
| `VLLM_ROCM_USE_AITER_RMSNORM` | env | 1 (when master on) | routes `RMSNorm.forward_hip` → `rmsnorm2d_fwd_with_add` |
| `use_model_sensitive_rmsnorm` | arg | 0 | forces CK tier; CK reduction is the "model-sensitive" (higher-parity) order |
| hidden dim N | data | — | `N>8192` ⇒ CK; `N≤8192` ⇒ asm; no tune ⇒ Triton |
| Triton `block_size` | derived | `min(65536//elt, next_pow2(N))` | single-pass vs two-pass blocked boundary |
| Triton `NUM_PRGMS` | derived | `min(rows, get_num_sms())` | persistent grid; one program sweeps strided rows |
| fusion entrypoint | call site | — | `_with_add` / `_with_dynamicquant` / `_with_smoothquant` — **the fusion is the tuning lever** |
| JIT warm | runtime | — | first call compiles `module_rmsnorm*` into `aiter/jit/`; warm before benching |

## Numerics / parity
- **fp32 accumulate**: `Σx²` is summed in fp32 (`vec8_t` accumulates in float; Triton casts `x.to(tl.float32)`
  before squaring). ε is added inside the mean: `rsqrtf(Σx²/N + ε)`.
- **Reduction order differs across tiers** (asm block-reduce vs CK vs Triton persistent sweep), so a
  backend/version swap can flip an argmax tie — **re-gate greedy parity** on any swap (aiter #1972 is a
  real instance of an SGLang↔aiter rmsnorm parity regression).
- **One-pass (this kernel) vs two-pass vs Welford**: RMSNorm only needs `Σx²` (no mean subtraction), so a
  single fp32 accumulator is numerically safe up to LLM hidden dims; there is no Welford here. The Triton
  tier switches to a **two-pass blocked** algorithm only when the row exceeds `65536/elt` elements
  (it can't hold the row in registers), not for accuracy.
- **γ dtype**: aiter promotes γ to fp32 before the multiply. Note the vLLM-native kernel had a regression
  (#42325, v0.20.0) where the *CUDA* RMSNorm always multiplied in fp32 ignoring weight dtype — aiter's
  path is unaffected, but it means aiter vs native parity can differ by the γ-multiply precision.
- **fp8 fused-quant output**: the `_dynamicquant` variants emit fp8 **fnuz** on gfx942 (off-by-2× trap if
  you treat it as OCP e4m3); gate with a task metric, not allclose. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- **vLLM**: `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_RMSNORM=1` (default on) →
  `vllm/model_executor/layers/layernorm.py` routes `RMSNorm.forward_hip` to `rmsnorm2d_fwd_with_add`
  (PR #14959). No site-packages edit needed; flip the env.
- **vLLM Inductor fusion passes** orchestrate the norm+quant fusion: the `rocm_aiter_fusion` RMSNorm+quant
  pass + `ActivationFusionPass` (**+8% throughput**). ⚠ Inductor-compiled torch-op quant can now **auto-fuse
  some patterns**, making the standalone SiLU+quant / RMSNorm+quant passes **obsolete except for custom-op
  cases** (attention, collectives, sub-byte quant) — check whether the pass is still needed before wiring it.
- **PTPC-FP8** (per-token-per-channel) fused quant is **up to 2.5× vs naive** on MI300X (vLLM PTPC-FP8 blog).
- **AITER v0.1.12** adds a fused **gated RMSNorm + group-quant** kernel (`gated_rmsnorm_fp8_group_quant`).
- **SGLang**: on by default with `SGLANG_USE_AITER=1`; norm routes through `rocm_linear_utils` / the fused
  norm+quant kernels.
- **Verify it engages**: `AITER_LOG_MORE=1` prints the `RMSNORM*` dispatch line; `rocprofv3` shows a CK/asm
  norm kernel name (e.g. an `*rmsnorm*` HIP kernel), **not** a Triton-mangled `_rms_norm_kernel` — a Triton
  name in the trace means no asm/CK tune matched the shape.

## Pitfalls & anti-patterns
- ⚠ Calling plain `rms_norm` where the block expects `_with_add` → the residual add becomes a separate
  kernel + extra HBM read/write (lose the fusion, ~2× the traffic). Match the framework's fused entrypoint.
- ⚠ fp8 fnuz/OCP dialect mismatch on gfx942 → 2× error in the quant variant (numerics).
- ⚠ A gfx942 shape with no asm/CK tune silently falls back to Triton (correct, sometimes slower) — confirm
  the kernel name in the trace.
- ⚠ Benching cold: first call JIT-compiles `module_rmsnorm*`; a cold bench reports compile time, not kernel
  time. Warm ≥3 iterations.
- ⚠ Assuming a global "USE_CK" flag — there isn't one for rmsnorm; the tier is chosen by `N>8192` and
  `use_model_sensitive_rmsnorm`.

## How to verify (bench + oracle)
- **Isolated**: `python3 op_tests/test_rmsnorm2d.py` at the model hidden dim (e.g. 4096, 8192) and a sweep
  of token counts; median of ≥3 warm repeats.
- **Bandwidth check**: compute `(in_bytes + out_bytes) / time` and compare to ~3.5 TB/s effective MI300X
  HBM; if you're far below, the wrong tier engaged or the shape is mask-bound.
- **e2e**: greedy parity vs the native path + `rocprofv3` norm-kernel name + `AITER_LOG_MORE=1` dispatch.

## Worked example (Llama-3-8B post-attn norm, MI300X)
Hidden `N=4096`, a decode batch of `M=256` rows, bf16. `N=4096 ≤ 8192` ⇒ the **asm** tier
(`module_rmsnorm_quant`) runs, not CK. Traffic: input `256·4096·2 = 2.0 MiB`, output `2.0 MiB`, γ `8 KiB`,
residual_out (if `_with_add`) another `2.0 MiB`. At ~3.5 TB/s effective the `_with_add` floor is
`6 MiB / 3.5 TB/s ≈ 1.8 µs`. The kernel is one block per row, 128-bit loads (`vec8_t`, 512 elts/8 = 64
vector loads per row), one LDS reduce. If you instead call `rms_norm` then a separate `+residual`, you add
a full 4 MiB round-trip (~1.1 µs) and a kernel-launch — that is exactly the fusion #14959 captures. For
`N=12288` (e.g. a wide MoE hidden) the dispatcher would route to **CK** instead.

## Alternatives / cross-links
[triton.md](triton.md) (portable tier + authoring) · [hip.md](hip.md) · [vllm_kernels.md](vllm_kernels.md) ·
[flydsl.md](flydsl.md) · [../overview.md](../overview.md) · [[fused_add_rmsnorm]] · [[fused_norm_quant]] ·
[[optimization/kernel_fusion_strategy]] · [[optimization/vectorization_and_coalescing]].

## Sources
- On-box dispatch + fused/CK/quant entrypoints (tier routing by `N>8192`):
  `ROCm/aiter@a6bb4993:aiter/ops/rmsnorm.py`.
- On-box asm/HIP forward (vec8 128-bit load, fp32 accumulate, `hipcub::BlockReduce` LDS reduction):
  `ROCm/aiter@a6bb4993:csrc/kernels/rmsnorm_kernels.cu`.
- Triton impl (persistent grid `min(rows,num_sms)`, two-pass blocked, `.cg`, `num_stages=2`):
  `ROCm/aiter@a6bb4993:aiter/ops/triton/normalization/rmsnorm.py` and
  `.../_triton_kernels/normalization/rmsnorm.py`.
- vLLM AITER RMSNorm integration (`forward_hip` → `rmsnorm2d_fwd_with_add`):
  https://github.com/vllm-project/vllm/pull/14959.
- Qwen3 norm+quant fusion tracking (e2e %, label-sourced): https://github.com/sgl-project/sglang/issues/18466.
- SGLang↔aiter rmsnorm parity regression precedent: https://github.com/ROCm/aiter/issues/1972.
- vLLM native RMSNorm γ-dtype regression (parity context, not aiter): https://github.com/vllm-project/vllm/issues/42325.
- PTPC-FP8 up to 2.5× vs naive (MI300X): https://blog.vllm.ai/2025/02/24/ptpc-fp8-rocm.html.
- vLLM Inductor fusion passes (ActivationFusionPass +8%; torch-op quant auto-fuse obsoletes some passes except custom-op): https://docs.vllm.ai/en/latest/design/fusions/.
- AITER v0.1.12 fused gated RMSNorm + group-quant: https://github.com/ROCm/aiter/releases.
