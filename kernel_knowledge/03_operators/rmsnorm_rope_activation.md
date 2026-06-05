# RMSNorm / RoPE / Activation Fused Kernels on AMD MI300X (CDNA3 / gfx942)

> AMD-only kernel knowledge. Target MI300X (CDNA3, gfx942); notes for MI355X
> (gfx950) where relevant. These are the **high-frequency, memory-bound** ops in
> every transformer layer: RMSNorm (+residual add, +fp8 quant), RoPE (+qk-norm),
> SiLU/GELU activation+mul (SwiGLU), LayerNorm. They contribute little FLOP but a
> LOT of HBM traffic and kernel launches. **Fusion is the entire game.** Every
> unfused op is a full read+write of the activation tensor (M×K elements) and a
> launch; on MI300X decode (bandwidth-bound) these add up fast.

---

## 1. Why these ops dominate the launch/bandwidth budget

A transformer layer at decode does, per token, roughly: input RMSNorm → QKV GEMM →
RoPE → attention → out GEMM → residual → post-attn RMSNorm → gate/up GEMM → SiLU·mul →
down GEMM → residual. The GEMMs are compute; **everything else is memory-bound
elementwise/reduction work.** Each unfused elementwise op is:

- 1 HBM read + 1 HBM write of the hidden state (M×K×2 bytes bf16), and
- 1 kernel launch (~few µs dispatch + occupancy ramp).

At small decode batch the launches alone can rival the math. So the rules are:
1. **Fuse the residual add into the norm** (you already read the hidden state).
2. **Fuse the quant into the norm / activation** (you already have the values in regs).
3. **Fuse RoPE + qk-norm + (optionally) kv-quant** into one pass over Q/K.
4. **Vectorize** loads (128-bit `float4` / `ds_read_b128`) — these ops are pure BW.
5. Keep the **reduction in FP32**; one workgroup per row (token) for RMSNorm.

---

## 2. RMSNorm fundamentals on CDNA3

RMSNorm: `y = x / sqrt(mean(x^2) + eps) * weight`. No mean-subtraction (cheaper than
LayerNorm). One row (token, K elements) per workgroup; reduce `x^2` across the row.

Memory-bound: arithmetic intensity ≈ a few FLOP/byte. Peak is HBM bandwidth
(5.3 TB/s). The kernel should be limited only by reading x once (and residual once if
fused) and writing y once.

CDNA3 specifics:
- **Wavefront = 64.** A K of 4096–8192 fits a few waves per workgroup; use a
  block-stride loop over K with a tree/`tl.sum` reduction.
- **FP32 accumulate** the sum-of-squares (bf16 sum loses precision and changes the
  norm). Promote x to fp32 for the reduction; cast back to bf16/fp8 on store.
- **LDS reduction across waves**: 64 KB LDS/CU; reduce per-wave partials in LDS.
- **Vectorize**: load 8 bf16 (128-bit) per thread to saturate HBM.

### Fusion table (RMSNorm-centric)

| Fused kernel | Fuses | HBM saved vs unfused | AITER / vLLM name |
|--------------|-------|----------------------|-------------------|
| **fused_add_rmsnorm** | residual add + RMSNorm (updates residual in-place) | 1 read + 1 write | `aiter.rmsnorm2d_fwd_with_add`, vLLM `fused_add_rms_norm` |
| **rmsnorm + fp8 quant** | RMSNorm + per-token fp8 cast | 1 write (fp8 not bf16) + downstream read | AITER "RMSNorm+quant"; vLLM `rms_quant_fusion` pass |
| **fused_add_rmsnorm + fp8 quant** | residual + RMSNorm + quant | 1 read + ~1.5 round-trips | vLLM `rocm_aiter_fusion` torch.compile pass (#26575 series) |
| **rmsnorm + block fp8 quant** | RMSNorm + 1×128 block quant (DeepSeek act) | round-trip | AITER layernorm/silu fp8-block-quant passes |
| **rmsnorm + pad** (gpt-oss) | residual + RMSNorm + hidden pad to AITER GEMM multiple | extra pad pass | ROCm-only, hidden=2880, O1+ |
| **allreduce + rmsnorm** | TP all-reduce + norm | comm/compute overlap | AITER "AR/AG fused with normalization", vLLM `#37646` |

---

## 3. Fused residual-add RMSNorm + FP8 quant (Triton, MI300X)

This is the canonical fused kernel: read x and residual once, add them (write the new
residual in-place for the next layer's skip connection), RMS-normalize in FP32, scale
by weight, then **per-token quantize to FP8 (E4M3-FNUZ)** in the same pass.

```python
import triton
import triton.language as tl

FP8_E4M3_FNUZ_MAX = 240.0   # MI300X native fp8 max (NOT 448). See quantization_fp8.md

@triton.jit
def fused_add_rmsnorm_quant_fp8(
    x_ptr,            # [M, K]  bf16 input (this layer's activation)
    res_ptr,          # [M, K]  bf16 residual IN/OUT (updated to x+res in place)
    w_ptr,            # [K]     bf16 rmsnorm weight (gamma)
    y_ptr,            # [M, K]  fp8 e4m3fnuz output (quantized normed activation)
    yscale_ptr,       # [M]     fp32 per-token scale
    M, K, eps,
    BLOCK_K: tl.constexpr,    # >= K (single-block) or tile if K large
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    xb = x_ptr + row * K
    rb = res_ptr + row * K

    # ---- load x and residual (one HBM read each), add, write residual back ----
    x   = tl.load(xb + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(rb + offs, mask=mask, other=0.0).to(tl.float32)
    h   = x + res                                  # new hidden = x + residual
    tl.store(rb + offs, h.to(tl.bfloat16), mask=mask)   # residual updated in place

    # ---- RMSNorm in FP32 ----
    var = tl.sum(h * h, axis=0) / K
    rms = tl.rsqrt(var + eps)
    g   = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    yf  = h * rms * g                              # normalized activation (fp32)

    # ---- per-token FP8 quant (FUSED: no extra HBM round-trip) ----
    amax  = tl.max(tl.abs(yf))
    scale = amax / FP8_MAX
    scale = tl.where(scale == 0.0, 1.0, scale)
    tl.store(yscale_ptr + row, scale)
    q = tl.clamp(yf / scale, -FP8_MAX, FP8_MAX)
    # AMD: only FP32->FP8 HW conversion exists; yf is already fp32 here -> safe cast.
    tl.store(y_ptr + row * K + offs, q.to(y_ptr.dtype.element_ty), mask=mask)
    # y_ptr.dtype.element_ty == tl.float8e4b8 (E4M3-FNUZ) for MI300X native MFMA.
```

Why this is the right shape on MI300X:
- **Residual updated in place** — the next layer reads it directly; no extra tensor.
- **All math in FP32** until the final fp8 store. CDNA3 has *no* bf16→fp8 hardware
  conversion; the source of the fp8 cast must be FP32. Computing the norm in fp32 and
  casting at the end satisfies this for free.
- **One read of x, one read of residual, one write of residual, one fp8 write.** The
  scale is a tiny [M] tensor. Versus the unfused chain (add kernel + norm kernel +
  quant kernel = 3 launches, ~5 round-trips) this is **1 launch, ~2 round-trips.**
- For **block (1×128) quant** (DeepSeek act), replace the single per-row amax/scale
  with per-128-K-tile amax → emit `[M, K/128]` scales (loop K in 128-chunks). The
  output then feeds a `w8a8_block_fp8_matmul` (see `quantization_fp8.md`).
- For large K (> a single block), do a two-pass variant: pass 1 reduces `h*h` (and
  writes the residual), pass 2 normalizes+quantizes; or use a persistent reduction
  with LDS across waves.

> AITER provides this natively: `rmsnorm2d_fwd_with_add` (+quant variants). On ROCm,
> prefer the library kernel (`VLLM_ROCM_USE_AITER=1`,
> `VLLM_ROCM_USE_AITER_RMSNORM=1`); the older `*_ck` (Composable Kernel) variant is
> the safe fallback — a newer `add_rmsnorm` variant had a chunked-prefill consistency
> bug (aiter #1972), so validate numerics if you switch variants.

---

## 4. RoPE (+ QK-norm) fused kernel

RoPE rotates Q and K by position-dependent angles in interleaved/half-split pairs.
It's elementwise over the head dimension and reads Q,K once. Fuse with **qk-norm**
(per-head RMSNorm of Q and K, used by many recent models) and optionally **kv-quant**.

### RoPE math

For head-dim pair `(x_{2i}, x_{2i+1})` (interleaved) or `(x_i, x_{i+d/2})` (half-split,
the GPT-NeoX/Llama layout), with `θ_i = pos * base^{-2i/d}`:
```
out_a =  x_a * cos(θ) - x_b * sin(θ)
out_b =  x_a * sin(θ) + x_b * cos(θ)
```

```python
@triton.jit
def fused_qknorm_rope_kernel(
    q_ptr, k_ptr,            # [T, H, D] bf16 (T tokens, H heads, D head-dim)
    qn_w_ptr, kn_w_ptr,      # [D] per-head qk-norm weights (optional)
    cos_ptr, sin_ptr,        # [T, D//2] precomputed rotary tables
    T, H, D, eps,
    QK_NORM: tl.constexpr,
    HALF: tl.constexpr,      # half-split (Llama/NeoX) vs interleaved
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0); h = tl.program_id(1)
    base = (t * H + h) * D
    d   = tl.arange(0, BLOCK_D)
    half = D // 2

    q = tl.load(q_ptr + base + d, mask=d < D, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + base + d, mask=d < D, other=0.0).to(tl.float32)

    # ---- optional QK-RMSNorm per head (FP32) ----
    if QK_NORM:
        qv = tl.sum(q * q) / D; q = q * tl.rsqrt(qv + eps) * \
             tl.load(qn_w_ptr + d, mask=d < D, other=0.0).to(tl.float32)
        kv = tl.sum(k * k) / D; k = k * tl.rsqrt(kv + eps) * \
             tl.load(kn_w_ptr + d, mask=d < D, other=0.0).to(tl.float32)

    # ---- RoPE (half-split layout shown) ----
    cos = tl.load(cos_ptr + t * half + (d % half), mask=d < D, other=1.0)
    sin = tl.load(sin_ptr + t * half + (d % half), mask=d < D, other=0.0)
    # gather the rotation partner: x_{i+half} for i<half, x_{i-half} for i>=half
    is_lo = d < half
    partner = tl.where(is_lo, d + half, d - half)
    qp = tl.load(q_ptr + base + partner, mask=d < D, other=0.0).to(tl.float32)
    kp = tl.load(k_ptr + base + partner, mask=d < D, other=0.0).to(tl.float32)
    # for the lower half: out = x*cos - partner*sin ; upper half: out = x*cos + partner*sin
    rot_sign = tl.where(is_lo, -1.0, 1.0)
    q_out = q * cos + rot_sign * qp * sin
    k_out = k * cos + rot_sign * kp * sin

    tl.store(q_ptr + base + d, q_out.to(tl.bfloat16), mask=d < D)
    tl.store(k_ptr + base + d, k_out.to(tl.bfloat16), mask=d < D)
    # (optional: quantize k_out to fp8 here and write to the kv-cache directly)
```

AMD/AITER notes:
- AITER: `aiter.rope_fwd()` / `rope_bwd()`; fused `fused_qk_rmsnorm`,
  `fused_qk_norm_mrope_3d`, `deepseek_scaling_rope`, and RoPE+KV-cache-write fusion.
- **Precision caveat:** SGLang's aiter fused RoPE backend is lower precision; disable
  with `USE_ROCM_AITER_ROPE_BACKEND=0` if accuracy matters. When writing your own,
  keep the rotation in FP32 (the table cos/sin can be bf16 but the multiply-add in
  fp32) to avoid drift.
- **Fuse the KV-cache fp8 quant**: after computing `k_out` you already have it in
  registers — quantize and write the fp8 byte to the paged KV cache in the same pass,
  saving a separate quant kernel. (See `quantization_fp8.md` §7 for KV layout.)
- Precompute cos/sin tables once; don't recompute trig per token.
- `mrope` (multimodal RoPE, 3-axis: temporal/height/width) is the
  `fused_qk_norm_mrope_3d` variant for VLMs.

---

## 5. Activation + mul (SwiGLU) fused

SwiGLU MLP: `down( SiLU(gate(x)) * up(x) )`. The `SiLU(g) * u` step (act_and_mul) is
elementwise over the gate/up GEMM outputs. Fuse it — and optionally the fp8 quant for
the down-projection input — into one pass.

SiLU: `silu(x) = x * sigmoid(x) = x / (1 + e^{-x})`.
GELU (tanh approx): `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.

```python
@triton.jit
def silu_mul_quant_fp8(
    gu_ptr,           # [M, 2N] bf16: concat of gate (first N) and up (last N)
    y_ptr,            # [M, N]  fp8 e4m3fnuz output (input to down-proj)
    yscale_ptr,       # [M] fp32 per-token scale
    M, N,
    BLOCK_N: tl.constexpr, FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); mask = offs < N
    g = tl.load(gu_ptr + row*2*N + offs,      mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row*2*N + N + offs,  mask=mask, other=0.0).to(tl.float32)
    silu = g * tl.sigmoid(g)
    act  = silu * u                           # SwiGLU activation (fp32)
    # fused per-token fp8 quant
    amax = tl.max(tl.abs(act)); scale = amax / FP8_MAX
    scale = tl.where(scale == 0.0, 1.0, scale)
    tl.store(yscale_ptr + row, scale)
    q = tl.clamp(act / scale, -FP8_MAX, FP8_MAX)
    tl.store(y_ptr + row*N + offs, q.to(y_ptr.dtype.element_ty), mask=mask)
```

Notes:
- Reading gate+up from a **single fused [M,2N] tensor** (the gate/up GEMM writes them
  contiguously) is one read; computing SiLU·mul and quantizing in one pass means the
  down-proj reads fp8 directly. Unfused = act kernel + quant kernel + 2 round-trips.
- **For MoE batched experts**: vLLM's `persistent_masked_m_silu_mul_quant` fuses
  SiLU+mul+per-group(128) fp8 quant. On ROCm the C++ kernel was a no-op
  (`#ifndef USE_ROCM`) so vLLM falls back to a **Triton** kernel — correct but less
  optimal; a dedicated HIP kernel is tracked in aiter #2420. If you write MoE kernels,
  this fusion is a concrete win to claim.
- **GELU/act epilogue in GEMM**: simple act (no second operand) can ride the GEMM
  epilogue (CK/hipBLASLt bias+activation). act_and_mul needs the second (up) tensor,
  so it's usually a standalone fused kernel rather than a pure epilogue.

---

## 6. LayerNorm (when used)

`y = (x - mean) / sqrt(var + eps) * weight + bias`. Same structure as RMSNorm but with
mean subtraction (two reductions: mean, then var). FP32 reductions, fuse residual add
(`aiter.layernorm2d_with_add_asm`) and quant identically. Most modern LLMs use RMSNorm;
LayerNorm appears in some VLM vision towers / older models.

---

## 7. Fusion-opportunity master table

| Op chain | Naive launches | Naive round-trips | Fused into | Fused launches |
|----------|----------------|-------------------|------------|----------------|
| residual add → RMSNorm → fp8 quant | 3 | ~5 | `fused_add_rmsnorm_quant` | 1 |
| RMSNorm → QKV GEMM | 2 | 2 | RMSNorm + quant, GEMM reads fp8 | 2 (1 BW saved) |
| QKV GEMM → RoPE → qk-norm → KV fp8 | 4 | 4 | `fused_qk_rmsnorm` + RoPE + kv-quant | 1 |
| gate/up GEMM → SiLU·mul → fp8 quant | 3 | 3 | `silu_mul_quant_fp8` | 1 |
| TP all-reduce → RMSNorm | 2 | comm+rd/wr | AR+RMSNorm fused | 1 (overlap) |
| GEMM → bias → GELU | 3 | 2 | GEMM epilogue (CK/hipBLASLt) | 1 |

**Priority on MI300X (decode, bandwidth-bound):** the residual+RMSNorm+quant and
SiLU·mul+quant fusions give the biggest, most reliable wins because they kill whole
HBM round-trips on the largest (M×K) tensors and remove launches at small batch.

---

## 8. Tuning knobs

| Knob | Effect on these kernels |
|------|-------------------------|
| **Vector width** (`float4`/128-bit load) | these are pure BW → use widest aligned loads; `ds_read_b128` if staging in LDS |
| **One workgroup per row (token)** | natural for RMSNorm/LayerNorm row reduction; keeps reduction in LDS |
| **FP32 accumulation** | mandatory for the sum-of-squares / mean / var; never bf16 |
| **`waves_per_eu` / num_warps** | enough waves to hide HBM latency; these are latency-bound at small M |
| **BLOCK_K = K (single block)** | if K (e.g. 4096–8192) fits regs/LDS, avoid two-pass; else two-pass reduction |
| **In-place residual** | write updated residual back to its own buffer; no extra tensor |
| **Fuse the cast to FP32 once** | promote on load, do all math fp32, single cast on store (also satisfies CDNA3 fp32→fp8 rule) |
| **AITER env flags** | `VLLM_ROCM_USE_AITER=1` (master), `_AITER_RMSNORM=1`; `USE_ROCM_AITER_ROPE_BACKEND=0` if RoPE precision matters |

---

## 9. Quick checklist

1. These ops are **memory-bound** — target HBM bandwidth, vectorize loads, minimize
   round-trips and launches.
2. **Fuse residual add + RMSNorm + fp8 quant** into one kernel (in-place residual).
3. **Reductions in FP32**; cast to bf16/fp8 only on store.
4. **CDNA3 has no bf16→fp8 HW cast** — keep the value in FP32 right up to the fp8 store.
5. Use **E4M3-FNUZ** (`tl.float8e4b8`, max 240) for MI300X-native fp8 output.
6. **Fuse RoPE + qk-norm + KV fp8 quant**; keep rotation math in FP32.
7. **Fuse SiLU·mul + fp8 quant** for SwiGLU (and MoE expert outputs).
8. Prefer **AITER library kernels** on ROCm (`VLLM_ROCM_USE_AITER=1`); CK variant is
   the safe fallback when a newer ASM/Triton variant has numeric regressions.
9. For **block (1×128) act quant**, emit `[M, K/128]` scales to feed block GEMM.
10. Watch the aiter RoPE precision flag and the ROCm MoE SiLU+quant Triton fallback —
    both are known gaps you can improve.

---

## Sources

- ROCm Blog — AITER: AI Tensor Engine for ROCm (rms_norm, rope_fwd/bwd, layernorm2d_with_add_asm, RMSNorm+quant/shortcut, AR+norm fusion): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- ROCm/aiter source (fused rmsnorm/rope/quant/activation kernels): https://github.com/ROCm/aiter
- HuggingFace — "Creating custom kernels for the AMD MI300" (fused residual+RMSNorm+FP8, the FP32→FP8 conversion constraint, Llama 3.1 405B): https://huggingface.co/blog/mi300kernels
- vLLM [FEAT][ROCm] AITER RMS Norm PR #14959 (rmsnorm2d_fwd_with_add integration, env flags): https://github.com/vllm-project/vllm/pull/14959
- vLLM fusion passes docs (fused_add_rms_norm + quant, rms_quant_fusion, rocm_aiter_fusion): https://docs.vllm.ai/en/latest/design/fusions/
- vLLM DeepSeek-V3 ROCm uplift (AITER RMSNorm quant fusion pass, silu+fp8 block quant): https://github.com/vllm-project/vllm/issues/26768
- aiter #2420 — Fused SiLU+Mul+FP8-Quantize for batched MoE on ROCm: https://github.com/ROCm/aiter/issues/2420
- aiter #1972 — add_rmsnorm chunked-prefill consistency vs rmsnorm2d_fwd_with_add_ck: https://github.com/ROCm/aiter/issues/1972
- vLLM ROCm V1 optimization guide (AITER env flags, RMSNorm/RoPE/MoE enablement): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
