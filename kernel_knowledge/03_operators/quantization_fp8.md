# FP8 on AMD MI300X (CDNA3 / gfx942)

> AMD-only kernel knowledge. Target: MI300X (CDNA3, gfx942). Contrast notes for
> NVIDIA Hopper/Blackwell are clearly marked. gfx950/MI355X (CDNA4) is mentioned
> where it changes the FP8 story. Everything here is oriented to writing
> SUPER-HIGH-QUALITY FP8 GEMM / quant / attention kernels for LLM inference.

FP8 is the workhorse low-precision format for MI300X inference: weights, activations,
and KV cache. Decode is memory-bound, so FP8 halves the bytes you move vs BF16;
prefill/GEMM is compute-bound, so FP8 doubles matrix-core throughput. The two
things you MUST get right on AMD are (1) the **FNUZ vs OCP** format split and
(2) **fusing the quant into the producer kernel** (RMSNorm / activation / GEMM
epilogue) so you never pay an extra HBM round-trip.

---

## 1. The FP8 formats — and the AMD FNUZ trap

FP8 is an 8-bit float. Two element encodings matter for inference:

| Format | Sign | Exp | Mant | Max normal | Min normal | Min subnormal | Special values | Typical use |
|--------|------|-----|------|-----------|-----------|--------------|----------------|-------------|
| **E4M3** | 1 | 4 | 3 | 448 (OCP) / 240 (FNUZ) | 2^-6 | 2^-9 | OCP: ±Inf? **no**, NaN=S.1111.111 | weights, activations, KV (decode) |
| **E5M2** | 1 | 5 | 2 | 57344 | 2^-14 | 2^-16 | Inf, NaN (IEEE-like) | gradients, KV (wider range), error-tolerant |

E4M3 trades range for precision (3 mantissa bits → finer steps); E5M2 trades
precision for range (5 exponent bits → matches BF16 range). For **inference** the
near-universal choice is **E4M3** for weights+activations; E5M2 is sometimes used
for KV cache when the range of cached K/V is wide.

### The critical AMD detail: OCP vs FNUZ

There are TWO incompatible E4M3/E5M2 variants in ROCm:

| Variant | dtype name (ROCm) | Negative-zero / NaN handling | Max E4M3 | Native HW |
|---------|-------------------|------------------------------|----------|-----------|
| **OCP** ("ocp") | `e4m3` / `e5m2` (torch `float8_e4m3fn`) | has ±0, Inf, NaN; E4M3 max = 448 | 448 | **gfx950/MI355X** native; gfx942 emulated |
| **FNUZ** ("Finite-and-NaN-Unsigned-Zero") | `e4m3fnuz` / `e5m2fnuz` | only one zero, only NaN (no Inf), exponent bias shifted by 1 | 240 | **gfx942/MI300X native** |

**MI300X (CDNA3) matrix cores natively run FNUZ.** The OCP `float8_e4m3fn`
checkpoints that ship from Hugging Face / vLLM are converted on load. Practical
consequences:

- E4M3-FNUZ max representable is **240**, not 448. Exponent bias is 8 (vs 7 for
  OCP E4M3). A naive reinterpret-cast of OCP bits to FNUZ is WRONG — values differ
  by a factor of 2 because of the bias shift.
- `torch.float8_e4m3fnuz` / `torch.float8_e5m2fnuz` are the MI300X-native dtypes.
  vLLM/SGLang detect `current_platform.is_fp8_fnuz()` and rescale weights +
  weight_scale by 2 on load (because FNUZ has one extra exponent bit of bias,
  the stored scale must compensate). Get this wrong and you get garbage / NaNs.
- ROCm 6.3+ supports BOTH OCP and FNUZ dtypes in software, but **CDNA3 matrix
  cores only accelerate FNUZ**. CDNA4/gfx950 adds native OCP fp8 (and packed MXFP8).
- `rocBLAS` deprecated 8-bit GEMM. Use **hipBLASLt** (or aiter / CK / Triton).

> Contrast (NVIDIA): Hopper/Blackwell use OCP `e4m3`/`e5m2` (max 448) natively.
> Porting an H100 FP8 kernel to MI300X without handling FNUZ = silent accuracy loss.

---

## 2. MI300X FP8 matrix-core throughput

| Metric | MI300X (gfx942) | MI355X (gfx950, contrast) |
|--------|-----------------|---------------------------|
| Peak FP8 matrix (dense) | ~2.61 PFLOP/s theoretical (2100 MHz × 304 CU × 4096 FLOP/CU/cyc); ~1307 TFLOP/s "marketing" w/o sparsity convention varies | ~10 PFLOP/s FP8 |
| Peak FP8 vs BF16 | 2× BF16 | 2× BF16 |
| HBM bandwidth | 5.3 TB/s (192 GB HBM3) | 8 TB/s (288 GB HBM3E) |
| Tuned microkernel efficiency | 81–92% of peak (aligned shapes) | ~85–92% |
| Sustained system-level (real inference) | ~45–50% of peak (ROCm stack, power/clock throttling) | higher |

**Why the peak↔sustained gap is bigger on MI300X than H100:** for the 1–128 MB
working-set sizes typical of FP8 weights + KV cache, MI300X effective bandwidth
is below NVIDIA's, and decode is bandwidth-bound. Compiler/scheduler maturity also
costs you. The lesson for kernel authors: **the win is real but you must tune** —
do not assume hipBLASLt default heuristics hit peak.

### MFMA instruction selection (occupancy is everything)

FP8 MFMA shapes on CDNA3:

| Instruction | A×B×K | C tile / wave | Notes |
|-------------|-------|---------------|-------|
| `mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | 16×16 | **preferred** for occupancy |
| `mfma_f32_32x32x16_fp8_fp8` | 32×32×16 | 32×32 | fewer, bigger blocks → low CU occupancy on small/medium GEMM |

Empirical rule from the GPU-MODE AMD FP8 challenge: using **16×16×32** to compute a
64×64 output tile (16 waves / 1024 threads / block) splits the GEMM into ~4× more
blocks than 32×32×16 computing 128×128, so you actually saturate the 304 CUs. With
32×32×16→128×128 a typical M,N can launch only ~16 blocks → ~16/304 CU utilization.
**On MI300X, prefer smaller MFMA tiles for FP8 GEMM unless M,N,K are all huge.**

> CDNA4 adds `V_MFMA_SCALE_F32_16X16X128_F8F6F4` and
> `V_MFMA_SCALE_F32_32X32X64_F8F6F4` — block-scaled MFMA where the scale is applied
> inside the matrix core. CDNA3 has **no** scale-in-core instruction; you apply
> scales in the epilogue (see §5).

---

## 3. Scaling granularities

The whole game in FP8 inference is choosing how many elements share one FP32 scale.
Finer = more accurate (handles outliers) but more scale storage + more epilogue work.

| Granularity | Scale shape (for A: M×K, W: N×K) | Accuracy | Cost | Where used |
|-------------|----------------------------------|----------|------|------------|
| **Per-tensor** | 1 scalar | lowest | cheapest; one `fp32` multiply in epilogue | static fp8, simple models |
| **Per-channel / per-column** (weights) | N (one per output channel) | good for weights | cheap | W8A8 weights |
| **Per-token** (activations) | M (one per row/token) | good for activations | one scale per token, computed online | dynamic act quant |
| **Block 128×128** (weights) | ⌈N/128⌉×⌈K/128⌉ | best (outlier-robust) | scales along K → must fuse into mainloop | DeepSeek-V3, all big MoE |
| **Tile 1×128** (activations) | M×⌈K/128⌉ | best | scales along K | DeepSeek-V3 activations |

### Dynamic vs static

- **Static (delayed) scaling**: scale is a calibrated constant baked at load time
  (per-tensor or per-channel). No reduction at runtime; just multiply. Cheapest,
  but needs calibration and can clip on activation outliers.
- **Dynamic (online) scaling**: compute `scale = amax(x) / FP8_MAX` at runtime per
  token / per tile, then quantize. One extra reduction, but combined with the
  measurement+quant in a single memory pass it is essentially free when **fused
  into RMSNorm/activation** (see §6). DeepSeek-V3 uses online per-1×128-tile
  activation scaling — no historical-amax bookkeeping, better accuracy.

**The DeepSeek block recipe (now the de-facto standard for MoE):** activations
tile-wise **1×128**, weights block-wise **128×128**, format **E4M3** everywhere,
FP32 accumulation. Because scales vary along the contracting (K) dimension, the
rescale MUST be fused into the GEMM mainloop — you cannot just scale the final C.

> UE8M0 evolution (DeepSeek-V3.1, CDNA4/Blackwell native): activation scales encoded
> as 1-byte E8M0 (power-of-two exponent) instead of FP32 → 75% scale-memory cut, and
> `scale_a * scale_b` becomes `exp_a + exp_b`. On CDNA3 you still carry FP32 scales;
> UE8M0 is the bridge to MX formats (see `quantization_fp4_fp6.md`).

---

## 4. Quant / dequant kernel logic

### 4.1 Per-token dynamic E4M3-FNUZ quant (Triton, MI300X)

```python
import triton
import triton.language as tl

# MI300X-native: E4M3-FNUZ max is 240.0 (NOT 448). Use the right constant!
FP8_E4M3_FNUZ_MAX = 240.0

@triton.jit
def per_token_quant_fp8_kernel(
    x_ptr,            # [M, K] bf16/fp16 input (e.g. RMSNorm output)
    y_ptr,            # [M, K] fp8 output (e4m3fnuz)
    scale_ptr,        # [M]    fp32 per-token scale (= amax/FP8_MAX)
    M, K,
    stride_xm, stride_xk,
    stride_ym, stride_yk,
    BLOCK_K: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    # ---- pass 1: amax over the row (K may be > BLOCK_K → loop) ----
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = offs < K
        x = tl.load(x_ptr + row * stride_xm + offs * stride_xk, mask=m, other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32))))
    # avoid div-by-zero; clamp tiny amax
    scale = amax / FP8_MAX
    scale = tl.where(scale == 0.0, 1.0, scale)
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + row, scale)
    # ---- pass 2: quantize ----
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = offs < K
        x = tl.load(x_ptr + row * stride_xm + offs * stride_xk, mask=m, other=0.0)
        q = x.to(tl.float32) * inv_scale
        q = tl.clamp(q, -FP8_MAX, FP8_MAX)
        # IMPORTANT: AMD path requires FP32 -> FP8; tl handles the cast to the
        # e4m3fnuz storage type. On MI300X the only HW conversion *to* fp8 is
        # from fp32, so promote first.
        y = q.to(y_ptr.dtype.element_ty)   # element_ty = tl.float8e4b8 (fnuz)
        tl.store(y_ptr + row * stride_ym + offs * stride_yk, y, mask=m)
```

Notes that bite people on AMD:
- **Promote to FP32 before casting to FP8.** CDNA3 has no direct bf16→fp8 path; the
  hardware conversion source is FP32. Skipping the promote can produce wrong rounding.
- The Triton fp8 element type for FNUZ E4M3 is `tl.float8e4b8` (bias-8); OCP E4M3 is
  `tl.float8e4nv`. Pick the FNUZ one for MI300X native GEMM.
- `dequant` is just `x_fp8.to(fp32) * scale` — but for block scaling the scale you
  multiply by depends on which (k-block) you are in (see mainloop below).

### 4.2 Block 128×128 weight dequant (reference)

```python
@triton.jit
def weight_dequant_block_kernel(w_q_ptr, w_s_ptr, w_ptr, N, K,
                                BLOCK: tl.constexpr):
    bn = tl.program_id(0); bk = tl.program_id(1)
    offs_n = bn * BLOCK + tl.arange(0, BLOCK)
    offs_k = bk * BLOCK + tl.arange(0, BLOCK)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    wq = tl.load(w_q_ptr + offs_n[:, None]*K + offs_k[None, :], mask=mask)
    s  = tl.load(w_s_ptr + bn * tl.cdiv(K, BLOCK) + bk)   # one scale / 128x128 block
    w  = wq.to(tl.float32) * s
    tl.store(w_ptr + offs_n[:, None]*K + offs_k[None, :], w.to(w_ptr.dtype.element_ty), mask=mask)
```

---

## 5. Scaled-MFMA GEMM (the fused mainloop)

The defining feature of block-scaled FP8 GEMM on CDNA3: because A-tile scales
(per-token, 1×128) and B-block scales (128×128) both vary along **K**, you must
apply scales *inside* the K-loop, per 128-wide chunk, accumulating in FP32. There
is no scale-in-core MFMA on gfx942, so you scale the partial product after each
MFMA group.

```python
@triton.jit
def w8a8_block_fp8_matmul(
    a_ptr, b_ptr, c_ptr,
    as_ptr,           # [M, K//128] fp32 activation scales (per-token, per-128-K-tile)
    bs_ptr,           # [N//128, K//128] fp32 weight block scales
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  # BLOCK_K=128
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    # ---- swizzled tile mapping for L2 reuse (group along M) ----
    num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * num_pid_n
    gid = pid // width
    pid_m = gid * GROUP_M + (pid % width) % GROUP_M
    pid_n = (pid % width) // GROUP_M

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak
    b_ptrs = b_ptr + offs_n[None, :]*stride_bn + offs_k[:, None]*stride_bk

    k_blocks = tl.cdiv(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(k_blocks):
        a = tl.load(a_ptrs)          # fp8 e4m3fnuz
        b = tl.load(b_ptrs)          # fp8 e4m3fnuz
        # MFMA partial in fp32 (tl.dot lowers to mfma_f32_16x16x32_fp8_fp8)
        p = tl.dot(a, b)             # fp32 partial for this 128-K slice
        # per-128-K scales: a_scale is [BLOCK_M] (per token), b_scale is scalar/block
        a_s = tl.load(as_ptr + offs_m * (K // BLOCK_K) + kb)        # [BLOCK_M]
        b_s = tl.load(bs_ptr + pid_n * (K // BLOCK_K) + kb)         # scalar
        acc += p * a_s[:, None] * b_s     # rescale BEFORE next K-block accumulates
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

Key points:
- **Accumulate in FP32**, rescale per K-block, then accumulate the rescaled partial.
  This is the AMD analogue of DeepSeek's "promote every 4 WGMMA to an FP32 register
  accumulator" trick — on CDNA3 the MFMA already accumulates in FP32, so you just
  multiply the partial by the two scales for that K-slice and add.
- `BLOCK_K = 128` is **fixed** to match the scaling granularity. The autotune space
  sweeps BLOCK_M ∈ {16,32,64}, BLOCK_N ∈ {32,64,128}, num_warps ∈ {4,8}, num_stages
  ∈ {2,3,4}. (DeepSeek's reference generates ~36 configs with K fixed at 128.)
- **GROUP_M tile swizzle** is essential on MI300X's 256 MB L2 — group programs along
  M so consecutive workgroups reuse the same B columns from L2.
- For **per-tensor / per-token-only** (no K-blocking), it's simpler: do the whole K
  loop with one MFMA accumulation, then in the epilogue multiply by
  `a_scale[:, None] * b_scale[None, :]`. That's the W8A8 fast path.

### Library backends for FP8 GEMM on MI300X

| Backend | What it is | When to use |
|---------|-----------|-------------|
| **hipBLASLt** | AMD's tuned GEMM lib; FP8 (FNUZ) support, per-tensor scaleA/scaleB, grouped GEMM (CDNA3-exclusive, ~29% MoE uplift) | default for plain/grouped FP8 GEMM; `hipblaslt-bench` to tune |
| **aiter** | AMD AI Tensor Engine; CK/ASM/Triton kernels for block-scale GEMM, fused MoE | default GEMM/MoE backend in vLLM & SGLang ROCm |
| **CK (Composable Kernel)** | C++ template kernel lib; underlies many aiter kernels | custom fused epilogues |
| **Triton** | portable; `w8a8_block_fp8_matmul` fallback | when no native block-scale path / rapid iteration |

> hipBLASLt scaled GEMM CLI: `hipblaslt-bench --a_type fp8_r --b_type fp8_r
> --c_type bf16_r --compute_type f32_r --scaleA 1 --scaleB 1 -m .. -n .. -k ..`

---

## 6. Fusion — the single biggest FP8 inference win

Each unfused quant is a full HBM read+write of the activation. Fusing the quant into
the producer kernel removes that round-trip. On MI300X (decode = bandwidth-bound)
this is often a bigger win than the GEMM speedup itself.

| Fusion | Saves | Status on ROCm/AITER |
|--------|-------|----------------------|
| **fused_add_rms_norm → fp8 quant** (residual add + RMSNorm + per-token quant) | 1 HBM read + 1 write of activation | AITER `rmsnorm2d_fwd_with_add` + quant; vLLM `rms_quant_fusion` / `rocm_aiter_fusion` torch.compile pass; "AITER RMSNorm quantization fusion pass" (#26575/#26575-series) |
| **SiLU + mul + fp8 quant** (SwiGLU gate) | 2 round-trips | AITER fused gated kernel; vLLM `persistent_masked_m_silu_mul_quant` (CUDA) → ROCm uses Triton fallback (aiter issue #2420 tracks dedicated HIP) |
| **GEMM epilogue → fp8 quant** (quant the GEMM output for the next layer) | 1 round-trip | epilogue cast in CK/hipBLASLt |
| **RoPE + qk-norm + quant** | 1–2 round-trips | AITER `fused_qk_rmsnorm`, `fused_qk_norm_mrope_3d` |

**AMD-specific constraint:** the only HW conversion to FP8 is from FP32. So a fused
RMSNorm→fp8 kernel must compute the norm in FP32 (or promote) before the fp8 cast.
Don't try to cast bf16 directly. (See `rmsnorm_rope_activation.md` for the full
fused RMSNorm+quant kernel.)

Enable on ROCm: `export VLLM_ROCM_USE_AITER=1` (master switch; also needed even when
you pass `--attention-backend`, because it gates GEMM/RMSNorm/MoE). Sub-flag
`VLLM_ROCM_USE_AITER_RMSNORM=1`.

---

## 7. FP8 KV cache

KV cache is the dominant memory consumer at long context; storing it FP8 halves it
and halves the bandwidth the attention kernel must read during decode.

| Choice | Format | Scale | Notes |
|--------|--------|-------|-------|
| E4M3-FNUZ KV | `kv_cache_dtype=fp8` (fp8_e4m3 on AMD → fnuz) | per-tensor (static, from calibration) or per-token dynamic | default; best precision for typical K/V ranges |
| E5M2 KV | `fp8_e5m2` | per-tensor | wider range, lower precision; for models with large K/V outliers |

- vLLM/SGLang store KV as `torch.uint8` underneath (the fp8 bytes). The attention
  kernel dequantizes K and V to FP32/BF16 in registers before the QK^T / PV MFMA, OR
  runs an FP8 attention path (below).
- Calibrate per-tensor KV scales offline (a small dataset amax) for static scaling;
  dynamic per-token KV scaling avoids calibration at a small runtime cost.
- AMD note: ensure you write the FNUZ-encoded bytes; reusing OCP-calibrated scales
  on CDNA3 needs the ×2 bias correction.

---

## 8. FP8 attention

Beyond FP8 KV storage, the attention matmuls themselves can run FP8:

- **QK^T in FP8**: quantize Q and K to E4M3, MFMA in fp8, accumulate scores in FP32,
  then softmax in FP32/BF16. Q is per-token-quantized; K uses the KV-cache scale.
- **PV (softmax·V) in FP8**: P (attention probs ∈ [0,1]) quantizes well with a
  per-row scale (often a single scalar since max prob ≤ 1); V uses the KV scale.
- Softmax stays FP32 — never quantize the exp/normalization.
- On MI300X this rides the same `mfma_f32_16x16x32_fp8_fp8` cores; aiter's
  MHA/MLA/PagedAttention backends implement fp8 paths. FlashAttention-style online
  softmax + fp8 QK/PV is the high-performance route; the accuracy hit is small if Q
  and P are per-token/per-row scaled.

> Contrast: this mirrors NVIDIA FlashAttention-3 fp8, but on AMD you use FNUZ and
> aiter/CK kernels rather than CUTLASS.

---

## 9. Accuracy guidance

| Lever | Effect |
|-------|--------|
| E4M3 vs E5M2 | E4M3 for weights/acts (more mantissa); E5M2 only when range demands |
| Per-tensor → per-token/block | finer scaling tracks outliers; block 128 ≈ near-lossless on big models |
| Keep RMSNorm/softmax/residual in BF16/FP32 | never quantize the norm or reductions |
| SmoothQuant / AutoSmoothQuant (Quark) | migrate activation outliers into weights before quant |
| Dynamic (online) over delayed | avoids stale amax, better on shifting activation distributions |
| FNUZ ×2 scale correction | mandatory on MI300X when loading OCP checkpoints |

Rule of thumb: per-tensor static FP8 can lose noticeable accuracy on outlier-heavy
models (e.g. some 70B), while DeepSeek-style 1×128 act / 128×128 weight block FP8 is
effectively lossless even on very large models. When in doubt, block-scale.

---

## 10. Quick checklist for an FP8 kernel on MI300X

1. Use **E4M3-FNUZ** (`float8_e4m3fnuz`, Triton `tl.float8e4b8`) for native MFMA.
2. Apply the **×2 bias correction** when loading OCP weights/scales.
3. **Promote to FP32 before the fp8 cast** (CDNA3 has no direct bf16→fp8).
4. Choose **16×16×32 MFMA** + small output tiles for CU occupancy on typical shapes.
5. **Accumulate in FP32**; for block scaling rescale per 128-K slice in the mainloop.
6. **Fuse the quant** into RMSNorm / SiLU·mul / GEMM epilogue — kill the HBM round-trip.
7. **GROUP_M tile swizzle** for the 256 MB L2.
8. Pick block scaling (1×128 act / 128×128 weight) for accuracy; per-token+per-channel
   for the W8A8 fast path.
9. Tune with `hipblaslt-bench`; don't trust default heuristics for peak.
10. KV cache fp8 (E4M3-FNUZ) for decode bandwidth; dequant in registers or run fp8 attn.

---

## Sources

- ROCm Blog — FP8 GEMM Optimization on AMD CDNA4 (MFMA shapes 16x16x128, scale instructions, TFLOPS): https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- AMD "Instinct MI300 CDNA3 ISA Reference Guide" (MFMA fp8 instructions, FNUZ): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- luongthecong123/fp8-quant-matmul — row-wise block-scale fp8 matmul, MFMA tile/occupancy analysis, GPU-MODE AMD challenge: https://github.com/luongthecong123/fp8-quant-matmul
- AMD MI300X workload optimization (hipBLASLt tuning, GEMM): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- DeepSeek-V3 inference kernels (act_quant, weight_dequant, fp8_gemm; 1×128 / 128×128 block scaling): https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
- DeepGEMM — fine-grained FP8 GEMM, UE8M0 scale layout: https://github.com/deepseek-ai/DeepGEMM
- vLLM FP8 / DeepSeek MLA+FP8 optimization (CUTLASS block fp8, fnuz handling): https://github.com/vllm-project/vllm/issues/26768
- ROCm/aiter — fused RMSNorm/quant, block-scale GEMM, MoE: https://github.com/ROCm/aiter
- Ambati & Diep, "AMD MI300X GPU Performance Analysis" (peak vs sustained fp8, hipblaslt-bench): https://arxiv.org/pdf/2510.27583
- "An Investigation of FP8 Across Accelerators for LLM Inference": https://arxiv.org/html/2502.01070v1
