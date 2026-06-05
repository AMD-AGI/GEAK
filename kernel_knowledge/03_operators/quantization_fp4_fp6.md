# FP4 / FP6 Microscaling on AMD (CDNA3 gfx942 emulated → CDNA4 gfx950 native)

> AMD-only knowledge. The 4-bit/6-bit story splits hard across generations:
> **MI300X (CDNA3 / gfx942) has NO native FP4/FP6 matrix path** — you emulate
> (dequant to fp8/bf16, then MFMA) or use weight-only schemes. **MI355X (CDNA4 /
> gfx950) has native block-scaled FP4/FP6 MFMA** via the `mfma_scale_*_f8f6f4`
> instructions. This file gives the OCP MX spec, the format tables, the native
> block-scaled GEMM kernel logic, the CDNA3 emulation path, and accuracy. Read
> `quantization_fp8.md` first for the scaling/fusion fundamentals.

---

## 1. Why FP4/FP6 — and the microscaling idea

4-bit/6-bit alone cannot represent LLM weights/activations: dynamic range collapses
and outliers destroy accuracy. **Microscaling (MX)** fixes this with **block floating
point**: a contiguous block of K=32 low-precision elements shares ONE power-of-two
scale (E8M0). Each block adapts its scale to its local magnitude, so a single outlier
block doesn't crush the rest of the tensor.

This is the OCP MX v1.0 standard (Microsoft/AMD/NVIDIA/Intel/Arm/Meta etc.). All MX
formats use **block size 32** and an **E8M0** (8-bit, pure biased exponent, power-of-2)
shared scale.

Storage win for MXFP4 (vs FP16): a 32-element block = 32×4 bits data + 8 bits scale
= 136 bits = **17 bytes** vs 64 bytes for FP16 → **73% smaller**; vs FP8 (33 bytes)
→ **47% smaller**.

---

## 2. The OCP MX format table

| Format | Element | Elem bits | Element range | Block (K) | Scale | Scale bits | Bytes / 32-blk | Native HW |
|--------|---------|-----------|---------------|-----------|-------|-----------|----------------|-----------|
| **MXFP8** (E4M3) | FP8 E4M3 | 8 | ±448 | 32 | E8M0 | 8 | 33 | CDNA3* + CDNA4 |
| **MXFP8** (E5M2) | FP8 E5M2 | 8 | ±57344 | 32 | E8M0 | 8 | 33 | CDNA3* + CDNA4 |
| **MXFP6** (E2M3) | FP6 E2M3 | 6 | ±7.5 | 32 | E8M0 | 8 | 25 | **CDNA4 only** |
| **MXFP6** (E3M2) | FP6 E3M2 | 6 | ±28.0 | 32 | E8M0 | 8 | 25 | **CDNA4 only** |
| **MXFP4** (E2M1) | FP4 E2M1 | 4 | ±6.0 | 32 | E8M0 | 8 | 17 | **CDNA4 only** |
| **MXINT8** | INT8 | 8 | ±127 | 32 | E8M0 | 8 | 33 | CDNA4 |

\* CDNA3 supports *packed* MXFP8 (E4M3/E5M2) since gfx942; FP4/FP6 matrix is CDNA4 only.

### Element bit layouts (the full set of small floats)

| Type | S | E | M | Subnormals | Max | Smallest normal | Notes |
|------|---|---|---|-----------|-----|-----------------|-------|
| **E2M1 (FP4)** | 1 | 2 | 1 | yes | 6.0 | 1.0 | only 16 codes: {0,±0.5,±1,±1.5,±2,±3,±4,±6} |
| **E2M3 (FP6)** | 1 | 2 | 3 | yes | 7.5 | 1.0 | more mantissa → finer steps, tighter range |
| **E3M2 (FP6)** | 1 | 3 | 2 | yes | 28.0 | 0.25 | more exponent → wider range, coarser steps |
| **E8M0 (scale)** | 0 | 8 | 0 | n/a | 2^127 | 2^-127 | unsigned, no sign/mantissa, **no Inf, no zero**; 1 NaN code (0xFF) |

**E2M3 usually beats E3M2** for MX: because the E8M0 block scale already supplies
dynamic range, the extra *mantissa* bit (E2M3) buys more accuracy than the extra
*exponent* bit (E3M2). Use E2M3 unless a tensor genuinely needs the wider per-element
range.

### E8M0 semantics (critical for kernels)

E8M0 is a raw biased FP32-style exponent: the encoded `uint8 = s`, actual scale =
`2^(s - 127)`. So `s = 127 → ×1` (no scaling). It cannot encode Inf, negative, or
zero; `0xFF` is the single NaN. Quantizer computes the per-block scale as
`floor(log2(max_abs(block)))` clipped to [-127,127], **minus 2** (head-room so the
block max maps below the element max), then **RNE** when encoding to E8M0.

> RNE on the scale is not optional. Plain floor systematically underestimates the
> block range → negative bias → measurable accuracy loss. AMD's Quark uses RNE.

---

## 3. Native block-scaled MFMA on CDNA4 (gfx950)

CDNA4 added matrix instructions that take the block scales **into the matrix core**.
This is the entire reason MXFP4/MXFP6 is fast on MI355X and only emulated on MI300X.

| Instruction | A×B×K | Per-A-block scale stride | Output C tile | On CDNA3? |
|-------------|-------|--------------------------|---------------|-----------|
| `V_MFMA_SCALE_F32_16X16X128_F8F6F4` | 16×16×128 | 32 (one E8M0 / 32 elems) | 16×16 | **no** |
| `V_MFMA_SCALE_F32_32X32X64_F8F6F4` | 32×32×64 | 32 | 32×32 | **no** |

Both accept **mixed types per operand**: A and B can independently be FP8/FP6/FP4.
Type codes: `0=E4M3, 1=E5M2(bf8), 2=E2M3(fp6), 3=E3M2(bf6), 4=E2M1(fp4)`.

### The intrinsic

```cpp
// CDNA4 / gfx950 only. Needs ROCm 7.0+ and hip_ext_ocp.h.
d_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg,            // 256-bit packed A fragment
            b_reg,            // 256-bit packed B fragment
            c_reg,            // FP32 accumulator fragment
            A_type, B_type,   // 0..4 (E4M3/E5M2/E2M3/E3M2/E2M1), per-operand
            OPSEL_A, scale_a, // OPSEL=0; scale_a = E8M0 byte for this A block
            OPSEL_B, scale_b);// scale_b = E8M0 byte for this B block
// scales applied AFTER the dot product, BEFORE accumulation.
```

Per-thread (lane) register layout for **32×32×64**: A=32×64, B=64×32, C=32×32, with
scale matrices Ax=32×2, Bx=2×32. Each lane holds **32 A entries, 1 Ax scale, 32 B
entries, 1 Bx scale, 16 C entries**. Note the structure mirrors the MX format exactly:
32 element values paired with 1 scale.

### FP4 packing — the 256-bit gotcha

The intrinsic's A/B args must be **256 bits**. But 32 FP4 elements = 128 bits. So you
define a `fp4x64_t` that is 256 bits — 128 bits of data, 128 bits zero-padded:

```cpp
using __amd_fp4x2_storage_t = uint8_t;   // 2 fp4 per byte; can't address <8 bits

uint8_t __amd_extract_fp4(__amd_fp4x2_storage_t x, size_t i) {
    return (i == 0) ? (x & 0x0Fu) : (x >> 4);
}
__amd_fp4x2_storage_t __amd_create_fp4x2(uint8_t lo, uint8_t hi) {
    return (uint8_t)(lo | (hi << 4));
}
// fp4x64_t: 256-bit; lower 128 bits = 32 packed fp4 (16 bytes), upper 128 = 0.
```

Cycle cost: if either operand is FP8 the instruction is slower (16×16×128: 16 or 32
cyc; 32×32×64: 32 or 64 cyc). All-FP4 / all-FP6 is the fast path. **FP6 and FP4 reach
the same peak on MI355X** (≈20 PFLOP/s each, 4× FP16) — unlike NVIDIA Blackwell where
MXFP6 has no throughput edge over MXFP8 and only MXFP4 doubles it.

### Headers / dtypes

- `hip_ext_ocp.h` (`__amd_fp8_storage_t`, OCP types) — the gfx950 accelerated path.
- CDNA4 FP8 is **OCP** (E4M3FN / E5M2), NOT FNUZ. (Opposite of CDNA3! See fp8 file.)

---

## 4. Native block-scaled GEMM kernel logic (CDNA4)

```cpp
// Sketch: MXFP4 x MXFP4 -> FP32, 32x32x64 tiles, gfx950.
// A: [M,K] mxfp4 (packed), Ascale: [M, K/32] E8M0 (uint8)
// B: [K,N] mxfp4 (packed), Bscale: [K/32, N] E8M0 (uint8)
__global__ void mxfp4_gemm(const uint8_t* A, const uint8_t* Asc,
                           const uint8_t* B, const uint8_t* Bsc,
                           float* C, int M, int N, int K) {
    // each wave computes a 32x32 C tile; loop over K in steps of 64 (= 2 blocks of 32)
    fp4x64_t a_frag, b_frag;   // 256-bit packed (128b data + 128b zero)
    float    acc[16] = {0};    // 16 C entries / lane for 32x32

    for (int k0 = 0; k0 < K; k0 += 64) {
        load_fp4_fragment(A, /*...*/ k0, &a_frag);   // gather 32 fp4 / lane, pad to 256b
        load_fp4_fragment(B, /*...*/ k0, &b_frag);
        // one E8M0 scale per 32-block; k-step of 64 covers two blocks → two MFMA calls
        uint8_t sa0 = Asc[/* row, k0/32     */];
        uint8_t sb0 = Bsc[/* k0/32, col     */];
        uint8_t sa1 = Asc[/* row, k0/32 + 1 */];
        uint8_t sb1 = Bsc[/* k0/32 + 1, col */];
        // type codes 4,4 = E2M1 (fp4) for both operands
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                  a_frag, b_frag, acc, /*Atype*/4, /*Btype*/4, 0, sa0, 0, sb0);
        // (second 32-block handled by the K=64 instruction's internal layout, or a
        //  second call for k0+32 depending on fragment packing)
    }
    store_c_tile(C, acc, /*...*/);
}
```

Key facts the kernel must respect:
- **Scale applied in-core, after dot, before accumulate** — you do NOT rescale the
  partial yourself (contrast the CDNA3 fp8 mainloop where you multiply by FP32 scales
  manually). Just feed the right E8M0 byte per 32-block.
- **Accumulate in FP32** (`acc` is fp32). Output bf16/fp16 in the epilogue.
- One E8M0 scale per **32** contracting elements — the K loop granularity is tied to
  32-blocks, like fp8 block GEMM is tied to 128.
- For **mixed** MXFP4×MXFP6 (e.g. fp4 weights, fp6 activations), set Atype/Btype to
  the two type codes; same instruction, different per-operand input layout.

### hipBLASLt / ROCm 7.0 path

ROCm 7.0 exposes MX GEMM without writing the intrinsic by hand:
- Set `HIPBLASLT_MATMUL_DESC_A_SCALE_MODE` /
  `..._B_SCALE_MODE` = `HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`.
- The `VEC32_UE8M0` enum literally encodes **32-element block, unsigned E8M0 scale**.
- ROCm 7.0 added a full MX GEMM pipeline for MXFP8/MXFP6/MXFP4; gfx950 only.

Triton: the `block_scaled_matmul` tutorial is generic over MXFP4/MXFP8 (and NVIDIA
nvfp4), HW-accelerated on CDNA4 matrix cores. Use it for portable MX GEMM.

---

## 5. CDNA3 / MI300X: emulation & weight-only

**MI300X has no FP4/FP6 matrix instruction.** Your realistic options:

| Strategy | How | Cost on MI300X | Use case |
|----------|-----|----------------|----------|
| **Weight-only MXFP4 (W4A16)** | store weights MXFP4, **dequant to BF16/FP16 in the kernel**, MFMA in bf16 | memory-bound layers win on bandwidth (4× smaller weights); compute is bf16 | decode of large dense/MoE; weights dominate HBM |
| **Dequant-to-FP8 then fp8 MFMA** | MXFP4 weights → fp8 (fnuz) on the fly → `mfma_*_fp8` | extra dequant ALU; compute at fp8 rate | when you want fp8 matrix speed + 4-bit storage |
| **Pure emulation (fp4 mul in ALU)** | unpack, multiply in VALU | slow; no matrix-core benefit | not recommended for perf |

Weight-only MXFP4 is the **practically useful** FP4 mode on MI300X: decode is
bandwidth-bound, so loading 4-bit weights (1/4 the bytes of FP16) is a real win even
though the matmul itself runs at BF16/FP8 rate after dequant.

### MXFP4 → BF16 dequant kernel (CDNA3 weight-only)

```python
import triton, triton.language as tl

# FP4 E2M1 lookup: 8 magnitudes (sign handled separately) -> bf16
# codes: 000=0, 001=0.5, 010=1.0, 011=1.5, 100=2.0, 101=3.0, 110=4.0, 111=6.0
FP4_E2M1 = tl.constexpr  # materialize as a small constant table in practice

@triton.jit
def mxfp4_dequant_to_bf16(
    wq_ptr,        # packed fp4 weights, 2 elems / byte: [N, K//2] uint8
    sc_ptr,        # E8M0 scales (uint8): [N, K//32]
    w_ptr,         # output bf16: [N, K]
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,   # BLOCK_K multiple of 32
):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)          # element index
    byte_k = offs_k // 2                                       # packed byte index
    nib_hi = (offs_k % 2) == 1
    packed = tl.load(wq_ptr + offs_n[:, None]*(K//2) + byte_k[None, :])
    nib = tl.where(nib_hi[None, :], packed >> 4, packed & 0xF) # 4-bit code
    sign = (nib >> 3) & 1
    mag_code = nib & 0x7
    # decode E2M1 magnitude (use a small select tree or a gather table)
    mag = decode_e2m1(mag_code).to(tl.float32)                # 0..6.0
    val = tl.where(sign == 1, -mag, mag)
    # apply block scale: one E8M0 per 32 elements along K
    blk = offs_k // 32
    s_e8m0 = tl.load(sc_ptr + offs_n[:, None]*(K//32) + blk[None, :])
    scale = tl.exp2((s_e8m0.to(tl.float32) - 127.0))          # 2^(s-127)
    out = (val * scale).to(tl.bfloat16)
    tl.store(w_ptr + offs_n[:, None]*K + offs_k[None, :], out)
```

Then feed `w_ptr` (bf16) into a normal bf16 MFMA GEMM. For **fused** weight-only you
do this dequant inside the GEMM mainloop (load fp4 + scale, dequant in registers, MFMA)
to avoid materializing the bf16 weights in HBM.

---

## 6. Full vs weight-only MXFP4

| Mode | Weights | Activations | MI300X | MI355X |
|------|---------|-------------|--------|--------|
| **W4A16** (weight-only) | MXFP4 | BF16/FP16 | yes (dequant→bf16) | yes |
| **W4A4** (full MX) | MXFP4 | MXFP4 | emulate (no native) | **native** (`mfma_scale` fp4×fp4) |
| **Mixed MXFP4/MXFP6** | MXFP4 | MXFP6 (or per-layer) | emulate | **native** (mixed type codes) |

Full W4A4 only pays off where the matrix core natively scales (CDNA4). On CDNA3,
stick to W4A16 weight-only for the bandwidth win, or use FP8 for compute.

---

## 7. Accuracy

From AMD Quark MXFP4/MXFP6 results (AutoSmoothQuant, MI355X):

| Model | MXFP4 | MXFP6 / mixed |
|-------|-------|---------------|
| DeepSeek-R1-0528 (671B) | **>99.5%** of FP16 acc (AIME24, GPQA-Diamond, MATH-500) | ≥ MXFP4 |
| Llama-3.1-405B | strong (near-lossless) | higher |
| Llama-3.3-70B | **visible degradation** | MXFP6 / mixed recovers it |
| gpt-oss-120b | lowest | MXFP6 > MXFP4-MXFP6 > MXFP4 |

Takeaways:
- **MXFP4 is near-lossless on very large models** (the per-32-block scale + huge
  parameter redundancy absorb the 4-bit noise).
- **Smaller models (≤70B) need MXFP6 or mixed MXFP4/MXFP6** to recover accuracy —
  often FP6 on sensitive layers (attention/first/last), FP4 on the rest.
- **AutoSmoothQuant** (per-layer outlier smoothing, no manual tuning) is the
  recommended Quark recipe; GPTQ / Quarot also available.
- Always RNE the E8M0 scales; keep norms/softmax/residual in BF16/FP32.

---

## 8. Quick checklist

1. MX = **block 32 + E8M0 power-of-two scale**; element ∈ {E2M1, E2M3, E3M2}.
2. **MI300X has no native FP4/FP6 matrix** — use **W4A16 weight-only** (dequant→bf16)
   for the bandwidth win, or dequant→fp8 for fp8 matrix speed.
3. **MI355X native**: `__builtin_amdgcn_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4`,
   type codes per operand, E8M0 byte per 32-block, scale applied in-core.
4. CDNA4 FP8 is **OCP** (not FNUZ like CDNA3) — different `hip_ext_ocp.h` path.
5. **FP4 needs 256-bit `fp4x64_t` packing** (128b data + 128b zero) for the intrinsic.
6. Prefer **E2M3** over E3M2 for FP6 (mantissa beats exponent when scaled).
7. Use **MXFP6/mixed** for ≤70B models; MXFP4 is fine for 100B+.
8. hipBLASLt: `HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` (ROCm 7.0+, gfx950).
9. **RNE the E8M0 scale**; head-room (-2) so block-max maps below element-max.
10. Accumulate FP32; quantize with AMD Quark AutoSmoothQuant.

---

## Sources

- OCP Microscaling Formats (MX) v1.0 Specification (block 32, E8M0, MXFP4/6/8): https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- ROCm Blog — Matrix Core Programming on CDNA3 & CDNA4 (mfma_scale_f32_32x32x64_f8f6f4 layout, fp4x64_t packing, E8M0, type codes, CDNA3 vs CDNA4): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- ROCm Blog — High-Accuracy MXFP4/MXFP6 & Mixed-Precision on AMD GPUs (format table, accuracy, Quark AutoSmoothQuant): https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html
- ROCm 7.0 release notes (FP6/BF6/FP4 gfx950, VEC32_UE8M0 block scaling, MX GEMM pipeline): https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html
- Microsoft microxcaling — MX reference emulation library (quant/dequant math): https://github.com/microsoft/microxcaling
- Rouhani et al., "Microscaling Data Formats for Deep Learning" (MX accuracy foundations): https://arxiv.org/pdf/2310.10537
- Triton block-scaled matmul tutorial (MXFP4/MXFP8 generic, CDNA4 accelerated): https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
- AMD Instinct MI355X datasheet (FP4/FP6 PFLOPS, CDNA4): https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
- FPRox — OCP MX Scaling Formats explainer (E8M0 semantics, 17-byte MXFP4 block): https://fprox.substack.com/p/ocp-mx-scaling-formats
