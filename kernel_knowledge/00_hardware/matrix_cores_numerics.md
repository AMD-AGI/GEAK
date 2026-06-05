# Matrix Cores & Numerics on CDNA3 (MI300X / gfx942)

> The MFMA (Matrix Fused-Multiply-Add) instruction family, supported shapes & dtypes, register/lane mapping, intrinsic usage, FP8/INT8 numerics, and rounding behavior. CDNA4 (gfx950, MI350/MI355) deltas are noted because agents often target both.
> Companion to `mi300x_cdna3_arch.md` and `memory_hierarchy_occupancy.md`.

---

## 0. What you must know in 10 lines

- MFMA is a **wavefront-collective** op: all **64 lanes** cooperate on one `D = A·B + C` tile. There is no per-lane matmul.
- Each CU has **4 Matrix Cores** (one per SIMD). Device total = 304 CU × 4 = **1216 matrix cores**.
- Inputs are low precision (FP16/BF16/FP8/INT8); **accumulation is FP32 (or INT32)** to preserve accuracy.
- Peak: **FP16/BF16 = 1307 TFLOP/s**, **FP8/INT8 = 2615 TFLOP/s** (8× / 16× over FP32).
- On MI300X, **`mfma_16x16x16`** typically beats `32x32x8` for GEMM even at large tiles (better LDS/VGPR behavior).
- CDNA3 FP8 is **FNUZ** (E4M3FNUZ / E5M2FNUZ), **not** OCP — conversion code must match. CDNA4 switches to OCP E4M3FN/E5M2 and adds FP6/FP4/MXFP.
- Operands & results live in **VGPRs**; accumulators can be parked in **AGPRs** to save occupancy.
- Inspect any instruction with the **amd_matrix_instruction_calculator** (`--detail-instruction`).
- FP16/BF16/TF32 MFMA use a **round-down (RD)** accumulation path on CDNA3 (a known bias source); FP8 path was specially adjusted. CDNA3 fully supports subnormals (fixed vs CDNA2).
- CDNA3 also has **SMFMAC** sparse instructions (4:2 structured sparsity, ~2× throughput).

---

## 1. The MFMA instruction naming scheme

```
v_mfma_<Dtype>_<M>x<N>x<K><AB-type>
        │        │  │  │   └─ input dtype of A and B (f16, bf16, fp8, bf8, i8, f32, f64)
        │        └──┴──┴───── tile dimensions: A is MxK, B is KxN, C/D is MxN
        └────────────────────  output/accumulator dtype (f32, i32, f64)
```
Example: `v_mfma_f32_16x16x16_bf16` → BF16 inputs, FP32 accumulate, 16×16 output tile, K=16.
Scaled (CDNA4) variants: `v_mfma_scale_f32_32x32x64_f8f6f4` carry per-block E8M0 scales.
Sparse (CDNA3): `v_smfmac_f32_16x16x32_f16`.

---

## 2. CDNA3 dense MFMA table (the ones that matter for inference)

Cycles are per-instruction on one matrix core. Per-thread element counts = `M·K/64` (A), `K·N/64` (B), `M·N/64` (C/D).

| Instruction | M×N×K | A,B in | C,D out | Cycles | A/B/C regs per lane | Notes |
|---|---|---|---|---|---|---|
| `v_mfma_f64_16x16x4_f64` | 16×16×4 | FP64 | FP64 | 64 | 1/1/4 | HPC |
| `v_mfma_f32_32x32x2_f32` | 32×32×2 | FP32 | FP32 | 64 | 1/1/16 | |
| `v_mfma_f32_16x16x4_f32` | 16×16×4 | FP32 | FP32 | 32 | 1/1/4 | |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | FP16 | FP32 | 32 | 4/4/16 | |
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | FP16 | FP32 | 16 | 4/4/4 | **preferred FP16 GEMM** |
| `v_mfma_f32_32x32x8_bf16` | 32×32×8 | BF16 | FP32 | 32 | 4/4/16 | opcode 96 |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | BF16 | FP32 | 16 | 4/4/4 | opcode 97, **preferred BF16** |
| `v_mfma_f32_32x32x16_fp8_fp8` | 32×32×16 | FP8(E4M3) | FP32 | 32 | 8/8/16 | FNUZ |
| `v_mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | FP8 | FP32 | 16 | 8/8/4 | **preferred FP8** |
| `v_mfma_f32_*_fp8_bf8` / `bf8_fp8` / `bf8_bf8` | same shapes | E4M3 / E5M2 mix | FP32 | — | — | A,B dtype chosen independently |
| `v_mfma_i32_32x32x16_i8` | 32×32×16 | INT8 | INT32 | 32 | — | quantized |
| `v_mfma_i32_16x16x32_i8` | 16×16×32 | INT8 | INT32 | 16 | — | **preferred INT8** |

CDNA3 does **not** have native FP6/FP4 or block-scaled (MX) MFMA — those are CDNA4-only (see §7).

### Peak-throughput formula
```
peak_TFLOPS = 2·M·N·K · num_matrix_cores · (clock_Hz / cycles) / 1e12
```
MI300X (1216 cores, 2.1 GHz). Check FP16 `32x32x8` (cycles=32):
`2·32·32·8 · 1216 · (2.1e9/32) / 1e12 ≈ 1307 TFLOP/s.` ✓
FP8 `16x16x32` (cycles=16): `2·16·16·32 · 1216 · (2.1e9/16)/1e12 ≈ 2615 TFLOP/s.` ✓

> Two instructions reach the same peak (e.g. FP16 16x16x16 @16cyc vs 32x32x8 @32cyc). Choose by **register/LDS pressure and tile fit**, not peak FLOPS. On MI300X, 16×16 wins more often.

---

## 3. Register & lane mapping (how A/B/C live in VGPRs)

An MFMA reads A and B from VGPRs (one vector register operand each holding the lane's slice) and reads/writes C/D from VGPRs or AGPRs. Layout is **row-major** with a fixed lane↔element mapping per instruction.

Per-lane element counts (wave64):
```
A entries/lane = M*K / 64
B entries/lane = K*N / 64
C entries/lane = M*N / 64
```
| Instruction | A/lane | B/lane | C/lane | Vector types used |
|---|---|---|---|---|
| `f32_32x32x2_f32` | 1 | 1 | 16 | `float`, `float`, `fp32x16_t` |
| `f32_16x16x16_f16` | 4 | 4 | 4 | `fp16x4_t`, `fp16x4_t`, `fp32x4_t` |
| `f32_32x32x8_f16` | 4 | 4 | 16 | `fp16x4_t`, `fp16x4_t`, `fp32x16_t` |
| `f32_32x32x16_fp8_fp8` | 8 | 8 | 16 | `fp8x8_t`(as long), `fp8x8_t`, `fp32x16_t` |
| `f32_16x16x32_fp8_fp8` | 8 | 8 | 4 | `fp8x8_t`, `fp8x8_t`, `fp32x4_t` |

To find which VGPR+lane holds a specific element, use the calculator:
```bash
# Where is A[3][2] for bf16 16x16x16?  (format Vx{lane}.sub)
./matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_16x16x16_bf16 \
    --get-register --A-matrix --I-coordinate 3 --K-coordinate 2
```
Stage A/B tiles in **LDS with a swizzled layout** that matches this mapping so `ds_read_b128` feeds MFMA conflict-free (see memory file §3.3).

---

## 4. Intrinsic usage — real HIP code

The Clang/HIP compiler exposes `__builtin_amdgcn_mfma_*`. Signature:
```cpp
d = __builtin_amdgcn_mfma_<out>_<MxNxK><in>(a, b, c, cbsz, abid, blgp);
//   cbsz/abid/blgp = broadcast/lane-group modifiers; 0 for ordinary GEMM.
```

### 4.1 FP32 32×32×2 (illustrative)
```cpp
using fp32x16_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;

__global__ void mfma_f32_32x32x2(const float* A, const float* B, float* C) {
    float a = A[/* lane-mapped index */];
    float b = B[/* lane-mapped index */];
    fp32x16_t acc = {0};
    acc = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, acc, 0, 0, 0);
    // store acc[0..15] back to C with the inverse lane mapping
}
```

### 4.2 FP16 16×16×16 (the GEMM workhorse)
```cpp
using fp16x4_t = __attribute__((__vector_size__(4 * sizeof(__half)))) __half;
using fp32x4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

fp16x4_t a_frag = *reinterpret_cast<const fp16x4_t*>(a_lds_ptr);
fp16x4_t b_frag = *reinterpret_cast<const fp16x4_t*>(b_lds_ptr);
fp32x4_t acc    = {0,0,0,0};

// K-loop: each call advances K by 16
for (int k = 0; k < K; k += 16) {
    a_frag = load_a(k); b_frag = load_b(k);
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);
}
```
BF16 is identical with `__builtin_amdgcn_mfma_f32_16x16x16bf16` and a `bf16x4_t` operand.

### 4.3 FP8 32×32×16 (note the `long` casts)
```cpp
#include <hip/hip_fp8.h>
using fp8_t   = __hip_fp8_storage_t;                                  // E4M3 FNUZ on CDNA3
using fp8x8_t = __attribute__((__vector_size__(8 * sizeof(fp8_t)))) fp8_t;
using fp32x16_t = __attribute__((__vector_size__(16*sizeof(float)))) float;

fp8x8_t a8 = load_a8(); fp8x8_t b8 = load_b8();
fp32x16_t acc = {0};
// FP8 MFMA expects A/B packed into 64-bit (long) registers:
acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long*>(&a8),
          *reinterpret_cast<long*>(&b8),
          acc, 0, 0, 0);
```
Mixed FP8: `__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8` (A=E4M3, B=E5M2), etc. INT8 uses `__builtin_amdgcn_mfma_i32_16x16x32_i8` with `int32x4_t` accumulators.

### 4.4 Keeping accumulators in AGPRs
For large output tiles, let the compiler place `acc` in AGPRs to preserve occupancy:
```cpp
// compile flags: -mllvm -amdgpu-mfma-vgpr-form=false -mllvm -amdgpu-agpr-alloc=256
```
Then `v_accvgpr_read_b32` is auto-inserted in the epilogue before `global_store`.

---

## 5. Numeric formats supported

| Format | Bits | Exp/Mant | CDNA3 variant | Notes |
|---|---|---|---|---|
| FP64 | 64 | 11/52 | IEEE | matrix + vector |
| FP32 | 32 | 8/23 | IEEE | matrix + vector |
| TF32 | 19 used | 8/10 | emulated | 653.7 TF; vector path |
| FP16 | 16 | 5/10 | IEEE | 1307 TF matrix; **RD accumulate** |
| BF16 | 16 | 8/7 | — | 1307 TF; **RD accumulate** |
| FP8 E4M3 | 8 | 4/3 | **FNUZ** (bias 8, ±240) | 2615 TF; only +0 & NaN |
| BF8/FP8 E5M2 | 8 | 5/2 | **FNUZ** (bias 16, ±57344) | larger range, less mantissa |
| INT8 | 8 | — | two's complement | 2615 TOPS, INT32 accumulate |
| INT4 | 4 | — | packed | quantized inference |

### 5.1 FP8 FNUZ vs OCP — critical for correctness
- **CDNA3 (gfx942)** uses **FNUZ** (`F`inite, `U`nsigned `Z`ero): E4M3FNUZ bias 8 (max ±240, no inf, only one zero, NaN as 0x80); E5M2FNUZ bias 16.
- **CDNA4 (gfx950) + most other vendors / OCP** use E4M3**FN** (bias 7, max ±448, ±0, ±NaN) and E5M2 (bias 15, ±57344, with inf).
- **Implication:** an FP8 model/checkpoint quantized for OCP must be **re-cast** for CDNA3 (different bias and saturation). Use `__hip_fp8_*` conversion helpers and the matching `__hip_fp8_storage_t` for the target arch; do not bit-copy OCP FP8 into CDNA3 MFMA.

### 5.2 Rounding & subnormals (accuracy caveats)
- CDNA3 TF32/FP16/BF16 MFMA conversion/accumulation uses an **asymmetric round-down (RD)** mode — it can introduce a small systematic bias; matters for long K reductions and training-like accumulation. The **FP8** path was specifically adjusted to mitigate this.
- CDNA3 **fully supports subnormals** (CDNA2 flushed some, which hurt training stability) — no need for the CDNA2 subnormal workarounds.
- Always accumulate in **FP32/INT32**; never down-convert the accumulator inside the K-loop.

---

## 6. Sparse matrix cores (SMFMAC, 4:2 structured sparsity)

CDNA3 supports `v_smfmac_*` for matrices with **4:2 structured sparsity** (in every group of 4 A-elements, 2 are zero), giving ~**2× throughput**.
- `D += A·B`; the usual C operand slot is replaced by a **compression-index VGPR (Src2)** that records which 2 of 4 are nonzero.
- Example: `v_smfmac_f32_16x16x32_f16`. Query the compression index layout:
```bash
./matrix_calculator.py --architecture cdna3 \
    --instruction v_smfmac_f32_16x16x32_f16 \
    --get-register --I-coordinate 2 --K-coordinate 31 --compression
# -> K[2][31] = v0{50}.[7:4]
```
Use only when the weight tensor genuinely has enforced 4:2 sparsity (post structured-pruning); otherwise dense MFMA.

---

## 7. CDNA4 (gfx950, MI350X/MI355X) deltas — for portable kernels

| Feature | CDNA3 (MI300X) | CDNA4 (MI350/355X) |
|---|---|---|
| FP8 variant | FNUZ | **OCP** E4M3FN / E5M2 |
| FP6 (E2M3 / E3M2) | ✗ | ✓ |
| FP4 (E2M1) | ✗ | ✓ |
| Block-scaled MXFP8/6/4 (`v_mfma_scale_*`, E8M0 scales) | ✗ | ✓ |
| New large-K shapes | — | 16×16×128, 32×32×64 |
| FP6/FP4 peak | — | ~10 PFLOP/s (~64× FP32) |
| LDS / CU | 64 KiB | 160 KiB |

Scaled intrinsic (CDNA4):
```cpp
// type codes: 0=E4M3 1=E5M2 2=E2M3 3=E3M2 4=E2M1 ; scales are E8M0 -> 2^(scale-127)
acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
          a, b, acc, /*Atype*/0, /*Btype*/0,
          /*opsel_a*/0, scale_a, /*opsel_b*/0, scale_b);
```
Guard arch-specific code paths with `__gfx942__` / `__gfx950__` and select the FP8 format (FNUZ vs OCP) accordingly.

---

## 8. The amd_matrix_instruction_calculator (your ground-truth tool)

```bash
# list every MFMA/SMFMAC for the arch
./matrix_calculator.py --architecture cdna3 --list-instructions

# full detail: opcode, M/N/K, blocks, cycles, FLOPs/CU/cycle, GPR usage,
# ArchVGPR/AccVGPR eligibility, co-execution with VALU, modifiers
./matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_16x16x16_bf16 --detail-instruction

# exact register/lane of one element (Vx{lane}.subreg format)
./matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_32x32x16_fp8_fp8 \
    --get-register --D-matrix --M-coordinate 5 --N-coordinate 7
```
`--detail-instruction` reports: encoding/opcode, M/N/K/blocks, **Execution cycles**, **FLOPs/CU/cycle**, **Can co-execute with VALU**, per-matrix GPR counts & alignment, in/out dtypes, and ArchVGPR/AccVGPR usability. Treat its output as authoritative over any blog table.

---

## Sources
1. Matrix Core Programming on AMD CDNA3 and CDNA4 — ROCm Blogs (full MFMA table, intrinsics, lane mapping, FP8 FNUZ): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
2. "Matrix Core Programming on AMD CDNA3 and CDNA4 architecture" — salykova (worked code for all dtypes incl. scaled FP8/FP4): https://salykova.github.io/matrix-cores-cdna
3. ROCm amd_matrix_instruction_calculator (instruction list, detail, register query, SMFMAC): https://github.com/ROCm/amd_matrix_instruction_calculator
4. AMD matrix cores — AMD GPUOpen lab notes (MFMA mapping & intrinsics): https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
5. AMD Instinct MI300 (CDNA3) ISA Reference Guide — Ch.7 Matrix Arithmetic Instructions: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
6. AMD Instinct MI300 Series microarchitecture (FLOPs/clock/CU table) — ROCm Docs: https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html
7. "MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores" — arXiv 2511.10909 (CDNA3 RD rounding, FP8 adjustment, subnormals): https://arxiv.org/html/2511.10909v1
8. Syntax of gfx942 Instructions — LLVM AMDGPU documentation: https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
