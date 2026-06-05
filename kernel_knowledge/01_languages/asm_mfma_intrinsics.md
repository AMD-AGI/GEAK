# CDNA3 Assembly, MFMA & Intrinsics — Low-Level Kernel Performance (MI300X / gfx942)

> Scope: the lowest level of the stack — CDNA3 (gfx942) ISA features that govern performance, the MFMA
> matrix-core instructions (encoding, register/lane layout, throughput), `s_waitcnt` and instruction
> scheduling for load/MFMA overlap, VGPR/AGPR banking & co-issue, and **inline GCN asm + compiler
> intrinsics in HIP**. CDNA4 (gfx950) deltas are flagged. This is the foundation under CK / ck_tile /
> rocWMMA — read those for the library-level view.

---

## 1. CDNA3 execution model (what the hardware actually is)

| Unit | count / size on MI300X | notes |
|---|---|---|
| **XCD** (accelerator complex die) | 8 per MI300X | each XCD ~38 CUs; 304 CUs active total |
| **CU** (compute unit) | 304 | 4 SIMDs each |
| **SIMD** | 4 per CU | 64-lane SIMD; one wavefront issues per SIMD |
| **Wavefront** | 64 lanes (wave64) | CDNA is **wave64 only** (RDNA can be wave32) |
| **VGPR file** | 512 × 32-bit per lane per SIMD | shared budget; drives occupancy |
| **AGPR file** | 256 × 32-bit per lane | accumulation GPRs for MFMA (CDNA-specific) |
| **LDS** | 64 KB per CU | 32 banks × 4B; `ds_read`/`ds_write` |
| **Matrix cores** | per-SIMD MFMA units | the XDL/MFMA engines |

**Occupancy** = how many wavefronts can be resident per SIMD, limited by `max(VGPR, AGPR, LDS, wave-slot)`.
With 512 VGPR/lane, a kernel using 256 VGPR/lane gets ≤2 waves/SIMD = 8 waves/CU. Spilling past 512
(or excessive AGPR↔VGPR traffic) collapses occupancy and is the #1 cause of MFMA kernels underperforming.

### 1.1 Instruction classes relevant to perf

| Class | examples | engine | counter |
|---|---|---|---|
| **VALU** | `v_add_f32`, `v_fma_f32`, `v_pk_*` (packed) | SIMD ALU | — |
| **MFMA / XDL** | `v_mfma_f32_16x16x16_bf16`, `v_mfma_f32_32x32x8_f16` | matrix core | — (in-order on the MFMA pipe) |
| **VMEM** | `buffer_load_*`, `global_load_*`, `buffer_load ... lds` | memory | **vmcnt** |
| **LDS (DS)** | `ds_read_b128`, `ds_write_b128` | LDS | **lgkmcnt** |
| **SMEM** | `s_load_dword*` (scalar/uniform loads) | scalar | **lgkmcnt** |
| **Scalar / control** | `s_waitcnt`, `s_barrier`, `s_setprio`, `s_nop` | scalar | — |

---

## 2. The MFMA instructions (CDNA3)

MFMA = **Matrix Fused Multiply-Add**: `D = A·B + C`, a **wave-level** op — all 64 lanes cooperate to
compute one M×N×K block, with A/B/C/D fragments distributed across the lanes' registers. Intrinsic form:

```
d_reg = __builtin_amdgcn_mfma_<ODType>_<M>x<N>x<K><InDType>(a_reg, b_reg, c_reg, cbsz, abid, blgp);
```

`ODType` = output/accum type (always 32-bit accumulate on CDNA: f32/i32/f64). `InDType` = A/B type.
`cbsz, abid, blgp` = broadcast/block-select flags (set `0,0,0` for plain GEMM).

### 2.1 The CDNA3 dense MFMA table (the ones that matter for LLM)

| Intrinsic | M×N×K | A/B → C | cycles | A regs/lane | B regs/lane | C regs/lane | peak* |
|---|---|---|---|---|---|---|---|
| `..._f32_32x32x8f16` | 32×32×8 | f16 → f32 | 32 | 4 (fp16x4) | 4 | 16 (fp32x16) | ~1.3 PF/s |
| `..._f32_16x16x16f16` | 16×16×16 | f16 → f32 | 16 | 4 | 4 | 4 | **higher** (power) |
| `..._f32_32x32x8bf16` | 32×32×8 | bf16 → f32 | 32 | 4 | 4 | 16 | — |
| `..._f32_16x16x16bf16` | 16×16×16 | bf16 → f32 | 16 | 4 | 4 | 4 | — |
| `..._f32_32x32x16_fp8_fp8` | 32×32×16 | fp8 → f32 | 32 | 8 (fp8x8) | 8 | 16 | 2× bf16 |
| `..._f32_16x16x32_fp8_fp8` | 16×16×32 | fp8 → f32 | 16 | 8 | 8 | 4 | 2× bf16 |
| `..._i32_16x16x32_i8` | 16×16×32 | i8 → i32 | 16 | 8 | 8 | 4 | int path |
| `..._f64_16x16x4f64` | 16×16×4 | f64 → f64 | 64 | 1 | 1 | 4 | HPC |
| `..._f32_16x16x4f32` | 16×16×4 | f32 → f32 | 32 | 1 | 1 | 4 | — |

`*` peak ≈ `2·M·N·K · num_matrix_cores · (clock / cycles)`. For 32×32×8 f16 on MI325X (1216 cores,
2.1 GHz): `2·32·32·8·1216·(2.1e9/32)/1e12 ≈ 1307 TFLOP/s`. fp8 variants double K density → ~2× peak.
fp8 e4m3/e5m2 can be **mixed** (`..._fp8_bf8`, `..._bf8_fp8`). gfx950 (CDNA4) adds native **mxfp4/fp4**
and larger-K MFMA tiles.

> **Critical MI300X tuning fact:** `mfma_16x16x16` usually **beats** `mfma_32x32x8` for GEMM, even at
> large sizes. The 32×32 op has higher *software* efficiency (bigger payload, fewer instructions) but
> draws more power and thus clocks lower; the 16×16 op is power-limited-friendlier and yields higher
> *max-achievable* FLOPs. Default to 16×16×16 and only test 32×32×8.

### 2.2 Register / lane layout (the part that bites)

The A/B/C fragments are scattered across the 64 lanes in a fixed pattern per instruction. Fragments are
in **packed registers with no guaranteed element order** — you must use the documented layout or the
[matrix-instruction-calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) to place data.

Worked example — `v_mfma_f32_16x16x16f16`: each lane holds 4 A elems, 4 B elems, 4 C elems. The
canonical load/compute/store (single wave, one 16×16×16 tile):

```cpp
using fp16_t  = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float))))  float;

__global__ void mfma_16x16x16(const fp16_t* A, const fp16_t* B, float* C) {
    fp16x4_t a, b; fp32x4_t c{};
    // A is 16x16 row-major: lane owns 4 contiguous-K elems at row=(tid/16-derived), see calculator
    a = *reinterpret_cast<const fp16x4_t*>(A + 4*(threadIdx.x/16) + 16*(threadIdx.x%16));
    for (int i = 0; i < 4; ++i)
        b[i] = *(B + i*16 + threadIdx.x%16 + (threadIdx.x/16)*64);
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);   // <-- one wave-wide MFMA
    for (int i = 0; i < 4; ++i)
        *(C + i*16 + threadIdx.x%16 + (threadIdx.x/16)*64) = c[i];
}
// launch one wavefront: mfma_16x16x16<<<1, 64>>>(...);
```

Query any layout precisely:

```bash
./matrix_calculator.py --architecture cdna3 --instruction v_mfma_f32_16x16x16_bf16 --detail-instruction
./matrix_calculator.py --architecture cdna3 --instruction v_mfma_f32_32x32x8_f16 --register-layout --A-matrix
# output format Vx{y}.z : x=register offset, y=lane, .z=sub-register (e.g. 16-bit half)
```

### 2.3 fp8 specifics

fp8 packs 8 elems/lane (`fp8x8_t`) and the intrinsic wants the A/B operands cast to `long` (64-bit):

```cpp
c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8((long)a_reg, (long)b_reg, c_reg, 0, 0, 0);
// mixed precision: ..._fp8_bf8 / ..._bf8_fp8 for e4m3 × e5m2
```
MI300X supports both OCP fp8 and the gfx942-only "NANOO"/`fnuz` fp8 encoding (different bias) — match
your dequant scale to the encoding.

---

## 3. VGPR / AGPR banking & co-issue

- **AGPRs** are CDNA's accumulation registers; classic MFMA codegen keeps the C accumulator in AGPRs.
  Reading/writing them from VALU code costs `v_accvgpr_read_b32` / `v_accvgpr_write_b32` moves.
- **The big trap:** at large tiles (e.g. >128×256 with 16×16 MFMA + SW pipelining) the compiler inserts
  *many unnecessary* `v_accvgpr_read/write` in the inner loop and/or spills — performance silently drops
  back to a 128×128-class kernel. **Symptom:** TFLOP/s plateaus or regresses as you grow the tile.
  **Check:** disassemble (`--save-temps` / `-S`) and grep the hot loop for `accvgpr` and `scratch_`.
- On gfx942 the compiler can keep MFMA accumulators in **VGPRs** (no AGPR) when register pressure
  allows, avoiding the move tax — but this competes with everything else for the 512-VGPR budget.
- **Co-issue:** a SIMD can overlap an in-flight MFMA (which occupies the matrix pipe for its cycle count)
  with independent VALU/VMEM/LDS instructions from the same or another wave. The MFMA pipe is in-order;
  back-to-back dependent MFMAs serialize, so the K-loop must interleave **independent** MFMAs across the
  MRepeat×NRepeat tiles to keep the pipe full.
- **VGPR banking:** the VGPR file is banked; the compiler handles operand-bank assignment, but very wide
  vector ops and bad operand patterns can cause bank stalls — rarely hand-tunable, mostly a "give the
  scheduler room" concern.

---

## 4. `s_waitcnt` and instruction scheduling — the heart of overlap

CDNA memory ops are **asynchronous**. The hardware tracks *outstanding* ops in counters; `s_waitcnt`
blocks until a counter **drops to a given value** (NOT "wait N instructions"):

| counter | tracks | typical use |
|---|---|---|
| **vmcnt(N)** | outstanding VMEM (`buffer_load`/`global_load`) | wait until ≤N global loads pending |
| **lgkmcnt(N)** | outstanding LDS (`ds_*`) + scalar (`s_load`) + msg | wait until ≤N LDS/scalar pending |
| **expcnt(N)** | outstanding exports | graphics; rare in compute |

Omitted fields default to **max** (i.e. "don't wait on this counter"). `s_waitcnt vmcnt(0) lgkmcnt(0)`
= full memory fence.

### 4.1 The relaxed-count trick (software pipelining)

The whole point of the "wait for N *remaining*" semantics is overlap. To overlap the **tail** of LDS
reads with MFMA:

```asm
    ds_read_b128  v[8:11],  v20            ; load A frag for next MFMA
    ds_read_b128  v[12:15], v24            ; load B frag for next MFMA
    s_waitcnt     lgkmcnt(1)               ; proceed once all-but-ONE ds_read returned
    v_mfma_f32_16x16x16_bf16 a[0:3], v[0:3], v[4:7], a[0:3]  ; compute on PREVIOUS frag
    s_waitcnt     lgkmcnt(0)               ; now the last ds_read is in; rotate buffers
```

This is exactly the pattern the LLVM scheduler emits for gfx942 MFMA loops (`s_waitcnt lgkmcnt(1)`
right before the `v_mfma`). The compiler's software pipeliner + the CK/ck_tile pipelines (v3/v4) generate
this for you; you only hand-write it for a micro-kernel or when the scheduler does it badly.

### 4.2 `s_setprio` and scheduling barriers

CK's `HotLoopScheduler()` emits `s_setprio` (raise wave priority during the compute burst so the MFMA
issuer isn't starved) and `sched_barrier` / `sched_group_barrier` intrinsics
(`__builtin_amdgcn_sched_barrier(mask)`, `__builtin_amdgcn_sched_group_barrier(mask, size, sync_id)`) to
**pin the interleave** of `buffer_load` / `ds_read` / `v_mfma` so the compiler can't reorder it away. The
`SchedGroupMask::MFMA` bit is how the scheduler identifies MFMA ops for software pipelining — **important
caveat: hand-written MFMA in inline asm is NOT recognized by `SchedGroupMask`, so it defeats the
software pipeliner.** Prefer intrinsics over asm for MFMA itself.

---

## 5. Memory path & LDS bank conflicts

Classic gfx942 GEMM dataflow:
`buffer_load` (global→VGPR) → `ds_write` (VGPR→LDS) → `ds_read` (LDS→VGPR) → `v_mfma`.

- **Direct-to-LDS (DGL):** `buffer_load ... lds` moves global→LDS in one op, skipping the VGPR stage,
  the `ds_write`, and all the cooperative-copy index math — saving VGPRs and instructions. On gfx942 the
  cache-swizzle support exists but the conventional two-step staging is still common; the **fully
  working** DGL data path for scaled-GEMM operands is primarily a **gfx950** story. Use it where the
  toolchain supports it (it's a major occupancy win).
- **LDS bank conflicts:** LDS has 32 banks × 4B. The `ds_read` that feeds MFMA must avoid lane→bank
  collisions. Two fixes:
  - **XOR swizzle** the LDS *write* address (`make_xor_transform` in CK) so the later `ds_read` is
    conflict-free — no extra LDS, the preferred path.
  - **LDS padding** (add a column) — simpler, doesn't touch the DMA lowering, but **costs extra LDS →
    lower occupancy**, and the DMA transfer must not cross a pad boundary.
- AMD's MI300X tuning guide: padding-to-avoid-conflicts "usually leads to extra LDS usage which might
  reduce occupancy" — prefer XOR swizzle when you can.

---

## 6. Inline GCN asm & intrinsics in HIP

### 6.1 When to use which

| approach | use it for | avoid for |
|---|---|---|
| **`__builtin_amdgcn_*` intrinsics** | MFMA, `ds_*` perms, `buffer_load`, sched barriers, ballot/permute | — (this is the default; scheduler-friendly) |
| **inline `asm volatile`** | a tight hand-scheduled micro-loop, latency probes, forcing a specific encoding | **MFMA** (breaks SchedGroupMask pipelining), anything the compiler schedules well |
| **full hand-written asm** | rarely — a peak micro-kernel where you out-schedule LLVM | maintainability; almost never worth it for production |

**Verdict:** hand-asm is worth it only when (a) you can prove the compiler's schedule is suboptimal via
disassembly, and (b) the kernel is small/hot enough that the maintenance cost pays off. For MFMA loops,
use intrinsics + `sched_group_barrier` to *guide* the compiler instead of replacing it.

### 6.2 Inline asm pitfalls (gfx942-specific, all real bugs)

```cpp
// WRONG: separate asm blocks collapse into the same registers / get reordered.
// RIGHT: one block, distinct early-clobber outputs, explicit base operand, memory clobber.
float4 v0, v1;
asm volatile(
    "global_load_dwordx4 %0, %2, off\n"
    "global_load_dwordx4 %1, %3, off\n"
    "s_waitcnt vmcnt(0)\n"
    : "=&v"(v0), "=&v"(v1)          // '&' = early-clobber: outputs must not alias inputs
    : "v"(ptr0), "v"(ptr1)
    : "memory");                    // prevent reorder/elimination
```

Rules:
1. **One asm block** for ordered sequences — don't rely on multiple `volatile` blocks for ordering.
2. **Early-clobber `"=&v"`** when an output reg must not be reused as an input (the classic
   "first load clobbers v[0:1], later loads break" bug).
3. **`"memory"` clobber + `volatile`** around timing/sync code, or `-O2`+ will reorder/delete it
   (the `s_memtime` latency-probe gives wrong results without it).
4. **Do not put MFMA in asm** if you want SW pipelining — use the intrinsic.

### 6.3 Building & inspecting

```bash
# emit gfx942 assembly to verify the hot loop (look for accvgpr moves, scratch spills, waitcnt counts)
/opt/rocm/bin/amdclang++ -x hip --offload-device-only --offload-arch=gfx942 -O3 -S kern.cpp -o kern.s
# or keep all temps from a normal build:
hipcc --offload-arch=gfx942 -O3 --save-temps kern.cpp -o kern
# grep the disassembly:
#   grep -E 'v_mfma|s_waitcnt|accvgpr|ds_read|buffer_load|scratch_' kern.s
```
Use `--offload-arch=gfx942` (not `gfx90a`); set `__launch_bounds__(256)` to cap VGPRs and control
occupancy; type fragments with `__attribute__((ext_vector_type(N)))` / `vector_size`.

---

## 7. Low-level perf checklist (gfx942)

| check | how | fix if bad |
|---|---|---|
| MFMA size | is it 16×16×16? | switch from 32×32×8 → 16×16×16 (power-limited FLOPs) |
| accvgpr moves in hot loop | grep `.s` for `v_accvgpr` | shrink tile / let acc stay in VGPR / fewer MRepeat |
| register spills | grep `scratch_` | reduce tile, KPerBlock, or prefetch depth; lower `__launch_bounds__` |
| load/MFMA overlap | look for `s_waitcnt lgkmcnt(1)` before `v_mfma` | add prefetch / use v3+ pipeline / `sched_group_barrier` |
| LDS bank conflicts | rocprof `LDSBankConflict` / profiler | XOR-swizzle LDS (preferred) or pad |
| occupancy | `rocprofv3` / `--amdgpu-spgr`; waves/SIMD | cut VGPR/AGPR/LDS; DGL `buffer_load...lds` |
| direct-to-LDS | is `buffer_load ... lds` used? | enable DGL (esp. gfx950) to skip VGPR staging |
| MFMA pipe full | dependent back-to-back MFMAs? | interleave independent MRepeat×NRepeat MFMAs |

---

## Sources
- AMD Instinct MI300 "CDNA3" ISA Reference Guide (Ch.7 Matrix Arithmetic; waitcnt; encodings; Aug 2025): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- ROCm/amd_matrix_instruction_calculator (per-instruction M/N/K, cycles, FLOPs, register & lane layouts): https://github.com/ROCm/amd_matrix_instruction_calculator
- Matrix Core Programming on AMD CDNA3/CDNA4 (intrinsics, fp16/fp8 examples, cycle table) — salykova: https://salykova.github.io/matrix-cores-cdna
- LLVM gfx940/gfx942 instruction syntax (v_mfma_*, ds_*, buffer_load, waitcnt): https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX940.html
- LLVM gfx9 `s_waitcnt` semantics (vmcnt/lgkmcnt/expcnt "wait for remaining"): https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
- LLVM issue #131954 — large MFMA tiles → spurious v_accvgpr_read/write moves & spills: https://github.com/llvm/llvm-project/issues/131954
- HIP issue #3333 — inline GCN asm multi-load register clobber pitfalls: https://github.com/ROCm/HIP/issues/3333
- iree issue #23765 — direct-to-LDS (`buffer_load ... lds`) + XOR-swizzle / LDS-pad bank-conflict tradeoff: https://github.com/iree-org/iree/issues/23765
- ROCm Blog — Measuring Max-Achievable FLOPs Part 2 (mfma_16x16 vs 32x32 power/clock on MI300X): https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
- AMD Instinct MI300X workload optimization (LDS padding vs occupancy guidance): https://rocm.docs.amd.com/en/docs-6.1.2/how-to/tuning-guides/mi300x/workload.html
- HIP compilers doc (amdclang++, offload-arch, intrinsics): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html
