# Low-Level HIP / AMDGCN Intrinsics for CDNA3 — MI300X / gfx942

> Scope: the hand-written-kernel layer for AMD Instinct MI300X (CDNA3, `gfx942`): MFMA matrix
> intrinsics (`__builtin_amdgcn_mfma_*`), buffer/global load-store builtins + buffer resource
> descriptors, LDS builtins (`__builtin_amdgcn_ds_*`), direct-to-LDS / async loads, barriers and
> wait-counters (`s_barrier`, `s_waitcnt`), the scheduling builtins (`sched_barrier`,
> `sched_group_barrier`, `iglp_opt`), and a complete MFMA microkernel with LDS double-buffering.
> Companion to `hip_cpp.md` (HIP basics) and `triton_amd.md`.

---

## 0. The register classes you must reason about

| Class | What | CDNA3 budget | Role in MFMA |
|---|---|---|---|
| **VGPR** | per-lane vector regs | 512 / lane-slot, granularity 16 | hold A/B operands, addresses |
| **AGPR** | accumulation regs | up to 256 / lane (shared budget with VGPR) | hold MFMA C/D accumulators |
| **SGPR** | scalar (wave-uniform) | ~102 usable | buffer descriptors, loop counters |

On CDNA3, MFMA accumulators live in **AGPRs**; moving them to/from VGPRs costs `v_accvgpr_read/write`.
A common pipelining bug (large tiles + software pipelining) is the compiler inserting spurious
`v_accvgpr_read_b32`/`v_accvgpr_write_b32` in the inner loop — kills perf back to small-tile levels
(LLVM issue #131954). Fix is the "tied" accumulator flag (input accum tied to output accum) which CK
relies on; with raw intrinsics, keep accumulators in a stable `__attribute__((vector_size))` variable
across iterations so the backend keeps them in AGPRs.

---

## 1. MFMA matrix intrinsics

General form (wavefront-wide, executes across all 64 lanes):
```
d = __builtin_amdgcn_mfma_<CDFmt>_MxNxK<ABFmt>(a, b, c, cbsz, abid, blgp);
```
- `MxNxK` = matrix shape; `ABFmt` = A/B input type; `CDFmt` = C/D type.
- `a`,`b`,`c` are **per-lane vector slices** of the full tiles (each lane holds `M*K/64`,
  `K*N/64`, `M*N/64` elements respectively).
- `cbsz, abid, blgp` = broadcast control (CBSZ = control broadcast size, ABID = A broadcast id,
  BLGP = B lane group pattern). **Set all to 0** for standard GEMM.

### gfx942 MFMA instruction table (the ones that matter for inference)

| Intrinsic | M×N×K | A/B | C/D | A elems/lane | B | C | cycles |
|---|---|---|---|---|---|---|---|
| `mfma_f32_16x16x16f16` | 16×16×16 | fp16 | fp32 | 4 | 4 | 4 | 16 |
| `mfma_f32_32x32x8f16` | 32×32×8 | fp16 | fp32 | 4 | 4 | 16 | — |
| `mfma_f32_16x16x16bf16_1k` | 16×16×16 | bf16 | fp32 | 4 | 4 | 4 | 16 |
| `mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | fp8(e4m3 fnuz) | fp32 | 8 | 8 | 4 | — |
| `mfma_f32_32x32x16_fp8_fp8` | 32×32×16 | fp8 fnuz | fp32 | 8 | 8 | 16 | — |
| `mfma_f32_16x16x32_fp8_bf8` | 16×16×32 | fp8/bf8 fnuz | fp32 | 8 | 8 | 4 | — |
| `mfma_i32_16x16x32_i8` | 16×16×32 | int8 | int32 | 8 | 8 | 4 | — |

Notes:
- **fp8 on gfx942 is FNUZ** (e4m3 fnuz / e5m2 "bf8"). The `_fp8_fp8` / `_fp8_bf8` suffixes pick A/B
  formats independently. (CDNA4/gfx950 adds the OCP `scale_f32_*_f8f6f4` block-scaled variants for
  fp8/fp6/fp4 — not on gfx942.)
- fp8 intrinsics expect the 8-element fp8 operand packed and **cast to `long`** (64-bit) per lane.
- Use the **AMD Matrix Instruction Calculator** to get the exact lane→element map for any instruction
  rather than reverse-engineering it.

### Single-MFMA example (fp16 16×16×16), per-lane layout
```cpp
using fp16_t   = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;

__global__ void mfma_16x16x16(const fp16_t* A, const fp16_t* B, float* C) {
    int lane = threadIdx.x;                 // 0..63
    fp16x4_t a_reg, b_reg;
    fp32x4_t c_reg = {0, 0, 0, 0};          // C/D accumulator -> AGPRs

    // A is 16x16 row-major: lane holds 4 contiguous K for its row group
    a_reg = *reinterpret_cast<const fp16x4_t*>(A + 4*(lane/16) + 16*(lane%16));
    // B is 16x16 row-major: gather 4 K-rows for this column
    for (int i = 0; i < 4; ++i)
        b_reg[i] = B[i*16 + lane%16 + (lane/16)*64];

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; ++i)
        C[i*16 + lane%16 + (lane/16)*64] = c_reg[i];
}
// launch: mfma_16x16x16<<<1, 64>>>(A, B, C);   // one wavefront
```
The lane mapping (`4*(lane/16) + 16*(lane%16)` for A) is the wavefront-distributed layout the matrix
core requires; reads/writes to global/LDS must respect it.

---

## 2. Buffer resource descriptors & buffer load/store builtins

`buffer_*` instructions use a **128-bit resource descriptor** (V#) in SGPRs: base address, stride,
num-records (bounds), and format flags. Benefits over plain `global_load`: **hardware bounds
checking** (out-of-range lanes return 0 / drop writes — no branch needed for masking) and sometimes
better address generation. Triton's `knobs.amd.use_buffer_ops` emits these; by hand:

```cpp
// Make a buffer resource descriptor (V#) for a float* of `num` elements.
__device__ inline int32x4_t make_buffer_rsrc(const void* base, uint32_t num_bytes) {
    int32x4_t rsrc;
    uint64_t addr = reinterpret_cast<uint64_t>(base);
    rsrc[0] = (uint32_t)(addr & 0xFFFFFFFF);     // base lo
    rsrc[1] = (uint32_t)(addr >> 32) & 0xFFFF;   // base hi (48-bit VA) | stride bits
    rsrc[2] = num_bytes;                          // num records (bounds)
    rsrc[3] = 0x00020000;                         // format/flags (gfx942 default OOB=0)
    return rsrc;
}

// 128-bit (dwordx4) buffer load with per-lane voffset; soffset=0, aux=0.
float4 v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, /*soffset=*/0, /*aux=*/0);
// store:
__builtin_amdgcn_raw_buffer_store_b128(value, rsrc, voffset, 0, 0);
```
- `raw_buffer_load_b32/b64/b128` (and `_store_`) cover 4/8/16-byte transactions. Prefer **b128**
  (`global_load_dwordx4` equivalent) in inner loops for bandwidth.
- Out-of-bounds lanes (`voffset >= num_records`) safely return 0 — this replaces predication masks in
  GEMM tail handling.
- The construction of the `0x00020000` flags word is gfx-specific; use AMD's `__amdgcn_make_buffer_rsrc`
  builtin where available instead of hardcoding.

Plain global builtins (no descriptor): `__builtin_amdgcn_global_load_*` / the compiler emits
`global_load_dwordx4` from `*reinterpret_cast<float4*>(ptr)` when the pointer is `__restrict__` and
16-byte aligned.

---

## 3. LDS builtins — `__builtin_amdgcn_ds_*`

LDS (64 KB/CU, 32 banks × 4 B). For peak bandwidth, **128-bit** LDS access (`ds_read_b128` /
`ds_write_b128`).

```cpp
extern __shared__ float lds[];
// Vectorized writes/reads usually emit ds_*_b128 automatically:
*reinterpret_cast<float4*>(&lds[off]) = v;                 // -> ds_write_b128
float4 r = *reinterpret_cast<float4*>(&lds[off]);          // -> ds_read_b128

// Explicit builtins (when you need to force a form / use swizzled offsets):
//   __builtin_amdgcn_ds_bpermute(addr, data)  -> cross-lane via LDS (byte addr = lane*4)
int x = __builtin_amdgcn_ds_bpermute(srcLane << 2, val);  // gather from srcLane
int y = __builtin_amdgcn_ds_permute (dstLane << 2, val);  // scatter to dstLane
// ds_swizzle: fixed permutation patterns within 32-lane groups
int z = __builtin_amdgcn_ds_swizzle(val, /*pattern=*/0x1F);
```
- `ds_bpermute`/`ds_permute` move data *through the LDS crossbar* (no LDS storage consumed) — useful
  for transposes/reductions without `__shfl` when you need arbitrary lane gather. Byte addressing:
  multiply lane index by 4.
- **Bank conflicts**: with 64-lane waves over 32 banks, pad/​swizzle the layout so the 64 lanes hit
  distinct banks across the two access phases (see `hip_cpp.md` §4). XOR-swizzle is the standard GEMM
  fix — and is **required** for direct-to-LDS (removing it caused 201M bank conflicts, −28% TFLOPS in
  the IREE study).

---

## 4. Direct-to-LDS & async loads (skip the register staging)

`global_load_lds` / `buffer_load ... lds` moves data **straight from global memory into LDS**,
bypassing VGPRs entirely — eliminates the `ds_write` and the staging registers (frees VGPRs for more
occupancy, removes a whole instruction class from the loop).

```cpp
// Each lane contributes one element; the 64-lane group fills a contiguous LDS chunk.
//   load width in bytes: 1, 2, or 4 (4 preferred). 64 lanes * 4B = 256B per call.
__builtin_amdgcn_global_load_lds(
    /*global*/ thread_global_addr,     // per-lane global address (gather: need NOT be coalesced)
    /*lds   */ subgroup_lds_addr,      // must be coalesced across the subgroup
    /*size  */ 4,                      // bytes per lane
    /*offset*/ 0,
    /*aux   */ 0);
```
- Availability: `global_load_lds` is gated to `gfx940-insts` (gfx942 family). The newer unified
  `llvm.amdgcn.load.to.lds` intrinsic lowers correctly on **gfx950**; on gfx942 use
  `global_load_lds`. Scratch→LDS exists post-gfx942 but only via inline asm.
- The LDS address must be **coalesced** across the subgroup; global addresses may be scattered (acts
  as a gather). Pair with `s_waitcnt vmcnt(0)` before reading the LDS.
- This is what Triton's `knobs.amd.use_async_copy` and CK's pipelined loaders use to overlap the next
  tile's load with the current tile's MFMAs.

---

## 5. Barriers & wait counters

CDNA3 memory ops are **asynchronous**; correctness requires explicit counters/barriers.

| Builtin / instr | Meaning |
|---|---|
| `__syncthreads()` → `s_barrier` | workgroup barrier (all waves in block) |
| `__builtin_amdgcn_s_barrier()` | raw block barrier |
| `__builtin_amdgcn_wave_barrier()` | single-wave barrier (`s_barrier` within wave scope) |
| `__builtin_amdgcn_s_waitcnt(n)` | wait on encoded counters |
| `s_waitcnt vmcnt(k)` | ≤ k vector-memory (global/buffer) ops outstanding |
| `s_waitcnt lgkmcnt(k)` | ≤ k LDS/GDS/const/message ops outstanding |
| `__builtin_amdgcn_s_dcache_inv()` | invalidate scalar cache |

Pattern for direct-to-LDS then compute:
```cpp
__builtin_amdgcn_global_load_lds(g, l, 4, 0, 0);   // issue async load to LDS
asm volatile("s_waitcnt vmcnt(0)");                 // wait until it lands
__builtin_amdgcn_s_barrier();                        // all waves see the LDS data
float4 a = *reinterpret_cast<float4*>(&lds[off]);    // ds_read_b128
asm volatile("s_waitcnt lgkmcnt(0)");                // wait LDS read before use
```
The compiler usually inserts `s_waitcnt` for you; you only hand-place them in microkernels where you
are also controlling scheduling (§6).

---

## 6. Instruction scheduling builtins (build a software pipeline)

To overlap MFMA with VMEM/LDS, AMDGCN exposes scheduler hints. They constrain the LLVM scheduler so
you can hand-build the interleave the default heuristic misses.

```cpp
__builtin_amdgcn_sched_barrier(mask);
// Hard barrier for instruction movement; mask = categories ALLOWED to cross. 0 = block all.

__builtin_amdgcn_sched_group_barrier(mask, size, sync_id);
// Form a scheduling group of `size` instructions of category `mask`, ordered by `sync_id`.
// Instructions are selected bottom-up from the barrier. Groups with the same sync_id are ordered.

__builtin_amdgcn_iglp_opt(variant);
// Apply a predefined IGLP (instr group-level parallelism) pipeline. variant 0/1 = built-in strategies.
```

**`SchedGroupMask` category bits** (as used in Composable Kernel GEMM pipelines):

| Mask | Category |
|---|---|
| `0x002` | VALU |
| `0x008` | **MFMA** |
| `0x020` | **VMEM read** (global/buffer load) |
| `0x040` | VMEM write |
| `0x100` | DS read |
| `0x200` | **DS write** |

Hand-built interleave (one "stage" of a GEMM loop — issue loads, then MFMAs, then LDS writes in a
controlled ratio so the MFMA units never starve):
```cpp
#pragma unroll
for (int i = 0; i < UNROLL; ++i) {
    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);   // 1 VMEM read  (prefetch next tile)
    __builtin_amdgcn_sched_group_barrier(0x008, 4, 0);   // 4 MFMA       (compute current)
    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);   // 1 DS write   (stage prefetched)
    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);   // 1 DS read    (feed next MFMA)
}
```
- Use only after the default scheduler proves inadequate (verify via ISA). Wrong ratios *hurt*.
- Critical interaction with MFMA: for the pipeline to work, the MFMA must be detected as
  `SchedGroupMask::MFMA` and the accumulator must be **tied** (or it injects `v_accvgpr_*` moves /
  spills — issue #131954). Inline asm alone doesn't give the tied semantics.

---

## 7. Cross-lane permute / swizzle builtins (besides `__shfl`)

| Builtin | Use |
|---|---|
| `__builtin_amdgcn_ds_bpermute(addr,val)` | gather from arbitrary lane (via LDS crossbar) |
| `__builtin_amdgcn_ds_permute(addr,val)` | scatter to arbitrary lane |
| `__builtin_amdgcn_ds_swizzle(val,patt)` | fixed swizzle within 32-lane group |
| `__builtin_amdgcn_mov_dpp(...)` / `update_dpp` | DPP: cheap neighbor shifts (row/bcast) |
| `__builtin_amdgcn_permlane16 / permlanex16` | 16-lane / cross-16 permute (CDNA3) |
| `__builtin_amdgcn_readlane / readfirstlane` | broadcast a lane's value to scalar/all |

DPP and `permlane` are cheaper than `ds_*permute` for fixed neighbor patterns (used in fast wave
reductions); `ds_bpermute` is the general fallback for arbitrary gather.

---

## 8. Worked example — MFMA microkernel with LDS double-buffering

A wavefront-level fp16 GEMM tile using `v_mfma_f32_16x16x16` with **two LDS buffers** so the next
K-block loads while the current one feeds the matrix core. Skeleton (one wave computes a 16×16 ×
several-K accumulation; real kernels tile multiple MFMAs per wave). Comments mark every AMD-specific
choice.

```cpp
using fp16_t   = _Float16;
using fp16x4_t = __attribute__((vector_size(4*sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4*sizeof(float)))) float;

#define BK 16                 // K per MFMA step
#define KTILES /*K/BK*/ 64    // number of K-blocks

__global__ void __launch_bounds__(64, 8)              // 1 wave; allow high occupancy
mfma_gemm_db(const fp16_t* __restrict__ A,
             const fp16_t* __restrict__ B,
             float* __restrict__ C, int M, int N, int K) {
    int lane = threadIdx.x;                            // 0..63

    // Two LDS buffers for A and B tiles (16x16 fp16 each = 512B; ×2 buffers ×2 operands = 2KB)
    __shared__ fp16_t As[2][16*16];
    __shared__ fp16_t Bs[2][16*16];

    fp32x4_t acc = {0, 0, 0, 0};                       // stays in AGPRs across the loop (tied)

    int buf = 0;
    // ---- prologue: load K-block 0 into buffer 0 ----
    auto load_tile = [&](int kblk, int b) {
        // cooperative: each lane loads its share of the 16x16 tile into LDS
        // (use global_load_lds on gfx942 to skip VGPR staging — shown inline below)
        for (int e = lane; e < 16*16; e += 64) {
            As[b][e] = A[/* tile-relative index for kblk */ e];
            Bs[b][e] = B[/* tile-relative index for kblk */ e];
        }
    };
    load_tile(0, 0);
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    #pragma unroll 1
    for (int k = 0; k < KTILES; ++k) {
        int nbuf = buf ^ 1;
        // ---- issue next tile's load into the OTHER buffer (overlaps with MFMA below) ----
        if (k + 1 < KTILES) load_tile(k + 1, nbuf);

        // ---- read current tile from LDS (ds_read_b128 via fp16x4) ----
        fp16x4_t a_reg = *reinterpret_cast<fp16x4_t*>(&As[buf][4*(lane/16) + 16*(lane%16)]);
        fp16x4_t b_reg;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            b_reg[i] = Bs[buf][i*16 + lane%16 + (lane/16)*64];

        // ---- MFMA: accumulate into AGPRs ----
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, acc, 0, 0, 0);

        // ---- scheduling hint: keep MFMA fed while next load is in flight ----
        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);   // VMEM read (next tile)
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);   // MFMA
        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);   // DS write (stage next tile)

        asm volatile("s_waitcnt vmcnt(0)");                  // next tile landed
        __builtin_amdgcn_s_barrier();                         // publish to all lanes
        buf = nbuf;                                           // swap buffers
    }

    // ---- epilogue: write 16x16 fp32 result ----
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        C[i*16 + lane%16 + (lane/16)*64] = acc[i];
}
```

Why each AMD detail matters:
- `acc` is a single `fp32x4_t` carried across iterations → the backend keeps it in **AGPRs**, avoiding
  `v_accvgpr_*` churn (issue #131954). Do **not** copy it in/out of a temp each iteration.
- **Two LDS buffers** (`[2]`) overlap the next-tile load with the current MFMA — the core idea of
  software pipelining. With `global_load_lds` the staging VGPRs disappear entirely.
- `ds_read` via `fp16x4_t` → 64-bit/128-bit LDS reads, not scalar `ds_read_b16`.
- `s_waitcnt vmcnt(0)` + `s_barrier` enforce that the LDS buffer is fully written before any lane
  reads it.
- `__launch_bounds__(64, 8)` requests high occupancy; check `.private_segment_fixed_size == 0`
  (no scratch spill) in the ISA, else lower the wave/EU target.

> Production note: hand-rolled MFMA microkernels are rarely worth it vs **rocWMMA** (C++ WMMA-style
> wrapper over MFMA) or **Composable Kernel** (templated, fully pipelined CDNA GEMM/attention),
> which already encode the tied-accumulator + sched-group-barrier + double-buffer patterns correctly.
> Reach for raw intrinsics only when those can't express your fusion.

---

## 9. Verification checklist (ISA-level)

Build with `--save-temps` / `AMDGCN_ENABLE_DUMP=1` and confirm in the inner loop:

| Look for | Good | Bad (retune) |
|---|---|---|
| Global loads | `global_load_dwordx4` / `buffer_load_dwordx4` | `global_load_dword` (scalar) |
| LDS access | `ds_read_b128` / `ds_write_b128` | `ds_read_b32` |
| MFMA | `v_mfma_f32_16x16x16` present, dense | sparse, gaps = starved |
| Accumulator | accumulators in `a[0:n]` (AGPR) | `v_accvgpr_read/write` in loop |
| Scratch | `.private_segment_fixed_size: 0` | nonzero → spilling to HBM |
| Waitcnt | minimal, overlapped | `s_waitcnt vmcnt(0)` after every load = no overlap |

---

## Sources

1. Matrix Core Programming on AMD CDNA3 and CDNA4 (MFMA intrinsic format, per-lane fp16/fp8 layouts, cbsz/abid/blgp) — ROCm Blogs: <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
2. AMD matrix cores deep dive (MFMA intrinsic list, lane→element mapping, cycle counts) — salykova.github.io: <https://salykova.github.io/matrix-cores-cdna>
3. `__builtin_amdgcn_sched_group_barrier` (mask/size/sync_id semantics, rules extension) — LLVM review D128158: <https://reviews.llvm.org/D128158>
4. SchedGroupMask usage (0x008 MFMA, 0x020 VMEM read, 0x200 DS write) in a CDNA GEMM pipeline — ROCm Composable Kernel source: <https://rocm.docs.amd.com/projects/composable_kernel/en/latest/>
5. MFMA + software pipelining AGPR spill / tied accumulator — llvm/llvm-project issue #131954: <https://github.com/llvm/llvm-project/issues/131954>
6. Direct-to-LDS `global_load_lds` / `buffer_load ... lds`, swizzle requirement, gfx942 vs gfx950 — iree-org/iree issue #23765 & AMDGPU backend: <https://github.com/iree-org/iree/issues/23765>
7. User Guide for AMDGPU Backend (buffer resource descriptors, ds builtins, s_waitcnt vmcnt/lgkmcnt, sched builtins) — LLVM docs: <https://llvm.org/docs/AMDGPUUsage.html>
8. AMD Matrix Instruction Calculator (exact register/element maps for every MFMA) — ROCm/amd_matrix_instruction_calculator: <https://github.com/ROCm/amd_matrix_instruction_calculator>
