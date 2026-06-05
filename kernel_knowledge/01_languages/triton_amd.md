# Triton on AMD (ROCm backend) — MI300X / CDNA3 / gfx942

> Scope: how Triton compiles and runs on AMD Instinct MI300X (CDNA3, `gfx942`), what is
> *different* from the NVIDIA path, and the AMD-specific patterns an agent must use to write
> high-performance Triton kernels for LLM inference. For the full autotuning knob set see
> `triton_autotune_amd.md`. For the underlying matrix hardware see `hip_intrinsics_async.md`.

---

## 0. TL;DR cheat sheet (read this first)

| Topic | NVIDIA (for contrast) | AMD MI300X / CDNA3 (`gfx942`) |
|---|---|---|
| Warp / wavefront width | 32 lanes | **64 lanes** (`num_warps=N` → `N*64` threads) |
| Matrix engine | Tensor Core (`mma`/`wgmma`) | **Matrix Core / MFMA** (`v_mfma_*`) via `tl.dot` |
| MFMA tile (`matrix_instr_nonkdim`) | n/a | **16** (mfma_16x16, preferred) or 32 (mfma_32x32) |
| Shared memory | 228 KB/SM programmable (H100) | **64 KB LDS / CU** (4× smaller — budget tightly) |
| VGPRs | 65536 / SM (256/thread cap) | **512 VGPRs / SIMD lane-slot**, alloc granularity 16 |
| FP8 matrix dtype | OCP `e4m3fn` / `e5m2` | **FNUZ** `e4m3fnuz` / `e5m2fnuz` (CDNA3 only!) |
| `num_stages` for GEMM | 3–4 typical | **2** single GEMM, **1** for fused 2-GEMM (FA) |
| `tf32` dot input | yes | `tf32` allowed **CDNA3 only**, else `ieee` |
| Backend dir | `third_party/nvidia` | `third_party/amd` |

**The five AMD mistakes that kill Triton perf:**
1. Assuming `warpSize==32` anywhere (grid/occupancy math, masks). It is **64**.
2. Over-budgeting `num_warps` → VGPR spill to scratch (HBM). On AMD, spills are catastrophic (3–5× slowdowns); cut warps first.
3. Using OCP `float8_e4m3fn` in `tl.dot` → `Unsupported conversion from 'f8E4M3FN'` on gfx942. Convert to **fnuz**.
4. Leaving `num_stages=3+` for a single GEMM — on AMD that pipelines worse than `num_stages=2`.
5. Forgetting LDS is only 64 KB — large `BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N` tiles silently drop occupancy or fail to compile.

---

## 1. The AMD Triton backend: what it is and how to get it

There are effectively two distributions, both built on the same upstream code:
- **Upstream `triton-lang/triton`** — the AMD backend lives in `third_party/amd/` and is built by default. CDNA3 is a first-class target.
- **`ROCm/triton`** (AMD's fork / staging) — carries AMD-specific perf patches and tuning utilities (e.g. `occ.sh`) ahead of upstream. ROCm PyTorch wheels ship a Triton built from here.

`third_party/amd/` layout (upstream):

```
third_party/amd/
├── backend/        # compiler.py (HIPOptions, pass pipeline), driver.py (HIP runtime, launch)
├── include/        # TritonAMDGPU dialect headers
├── lib/            # MLIR passes: TritonGPU -> TritonAMDGPU -> AMDGCN/LLVM lowering, MFMA conv
├── language/hip/   # AMD device-library hooks
├── python/         # python bindings (triton.backends.amd)
└── test/
```

Pick the target with the canonical CDNA3 arch string:

```python
import torch, triton
print(torch.cuda.get_device_properties(0).gcnArchName)   # 'gfx942' on MI300X / MI300A
# Triton auto-detects the arch from the active HIP device; no manual flag needed.
```

> `gfx942` covers MI300X, MI300A (APU), MI325X. `gfx950` is CDNA4 (MI350/MI355X). Most of this
> doc is gfx942; CDNA4 notes are called out inline.

---

## 2. Programming model — identical syntax, different hardware mapping

Triton's Python API is **the same** on AMD and NVIDIA: `tl.program_id`, `tl.arange`, `tl.load/store`,
block/`make_block_ptr`, `tl.dot`, `@triton.jit`, `@triton.autotune`. What changes is the *lowering*.

### 2.1 program_id / grid
A Triton "program" == one workgroup (HIP block) == `num_warps` wavefronts of **64 lanes** each.
`tl.program_id(axis)` maps to `blockIdx`. Grid is the launch dim. **MI300X has 304 CUs**; aim for
**≥ 1024 programs** in the grid so the scheduler can hide latency and balance across the 8 XCDs.

### 2.2 The compilation pipeline (TritonGPU → AMDGCN)
```
@triton.jit (Python AST)
  → Triton IR (TTIR)               # arch-independent
  → TritonGPU IR (TTGIR)           # blocked/MFMA layouts assigned here
  → TritonAMDGPU IR                # AMD-specific: MFMA dot conversion, LDS layouts,
                                   #   stream-pipeliner, sched-group barriers
  → LLVM IR (AMDGPU target)        # adds amdgpu-waves-per-eu, denormal-fp-math attrs
  → AMDGCN ISA (gfx942)            # v_mfma_*, ds_read/ds_write, global_load_dwordx4, buffer_load
  → HSACO (code object)            # loaded by the HIP runtime
```
Inspect any stage with env vars (see §7). The key AMD-only stage is the **TritonAMDGPU** dialect
where `tl.dot` becomes an `MFMA` layout op and the loop is software-pipelined.

### 2.3 `tl.dot` → MFMA (the heart of GEMM/attention)
`tl.dot(a, b, acc)` lowers to a sequence of `v_mfma_f32_*` instructions. The chosen MFMA shape is
controlled by `matrix_instr_nonkdim` (16 or 32) and the dtype:

| `tl.dot` input dtype | MFMA instruction (gfx942) | K per instr | recommended `BLOCK_K` |
|---|---|---|---|
| fp16 / bf16 | `v_mfma_f32_16x16x16` (nonkdim=16) | 16 | 32–64 |
| fp16 / bf16 | `v_mfma_f32_32x32x8` (nonkdim=32) | 8 | 32–64 |
| fp8 (fnuz) | `v_mfma_f32_16x16x32_fp8_fp8` | 32 | 64–128 |
| fp8 (fnuz) | `v_mfma_f32_32x32x16_fp8_fp8` | 16 | 64–128 |
| int8 | `v_mfma_i32_16x16x32_i8` | 32 | 64–128 |

**Rule of thumb (AMD docs):** for GEMM, `mfma_16x16` (`matrix_instr_nonkdim=16`) usually beats
`mfma_32x32` even at large tiles — better power efficiency and finer scheduling granularity.

Each MFMA is **wavefront-wide**: the 64 lanes collectively hold the A/B/C tiles in VGPRs/AGPRs.
You do not write MFMA by hand in Triton — `tl.dot` does it — but the layout it picks dictates VGPR
pressure (see occupancy, §5).

### 2.4 Dot input precision on AMD
```python
acc = tl.dot(a, b, acc, input_precision="ieee")   # default-safe on AMD
# input_precision="tf32"   # allowed on CDNA3 ONLY; rounds fp32 inputs to tf32 for the MFMA
```
On AMD the valid `input_precision` values are `"ieee"` and (CDNA3-only) `"tf32"`. NVIDIA's
`"tf32x3"` is not an AMD path.

---

## 3. Wave64: the single most important AMD fact

CDNA3 wavefronts are **64 lanes**. Consequences for Triton:

- `num_warps=4` → **256 threads/block** (not 128 as on NVIDIA). `num_warps=8` → 512 threads.
- VGPRs are shared across all waves resident on a SIMD. **512 VGPRs** per lane-slot. With
  `num_warps=8`, 2 waves land on one SIMD → each wave gets ~256 VGPRs. Exceed that → spill to
  scratch (HBM) → 3–5× slowdown. **Reducing `num_warps` is the #1 cheapest AMD perf fix.**
- Cross-lane ops inside a Triton reduction operate over 64 lanes. (Triton hides this, but it affects
  how much work one warp does and thus register pressure.)
- Block-level reductions (`tl.sum`, `tl.max` over an axis) finish with a 64-lane wave reduce; tiles
  whose reduced dimension is < 64 waste lanes.

> Porting heuristic: an NVIDIA Triton config with `num_warps=8` is often **too many** warps on AMD;
> start by trying `num_warps=4` and re-tuning up only if occupancy allows.

---

## 4. LDS (shared memory) — 64 KB and bank-conflict aware

- **64 KB LDS per CU** on CDNA3 (vs 228 KB on H100). This is *the* capacity constraint for GEMM/FA.
- Triton stages `tl.dot` operands and `make_block_ptr` tiles through LDS. The required LDS ≈
  `(BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * dtype_bytes * num_stages_buffers`. Blow past 64 KB and
  occupancy collapses to 1 block/CU (or the kernel won't pipeline).
- **32 LDS banks × 4 B** on CDNA3 (128 B/cycle). With 64-lane waves, naive layouts conflict; Triton
  inserts swizzled/padded shared layouts. You influence this indirectly via `kpack` and tile shape.
- **Inspect the ISA**: good Triton GEMM should show `ds_read_b128` / `ds_write_b128` (128-bit LDS
  ops) and `global_load_dwordx4` (128-bit global loads) inside the main loop. If you see narrow
  `ds_read_b32`/`global_load_dword`, the layout is suboptimal — retune tile/`kpack`.

`OPTIMIZE_EPILOGUE=1` removes a `convert_layout` in the epilogue (an LDS round-trip of the
accumulator). Turn it **on** in most GEMM kernels — frees LDS and removes traffic.

---

## 5. Occupancy & register pressure on MI300X

Occupancy is governed by **VGPRs, LDS, and waves/workgroup**:

```
VGPRs/EU      = 512, allocated in blocks of 16
occ_vgpr      = floor(512 / round_up_16(vgpr_used))        # waves per SIMD from VGPR
occ_lds       = floor(65536 / lds_bytes_used)              # blocks per CU from LDS
nW            = num_warps                                  # waves per workgroup
occ (wg/CU)   = min( floor(occ_vgpr * 4 / nW), occ_lds )   # 4 SIMD per CU
```

Example: vgpr_used = 170 → rounds to 176 → `512/176 = 2.9` → **2 waves/EU** (since 176×3 > 512).
Set `waves_per_eu=3` to *ask* LLVM to squeeze VGPRs below 170 to reach 3 waves/EU.

Get the numbers from a compiled kernel:
```bash
AMDGCN_ENABLE_DUMP=1 MLIR_ENABLE_DUMP=1 python my_kernel.py 2> dump.txt
grep ".vgpr_count"        dump.txt    # VGPRs
grep "triton_gpu.shared"  dump.txt    # LDS bytes
grep "num-warps"          dump.txt    # nW
# ROCm/triton ships occ.sh to compute occupancy from these.
```

---

## 6. FP8 on MI300X — use the FNUZ dialect (critical)

CDNA3 matrix cores consume the **FNUZ** FP8 variants, **not** the OCP variants used by NVIDIA H100
and AMD CDNA4 (MI350):

| dtype | bits | gfx942 (MI300X) | gfx950 (MI350) / H100 |
|---|---|---|---|
| `float8_e4m3fnuz` | E4M3, unsigned-zero, no inf | **native MFMA** | not native |
| `float8_e5m2fnuz` | E5M2, unsigned-zero | **native MFMA** | not native |
| `float8_e4m3fn` (OCP) | E4M3FN | **fails in `tl.dot`** | native |
| `float8_e5m2` (OCP) | E5M2 | partial | native |

- FNUZ vs OCP differ in **exponent bias by 1** → reading the wrong dialect is off by exactly 2×.
- Passing `torch.float8_e4m3fn` into `tl.dot` on gfx942 raises:
  `Unsupported conversion from 'f8E4M3FN' to 'f16'`. SGLang/vLLM normalize OCP checkpoints with a
  `normalize_e4m3fn_to_e4m3fnuz` helper before the matmul.
- In Triton, the AMD-native fp8 element types are `tl.float8e4b8` (E4M3 fnuz, bias 8) and
  `tl.float8e5b16` (E5M2 fnuz, bias 16).

```python
# W8A8 block-scaled fp8 GEMM tile (MI300X-correct dtypes)
a = a.to(tl.float8e4b8)          # fnuz, NOT float8e4nv
b = b.to(tl.float8e4b8)
acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
```

---

## 7. AMD-specific environment variables (debug & control)

| Variable | Effect |
|---|---|
| `AMDGCN_ENABLE_DUMP=1` | Print final AMDGCN ISA to stdout/stderr |
| `MLIR_ENABLE_DUMP=1` | Dump TTGIR / TritonAMDGPU IR after each pass |
| `TRITON_PRINT_AUTOTUNING=1` | Print the winning autotune config + timing |
| `OPTIMIZE_EPILOGUE=1` | Drop epilogue `convert_layout` (recommend ON for GEMM) |
| `TRITON_HIP_STREAM_PREFETCH=1` | Enable global-load prefetch in the stream pipeliner |
| `TRITON_HIP_USE_BLOCK_PINGPONG=1` | Ping-pong scheduling (overlap two warp groups) |
| `TRITON_ALWAYS_COMPILE=1` | Bypass the kernel cache (force recompile) |
| `AMD_LOG_LEVEL=3` | HIP runtime launch logging |
| `ROCBLAS_LAYER` / `HIPBLASLT_LOG_LEVEL` | log lib fallbacks when comparing vs Triton |

> Exact `TRITON_HIP_*` flags drift between ROCm/Triton releases — grep
> `third_party/amd/backend/compiler.py` for the current `HIPOptions` fields.

---

## 8. Worked example A — annotated FP16 GEMM for MI300X

A complete, AMD-tuned matmul. Note: `num_stages=2`, `matrix_instr_nonkdim=16`, `waves_per_eu`,
`kpack`, `GROUP_SIZE_M` for L2 reuse, and the `group_id` swizzle.

```python
import torch
import triton
import triton.language as tl

def _amd_configs():
    cfgs = []
    for BM, BN, BK in [(128, 128, 64), (128, 256, 64), (256, 128, 64), (128, 64, 64)]:
        for nw in (4, 8):                     # wave64: 4 warps = 256 threads
            for we in (2, 3):                 # waves_per_eu occupancy hint
                cfgs.append(triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK,
                     "GROUP_SIZE_M": 8,        # multiple of XCD count (8) for L2 reuse
                     "matrix_instr_nonkdim": 16,  # mfma_16x16 — preferred on MI300X
                     "kpack": 2,               # pack 2 K-slices per LDS read (b128)
                     "waves_per_eu": we},
                    num_warps=nw, num_stages=2))   # AMD: 2 for a single GEMM
    return cfgs

@triton.autotune(configs=_amd_configs(), key=["M", "N", "K"])
@triton.jit
def amd_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ---- L2-friendly block swizzle: group rows so consecutive blocks reuse B in L2 ----
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # MFMA accumulates in fp32
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)        # want global_load_dwordx4
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)                            # -> v_mfma_f32_16x16x16
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)                      # OPTIMIZE_EPILOGUE=1 drops convert
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b):
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    amd_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

**AMD-specific annotations in the kernel above**
- `matrix_instr_nonkdim=16`, `kpack=2`, `waves_per_eu` are **passed as kernel kwargs in the
  `triton.Config`** — they are AMD backend knobs, not standard Triton args (see `triton_autotune_amd.md`).
- `num_stages=2` (not 3): the AMD stream pipeliner pipelines a single GEMM best at 2.
- `GROUP_SIZE_M=8` aligns block grouping with the **8 XCDs** for L2-cache reuse.
- Avoid leading dims that are multiples of 512 B (Tagram channel hotspotting on MI300X) — if
  `K % 256 == 0`, pad `lda = ldb = K + 128`.

---

## 9. Worked example B — fused softmax (memory-bound, wave64-aware)

Row-wise softmax fused into a single kernel — the canonical reduction pattern. The reduce runs over
64-lane waves; pick `BLOCK_SIZE` ≥ row width rounded to a power of two so the wave reduce is full.

```python
@triton.autotune(
    configs=[triton.Config({}, num_warps=nw) for nw in (2, 4, 8)],
    key=["n_cols"],
)
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    in_ptrs = in_ptr + row * in_row_stride + cols
    mask = cols < n_cols
    x = tl.load(in_ptrs, mask=mask, other=-float("inf"))    # one global_load per row
    x = x - tl.max(x, axis=0)                                # 64-lane wave max reduce
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)                              # 64-lane wave sum reduce
    y = num / denom
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y
```

For LLM inference the high-value fused kernels are **Flash-Attention** (2 chained `tl.dot`s,
`num_stages=1` on AMD) and **fused RMSNorm/SiLU/residual** — same patterns: keep `num_warps` low to
avoid spills, target ≥1024 programs, and verify `global_load_dwordx4` + `ds_*_b128` in the ISA.

---

## 10. AMD vs NVIDIA pitfalls (porting checklist)

| Pitfall | Symptom | Fix |
|---|---|---|
| Hardcoded warp=32 in grid/occupancy math | wrong block sizing, half-utilized waves | use 64; recompute |
| `num_warps=8` carried over from NVIDIA | VGPR spill, 3–5× slower | try `num_warps=4`, retune |
| `num_stages=3/4` for GEMM | worse pipeline than `=2` | `num_stages=2` (single GEMM), `1` (FA) |
| OCP fp8 `e4m3fn` into `tl.dot` | `Unsupported conversion 'f8E4M3FN'` | normalize to `e4m3fnuz` |
| Big tiles ignoring 64 KB LDS | occupancy → 1, or compile fail | shrink tile / `num_stages` / use `OPTIMIZE_EPILOGUE` |
| Tuning args set as Python vars not Config kwargs | knobs silently ignored | put `matrix_instr_nonkdim`/`kpack`/`waves_per_eu` in `triton.Config({...})` |
| Leading dim multiple of 512 B | TN GEMM slow (Tagram hotspot) | pad `lda/ldb` by 128 when `K%256==0` |
| Narrow `ds_read_b32` in ISA | poor LDS layout | bump `kpack`, change tile, check swizzle |
| `mfma_32x32` everywhere | leaves perf on table | prefer `matrix_instr_nonkdim=16` |

---

## Sources

1. Optimizing Triton kernels — ROCm Documentation: <https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html>
2. AMD Instinct MI300X workload optimization (Triton + GEMM tuning) — ROCm Documentation: <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
3. Triton AMD backend source (`third_party/amd/backend/compiler.py`, HIPOptions/pass pipeline) — triton-lang/triton: <https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py>
4. Enabling vLLM V1 on AMD GPUs With Triton — PyTorch blog: <https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/>
5. Unlock Peak Performance on AMD GPUs with Triton Kernel Optimizations — ROCm Blogs: <https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html>
6. AMD FP8 (fnuz) for DeepSeek-V3 in `tl.dot` — sgl-project/sglang PR #2601: <https://github.com/sgl-project/sglang/pull/2601>
7. `triton.language.dot` API (supported dtypes, `input_precision`) — Triton docs: <https://triton-lang.org/main/python-api/generated/triton.language.dot.html>
8. A Deep Dive Into AMD Triton Compilation — Medium (Nzhangnju): <https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45>
