# Batched GEMM on AMD MI300X (CDNA3 / gfx942)

> Scope: **batched GEMM** — `B` independent GEMMs with **uniform shapes** `[M,K]·[K,N]` for every batch
> entry, fused into one launch. Contrast with grouped GEMM (`grouped_gemm.md`, *variable* M per group)
> and dense GEMM (`gemm.md`, single GEMM). AMD-only (gfx942). See `gemm.md` for the tiled-MFMA inner
> loop that batched GEMM reuses per batch slice.

The defining property: **all `B` GEMMs have identical M, N, K** — only the data pointers differ. Because
the shape is static and known at launch, batched GEMM is simpler and more predictable than grouped GEMM:
the tile grid is just `(#M_tiles · #N_tiles) × B`, no dynamic scheduler needed.

---

## 1. Where it shows up in LLM inference

| Use case | Batch dim B | Per-GEMM shape | Regime |
|---|---|---|---|
| **Multi-head attention scores** `QKᵀ` | num_heads (·batch) | `[S, d_head]·[d_head, S]` | prefill: large; decode: skinny-M |
| **Attention `·V`** | num_heads (·batch) | `[S, S]·[S, d_head]` | same |
| **Grouped/MHA QKV when done per-head** | heads | `[S, d]·[d, d]` | usually folded into one big GEMM instead |
| **Fixed-capacity / padded MoE** | E experts (capacity C) | `[C, H]·[H, N]` | when you pad every expert to capacity C → uniform M=C |
| **Batched LoRA / adapters** | num_adapters | `[M, r]·[r, N]` | low-rank, skinny K |
| **Beam/parallel-sample decode** | beams·batch | `[1, K]·[K, N]` | very skinny-M, bandwidth-bound |

Note: most modern attention is done by a **fused FlashAttention** kernel, not two raw batched GEMMs — but
batched GEMM is still the right mental model and the fallback path, and is exactly what CK's
`DeviceBatchedGemm` / `FmhaBatch*` are built on. Fixed-capacity MoE (pad to capacity C) turns the dynamic
MoE problem into a *batched* GEMM — simpler scheduling at the cost of padding waste (see trade-off below).

---

## 2. Two memory layouts: strided vs array-of-pointers

| Layout | How batch entry `b` is addressed | When |
|---|---|---|
| **Strided batched** | single A,B,C tensors; offset `b` by `stride_a, stride_b, stride_c` (e.g. `A + b*M*K`) | contiguous 3D tensors `[B,M,K]` — the common case (attention, padded MoE) |
| **Array (pointer) batched** | device arrays `A_ptrs[b], B_ptrs[b], C_ptrs[b]` | non-contiguous / gathered slices, ragged storage |

Strided is preferred on MI300X: one descriptor, perfectly coalesced loads, no pointer-chasing. Use the
array form only when the slices genuinely aren't contiguous.

---

## 3. Batched-GEMM kernel logic

The kernel = the dense-GEMM tiled-MFMA loop with a third grid axis for the batch, which just shifts the
base pointers. fp32 accumulation in VGPRs, MFMA inner loop identical to `gemm.md`.

```python
@triton.autotune(configs=[
    triton.Config({'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':64,
                   'matrix_instr_nonkdim':16,'waves_per_eu':2,'kpack':2},
                   num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M':32,'BLOCK_N':64,'BLOCK_K':128,   # skinny-M decode head
                   'matrix_instr_nonkdim':16,'waves_per_eu':4,'kpack':2},
                   num_warps=4, num_stages=2),
], key=['B','M','N','K'])
@triton.jit
def batched_gemm_kernel(A, B_, C, B, M, N, K,
                        stride_ab, stride_am, stride_ak,      # per-batch + per-element strides
                        stride_bb, stride_bk, stride_bn,
                        stride_cb, stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid   = tl.program_id(0)                       # tile within one batch entry
    pid_b = tl.program_id(1)                       # which batch entry  -> grid axis 1
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid %  num_pid_n

    a_base = A  + pid_b * stride_ab               # shift base pointers by batch stride
    b_base = B_ + pid_b * stride_bb
    c_base = C  + pid_b * stride_cb

    offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_base + offs_m[:,None]*stride_am + offs_k[None,:]*stride_ak
    b_ptrs = b_base + offs_k[:,None]*stride_bk + offs_n[None,:]*stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):        # MFMA inner loop -> v_mfma_f32_16x16x16_*
        a = tl.load(a_ptrs, mask=offs_m[:,None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None,:] < N, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(C.dtype.element_ty)                  # epilogue: cast (+bias/+act fused)
    c_ptrs = c_base + offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:,None] < M) & (offs_n[None,:] < N))

# launch grid: 2D — (tiles per entry, batch)
grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), B)
```

Why this is easier than grouped GEMM: the grid axis-1 = `B` gives perfect static load balance (every
batch entry has equal work), so there is no `tile_idx`-striding scheduler and no skew. The only judgment
call is the per-entry tile size, driven by the **per-GEMM** M,N,K regime (not the batch count).

---

## 4. Shape regimes & the small-batch trap

| Regime | per-GEMM M | What dominates | Strategy |
|---|---|---|---|
| **Prefill attention** (large S) | S (1k–32k) | MFMA compute | big tiles, 16×16 MFMA, occupancy from B·tiles ≥ 304 |
| **Decode attention** (S=1 query) | 1–few | launch + bandwidth | tiny BLOCK_M; the per-GEMM is too small → rely on B for parallelism |
| **Padded MoE (capacity C)** | C | compute, but padding waste | choose C near mean tokens/expert; grouped GEMM if skew is high |
| **Batched LoRA** | M | skinny-K | small BLOCK_K; fuse into base GEMM epilogue when possible |

**The small-per-GEMM trap.** When each batched GEMM is tiny (decode: M=1, d_head=128), a single tile
barely uses one wave and you depend entirely on `B` to fill 304 CUs. If `B` (heads·batch) is also small,
the chip is starved. Fixes: (1) collapse the batch into a single bigger GEMM when shapes allow (e.g. fold
multi-head QKV into one `[S, 3·d_model]` GEMM instead of per-head batched); (2) use a fused attention
kernel (FlashAttention / CK FMHA) that keeps everything on-chip instead of materializing batched scores;
(3) split-K within each entry if K is large.

> Padded-MoE vs grouped trade-off: padding every expert to capacity `C` makes M uniform → batched GEMM
> (static, simple, balanced) but wastes `C - M_e` rows of compute per expert. Worth it only when routing
> is near-uniform or `C` is tuned tight; with the typical skewed/decode distribution, grouped GEMM
> (`grouped_gemm.md`) wins because it processes exactly `sum M_e` rows with no padding.

---

## 5. Backend ladder (MI300X batched GEMM)

| Tier | Mechanism | Edit? |
|---|---|---|
| A — backend select | bench batched impls on `(B,M,N,K,dtype)` | no |
| B — per-backend tune | autotune tile/instance | no |
| C — code rewrite | edit Triton/CK batched kernel (fuse, split-K) | yes |
| D — quant | fp8/int8 batched | flag → accuracy gate |

| Backend | Notes (gfx942) |
|---|---|
| **hipBLASLt batched** | `hipblasLtMatmul` with batch count + strides; default; tune solution per `(B,M,N,K)` |
| **rocBLAS `*StridedBatched`** | `rocblas_gemm_strided_batched_ex`; mature; good for attention-shaped batches |
| **CK `DeviceBatchedGemm*_Xdl`** | the building block for FMHA & SmoothQuant int8 (`DeviceBatchedGemmMultiD_Xdl`); tunable instance, MFMA 16/32, fused multi-D epilogue |
| **aiter batched / fused** | tuned batched + fused epilogue (bias/act/quant); first try for fused decode |
| **Triton batched** | the §3 kernel; editable; autotune tiles + `matrix_instr_nonkdim=16`, `waves_per_eu`, `kpack`; path to fusion |
| **Fused attention (CK FMHA / FlashAttn / aiter)** | *replaces* the two batched GEMMs entirely for attention — usually the right answer, not raw batched GEMM |

**Ranked first-moves:** attention-shaped → CK FMHA / FlashAttention first (don't materialize batched
scores); generic uniform batch → hipBLASLt batched → rocBLAS strided-batched → CK → Triton. int8/fp8 →
CK `BatchedGemmMultiD` / aiter.

**CK instance knobs (Tier B):** `BlockSize`, `MPerBlock/NPerBlock/KPerBlock`, MFMA (`mfma_16x16`
preferred), pipeline v1/v2, padding on/off, multi-D epilogue ops. Build for gfx942
(`cmake-ck-dev.sh ../ gfx942`).

---

## 6. Profiling checklist

- Confirm `B × tiles_per_entry ≥ 304` — if the batched grid can't fill the chip, the batch is too small;
  collapse to a single GEMM or fuse.
- Strided layout: verify coalesced 128-bit loads per batch slice (no pointer-chase penalty from the
  array form).
- For padded MoE: measure actual vs padded rows — if padding waste > ~30%, switch to grouped GEMM.
- For attention: prefer a fused FMHA kernel; only fall to batched GEMM when no fused path covers the
  dtype/shape, and watch for the score-matrix HBM round-trip.

---

## Sources
- Composable Kernel: SmoothQuant int8 GEMM via `DeviceBatchedGemmMultiD_Xdl` (ROCm Blogs): https://rocm.blogs.amd.com/software-tools-optimization/ck-int8-gemm-sq/README.html
- Optimizing with Composable Kernel (batched GEMM / FMHA building blocks): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- Hands-On with CK-Tile: optimized GEMM on AMD GPUs (BatchGemm in ck_tile API): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- hipBLASLt documentation (batched matmul, strides, solution tuning): https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/
- rocBLAS gemm_strided_batched_ex reference: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-3.html
- AMD MI300X workload optimization (MFMA 16×16 vs 32×32, occupancy): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Triton matmul tutorial (tiling/autotune base for batched variant): https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
