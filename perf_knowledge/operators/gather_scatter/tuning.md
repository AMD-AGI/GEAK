---
title: gather_scatter — tuning (coalescing, 128-bit rows, atomics, LDS-staged gather)
kind: technique
operator: gather_scatter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
  - https://llvm.org/docs/AMDGPUUsage.html
---

# gather_scatter — tuning

Pure bandwidth, but with **irregular addresses**. The whole game is to keep the *bytes* coalesced and
vectorized even though the *row index* is random.

## 1. Tile the hidden dim (the #1 lever)
Assign one workgroup (or warp) per gathered **row**, and tile the hidden dim `H` into 128-bit chunks so the
64 lanes of a wave load **contiguous** bytes of that row:
```python
# Triton: one program per output row, BLOCK_D over hidden
row = idx[tl.program_id(0)]                 # scattered row pointer (1 indirection)
for d0 in range(0, H, BLOCK_D):
    cols = d0 + tl.arange(0, BLOCK_D)
    v = tl.load(in_ptr + row*H + cols, mask=cols < H)   # COALESCED within the row
    tl.store(out_ptr + pid*H + cols, v, mask=cols < H)
```
Now only the **row base** is irregular (one scattered address per program); the `H` payload is a clean
`global_load_dwordx4` stream. This is exactly the "BLOCK_D tiling over hidden for coalesced access" that the
Triton MoE permute uses to hit ~54% peak BW on the scatter.

## 2. Vectorize the row (128-bit)
`float4`/`int4`, `__restrict__`, 16-B-aligned `H` → `global_load_dwordx4`/`global_store_dwordx4`. A 4-B
per-lane gather wastes ~half the bus. If `H` isn't 8-aligned (fp16) pad or fall back to dwordx2 on the tail.

## 3. Scatter-reduce: atomics
When multiple inputs map to one output (MoE unpermute: `topk` rows → 1 token), the write **must** reduce:
- HIP: `atomicAdd` with `-munsafe-fp-atomics` → `global_atomic_add_f32` (HW path, no CAS loop). bf16/fp16
  atomics are supported on CDNA3/4; check the ISA emits the HW atomic, not a software CAS.
- Triton: `tl.atomic_add(out_ptr + ..., v, mask)`.
- ⚠ Triton **can't do 2-D scalar indexing into an accumulator** (`acc[m,:]` with loop `m`), which blocks a
  fully-fused down-proj+scatter — the down GEMM stays a separate grouped-GEMM kernel and the scatter is its
  own kernel (or epilogue). Known Triton limit.
- Prefer **gather over scatter** when you have the choice: a gather (read-scattered, write-contiguous) has no
  atomic contention; reformulate unpermute as a gather where possible.

## 4. Reduce coalescing pressure on the index side
- **Sort/group indices first** (MoE align-sort) so consecutive programs touch nearby rows → better L2 reuse.
  AMD measured only **~30% coalescing / 61% L2 hit** on the MoE sort path — still a live bottleneck, so the
  sort quality matters.
- For **embedding-bag**, accumulate the bag in **registers/LDS** and do one vectorized store per output row
  rather than `atomic_add` per index.

## 5. LDS-staged / direct-to-LDS gather (HIP)
`__builtin_amdgcn_global_load_lds` can move a **scattered** per-lane global address straight into a
**coalesced** LDS destination — i.e. gather into LDS with no VGPR staging — provided the LDS side is
coalesced (and swizzled, see [[operators/transpose/tuning.md]] §3). gfx942: `global_load_lds`; gfx950:
unified `llvm.amdgcn.load.to.lds`. Useful when many lanes gather rows reused within the workgroup.

## 6. Grid / occupancy
≥1024 workgroups; block = multiple of 64. For one-row-per-program, that's automatic when `N` is large; for
small `N` (decode), grid-stride over rows or split `H` across programs to fill 304 CUs.

## Verify
rocprof-compute → memory-chart: high **uncoalesced/L2-miss** = the index side dominates (sort harder / stage
in LDS); want the row payload near HBM roofline. ISA: `global_*_dwordx4`, HW `global_atomic_add_*` (no CAS
loop). Oracle: gather exact; scatter-reduce `allclose` (atomics reorder).

## Sources
- BLOCK_D tiling, 54% BW, 30% coalescing / 61% L2, Triton 2-D scalar-index limit: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang · https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- Vectorize / LDS-stage guidance: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- HW fp atomics, global_load_lds gather: https://llvm.org/docs/AMDGPUUsage.html · [[languages/hip_cpp/lds_async.md]].
