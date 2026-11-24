# Gluon — shared memory / LDS layouts

> **Lookup rather than recall:** CDNA4 LDS banks (64 × 4 B), conflict rules and `ds_read_*_tr` transpose loads — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 lds`, `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 search ds_read_b64_tr`; capacities and granularity — `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 limits`.

## CDNA (gfx942/gfx950)

- **gfx950: 160 KiB LDS per CU, 64 banks; gfx942: 64 KiB, 32 banks** — a swizzle conflict-free on
  one is not known to be on the other. Per-arch keys in `perf_knowledge/hardware/data/hw_constants.json`
  (`lds_per_cu_kib`, `lds_banks`, `ds_read_b128_*_stride_bytes`).
- **gfx950 allocates LDS in 1280 B granules** (320 dwords; LLVM AMDGPUUsage `GRANULATED_LDS_SIZE`
  "GFX950: roundup(lds-size / (320 * 4))", `getLDSAllocGranule`); gfx942 in 512 B. `hw_constants.json`
  stores it as `lds_alloc_granule_bytes` (gfx950 1280, gfx942 512; also under the older `lds_align_bytes` /
  `lds_min_alloc_bytes` keys). 160 KiB is
  128 granules. GPU-measured on gfx950 / Triton 3.8.0 (resident workgroups per CU vs `shared`, 256 B steps): the
  largest `shared` that fits k workgroups is a multiple of 1280 B for every k tried, so a power-of-2 tile can lose
  a workgroup to rounding (32 KiB fits 4 per CU, not 5).
- Shared-layout choice (swizzled is the usual default; the 3.8 padded factory applies to MFMA-operand
  buffers only, and is worth an A/B against swizzled): `layout-reference.md ## Shared Layouts: The Three
  Constructors`, `layout-reference.md ## Let 3.8 Compute The Padded Layout For You (CDNA4)`,
  `layout-reference.md ### The selection rule is a starting point, not a result — the write side
  can reverse it`.
- Pipeline stages are separate, hand-sized LDS buffers; `num_stages` is inert on the Gluon path (3.6.0–3.8.0)
  (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).

## LDS is a second, independent occupancy limiter

`workgroups_per_CU = lds_capacity // lds_per_wg_allocated` (floor division — savings are step
functions: at 40 KiB/WG gfx950 fits 4; 5 needs <= 32000 B, i.e. 25 granules). Round the per-WG bytes up
to the granule first: `lds_capacity // (ceil(shared / 1280) * 1280)` on gfx950 (512 on gfx942). Registers and LDS cap occupancy
independently; a kernel can be capped by both.

**`lds_per_wg_allocated` comes from the Triton cache metadata `shared` field; a value derived from the
source's allocations misses conversion scratch.**
`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py --meta <ir-dir>` (pack shim:
`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`). **The kernel descriptor's `group_segment_fixed_size` is 0 for every Triton
kernel** (LDS is passed as dynamic at launch; compile-verified 3.8.0, gfx950 and gfx942) — a 0 there is not a
small allocation (**Overrides KB** for JIT-compiled Triton kernels, 3.8.0: `perf_knowledge/languages/triton_amd/isa_verify.md`
reads LDS bytes from `.group_segment_fixed_size` and `perf_knowledge/languages/triton_amd/deep_codegen.md` says the
allocator sets it; verified by a compile-only check: a kernel with `metadata.shared` 1024 B emits
`.amdhsa_group_segment_fixed_size 0` on gfx950 and gfx942; an AOT binary with static LDS is outside this scope).
Measured occupancy follows `shared` rounded up to the granule (GPU-measured, gfx950 / 3.8.0, descriptor 0 on every
kernel swept), so the dynamic allocation is what the dispatch enforces. Profiler LDS columns depend on whether
they read the descriptor or the dispatch packet: rocprofv3 1.3.5 (`--kernel-trace` CSV `LDS_Block_Size`) reported
0 for every dynamic-LDS kernel, Triton/Gluon included (gfx950), while rocprof-compute 3.11.0's LDS-per-workgroup
metric (block 7.1.8) reported the dynamic allocation (equal to `shared`); cross-check any profiler value against
`shared`. `shared` is
the peak of simultaneously live buffers — `max(explicit, scratch)` only when the allocation is dead at the
convert, their sum when it is live (`layout-reference.md ### The scale operand is
register-resident, so staging it through LDS is a choice`). `shared` is the un-rounded request (the
hardware rounds up to the granule at dispatch), so reading `shared` alone does not reveal the granule; a measured
occupancy sweep over `shared` does.

- **Padding costs capacity; swizzle does not.** Compute the new workgroups/CU before adopting a
  padded layout; on an async-filled buffer swizzle is also the destination that lowers
  unconditionally (`memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`).
- **Carving vs allocating twice**: `.index` / `.slice` / `.permute` / `._reinterpret` views over one
  allocation cost no extra LDS (`../tile-programming/slicing.md ## LDS dedup (carve reuse from one allocation)`).
- If the SPI occupancy limiter is empty, neither resource caps you (`../method/profile.md`).

## Reshaping an allocation without reallocating it

`.index(i)` (subview on dim 0), `.slice(start, length, dim)`, `.permute(order)` and
`._reinterpret(dtype, shape, layout)` return a new descriptor over the same bytes. `_reinterpret` is
how you stage as one dtype and read as another, or give the reader a different layout than the
writer (the gfx942 `order=[1, 0]` async-destination case in
`pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`).

**`_reinterpret` signature changed in 3.8.0:** before 3.8.0 `dtype`, `shape`, `layout` are all
required; from 3.8.0 each defaults to the descriptor's current value, so `d._reinterpret(dtype=gl.int8)`
is 3.8.0-only (`TypeError` on 3.7.1). Passing all three works everywhere. Underscore = not stable.

## _keep_alive() — the name is the effect, the mechanism is its opposite

**`shared_memory_descriptor._keep_alive()` emits `ttg.local_dealloc`: it marks the END of a live
range** (checked v3.8.0; body is `_semantic.shared_dealloc(self)`). Without an explicit dealloc the
compiler ends the range at the first point post-dominating all uses, and the allocator may reuse the
bytes from there. Placing `_keep_alive()` after the last real use **extends** the range to that point.

- It can remove an allocator-reuse hazard instead of fencing it (a third escape hatch next to
  `../tile-programming/warp-pipeline.md ## Membar is a standing risk, not a one-time bug`); gate it
  with the determinism race-test, since you now own the guarantee.
- A longer range can raise peak LDS (cost a workgroup/CU): read `shared` before and after.

## Indexed access to LDS — .gather / .scatter

**3.7.0+** (absent on 3.6.0).

- `smem.gather(indices, axis)`: `result[I] = src[I[0], ..., indices[I], ..., I[n]]`; result has the
  **shape of `indices`**.
- `smem.scatter(values, indices, axis)`: `dst[..., indices[I], ...] = values[I]`; `values` and
  `indices` broadcast to a common shape first.
- `smem.gather` always costs `ds_read`; `gl.gather` on a register tensor may cost nothing
  (`layout-reference.md ## gl.gather — a lane exchange only when the layouts allow it`).
- No documented cross-wave ordering or collision semantics: barrier between scatter writes and any
  read, and treat colliding indices as unspecified (not last-writer-wins).

## Atomic RMW in LDS — the atomic_scatter_* family

**3.8.0 only** (version row: `../workloads/moe.md ## Version gates`). **There is no pointer-form
`smem.atomic_add`** (3.8.0, `hasattr`-checked) — the only spelling is `atomic_scatter_<op>`, so grepping for `atomic_add` on a
descriptor finds nothing whether or not it is used.

```python
prev = smem.atomic_scatter_add(values, indices, axis, mask=active)   # returns PRE-op value per lane
```

- Family: `atomic_scatter_add`, `_max`, `_min`, `_and`, `_or`, `_xor`, `_xchg`.
- **The parameter is `axis`, not `dim`** — `dim=` is a `TypeError`. `values`/`indices` broadcast as
  for `.scatter`; masked-off lanes do no RMW.
- **`add` and `xchg` accept int and float; `max`, `min`, `and`, `or`, `xor` are integer-only** and
  raise at compile time otherwise (float max -> order-preserving integer key). `max`/`min` are
  signedness-aware.
- Use it when a lane's destination is data-dependent (histogram, CTA-local counting sort, slot
  reservation, conflict-tolerant `_xchg` publish). When the destination follows position, barrier +
  `gl.sum`/`gl.max` is the deterministic alternative available on every version.
- **Ordering among colliding lanes is nondeterministic across waves** (GPU-checked gfx950 / 3.8.0: within
  one wave colliding lanes serialized in a fixed lane order; at 4 warps the wave order varied run to run, so
  `prev` slots differ) — harmless
  only if a later ordered step folds by a stable key (`../workloads/moe.md ## Bucketing: three
  shapes, and what selects one`).
- The atomic is not a rendezvous: seed the table (`allocate_shared_memory` accepts an initial
  value), `gl.barrier()`, atomics, `gl.barrier()`, then read back with `.gather`.

```python
COUNT_SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])      # 1-D, unswizzled
counts = gl.allocate_shared_memory(gl.int32, [NBUCKET], COUNT_SH,
                                   gl.zeros([NBUCKET], gl.int32, layout=LIN))
gl.barrier()
slot = counts.atomic_scatter_add(gl.full(bucket.shape, 1, gl.int32, bucket.type.layout), bucket, 0, mask=active)
gl.barrier()
total = counts.gather(bucket, 0)
```

Below 3.8.0 the methods do not exist (trace-time attribute error; `hasattr` probe). Downgrades:
`gl.atomic_add` on a global scratch buffer (all versions; wider nondeterminism), or
`gl.associative_scan` + `.gather` (deterministic, extra pass).

## vs CuTeDSL smem/TMEM

| CuTeDSL | Gluon (CDNA) |
| --- | --- |
| TMEM accumulator (sm_100) | none — accumulators are VGPRs or AGPRs as the allocator places them (`../tile-programming/llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`) |
| TMA swizzle modes | LDS swizzle + buffer resource |
| `make_swizzle` XOR | `SwizzledSharedLayout` (padding = `PaddedSharedLayout`) |

gfx1250 (`PartitionedSharedLayout`, 320 KiB LDS) is out of scope.
