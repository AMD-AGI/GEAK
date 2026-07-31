# Gluon — pipeline (manual buffer sync, no Pipeline* library)

Backbone layer 4. Gluon has **no** CuTeDSL `PipelineTmaAsync` / mbarrier library and
**no** TileLang `num_stages` auto-pipeliner. Overlap is **authored**:

```text
buffer_load_to_shared / cdna4.async_copy / gfx1250 TDM
  → commit_group (producer)
  → wait_group   (consumer)
  → mfma / wmma
```

LLVM passes (`LLIR_SCHED`, `RA_HINTS`, `AMDGCN_AS`) interleave the hot loop — toggle +
IR-verify (`../tile-programming/compiler-contract.md`).

## vs CuTeDSL (concept filter)

| CuTeDSL | Gluon equivalent |
| --- | --- |
| `PipelineTmaAsync` + mbarrier tx_count | `commit_group`/`wait_group` + `s_waitcnt` |
| TMA multicast | gfx1250 `cluster` + multi-CTA (CDNA: XCD remap, not multicast) — note TMA is NVIDIA-proprietary; gfx1250 uses `cluster`/TDM, not NVIDIA TMA |
| `cutlass.range(prefetch_stages=)` | manual prologue/steady-state in source |
| warp-spec warpgroups | `warp_specialize` + `warp_pipeline_stage` (gfx950) |
| CLC | **none** on CDNA; gfx1250 cluster scheduling only |

## gfx950 CDNA4 path

```python
# producer
gl.amd.cdna4.async_copy(gmem_ptr, lds_ptr, ...)
gl.amd.cdna4.commit_group()
# consumer
gl.amd.cdna4.wait_group(n)
# mfma
acc = gl.amd.cdna4.mfma(...)
```

Plain Triton on gfx950 may get `tritonamdgpu-pipeline` — **Gluon does not**; hand-write
sync when escalating from plain.

## gfx1250 path

TDM descriptor async copy replaces buffer_load_lds for matrix operands
(`gl.amd.gfx1250.tdm`). Wave **32** — separate minimal anchor from gfx950.

## gfx942 downgrade

Async hardware exists; verify `commit_group` lowering. No scaled MFMA.

## Roll the loop (cut i-cache pressure)

Gluon has **no** `num_stages`/`loop_unroll_factor` knob (only `ttgl.static_range`, which
fully unrolls, and plain `range`, which does not). When the Instr Cache hit rate is low,
this is a structural edit, not an API call: use plain `range` for large trip counts and
factor a fat hot body into a shared `@gluon.jit` helper to avoid duplicated code.

```python
for k in range(0, K, BLOCK_K):        # plain range -> rolled loop (static_range would unroll all)
    acc = _dot_step(a, b, acc)        # fat body factored into one @gluon.jit subroutine
```

There is no arch-specific primitive here (unlike `tl.range(loop_unroll_factor=)` in
plain Triton). Verify: L1I hit up, fetch-latency bubble down.

## Footguns

- `wait_group` depth mismatch → hang or stale LDS read (`../debug-async.md`).
- Deeper stages without occupancy headroom → regression (`../tile-programming/slicing.md`).

## Anchors

- [triton-lang/triton `cdna4/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd/cdna4)
- [triton-lang/triton `gfx1250/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd/gfx1250)
