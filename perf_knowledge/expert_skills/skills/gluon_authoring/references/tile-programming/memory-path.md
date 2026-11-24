# Memory Path Layer (gfx950)

HBM -> MFMA operands with the fewest, widest, best-hidden transactions (after the transcription anchor,
before LDS layout / pipeline).

## The memory-path ladder

1. **Generic** (anchor default): `gl.load` -> VGPR; masked-load overhead + register staging to LDS.
2. **Buffer ops**: `gl.amd.cdna4.buffer_load` / `buffer_store` (gfx942: `gl.amd.cdna3.*`) with a scalar
   base + **element offsets, not bytes** (the emitter multiplies by the element width; pre-multiplying
   scales the address twice). Hardware OOB removes mask branches (plain masked `global_load` ~140
   branches, buffer ops ~4).
3. **Direct-to-LDS async** (gfx950 default): `gl.amd.cdna4.async_copy.buffer_load_to_shared`
   moves HBM -> LDS directly (no register staging, no `ds_write`); `commit_group` / `wait_group`.
   **Per-lane width (Triton 3.8.0, compile-checked): gfx950 exactly 128 or 32 bit; 64 bit fails.**
   **gfx942:** 32 bit only, destination `order` = the source's fast dim (`[1, 0]` row-major) (padded destination: interval >= 256 B and an
   `offset_bases`-derived producer; swizzled is the simpler destination to bring up); measured **slower than
   synchronous staging** in the case recorded (`../gluon/pipeline/authored-overlap.md` row A1), so rung 2 +
   sync staging (GR -> LW -> LR) is the usual gfx942 form — A/B the async form before closing it.
4. **Side path** (GR -> LW -> LR) for a tensor whose per-lane width misses the legal set or whose
   per-lane run is not contiguous in the source (the direct path does not lower for those). A 4 B/thread scale is the legal 32-bit form and goes
   direct (`low-precision.md`).

```python
values = gl.amd.cdna4.buffer_load(base, elem_offsets)          # element offsets
# destination is the FIRST argument; returns nothing
gl.amd.cdna4.async_copy.buffer_load_to_shared(shared_buffer, base, elem_offsets)
```

### Async copy: smoke-test before wiring

**A minimal async-copy smoke compile on the exact shared shape / dtype / offset layout / consumer load
isolates lowering failures before `buffer_load_to_shared` is wired into the loop.** Its lowering failures
have mostly been layout-contract violations rather than missing backend support (checklist: `compiler-contract.md ## Async / buffer path failures are layout-first`).

- **Per-lane width is a set, not a floor** (3.8.0: gfx950 {128, 32} bit, gfx942 {32}; 64 bit illegal on both).
  Failure strings: `failed to translate module to LLVM IR` (`LLVM Translation failed for operation:
  builtin.unrealized_conversion_cast`) / `PassManager::run failed` (`failed to legalize operation
  'ttg.async_copy_global_to_local'`). Fix `size_per_thread` on the contiguous dim, or use the side path.
  The same error fires for a legal width whose per-lane run is not contiguous in the source. Table:
  `../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`.
- **Pre-coalesced offsets: Gluon runs no `CoalesceAsyncCopy` pass on 3.8.0** (plain does). The offset layout
  must already pass `canLoadDirectToLDS` (contiguous threads on the LDS-fast dim, legal-width run — 128 bit =
  `vec=8` fp16/bf16 — and `vec == contig`); otherwise lowering fails with `unrealized_conversion_cast`
  (compile-checked) rather than being rewritten. Testable predicate: exact cover,
  `threads_per_warp[contig] * size_per_thread[contig] == tile extent along contig`
  (`../gluon/memory-reference.md ## Async Copy To Shared`).
- **Shared-layout family:** `SwizzledSharedLayout` lowers; a padded destination lowers only when the
  padding interval covers one wave's copy (64 lanes x per-lane bytes: 1 KiB at 128 bit) and the producer
  layout is derived from its `.offset_bases` (`layout-recipes.md` `producer_ll_for`;
  `../gluon/memory-reference.md ## Deriving the producer layout for a padded async destination`; for
  `with_identity_for` an exact-cover row-major `BlockedLayout` is that producer); a row-major
  `BlockedLayout` producer into a `compute_efficient_padded_shared_layout` destination hits
  `unrealized_conversion_cast`. Swizzled is the simpler destination to bring up.
- Accept on asm: `buffer_load ... lds` present, **no** `ds_write` on the load path.

```python
gl.amd.cdna4.async_copy.buffer_load_to_shared(shared_buffer, base, elem_offsets)
gl.amd.cdna4.async_copy.commit_group()
gl.amd.cdna4.async_copy.wait_group(0)
# Read shared_buffer only after the wait; verify no ds_write in asm.
# gfx942 downgrade: same calls, size_per_thread at 32 bit, shared layout order = source fast dim ([1, 0] row-major).
```

## Decision signals

- Masked `global_load` with many `s_cbranch` in `.amdgcn`, or high VGPR from register staging -> buffer ops.
- `ds_write` on the HBM -> reg -> LDS path -> `buffer_load_to_shared`.
- Memory-bound: in-flight bytes per CU vs the TCP cap (~32 KiB/CU, unverified constant;
  `../hardware/roofline-models.md ## HBM / TCP in-flight model`); per that model, once capped more occupancy
  adds no bandwidth, and the request pattern or tile is what moves it.

## Epilogue stores: MFMA layout vs an LDS-staged row-major store

**Storing straight from the MFMA layout gives `buffer_store_dwordx2` (8 B/lane)** for a gfx950 16x16
(`transposed=True`) fp16 tile (compile-verified, Triton 3.8.0). When the output write is a visible share
(prefill GEMM, wide-N epilogue), staging through LDS (aliasing the no-longer-used operand ring) and reading
back in a row-major `BlockedLayout` with 16 B/lane gives `buffer_store_dwordx4`. GPU-measured on gfx950 / Triton
3.8.0 (fp8 GEMM): the staged 16 B/lane store paid where the epilogue is a large share of the time (short K, many
tiles) and tied at long K. Store cache modifiers `.wb`, `.cg`, `.cs`, `.wt`: in the plain-Triton GEMM epilogues
measured (gfx950 / 3.8.0) `.wt` and `.cs` lost on output-heavy and square shapes and tied at decode, while `.cg`
/ `.wb` set no bits and equal the default — keep the default store and A/B a modifier only with a measured
reason (a `.wt` win is on record for an in-place state RMW, `../workloads/linear-attention.md ### Cache
modifiers on the state slab (.cg load, .wt store)`, not for a GEMM epilogue). Costs: the LDS round
trip + barrier, and register pressure (fence loads hoisted above the drain or its MFMAs can spill).
Accept on ISA (`dwordx4`, no new `scratch_*`) and a timing A/B.

## s_waitcnt vmcnt counts stores too; wait_group(0) is not a store fence

- **On CDNA3/4 `vmcnt` counts global stores as well as loads, and they retire in order**: a `vmcnt`
  drain for a store also waits behind (and blocks the waits of) later-issued loads, so a loop that waits
  for its own stores pays for any global loads issued after them. Verified on Triton 3.8.0 / gfx950
  (multi-token decode, and a store-heavy microbenchmark whose loop got an `s_waitcnt vmcnt(0)` per iteration
  once a global load sat in it); the cost was largest at low occupancy, where no other wave hides it.
- **`wait_group(0)` only tracks async direct-to-LDS copies** (lowers to `llvm.amdgcn.wait.asyncmark`):
  with no async copy outstanding it emits **no** `s_waitcnt vmcnt` for ordinary `gl.load` / `gl.store`,
  but still costs an `s_barrier` per call. It does not order or fence stores. Compile-verified, 3.8.0,
  gfx950 + gfx942 (`pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`).
  GPU-checked on gfx950 / 3.8.0: between waves of one CTA the missing drain did not show (same-CU readers saw
  the stores); it shows as stale payload for a consumer on another XCD, where even a per-wave `vmcnt(0)` +
  barrier + `.wt` flag was not enough on cacheable memory — publish with a release atomic (writeback),
  `../workloads/collective.md`.

## Output / reduction path (atomics)

- **In-loop atomics and `sem`.** The default semantics emit a per-op fence (`buffer_wbl2` / `buffer_inv` plus
  `s_waitcnt vmcnt(0)`) around each atomic, serializing the loop; `sem="relaxed"` (no-return
  `buffer_atomic_add_f32`) removes it (compile-checked 3.8.0 gfx950) — correct where nothing reads the
  result inside the kernel without another sync. The asm shows whether the trailing `vmcnt(0)` is gone. Residual cost is L2 atomic
  throughput under contention.
- A cross-partition `buffer_atomic_add` reduction writes N x the output for N contributors: a bandwidth
  cost set by the parallelization (`mental-model.md ## Reduction & parallelization structure (decide
  before tiles)`); fewer contributors (coarser partition, few split-K splits) reduce it, scheduling does not.
- **Packed atomics (`*_atomic_pk_add_{f16,bf16}`, two elements per instruction) pay only where the atomic
  epilogue is atomic-throughput bound and the lowering is clean.** GPU-measured on gfx950 / Triton 3.8.0, two
  split-style atomic epilogues: with `sem="relaxed"`, a lane-coalesced layout and a loop of pure
  `*_atomic_pk_add_*`, bf16 / fp16 beat fp32 atomics at large-M (mid and prefill-scale) shapes and tied at
  decode-scale shapes; with the default `sem` (acq_rel), where the bf16 lowering also carried a
  `global_atomic_cmpswap` path and a `buffer_wbl2` / `buffer_inv` pair per atomic (the fp32 form got one
  fence for the whole kernel, compile-checked), the packed form bought little. So, before timing: `sem="relaxed"`
  where nothing reads the result in-kernel (bullet above), then confirm in the asm that the epilogue is
  `*_atomic_pk_add_*` with no `cmpswap` and no per-atomic fence. The layout is not a reliable predictor:
  `gl.atomic_add` at `vec=1` still emitted packed atomics plus a `cmpswap` path, and `vec=2` was pure packed in
  one kernel and mixed in the other; `cdna4.buffer_atomic_add` at `vec=1` aborts the compiler
  (`Assertion vec >= 2`; compile-checked 3.8.0 gfx950). **Layout outweighed dtype:** a wide per-thread vector
  (consecutive lanes far apart) made each atomic instruction scatter and cost far more than the dtype choice —
  keep consecutive lanes on consecutive addresses. bf16 and fp16 timed the same; pick bf16 for its fp32
  exponent range, not for speed.

## Reuse a shared operand (load once, reuse)

Operand read more than once per work-item (attention K/V across query rows): one `buffer_load_to_shared`
followed by a `.load()` (or `.permute([1, 0]).load()`) per consumer tile reads HBM once. gfx942: sync staging
(`gl.load` -> `smem.store` -> `smem.load`; no `ds_read_tr` behind `permute`).

## Cut HBM bytes (narrow / cache / fuse)

When HBM is saturated only fewer bytes help (time = bytes / bandwidth): narrowing the dtype at the load
(`low-precision.md`), biasing L2 with the cache modifier, or fusing streaming passes.

```python
a = gl.amd.cdna4.buffer_load(a_ptr, offs, cache=".cg")  # leading dot: "cg" fails to compile (3.8.0)
```

No L2 prefetch op in the 3.8.0 Gluon `cdna3` / `cdna4` modules (`tdm.prefetch` is gfx1250-only). Verify: achieved HBM bytes down.

## IR / asm acceptance signals

| Change | What to confirm in IR/asm |
| --- | --- |
| buffer ops | `buffer_load_dwordx4` + `v_cndmask`; branch count drops vs `global_load` + `s_cbranch` clusters |
| async copy | `buffer_load ... lds` present and **no** `ds_write` on the load path; `wait_group` retires the buffer before reuse (gfx942 downgrade: `buffer_load_dword ... lds` only — a wider form means the 32-bit gate was not the one applied) |
| coalescing | per-lane HBM offsets form wide, aligned (128-bit) transactions; contiguous dim aligned with `size_per_thread` on the fast axis |

Dump with `kernel_workflow/scripts/kernel_tools/dump_ir.sh` (`compiler-contract.md ## IR dump workflow`).

## A load's width is issue-side; a transaction's width is access-side

Adjacent lanes on adjacent addresses coalesce into wide transactions whatever the per-lane opcode width,
so counting narrow-load opcodes counts issue slots, not bytes or transactions; removing them can measure
nothing. Read instead: the address delta between adjacent lanes on the contiguous dim (1 element = already
one wide run); the contiguous run per instruction = lanes on the contiguous dim *within one group* x
per-lane width (exact cover can hold while the run shrinks, which costs transaction width); and the
profiler's coalescing / partial-line rates (`../hardware/bound-class-signals.md`). If bytes, in-flight and
occupancy are unchanged, the arm only tests the issue rate.

## Common failures

- Buffer ops with ambiguous fallback value, offset unit or pointer shape -> silent wrong results.
- `ds_write` contending with `buffer_load_to_shared` on the LDS write port (hundreds of cycles, paid at the
  `lgkmcnt` wait that consumes the write while DMA data is landing, not at the `ds_write` issue; ATT,
  gfx950): `low-precision.md ## a4w4 (MXFP4, e2m1)`.
- An "8-entry LDS request FIFO per SIMD pair" is not visible in ATT on gfx950 (rocprofiler-sdk 1.3.5): one
  wave queued about 16 `ds_read_b128` before issue stalled, and outstanding LDS-DMA did not block `ds_read`
  issue — LDS-DMA costs show when its data lands
  (`../hardware/isa-mechanisms.md ## LDS instruction behavior (gfx950)`).
