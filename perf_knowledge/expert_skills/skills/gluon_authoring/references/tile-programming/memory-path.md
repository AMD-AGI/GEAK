# Memory Path Layer (gfx950)

Read this for the memory-path layer of the backbone (after the transcription
anchor, before LDS layout / pipeline). Goal: get data from HBM to the MFMA
operands with the fewest, widest, best-hidden transactions.

## The memory-path ladder

1. **Generic** (anchor default): HBM -> `gl.load` -> VGPR. Correct but pays full
   masked-load overhead and a register staging round-trip to LDS.
2. **Buffer ops**: `gl.amd.cdna4.buffer_load` / `buffer_store` (gfx942 spelling:
   `gl.amd.cdna3.*`, same op) with a
   scalar base + **element** offsets (the emitter multiplies by the element
   width itself; pre-multiplying scales the address twice). Hardware OOB
   handling removes mask branches
   (plain masked `global_load` can emit ~140 branches; buffer ops ~4).
3. **Direct-to-LDS async** (gfx950 default): `gl.amd.cdna4.async_copy.buffer_load_to_shared`
   moves HBM -> LDS directly, removing the register staging and the `ds_write`
   on the load path. Pair with `commit_group` / `wait_group`. Legal per-thread widths on
   gfx950 are **128 bit and 32 bit**.
   **gfx942 downgrade:** the same op lowers only at **32 bit per thread**, into a shared
   destination with `order=[1, 0]` (a padded destination fails to translate — build swizzled
   first), and it measured **slower than synchronous staging** at that width
   (`../gluon/pipeline/authored-overlap.md` row A1). On gfx942 stop at rung 2 plus sync staging
   (GR -> LW -> LR) unless an A/B says otherwise.
4. **Side path** (small metadata / scales): GR -> LW -> LR for a tensor whose per-thread width
   misses the legal set (gfx950 {128, 32} bit, gfx942 {32}), or whose per-thread run is not
   contiguous in the source. It is not "tensors below ~16 B/thread": a 4 B/thread scale is the
   legal 32-bit form and goes direct (`low-precision.md`;
   `../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`).

Copy the wider-load form, then specialize the scalar base and element offsets:

```python
# Start from the generic load, then specialize the wider buffer path (gfx950).
values = gl.load(ptr)
values = gl.amd.cdna4.buffer_load(base, elem_offsets)
# If the layout contract passes, specialize to direct-to-LDS.
# Note the destination is the FIRST argument, not the last.
gl.amd.cdna4.async_copy.buffer_load_to_shared(shared_buffer, base, elem_offsets)
```

`elem_offsets` counts elements, not bytes, on all three of these — the buffer
emitter converts to bytes itself.

### Async copy: smoke-test before wiring (mandatory)

`buffer_load_to_shared` (step 3) fails in **lowering for layout-contract reasons
before backend reasons**. Before wiring it into the loop, run a minimal async-copy
smoke on the **exact** shared-buffer shape / dtype / consumer load + offset
layout, in isolation. This catches the layout mismatch in one shot instead of
through repeated in-loop debug rounds. Full layout-first checklist:
`compiler-contract.md ## Async / buffer path failures are layout-first`.

The most common async-copy lowering failure is the **per-thread width**: the per-thread load
must land **on** a legal width (gfx950 {128, 32} bit; gfx942 {32}) — it is a set, not a floor, and
8 B (64 bit) is illegal on both. It is rejected at lowering (not a build ceiling, not a
`BlockedLayout`-family issue): move `size_per_thread` on the contiguous dim onto a legal width, or
use the side path. The same error also fires when the width is legal but the per-thread run is not
contiguous in the source (a permuted element order), so check the offsets too. The width table, the
per-dtype element counts and the scattered-run case are stated once in
`../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`.

**Direct-to-LDS needs a *pre-coalesced* offset layout — the Gluon lowering does
not coalesce for you.** `buffer_load_to_shared` lowers to a direct global->LDS
copy only if the offset layout already passes `canLoadDirectToLDS`: contiguous
threads on the LDS-fast dim, a legal-width contiguous run (128-bit = `vec=8` bf16/fp16 on gfx950),
and `vec == contig`. The **plain** path gets this automatically from the
`CoalesceAsyncCopy` compiler pass; the **Gluon** lowering runs **no** such pass, so
a hand-written offset that is *not* already coalesced silently falls back (register
staging + `ds_write`, or an `unrealized_conversion_cast` lowering failure). Build
the offset layout coalesced on the fast dim from the start (natural tile:
threads-contiguous along the row; transposed tile: along the column, paired with the
matching swizzle order) — the testable predicate is exact cover,
`threads_per_warp[contig] * size_per_thread[contig] == tile extent along contig`
(`../gluon/memory-reference.md ## Async Copy To Shared`) — and confirm in asm that
`buffer_load ... lds` appears with **no** `ds_write` on the load path. The async path also
constrains the **shared-layout family**: `SwizzledSharedLayout` lowers, and a **padded destination
has failed to translate on both constructors** (`with_identity_for` and
`compute_efficient_padded_shared_layout`), so build swizzled first. It supports
**transpose-on-read** (`smem.permute`) for reusing one tile in two operand orientations:
`../gluon/memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`.

Copy this smoke-test skeleton, then specialize the shared buffer shape, dtype,
offset layout, and consumer before wiring it into the loop:

```python
# gfx950. Copy, then specialize the exact buffer/layout contract.
# The destination is the FIRST argument, and the call returns nothing.
gl.amd.cdna4.async_copy.buffer_load_to_shared(shared_buffer, base, elem_offsets)
gl.amd.cdna4.async_copy.commit_group()
gl.amd.cdna4.async_copy.wait_group(0)
# Read shared_buffer only after the wait; verify no ds_write in asm.
# gfx942 downgrade: same calls, size_per_thread at 32 bit, shared layout order=[1, 0].
```

## Decision signals

- Masked `global_load` with many branches in `.amdgcn`, or high VGPR from
  register staging -> move to buffer ops.
- `ds_write` on the HBM->reg->LDS path showing in the IR -> move to
  `buffer_load_to_shared` (async), which removes the `ds_write`.
- Memory-bound by roofline (`intensity << ridge`) -> prioritize this layer; aim
  to saturate effective in-flight bytes vs the 32 KiB/CU TCP cap
  (`../hardware/roofline-models.md ## HBM / TCP in-flight`).

## Budget to compute first

```text
data_per_request_per_wave = block_size / num_waves_per_workgroup
inflight_bytes_per_CU     = num_req_per_wave * data_per_request * active_waves
effective_inflight        = min(inflight_bytes_per_CU, 32 KiB)
```

If `effective_inflight` is already capped at 32 KiB, more occupancy will not add
bandwidth; restructure the request pattern or tile instead.

## Output / reduction path (atomics)

The load path is only half the traffic. A cross-partition reduction written with
`buffer_atomic_add` (e.g. a gradient accumulated across program instances) carries
**write amplification proportional to the number of contributing partitions**: N
partitions each atomic-add the output once -> N x the output write volume.

- This is **bandwidth, not latency**: scheduling / `sem="relaxed"` / overlap hide
  the atomic *latency*, never the write *volume*; on a memory-heavy loop the atomic
  traffic adds on top and does not overlap away.
- **Use relaxed/no-return atomic semantics — the default serializes the loop.** An
  in-loop atomic compiled with the **default** (acquire-release-ish) semantics emits
  a per-op **completion fence** (`s_waitcnt vmcnt(0)`) that drains every atomic
  before continuing — turning fire-and-forget accumulation into a serial chain.
  Passing `sem="relaxed"` (CDNA4 `buffer_atomic_add_f32` no-return / fire-and-forget)
  removes that per-op fence and is a large, separate codegen win on top of any volume
  reduction. Confirm in asm the atomic no longer has a trailing `vmcnt(0)` drain.
  (After this, the residual cost is the L2 atomic throughput under contention —
  hardware, not codegen.)
- It is a **structural** cost set by the parallelization choice
  (`mental-model.md ## Reduction & parallelization structure (decide before tiles)`),
  not a kernel bug.
  Reduce it by reducing #contributors (coarser partition / split-K with few splits
  / a different parallel axis), not by scheduling.
- **Narrow the atomic dtype (packed atomics) — a count reduction, not a measured
  win.** Independently of #contributors, writing the reduction in a 16-bit float
  halves the bytes per contributor and packs **2 elements per atomic op**: CDNA4 has
  `BUFFER_ATOMIC_PK_ADD_{F16,BF16}`. Both of those are counts — of bytes and of
  atomic ops — and on a measured AMD atomic epilogue the halved count produced **no
  observed time change**. So this is a candidate to be timed, not a volume-to-time
  conversion you may assume. Constraints:
  - **`vec >= 2`**: the packed op needs 2 adjacent elements on the atomic's
    contiguous (output) axis held by one thread, so the value layout must be
    coalesced on that axis, or repacked into it (a `convert_layout` to a `vec=2`
    blocked layout) before the atomic.
  - **dtype range**: `fp16` has a narrow exponent range and overflows for
    large-magnitude accumulation; prefer **`bf16`** (same exponent range as fp32)
    for gradient-style reductions, accepting bf16 mantissa rounding (compare against
    a higher-precision oracle).
  Confirm in asm that it lowered to `buffer_atomic_pk_add_*`, not a scalar fallback.
  That confirmation is the **falsifier** — a scalar fallback means the edit did not
  happen, so stop — and it is not the evidence. Keep it on a timed A/B against the
  fp32 form under the same protocol (`../method/benchmark-hygiene.md`), and report the
  identity you got: halved bytes and halved op count are the thing that was
  established; the time effect is a separate claim that has to be observed to be made.

## Reuse a shared operand (load once, reuse)

When the kernel reads the SAME operand region more than once per work-item (attention
K/V across query rows; a GEMM operand across N tiles), stage it into LDS **once** and
reuse from LDS, instead of re-reading HBM each time.

```python
# gfx950: HBM -> LDS once, then reuse from LDS per consumer tile
gl.amd.cdna4.async_copy.buffer_load_to_shared(kv_smem, kv_ptr, offs)
gl.amd.cdna4.async_copy.commit_group(); gl.amd.cdna4.async_copy.wait_group(0)
for q in range(0, N_Q, BLOCK_Q):                 # attention: reuse K/V across all query rows
    k = kv_smem.index(0).permute([1, 0]).load(k_layout)
# gfx942 downgrade: sync staging gl.load -> smem.store -> smem.load (the 32-bit async form
# with order=[1, 0] lowers but measured slower than sync staging); no ds_read_tr behind permute.
```

Ref: FA K/V reuse + `buffer_load_to_shared` (upstream `f16_fa_gfx1250.py`,
`test_amd_direct_load_to_shared`). Verify: fewer HBM transactions / vmem %-of-peak
down when the reuse was the redundant traffic.

## Cut HBM bytes (narrow / cache / fuse)

When the HBM pipe is saturated, adding in-flight is inert — the only win is fewer bytes.
Narrow the operand dtype at the load boundary, bias L2 residency with the `cache`
modifier, or fuse a streaming read->write pass into one kernel.

```python
# narrow the dtype at the HBM boundary: fp8/fp4 buffer_load feeding scaled MFMA
a = gl.amd.cdna4.buffer_load(a_ptr, offs, cache="cg")   # cache modifier biases L2
acc = gl.amd.cdna4.mfma_scaled(a, a_scale, b, b_scale, acc)   # consume fp8/fp4 directly
```

Ref: `cdna3/cdna4.buffer_load(cache=)` + `mfma_scaled` narrow-dtype path (upstream
`test_amd_mfma_scaled`). gfx1250 (CDNA5 / MI450, not RDNA4) has an L2 prefetch, `tdm.prefetch`;
it does NOT exist on gfx950/gfx942. **gfx942 downgrade:** no `mfma_scaled` and no fp4/fp6 — the
narrowest matrix operand is fp8 in the **FNUZ** encoding (`e4m3fnuz` / `e5m2fnuz`) through the
regular `gl.amd.cdna3.mfma`; the `cache=` modifier carries over. Verify: achieved HBM bytes down
(vmem %-of-peak down); time moves iff truly BW-bound.

## IR / asm acceptance signals

| Change | What to confirm in IR/asm |
| --- | --- |
| buffer ops | `buffer_load_dwordx4` + `v_cndmask`; branch count drops vs `global_load` + `s_cbranch` clusters |
| async copy | `buffer_load ... lds` present and **no** `ds_write` on the load path; `wait_group` retires the buffer before reuse (gfx942 downgrade: `buffer_load_dword ... lds` only — a wider form means the 32-bit gate was not the one applied) |
| coalescing | per-lane HBM offsets form wide, aligned (128-bit) transactions; contiguous dim aligned with `size_per_thread` on the fast axis |

Use `scripts/dump_ir.sh` (shim; the tool lives in `kernel_workflow/scripts/kernel_tools/dump_ir.sh`)
and the recovery map in `layout-recipes.md`. See
`../gluon/memory-reference.md` for buffer-op offset units and fallback-value
rules.

## A load's width is issue-side; a transaction's width is access-side

The width in a load **instruction** is how many bytes one lane asks for. The width
of a memory **transaction** is how many contiguous bytes the access hardware moves
in one go, and that is decided by the addresses the lanes present together, not by
the opcode. Adjacent lanes on adjacent addresses are combined into wide
transactions whatever the per-lane request width was — so a narrow per-lane load
over a contiguous wave footprint already moves what a wide one would.

**What this does to a census.** Counting narrow-load opcodes counts **issue slots**.
It does not count bytes and it does not count transactions. An arm justified by
"the comparator emits none of these and we emit many" can therefore remove every
one of them, confirm the removal in the disassembly, and still measure nothing:
the transactions were wide before the edit and the only quantity that moved was
the issue count.

Read these instead of the histogram:

1. **The address delta between adjacent lanes** on the contiguous dim. If it is one
   element, the wave's footprint is already a single wide run and the opcode width
   is an issue-side detail.
2. **The contiguous run one instruction produces** = (lanes on the contiguous dim
   *within one group*) × (per-lane width). **The group count does not enter that
   product.** Splitting the contiguous dim across more groups leaves exact coverage
   intact and cuts the run — exact coverage and widest transaction are two
   independent constraints, and a layout can pass the first while losing the
   second. When they disagree, preserve transaction width first and byte count
   second.
3. **A transaction-side rate from the profiler** rather than an instruction count —
   the coalescing / partial-line rows in `../hardware/bound-class-signals.md` are
   measured on the access, which is the quantity under discussion.

When the bound behind an arm has the shape *"this saves instructions, not bytes"*,
ask first whether the machine is issue-bound at all. If bytes moved, bytes in
flight and occupancy are all unchanged by the edit, the bound is over the issue
rate alone — say so before spending the round, because that is the hypothesis the
arm is actually testing.

> `../method/benchmark-hygiene.md ## A compile-only kill step is a falsifier, not a price`
> says a structural reading does not price a change. This says a structural reading
> may not even be reading the quantity you think it is.

## Common failures

- Buffer ops with ambiguous fallback value, offset unit, or pointer shape ->
  silent wrong results; pin them explicitly.
- `ds_write` contending with `buffer_load_to_shared` on the LDS write port ->
  ~400-cycle stall (CDNA4); schedule MFMA after the `ds_write`, or move the small
  tensor to a GR/LW path off the critical write port.
- Treating `buffer_load_to_shared` as free: it consumes the SP-to-LDS FIFO
  (8 slots / SIMD pair); over-issuing stalls.
