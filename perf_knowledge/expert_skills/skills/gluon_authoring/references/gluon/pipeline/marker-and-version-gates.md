# Gluon pipeline — the stage marker, the fork-only predicate, and the gfx942 async width gate

## `gl.amd.warp_pipeline_stage` — the official marker path

Upstream, opt-in, independent of `num_stages`. The symbol exists on **3.7.0+** (absent on 3.6.0 —
`../pipeline-reference.md ## Which spelling exists on which build — the four-version probe table`); the phase
offset was GPU-verified race-free on gfx950 with 3.8.0 and 3.9.0 only — on 3.7.1 the conversion emitted no in-loop
stage-border barriers and the corrected snippet returned wrong results
(`../../tile-programming/warp-pipeline.md`).
`gluon_to_ttgir` already calls `add_warp_pipeline`; `WarpPipeliner.cpp` ignores loops with no border marker.

```python
for k in range(K // BK):                       # dynamic loop (range / tl.range -> scf.for); static_range unrolls: no pipelined_for, but the phase offset, border barriers and s_setprio are still emitted (straight-line ping-pong; 3.8.0 compile)
    with gl.amd.warp_pipeline_stage("mem", priority=1):
        ...                                    # LDS reads, refills, address VALU
    with gl.amd.warp_pipeline_stage("mfma", priority=0):
        ...                                    # matrix ops
```

- **Shipped gfx950 GEMMs give the memory stage the HIGHER priority** (AMD's gfx950 tutorial inter-wave
  GEMMs, aiter, the installed `warp_pipeline.py` docstring); the measured exception is a four-stage
  attention loop where compute-high came out ahead with P's `exp2` pinned in the PV stage (without the pin
  the polarities tied; gfx950 / 3.8.0). Time both polarities
  (`../../tile-programming/warp-pipeline.md ## Stage priority: the shipped memory-high convention`).
- **Priority-reset rule:** if any stage sets `priority`, every stage that does not is reset to 0; if no
  stage sets one, **no `s_setprio` is emitted at all** — you get `sched_barrier` region cuts without the
  priority mechanism. So `mem=1` with `mfma` bare equals the pair above.
- **Efficacy before timing:** dump the asm (`kernel_workflow/scripts/kernel_tools/dump_ir.sh`) and confirm
  `s_setprio` at the marked boundaries in the hot loop; then time **both** priority assignments.
  `s_setprio` costs no LDS/VGPR/AGPR/MFMA slot. Hand-written `s_setprio` inline asm in the same loop is a
  second writer of the same wave priority; the last `s_setprio` executed is what holds, which need not be the marker's plan.
- **Pass constraints (3.8.0):** ≥ 2 clusters or it errors (compile-checked); an `async_wait` may sit only on a border (between
  stages or as a stage's first op) — *which* border is a correctness choice: a publish-wait directly before the stage that reads the tile races across the two groups (GPU-verified gfx950 / 3.8.0, `../../tile-programming/warp-pipeline.md ## LDS hazards close a stage early`) — and `gl.barrier()` nowhere in the pipelined loop; no `scf.for`/`scf.while` inside a stage; a stray op between stages still compiles (silently).
  Lowers to `s_setprio` + `sched_barrier` around each cluster in `make_llir`.
- **Gate 0: `num_warps >= 8`.** On 3.8.0 the two groups split at thread id 256, so below 8 warps every
  wave is in one group and there is no inter-wave phase shift (compile-checked), while the
  `sched_barrier` still acts as an intra-wave wall — a timing change there comes from the wall, not
  the marker (`../../tile-programming/warp-pipeline.md ## Gate 0: does the launch have two wave groups at all?`).
  Measured wins were on compute-bound loops staging through LDS at 2 waves/SIMD; on attention bodies
  where 4 -> 8 warps cut occupancy it measured slower — A/B it on the loop at hand. Loop covers
  `K_ITERS - (NUM_BUFFERS - 1)` tiles with the remainder in an epilogue outside any stage. Skeleton and
  gates: `../../tile-programming/warp-pipeline.md`.

## `gl.warp_predicate` — a fork-only extension, and a design boundary

**Absent from upstream 3.6.0–3.8.0** (vendor fork only; 3.8.0 re-checked by `hasattr`). It runs a body under a per-wave condition
(`s_and_saveexec_b64` + `s_cbranch_execz`: warp-uniform branch, no cross-wave reduction, no barrier).

- **On stock Triton 3.8.0 there is no explicit per-wave predicate:** inline asm is elementwise and cannot enclose a Gluon region
  (`../inline-asm-reference.md ## Class 4 — synchronization`) and a 0-d `if` is CTA-uniform. A per-row
  branch on a per-lane value still lowers to an exec-masked block with `s_cbranch_execz`, which skips when
  no lane needs it (`../../workloads/attention.md ### Conditional rescaling (skip the per-block acc *= alpha)`).
  Record a missing predicate as an API ceiling with the build identity (`../../tile-programming/compiler-contract.md ## Toolchain identity`).
- Where the fork has it: in a compute region the predicate's control flow issues ahead of the region's
  first MFMA, in a memory region it does not delay an MFMA (`../../tile-programming/llir-codesign.md ## Attention: the co-execution budget`).

## The gfx942 async-copy width gate — available, and narrow

**On Triton 3.8.0 `cdna4.async_copy` lowers on gfx942 only at 32 bits/thread** (`supportsDirectToLdsLoadBitWidth`:
CDNA3 = {32}, CDNA4 = {128, 32}). Shared contract (native width per lane, exact cover of the contiguous
dim, swizzle within a warp): `../memory-reference.md ## Async Copy To Shared`. Verified on Triton 3.8.0.

| | gfx950 (CDNA4) | gfx942 (CDNA3) |
| --- | --- | --- |
| legal per-thread widths | 128 / 32 bit | **32 bit only** |
| 64 bit | **fails** | **fails** |
| bytes per lane per instruction (widest) | 16 B | 4 B |
| destination `order` | per contract | **must match the source layout's fast dim** (`[1, 0]` for a row-major tile) |
| vs sync staging | A/B it | measured slower on the one kernel tried — A/B it |
| namespace | `gl.amd.cdna4.async_copy` | same `cdna4` entry points (`gl.amd.cdna3` has no `async_copy`) |

| bytes/thread on gfx942 | result |
| --- | --- |
| **4 (32-bit)** | lowers, runs, bit-exact (bf16, clean tiling) |
| 8 / 16 / 32 | `failed to translate module to LLVM IR` (`buffer_load_to_shared`: `builtin.unrealized_conversion_cast`); `global_load_to_shared` fails `PassManager::run failed` (`failed to legalize operation 'ttg.async_copy_global_to_local'`) |

- **Same error for a layout-contract violation:** a 32-bit chunk with ragged tiling (`4 threads × 1 elem`
  over a 64-wide dim) fails identically, `64 × 1` succeeds. Read a lowering failure as a layout question first.
- **`order` gates on its own:** with width and tiling correct on a row-major source, `order=[0, 1]` turns a
  bit-exact lowering into `failed to translate module to LLVM IR` (at `vec=1, max_phase=1` and `vec=4, max_phase=16`). If the
  consumer needs `[0, 1]` (e.g. V for `amdg.in_thread_transpose`), give the async destination a second
  shared layout differing only in `order` — same `memdesc`, round trip stays bit-exact.
- **Measured on one real kernel:** converting one sync-staged KV buffer of an attention decode to
  `async_copy` stayed bit-exact (plus a determinism race-test) on all four minors and was slower on
  each of them, by more on the newer builds because the *sync* arm got faster while async stayed flat.
  A different kernel (more reuse per staged byte, a deeper ring) could come out otherwise — A/B before
  closing the direction. Compare absolute times per build, not ratios
  (`../../method/benchmark-hygiene.md ## Cross-version ratios: attribute the move before claiming it`).
