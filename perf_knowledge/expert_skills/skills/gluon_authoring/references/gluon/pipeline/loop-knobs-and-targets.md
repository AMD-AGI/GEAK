# Gluon pipeline — per-target paths (gfx950 first) and the loop knobs

## gfx950 CDNA4 path

`async_copy` is a module holding the group ops (no `cdna4.commit_group`); destination first:

```python
gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_ptr, gmem_base, elem_offsets)   # producer
gl.amd.cdna4.async_copy.commit_group()
gl.amd.cdna4.async_copy.wait_group(n)                                              # consumer
acc = gl.amd.cdna4.mfma(a, b, acc)
```

Entry points: `global_load_to_shared`, `buffer_load_to_shared`, `commit_group`, `wait_group`,
`load_shared_relaxed`. Plain Triton gets `tritonamdgpu-pipeline` automatically; Gluon does not — the ring
is yours (`async-ordering.md ## The async forms — gfx950 only, and the shape differs from A5`). Copy
width per lane is **128 or 32 bit** (`../memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`);
LDS is **160 KiB/CU** (`../smem-lds-reference.md ## LDS is a second, independent occupancy limiter`).

## gfx942 downgrade

| gfx950 form | gfx942 downgrade | owner |
| --- | --- | --- |
| async ring at 128-bit/thread | **sync staging (A5) is the usual form**; async only at **32-bit** (3.8.0), clean tiling, destination `order=[1, 0]`, measured slower than sync on the one kernel tried | `marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`; `authored-overlap.md ### Worked examples — every one compiled and run on gfx942` |
| 160 KiB LDS/CU, 64 banks | **64 KiB, 32 banks** — re-derive ring depth and swizzle | `../smem-lds-reference.md ## LDS is a second, independent occupancy limiter`; `../../hardware/cdna3-gfx942.md` |
| `ds_read_*_tr_*` transpose reads | **absent** | `../atoms-reference.md ## CDNA3 gfx942 downgrade` |
| `cdna4.mfma_scaled` | **no scaled MFMA** | same |
| OCP fp8 (`e4m3fn`) | **FNUZ** fp8 (`float8e4b8`); OCP fp8 MFMA fails | same |
| `cdna4.buffer_load` / `mfma`, `AMDMFMALayout(version=4)` | `cdna3.*`, `version=3` | `../../hardware/cdna3-gfx942.md` |

The downgrade is a structural edit (sync two-barrier ring instead of wait + barrier), not a namespace swap.
gfx1250 (TDM descriptors, wave32) is out of scope.

## Roll the loop (cut i-cache pressure)

`tl.range(..., num_stages=, loop_unroll_factor=)` is usable in a Gluon body and sets `tt.num_stages` /
`tt.loop_unroll_factor` on the `scf.for` on 3.6.0–3.8.0 — accepted on each. What differs is whether a pass reads it:

- **`num_stages` is inert on the Gluon path (3.6.0–3.8.0; 3.8.0 source-checked):** its consumers
  (`add_schedule_loops` / `add_pipeline`) live in `make_ttgir`, and `add_stages` routes a Gluon kernel
  to `gluon_to_ttgir` instead. It is only a buffer-count
  budget parameter; a plain winner's value is not a Gluon depth
  (`../../tile-programming/pipeline.md ## Gluon: the pipeliner does not run`).
  The ring depth is the buffer count you allocate.
- **`loop_unroll_factor` is live on 3.8.0 only** (`gluon_to_ttgir` gained `passes.ttir.add_loop_unroll`);
  on 3.6.0–3.7.1 the identical line compiles and does nothing.

Both failures are silent. Verify separately: `tt.loop_unroll_factor` in the TTGIR (attribute landed),
body instruction count in the asm (pass ran), L1I hit rate / fetch bubble (it helped).

### Where the factor goes: on the loop, never on the launch

**`loop_unroll_factor` is a keyword of the loop construct, not a launch kwarg** —
`kernel[grid](..., loop_unroll_factor=8)` is not a spelling of anything (3.8.0 `HIPOptions` has no such
field), so its failure says nothing about the knob. Shape-dependent factors go through a constexpr:

```python
def _unroll_for(m: int) -> int:
    return 8 if m <= 256 else 1                     # host-side table
kernel[grid](..., UNROLL=_unroll_for(m), num_warps=4)

@gluon.jit
def kernel(..., UNROLL: gl.constexpr):
    for k in tl.range(0, K_TILES, loop_unroll_factor=UNROLL):
        ...
```

An annotated loop inside a `@gluon.jit` helper carries its literal factor into every caller — make it a parameter.

### `=1` is documentation, not a muzzle — and the amplify case is narrower than it looks

- Factor < 2 means no unrolling (`triton/language/core.py` `range` docstring), so `=1` on an authored ring
  documents but enforces nothing; the defence is the authored ordering plus `disable_licm=True` where
  hoisting is the hazard.
- **Amplify (`>1`):** the compiler-side way to **partially** unroll a compile-time loop (`gl.static_range` is
  all-or-nothing) and to unroll a runtime-bounded loop (bounds loaded from memory); the hand-written
  alternative is the nested-loop form in the downgrade table below. The factor
  need not divide the trip count (the pass emits a remainder). Clean idiom: annotate the steady loop,
  peel the tail with `gl.static_range`. Raising it moves VGPRs → occupancy/spill threshold — sweep it with
  tile, `waves_per_eu` and the `llvm_fn_attrs` sched strategy (`../../tile-programming/instruction-scheduling.md`).

```python
begin, end = gl.load(starts + seq), gl.load(starts + seq + 1)     # runtime bounds
for c in tl.range(0, gl.cdiv(end - begin, CHUNK), loop_unroll_factor=UNROLL):
    state = _chunk_step(state, c)                                  # body x UNROLL + remainder
```

### Auditing a tree for this knob — do not grep for `tl.range`

`range` is often imported under an alias (`from triton.language.core import range as loop_range`): walk
the AST and select every `Call` with a `loop_unroll_factor` keyword, whatever the callee name.

### The downgrade is a structural edit, not a different keyword

| wanted (below 3.8.0) | write instead |
| --- | --- |
| suppress (`=1`) | nothing — no pass unrolls `range`/`tl.range`; but `gl.static_range` still fully unrolls at trace time |
| amplify, compile-time trips | `gl.static_range` |
| amplify, runtime trips | outer dynamic loop over `count // F` with inner `gl.static_range(F)`, plus a remainder loop |

On any version: plain `range` keeps a loop rolled, `gl.static_range` unrolls it fully; factor a fat body
into a shared `@gluon.jit` helper to avoid duplicated code.

## Footguns

- `wait_group` depth mismatch = silent wrong answer that benchmarks faster: count groups, commit
  unconditionally, gate on bit-exactness (`async-ordering.md ## Getting wait_group right`).
- `wait_group(expr)` with a loop-variable term inside a dynamic `range` fails at trace time (compile-checked 3.8.0) — the count must be a trace-time int.
- Ring depth costs LDS and VGPRs (`../../tile-programming/slicing.md`): 2 buffers is the smallest ring;
  where waves are already VGPR- or LDS-capped, added depth or prefetch has measured as a regression;
  aggressive unrolling can raise the `s_nop` count.
- CuTeDSL concepts with no CDNA counterpart: `PipelineTmaAsync`/mbarrier (→ `commit_group`/`wait_group`),
  TMA multicast, CLC, `cutlass.range(prefetch_stages=)` (→ manual prologue); warpgroup specialization →
  `warp_pipeline_stage` at `num_warps=8` (`warp_specialize` fails on gfx950 / gfx942 on 3.8.0: "only supported on gfx1250").
