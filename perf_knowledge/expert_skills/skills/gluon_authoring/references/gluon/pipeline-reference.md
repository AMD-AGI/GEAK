# Gluon — pipeline

Backbone layer 4. Gluon has **no** CuTeDSL `PipelineTmaAsync` / mbarrier library. It also
ships no auto-pipeliner: `gluon_to_ttgir` never calls `add_schedule_loops` / `add_pipeline`
on **any** upstream version (checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0). So overlap is either
**authored** (§ below) or **re-injected** — and re-injection is measured, not theoretical:
it recovered the full pipeline gap on two kernels and on all four versions.

---

## Re-injecting plain's pipeliner — the measured recipe

`add_schedule_loops` and `add_pipeline` are present in `libtriton` on all four versions;
only the Python pass list omits them. Two ways to reach them, and the first edits nothing:

```python
import gluon_swp                        # scripts/gluon_swp.py -- wraps gluon_to_ttgir
with gluon_swp.pipelined(2, buffer_ops=True):
    out = my_kernel[grid](...)          # compile INSIDE the block; Triton caches
```

`gluon_swp.py` runs the passes as a second pass manager over the module the stock
`gluon_to_ttgir` returns. **Verified byte-identical TTGIR to the `compiler.py` splice on
3.6.0 / 3.7.0 / 3.7.1 / 3.8.0** — same md5 both armed and unarmed — so nothing is given up by
not touching the file, while a read-only or shared site-packages, a `pip --force-reinstall`,
and a crash between apply and revert all stop being hazards. `scripts/patch_reinject.py` is
the on-disk form, kept for when you want the pass list visible in `compiler.py` while reading.

**`TRITON_GLUON_SWP_PIPELINE` is not the knob.** It, `TRITON_GLUON_COOP_LDS` and
`TRITON_GLUON_PINGPONG` belong to a vendor fork's `GetEnv.h`; no upstream version reads them.
Measured on clean 3.7.1 and 3.8.0 they are *tolerated and inert* — as is a knob invented on the
spot — so the failure is silent and reads as "the technique does not work here".

**Two conditions are required and neither alone does anything.** From a 2×2 on aiter's
rmsnorm, all four cells bit-exact, `PIPELINED` read off the IR (peeled prologue loads +
a loop-carried `iter_arg`):

| loads written as | loop | pipelined |
| --- | --- | --- |
| `gl.amd.cdna3.buffer_load` | `range(...)` | ✗ |
| `gl.amd.cdna3.buffer_load` | `tl.range(..., num_stages=2)` | ✗ |
| `gl.load` | `range(...)` | ✗ |
| **`gl.load`** | **`tl.range(..., num_stages=2)`** | **✓** 2→4 loads, `iter_args` 0→1 |

1. **The loop must be a pipelining CANDIDATE, and whether it is depends on the DOT.**
   `add_schedule_loops` takes the launch-level `num_stages` as the default for a loop it
   considers a candidate, and it decides that from the loop's contents:

   | loop | needs an annotation? |
   | --- | --- |
   | contains a `tl.dot` | **no** — a bare `range` pipelines from `num_stages` alone |
   | dot-free | **yes** — `tl.range(..., num_stages=N)`, or nothing happens |

   Measured both ways on plain, same launch `num_stages` 1/2/3: a bare-`range` **GEMM** went
   2 → 4 → 6 loads with `memdesc_index` 0 → 4 → 6 and **no `tt.num_stages` in the IR at all**,
   while a bare-`range` **dot-free reduction** stayed byte-identical. Confirmed on the Gluon
   side under injection: an un-staged GEMM with a bare `range` pipelines (`memdesc_index 4`,
   1.087× vs hand-staged) exactly like the `tl.range` version (1.100×).

   That is why aiter's `moe_op` is pipelined while writing plain `for k in range(...)`, and
   why `rmsnorm` — dot-free — has to write `tl.range(num_stages=2)` and does.

   Gluon exposes no `range` of its own (only `static_range`, which unrolls), but `tl.range`
   **is** usable from a `gluon_jit` body when you need the dot-free case. And
   `tl.range(..., num_stages=None)` **inherits** the launch value, which is how aiter's
   `_attn_fwd` works: `num_stages = None if ENABLE_PIPELINING else 1` on the inner loop with
   the number coming from the autotune config. So `None` on the loop is not "unset".
2. **The loads must still be `tt.load` when the pipeliner runs.** Plain's own `make_ttgir`
   orders `add_schedule_loops` #15, `add_pipeline` #16, `add_convert_to_buffer_ops` **#28** —
   twelve passes later. So plain's pipeliner only ever sees `tt.load`. An anchor written the
   way the transcription runbook asks (explicit `gl.amd.cdna3.buffer_load`, because
   `gluon_to_ttgir` runs no buffer conversion) hands the pipeliner ops it cannot recognise.

> **Those two pieces of guidance pull against each other**, and that is not a bug in either.
> Buffer ops are worth real performance on a memory-bound body; the pipeliner needs
> `tt.load`. The splice resolves it by restoring plain's ORDER: pipeline first, then
> `add_convert_to_buffer_ops`. Write `gl.load`, arm `TRITON_GLUON_SWP_BUF=1`, and the final
> IR is buffer ops *and* pipelined — measured 4 `amdg.buffer_load`, 0 `tt.load`.
> That half is **opt-in**: arming it on an anchor whose loads are already `buffer_load`
> aborts with `PassManager::run failed`.

### On a dot kernel: un-write the staging

The faithful anchor shape — `allocate_shared_memory` + `gl.barrier()` +
`smem.load(A_DOT_OPERAND)` — carries every blocker at once, and building the LDS path is
exactly what the pipeliner exists to do. So hand the loop back to it. Measured on a 2048³
fp16 GEMM, all arms numerically identical:

| arm | LDS ops (alloc/store/load) | memdesc_index | vs hand-staged |
| --- | --- | --- | --- |
| hand-staged, no pipeline | 2 / 2 / 2, 2 barriers | 0 | 1.000 |
| un-staged, splice **off** | 0 / 0 / 0 | 0 | **0.811** |
| un-staged, splice **on** `ns=2` | **2 / 4 / 4**, 0 barriers | **4** | **1.088** |

The pipeliner built the staging *and* multi-buffered it (`memdesc_index 4`) with no explicit
barrier. Note the middle row: **un-staging without the splice is a 19% regression** — the
two halves go together or not at all.

### And on attention — two dots chained through a softmax

The shape worth checking separately, because the second dot's A operand is the first dot's
output and `acc` / `m` / `l` are all loop-carried, so a pipeliner that prefetches a GEMM's
operands might still refuse it. It does not. Minimal FA-forward body (BM=128, BN=32, D=128,
layouts recovered from a tuned `_attn_fwd` champion), un-staged, all arms numerically
identical at `max_rel 2.2e-3`:

| arm | loads | LDS alloc/store/load | memdesc_index | barriers | vs un-pipelined |
| --- | --- | --- | --- | --- | --- |
| splice off | 3 | 0 / 0 / 0 | 0 | 0 | 1.000 |
| `ns=2` | 5 | 2 / 4 / 4 | 4 | 0 | **1.121** |
| `ns=3` | 7 | 2 / 6 / 6 | 6 | 0 | **1.126** |

Same signature as the GEMM: prologue peeled, K/V staging created by the pass, multi-buffered,
no barrier authored. So all three shapes this pack cares about — dot-free reduction, GEMM,
attention — pipeline under the same two conditions.

**But size the prize before spending the round.** On that champion the pipeline is worth much
less than on rmsnorm, and it is not monotone in shape: at its own tile, `num_stages` 2 vs 1
measured 1.083 at S=2048, **0.980** at S=4096 (a small pessimisation) and 1.040 at S=8192.
Its patch notes' "~1.5–1.7×" is the *combined* effect of `BLOCK_N=32` + `num_stages=2` against
the shipped `BLOCK_N=64` + `num_stages=1`, not the pipeline alone. Measure `plain@ns=1` at the
champion's own tile, which is the only control that separates the two.

### Recovery measured, per version

aiter's rmsnorm, ratio vs the **shipped** plain kernel (which pipelines its row loop at
`tl.range(num_stages=2)`), interleaved arms in one process, all bit-exact:

| | 3.6.0 | 3.7.0 | 3.7.1 | 3.8.0 |
| --- | --- | --- | --- | --- |
| faithful anchor (`buffer_load`, bare `range`) | 0.769 | 0.749 | 0.748 | 0.758 |
| **re-injected** (`gl.load` + `tl.range(2)` + splice) | **1.062** | **1.039** | **1.030** | **1.026** |
| re-injected at `ns=3` | 1.006 | 0.991 | 0.986 | 0.994 |

So the gap closes on every version and the recovered anchor slightly **exceeds** shipped
plain. `ns=2` beat `ns=3` on all four — depth is a knob, not a monotone. 3.6.0 works even
though it has no `add_warp_pipeline` at all, which confirms the win is `schedule_loops` +
`pipeline` and not the warp pipeliner.

### The other two AMD pipeline mechanisms, measured on gfx942

They are **not** the same lever and only one of them is reachable from plain here.

**Block ping-pong (`add_block_pingpong`) does fire on gfx942, in a narrow window.** It runs
only after the pipeliner and adds its own constraints. Measured on a plain fp16 GEMM,
`s_setprio` / `sched_barrier` counted in the `.amdgcn`:

| tile | `num_warps` | `num_stages` | `s_setprio` |
| --- | --- | --- | --- |
| 256×256×64 | **8** | **2** | **8** ✓ fired |
| 128×128×64 | 8 | 2 | 0 — tile below `mediumTile` |
| 256×256×64 | 4 | 2 | 0 — different window for `nw4` |

So **read `s_setprio` out of the ISA; never infer ping-pong from a source config.** And on a
Gluon anchor it is unreachable while the staging is hand-written: it collects only
`local_load`s whose source is a loop-carried `BlockArgument`, and a hand-written one is
sourced from `memdesc_index`. Un-writing the staging is what puts it back in reach.

**Async copy / direct-to-LDS is NOT reachable from plain on gfx942, even forced.** Setting
`triton.knobs.amd.use_async_copy = True` changed nothing: `async_copy` / `buffer_load_to_local`
count stayed **0** in the TTGIR, identical to the knob off. The hardware supports it
(`supportsBufferLoadToLocal()` covers CDNA3 at 32-bit vector width) and the Gluon surface
imports on every version including 3.6 — but plain's lowering will not emit it here, so on
gfx942 this form has to be **authored explicitly** in Gluon
(`gl.amd.cdna4.async_copy.*`), not recovered from a plain champion. Untested by this pack.

### What this does not do

- **It is not upstream.** No upstream version calls these passes from `gluon_to_ttgir`; a
  patched `compiler.py` is a local change and every measurement taken under it must say so.
- `add_block_pingpong` still will not fire on hand-authored staging: it only collects
  `local_load`s whose source is a loop-carried `BlockArgument`, and a hand-written one is
  sourced from `memdesc_index`. Un-writing the staging is what makes it reachable.
- `warp_pipeline_stage` is a **different** mechanism (below), not this one.

---

## Authored overlap (no compiler patch)

The upstream-only path. Still the right choice when a patched `compiler.py` is unacceptable,
and the only option on 3.6 for warp-level pipelining:

```text
buffer_load_to_shared / cdna4.async_copy / gfx1250 TDM
  → commit_group (producer)
  → wait_group   (consumer)
  → mfma / wmma
```

LLVM passes (`LLIR_SCHED`, `RA_HINTS`, `AMDGCN_AS`) interleave the hot loop — toggle +
IR-verify (`../tile-programming/compiler-contract.md`).

### `gl.amd.warp_pipeline_stage` — the official marker path

Upstream, opt-in, and **independent of `num_stages`**: `WarpPipeliner.cpp` bails out unless
the loop body contains at least one border marker, so nothing happens to a loop you did not
annotate. `gluon_to_ttgir` already calls `add_warp_pipeline` on 3.7+, so this needs **no
compiler patch**.

```python
for k in range(K // BK):                       # bare range is fine here
    with gl.amd.warp_pipeline_stage("load", priority=1):
        ...                                    # cluster 1
    with gl.amd.warp_pipeline_stage("mfma", priority=3):
        ...                                    # cluster 2
```

Constraints, from the pass itself: **≥ 2 clusters** or it errors; a barrier / `async_wait` /
`AsyncTDMWait` may sit only *between* stages, never inside one; and no `scf.for` / `scf.while`
inside a stage. It emits `s_setprio` + `sched_barrier` around each cluster in `make_llir`.

**Availability is the thing to check first — it is the one capability that is version-gated
rather than probe-gated:**

| | 3.6.0 | 3.7.0 | 3.7.1 | 3.8.0 |
| --- | --- | --- | --- | --- |
| `gl.amd.warp_pipeline_stage` | **absent** | ✓ | ✓ | ✓ |
| `add_warp_pipeline` in `libtriton` | **absent** | ✓ | ✓ | ✓ |
| `add_schedule_loops` / `add_pipeline` in `libtriton` | ✓ | ✓ | ✓ | ✓ |
| called from `gluon_to_ttgir` | ✗ | ✗ | ✗ | ✗ |
| `gl.amd.cdna4.async_copy` (all 5 entry points) | ✓ | ✓ | ✓ | ✓ |

On 3.6 there is no marker path at all — re-injection is the only pipelining available.
`cdna4.async_copy` imports on every version including 3.6; the `cdna4` naming is a module
name, not a hardware gate — `TargetFeatures::supportsBufferLoadToLocal()` covers CDNA3 and
CDNA4, so direct-to-LDS exists on gfx942 at 32-bit vector width (128-bit needs CDNA4).

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
