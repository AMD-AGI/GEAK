# Gluon pipeline — re-injecting plain's pipeliner (diagnostic)

**Single home for running plain's `add_schedule_loops` + `add_pipeline` over Gluon TTGIR.** It is the
diagnostic end of the overlap mechanisms (`../../tile-programming/pipeline.md ### The overlap options and the signal for each`)
and has two recorded uses: (1) below the parity gate, to size the `lost_pipeline` debt (the `plain@ns=1`
control in `../../method/recover.md` is the cheaper half); (2) a fallback when a hand-written ring has
not reached parity. Numbers from it are `injected`: they describe plain's overlap on a plain-style body,
not an incumbent (already-Gluon, hand-staged) kernel, and its ceiling is plain's own overlap. Measurements below are **gfx942** unless stated; gfx950
statements are marked (the compile behaviour in the next section, and the gfx950 / Triton 3.8.0 FA-forward
depth result in `### Depth: ns=2 measured best on gfx942`). On gfx950 `add_pipeline`
defaults to its async path; GEMM bodies there were not measured.

### Precondition for re-injection: plain loads and a dot in the loop

The anchor that `transcribe.md ## Port checklist` produces (explicit `buffer_load`, explicit
`allocate_shared_memory` staging, `mfma` / `mfma_scaled`) is **not** a re-injection candidate, and the
two ways it fails look different (verified on Triton 3.8 / gfx950):

- `gluon_swp.pipelined(2, buffer_ops=True)` on a body with `buffer_load` stops the compile with
  **`failed to legalize operation 'amdg.buffer_load' that was explicitly marked illegal`**.
- `gluon_swp.pipelined(2)` on the same body compiles and changes nothing: the TTGIR has no
  `ttg.memdesc_index` and no `async_wait` — a silent no-op that times like the anchor.

Re-injection needs a separate diagnostic body written plain-style: `gl.load` on pointer tensors, the dot
fed by `gl.convert_layout` to dot-operand layouts (no user `allocate_shared_memory`), inside a `range`
loop. Confirm it fired before timing (`### Confirm it fired from the IR, before timing` below). On gfx950 the
cheaper diagnostic is usually the `plain@ns=1` control (`recover.md` §2.1), and the authored ring
(`recover.md ### The async-ring loop form that reaches plain parity (gfx950)`) reaches plain's overlap
without this route.

## Re-injecting plain's pipeliner — the measured recipe

**No upstream `gluon_to_ttgir` (3.6.0–3.8.0) calls `add_schedule_loops` / `add_pipeline`, but both ship in
`libtriton`** — so no rebuild and no edit to an installed file: wrap the stage function and run a second
pass manager over the module it returns (verified 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0).

### Seams and tools: add_stages_inspection_hook, gluon_swp.py, patch_reinject.py

**Seam 1 — upstream `add_stages_inspection_hook` (prefer for new work).** The hook is called twice per
compile: `hook()` for the cache key, then `hook(backend, stages, options, language, capability)`.
The no-argument call must return a `(key, hash)` pair (3.8.0 `triton/runtime/jit.py` unpacks it into the
cache key; contract also in `triton/compiler/compiler.py`); a five-argument-only hook raises `TypeError`.

```python
from triton.knobs import runtime
from triton.compiler.compiler import Language
def _reinject(backend=None, stages=None, options=None, language=None, _=None):
    if backend is None:
        return ("reinject", "1")                  # (key, hash) -- required
    if language is not Language.GLUON:
        return
    stock = stages["ttgir"]
    def ttgir(src, metadata):
        mod = stock(src, metadata)
        ...                                       # second pass manager: the pass list below
        return mod
    stages["ttgir"] = ttgir
runtime.add_stages_inspection_hook = _reinject
```

**Seam 2 — `scripts/gluon_swp.py` (pack script, reference pass list).**
`with gluon_swp.pipelined(2, buffer_ops=True): out = k[grid](...)` — compile inside the block. Bare CLI prints `capabilities()` (probed);
`--selftest` runs offline. Refuses `num_stages < 2` and a tree that already splices the passes. Its default
post recipe `plain_itt` splices plain's tail and reports what it applied (an absent pass is listed as
`-name`); on gfx950 it adds `add_coalesce_async_copy` (the pipeliner's async path needs it). Byte-identical
TTGIR to the on-disk splice on all four minors (gfx942 only — re-check on gfx950).

**On-disk form — `scripts/patch_reinject.py apply|revert|status`.** Splices before `add_warp_pipeline`
(3.7+; after the last `add_*` on 3.6), `.orig_swp` backup. It stops at `add_pipeline` (no tail).

**Env knobs: no Triton env var re-injects.** `TRITON_GLUON_SWP*` / `TRITON_GLUON_ASYNC` are this pack's
knobs, read only by a `compiler.py` patched with `patch_reinject.py` / `patch_async_reinject.py` (not in the
cache key); `TRITON_GLUON_SWP_PIPELINE` / `_COOP_LDS` / `_PINGPONG` are read by nothing (silent null).

`scripts/pipeline_survey.py <root>` screens a plain source tree by pipeline form (A cross-iteration
pipeline — the re-injectable one, B block ping-pong, C async/direct-to-LDS) from source text; only a dump
settles what a kernel actually dispatches.

Pass list (plain's own order):

```text
add_optimize_dot_operands; add_schedule_loops(ns); add_pipeline(use_async_copy, use_block_pingpong)
[gfx950 async path: add_coalesce_async_copy]
tail: add_convert_to_tensor_ops; canonicalizer; add_remove_layout_conversions; add_reduce_data_duplication;
      add_in_thread_transpose (arch-gated: is_in_thread_transpose_enabled); add_move_up_prologue_loads;
      add_block_pingpong(ns)          # 3.6.0 lacks convert_to_tensor_ops and move_up_prologue_loads
buffer_ops=True / TRITON_GLUON_SWP_BUF=1: canonicalizer; canonicalize_pointers; canonicalizer; convert_to_buffer_ops
```

**The pipeliner and the tail are a pair.** Pipeliner without the tail: every operand takes a second LDS
round trip (`local_load` stays blocked + a `convert_layout`) → `OutOfResources: Required <n>, Hardware limit`
on gfx942 (64 KiB), a silent slowdown on gfx950 (160 KiB), and can be worse than no injection by a multiple;
the tell is **register spill** (64-bit pointer tensors live across peeled stages) — check `spill=`, not
`shared`. Without `in_thread_transpose` one wide LDS read collapses into many narrow ones; with it the
pipeliner emits the rotating shared layout Gluon has no constructor for (verify can reach PASS instead of
RECONCILED). Tail without the pipeliner returns **NaN** (`add_block_pingpong` assumes the pipeliner ran).

### Conditions for the pass to fire, and for it to pay

Items 1-3 are what the pass needs to fire at all; items 4-5 decide whether it pays (measured, gfx942).

1. **The anchor is a `tt.dot`:** a dot loop pipelines on a bare `range`; a dot-free loop needs
   `tl.range(..., num_stages=N)` (usable from a `gluon.jit` body; semantics: `triton/language/core.py` `range`
   docstring) and `gl.load`, not `buffer_load`.
2. **The loads must still be `tt.load` when the pipeliner runs** — `gl.load`, not
   `gl.amd.cdna{3,4}.buffer_load`. In plain `make_ttgir` `add_schedule_loops` /
   `add_pipeline` run before `add_convert_to_buffer_ops` (`triton/backends/amd/compiler.py` `make_ttgir`,
   3.8.0 source-checked), so in plain's own flow the pipeliner does not see buffer ops. Write `gl.load` and arm
   `buffer_ops=True` to end on buffer ops anyway (measured 4 `amdg.buffer_load`, 0 `tt.load`). `buffer_ops`
   fails three ways: loads already `buffer_load` → `PassManager::run failed` / `failed to legalize operation
   'amdg.buffer_load/store'`; a `buffer_store` → **`LLVM ERROR: Fatal pipeliner error`, which kills the
   interpreter**; one buffer op anywhere in the function, even outside the loop
   (`TritonAMDGPUCanonicalizePointers` runs function-wide). It also cost extra on one kernel.
3. **The staging must be the pipeliner's own.** Remove hand `allocate_shared_memory` + barriers + sliced
   loads together: one `ttg.barrier` left in the loop makes `isSafeToPipeline` skip it silently.
   **`'ttg.local_alloc' op pipeliner doesn't know how to predicate this op` = hand staging not removed** (a
   mutable `local_alloc`), not a language limit. Scope is per dot: it needs one dot whose operands SSA-trace
   back to a `tt.load` (a `local_store` breaks the walk); on an attention body with one pure-register dot it
   fired and multi-buffered the loop (3 dots / 104 `v_mfma` → 6 / 208, that kernel's best arm).
4. **Hand register-prefetch and loop-variant `scf.if` left in the loop.** The pipeliner is the prefetcher:
   on gfx942 FA-fwd in-body K/V loads gained clearly from `ns=2`, a manually prefetched loop about nothing. Split a causal mask into
   two loops (plain's shape) — that also unblocks `BlockPingpong`.
5. **A loop with a trip count.** A loop that runs once only pays a peeled prologue/epilogue.

### Un-stage and inject together: measure three arms

| arm | reads as |
| --- | --- |
| hand-staged, no injection | the faithful baseline |
| **un-staged, injection OFF** | **a regression** — staging removed, nothing rebuilt it |
| un-staged, injection ON | the recovery (`injected`) |

Judge the net on the same-window per-rep ratio (armed-and-de-staged ÷ original anchor), each arm in its own
process. **Pre-check: plain's own `ns=1` vs `ns=2`** — where plain gains nothing on a version, the recovery did
not pay there either (held on every version tested, gfx942). Re-injection and hand-written staging do not
combine in one loop (the pass skips or rejects it, items 3-4); the explicit-smem version stays the starting
point of a hand-built ring.

### Confirm it fired from the IR, before timing

- **dot loop:** `ttg.memdesc_index` appears; `local_alloc` / `local_store` / `local_load` counts rise toward
  plain. **dot-free loop:** `memdesc_index` stays 0 (register prefetch) — read `iter_args` 0→1, load count
  scaling with depth (2→4→6), `tt.num_stages` on the `scf.for`, a peeled prologue.
- Either way `s_waitcnt lgkmcnt(0)` relaxes to `lgkmcnt(N>0)` (`asm_loop_audit.py`). IR identical armed vs
  unarmed = the pass ran and rewrote nothing (staging, `buffer_load` or a hand prefetch left in).
  `probe_levers.py --all` only says the symbols exist.
- A missing `s_setprio` is not counter-evidence unless plain's own `ns=2` build shows it.

**Cache trap (both halves give false negatives).** In process: with Seam 1 the hook's hash is part of
the specialization key (`triton/runtime/jit.py`, `("custom_pipeline", hash)`), so arms with different hooks
recompile; the Seam 2 wrapper and the `patch_reinject.py` knobs are **not** in the key (`(function, signature, constexprs)` only) and
**`TRITON_ALWAYS_COMPILE=1` does not fix that** (it only bypasses the disk cache) — run one wrapper arm per
process. On disk, the wrapper / pack env knobs are not in the key and a per-arm `TRITON_CACHE_DIR`
does not encode depth: use `triton.knobs.cache.dir = f"/tmp/run_{arm}_{gluon_swp.cache_tag()}"` (encodes
depth, post recipe, buffer_ops, ping-pong, async) and confirm the tell in each arm's own `.ttgir`.

### Depth: ns=2 measured best on gfx942

Recorded on gfx942 in the Triton 3.6.0–3.8.0 campaign of `### What to expect, and what to measure yourself`.
`num_stages − 1` buffers (`ns=2`: single buffer + peeled prologue). **`ns=3` was worse than `ns=2` on each gfx942 kernel measured; one refused to launch**; on FA-fwd
`ns=4` took the chained-dot path and regressed via the occupancy cliff. Two different ceilings: the
per-workgroup allocation refuses the launch; the LDS/CU divisor (gfx950 160 KiB, gfx942 64 KiB) silently
halves workgroups/CU. `recover_gluon.py`'s `LDS:` line is an upper bound (no liveness reuse); measure with
`$KT/probe.py measure`. A depth plain cannot compile is unavailable here too (same pass). On FA-fwd the
injected prologue/epilogue lost below `S = 2048` (gfx942 FA-fwd), where the non-pipelined variant was faster.
On gfx950 / Triton 3.8.0 the one FA-forward body measured differed on depth (`ns=4` best, no cliff) and
matched on length (crossover near `S = 2048`): these depths and thresholds are per arch and per body (`../../workloads/flash-attention-structural-insights.md ### Software-pipeline reproduction — the kernel shape re-injection needs`).

### And on attention — two dots chained through a softmax

A minimal FA-forward body pipelines like a GEMM (prologue peeled, K/V staging built by the pass, numerics
unchanged), and a gfx942 attention kernel reached near-parity with plain. **A minimal body does not settle
attention:** on a sparse-paged attention kernel the injection fired (op census = plain's) and was a **large
regression**, because the V staging landed on `swizzled_shared<vec=1, perPhase=1, maxPhase=1>` where plain gets
`amd_rotating_shared<vec=4, perPhase=4, maxPhase=4>` (inferred: re-check with the `in_thread_transpose` tail,
which is what emits the rotating layout). On a tuned attention champion the pipeline's own share was small
and **changed sign across sequence lengths**; only `plain@ns=1` at the champion's tile separates it from a
tile change. **`convert_layout` scratch allocated after the pipeliner is outside hazard analysis** — a layout
conversion left in the loop is a correctness question first.

### What to expect, and what to measure yourself

Measured outcomes (gfx942, 3.6.0–3.8.0, numerics unchanged everywhere): it **fires** on dot-free reductions,
GEMMs and a minimal FA-fwd on every version; on two kernels with a real `plain@ns=1` debt it **closed the gap**
(reached or slightly exceeded shipped plain); on one de-staged dot kernel the injection itself was a consistent
speedup over un-staged-OFF, but de-staged + injected vs the hand-staged anchor was **net negative on 3 of 4
minors**. Magnitudes do not transfer across kernel, shape or Triton version; carry only what you measure.

### The ping-pong window (gfx942)

`add_block_pingpong` fired under injection only at 256×256×64, `num_warps=8`, `ns=2` (8 `s_setprio`);
`num_warps=4`, or 128×128×64 at 8 warps → 0. It did not fire on hand staging in the cases tried, as its
source predicts (it collects `local_load`s sourced from a loop-carried `BlockArgument`; hand staging comes
from `memdesc_index`).
`is_pingpong_schedule_enabled` is consulted only inside `make_ttgir`, so satisfying it from Gluon buys
nothing. `warp_pipeline_stage` is a different mechanism (`marker-and-version-gates.md`).

### `patch_async_reinject.py` belongs to the authored async ring, not here

`apply|revert|status`, armed by `TRITON_GLUON_ASYNC=1`, splices `add_coalesce_async_copy` into
`gluon_to_ttgir`. Native-width async copy (gfx950 128 or 32 bit per lane, gfx942 32 bit, one access per lane)
lowers without it; it only repairs a non-native pattern (64-bit, or a per-lane access split across
repetitions, e.g. a `BlockedLayout` covering `[64, 16]` on a `[32, 32]` tile) by adding a bounce (a
`convert_layout` that clips the per-lane width; the data still lands direct-to-LDS, the address tensor makes
an LDS round trip: `../../pitfalls/platform-known-issues.md`); a layout
that lands on a native width needs no bounce (`../memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`).
