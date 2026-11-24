# Triage — compile, lowering, correctness and runtime failures (Gluon, gfx950)

Symptom → fix for Gluon on Triton 3.8.0 / gfx950 (gfx942 downgrade notes). Authoring rules live in
`../gluon/layout-reference.md`, `../gluon/memory-reference.md`, `../gluon/matrix-reference.md`;
known platform bugs in `../pitfalls/platform-known-issues.md`; silently-not-lowering expressions in
`../pitfalls/negative-patterns.md`. A correct-but-slow candidate is a `profile.md` question.

Suggested first response: roll back the newest structural layout/memory/matrix change before touching launch
parameters; reduce to one executable subpath; add shared memory / async only after the simple path is
correct. Before calling a feature unavailable, run the cheapest falsifying probe (`hasattr(module, op)`,
a one-line compile of the op, an alternative namespace such as `cdna4.async_copy` when `cdna3` lacks it)
and keep its exact error.

## Symptom -> inspect -> fix

Compile / lowering / layout time. Runtime failures after a clean compile: `## Async-handoff debug
worksheet (runtime)`.

| Symptom | Inspect first | Fix |
| --- | --- | --- |
| `arange()` missing `layout` | leftover `tl.arange` / layout-less `gl.arange` | explicit layout-aware `gl.arange` |
| `cannot convert SliceLayout(...) to tensor` | a layout object assigned to an ordinary (un-annotated) local inside `@gluon.jit` | build layouts on the host and pass them as `gl.constexpr` arguments, or bind them in the kernel to a `gl.constexpr` name (both compile on 3.8.0 / gfx950) |
| `expected expand_dims input to have a SliceLayout` | slice parent + expand axis | regenerate the index from `SliceLayout(axis, parent)` |
| `Layout mismatch in broadcast` / broadcast dim mismatch | parent layout + slice lineage | rebuild both sides from one parent layout |
| `expected broadcast left input to be a distributed_type` after `tl.dot` | mixed plain/Gluon boundary | do not feed `tl.dot` results to Gluon ops; lower the dot through MFMA |
| buffer reads wrong addresses | offset unit | **buffer offsets are ELEMENT counts, not bytes**: drop any `* bytes_per_element` (else 2x off on 2-byte, 4x on 4-byte dtypes) |
| buffer fallback has no dtype | `other` value | typed `gl.full(..., ptr.dtype.element_ty, layout=...)` |
| buffer offset verifier failure | offset dtype/layout | distributed int32/uint32 offsets |
| `buffer_load_to_shared` / `global_load_to_shared` fails: `failed to translate module to LLVM IR` (`LLVM Translation failed for operation: builtin.unrealized_conversion_cast`) or `PassManager::run failed` (`failed to legalize operation 'ttg.async_copy_global_to_local'`) | per-lane width, then warp-to-row cover, then shared-layout family | **per-lane width must be exactly 128 or 32 bit on gfx950, 32 bit only on gfx942; 64 bit fails on both** (`../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`). Offset layout must exactly cover the contiguous dim (Gluon runs no `CoalesceAsyncCopy`). Padded destination: `interval_bytes >= 64 lanes × per-lane bytes` and producer derived from its `offset_bases` (`../gluon/memory-reference.md ## Deriving the producer layout for a padded async destination`; `../tile-programming/compiler-contract.md ## Async / buffer path failures are layout-first`). gfx942: swizzled destination first, or sync staging |
| other `builtin.unrealized_conversion_cast` | newest index/layout rewrite | revert it and shrink the subpath |
| `convert_layout` input type failure | input not distributed | convert only distributed tensors |
| `BlockedLayout size_per_thread` verifier failure | tile, `threads_per_warp`, `warps_per_cta` | recompute for wave64 |
| MFMA layout verifier failure | `instr_shape`, version | `version=4` gfx950, `version=3` gfx942 |
| **CDNA4 INT8 MFMA rejects an fp32 accumulator** | acc dtype | int32 accumulator for `cdna4.mfma`, or keep plain `tl.dot` |
| **`scaled_upcast` rejects `bf16` although the message expects bf16** | string vs dtype object | pass `tl.bfloat16`, not `"bf16"` |
| `kpack is deprecated starting from gfx950` | stale compile option | `kpack=1` or remove it |
| `sched_barrier` / scheduling helper missing | installed API | version/API blocker for that spelling (absent on 3.8.0); other scheduling routes remain (`../tile-programming/instruction-scheduling.md`) |
| JIT boundary mismatch | decorator | `@gluon.jit` for Gluon helpers; `@triton.jit` only for separate Triton kernels |
| scalar return `int32` vs `int64` after a Triton update | wrapper/oracle dtype | normalize at the wrapper boundary |
| AOT scratch failure | metadata + scratch policy | keep the fallback; compile success ≠ runtime success |
| preshuffled GEMM wrong | `DistributedLinearLayout`, unshuffle order, K divisibility | preserve reshape/permute/trans order |
| **multi-region kernel wrong on some shapes only (large abs / cosine ~1.0)** | one combined capability flag + a dummy pointer for a region without the feature | per-region flags; give every region a valid pointer (data-dependent: passes small shapes) |
| `ttgir_bridge.py recover` reports `UNRECOVERABLE` | which layout; dumped with pipeliner active? | `transcribe.md ## 2. Recover` (probe build → re-dump ns=1 → `convert_layout` substitute); a hand-written basis cannot be verified |

**Read the failure magnitude before blaming numerics**: near-1.0 cosine diff or large abs error is a
bug, not reduction-order noise. Deterministic but shape-dependent = index/ownership bug; varies
run-to-run on one input = race (`benchmark-hygiene.md ## Determinism race-test (async / barrier /
pipeline / layout changes)`).

## Async-handoff debug worksheet (runtime)

For a deadlock / launch timeout, a crash / poisoned context, or wrong values after a clean compile.
Suggested before editing: write per async op **Roles** (which lanes/waves issue it), **Storage** (where each tile
lives each step), **Handoff** (producer, consumer, wait count, the drain that makes the write visible),
**Lifetime** (earliest reuse of each slot). Then dump IR/asm and check it against that table.

| Symptom | Likely area | First check |
| --- | --- | --- |
| hang / launch timeout | deadlock | wait counts vs issued groups; barrier reached by all waves |
| illegal access, later calls also fail | crash | restart the process first; OOB offsets, wrong lane set |
| wrong values in row / tile stripes | race or tile ownership | deterministic vs run-to-run |
| `NaN` | layout/swizzle, uninitialized accumulator | accumulator init |
| finite, patterned-wrong | stale data | missing drain, slot reused too early |

Change one handoff at a time; re-run correctness and the determinism test after each.

### gfx950 / Gluon fill

Dump with `kernel_workflow/scripts/kernel_tools/dump_ir.sh`; audit with `asm_loop_audit.py`. Grep the
asm for `buffer_load ... lds` / `global_load_lds_*` (direct-to-LDS issued), `ds_read` / `ds_write`,
`s_waitcnt` / `s_barrier`, `v_mfma*`, `sched_barrier`.

Authored ring (`async_copy` + `commit_group` / `wait_group`; gfx942: sync staging) — check:
- **`wait_group(N)` vs outstanding `commit_group`s**: N = groups allowed still in flight; recompute it
  when the prologue depth or unroll changes. `wait_group` lowers to `llvm.amdgcn.wait.asyncmark(N)` and
  the compiler emits an `s_barrier` after it (`../tile-programming/pipeline.md ### What the compiler
  inserts after wait_group: barriers and load_shared_relaxed`).
- **Buffer count**: an authored ring needs one slot per tile live at once — the copies still
  outstanding after `wait_group(N)` plus the slot being read — or a refill overwrites a slot being
  consumed (inferred). The plain pipeliner's count does not transfer: plain at `num_stages=S` allocates
  `S-1` buffers (`memdesc<(S-1)x…>`; `ns=2` is single-buffered with a peeled prologue) because its
  staging differs.
- **A runtime `smem.index(k % nBuffers)` is legal and is not a deadlock cause**; unrolling to literal
  indices pays when the ISA shows it (`../tile-programming/pipeline.md ### Hand-built buffering
  rules (correctness + scheduling footguns)`).
- `DOT(k)` consuming a slot before the `wait_group` that retires its copy.
- `sched_barrier` / barrier inside a branch only some waves take.
- **A non-coalesced async-copy offset layout fails to lower on stock 3.8.0** (`unrealized_conversion_cast`
  / `failed to legalize ... async_copy_global_to_local`) — compile-time, see the table above; it is not a
  silent runtime fallback. If the asm shows `ds_write` but no direct-to-LDS load on the load path, the
  code is not using the async copy you think (e.g. a `.store()` path), so `wait_group` does not cover it.

A layouts-only anchor (no ring yet) that "hangs" is usually a compile/lowering issue — check the table.
A build with a re-injected pipeliner is a diagnostic whose sync the pipeliner owns
(`../gluon/pipeline/reinjection.md`), not a debugging target. Restart the HIP process after an illegal
access before testing the next fix. Asm unchanged after an edit = the expression did not lower as
intended (`../pitfalls/negative-patterns.md`).
