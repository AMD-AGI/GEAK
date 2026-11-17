# FlyDSL idioms, resource limits and pitfalls for a decode operator

API-level facts collected while writing ten decode operators on gfx942 with FlyDSL 0.4.x. **Check every
name against the FlyDSL and aiter you have installed**: the source tree that was read (0.5.0) and the
installed package (0.4.0) differed in about half of their common files, and an idiom copied from the
newer tree did not always exist in the older one. Import-check an API before designing around it.

General FlyDSL authoring is covered by the public cards and is not repeated here:
[tile programming](../../../../languages/flydsl/authoring_tile_programming.md),
[optimization workflow and the register-allocation gate](../../../../languages/flydsl/authoring_optimization.md),
[debugging](../../../../languages/flydsl/debugging.md),
[attention levers](../../../../languages/flydsl/authoring_attention_levers.md).

## Idioms

| need | idiom |
|---|---|
| `if` / `while` / `for` on traced values inside a helper function | decorate the helper with the AST rewriter (`ASTRewriter.transform`); a `@flyc.kernel` rewrites only its own body |
| trace-time loop or branch | `range_constexpr(n)`, `const_expr(cond)`; a plain Python `if` on a build parameter removes the other branch from the kernel entirely |
| loads / stores with a cache policy | aiter's `buffer_ops` on a resource built from a raw 64-bit address; the cache-policy bits select device scope (SC1), system scope, or non-temporal (weights read once) |
| offsets past 4 GiB (a large KV cache, a big request-to-token table) | a raw-pointer load (`llvm.LoadOp` on an `inttoptr` address) with the offset formed in 64 bits: buffer offsets are 32 bits |
| a wave-uniform value kept scalar | `rocdl.readfirstlane` (both halves of a 64-bit value): keeps buffer descriptors scalar, no waterfall loops |
| issue every load of a step before the math | collect the loads in a list, `rocdl.sched_barrier(0)`, then the arithmetic in the same order |
| stop LLVM from moving FMAs into the next round, or spilling an accumulator | an opaque register move (`v_mov_b32` through `llvm.inline_asm`) on the accumulator |
| wave reductions | DPP (`update_dpp_i32`) for xor distances 1 to 8, a shuffle (`shuffle_xor`, `ds_bpermute`) for 16 and 32 |
| matrix cores | `rocdl.mfma_f32_16x16x32_fp8_fp8` (a lane holds 8 FNUZ bytes as one i64), `rocdl.mfma_f32_16x16x16bf16_1k` |
| fp8 / bf16 conversion | `rocdl.cvt_pk_fp8_f32` (it does **not** saturate: clamp first), `cvt_pk_f32_fp8`, `perm_b32` for exact bf16 pairs |
| LDS | a struct / union of arrays allocated from the shared allocator; overlay the areas of phases that are never live together in one union; keep arrays 16-byte aligned |
| per-phase timing | a separate build that stores the device clock at numbered points from thread 0 |
| rebuild on any source edit | put a digest of the kernel's source files in the launcher name |
| compile without a GPU | hide the devices and set the compile-only switch and the target arch in the environment; a plain compile call otherwise also launches |

## Resource limits on gfx942

Hardware constants are from the public cards: [architecture](../../../../hardware/cdna3_mi300/arch.md),
[occupancy](../../../../hardware/cdna3_mi300/occupancy.md),
[memory hierarchy](../../../../hardware/cdna3_mi300/memory_hierarchy.md).

- **304 CUs**, 64 lanes per wave, 4 SIMDs per CU. One decode row has to be spread over tasks to use them.
- **Registers.** A SIMD has 512 vector registers shared by its resident waves. A workgroup of **512
  threads** is 8 waves, two per SIMD, so each wave gets **at most 256**. At 256 threads or fewer (one
  wave per SIMD) a kernel may report up to 512, but everything past 256 lives in the accumulator file
  and is reached through register moves. The counts quoted in the per-op references are the ISA's
  next-free-register figure, which includes those accumulator registers.
- **LDS: 64 KiB per workgroup.** A design that uses more than half of it runs one workgroup per CU.
- **L2 is per XCD.** A plain load in one workgroup can miss a store made by a workgroup on another
  XCD in the same launch. Cross-workgroup data needs coherent accesses (below).
- **Extra state held across a loop** (a second K/V step in registers, a deeper prefetch, shared weight
  reads) is where spills start. Read the ISA after every such change; a spill turns a win into a loss.

Measured register counts, for orientation (256-thread workgroups unless noted):

| operator | registers | LDS |
|---|---|---|
| GEMV / linear | 77 to 112 (up to 256 with 16-element steps and an activation epilogue) | 0 |
| RMSNorm | 25 to 77 | 0 |
| RoPE / KV write | 22 to 67 | 0 |
| decode attention | 172 to 339; 114 to 237 in the 512-thread form | 1 to 33 KB |
| MLA decode | 262 to 376; 254 in the one path that fits 512 threads | about 35 to 52 KB |
| MoE experts | 110 to 191 (4-bit weights), 312 (fp8 with activation quant), 166 (bf16) | 2 x hidden + small |
| router top-k | 70 to 178 (one shape 272) | 2 x hidden + small |
| sparse indexer | 109 to 208 | about 41 KB |

## Compiler pitfalls seen

1. **An LDS flag under a divergent branch was miscompiled.** A verdict ("this task is the last to
   arrive") was written to LDS by lane 0 inside a divergent `if` and read back by the whole wave; the
   other lanes read before lane 0's store. Carry such a verdict as SSA values (the result of the `if`,
   then a `readlane`), never through an LDS flag.
2. **Rows unrolled at build time can exhaust the registers.** A stream-mixing operator unrolled over
   rows needed 466 registers at 3 rows; with the rows as runtime loops the same shape needs 244 and the
   count no longer grows with the width. Small operators (norm, RoPE, delta rule) unroll rows without
   trouble; check the count at the largest width, not at width 1.
3. **Unrolling a K step on top of a wider step spills.** In the GEMV, unrolling with hoisted loads
   helped the 8-element step by about 14 %, but combined with the 16-element step it spilled into the
   accumulator file and lost.
4. **Global-to-LDS copies can serialise a step.** Without alias information LLVM may insert a wait for
   all outstanding vector-memory operations before the next LDS read. Read the ISA before relying on a
   direct global-to-LDS load to overlap with anything.
5. **Loop-carried values come back wrapped.** A Python `for` / `if` in a kernel body becomes `scf.for` /
   `scf.if`, and values carried through it come back as DSL wrapper objects without an MLIR type. Unwrap
   them before handing them to raw builders.
6. **A missing barrier after a phase that reuses LDS** fails only now and then. Stress the unit test
   with many bit-exact repeats after any change to LDS use.
7. **Above 256 threads a structural change is needed, not a tweak.** The attention design switches to
   half the positions per wave per round and loads V only after the scores are consumed; the MLA design
   refuses two of its three operand paths there because they were measured to spill.
8. **A workgroup barrier needs every thread.** Keep the staging branch-free (clamp an index instead of
   guarding it); see the barrier-deadlock section of the public debugging card.

## JIT cache and trace-time pitfalls

- **The cache key is the launcher's source plus its closure scalars.** Helper modules and module-level
  globals are not part of it. Editing a helper, or changing a module-level parameter between two builds
  in one process, can silently run the previously compiled kernel.
- Therefore: one launcher name per configuration (width, workgroup count, each variant switch), and a
  source digest in the name while developing. A profiling build gets its own name.
- **Module globals are read at trace time.** A kernel body that reads `WIDTH`, a thread count or a
  format switch from module scope bakes in the value seen at the first trace. Two widths built in one
  process must not share a launcher name, and a parameter must not be mutated between traces while the
  name stays the same (*derived* from the cache-key rule above; treat it as a rule).
- If a fix looks ineffective, first re-run once with the disk cache bypassed (the public debugging card
  gives the switch), then suspect the code.
- The first launch of each configuration compiles. Keep it outside the timed region and outside graph
  capture.

## Cross-workgroup patterns

Three patterns covered every case in the ten operators. All of them are correct for any workgroup
count and need no grid-wide barrier.

1. **Exclusive ownership, no communication.** Each task owns its outputs and derives everything else
   itself, even redundantly (the RMSNorm statistic per wave, the delta rule's `q` / `k` normalisation
   per split). Cheapest to write, and the right first version.
2. **Last arrival finishes.** The tasks of one row (or head) write their partial results into a small
   device workspace with coherent stores, drain them, then add 1 to an arrival counter with a
   device-scope atomic. The task whose add returns `n - 1` is the finisher: it reads all partials, sums
   them **in a fixed index order** (so the result does not depend on who arrived last), writes the
   output, and re-zeroes counter and accumulators. Nobody waits. Used by the router, the MoE experts,
   the K-split GEMV and the delta rule.
3. **Owner waits on lower indices.** Non-owner tasks store a record and post an epoch-tagged flag; the
   owner is the **highest** task index of its set, so with a grid-stride loop every task it waits for
   has already been scheduled. Used by split-K attention, MLA decode, the indexer and the stream mix.

Rules that go with patterns 2 and 3:

- the workspace is zero before the first call and every call leaves it zero again (or tags flags with a
  launch epoch so nothing needs clearing);
- one workspace per (device, width); calls that share it must not overlap, which a single stream
  guarantees;
- an aborted launch can leave a counter non-zero, after which that row never finishes until the
  workspace is zeroed again;
- sums of partials use a fixed order. An atomic float add is acceptable only where run-to-run
  bit-reproducibility is not required (the MoE accumulators use it and are *not* bit-reproducible);
- a value-only test cannot see a protocol bug: a variant in which every arrival finishes passes every
  numeric check and still releases its output early and repeatedly. Reason about "each output unit
  written exactly once for every arrival order" separately.

## What to check before handing in

- ISA: register count under the cap for the thread count, scratch 0, LDS under 64 KiB.
- Correctness at 1, 7 and the full number of workgroups; then repeated runs for bit-exactness where
  the design promises it.
- The launcher is graph-capturable, and its first call ran eagerly.
- A changed parameter really produced a new kernel (the name changed, or the ISA differs).
