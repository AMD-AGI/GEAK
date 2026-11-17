# FlyDSL idioms and kernel pitfalls for a persistent layer kernel

How to express the levers of this skill in FlyDSL, the gfx942 resource facts behind them, and the kernel
robustness issues that a stage's own checks do not see. Pitfalls of the serving-engine integration and of
serving benchmarks are in
[apply_mono_kernel_to_framework](../../apply_mono_kernel_to_framework/skill.md#knobs--pitfalls).

## FlyDSL idioms

API-level idioms (FlyDSL 0.4.x, gfx942). Check names against the installed FlyDSL and aiter.

| need | idiom |
|---|---|
| `if` / `while` / `for` on traced values inside a helper | decorate the helper with `ASTRewriter.transform`; a `@flyc.kernel` rewrites only its own body |
| trace-time loop / branch | `range_constexpr(n)`, `const_expr(cond)` |
| loads / stores with a cache policy | aiter's `buffer_ops` on a resource built from a raw Int64 address; the cache-policy bits select device scope (SC1), system scope (peer over XGMI) or non-temporal (weights read once) |
| > 4 GiB offsets (KV cache) | `llvm.LoadOp` on an `inttoptr` address: buffer offsets are 32 bits |
| wave-uniform value in an SGPR | `rocdl.readfirstlane` (both halves of an i64): keeps buffer descriptors scalar, no waterfall loops |
| issue every load before the math | loads into a list, `rocdl.sched_barrier(0)`, then the MFMA chain in the same order (same bits) |
| keep LLVM from reordering FMAs into the next round / spilling | an opaque `v_mov_b32` via `llvm.inline_asm` on the accumulator |
| wave reductions | DPP `update_dpp_i32` for xor 1-8, `shuffle_xor` for 16 / 32 |
| MFMA | `rocdl.mfma_f32_16x16x32_fp8_fp8` (lane = 8 FNUZ bytes as i64), `mfma_f32_16x16x16bf16_1k` |
| fp8 / bf16 conversion | `rocdl.cvt_pk_fp8_f32` (saturation off: clamp first), `cvt_pk_f32_fp8`, `perm_b32` for exact bf16 pairs |
| LDS | `@fx.union` / `@fx.struct` of `fx.Array`, `fx.SharedAllocator().allocate(...)`; overlay the areas of stages that are never live together in one union |
| per-stage timing | a `timeline` build flag: thread 0 stores `s_memrealtime` at numbered points per CTA ([measuring.md](measuring.md#per-stage-timeline)) |
| rebuild on any source edit | put a digest of the package sources in the kernel name (FlyDSL's JIT cache key) |
| compile without a GPU | `HIP/CUDA/ROCR_VISIBLE_DEVICES=-1 COMPILE_ONLY=1 ARCH=gfx942` (`flyc.compile` otherwise also launches) |

## Resource facts (gfx942)

- 512 threads per CTA (two waves per SIMD): <= 256 VGPRs per lane. Extra state held across a loop (a
  second K/V step in registers, deep prefetch) is where spills start; check the ISA after every such
  change.
- LDS: 64 KiB per CTA.
- LDS DMA (global -> LDS) can make LLVM insert `s_waitcnt vmcnt(0)` before the next `ds_read` (no alias
  info), which serializes the step's loads; read the ISA before relying on it.

## Kernel pitfalls

- **LDS races after a stage.** A missing barrier at the end of a stage that reuses LDS (or between staging
  a KV block and the current-token write) shows only now and then. Stress the unittest with many
  bit-exact repeats.
- **Spills turn wins into losses.** Every lever that holds more state (prefetch, shared weight reads,
  next-round loads) can hit the VGPR cap. Check scratch in the ISA.
- **LDS alignment.** Arrays not 16-byte aligned cost wide LDS accesses; align them.
- **Uncached peer / flag accesses.** Peer buffers must be uncached; waits that clear flags need
  system-scope ACQUIRE.
- **Tag wrap**: see [tagged hand-offs](handoffs_and_sync.md#tagged-value-tag-hand-offs).
