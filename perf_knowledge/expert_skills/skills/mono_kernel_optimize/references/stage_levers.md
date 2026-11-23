# Stage levers: weights, attention, MoE, and what a schedule change may not alter

Open this file when a stage sits far above its byte floor ([measuring.md](measuring.md)). In the measured
FlyDSL reference the expert stages held 57% of the layer (gate/up 38%, down + final sum 19%), attention 7%.

## 1. Stream the weights at memory speed

**When.** A GEMV stage (QKV, o_proj, gate/up, down, expert GEMMs) is well above its floor. At decode
widths every weight byte is read once per step for a handful of FLOPs: memory-bound, not compute-bound.

**Recipe.**
1. Wide, coalesced loads: 16 bytes per lane, consecutive lanes on consecutive addresses along K; a
   workgroup streams contiguous rows. Read fp8 weights as fp8; never upcast in memory.
2. Dequant in registers: for block-scaled fp8 (128×128 blocks) load one scale per block per row tile and
   multiply after the fp32 accumulation of the block, not per element.
3. Enough bytes in flight: with one workgroup per CU, latency must be hidden inside the workgroup —
   several outstanding loads per lane (unroll K) or double buffering through LDS. Too few rows per
   workgroup: split K across workgroups and reduce in the same fixed order on every rank.
4. Activation reuse: the input vector (width × hidden) is small; keep it in LDS or registers for the
   whole stage instead of re-reading it per row.
5. Prefetch by idle workgroups: workgroups idle during attention or a wait can touch the next stage's
   first tiles so they land in the per-XCD L2 just before use — only when the timeline shows idle CUs.
6. Block scales: they are small but can cost more requests than the weights when the framework's layout
   scatters one wave's lanes over many cache lines. Check the addresses one wave touches; either keep a
   copy of the scales in the kernel's read order (made once, keyed by the weight's data pointer, its
   bytes stated) or issue the scales before the weights.
7. Generic FlyDSL levers: [authoring_gemm_levers.md](../../../../languages/flydsl/authoring_gemm_levers.md)
   (tiling, prefetch, epilogue), [patterns.md](../../../../languages/flydsl/patterns.md) (split-K for
   skinny shapes, the small-M family, fp8 preshuffle) and the existing skill `flydsl_fp8_blockscale_gemm`
   (fp8 block-scale parity and tiling; validated on gfx950, re-measure on gfx942).

**Already applied if** loads are 16 bytes per lane and issued before the math (a load helper, then an
MFMA helper), fp8 bytes feed fp8 MFMA, the scale multiply follows the block sum, the activation is in LDS.

**Pitfalls.** A weight or scale repack is out under a stated memory limit unless it replaces the
original in place or the task statement allows it; otherwise once, keyed by the data pointer, bytes stated; never per call. MFMA is not
the bottleneck at width 1: do not trade load width for an MFMA-friendly layout unless the timeline says
compute. Changing the K split changes the summation order (within tolerance; never the all-reduce
order). **Check.** Stage span against its floor; the oracle at every width.

**Evidence (measured on the expert gate/up stage, the largest of the layer).** Activation hoisted to LDS
and the next task's weights prefetched: −19% of the stage. Then an XCD-aware tile map, cached block
scales, lane redistribution through an LDS permute, a depth-2 loop and a one-wave epilogue: a further
−23%. Down weights coalesced with an LDS transpose (flags polled first): −15% of the down stage, partly
lost again by the end. Cacheable loads for 1-row expert weights: no gain (reverted).

## 2. Split-KV decode attention with an in-kernel combine

**When.** Attention dominates the long-context cases, or its span grows with context while few
workgroups work on it. At width 1 there are only a few query heads per rank, far fewer than the CUs.

**Recipe.**
1. Split the KV sequence of each (KV head, sequence) into chunks, one per workgroup, enough to cover the
   attention share of the grid. Each workgroup computes over its chunk the running max m, the sum l and
   the unnormalized output acc for every query head of the group (flash-decoding).
2. Combine in the kernel: each split writes (m, l, acc) to scratch and publishes; one workgroup per head
   (or the consumer of the attention output) waits on the head's splits and merges them in split order:
   m* = max mᵢ, l* = Σ lᵢ·e^(mᵢ−m*), out = Σ accᵢ·e^(mᵢ−m*) / l*.
3. Read KV in the framework's layout: paged cache through the block table; fp8 KV with its scales as
   the native path applies them; dequant in registers.
4. Sliding window and sinks: only the last `window` tokens count, so fewer chunks are needed; sinks
   enter the softmax where the baseline applies them (once per head, not per split).
5. Generic levers: [authoring_attention_levers.md](../../../../languages/flydsl/authoring_attention_levers.md).

**Already applied if** attention tasks are per (request, key chunk) and a combine step runs per
(token, head).

**Pitfalls.** The chunk count depends on the context, which changes every step: derive it on the device
from the sequence lengths (graph replay keeps arguments fixed) and size scratch for the longest case.
Merge in a fixed split order; arrival order is nondeterministic. The KV write of the current token must
be visible before any split reads the last block (same workgroup, or behind a hand-off). Empty chunks
must still publish, with l = 0. **Check.** The oracle at every context bucket, graph replay, sequences
across steps; the timeline at the longest context.

**Evidence (measured, global-attention stage at the longest context).** Block table in LDS + K/V loaded
one step ahead + combine loads first: −16% of the stage. Dead ends: exact-fill split (one chunk per CU)
— no gain, and the output drifted (relative mean error about 0.4% where every kept change was
bit-identical); combine split 4× over more workgroups — the next stage 2.3× slower; idle workgroups
helping the KV loop — slower, the loop was already memory-bound.

## 3. Wave-local attention, K straight from the cache

**When.** Attention dominates, or its span grows faster with context than its KV bytes do (barriers and
LDS round trips per key block).

**Recipe.**
1. Each wave runs its own online softmax on its key block: Sᵀ = K·Qᵀ on MFMA with K loaded from the KV
   cache directly as the MFMA operand — no LDS staging of K, no workgroup barrier inside the key loop.
   P stays in registers as the operand of PV; the waves merge (m, l, acc) once at the end.
2. Vᵀ through LDS with a swizzle (a row-dependent XOR per small token unit removes the bank conflicts of
   the transposed write); a wave-private LDS slice for Vᵀ avoids workgroup barriers.
3. Quantize q cooperatively: for FP8 KV the whole workgroup writes the FP8 q into LDS once and reads it
   back in MFMA layout, instead of every lane quantizing its own copy.
4. Long context: split-KV over many workgroups plus a combine (section 2). Size the key blocks so the
   split covers the free workgroups; spread the combine over (head, dim segment) so it is not a
   few-workgroup tail. If the native FP8 decode rounds P against a running max per segment, keep the
   same segments (or exchange the in-segment max between workgroups that share one).
5. Long tasks read K once: when a task's scores exceed the LDS, write them to a per-workgroup scratch in
   memory during a pre-scan and let later passes read the scores back instead of K.
6. Write KV after the attention loop (its stores would turn the K/V waits into full memory waits). RoPE
   and the KV write can live in the attention workgroups.
7. Sliding window: one workgroup per KV head with the sinks in the same softmax; align the window start
   down to a tile and treat tokens outside the window as zero without reading them.

**Already applied if** there is no barrier inside the key loop, K loads feed MFMA directly, and the KV
store comes after the loop.

**Pitfalls.** Narrow V loads shaped to avoid the transpose can be slower than the swizzled transpose.
Every added buffer (score scratch, double-buffered K/V) costs VGPRs or LDS: check the ISA for scratch.
The current token's K/V must go through the same FP8 quantize–dequantize as the cache. **Check.**
Attention span against its KV byte floor at every context bucket; a pure schedule change must be
bit-identical to the previous version.

**Evidence (measured; serving P90 unless stated).** Cooperative q quant + KV write after the loop: −5%.
K straight from memory per wave + Vᵀ swizzle: neutral. Long tasks reading K once: −15%. Wave-local
global attention without a start barrier, together with FP4 multi-accumulators: P90 flat, median −10%.
Wave-local sliding-window attention (with LDS alignment and scales before weights): −1%. FlyDSL kernel,
layer time: Q staged in LDS without spills + batched chunk combine −5.7% (the only keeper of five
authoring rounds); software-pipelined K/V loads — slower; two dim tiles per merge round — slower; a
register double buffer for K/V — 19 to 81 spilled registers.

## 4. MoE decode: router, top-k, FP4 experts

For MoE layers with block-scaled FP4 experts. The routing rules (scoring function, bias, grouping,
renormalization) come from the framework's own source. **When:** router, top-k or expert stages each
hold a visible share of the timeline, or the experts stay well above their weight-byte floor.

**Recipe.**
1. Router logits across workgroups, selection in every workgroup that needs it: a few workgroups compute
   and publish the logits; each consumer reads all of them, applies the native scoring, picks top-k and
   normalizes, so there is no second hand-off. Keep the readers few when they read the same few KB: in
   the FlyDSL kernel selection in every workgroup was 2.2× slower than 8 readers per token (*measured*),
   although both HIP designs select in every workgroup.
2. Top-k at wave level: each lane sorts its few candidates, then k rounds compare only the lanes' heads (cross-lane
   max, then a ballot); tie side and id order exactly as the native router kernel has them (`native_decode_numerics`, tie table).
3. FP4 decode in registers: a byte permute builds FP16 (or FP8) bytes directly from the 4-bit codes,
   since every code is exact in those formats — no LDS lookup table.
4. Scales in read order (section 1, item 6).
5. Several independent accumulators for the FP4 dot products, so the FMA chain is not latency-bound.
6. Rounding points as native: fold the routing weight where the native kernel does and round each
   expert's result where it does (often BF16 before the sum).
7. Cacheable loads for expert weights when several rows hit the same expert; for one row it is moot.

**Already applied if** a handful of reader workgroups per token (one per XCD) do the top-k with
cross-lane operations, FP4 is decoded by a permute, and a permuted scale copy exists.

**Pitfalls.** MFMA accumulation order differs from the native MoE kernel: tolerance, not bit-identity.
Near-tied routing scores flip experts: compare with the routes native actually executed, not a
recomputed top-k. A scale copy costs KV-cache memory: under a stated memory limit, only if the task statement allows it. Multi-row expert
paths reuse LDS between rows: a missing barrier at the end is a real race. **Check.** Expert span
against its bytes (k experts × gate / up / down at FP4, plus scales); a case with near-tied routing
scores.

**Evidence (measured).** 8 readers per token (one per XCD) + wave-level top-k: −45% of the top-k stage;
moving top-k into every workgroup: 2.2× slower. FP4 permute decode (with shared QKV / o_proj rows and
more row counts): −14% P90. Per-lane sorted candidates alone: no gain (P90 about 5% worse that round,
unexplained). Expert tasks in rounds of six: slower. Ranked but not done: pipelining gate/up and down
per expert (estimated 13% of the layer).

## 5. What a schedule change may and may not alter

| change | effect on the output | rule |
|---|---|---|
| pure placement, schedule or prefetch | none | must be bit-identical to the previous version: check it |
| K split or accumulation order inside a GEMV / MoE | about 1 ulp now and then | within tolerance; say that bit-identity is gone |
| order of the RMSNorm sum of squares | can flip the FP8 code of the next input | keep the baseline's reduction tree |
| where a value is rounded (sum + residual to BF16; FP8 quant from the FP32 normed value; per-expert rounding before the sum) | drift downstream | keep the baseline's rounding points |
| softmax form (exponent base, sinks, P rounded against a running max per segment) | P rounds differently | keep the baseline's form and segments |
| causal limit / window from sequence lengths, RoPE from positions | wrong for multi-token steps if mixed | keep the two index sources apart |
| all-reduce order | ranks stop being bit-identical | never change it (rank order, fp32) |

For norm / quant changes also count FP8 code flips against the baseline on the captured inputs (expect
none). What then remains — accumulation orders about 1 ulp off the native library, forking greedy outputs
only at near-ties where native forks against itself — is not worth chasing. See `native_decode_numerics`.

## 6. Expressing a lever in FlyDSL

Every name below exists in public FlyDSL at the commit in the skill's Sources (*names verified, behaviour
read from source only*): check against the installed version. In HIP C++ read the kernel's own helpers.

| need | idiom |
|---|---|
| `if` / `while` / `for` on traced values in a helper | decorate the helper with `ASTRewriter.transform`; a `@flyc.kernel` rewrites only its own body |
| trace-time loop / branch | `range_constexpr(n)`, `const_expr(cond)` |
| loads / stores with a cache policy | `buffer_ops` on a resource built from a raw 64-bit address; the policy bits select device scope, system scope (peers) or non-temporal (weights read once) |
| offsets beyond 4 GiB (KV cache) | `llvm.LoadOp` on an `inttoptr` address: buffer offsets are 32 bits |
| wave-uniform value in a scalar register | `rocdl.readfirstlane` on both halves of a 64-bit value: buffer descriptors stay scalar, no waterfall loops |
| issue every load before the math | loads into a list, `rocdl.sched_barrier(0)`, then the MFMA chain in the same order |
| keep the compiler from reordering FMAs or spilling | an opaque register move through `inline_asm` on the accumulator |
| wave reductions | `update_dpp_i32` for XOR distances 1–8, `shuffle_xor` for 16 / 32 |
| MFMA | `rocdl.mfma_f32_16x16x32_fp8_fp8` (a lane holds 8 fp8 bytes as one 64-bit value), `mfma_f32_16x16x16bf16_1k` |
| fp8 / bf16 conversion | `rocdl.cvt_pk_fp8_f32` (saturation is off: clamp first), `cvt_pk_f32_fp8`; `perm_b32` for exact bf16 pairs |
| LDS | `@fx.union` / `@fx.struct` of arrays from `fx.SharedAllocator`; overlay the areas of stages that are never live together |
| per-stage timing | a timeline build: thread 0 stores the realtime counter at numbered points ([measuring.md](measuring.md)) |
| rebuild on any source edit | a digest of the package sources in the kernel name (the JIT cache key). The cache does not see edits in helper modules: without it you measure the old kernel |
| compile without a GPU | hide the devices and set the `COMPILE_ONLY` option with the target architecture |

Resources on gfx942: 512 threads per workgroup (two waves per SIMD) leave 256 registers per wave, architected VGPRs and
accumulator file together; state held across a loop is where spills start (count accumulator moves as well as scratch);
LDS is 64 KiB per workgroup; direct global → LDS loads can add a full memory wait before the next LDS read.
