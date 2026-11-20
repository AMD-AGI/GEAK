# Per-stage kernels of a decode layer: weights, attention, MoE

Levers for the stages that sit far above their byte floor in the stage timeline
([measuring.md](measuring.md#floors)). Numerics that drift only downstream are in
[numerics.md](numerics.md). All items are drafts: measure each against the plain version.

Contents: [Stream the weights at HBM speed](#stream-the-weights-at-hbm-speed) ·
[Split-KV decode attention](#split-kv-decode-attention) ·
[Wave-local decode attention](#wave-local-decode-attention) ·
[Decode MoE](#decode-moe-router-top-k-fp4-experts)

## Stream the weights at HBM speed

**When.** A GEMV stage (QKV, o_proj, gate_up, down, expert GEMMs) is well above its floor in the timeline.
At decode width 1–8 every weight byte is read once per step and used for a handful of FLOPs: the layer is
HBM-bound ([floors](measuring.md#floors): compute the GEMV floor before deciding a stage is slow).

**Recipe.**
1. **Wide, coalesced loads.** 16 B per lane (`buffer_load_dwordx4`), consecutive lanes on consecutive
   addresses along K; a workgroup streams contiguous rows. Read fp8 weights as fp8, never upcast in
   memory.
2. **Dequant in registers.** Block-scaled fp8 (`weight_scale_inv`, 128×128 blocks): load one scale per
   block per row-tile, multiply after the fp32 accumulation of the block, not per element.
3. **Enough bytes in flight.** One workgroup per CU means latency must be hidden inside the workgroup:
   several outstanding loads per lane (unroll K), or double buffering through LDS. Too few rows per
   workgroup → split K across workgroups and reduce (in the same fixed order on every rank).
4. **Activation reuse.** The input vector (width × hidden) is small: keep it in LDS or registers for the
   whole stage instead of re-reading it per row.
5. **Prefetch.** Weights do not depend on the step's data. Workgroups idle during attention or a wait can
   touch the next stage's first tiles so they land in L2 (per-XCD, 4 MiB) just before use; only worth it
   when the timeline shows idle CUs. When to start each stream:
   [early-issue prefetch](handoffs_and_sync.md#issue-the-next-stages-loads-before-its-inputs-arrive).
6. Generic FlyDSL levers: [`languages/flydsl/authoring_gemm_levers.md`](../../../../languages/flydsl/authoring_gemm_levers.md)
   (tiling, prefetch, epilogue), [`patterns.md`](../../../../languages/flydsl/patterns.md) §4–6 (split-K
   for skinny shapes, small-M family, fp8 preshuffle). The expert skill
   [`flydsl_fp8_blockscale_gemm`](../../flydsl_fp8_blockscale_gemm/skill.md) documents the fp8 blockscale
   parity fix and per-shape tiling (validated on gfx950; re-measure on gfx942).

**Pitfalls.**
- A one-time weight repack (preshuffle) is allowed only keyed by `data_ptr`, with its bytes reported
  (gate 7, [acceptance_gates.md](acceptance_gates.md#hard-gates)); never per call.
- MFMA is not the bottleneck at width 1; do not trade load width for MFMA-friendly layouts unless the
  timeline says compute.
- Changing the K split changes the summation order (still within the tolerance; never the all-reduce
  order, [all-reduce fusion](handoffs_and_sync.md#fuse-the-tp-all-reduce-into-its-neighbours)).

**Check.** Timeline span vs floor, the oracle per width, the gates.

**Related.** Block scales are small but can cost more requests than the weights when the framework's
layout scatters a wave's lanes over many cache lines: check the addresses one wave touches; a copy of the
scales in the kernel's read order (made once, keyed by `data_ptr`, bytes reported) or issuing the scales
before the weights are the options. MoE specifics: [Decode MoE](#decode-moe-router-top-k-fp4-experts).

## Split-KV decode attention

Inside the persistent kernel.

**When.** Attention dominates the timeline at the long context bucket (e.g. 20k tokens), or its span grows
with context while few workgroups work on it. At width 1 there are only `heads_per_rank` query heads (a
few KV heads with GQA), far fewer than the CUs.

**Recipe.**
1. **Split the KV sequence** of each (KV head, sequence) into chunks, one chunk per workgroup, enough
   chunks to cover the attention partition. Each workgroup computes over its chunk the running max `m`,
   the sum `l` and the unnormalized output `acc` for every query head of the group (flash-decoding).
2. **Combine in the kernel.** Each split writes `(m, l, acc)` to a scratch buffer and posts a mailbox; one
   workgroup per head (or the consumer of the attention output) waits on the head's splits and merges
   them in split order: `m* = max m_i`, `l* = Σ l_i·e^{m_i−m*}`, `out = Σ acc_i·e^{m_i−m*} / l*`.
3. **Read KV in the framework's layout.** Paged cache via the block table; fp8 KV with its k/v scales as
   the native path applies them (the layer's reference implementation says which). Dequant in registers.
4. **Sliding window / sinks.** For SWA layers only the last `window` tokens count: fewer chunks, and the
   split count can be lower than for full attention. Attention sinks enter the softmax as extra logits in
   the combine (once per head, not per split).
5. Generic levers:
   [`languages/flydsl/authoring_attention_levers.md`](../../../../languages/flydsl/authoring_attention_levers.md)
   (fusion boundary, MFMA orientation, which axis goes on atomics).

**Pitfalls.**
- Chunk count depends on the context, which changes every step: derive it on the device from `seq_lens`
  (graph replay keeps the arguments fixed), and size scratch for the largest bucket.
- Merge in a fixed split order; an arrival-order merge is nondeterministic.
- The KV write of the current token must be visible before any split reads the last block (same
  workgroup, or a mailbox between the write and the reads).
- Empty chunks (context shorter than the split count) must still post their mailbox, with `l = 0`.

**Check.** The oracle (both context buckets, graph replay, sequences across steps), timeline at the long
bucket, the gates. Per-workgroup levers: [wave-local attention](#wave-local-decode-attention); for widths
> 1, give workgroups to rows by context length ([multi-row decode](layer_structure.md#multi-row-decode)).

## Wave-local decode attention

K straight from the KV cache. Builds on [split-KV attention](#split-kv-decode-attention) (how to split KV
across workgroups) with what makes each workgroup fast. Draft: measure every item against the plain
version.

**When.** `attn` dominates the timeline, or its span grows with context faster than the KV bytes do
(barriers / LDS round trips per key block).

**Recipe.**
1. **Each wave runs its own online softmax** on its key block: Sᵀ = K·Qᵀ on MFMA (e.g. 16x16x16 BF16),
   with K loaded from the KV cache directly as the MFMA operand — no LDS staging of K, no CTA barrier
   inside the key loop. P stays in registers as the A operand of PV; the waves merge (m, l, acc) once at
   the end.
2. **V^T through LDS with a swizzle:** a row-dependent XOR per small token unit removes the bank
   conflicts of the transposed write. A wave-private LDS slice for V^T avoids CTA barriers.
3. **q quantized cooperatively:** for FP8 KV the whole CTA writes the FP8 q into LDS once and reads it
   back in MFMA layout, instead of every lane quantizing its own copy.
4. **Long context: split-KV over many CTAs + combine** ([above](#split-kv-decode-attention)). Size the key
   blocks so the split covers the free CTAs; spread the combine over (head, dim segment) so it is not a
   few-CTA tail. If the native FP8 decode rounds P against a running max per segment (read its source in
   the task's native-source list), keep the same segments, or exchange the in-segment max between the CTAs
   that share one, so P rounds identically.
5. **Long tasks read K once:** when a task's scores exceed the LDS, write them to a per-CTA HBM scratch in
   a pre-scan and let the later passes read the scores back instead of K.
6. **Write KV after the attention loop** (its stores would otherwise turn the K/V waits into `vmcnt(0)`);
   RoPE and the KV write can live in the attention CTAs.
7. **Sliding window:** one CTA per KV head with the sinks in the same softmax; align the window start down
   to a tile and treat the tokens outside the window as 0 without reading them.

**Pitfalls.**
- Loads shaped to avoid the transpose (narrow V loads) can be slower than the swizzled transpose; measure.
- Every buffer you add (score scratch, double-buffered K/V) costs VGPRs or LDS; check the ISA for scratch.
- The current token's K/V must go through the same FP8 quantize–dequantize as the cache, or the layer
  drifts.
- Widths > 1: allocate long-context attention CTAs by context length, not per row group
  ([multi-row decode](layer_structure.md#multi-row-decode)).

**Check.** Attention span vs its KV byte floor at every context bucket; bit-identical output vs the
previous version when the change is a pure schedule change.

## Decode MoE: router, top-k, FP4 experts

For MoE layers with block-scaled FP4 experts (e.g. MXFP4 with E8M0 scales). Draft: measure each item. The
native routing rules (scoring function, bias, grouping, renormalization) come from the layer's reference
implementation ([native_reference_notes.md](native_reference_notes.md#mxfp4-moe-experts-moe_backendaiter_triton_mxfp4_bf16-aiterw4a16expertsmonolithic))
and the framework sources in the task's native-source list.

**When.** Router / top-k / expert stages are several µs each in the timeline; experts stay well above
their weight-byte floor.

**Recipe.**
1. **Router logits across CTAs, selection in every CTA that needs it.** A few CTAs compute the logits and
   publish them; every consumer CTA reads all logits, applies the native scoring, picks top-k and
   normalizes, so there is no second hand-off. Keep the number of readers small if they all read the same
   few KB ([task tables](layer_structure.md#task_tables)).
2. **Top-k at wave level:** each lane first sorts its few candidates (score desc, id asc on ties); then k
   rounds compare only the lanes' heads (DPP max + ballot for the lowest lane holding it).
3. **FP4 decode in registers:** `v_perm` can build FP16 (or FP8) bytes directly from E2M1 codes, since
   every E2M1 value is exact in those formats — no LDS lookup table.
4. **Scales in read order.** If the framework's scale layout scatters a wave's lanes over many cache
   lines, keep a copy of the scales permuted to the kernel's read order (made once at bind time, keyed by
   `data_ptr`, bytes reported to the memory gate), or at least issue the scales before the weights.
5. **Several independent accumulators** for the FP4 dot products, so the FMA chain is not latency-bound.
6. **Rounding points as native:** fold the routing weight where the native kernel does, round each
   expert's result where it does (often BF16 before the sum); then the second all-reduce produces the next
   layer's input.
7. **Cacheable loads** for expert weights when several rows hit the same expert (widths > 1); for one row
   it does not matter.

**Pitfalls.**
- MFMA accumulation order differs from the native MoE kernel: no bit-identity, only tolerance. Near-tied
  routing scores flip experts; compare with the routes native actually executed, not a recomputed top-k.
- A scale copy costs memory (KV-cache capacity); account for it in the memory budget.
- Multi-row expert paths reuse LDS between rows: a missing barrier at the end is a real race.

**Check.** Expert span vs its weight bytes (k experts × gate/up/down at FP4 + scales); the oracle with
near-tied routing.

## Sources

- In this repository: [`languages/flydsl/`](../../../../languages/flydsl/),
  [`flydsl_fp8_blockscale_gemm`](../../flydsl_fp8_blockscale_gemm/skill.md).
