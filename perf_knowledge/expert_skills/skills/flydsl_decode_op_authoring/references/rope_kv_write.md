# RoPE of q / k, with or without the paged KV-cache write

Rotary embedding of the current token's q and k, optionally with the write of its K and V into a paged
KV cache. Maturity: the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over 16 shapes;
gfx950 and tensor parallelism above 1 are untested; **there is no timing against a deployed kernel**.

## What the design covers

| covered | not covered |
|---|---|
| one token per decode row; separate `q [rows, heads, dim]`, `k [rows, kv_heads, dim]` (and `v`) inputs, **or** one packed `qkv [rows, N]` row laid out as q, then k, then v, with each head contiguous | more than one token per row; a packed row with padding, an interleaved K / V part, or an extra trailing part |
| the rotation of the leading `rotary_dim` dims in halves or in adjacent pairs, the rest passed through; **no rotation at all** (`rotary_dim` 0) | multi-section rotary embeddings, a rotation offset, the inverse rotation |
| any cos / sin table the framework built, stored bf16 or fp32 | a kernel that computes its own angles |
| **rope only** (another op writes the cache) or **rope + cache write** into a paged cache addressed by element strides, bf16 or the arch's fp8 with a per-tensor scale | a cache with packed key dims, V in another allocation, per-token scales, a latent-attention cache (see [`mla_decode.md`](mla_decode.md)) |
| GQA, MQA, MHA; `v_dim` different from the q / k dim | `k_dim` different from `q_dim` |
| a constant multiply of v by the model ahead of its hand-off or cache write | a multiply of k; a per-head or per-token v scale |
| padding rows (negative slot): rotated, never written | |
| a fused **per-head RMSNorm of q and k** ahead of the rotation, weighted or weightless | a norm over the whole row, a norm on v, a weight applied before the statistic |

## Two design rules that are not about speed

1. **Never compute an angle.** Read the layer's own cos / sin table at each row's position. Linear,
   NTK, YaRN and other scalings only change the table, and a model whose layers use different rotary
   bases simply has a different table per layer. No parameter names a base.
2. **The cache is the product, not the rotated q.** With the write, nothing in the rotated q shows
   whether the current token was written, written to the right slot, or written to the right head. A
   test that compares only the outputs passes a kernel that writes nothing. Compare the whole cache
   after the call; without the write, the cache must be bit-identical to before (a stray write fails).

Rounding points (the rotation in fp32 with one rounding or in bf16 operations, where k is rounded
before the cache quantisation, divide versus reciprocal-multiply for an fp8 cache, the table's dtype as
read) are parity facts: see `native_decode_numerics`. Several of them cannot be told apart by a
tolerance test at all, so they must come from the backend's source.

## Task decomposition and ownership

**With the cache write:**

- Task `h` (one per query head) is **query head `h` over all rows**. It also rotates kv head
  `g = h // group` of k and writes `g`'s K and V into the cache, and only *then* releases the rotated
  q of head `h`.
- Reason: an attention that waits on the rotated q of head `h` then finds the current token in the
  cache without needing a separate signal for the write.
- The `group` tasks of one kv head write the same bytes. Only the first of them stores and releases the
  rotated k and the v of that kv head, so each of those outputs is produced exactly once.

**Without the cache write:**

- A query-head task rotates q only. One extra task per kv head rotates its k and moves (or multiplies)
  its v. Nothing is repeated.

Either way task `i` belongs to workgroup `i mod n_blocks`, so any workgroup count is correct. A task
waits only for what it reads: its q head, and the k / v head it rotates or copies. On a packed input
that is the 16-column tiles its head's columns touch; a head need not start on a tile boundary.

## Lane and wave layout, and why

- **All threads of the workgroup work, with no cross-lane operation.** Thread `t` owns rotation pair
  `t`, `t + threads`, ..., then the pass-through dims and the v dims the same way.
- Why: this op is memory-bound with no cross-thread reuse. Each element is read once and written once,
  so LDS staging would only add synchronisation. The public
  [optimization card](../../../../languages/flydsl/authoring_optimization.md) says the same for this
  class: vectorise and coalesce, do not stage.
- Rows are a Python loop unrolled at build time; a negative slot masks only the cache stores.
- With the per-head norm, **every wave reduces the whole head by itself**: lane `l` sums dims
  `l, l + 64, ...`, then an XOR butterfly across the wave. Still no LDS and no barrier. Each element the
  rotation reads is then the normalised value.

## Synchronisation

None inside the op. The one ordering requirement is that the cache stores of a task are issued, with
the same coherence as its outputs, before the release of the rotated q that consumers wait on.

A caution that matters when another kernel region reads the cache in the same launch: L2 is per XCD,
so a plain load of the cache from a workgroup on another XCD can see a stale line unless the reader's
wait invalidates its cache or the reader uses coherent loads. In a single-op unit test the cache is
read after the launch, so this does not show there.

## Resource budget

- Registers: 22 to 43 without the norm (rope-only on a packed row 22 to 33; with the cache write 32 to
  43); 28 to 67 with the per-head norm; 49 for the reciprocal-multiply fp8 write. Plan for 48, or 80
  with the norm.
- LDS 0, scratch 0.

## Where the time goes

- **Every element is a separate 2-byte (bf16) or 1-byte (fp8) load and store.** Vectorising them is the
  first optimisation step and has not been done.
- With the write, each kv head's k rotation and cache write is repeated by all the query-head tasks of
  its group: 16 times for a 16-to-1 layout.
- The per-head norm statistic is reduced redundantly by every wave (each wave reads the whole head
  once more): cheap for head dims up to a few hundred, and the first thing to share through LDS.
- A packed input is waited on per 16-column tile in a loop (12 waits for a 192-dim head) when the op
  runs inside a kernel with hand-offs; a standalone op has no such waits.

## Measured dead ends

None recorded: no optimisation round has been spent on this op.

## What to try next (in this order)

1. Vector loads and stores: 8 or 16 bytes per access, a thread owning a run of adjacent pairs.
2. Untested suggestion, with the write: let one task per kv head do the k rotation and the cache
   write, and have the query-head tasks of its group depend on it, instead of repeating the write. This
   needs a release order (owner pattern), so do it only if the repeated write shows in a profile.
3. Share the norm statistic through LDS within a workgroup.

## What to check

- **Who writes the cache.** In one common backend the rotation is done by the model's forward and the
  cache write by the attention call. A rope op that also wrote would write twice, which is harmless
  only if both writes round identically; one that is expected to write and does not leaves the cache
  stale. Read which side writes before choosing the variant.
- **The packed split.** q, k and v are three column ranges whose offsets follow from the head counts
  and dims of the outputs. When the v dim differs from the q / k dim, an offset computed with one dim
  reads the wrong columns (off by hundreds of tolerances).
- **The v multiply** is `bf16(fp32(v) * fp32(c))`; dropping it is a large error, keeping v in fp32 into
  the cache write differs in the last bf16 bit.
- **An fp8 write that saturates.** If the backend clamps before the fp8 convert and yours does not, an
  out-of-range element becomes NaN in the cache. Include a case that overflows.
- Padding rows: rotated, nothing written.
- No rotation (`rotary_dim` 0) must read neither the table nor the positions.
- The cache comparison resets the cache to random contents before every call; otherwise a write to the
  wrong slot can go unnoticed.
