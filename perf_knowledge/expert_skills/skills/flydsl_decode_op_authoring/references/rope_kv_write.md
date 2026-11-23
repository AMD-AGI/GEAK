# RoPE of q / k, with or without the paged KV-cache write

Rotary embedding of the current token's q and k, optionally with the write of its K and V into a paged
KV cache. Maturity: the design passed every correctness leg on gfx942 with FlyDSL 0.4.x over 16 shapes;
gfx950 and tensor parallelism above 1 are untested; **the design itself has no timing against a
deployed kernel**. It is shaped for a slot inside a larger kernel. For a standalone RoPE + KV-write
launch, two kernel lanes measured kernels of their own against the deployed kernel, and two of their
results go against steps that look obvious (vector loads and stores; doing the kv work on the
query-head task): read
[Measured in a kernel lane](#measured-in-a-kernel-lane) first.

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
- Use this form only when a consumer in the same launch needs that ordering. *Measured in a kernel
  lane* (one run): in a standalone launch, keeping the kv chain on a query-head wave was 4 % slower than
  giving the kv heads their own waves, as in the form below.

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

- Registers per wave: 22 to 43 without the norm (rope-only on a packed row 22 to 33; with the cache
  write 32 to 43); 28 to 67 with the per-head norm; 49 for the reciprocal-multiply fp8 write. Plan for
  48, or 80 with the norm.
- LDS 0, scratch 0.

## Where the time goes

- **Every element is a separate 2-byte (bf16) or 1-byte (fp8) load and store.** Vectorising them looks
  like the first optimisation step. One of the lanes below measured the opposite at decode
  widths: narrow lanes with a short per-lane instruction chain won.
- With the write, each kv head's k rotation and cache write is repeated by all the query-head tasks of
  its group: 16 times for a 16-to-1 layout.
- The per-head norm statistic is reduced redundantly by every wave (each wave reads the whole head
  once more): cheap for head dims up to a few hundred, and the first thing to share through LDS.
- A packed input is waited on per 16-column tile in a loop (12 waits for a 192-dim head) when the op
  runs inside a kernel with hand-offs; a standalone op has no such waits.

## Measured in a kernel lane

Two kernel lanes each authored a FlyDSL
fused RoPE + paged KV write on gfx942 against the deployed Triton kernel: 64 query heads and 8 kv heads
of 64 dims, rotation in halves, bf16 cache, 7 timed cases at 1, 2, 4 and 8 tokens, all under graph
replay, bit-exact. Every figure below is *measured in kernel lanes*, one run each, one optimisation
round each: the weighted results were re-measured independently, the per-move shares are the lanes' own
same-process A/B figures. A single run resolves about 4 % (four runs of one build gave 1.31x, 1.20x,
1.31x, 1.30x).
Do not carry the figures to another operator, shape or GPU generation.

**Result.** Weighted speedup over the deployed kernel: 1.40x in one lane and 1.30x
to 1.32x in the other.

**Read the oracle's case weights first.** The two 4-token cases carried 83 % of the weight.

**Measure the floor first.** An empty kernel under the oracle's timing cost about 64 % of the deployed
kernel's call, for any grid tried (1 to 512 workgroups, 64 to 512 threads). The two lanes ended at
1.10x to 1.15x and at 1.17x to 1.20x of that floor; a kernel with an empty body would score about
1.55x to 1.58x.

**What paid.**

| move | measured |
|---|---|
| one dependent chain per wave: query heads and kv heads in separate single-wave workgroups, so no wave carries both the q chain and the slot, k, v chain | 17 % at 4 tokens (one lane); the kv work became free |
| every independent load issued at entry (row, rotation partner, v row), position and slot through scalar loads, an issue fence, and strides as build constants (a 40-byte block of 5 pointers instead of 156 to 196 bytes) | 1.16x (other lane) |
| one rotation pair per lane, which cut the instructions after the last wait from about 165 to about 40 | 1.07x; two pairs per lane were 1.9 % slower, four pairs 7 % |
| write-through (device-scope) policy on the stores | 5 % per call at 4 and 8 tokens (one lane) |
| store predication without a branch (a zero-length buffer descriptor for rows that must not be written) | enabling: a branch on the slot let the compiler sink the k / v loads behind the test |
| dropping intermediate NaN selects (proven bit-exact) and folding the rotation sign into the sine | 1.9 % |
| ignoring the stride of a size-1 dimension in the layout guard: a one-token view reports an arbitrary token stride, and a strict guard silently sent that case to the generic kernel | 7 % on the one-token case in one lane, 1.01x to 1.23x and more in the other; both lanes hit it |

**Wide inputs need their own kernel.** The decode kernel (one row per wave) ran at 0.27x to 0.84x of
the deployed kernel from 128 tokens up. A kernel with several rows per wave ran at 1.4x to 2.3x from 16
to 16384 tokens; 16 tokens was the crossover. Those inputs had no weight in this task's metric and were
kept so that non-decode calls do not regress.

**Dead ends.**

| tried | result |
|---|---|
| kernarg preload (both lanes) | 3 % to 4 % slower, at best neutral |
| non-temporal stores | slower than write-through in one lane, neutral in the other |
| cache-policy variants on the loads | neutral: the rows are cold after the oracle's eviction |
| kv heads first in the grid; loading the v row from the query wave | noise |
| the rows-per-wave kernel at 8 tokens or fewer | slower than one row per wave |
| knock-out probes in one lane: no rotation arithmetic, no cos / sin hop | not faster in that lane, which therefore tried no trims; the other lane gained 1.9 % from arithmetic trims, so treat these as unproven; removing the row loads entirely would have saved 7 % |

**Where it ended.** The launch floor is 85 % to 89 % of the call; the rest is the argument fetch and
the cold read of the row.

**A knock-out probe does not close a path.** In one lane a probe that removed some stores was not
faster, so the lane closed the whole store path and tested only non-temporal stores. The other lane
measured 5 % from a device-scope policy on the same stores, most of the gap between the two results.

## What to try next

1. Take the lane geometry above for a standalone launch, including the store policy and the
   branch-free predication.
2. Share the norm statistic through LDS within a workgroup (for the fused per-head norm; untested).
3. Not this first: wider lanes. Two and four pairs per lane lost. Wider stores were not measured; one
   lane ruled them out with a knock-out probe only.

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
