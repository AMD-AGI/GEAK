# Two further decode ops: a multi-stream residual mix and a gated delta-rule step

Both are described structurally. Both were gated on gfx942 only, with FlyDSL 0.4.x, on four shapes each;
gfx950 is untested. The stream mix **has** a timing against the deployed kernel and is about 3x slower.
The delta-rule step has **no** timing against a deployed kernel.

---

## A. Multi-stream residual mix (hyper-connection style)

### What it is

The layer carries `n` residual streams instead of one (`n` from 1 to 8). One call does three things for
every row:

1. the previous sub-block's **post** step: new stream `j` = `x x post[j] + sum_i comb[i, j] x stream[i]`;
2. this sub-block's **pre** step: a mixing projection over all `n x hidden` new stream values, normalised
   by their RMS, gives `2n + n^2` mix values; from them come sigmoid pre-weights, sigmoid post-weights,
   and an `n x n` combination matrix normalised by a few Sinkhorn iterations;
3. optionally the sub-block's **RMSNorm**, folded in on the pre-weighted collapse of the streams.

Four outputs: the new streams, the new post-weights, the new combination matrix, the (normed) collapsed
row. Not covered: the very first call of the model (no post step), zero Sinkhorn iterations, a packed
mixing weight, more than 8 streams, tensor parallelism above 1.

### Task decomposition and ownership

Three kinds of task; a task waits only for lower indices.

- **Column tasks** `0 .. hidden / 16 - 1`: task `t` owns 16 columns of every stream of every row. It
  forms and stores the new streams, and reduces its **partial** mix dot products and its share of the
  sum of squares; the partials go to a device workspace, then the task posts its flag.
- **One owner task**: waits for all column flags (thread `j` polls flags `j, j + threads, ...`), sums
  the partials **in a fixed order** (threads over tasks, a wave butterfly, the waves in index order
  through LDS) so the result does not depend on the workgroup count, forms the mixes, runs Sinkhorn,
  stores the two small outputs, and with the folded norm reduces the norm statistic over the whole row.
  It hands the pre-weights and the inverse RMS to the workspace and posts its flag.
- **Norm tasks**, one per 16 columns: wait for the owner's flag and write the normed row.

`2 x hidden / 16 + 1` tasks: 513 at a hidden size of 4096.

### Lane and wave layout

- Column task: wave `r mod waves` takes row `r`. Lane `L` holds stream values `L + 64 i`, that is
  stream `(L + 64 i) / 16`, column `(L + 64 i) mod 16`. Each lane keeps `2n + n^2 + 1` accumulators (its
  partial dots with the mixing weight and its share of the sum of squares); an XOR butterfly reduces
  them over the wave.
- Owner: **one combination-matrix element per lane** (lane `L < n^2` holds element `(L / n, L mod n)`);
  its row and column sums are gathered from the lanes of its row / column in index order. The
  iterations are a runtime loop with one loop-carried register.
- **Rows are runtime loops, not unrolled.** The first version unrolled them and needed 466 registers at
  3 rows; with loops the same shape needs 244.
- The owner and the norm tasks **recompute the stored streams from the inputs**, with the column tasks'
  exact steps (the same helper, hence the same bits), instead of reading the stream output back. That
  output may be written with plain stores by a workgroup on another XCD.

### Resources

- Registers: about `96 + 6 x (2n + n^2 + 1)`: 246 at `n = 4`, 192 at `n = 3`, 150 at `n = 2`. Measured
  90 to 244.
- LDS: a few hundred bytes (`4 x waves x (2n + n^2 + 2) x rows`).
- Workspace: about 108 KB at `n = 4`, hidden 4096, one row.

### Where the time goes, and status

- **0.32x of the deployed fused kernel pair** at `n = 4`, hidden 4096, one row: about 3x slower.
- The three phases are chained by workspace flags, and the owner is **one workgroup**: the mix
  reduction, the Sinkhorn rounds and the norm statistic (a pass over the whole row) are serial per row.
  Time grows with the width (about 2.5x more at 3 rows and a hidden size of 7168).
- The deployed kernel uses small matrix-core tiles for the mixing projection with a split-K reduction.
- No dead ends were measured; the time was not broken down further than the above.

### What to try next (untested)

Move the owner's serial work out of one workgroup: let the last-arriving column task finish the mixes
(the last-arrival pattern) instead of a separate owner phase, and fold the norm statistic into the
column tasks' partials so that the norm tasks need no second pass over the row.

### What to check

- Which values the mixing projection reads: the unrounded fp32 new streams or the stored bf16 ones. This
  one is visible in a test (several tolerances); the store rounding mode is visible only when the mixes
  read the stored streams. Both are parity facts for `native_decode_numerics`.
- The Sinkhorn iteration count: off by one is visible unless the iteration has already converged.
- Any workgroup count gives bit-identical small outputs (the fixed reduction order).

---

## B. Gated delta-rule decode step (linear attention with a recurrent state)

### What it is

One decode token of a gated delta rule on a per-sequence recurrent state, followed by the layer's gated
RMSNorm. Per row and value head, with state `S` of shape `[v_dim, k_dim]`:

`S <- decay x S`, `u = beta x (v - S k)`, `S <- S + u k^T`, `o = S q`, then the gated norm of `o`.

The state is updated **in place**. The decay is per head or per key channel. Not covered: multi-token
verify, prefill, a packed input row, the short convolution in front (it has its own recurrent state and
needs a different task split; keep it a separate op, as the deployed decode path does).

### The structural fact that makes it parallel

**The rule is independent per state row.** Row `i` of the new state, and `o_i`, read only row `i` of the
old state, plus `k`, `q`, the decay and `beta`. So the state rows of a head can be split across
workgroups with exactly one owner per state element, and the state needs no synchronisation.

### Task decomposition and ownership

- One task is one **(value head, split of the state rows)** over all decode rows. The split count is the
  largest power of two up to 16 that leaves every wave at least 4 state rows (8 at `v_dim` 128 with 256
  threads, 4 at `v_dim` 64).
- Every split re-derives `q`, `k`, the decay and `beta` of its head. That is a few hundred bytes of
  redundant reads, cheaper than sharing them.
- Decode rows must have distinct state slots (a group's rows are loaded before any is stored).

### Lane and wave layout, and why

- A wave updates a few state rows at once, and **a state row belongs to a group of adjacent lanes**, each
  lane holding a run of contiguous columns (16-byte loads and stores when the layout allows).
- A row dot (`S k`, `S q`) is a per-lane partial plus a short XOR butterfly **inside the row's lane
  group**. The first version striped a row over all 64 lanes, which cost a 6-step butterfly for each
  dot.
- `o = S' q + u (k . q)` reuses the decayed row, so one reduction pass per row is enough.
- **Loads first.** The loads of a group of decode rows (slot, state rows, `q`, `k`, gates, `v`) are all
  issued before any arithmetic, behind a scheduling barrier, so the rows' memory latencies overlap.
- Per-channel decays are computed once per column and shared through the wave's own LDS.

### Synchronisation

Only for the gated norm, which needs the sum of `o^2` over all state rows of a head and therefore over
all its splits. **Last arrival finishes:** each wave stores its `o` values and its partial sum with
coherent stores; the workgroup drains them and adds 1 to the head's arrival counter. The workgroup
whose add returns `splits - 1` reads all partials, sums them in split order (independent of arrival
order), writes all outputs of the head and resets the counter. Nobody waits.

### Resources

Registers 64 to 96 (four shapes). LDS 16 to 32 bytes with a per-head decay, about 8 KB with per-channel
decays. Workspace 2.6 to 10 KB. No scratch. The state is never held whole in registers.

### Where the time goes, and status

- The first version (one workgroup per value head, a serial chain per state-row block) against the
  split version: **2.0x to 2.6x faster** warm, with bit-identical outputs on the test shapes.
- What is left: an empty launch is about 26 to 37 % of the warm time; the rest is two dependent global
  round trips (the slot, then the state), the row-dot butterflies, and the cross-split merge (drain,
  atomic, workspace read: about 15 to 22 % of the total).
- Cold, everything sits on the launch floor.
- No timing against a deployed kernel.

### What to check

- **A value test cannot see a protocol bug.** A variant in which every arriving split finishes passes
  every numeric check, because the last finish rewrites the complete output; its defect is an early,
  repeated release of the output. Check "exactly one finisher for every arrival order" by reasoning or
  with a small model of the counter, and by running at 1 and 7 workgroups.
- Compare the whole state buffer after the call, counting stray writes, not just the output.
- Padding rows: the state is neither read nor written and the output is zero. Which state index means
  padding differs between backends.
- Shapes with a small state, where the norm's epsilon and the query scale become visible.
