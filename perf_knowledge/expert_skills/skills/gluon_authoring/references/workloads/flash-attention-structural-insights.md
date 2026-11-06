# Flash Attention — Structural Insights

Read this **before** importing a published attention technique (especially FA-4) or
deciding which non-MMA lever to pull. Operational how-to (online softmax, layout,
pipeline) stays in `attention.md`. **Full FA-4 technique catalogue:** only in
`tile-programming-cutedsl` (B200 / sm_100). **AMD (CDNA3/CDNA4):** validated optimization
strategies only — in this skill's fill below the portable core marker.

<!-- BEGIN portable core — byte-identical across the 4 tile skills (gluon / cutedsl / flydsl / tilelang). Keep in sync; everything below the END marker is this skill's per-target fill. -->

## Structural model

Attention is **not** one GEMM repeated. It is two matrix phases with real work
wedged between them:

```text
QK MMA  →  online softmax (running m, l, P)  →  PV MMA  →  epilogue
```

Per streamed K/V block the running state updates; whenever the row max grows, the
accumulated output `O` must be **rescaled** before the next PV accumulate is safe.
`P` is the **per-block softmax numerator**, not the final normalized attention matrix
(normalization is deferred to the epilogue).

## Tile-primitive graph (one K/V block)

```text
Q, K, V in GMEM
  → shared staging          (async / buffer load)
  → S in accumulator tier   (score MMA: QK^T)
  → P in accumulator tier   (softmax: exp + row reduce; may need matrix-readable layout)
  → O in accumulator tier   (value MMA: P @ V)
  → rescale O (when m grows) then epilogue normalize
```

Each arrow is a tile-op — fill a contract card per stage (`tile-programming/tile-op-contract.md`:
Scope, Layout, Dispatch, Handoff).

## Why GEMM-only levers fail here

- A **VALU stage** (softmax: exp, row-max, row-sum, rescale) sits on the hot-loop
  critical path between the two matmuls — GEMM-class instruction-scheduling knobs
  (compiler misched assumptions; e.g. a GEMM throughput-pairing scheduler on AMD) often assert or regress.
- **Online-softmax recurrence** carries `(m, l, O)` across blocks — blocks symmetric
  ping-pong where two warps run identical-phase-offset softmax.
- The PV accumulator may be **read-modified every iteration** (rescale then the inner
  matmul), not write-only-until-epilogue — this can force accumulator register-file
  round-trips (e.g. AGPR↔VGPR on AMD) unless the common path accumulates in place.

## Bound-class porting filter

Before importing a published technique, classify **which bottleneck it fights**:

| Question | Blackwell (B200) | AMD CDNA3/4 |
| --- | --- | --- |
| Targets exp / shared **throughput** (asymmetric scaling)? | Primary FA-4 fight | Only if profile proves exp/smem **saturated** — uncommon on occupancy-first CDNA fwd |
| Targets latency / accumulator **read-modify-write**? | Decoupled correction warpgroup | **Low-occupancy decode** where PV cannot hide AGPR traffic |
| Targets scheduling **makespan** (LPT)? | Yes | Yes — causal load imbalance; confirm with profile, not throughput |
| Requires TMEM / `tcgen05` / separate MUFU / 2-CTA cluster MMA? | Native on B200 | **Not available** — do not plan around |

## FA-4 technique names (verdicts are per-skill fill below)

FlashAttention-4 ([arXiv:2603.05451](https://arxiv.org/html/2603.05451v1)) fwd techniques:

1. **Pipeline** — TMEM + fully-async MMA + warp-specialized QK / softmax / PV / correction
2. **Polynomial exp** — partial `2^x` on FMA while rest on hardware exp unit
3. **Conditional rescale skip** — skip `O *= alpha` when `m_j - m_{j-1} <= tau`
4. **Backward** — 2-CTA MMA + TMEM traffic reduction
5. **LPT causal** — longest-processing-time-first grid (architecture-agnostic)

<!-- END portable core -->

## gfx942 / gfx950 (CDNA3/CDNA4) — validated strategies

Written gfx950-first (CDNA4, MI350X / MI355X); measurements taken on gfx942 (CDNA3) are labelled
as gfx942 downgrade evidence where they appear. CDNA3/CDNA4 lack Blackwell FA-4 hardware (no TMEM, no `tcgen05`, no separate
MUFU/FMA ports, no 2-CTA cluster MMA) and use an **occupancy-first** model: latency
is usually hidden by resident waves, not by register-heavy warp-specialized pipelines.
**Do not treat FA-4 as a porting checklist on AMD.** Use the strategies below when
profile + asm show the **same bottleneck class** they target.

### LPT causal (longest-processing-time-first)

- **Bottleneck class:** grid **makespan** / load imbalance — causal masking makes
  per-worktile KV work grow with `mblock`; ascending block order leaves expensive
  tiles on the tail while CTAs sit idle.
- **Why it works:** reversing `mblock` order (longest first) is a pure `pid → worktile`
  remap — zero kernel-body change, correctness preserved. It shortens the critical
  path of the **grid**, not the inner loop.
- **Apply when:** causal forward (or backward with the same imbalance), before editing
  softmax or pipeline. Pair with batch-outermost + head-section swizzle for L2
  (`attention.md ## Scheduling`, `gemm.md ## Worktile scheduling`).
- **CONFLICTS with rasterization / `pid`-swizzle (do not stack blindly).** LPT and an
  L2-reuse `pid → worktile` swizzle are BOTH grid remaps; stacked, they can fight — the
  LPT reversal can break the swizzle's L2-reuse ordering (a sibling gfx942 FA kernel
  measured LPT **−6% to −19%** on a swizzle-enabled kernel, the opposite of the +30-40%
  LPT gives swizzle-free). They are NOT independently additive: do not assume "LPT + swizzle"
  beats either alone. A/B LPT-only vs swizzle-only vs both and keep the winner; if the grid
  already rasterizes for L2 reuse, LPT may add nothing.
- **Needs a monotone cost axis (non-uniform varlen packing breaks it).** Same shape of failure
  as the swizzle conflict above — the precondition, not the arch, decides. The gate, knob and
  acceptance are stated once, in `attention.md ## Scheduling (load-imbalanced grids)` (**LPT
  uniformity gate (varlen)**); apply it before keeping LPT on a varlen dispatch.
- **Verify:** profile or schedule analysis shows spread in per-worktile time; A/B is
  a host-side remap only — re-run correctness on causal shapes; include the
  LPT-vs-swizzle-vs-both A/B above.

### Conditional rescale skip

- **Bottleneck class:** **latency** on the PV path — each K-block does
  `acc *= alpha` then `mfma(p, v, …)` where `acc` lives in AGPR. The VALU rescale
  forces `v_accvgpr_read` / writeback every iteration, breaking the efficient
  write-only-until-epilogue MFMA cadence.
- **Why it works:** when the running row-max is **stable** across consecutive blocks,
  `alpha = 1` and the rescale is dead work. Skipping it (with slack `tau` and final
  epilogue renormalize) lets the common path use `mfma(p, v, acc)` in-place — no AGPR
  round-trip on that iteration. See `attention.md ### Conditional rescaling (skip the per-block acc *= alpha)`.
- **Apply when — path A, latency (all should hold):**
  - **Low occupancy** — often one wave per SIMD on decode; rescale is **exposed**, not
    hidden by other waves.
  - **Asm shows AGPR read-modify** on the PV hot path every iter.
  - **Running max often stable** — typical in decode over long KV with similar logits;
    rare max jumps mean the branch still pays but savings are small.
- **Apply when — path B, co-execution budget (a different mechanism, opposite occupancy
  precondition):** the kernel runs **two waves per SIMD** with the softmax riding in the compute
  clusters, and one cluster's vector **demand exceeds its MFMA-shadow capacity** — the budget
  arithmetic shows the overflow (`../tile-programming/llir-codesign.md ## Attention: the
  co-execution budget`). What the skip buys there (window capacity, not an AGPR round-trip), why
  it must be a **real branch** placed in a **memory** cluster, and whether the build can express a
  warp-uniform branch at all are stated once in
  `attention.md ### Conditional rescaling (skip the per-block acc *= alpha)`.
- **Do not apply when:**
  - **Multi-wave occupancy** already overlaps rescale with memory or other VALU work **and no
    cluster is over its co-execution budget** — then rescales are effectively free and the
    warp-uniform branch is pure overhead. This is the exclusion path B carves out: multi-wave is
    not by itself a reason to skip the lever, an *unconstrained budget* is.
  - **Compiler auto-pipeliner** interleaves rescale with MFMA/LDS — adding a
    data-dependent skip **breaks** the steady schedule (common on plain-Triton /
    TileLang `num_stages>1` dense prefill).
  - **Dense prefill, hand-built Gluon pipeline** with neither profile in hand — branch + dual
    softmax paths often cost more than the saved rescale unless the profile proves AGPR traffic
    dominates (path A) or the budget shows an over-capacity cluster (path B). The failure mode is
    applying it on a hunch, not the workload being prefill.
- **Verify before keeping:** `waves_per_eu` / occupancy counters; grep asm for
  `v_accvgpr_read` on the PV loop; determinism race-test after any handoff change. A recorded
  measured negative at the occupancy path A asks for is in
  `../pitfalls/negative-patterns.md ## Conditional rescale skip — measured negative at the occupancy its own path A asks for`. On path B also
  confirm **something spends the freed budget** — the skip alone frees capacity without converting
  it into cycles, so measured alone it can read as a null result and get reverted
  (`../method/close.md ## Attributing a change that only creates headroom`).

### VALU folds (operational, not FA-4)

Fold `log2e` into the scale, pre-scale `Q` once — reduces per-block softmax VALU and
register pressure. Mechanism: `attention.md ## Online softmax`.

### Software-pipeline reproduction — the kernel shape re-injection needs

**Lowest rung — read only after the hand-written route.** When a Gluon FA loop is
**schedule/overlap-bound** (equal VGPR+AGPR and occupancy as plain but lower `MfmaUtil` + more
full-drain `s_waitcnt lgkmcnt(0)`), the default repayment is the hand-written overlap
(`attention.md`, layer-roadmap step 6, and
`../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`).
Re-injecting plain's TTGIR software pipeliner
(`../tile-programming/pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`;
lever `reinject_ttgir_pipeliner`) is admissible only as a **diagnostic below the parity gate** (to
size the `lost_pipeline` debt) or as a **last resort** when the hand-written ring cannot reach
parity. Its ceiling is plain parity; its numbers are labelled *injected* and never reported as a
win; it is never applied to an incumbent (already-Gluon) kernel; and on a loop that already
hand-authors its staging it cannot fire at all. When it is used, shape the **attention kernel** so
the pipeliner has room. All figures below are *injected* measurements on gfx942 FA-fwd (gfx942
downgrade evidence; not re-measured on gfx950):

- **Load K/V IN-BODY, with NO hand register-prefetch.** The pipeliner *is* the prefetcher; a manual
  register-prefetch "uses up" the double-buffer slot (in-body loads gained +7-15% from `num_stages=2`;
  a manually-prefetched loop gained ~+0.8%).
- **Split the causal mask into TWO loops (plain's shape), not one loop with an `scf.if`.** The clean
  full-region loop is branch-free, so it pipelines exactly like plain's first `scf.for` — and it also
  un-blocks `BlockPingpong` (which bails on a loop-variant mask). A loop-variant predicate is exactly
  the "data-dependent skip breaks the steady schedule" hazard above (## Conditional rescale skip).
- **Compose on top of LPT causal remap + XCD** (## LPT causal; `attention.md ## Scheduling`) — they
  stack for free; the recorded champion shape is two-loop mask-split + in-body loads + LPT + XCD at `num_stages=2`.
- **`num_stages=2`** is the sweet spot *for the injected pass* (ns3 worse; ns4 takes the chained-dot path
  and regresses via the occupancy cliff) — this is the parameter handed to the re-injected pipeliner,
  not a Gluon knob: on the Gluon path itself `num_stages` is dead in 3.8.0. The pipeliner's
  prologue/epilogue overhead loses at tiny `S` (< 2048) — dispatch the non-pipelined variant there.

### Do not plan on AMD

Polynomial exp emulation (no separate exp issue port — adds VALU on the same SIMD),
TMEM/warp-spec FA-4 pipelines, 2-CTA MMA backward tricks. **gfx1250 (CDNA5 / MI450)
is a separate data-center fork, not yet profiled in this skill** — its asynchrony-first
primitives (named-barrier warpgroups + TDM) make FA-4-style warp-spec plausible there,
but probe + profile on-target before planning. Full FA-4 map: `tile-programming-cutedsl`.

## Cross-links

- tile-op contract card: `../tile-programming/tile-op-contract.md`
- operational attention workload: `attention.md`
- bound-class / exp unit: `../method/profile.md`, `../hardware/planning-constants.md`
- overlap order on the Gluon path (hand-written first, re-injection last): `../tile-programming/pipeline.md`
- LPT pid remap: `attention.md ## Scheduling`, `gemm.md ## Worktile scheduling`
