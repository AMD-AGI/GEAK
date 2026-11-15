# Layer structure: layouts and the levers that change them

How the CTAs of a persistent layer kernel are divided among the stages, and the levers that rebalance,
re-place or overlap them. Structural changes (which steps the kernel covers, how stages overlap) tend to
pay more than per-stage polish, so decide the layout first, then tune it from the timeline
([measuring.md](measuring.md)).

Contents: [Layouts](#layouts) · [Rebalance the CU plan](#rebalance-the-cu-plan) ·
[Multi-row decode](#multi-row-decode) · [XCD placement](#xcd-placement)

## Layouts

A layout fixes, before the stage code is written, the thread count, the number of CTAs, the
synchronization style and the register / LDS budgets. Three layouts exist; the parameter values below are
starting points (marked draft where they were not measured) that a **calibration** run adjusts: a measured
trial of the layout, with per-stage times and the ISA resource report. Names in code font such as
`grid.threads`, `grid.n_blocks`, `sync.grid_barriers`, `waves`, `blocks`, `tasks`, `in_group`,
`replicate`, `budget.vgpr`, `budget.lds_bytes`, `prefetch` and `unit` are the field names of the layout
description these templates were written for; read them as parameter names.

| layout | idea | pick it when |
|---|---|---|
| [`staged_all_ctas`](#staged_all_ctas) | every CTA runs every stage in order and waits only on the tiles it reads | default for decode layers |
| [`task_tables`](#task_tables) | each stage is a table of tasks on staggered base CTAs | stages have very different task counts |
| [`wave_split`](#wave_split) | two wave groups per CTA run different stages at the same time | one long stage stalls the stages behind it and they could overlap |

### `staged_all_ctas`

Every resident CTA runs every stage in stage-list order and waits only on the tiles it reads
([tagged hand-offs](handoffs_and_sync.md#tagged-value-tag-hand-offs),
[per-tile hand-offs](handoffs_and_sync.md#per-tile-hand-offs-instead-of-stage-wide-barriers)).

**Fixed structure.**
- One group (`main`) with all waves (`waves: "all"`); `grid.threads` 256 (one wave per SIMD, 512 VGPR per
  lane).
- `grid.n_blocks` <= CUs, all CTAs resident; `sync.grid_barriers` 0 (no grid barriers).
- Every stage has `blocks: "all"` unless a stage is too small to feed all CTAs (then an integer or
  `{"share": f}`; the idle CTAs move on to the next stage and wait there).

**Free parameters.**

| parameter | start | notes |
|---|---|---|
| `n_blocks` | CUs (304 on MI300X) | fewer only if calibration shows the tail stage idles |
| hand-off `unit` | ~1 tile per CTA (draft) | GEMV outputs: one MFMA tile of output channels (e.g. 16); attention: one head |
| `budget.vgpr` | 512 | lower it when a stage is latency-bound and occupancy does not matter (one CTA per CU anyway) |
| `budget.lds_bytes` | 64 KiB | stages run one after another, so each may use all of it |
| `prefetch` | weights of the next GEMV at `kernel_start` | [early-issue prefetch](handoffs_and_sync.md#issue-the-next-stages-loads-before-its-inputs-arrive) |
| XCD order | stage-list order | [XCD placement](#xcd-placement) |

**When to use.** Default for decode layers. A stage whose tiles finish unevenly (attention by context
length) hurts every CTA behind it; if calibration shows a long wait at one hand-off, try
[`wave_split`](#wave_split) before shrinking that stage's blocks.

**Pitfalls.**
- One epoch per step: tag = step × L + layer + 1 (L >= layers), so a wait never matches a stale tile
  ([tagged hand-offs](handoffs_and_sync.md#tagged-value-tag-hand-offs)).
- KV cache pointers must be re-read every replay
  ([serving pitfalls](../../apply_mono_kernel_to_framework/skill.md#knobs--pitfalls)).

### `task_tables`

Every stage is a table of tasks. A stage of `n` tasks puts task `i` on block `(base + i) mod n_blocks`,
and a block loops over its tasks with stride `n_blocks`. A block works through its tasks stage by stage in
stage-list order and a task waits only on earlier stages, so the resident grid cannot deadlock. Bases are
chosen so that stages that can overlap run on different CUs.

**Fixed structure.**
- One group (`main`), `waves: "all"`, `blocks: "all"`; `grid.threads` 512 (8 waves, 256 VGPR per lane).
- Every stage has `tasks: {"n": <int or expression over rows>, "base": <first block>}` and
  `blocks: "all"`.
- `n` may use `rows`, ints, `+ - * //` and parentheses; it must be >= 1 at every width.
- `in_group` hand-offs only between stages with the same task table (task `i` on the same block).
- `sync.grid_barriers` 0.

**Free parameters.**

| parameter | start (draft) | notes |
|---|---|---|
| `tasks.n` | a task per tile of the first output; a task per row for norms | GEMV: output rows / MFMA tile rows; attention: (request, KV head) or (request, key chunk); MoE: tiles per expert |
| `tasks.base` | cumulative sum of earlier stages' tasks, mod `n_blocks` | move a prefetching stage off the blocks that finish its producer last |
| hand-off `unit` | a tile per task | flags per task; consumers that read every tile wait for the whole previous stage |
| `budget.vgpr` | 256 | the cap at 512 threads |
| `budget.lds_bytes` | 64 KiB | stages run one after another in a block: overlay their areas |
| `prefetch` | none | next stage's weights before its wait; inputs whose address is known before the wait |

**When to use.** Decode layers whose stages have very different task counts (a CTA per token for norms,
hundreds of GEMV tiles, a few attention heads): `staged_all_ctas` gives every stage every block, which
leaves most blocks idle in small stages and delays the next stage's prefetch on the busy ones.

**Placement recipe** (the lever behind the `base` values).

*Symptom.* Every CTA runs every stage in order, and a later stage cannot start on a CU until that CU's
earlier tasks finish; or a stage's weight prefetch starts late because its CTAs are still busy with the
previous stage.

1. Choose bases so stages that can overlap use different CTAs: start a stage where the previous stage's
   tasks end (cumulative task counts), so small stages (a task per row for norms) do not all land on the
   first CTAs.
2. Place a stage whose weights are prefetched before its wait away from the CTAs that finish the previous
   stage last. A simple way: score every candidate base by the tasks it puts on "late" CTAs (very high
   cost), on CTAs another concurrent stage needs (medium) and on CTAs that start their prefetch late
   (low); take the cheapest.
3. Map tasks that share a cache line of scales or weights to one XCD (confirm the block → XCD mapping
   first, [XCD placement](#xcd-placement)).

**Pitfalls.**
- Spreading a small, hot read over all blocks makes a hot spot (every block reading the same few KB of
  router logits). A few reader CTAs, one per XCD, writing XCD-local copies is the alternative.
- Placement does not shorten a stage bound by its own dependency chain plus DRAM time; check the stage's
  floor ([measuring.md](measuring.md#floors)) before trying placement variants.
- Index-mapping changes can push a build over the VGPR cap; run the ISA check
  ([FlyDSL idioms and pitfalls](flydsl_idioms_and_pitfalls.md#kernel-pitfalls)).

**Check.** The timeline: each stage's start relative to its producers' end, and the prefetching stage's
first-use gap; then the oracle and the gates.

### `wave_split`

Each CTA has two wave groups that run different stages at the same time: side A streams the big loads
(QKV / attention / gate-up), side B does the short dependent stages (norms, o_proj, router, down,
all-reduce). Draft: measure against [`staged_all_ctas`](#staged_all_ctas) on the same layer.

**Fixed structure.**
- Exactly two groups with disjoint `waves` lists (draft: A = waves 0-3, B = waves 4-7); `grid.threads`
  512.
- Two waves per SIMD, so the per-lane VGPR cap is 256 (256 arch + 256 acc on gfx942).
- LDS per CTA = max over A's stages + max over B's stages <= 64 KiB (draft: 32 KiB each).
- `sync.grid_barriers` 0; hand-offs between A and B in the same CTA still go through mailboxes unless the
  producer is `replicate: true` and in the same group.

**Free parameters.**

| parameter | start | notes |
|---|---|---|
| stage → group | A: kv / attention / qkv stages (draft) | move a stage when calibration shows one group idling |
| `budget.vgpr` | 256 | spills start here: a stage needing more belongs in `staged_all_ctas` |
| `budget.lds_bytes` | 32 KiB per group | shift between groups as measured |
| hand-off `unit` | ~1 tile per CTA | as in `staged_all_ctas` |

**When to use.** When one long stage (attention over long contexts, MoE expert streaming) stalls the
stages behind it and they could overlap. The symptom this layout addresses: with each stage running on
all CTAs one after another, memory-bound stages (norm, small GEMVs, all-reduce polling) and the
latency-bound attention never overlap, and CUs idle at every hand-off. It halves the registers per lane
compared with `staged_all_ctas` and spills start at the 256-VGPR cap, so check the ISA resource report of
a trial build before choosing it; it is a lever, not a requirement.

**Recipe.**
- 512 threads per CTA = side A (4 waves) + side B (4 waves). A takes the long streaming stages (e.g. QKV,
  attention, gate/up); B the short dependent ones (input norm, o_proj, router, down, all-reduce
  reduction).
- While B does the input norm, A issues its QKV weight rows and the first K tile; while A is in
  attention, B loads the o_proj weights; down weights go out together with gate/up.
- For widths > 1, both sides can share the shared-weight GEMVs (QKV, o_proj) so neither side idles.

**Alternative.** 256 threads, every CTA runs every stage in order (`staged_all_ctas`), and the overlap
comes from issuing the next stage's loads early
([early-issue prefetch](handoffs_and_sync.md#issue-the-next-stages-loads-before-its-inputs-arrive)) and
waiting only on consumed shards ([tagged hand-offs](handoffs_and_sync.md#tagged-value-tag-hand-offs)).
It has twice the registers per lane and no cross-side coordination, so the split is a lever, not a
requirement: pick by measuring the stage timeline.

**Pitfalls.**
- The register budget is per side (256 VGPRs per lane with two waves per SIMD); moving more work to a
  side spills. Check spills on every variant.
- Side A and side B communicate only through tagged hand-offs; a CTA-wide barrier stalls both sides.

**Check.** Stage timeline per side ([measuring.md](measuring.md#per-stage-timeline)): the B-side chain
should finish inside A's attention span.

## Rebalance the CU plan

**When.** The timeline shows one stage's span setting the critical path while other partitions finish
early and wait. The CU plan that came with the task was an estimate; the fused kernel may have kept it
as is.

**Recipe.**
1. Decode GEMV stages are bandwidth-bound: their time ≈ bytes ÷ (bandwidth share). Give each such stage
   CUs in proportion to the bytes it reads (its weights for this rank, plus KV for attention at the
   weighted context). With one workgroup per CU, bandwidth share scales roughly with CU share until a
   stage has too few rows/tiles to feed its CUs.
2. Stages that run **concurrently** (independent data, e.g. a prefetch next to attention) split the
   CUs; stages that run **in sequence** (each waits for the previous) can each use all CUs. A pure
   pipeline rarely wins at width 1: every stage depends on the previous one's output, so the usual best
   plan is "all CUs on the current stage", with
   [per-tile hand-offs](handoffs_and_sync.md#per-tile-hand-offs-instead-of-stage-wide-barriers) so the
   next stage starts on the rows that are ready.
3. Tiny stages (RMSNorm, RoPE, residual add; a few µs or less) do not deserve their own partition: run
   them redundantly in every workgroup that needs their output, or fold them into the neighbouring
   stage's prologue/epilogue.
4. Change one share at a time and re-measure the timeline; keep the plan in one table in `mk_<kind>.py`
   (stage → block range) so the timeline and the plan stay in sync.

**Pitfalls.**
- All workgroups must stay co-resident (≤ one persistent workgroup per CU, `PHYSICAL_CU_COUNT`), or the
  grid barriers deadlock until the deadline.
- A different split of a reduction (K split across more workgroups) changes the summation order: still
  within the tolerance normally, but never change the **all-reduce** order (rank order, fp32), which
  keeps ranks bit-identical.
- Every width shares the same compiled structure per build; check the plan for each width in `WIDTHS`.

**Check.** The oracle (correctness and the per-width speedup), then the gates.

## Multi-row decode

2–8 decode rows in the same layer kernel; relevant to every width > 1 in `WIDTHS`. Draft: measure each
design against native at every width.

**When.** The workload runs concurrent decode steps (several requests, or speculative steps), and those
steps set the tail latency while only the 1-row path runs in the kernel. Polishing width 1 does not move
that metric.

**Two designs.**
- **Rows as MFMA B columns.** Shared-weight GEMVs (QKV, o_proj, dense MLP) load each weight tile once for
  all rows; attention by (row, KV head) for sliding-window layers and a share of the CTAs per row for
  full attention; experts by (row, top-k slot, chunk of the intermediate dim) over all CTAs, the next
  task's loads in flight; row vectors read coalesced into LDS. Shares weight reads without extra
  registers per row.
- **Block groups.** The CTAs split into NR groups (rows rounded up to a power of two), each group runs
  the 1-row flow for its row with per-row mailboxes. Simple, but every group re-reads the dense weights,
  so it loses as the row count grows unless those reads hit L2 ([XCD placement](#xcd-placement)).

**Recipe items.**
1. **Long-context attention CTAs by context length** over all CTAs (with a minimum per row), not a fixed
   share per row group: otherwise one long row runs on a few CTAs while the rest wait.
2. **XCD-aware order** for block groups: the same index in every group on the same XCD.
3. **Padded rows return early:** graphs pad to the next captured size; a padded row (e.g. KV slot −1)
   should exit at once.
4. **Pairs of rows can share reductions and barriers** (norm, expert down), with the next pair's weights
   and inputs in flight.
5. A row's output must be bit-identical whatever row index it gets (unittest).

**Pitfalls.**
- Sharing weight reads across rows inside a group design can push past the VGPR cap; the B-column design
  avoids it.
- Next-round expert prefetch is a spill risk at the VGPR cap.
- Measure every width against native at short and long context before routing that width to the kernel.

**Check.** Standalone per-width step time vs native at the same number of concurrent requests and several
contexts; a mixed long + short row case.

## XCD placement

Keep producers and consumers on one XCD.

**When.** Stages are balanced and hand-offs are per tile, but each hand-off still costs more than
expected, or a consumer re-reads data its producer just wrote. MI300X is 8 XCDs of 38 CUs, each with its
own 4 MiB L2; the only device-shared cache is the Infinity Cache
([`hardware/cdna3_mi300/arch.md`](../../../../hardware/cdna3_mi300/arch.md),
[`xcd_chiplet.md`](../../../../hardware/cdna3_mi300/xcd_chiplet.md)). Data written on one XCD and read on
another goes through the Infinity Fabric.

**Recipe.**
1. **Find the mapping first.** The hardware scheduler round-robins workgroups across the 8 XCDs
   ([`xcd_chiplet.md`](../../../../hardware/cdna3_mi300/xcd_chiplet.md)), so consecutive block ids
   usually land on different XCDs. Confirm it on the device: in a probe kernel, record each block's XCC
   id (`s_getreg_b32` of `HW_REG_XCC_ID`) and build the block → XCD table; do not rely on `bid % 8`
   without that check.
2. **Assign blocks to stages by XCD.** When a stage's tile i feeds the next stage's tile i, give both to
   blocks on the same XCD, so the consumer reads from the shared L2. Keep the stage → block table in one
   place ([rebalance](#rebalance-the-cu-plan)).
3. **Spread every stage over all 8 XCDs** (multiples of 8 blocks per stage), so HBM bandwidth and L2
   capacity are used evenly.

**Pitfalls.**
- The mapping is a property of the dispatcher, not a guarantee; a different ROCm or a partition mode
  (CPX/SPX) can change it. Fall back to the plain plan when the probe disagrees.
- Same-XCD placement does not change visibility rules: hand-offs still need release/acquire through
  `mk_runtime`.

**Check.** Timeline (wait times at the hand-offs), the oracle, the gates.

**Variant.** When several block groups read the same weights (one group per decode row,
[multi-row decode](#multi-row-decode)), put the same index of every group on the same XCD so the
duplicate reads hit that XCD's L2. A pure placement change keeps the output bit-identical; check that it
does.

## Sources

- Hardware figures: [`hardware/cdna3_mi300/arch.md`](../../../../hardware/cdna3_mi300/arch.md),
  [`hardware/cdna3_mi300/xcd_chiplet.md`](../../../../hardware/cdna3_mi300/xcd_chiplet.md).
