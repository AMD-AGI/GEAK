# Structure levers: how many rows, who runs what, and where

Structural changes (which steps the kernel covers, how stages overlap, how many rows it takes) tend to
pay more than per-stage polish, and cost more rounds. Each lever lists the symptom, the recipe, how to
tell it is already applied, pitfalls, the check and the evidence. "Measured" figures come from
[measured_rounds.md](measured_rounds.md); everything else is a direction to measure.

Order of attack that usually pays on a fresh seed: rebalance → hand-offs → all-reduce → per-stage
kernels (weights, attention) → placement. On a deployed kernel most of that is done: inventory first.

## 1. Cover every width the cases use (several decode rows per step)

**When.** The unit test's cases (or the serving trace behind their weights) contain steps with more
than one decode row — several requests, speculative or multi-token steps — and those widths run slower
per row than width 1. Polishing width 1 does not move them.

**Two designs.**
- *Rows as MFMA columns.* Shared-weight GEMVs (QKV, o_proj, dense MLP) load each weight tile once for
  all rows; attention by (row, KV head) for sliding-window layers and a share of the workgroups per row
  for full attention; experts by (row, top-k slot, chunk of the intermediate dim) over all workgroups
  with the next task's loads in flight; row vectors read coalesced into LDS. Shares weight reads
  without extra registers per row.
- *Workgroup groups.* The grid splits into groups (rows rounded up to a power of two); each group runs
  the 1-row flow for its row with per-row hand-off slots. Simple, but every group re-reads the dense
  weights, so it loses as rows grow unless those reads hit the per-XCD L2 (lever 5).

**Recipe items.**
1. Give long-context attention workgroups by context length over the whole grid (with a minimum per
   row), not a fixed share per row group — otherwise one long row runs on a few workgroups while the
   rest wait.
2. XCD-aware order for groups: the same index of every group on the same XCD.
3. Padded rows return early: graphs pad to the next captured size; a padded row (for example KV slot
   −1) exits at once.
4. Pairs of rows can share reductions and barriers (norm, expert down) with the next pair's weights and
   inputs in flight.
5. A row's output must be bit-identical whatever row index it gets.

**Already applied if** stage functions take a row count or come in a multi-row family, and the cases
with width > 1 run the kernel in the candidate leg.

**Pitfalls.** Sharing weight reads across rows in a group design can exceed the VGPR cap (the column
design avoids it). Next-round expert prefetch is a spill risk at the cap. Measure every width at short
and long context. Never send a width that is in the cases to the native path to hide a slow width.

**Check.** Per-width time against the baseline at the same number of rows and several contexts; a mixed
long + short row case; row-index independence.

**Evidence (measured, serving P90 at concurrency 1).** The largest single step in both HIP designs:
2–8 rows in one round −59%. In the other design rows arrived inside bundled rounds, so their figures are
not rows alone: 2 rows + early V issue −20%; 1 / 2 / 4 rows + in-register FP4 decode −14%; 8 rows +
scales in read order + XCD order −12% (the source credits this one to the scale layout). Attention
workgroups by context length: −6% against the previous round, −1.5% against the best earlier one. Pairs
of rows sharing norm / down reductions: −1.5%. The 1-row path had plateaued at about 0.66–0.70 of native
latency. A lane's own metric moves only for widths present in its cases: read the case weights.

## 2. Rebalance the CU plan

**When.** One stage's span sets the critical path while other partitions finish early and wait. A plan
that was an estimate is often kept unchanged when stages are first fused.

**Recipe.**
1. Decode GEMV stages are bandwidth-bound: time ≈ bytes ÷ bandwidth share. Give each such stage CUs in
   proportion to the bytes it reads (its weights for this rank, plus KV for attention at the weighted
   context). With one workgroup per CU the share scales roughly with CUs until a stage has too few rows
   or tiles to feed them.
2. Stages that run concurrently (independent data, such as a prefetch next to attention) split the CUs;
   stages in sequence can each use all CUs. A pure pipeline rarely wins at width 1: the usual best plan
   is "all CUs on the current stage" with per-tile hand-offs, so the next stage starts on the rows that
   are ready.
3. Tiny stages (RMSNorm, RoPE, residual add) do not deserve a partition: run them redundantly in every
   workgroup that needs the output, or fold them into a neighbour's prologue / epilogue — after checking
   that the fold does not add reads (see the evidence).
4. Change one share at a time and re-measure; keep stage → workgroup range in one table so plan and
   timeline stay in sync.

**Already applied if** every workgroup takes a fixed share of each stage from one table and no stage
owns a private partition that idles.

**Pitfalls.** All workgroups must stay co-resident (at most one persistent workgroup per CU), or
device-wide waits stall until the deadline. Splitting a reduction across more workgroups changes the
summation order: normally within tolerance, but never the all-reduce order. One build serves every
width: check the plan per width.

**Check.** Correctness and the per-width speedup; the timeline after each single change.

**Evidence.** Not measured in isolation. Measured neighbours: moving o_proj off the workgroups that
finish attention last was one item of a round worth 1.19× (weighted); folding the input norm into the
QKV workgroups was a loss twice (+14% time in one attempt, the QKV stage finishing about 20% later in
a second) because the norm was already hidden behind the QKV weight stream and every QKV workgroup had
to re-read the hidden state and residual; fusing the expert down projection with the final sum was a
win (−36% of those stages).

## 3. Per-stage task tables with staggered bases

**When.** Every workgroup runs every stage in order and a later stage cannot start on a CU until that
CU's earlier tasks finish; or a stage's weight prefetch starts late because its workgroups are busy.

**Recipe.**
1. A stage = n tasks; task i on workgroup `(base + i) mod grid`, each workgroup looping over its tasks
   with stride grid. A task waits only on earlier stages, so the resident grid cannot deadlock.
2. Choose bases so stages that can overlap use different workgroups: start a stage where the previous
   stage's tasks end (cumulative counts), so small stages do not all land on the first workgroups.
3. Keep a stage that prefetches before its wait away from the workgroups that finish the previous stage
   last. A simple scoring: tasks on "late" workgroups cost most, tasks on workgroups a concurrent stage
   needs cost less, tasks on workgroups that start their prefetch late cost least; take the cheapest
   base.
4. Put tasks that share a cache line of scales or weights on one XCD (lever 5).

Starting points: a task per tile of the stage's first output; a task per row for norms; attention by
(request, KV head) or (request, key chunk); MoE by tiles per expert; one flag per task.

**Already applied if** the code maps tasks with `(base + i) mod grid` and per-stage base constants.

**Pitfalls.** Spreading a small hot read over all workgroups makes a hot spot (every workgroup reading
the same few KB of router logits); a few readers, one per XCD, writing XCD-local copies, are the
alternative. Placement does not
shorten a stage bound by its own dependency chain plus memory time: check its floor first. Index-mapping
changes can push a build over the VGPR cap: check the ISA.

**Check.** Timeline: each stage's start relative to its producers' end; the prefetching stage's
first-use gap.

**Evidence (measured, all inside an existing task-table design).** Moving top-k into every workgroup made
that stage 2.2× slower (a hot spot in the device-shared cache on a few KB of logits); 8 readers per
token, one per XCD, with wave-level top-k cut it by 45%. Splitting the attention combine 4× over more workgroups made the following o_proj
2.3× slower (it collided with o_proj's placement and quadrupled the flags). Placement variants of the
expert down stage left its tail unchanged.

## 4. Two wave groups per workgroup

**When.** Stages run on all workgroups one after another, so memory-bound stages (norm, small GEMVs,
all-reduce polling) and latency-bound attention never overlap. Needs known register headroom.

**Recipe.** 512 threads = side A (4 waves) + side B (4 waves). A takes the long streams (QKV, attention,
gate/up); B the short dependent stages (input norm, o_proj, router, down, the all-reduce sum). While B
does the input norm, A issues its QKV weights and the first K tile; while A is in attention, B loads the
o_proj weights; down weights go out with gate/up. For widths > 1 both sides can share the shared-weight
GEMVs so neither idles.

**Alternative.** 256 threads, every workgroup runs every stage, overlap from early issue and from
waiting only on consumed shards: twice the registers per wave and no cross-side coordination.

**Already applied if** the code branches on the wave index and splits the LDS between two sides.

**Pitfalls.** The register budget is per side (256 VGPRs per wave with two waves per SIMD): moving work
to a side spills; check every variant. The sides communicate only through tagged hand-offs; a
workgroup-wide barrier stalls both. LDS: the largest A stage plus the largest B stage must fit 64 KiB.

**Check.** Timeline per side: the B chain should finish inside A's attention span.

**Evidence (measured comparison, not an isolated A/B).** Of two independent HIP designs, the 256-thread
fixed-order one reached the better serving result in fewer rounds (7 against 13) than the two-sided
one. Converting a working kernel to two wave groups is unproven and expensive: not a one-round job.

## 5. Keep producers and consumers on one XCD

**When.** Stages are balanced and hand-offs are per tile, yet each hand-off costs more than expected, or
a consumer re-reads what its producer just wrote. MI300X is 8 XCDs of 38 CUs, each with its own 4 MiB
L2; the only device-shared cache is the Infinity Cache
([xcd_chiplet.md](../../../../hardware/cdna3_mi300/xcd_chiplet.md)).

**Recipe.**
1. Find the mapping first. The hardware scheduler round-robins workgroups over the XCDs, so consecutive
   ids usually land on different XCDs. Confirm on the device: a probe kernel records each workgroup's
   XCD id (a hardware register read) and builds the table; do not trust `id mod 8` unchecked.
2. When a stage's tile i feeds the next stage's tile i, give both to workgroups on the same XCD.
3. Spread every stage over all XCDs (multiples of 8 workgroups per stage), so memory bandwidth and L2
   capacity are used evenly; data written on one XCD and read on another crosses the fabric.
4. When several groups read the same weights (a group per row), put the same index of every group on
   the same XCD so the duplicate reads hit its L2.

**Already applied if** a tile map or task order is keyed by the workgroup's XCD, or loops say "one
reader per XCD".

**Pitfalls.** The mapping belongs to the dispatcher, not to a guarantee: a different ROCm version or
partition mode can change it; fall back to the plain plan when the probe disagrees. Placement does not
change visibility: hand-offs still need release / acquire.

**Check.** Wait times at the hand-offs; a pure placement change must keep the output bit-identical.

**Evidence (measured, combined with other items).** XCD-aware tile map + cached block scales + lane
redistribution: −23% of the expert gate/up stage. Scales in read order + XCD-aware order + padded-row
return + 8 rows: −12% serving P90.

## 6. Launch structure

- Never add a launch and never split one (universal rule, [acceptance.md](acceptance.md)).
- What a single launch already banked (*measured* on the native path of the reference model, two
  different runs): the native layer was about 16–17 device kernels; only about three quarters of a
  native decode step was kernel time — the rest was gaps between small dependent kernels.
- Merging a front launch into the tail launch is legal but is a structural project: the merged body
  needs the registers and LDS of both, and the KV write must be visible before any read of the last
  block. Plan it only with several rounds left, in steps that each pass the oracle.
