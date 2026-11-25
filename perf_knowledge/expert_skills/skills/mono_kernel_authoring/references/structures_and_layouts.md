# Layer structures and block layouts

Open this for step 3 of the procedure in [`../skill.md`](../skill.md): after the contract, before kernel code.

**Status.** Layer structure `full_one_launch` and block layouts `staged_all_ctas` and `task_tables` were built and
exercised in an internal mono-kernel generation workflow (not public). Layout `wave_split` and the two candidate
structures were designed only. The names are labels for patterns; no template files, layout checker or assembler
exist in this repository. Hardware constants are from the
[MI300X arch card](../../../../hardware/cdna3_mi300/arch.md).

## 1. Two separate choices

- **Layer structure**: how much of the layer the kernel covers, and in how many launches.
- **Block layout**: how the resident workgroups ("blocks", at most one per CU) share the stages inside a launch.

Choose the structure first, then the layout. **Select on structural facts only**: coverage, number of launches,
the roles of the stages, the ratio of stage task counts. Never on a model name, family or architecture string,
and never on a number measured on one model.

## 2. Layer structures

| structure | recognize it by | hand-off | all-reduce | status |
|---|---|---|---|---|
| `full_one_launch` | covers input norm to layer output, attention and FFN/MoE, in one launch | per-tile mailboxes, epoch + deadline | both inside the launch | **exercised**; the only structure the internal flow built and accepted |
| `attn_then_ffn_k2` (partial layer) | the framework still runs attention (or a linear-attention recurrence); the kernel starts at the output projection or post-attention norm; no attention-core stage | the boundary to the framework is a tensor port; mailboxes inside | inside the launch | candidate: designed, no check built, no evidence; specified with `staged_all_ctas` or `task_tables` only |
| `dual_launch_k1k2` | two or three launches: a front launch writes KV (or a key table), a tail launch runs attention core + FFN/MoE + both all-reduces | between launches a device-level release, no tile wait across launches; inside each launch epoch + deadline | in the tail launch | candidate: designed, no check built, no evidence |
| many launches per layer (4 or more) | — | at launch boundaries | — | reference only |
| several layers per launch | — | one grid across layers | per layer | reference only; out of scope |

- "One launch per layer per rank" is a review criterion for a kernel you author to `full_one_launch`. It is not a
  universal law: deployed mono kernels may use two launches per layer (for example the public DeepSeek-V4.1 mono
  kernels; read from an internal index of public kernels, not re-checked here). Optimizing an existing kernel of
  any structure is skill `mono_kernel_optimize`.
- Consider the partial layer when the product path keeps attention in the framework, or the attention backend's
  numerics cannot be matched. Consider the dual launch when the contract is already cut front/tail, when one
  launch cannot fit attention + MoE in the register and LDS budget, or when the KV write and the attention core
  need a device-level barrier. Either way you leave what was exercised: write down the structure id, the facts
  and the risk, and expect to build the checks yourself.
- A candidate becomes a default only after it passes the layout check and calibration on at least two
  configurations with different structures.
- A failed wait in any launch fails the whole step. Inside each launch the block layouts below still apply.

## 3. Block layouts

Common to all three: every block stays resident for the whole launch (at most 304 on MI300X); no grid barrier
inside the layer (the all-reduce brings its own); a port is published in tiles the consumer can start on; a
consumer waits only for the tiles it reads. Every exercised hand-off went through a mailbox; a mailbox-free
hand-off inside a block (the producer computes its whole output on every block, the consumer runs in the same
group on a subset of those blocks) was designed only.

### 3.1 `staged_all_ctas`: every block runs every stage

Every block runs every stage in contract order. **Default for decode layers.**

- Fixed: one group on all waves; 256 threads per block (one wave per SIMD, so 512 VGPRs per wave); every stage
  on all blocks, unless a stage is too small to feed them all (then a count or a share; the idle blocks move on
  to the next stage and wait there).
- Free: blocks (start at the CU count; fewer only if calibration shows the tail stage idling); hand-off unit
  (about one tile per block; GEMV outputs one MFMA tile of output channels, for example 16; attention one head);
  VGPR budget (512, one value for the whole group: see section 6);
  LDS (64 KiB; stages run one after another, so each may use all of it); prefetch (the next GEMV's weights at
  kernel start); XCD order (contract order).
- Pitfall: a stage whose tiles finish unevenly (attention by context length) delays every block behind it. If
  one hand-off shows a long wait, try `wave_split` before shrinking that stage's blocks.
- Status: built; passed the copy-body run (section 8) on three layer kinds; an exercised direct-route kernel
  was written on this mechanism.

### 3.2 `task_tables`: a table of tasks per stage

Task `i` of a stage runs on block `(base + i) mod n_blocks`, then on every `n_blocks`-th. A block works through
its tasks stage by stage in contract order and waits only on earlier stages, so the resident grid cannot
deadlock. Bases are chosen so that stages that can overlap run on different CUs.

- Fixed: one group on all waves and all blocks; 512 threads per block (8 waves, so 256 VGPRs per wave); a task
  count per stage (an integer, or an expression over the row count using `+ - * //`, at least 1 at every width)
  and a base block; a mailbox-free in-block hand-off (designed only) only between stages with the same task table.
- Free: task count (a task per tile of the first output; a task per row for norms; GEMV: output rows ÷ MFMA tile
  rows; attention: per (request, KV head) or (request, key chunk); MoE: tiles per expert); base (cumulative sum
  of the earlier stages' tasks, mod `n_blocks`); hand-off unit (a tile per task; a consumer that reads every
  tile waits for the whole previous stage); VGPR budget 256; LDS 64 KiB with stage areas overlaid.
- Use when stages have very different task counts (a block per token for norms, hundreds of GEMV tiles, a few
  attention heads): `staged_all_ctas` would leave most blocks idle in the small stages and delay the next
  stage's prefetch on the busy ones.
- Pitfalls: spreading a small hot read over all blocks makes a hot spot (every block reading the same router
  logits). A stage that prefetches before its wait must not sit on the blocks that finish the previous stage
  last. Confirm the block → XCD mapping on the device before mapping tasks that share weight lines to one XCD.
  *(Measured: publishing a row-split stage in 16-element tiles from one block, 256 publishes in series, made
  the layout's overhead 3× to 5× higher than one publish per row.)*
- Status: built; passed the copy-body run on three layer kinds.

### 3.3 `wave_split`: two wave groups per block

Each block has two groups of waves that run different stage chains at the same time: group A streams the big
loads (QKV, attention, gate/up), group B runs the short dependent stages (norms, output projection, router,
down projection, all-reduce).

- Fixed: exactly two groups with disjoint wave lists (draft: waves 0–3 and 4–7); 512 threads per block; two waves per
  SIMD, so the per-wave VGPR cap is 256; LDS per block = largest A stage + largest B stage ≤ 64 KiB (start at
  32 KiB each); hand-offs between A and B still go through mailboxes.
- Free: which stage is in which group; LDS split between the groups; hand-off unit.
- Use when one long stage (attention over long contexts, MoE expert streaming) stalls the stages behind it and
  they could overlap. It halves the registers per wave: spills start at the 256 cap, and a stage that needs
  more belongs in `staged_all_ctas`.
- Status: **designed only.** It passes the static rules, but the internal tooling could not assemble it (two
  wave groups in one block need named barriers) and it never ran. Choose it only if you can verify it yourself.

## 4. Selection rules

First match wins *(a draft heuristic of the internal flow; its thresholds were not validated by measurement)*:

1. The largest stage task count is at least **8×** the smallest → `task_tables`.
2. A long producer exists (attention core or MoE) **and** register headroom is known (every stage has a register
   estimate, none above half of that layout's per-wave cap) → `wave_split`. Never exercised.
3. Otherwise → `staged_all_ctas`.

Unresolved conflict in rule 1. The internal selector took every stage of the contract, small ones included, and
counted a stage's tasks as the task count of its tested op code where it had one, otherwise as the element count
of its first output. With the task counts section 3.2 suggests (one task per row for a norm, hundreds for a GEMV)
the ratio reaches 8 for almost any decoder layer, and the declared default would never be chosen. The sources do
not settle this. When in doubt, take `staged_all_ctas`: it was the internal flow's fallback.

If the register need is unknown, do not pick `wave_split`. If the compiled kernel later fails the register check,
fall back to `staged_all_ctas`. A deviation needs a written reason that cites a structural fact or a measurement.
Recognize a stage's role (norm, attention front, attention core, attention out, FFN, MoE) by what it computes, not
by its name. *(Measured, the only comparison of the two built layouts: with copy bodies the overhead floor of
`task_tables` was between equal and 22% higher than that of `staged_all_ctas` on three layer kinds.)*

## 5. Structure fields and parameters

| fixed before stage code is written (structure) | calibrated afterwards (parameters) |
|---|---|
| layout; threads per block and wave division; which stages share a group; the hand-off unit of every port; what is prefetched; the multi-row method; the set of hand-offs and all-reduces | blocks (or share) per stage; numeric VGPR and LDS budgets; order inside a group; when a prefetch is issued; block → XCD arrangement |

A change to a structure field is a new structure version and the affected stages are rewritten. A change to a
parameter needs only a re-check. That split works because a stage must be correct for any number of blocks and
any subset of block ids. For decode widths above 1 two multi-row methods were designed (rows as columns of the
matrix-core operand, or groups of blocks); every exercised kernel had width 1.

## 6. Drafting and calibrating the numbers

Draft, before any stage exists:

- Hand-off unit: along the first non-width dimension, the smallest power of two that divides it and is at least
  dimension ÷ blocks. The producer's natural unit wins over the consumer's.
- Blocks: the CU count. Budgets: the device caps. They are upper limits, deliberately loose; calibration takes
  back what is unused. With an estimate: VGPR = estimate × 1.25 rounded up to a multiple of 8 (at least 32),
  LDS = estimate × 1.25, not above the group's share.

Calibrate, once the stages exist *(implemented as scripts; exercised on a pilot of three stages)*:

- Time each stage alone at ½×, 1× and 2× its block count (¼× when 2× exceeds the grid), per width. Read VGPR,
  LDS and register spill from the compiled ISA.
- Blocks = the smallest count within 5% of that stage's fastest time (`task_tables` stages are left unchanged).
- VGPR budget = measured + 10%, rounded up to 8, not above the per-wave cap. **Every stage of a group gets the
  group's maximum**: the fused kernel's VGPR count is the maximum over its stages, so a lower budget for one
  stage buys no occupancy and only rejects a later fix.
- LDS budget = measured + 10%, plus the group's share of what the block has left (one group: all of it).
- Report instead of adjusting: a stage over the hardware limit or with register spill goes back to its author
  with the excess stated; a stage that takes more than half of the summed stage time at some width marks the
  structure as needing revision.
- The structure fields must come out identical; otherwise it is not a calibration.

Design only, never run: tuning the order inside a group, the prefetch issue point and the block → XCD arrangement;
re-calibration when optimization plateaus; structure revision; merging or splitting stages during optimization.
*(Measured on the pilot: the calibrated block counts raised the copy-body overhead floor by about 15%; the effect
on real speed was not measured.)*

## 7. Static rules of a layout

1. **Placement.** Every stage is in exactly one group. Groups use disjoint wave sets or all use every wave. A
   stage's blocks lie inside its group's range.
2. **Coverage.** Every stage input is bound exactly once, to a layer input or to one hand-off. Every output is
   handed off or is a layer output, never both (one buffer per port), and has at most one hand-off.
3. **Order.** Producers come before consumers in contract order; no stage consumes itself.
4. **Units.** A unit's dimension is in the output's shape and its size divides that dimension (for the width
   dimension: every width).
5. **All-reduce.** The stages that reduce in the kernel are exactly the stages with a collective.
6. **Widths.** A subset of the contract's feasible widths; only width 1 without a multi-row method.
7. **Blocks.** At most one per CU; a stage has no more blocks than its group.
8. **Registers.** Waves per SIMD = ceil(threads ÷ 64 ÷ 4); per-wave VGPR cap = 512 ÷ that (512 at 256 threads,
   256 at 512 threads). No stage budget above the cap. A deliberately lower kernel cap needs a stated reason
   (which occupancy it buys).
9. **LDS.** Stages on one wave set run one after another: take the maximum. Groups on different wave sets run
   together: add. LDS that stays live across stages (a prefetch into LDS) cannot be reused: add it on top. The
   total stays within 64 KiB. Use one LDS arena per kernel, reused from offset 0 by each stage, with a block
   barrier between a stage that used LDS and the next.
10. **Mailboxes.** The tiles of all mailbox hand-offs sum to at most 8192 (a convention: a 64 KiB control
    region of 8-byte mailboxes). At width above 1 a tile is still one mailbox.
11. **Prefetch.** "Issue after stage X" requires X to run earlier in the same block.
12. **Barriers.** No grid barrier inside the layer.
13. **Contract hash.** The layout records the hash of the stage contract; a changed contract voids it.

## 8. Verify the layout before any stage exists: the copy-body run

*(Implemented and exercised: three layer kinds × two layouts passed on four ranks.)* Build the layer kernel with a
**copy body** in every stage slot. A copy body:

1. waits for each input tile and compares what it read, bit for bit, with its own reference input (counting
   mismatches: this finds a read before the wait, a wrong tile, a wrong binding);
2. copies each output's reference value into its buffer, tile by tile: for a hand-off the first reader's
   reference input, for a layer output the producer's reference in the deployed dtype;
3. publishes each tile.

Before the run, check the references: all readers of one value must have bit-identical reference inputs, and the
value must agree with its producer's reference within the producer's tolerance *(exercised: this found wrong port
bindings in two of three layer kinds)*. For a stage with a collective, rank 0 writes the reference as its partial
and the other ranks write `-0.0`. `x + (-0.0) = x` for every `x`, so the sum stays bit-identical while the partial
buffers, both barriers and the rank-order summation really run. Pass criteria, per width and captured step:

| check | requirement | catches |
|---|---|---|
| buffers | every hand-off buffer and layer output is bit-identical to the reference value copied into it | hand-off, tag, tile mapping, all-reduce order |
| input compare | every mismatch counter is 0 | read before wait, stale data |
| error word | 0 on every rank | timeouts, peer errors |
| dropped publish | removing one tile's publish (one per hand-off) ends in a timeout error | a wait that does not really wait |
| replay | three graph replays agree and pass, with every intermediate buffer poisoned (NaN or 0xFF) before each | a tag passed from the host; a buffer that must be zeroed |

The device time of this run is the layout's **overhead floor**: what synchronization and hand-off cost with no
compute. Record it; no threshold was set. *(Measured: cheap hand-offs cut it about 13×, see
[`sync_patterns.md`](sync_patterns.md).)*

Limits: copy bodies do not reproduce the timing of real stages, so races can still appear later; the layer
oracle and the robustness checks stay the verdict. Even a run with real stages in some slots did not detect a
hand-off read through a plain load *(exercised: one was injected and went unreported)*; only the ISA check covers
it, coarsely.
