# Search template: evo (islands + quality-diversity + crossover)

**State used:** the archive (`archive_views.md`): best, the noise-aware front, the dispatch composite,
families, and the roofline gap per regime.

## Islands
An island is one design family (`family` tag). Keep at most `min(GPUS + 1, 4)` islands alive:
- the shipped node's family (always);
- the family whose best node leads at least one regime on the front;
- a family seeded by `fresh` when the trigger below fires.
An island with no gain beyond noise for 2 rounds and no front membership is retired. Write its lesson
and mechanism to `LEARNINGS.md`; its nodes stay in the archive as possible crossover parents.

## Parent selection (per mutation)
1. Pick the regime / size band holding the largest **weighted gap to the cap** (roofline). Two actions in
   the same round should pick different regimes or bands, unless one regime holds > 60% of the gap.
2. Among front nodes, take the one best on that regime (tournament over front nodes: compare the
   regime score, and break ties toward fewer descendants). Do **not** default to the overall best.
3. If the chosen node is from a different island than the shipped node, say why (it is ahead on that
   regime, or its design still has an open door there).

## Mix per round (scale to `GPUS` and the eval cost)
- **~60% mutations** on selected parents (one cut each).
- **~20% crossover:** `merge` when parents changed disjoint code, or `dispatch` when the dispatch composite
  beats the best node by more than `NOISE`. Dispatch conditions use regimes or size bands only.
- **~20% exploration:** `fresh` (new island) or an `investigate` that can reopen a closed door
  (`ceiling`, `prior-art`). With one GPU, alternate rounds instead of splitting one.

## Triggers
- **fresh:** one island has produced every front node for 2 rounds and the weighted gap is ≥ 5%; or a
  regime's gap is ≥ 10% while its best node is a descendant of the shipped line.
- **migration:** when an island's best regime score beats another island's on a regime, the next
  mutation on the lagging island may take that node's mechanism as its direction ("port X from c123").
- **restart an island** from an older front node if its last 2 rounds of descendants all lost on the
  regime they targeted.

## Bookkeeping
Each round's decision file lists the island and the regime of every action. At round end, record per
action: did its child join the front, did it ship. Every 3 rounds, tell `SYNTHESIS.md` which action types
paid (mutate / crossover / fresh / investigate), and shift the mix toward them.

**Evidence that tempers this template** (`../EVIDENCE.md`): over 4 kernels of the EAT run, recombining
already-measured nodes by per-regime dispatch would have added only 0–0.6% (per-case: 0.2–1.6%, partly
noise). Most value is expected from diverse parents and new islands, not from mixing old ones.
