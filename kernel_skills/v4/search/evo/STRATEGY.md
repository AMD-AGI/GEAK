# Strategy: evo (islands + front tournament)

Keep several design families alive. Pick parents by where the headroom is, not by the overall best.

## State used
Core views, roofline gap per regime, and `RUN/search/evo/state.json` from `policy.py`: islands (one per
`family` tag) with their best score, rounds since their last gain, and front membership; plus tournament
proposals.

## Actions used
mutate, fresh, combine, measure, investigate, ship, stop

## Parent selection
1. Take the regime with the largest weighted gap to the cap (roofline). Spread a round's mutations over
   different regimes, unless one regime holds > 60% of the gap.
2. Run a tournament among the front nodes on that regime's score; break ties toward fewer children. The
   `policy.py` proposals do this; you may override them with a reason.
3. The parent may come from any live island. Say why when it is not the shipped node's island.

## Round composition
About 60% `mutate`, 20% `combine` (`merge` for disjoint code, `dispatch` when the band composite beats the
best node by more than NOISE), 20% exploration (`fresh` or `investigate`). With one GPU, alternate rounds
rather than splitting one. Add a `measure` when two front nodes are within noise on the regime you
picked.

## Triggers
- `fresh` (a new island): one island has produced every front node for 2 rounds and the gap is ≥ 5%.
- Retire an island after 2 rounds with no gain and no front node. Write its lesson to `LEARNINGS.md`; its
  nodes stay selectable.
- Migration: a `mutate` on a lagging island may take another island's winning mechanism as its direction.

## Stop
Shell rule, plus: no live island has an open question that could move the front.

## Tags
`family` (= island): the parent's value, or a new one for `fresh`. At most `min(GPUS + 1, 4)` live islands.
