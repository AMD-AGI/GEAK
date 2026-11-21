# Strategy: hillclimb

Climb from the shipped node; merge the round's winners. The v3 behaviour, kept as the control arm.

## State used
Core views (best, per-regime bests), roofline gap per regime, `LEARNINGS.md` closures. No strategy state.

## Actions used
mutate, fresh, combine, investigate, ship, stop

## Parent selection
`parent` = the shipped node. An older node only when the shipped line closed a door it still has open (say
which in `why`).

## Round composition
One `mutate` per cut holding a large weighted gap to the cap, at most 2–3 cuts per round (scaled to the
GPUs). Cut by regime, size band or cost component. Plus one `combine kind=merge` of the round's winners on
disjoint cuts, and `ship`.

## Triggers
- `fresh`: after a 2-round plateau with one design family and a weighted gap ≥ 5%; at once if less than half
  the time is left.
- `investigate`: on a declared ceiling (`ceiling`), flipping verdicts (`instrument`), or ideas only from inside
  the team (`prior-art`).

## Stop
Shell rule only.

## Tags
`family`: the parent's value, or a new one for `fresh`.
