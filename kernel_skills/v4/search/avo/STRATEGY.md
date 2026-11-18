# Strategy: avo (agentic variation operators)

The agent is the variation operator: few, well-briefed, directed mutations rather than many blind ones.
Good when evals are expensive.

## State used
Core views, roofline gap, past decisions and their outcomes (each sibling's change, result and mechanism).
No strategy state.

## Actions used
mutate, measure, investigate, ship, stop

## Parent selection
Front nodes, weighted toward the one whose diagnosis is freshest (a recent budget or roofline breakdown, with
a known top-gap part).

## Round composition
At most `2 × GPUS` mutations. Each carries:
- `refs`: its 3–5 closest siblings, so the engineer does not repeat them without a new reason;
- a `direction` naming one operator: `targeted` (the top-gap part), `contrastive` (port the mechanism
  that makes node A beat B), `counterfactual` (invert the precondition of a closed lever), `numerics` (use
  unused oracle tolerance), or `structural` (change the work decomposition).
Rotate operators across a parent's children. Screen with the fast probe and run `measure` only on
children that beat their parent beyond noise on the targeted regime.

## Triggers
Stop proposing an operator on a cut after it failed on 3 different parents with agreeing mechanisms.

## Stop
Shell rule only.

## Tags
`operator`: the operator used. `family`: inherited.
