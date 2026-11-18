# Search template: avo (agentic variation operators)

The agent itself is the variation operator. Each mutation reads the archive and the failure record, and
proposes a *directed* change, not a random perturbation. The loop is: select → diagnose → propose varied
edits → evaluate → keep the archive. Use it when evals are expensive and a few well-reasoned candidates
beat many blind ones.

## Selection
- Parents come from the front (as in `evo`), weighted toward the nodes whose **diagnosis is freshest**:
  a node with a recent `budget` / roofline breakdown and a known top-gap part beats a node no one has
  analysed.
- Every mutation gets a **failure brief** built by the lead from the archive: the 3–5 closest siblings
  (same parent or same cut), what each changed, its per-regime result, and the mechanism recorded for
  why it tied or lost. The engineer must not repeat a sibling's change without a new reason.

## Variation operators (name the one used in `direction`)
- **targeted**: attack the top-gap part of the parent's time budget for this cut.
- **contrastive**: take the mechanism that makes node A beat node B on a regime, and apply it to B's family.
- **counterfactual**: invert one assumption a closed lever rested on (`LEARNINGS.md` closure → precondition),
  and test the cheapest consequence first.
- **numerics**: use oracle tolerance that the parent leaves unused (requires a measured tolerance).
- **structural**: change the work decomposition (grid, launches, ownership of data), a `fresh` in miniature.
Rotate operators across a parent's children: no parent gets the same operator twice in a row.

## Evaluation budget
- Engineers screen with the validated fast probe and report the per-regime vector. Only children that
  beat the parent beyond noise on the targeted regime get an official run.
- At most `2 × GPUS` children per round. The lead prefers fewer, better-briefed mutations to more of them.

## Archive and stop
- Keep every child, including losers. Losers carry the mechanisms the next failure briefs need.
- Stop under the lead's end rule. In addition, stop proposing an operator on a cut after it failed on
  3 different parents with mechanisms that agree.
