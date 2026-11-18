# Search template: evo (GEAK-evo, PR #580, as a skill)

Retained-candidate search. Every attempt with source stays in the archive: correct, slower, incorrect,
unverified. Each direction starts from a **selected parent** with optional **donors**. Alternatives get
**sustained development** next to the incumbent. Independent **lineages** can exchange checkpoints. The best
validated candidate is tracked separately: retention never promotes, and the v4 ship rule still decides.

## Start (`EVOLUTION_START`)
- `first_stall` (default): run `hillclimb` until the first round without qualifying progress, then switch.
  The archive keeps everything from round 0, so earlier attempts are already selectable.
- `search_start`: evo from round 1. Required when this run is one lineage of a population.

## Each direction = parent + donors + approach
In `decisions/round_<n>.json`, every `mutate` carries:
- `parent_id`: any archive entry with retained source, or `ORIGINAL`. Choose the source that implements the
  starting point this direction needs, not the best score. A slower node is a fine parent when its
  computational organization is the one to develop, or when it wins the cases this direction targets.
- `donor_ids`: optional entries whose ideas or source this direction should combine (crossover). For each
  donor, name the contribution to carry over. The engineer must check that it is present in the executed
  code before interpreting the result.
- `approach`: a stable label for the computational organization. Keep it while the organization is
  unchanged; a new label means a distinct approach, a claim to be checked, not proven by the label.
- `question`: the unresolved question and the next experiment whose outcome would change the decision.

## Breadth vs depth
- Every round: continued exploitation of the best approach plus development of the alternatives whose
  unresolved question justifies the work. Do not abandon a formulation because its first implementation was
  slow: ask for the best correct realization (`RETAIN=development`) and judge it on its own measurements.
- Size the fan-out to the GPU pool and the eval cost (v4 lead "Each round"). One deep / fresh-arch direction
  takes a dedicated round.
- Read the `archive_views.md` front and the per-regime gaps to the roofline cap. An alternative that leads a
  regime with a large gap is a strong parent even if it trails overall.

## Ledger discipline (in `ledger.md` / `SYNTHESIS.md`)
- Treat the latest revision of an approach as a continuation, not a new approach.
- Classify each outcome: unrealized intervention, incorrect source, correct but slower realization,
  missing or inconclusive measurement, or improvement.
- Never close a whole approach because one implementation failed. Mark contributions already incorporated
  elsewhere as subsumed. Reopen a conclusion when its premise or applicability changes.

## Planner view
Inline: the latest 12 archive entries plus the best entry (`archive.py views`). Search the full
`RUN/archive.json` and the referenced evidence for older approaches before choosing parents.
`RUN/evolution/` contains `base/` (the fixed start), `patches/<id>.diff` (cumulative against that base) and
`archive.json` (schema `geak.evolution/1`), kept current by `archive.py build`.

## Population (optional; `LINEAGE`, `POP_ROOT`, `PEERS`)
- Run several leads as independent lineages, each with its own RUN, GPUs, budget and incumbent (e.g. one
  lineage per GPU pair).
- After each ship, publish the committed checkpoint:
  `python3 $SK/bin/population.py publish --run $RUN --root $POP_ROOT --lineage $LINEAGE --id <shipped>`.
- At each round start, import from the named peers:
  `python3 $SK/bin/population.py import --run $RUN --root $POP_ROOT --lineage $LINEAGE --peers $PEERS`.
  Only checkpoints with the same base hash and metric are imported.
- Imported nodes (`peer_<B>_r<N>`) are parents and donors only. Before any decision, measure them locally
  (geak-benchmark), then run the geak-ship audit. They never replace the incumbent on peer evidence.

## Stop
The v4 lead's end rule, plus: no alternative's unresolved question remains whose answer could change the
front.
