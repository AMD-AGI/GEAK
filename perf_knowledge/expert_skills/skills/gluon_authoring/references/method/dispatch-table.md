# Shipping a per-shape dispatch table

Per-shape specialization is a routine outcome, not an exotic one: one tile, one pipeline
depth or one whole kernel wins on part of the served range and loses on the rest, so what
ships is a **table** — a predicate on the incoming shape selecting an implementation — with
a default. This page is the interface that table has to satisfy before it counts as a
result. It is not advice about *whether* to specialize; the bound and the budget decide
that. It is about what a shipped table owes a reader.

A dispatch table is the one artifact in a run that **outlives the run and is read by people
who cannot re-measure it**. That is what makes an unmarked inferred row expensive: it is
indistinguishable from a measured one, it is trusted for as long as the table lives, and
the first thing that disproves it is a production regression.

## The four obligations

### 1. Every row says whether it was measured or inferred, at a glance

Not in a footnote, not in the commit message, not derivable by comparing the row list to a
results file. **In the row.** A reader scanning the table must be able to see, without
reading prose, which rows rest on a timing taken on this box and which rest on an argument.

```text
shape predicate                | impl          | basis    | evidence
-------------------------------|---------------|----------|-------------------------------
m <= 64, k % 64 == 0           | decode_v2     | MEASURED | round 7, A/B ..., null checked
64 < m <= 512                  | tiled_128     | MEASURED | round 9, A/B ...
m > 512                        | tiled_128     | INFERRED | rule R1 below
any, k % 64 != 0               | generic       | DEFAULT  | never timed; correctness only
```

Three values, and they are not a spectrum: `MEASURED` (this predicate's representative
shape was timed, interleaved, against the incumbent, at the deployment boundary),
`INFERRED` (the choice follows from a stated derivation over measured rows), `DEFAULT`
(nothing was claimed; this is the fall-through). Anything you cannot honestly call
`MEASURED` is one of the other two. A table whose rows are all `MEASURED` because the
distinction was never drawn is the failure this obligation exists to prevent.

### 2. Every inferred row carries its derivation, as a rule and not as a vibe

An inferred row is admissible — you cannot time a continuum — but it has to be inferred
*from something*, and the something has to be written next to the table so the next reader
can attack it:

- **which measured rows it interpolates or extrapolates from**, by name;
- **the monotonicity or mechanism it assumes** — "above the measured band this kernel is
  bandwidth-bound and the tile choice stops mattering", "the winner flips at the ridge and
  every shape above it is on the same side";
- **the direction of the risk** — which way the row is wrong if the assumption fails, and
  whether that direction is a slowdown or a correctness event;
- **what would falsify it**, concretely enough to be run later without re-deriving the
  argument.

Extrapolation past the largest and below the smallest measured shape is the fragile kind:
say so on those rows specifically rather than treating the whole inferred class as one
risk.

### 3. Compile-and-numerics gate on **every** row, including the untimed ones

Timing is optional per row. Building and being correct is not. A row that was never timed
is still a row the dispatcher can select in production, so:

- every predicate has at least one representative shape that is **compiled** and run
  against the **correctness oracle**, including `INFERRED` and `DEFAULT` rows;
- the gate runs over the table as a whole, in CI or in the close audit, not once by hand
  during the round that wrote the row;
- a row that cannot be compiled or cannot pass the oracle is **removed from the table**,
  not shipped with a caveat — the dispatcher has no way to honour a caveat;
- shape-dependent constraints (alignment, a `k` divisibility a tile requires, a minimum
  sequence length) belong in the **predicate**, so the untestable combination never
  reaches the specialized arm in the first place.

The asymmetry is the point: an untimed row costs you performance you did not know you
were missing, while an unbuilt or numerically wrong row costs a caller.

### 4. An explicit boundary policy

The table has edges, and every edge is a decision someone has to be able to find:

- **What happens outside the measured range** — fall through to the default, extrapolate
  the nearest rule, or refuse. Pick one and write it down. "Extrapolate" is a legitimate
  answer and an `INFERRED` row is how it is expressed; "nobody thought about it" is not.
- **Where the boundaries sit, and why there.** A boundary placed at a measured crossover
  is a finding; a boundary placed at a round number is a convenience, and the two should
  not look alike in the table.
- **What happens exactly on a boundary.** Half-open intervals throughout, stated once, so
  there is no shape that matches two rows or none.
- **Hysteresis or the lack of it**, where the selector runs per call on a drifting shape.
- **Who owns re-validation**, and on what event — a toolchain bump, a new SKU, a changed
  served mix. A table is pinned to the box and build it was measured on; that pin is part
  of the table, not metadata about it.

## Before you read a table's numbers as evidence

Two hazards on this page's neighbours will manufacture a per-shape table out of nothing:

- a **fixed-graph A/B** gives each arm one allocator placement and therefore a persistent,
  one-directional per-arm bias, which reads as a stable per-shape pattern
  (`benchmark-hygiene.md ## Measurement basis + benchmark-artifact pitfalls`). Run the
  identical-arm null first; a table built on top of an un-nulled harness is a picture of
  the harness.
- a win measured at an isolated boundary can be negative at the shipping boundary — the
  **boundary-local win**, same page. Every `MEASURED` row means measured at the deployment
  boundary, or it does not mean `MEASURED`.

And weigh the table by the **served** mix rather than by per-shape microseconds
(`../../scripts/served_envelope.py`): a table that wins on most rows and loses on the rows that
carry the GPU-seconds is a loss.

## Recording it

The table is part of the result, so it travels with the result rather than only with the
source: name it in `caveats[]` / `deferred[]` where rows are inferred, and keep the per-row
evidence refs resolvable from the close audit. A shipped table with no `INFERRED` rows and
no boundary policy written down has usually not been examined for either — that absence is
itself worth a challenge at the close.

## Sources

Adapted from `TileProgrammingAgentSkills@a649eaaf`, `.claude/skills/tile-programming-gluon/references/dispatch-table.md`.
