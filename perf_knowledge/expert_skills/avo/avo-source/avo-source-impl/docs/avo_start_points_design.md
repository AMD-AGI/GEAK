# AVO Start-Point Selection Design (avo-workflow)

Where each optimization step **begins** — always the single best, or a diversified
set of earlier/alternative versions — and why that choice matters for escaping local
optima on the road to the roofline. Covers the `best | topk | diverse | elites`
policies, when they apply, and the safety invariant that keeps diversification from
ever regressing the result.

> Companion: [`avo_on_team.md`](avo_on_team.md) (the fan-out operator that consumes
> these starts), [`avo_roofline_design.md`](avo_roofline_design.md) (the `regime`
> dimension `elites` buckets on). Pure logic: `lib/start_select.mjs` + `lib/archive.mjs`
> (unit-tested). Source: `GEAK/docs/developer/avo_parallel_start_points_design.md`.

---

## 1. The problem: a single chain collapses to ONE local optimum

Faithful AVO resets every step to the **global best** and edits from there, so the
lineage is a chain `avo-v0 → v1 → v2 → …`. That is the cleanest single-lineage depth
search, but it has a structural weakness: **once a branch becomes the best, the whole
search is trapped in that branch's family.** A different version that was briefly
*behind* — a different tiling, a different regime — but had **more headroom** is
abandoned, even if branching off it would eventually overtake the current best.

Start-point selection fixes this by letting fan-out steps begin from a **diversified
frontier** while never giving up the exploit of the current best.

## 2. Exploit + explore (the global best is always a start)

Every policy returns a list of start refs with the **global best first** (exploit),
the rest chosen for exploration. So a fan-out step always (a) keeps pushing the known
best, and (b) probes alternatives in parallel — one step, both behaviors.

## 3. The four policies (`start_policy`)

`selectStartPoints(committed, {policy, k})` (`lib/start_select.mjs`):

| `start_policy` | Start set | Draws from | Lineage shape |
|----------------|-----------|------------|---------------|
| **`best`** | the single global best | — | **chain** (faithful) |
| **`topk`** | top-k committed versions by speedup | raw speedup ranking | tree |
| **`diverse`** (default) | best version **per strategy bucket** | 1-D strategy cells | tree |
| **`elites`** | best **per (strategy × regime) cell** (MAP-Elites) | `lib/archive.mjs` `selectElites` | tree |

- `best` and `k=1` always collapse to `[best]`.
- The baseline `v0` is excluded from exploration starts (unless it is itself the best).
- `elites` is the richest: a 2-D quality-diversity archive (mechanism × binding
  ceiling), so the frontier keeps e.g. a memory-bound winner **and** a compute-bound
  winner **and** a basement winner simultaneously — maximal route diversity.

## 4. Where each applies

| Context | Start | Notes |
|---------|-------|-------|
| **`operator=single`** (explicit) | **always the best** | the paper-faithful chain; one focused edit/step |
| **supervisor backtrack** | an **earlier tag** (`backtrack_to`) | non-best start to escape a local trap (anti-stall) |
| **`operator=team`** (default) | `pairStartsWithDirections` → `selectStartPoints(policy)` | K directions × K starts, zipped; one worker per start/GPU |
| **ESCALATE** (stall rescue) | diversified: `best → topk`, else the chosen policy | a single point is pointless when stalled, so it always widens |

So **not every start is the best**: single-operator chains stay on best, but fan-out
(team / ESCALATE) and backtrack deliberately start elsewhere.

## 5. Safety invariant — diversification never regresses

However scattered the starts, the result cannot get worse:

- A candidate is committed only if it passes the **deterministic commit gate**
  (correct ∧ independently-verified geomean beats the global best beyond the noise
  band). A behind-branch candidate either **verifiably beats the global best** (then it
  advances best) or it is rejected.
- A winning off-branch commit **parents on its own start ref** (a tree edge), so it
  **never demotes** the current best; the global best advances *only* on a verified
  global improvement.
- Storage is a **git-tagged DAG** either way: `best` uses it as a chain;
  `topk/diverse/elites` use it as a tree. Every version is a real `avo-v{N}` tag, so
  any start is reproducible and auditable.

This is the whole point: **explore aggressively, advance conservatively.**

## 6. Deterministic vs LLM split

| Concern | Owner |
|---------|-------|
| Which refs to start from (`best`/`topk`/`diverse`/`elites`) | **pure** `lib/start_select.mjs` (+ `lib/archive.mjs` for `elites`) |
| Pairing starts × planned directions | **deterministic** `pairStartsWithDirections` in `avo_workflow.js` |
| Advancing the global best (commit gate) | **deterministic** `passesCommitGate` |
| What edit to make from a given start | **LLM** `variation_engineer` |

Selection is **code** (reproducible, unit-tested in `tests/start_select.test.mjs` +
`tests/archive.test.mjs`); only the edit is judgement.

## 7. Why diverse starts help approach the roofline / beat experts

- **Escape local optima:** the binding ceiling shifts as the kernel evolves; a branch
  that was memory-bound-optimal may be a dead end once compute-bound — `elites` keeps a
  live candidate in *each* regime to branch from.
- **Feeds the Integrator merge:** complementary winners from different regime starts
  (same base) are exactly what `selectMergeGroup` + the `avo_integrator` combine into a
  compounding candidate (see `avo_on_team.md` §7).
- **Quality-diversity, not just elitism:** keeping mechanism-distinct elites is how an
  evolutionary search reaches solutions a greedy single-chain (or a human iterating one
  kernel) would never stumble onto.

## 8. Tunables

| arg | default | meaning |
|-----|---------|---------|
| `start_policy` | `diverse` | `best` \| `topk` \| `diverse` \| `elites` |
| `start_k` | `operator_directions` | number of distinct start points for fan-out |
| `operator` | `team` | `single` (chain, start=best) \| `team` (fan-out uses `start_policy`) |

Guidance: the default is `team` + `diverse`. Explicitly configure
`operator=single,start_policy=best` for faithful single-chain depth. `elites` is
available when a richer strategy × regime frontier is useful. Cost grows ~K×
per step with fan-out.

## 9. Acceptance ("start selection is working")

- `start_policy=best` (or `operator=single`) → a **linear** `parent_id` chain.
- `start_policy∈{topk,diverse,elites}` → at least one committed node whose `parent_id`
  is **not** the immediately-preceding best (a real tree branch off an earlier version).
- The global best in `lineage.json` always has the max verified speedup (a behind-branch
  commit never demotes it).
- Unit tests green: `selectStartPoints` (best/topk/diverse/k=1/empty) +
  `selectElites`/`crossoverPairs` (`lib/archive.mjs`).

## 10. Deviations & follow-ups

1. **No island model** (persistent per-branch budgets/bests) — `diverse`/`elites` are
   single-best-advancing tree exploration, not islands. Out of scope for now.
2. **Cross-base crossover** is not yet a start: the Integrator merge only combines
   complementary winners that **share** a base ref (`selectMergeGroup`). A true
   cross-lineage crossover *edit* is a follow-up.
3. **Cross-run seeding** (start a new kernel from elites learned on a related
   `(op, dtype, arch)`) is future work — the same cross-run-memory gap noted elsewhere.
