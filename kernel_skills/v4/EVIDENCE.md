# v4: search as state + policy

v4 keeps every v3 role and rule (`../v3/EVIDENCE.md`). Its only change is how a lead picks what to do
next. Optimization is a search over an **archive** of measured nodes (`bin/archive.py`). Each round the lead
reads a **search template** (`search/hillclimb.md`, `evo.md`, `avo.md`) and writes its **decision** as data
(`decisions/round_<n>.json`: mutate / crossover / fresh / investigate / ship / stop, each with a reason and,
later, its outcome).

Why:
- In v3 the parent choice existed in the data model (candidate tree, any node can be a parent). It was
  invisible as a policy, so every lead and the EAT supervisor defaulted to "climb from the shipped node,
  merge at round end". The EAT trees show that: 20–21-node mainlines; side branches mostly variants of
  one engineer's own candidate; 0–9 multi-parent merges per task.
- A written policy can be compared (hillclimb vs evo vs avo at equal budget), changed per kernel, and
  harvested from decision files rather than reconstructed from transcripts.

Offline replay, before any GPU time: `archive.py build --ut-reports` over all EAT official evals,
grouped by code snapshot (21–36 nodes per task). Per-regime / per-band dispatch of measured nodes would
have beaten the best single node by:

| task | best node | regime dispatch | regime × band | per case (noise-inflated upper bound) |
|---|---|---|---|---|
| compact | 1.1746 | +0.00% | +0.00% | +0.43% |
| gqa | 1.1930 | +0.00% | +0.00% | +1.10% |
| splitk | 1.1855 | +0.00% | +0.00% | +0.23% |
| norm | 1.3036 | +0.58% | +0.58% | +1.63% |

So recombining what a hillclimb already measured buys little. The archive held mostly mainline snapshots
(only shipped or contested nodes got official evals). `evo` therefore spends its budget on diverse parents and
new islands (where EAT's largest steps came from: gqa's architect family, norm's prefill rewrite), not on
crossover. Its 20% crossover share is the first knob to cut if runs confirm this.

Not yet run on GPU. Next: hillclimb vs evo (vs avo) from c000, same GPUs and hours, ≥ 2 seeds per arm.
"avo" here means agentic variation operators (the agent as a directed mutation operator with failure
briefs and named operators). If a specific published AVO method is meant, `search/avo.md` should follow it.

## Mapping to GEAK-evo (PR #580)

`search/evo.md` is meant to reach PR #580's evolutionary search as skills. Below, each PR #580 feature
next to where v4 has it:

| GEAK-evo (PR #580) | v4 |
|---|---|
| archive of all attempts with source, schema `geak.evolution/1`, `best_id` tracked separately | `bin/archive.py build` → `RUN/archive.json` (`schema`, `entries`, `best_id` from `RUN/shipped/ID`) |
| status correct / slower / incorrect / unverified; retention ≠ promotion | `archive.py` status mapping; COMMON "retention never promotes"; ship rule unchanged |
| planner picks `parent_id`, `donor_ids`, `approach` per direction | lead decision `mutate {parent_id, donor_ids, approach, question}` |
| private workspace = fixed base + parent's cumulative patch | `bin/mkws.sh $RUN <ws> cands/<parent>/patch.diff` (base = `RUN/orig`) |
| donors as references; the engineer combines and checks the contribution is present | engineer `DONORS`: name the contribution, verify it is in the executed code |
| `evolution_return_development` / `preserve_founder_attempts` | engineer `RETAIN=development`: the best correct realization even if slower, retained as `slower` |
| `evolution_start` first_stall / search_start | lead `EVOLUTION_START` |
| planner sees the latest 12 entries + best, full archive on disk | evo.md "Planner view" |
| ledger: approach continuity, outcome classes, subsumed / reopen | evo.md "Ledger discipline" |
| `population_lineage` checkpoint migration (publish / import named peers; imports become parents and donors, never auto-incumbent) | `bin/population.py publish / import` (base-hash + metric + patch-hash checks); evo.md "Population" |
| `evolution_evidence_handoff` / case-evidence views | the decision `question` + failure brief (avo); `archive.py views` per-regime front |
| greedy remains selectable | `SEARCH=hillclimb` |

Not reproduced: PR #580's JS runtime (budget/deep_cost accounting, process-bounded clocks, native evaluator,
Hyperloom handoff). v4 relies on the lead's time boxes and the v3 instrument instead. PR #580's reported
selected results (+5.85% over GEAK on 2 AgentX comparisons; +8.59% on 3 InferenceX rows) are its own;
v4's evo has not been run yet.
