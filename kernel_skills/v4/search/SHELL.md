# Search shell: the contract a search strategy plugs into (design, not yet wired)

Goal: anyone can write a search strategy without touching roles or tools. A strategy chooses **what to
try next**. The shell owns **how anything is tried and judged**: workspaces, measurement, ship rule,
roofline, end rule, GPU scaling.

```
          ┌────────────── shell (fixed) ──────────────┐
 state ──►│ observe ─► [policy hook] ─► reason ─► check ─► dispatch ─► record outcome ─┐
   ▲      └──────────────────────────────────────────────────────────────────────────┘ │
   └───────────────────────────── archive + strategy state ◄──────────────────────────┘
```

## 1. State (read by every strategy)
| piece | owner | content |
|---|---|---|
| `RUN/archive.json` | shell (`bin/archive.py build`) | one node per attempt with source: `id`, `parents[]`, `tags{}`, `status`, `score`, per-case rows `{sig, regime, weight, speedup}`, `patch`, `notes` |
| core views | shell (`archive.py views`) | best, noise-aware front, per-regime bests, tag groups |
| `RUN/roofline/cap.json` | shell (geak-roofline) | per-case floors, weighted cap, gap closed |
| `RUN/decisions/round_<n>.json` | shell | every past decision and its outcome |
| `RUN/search/<name>/state.json` | **strategy** | anything the strategy keeps between rounds (scores, bins, islands, temperatures…) |

Node `tags` are free-form key/values a strategy may ask executors to set (e.g. `approach`, `island`,
`cell`). The shell stores them and never interprets them.

## 2. Strategy package: `search/<name>/`
| file | required | contract |
|---|---|---|
| `STRATEGY.md` | yes, ≤ 60 lines | the sections in `_template/STRATEGY.md`: State used, Actions used, Parent selection, Round composition, Triggers, Stop, Tags |
| `policy.py` | no | `python3 policy.py --run RUN --archive RUN/archive.json --state RUN/search/<name>/state.json` → prints JSON `{"state": {...}, "proposals": [<action>...], "notes": "..."}`. It may update its own state; it never edits the archive or dispatches. Proposals are advice: the lead's reasoning makes the decision |
| `roles/<role>.md` | no | prompts for `custom` actions, ≤ 100 lines each, following `COMMON.md` |

## 3. Actions (the only verbs a decision may use)
| op | required fields | executed by |
|---|---|---|
| `mutate` | `parent`, `cut`, `direction` (+ optional `refs[]`, `retain: best|development`, `tags`) | geak-engineer |
| `fresh` | `cut`, `direction` (+ `tags`) | geak-engineer `MODE=fresh-arch` |
| `combine` | `parents[]`, `kind: merge|dispatch` | geak-ship |
| `measure` | `nodes[]`, `instrument`, `reps` | geak-benchmark (re-measure, de-noise) |
| `investigate` | `mode`, `question` | geak-investigator |
| `ship` | `node` | geak-ship (shell ship rule) |
| `custom` | `role`, `args{}` | `search/<name>/roles/<role>.md` |
| `stop` | — | lead, only under the shell end rule |
Every action also carries `why` (citing state) and gets `outcome` filled at round end (new node ids,
status, score delta, whether shipped).

## 4. One round (the shell's loop)
1. **observe**: rebuild the archive and core views; load cap, past decisions, strategy state.
2. **policy hook**: run `policy.py` if present; keep its proposals and state.
3. **reason**: the lead reads `STRATEGY.md` + state + proposals, and writes `decisions/round_<n>.json`.
4. **check**: `bin/check_decision.py` (to build) validates the schema (`decision.schema.json`), that
   referenced nodes exist with source, that ops are in the strategy's declared set, and that the GPU-heavy
   count fits the pool.
5. **dispatch**: the shell maps ops to executors; executors write nodes with the requested tags.
6. **record**: fill each action's `outcome`; the strategy updates `state.json` (via `policy.py` next round).

## 5. Fixed by the shell, whatever the strategy
Measurement and noise, ship rule, retention ≠ promotion, roofline-gated end rule, deadline, GPU scaling,
restart protocol, audit. A strategy may stop earlier than the end rule allows only for budget or deadline.

## 6. Expressibility check (each is a STRATEGY.md + optional policy.py, nothing else)
- hillclimb: parent = shipped; mutate per top-gap cut; combine(merge) at round end.
- islands / evo: tag `island`; parent = tournament on the front within an island; fresh opens an island.
- MAP-Elites: tag `cell` (e.g. regime × design); `policy.py` keeps the elite per cell; parent = random elite.
- bandit over directions: `policy.py` keeps UCB per `direction-class` tag; mutate the arm with the best bound.
- avo: mutate with `refs` = failing siblings; `direction` names the variation operator.
- annealing / restarts: `state.json` holds a temperature; accept slower parents with a probability.
- lineage migration: `custom` role that imports peer checkpoints as nodes (tag `origin`).

## 7. Comparing strategies
Same task, GPUs, hours and model; ≥ 2 seeds. Report: final verified score, gap closed, score vs cost curve,
and from the decision files the yield per op type (share of actions whose child joined the front / shipped).
