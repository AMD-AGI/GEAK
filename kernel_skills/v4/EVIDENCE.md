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
