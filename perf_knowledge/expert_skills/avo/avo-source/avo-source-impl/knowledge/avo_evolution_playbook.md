# AVO Evolution — Playbook (the AVO method, and why)

Condensed from GEAK's `skills/avo-evolution` and the NVIDIA AVO paper
(*Agentic Variation Operators for Autonomous Evolutionary Search*,
arXiv:2603.24517). In AVO, **one agent run = one variation step**; a controller
owns the lineage, the wall-clock budget, and the anti-stall supervisor.

## The operator

`Vary(P_t) = Agent(P_t, K, f)` — a self-directed coding agent that, each step:
1. starts from the **best** committed version (the worktree is reset to it),
2. consults the lineage `P_t` (prior versions + scores) and knowledge `K`,
3. makes ONE focused change aligned with the active direction,
4. tests via the scoring function `f` (correctness + benchmark),
5. commits ONLY a correct, verified improvement.

## Commit gate (the integrity core)

A version enters the lineage only if:
- correctness passes, AND
- the **independently-verified** geomean speedup ≥ running best (within epsilon),
- above an anti-lazy floor (a ≈1.0x trivial rewrite is rejected).

Self-reported speedups never count. Verification re-applies the patch on a clean
base, so an agent editing the test/harness cannot fake a win.

## Anti-stall supervision

- A deterministic stagnation detector always fires on stall / cycle /
  no-improvement counters → `NUDGE → REDIRECT → ESCALATE`.
- On `REDIRECT`, an LLM supervisor reviews the trajectory and proposes new
  untried directions (and may backtrack to an earlier version).
- `ESCALATE` runs a diversified rescue (several distinct directions) and keeps the
  best verified one.

## Anti-cheat (non-negotiable)

- Do NOT "optimize" by simplifying away the kernel structure (tiling / MMA /
  pipeline) or falling back to PyTorch / cuBLAS / cuDNN.
- Do NOT change precision or I/O shapes to inflate a number.
- Honor the optimization stage: structural-first, profiling-guided later.

## Paper-backed expectations (§4.4)

- Improvements arrive in **discrete jumps**, not smooth gradients.
- Early steps: coarse architectural changes; late steps: micro-arch tuning
  (registers, fences, pipeline overlap).
- Many internal attempts per commit is normal — failed attempts are signal.

## Measurement rigor (Tier-1 — important)

The paper measures each config with warm-up + **10 runs (mean/stddev)**. In a long
evolutionary run, benchmark noise **compounds** (a noisy "win" commits and every
later step branches from a false best). This workflow defaults to
`verify_repeats >= 3`, and the verifier aggregates **robustly** (drop outliers via the
MAD rule, take the median) and reports the measured **cv** (`lib/measure.mjs`). The
commit gate is then **variance-aware**: the required improvement is
`max(significanceMargin, commit_noise_k·cv)`, so a within-noise gain cannot commit.
Prefer rocprof-**measured** HBM bytes for the roofline regime. Keep timing stable.
