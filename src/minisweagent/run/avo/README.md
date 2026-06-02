# AVO — Agentic Variation Operators on GEAK

`geak-avo` runs a **single-lineage, continuous kernel-evolution** loop on top of
GEAK. It implements the NVIDIA AVO paper (*Agentic Variation Operators for
Autonomous Evolutionary Search*, arXiv:2603.24517) as an **additive layer** — it
reuses GEAK's preprocess, `OptimizationAgent`, `save_and_test`,
`strategy_manager`, RAG, and `RunBudget`, and modifies no core module.

For the full design and rationale, see
[`docs/developer/avo_design.md`](../../../../docs/developer/avo_design.md).

---

## What it does

Instead of GEAK's default fixed/planned **round loop**, AVO runs an outer loop of
**variation steps**. Each step is one `OptimizationAgent` run that edits the
current best kernel, tests it, and — only if it is **correct and verifiably
faster** — commits it to a growing *lineage* (`P_t`). A two-layer **supervisor**
keeps the run from stalling: a deterministic detector that always fires, plus an
LLM `avo-supervisor` subagent that re-plans the direction when a path is dead.

```text
preprocess (reused) → COMMANDMENT.md (= scoring f)
        │
        ▼   outer loop until wall-clock budget
  reset worktree to best ──► variation step (OptimizationAgent + save_and_test)
        ▲                          │
        │                    commit gate (correct AND speedup ≥ best·(1-ε))
        │                          │
        └── supervisor ◄── StagnationDetector (stall / cycle / no-improvement)
        │
        ▼   budget exhausted
   auto_finalize → final_report.json
```

---

## Prerequisites

Same as `geak` (see [`docs/quick_start.md`](../../../../docs/quick_start.md)):

- A GEAK install: `make install` (or `pip install -e .`).
- A configured model + key, e.g.

  ```bash
  export MSWEA_MODEL_NAME="anthropic/claude-opus-4-6"
  export ANTHROPIC_API_KEY="YOUR_KEY"
  # or the AMD LLM gateway:
  export AMD_LLM_API_KEY="YOUR_KEY"
  ```

- A GPU and the kernel's toolchain (Triton / PyTorch / ROCm / hipcc, etc.).

---

## Quick start

```bash
geak-avo \
  --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel. Metric: latency (lower is better)." \
  --mode quick          # 1-hour cap for a first smoke run
```

A longer, paper-style run:

```bash
geak-avo \
  --repo /path/to/kernel/repo \
  --task "Optimize the attention kernel." \
  --kernel-language triton \
  --mode full           # 7-day wall-clock cap (see geak_avo.yaml)
```

The run **resumes automatically**: if the output directory already contains
`avo_state/lineage.json`, AVO continues from the existing best version.

---

## CLI options

| Option | Default | Meaning |
|--------|---------|---------|
| `--repo` | (required) | Target kernel repository root. |
| `-t`, `--task` | (required) | Optimization task description (include the metric). |
| `-o`, `--output` | `optimization_logs/avo_<repo>_<ts>/` | Run output directory. |
| `-m`, `--model` | from config | Model name override. |
| `--mode` | `full` | Budget mode: `quick` (1 h) or `full` (7 d). |
| `--total-budget-s` | — | Override the mode's wall-clock cap, in seconds. |
| `--kernel-language` | `python` | `triton` \| `hip` \| `flydsl` \| `python` — selects which skills to inject. |
| `--gpu-ids` | `0` | Comma-separated GPU device indices used for per-step verification. |
| `-c`, `--config` | — | Extra YAML merged last over `geak.yaml` + `geak_avo.yaml`. |

---

## Configuration

`geak-avo` loads, in order (deep-merged): `geak.yaml` → `geak_avo.yaml` →
optional `--config`. The AVO-specific knobs live in
[`config/geak_avo.yaml`](../../config/geak_avo.yaml):

```yaml
avo:
  variation_step_limit: 200      # OptimizationAgent step_limit per variation step
  commit_epsilon: 0.001          # min relative speedup to enter the lineage
  min_commits_before_stop: 5     # budget-remaining runs should not stop empty-handed
  stagnation:
    steps_without_commit: 80
    consecutive_no_improvement: 8
    consecutive_correctness_failures: 5
    patch_hash_repeat: 3
    supervisor_cycles_without_commit: 3   # → triggers an ESCALATE rescue round
  escalate:
    enabled: true
    rescue_mode: planned
    rescue_workers: 4
```

The wall-clock caps come from `run.budgets.{quick,full}` (the `full` mode is
overridden to 7 days in `geak_avo.yaml`).

---

## Outputs

```text
optimization_logs/avo_<repo>_<ts>/
├── COMMANDMENT.md            # the test/benchmark contract (= scoring f), from preprocess
├── baseline_metrics.json     # preprocess baseline
├── profile.json              # preprocess profiling
├── .optimization_strategies.md   # run-wide strategy state (shared by agents + supervisor)
├── avo_state/
│   ├── lineage.json          # committed versions (P_t) + best_id + active tip
│   ├── attempts.jsonl        # every attempt, including failures
│   ├── supervisor_log.jsonl  # supervisor interventions
│   ├── direction.json        # current assigned strategy
│   ├── heartbeat.json        # liveness + resume anchor
│   ├── notebook/             # cross-step working notebook (memory summary source)
│   └── patches/v{N}.patch    # committed patches, one per lineage node
├── variation_0001/           # one directory per variation step (logs + strategies)
│   ├── agent.log
│   └── .optimization_strategies.md
├── results/round_1/avo-worker/   # GEAK-canonical layout (patches + best_results.json)
│   ├── patch_*.patch / patch_*_test.txt
│   └── best_results.json
├── round_1_evaluation.json   # verified FULL_BENCHMARK + per-shape geomean (from evaluate_round_best)
├── avo_controller.log
└── final_report.json         # best result (canonical GEAK shape + an "avo" block)
```

The **commit gate**: after each step the controller runs GEAK's
`evaluate_round_best` (apply best patch in a temp worktree → `FULL_BENCHMARK` +
`PROFILE` → **per-shape geomean** verified speedup). A version enters
`lineage.json` only if its candidate is correct, its *verified* speedup is at
least the running best (within `commit_epsilon`), **and** it exceeds the
anti-lazy-optimization floor `min_commit_speedup` (default `1.0` → must be
genuinely faster than baseline; raise to e.g. `1.05` to require ≥5% per commit).
Non-improving / failing attempts are recorded in `attempts.jsonl` but never
committed — mirroring the paper's "500+ internal attempts → ~40 commits". Set
`avo.verify_each_step: false` to skip per-step FULL_BENCHMARK and use a
lightweight log parse instead.

Each step's prompt also injects the **current-best diff + its verified metrics**
(`avo.inject_best_exemplar`, default on) so the agent edits from the best
implementation rather than re-deriving it.

**Delayed profiling** (CuTeGen): the first `avo.profiling_after_step` steps
(default 3) are marked *structural-first* (profiler-driven micro-tuning withheld
to avoid premature local optima); later steps are *profiling-guided*. Set it to
`0` for simple/elementwise kernels. The `avo-evolution` skill also carries
kernel-class optimization menus and non-negotiable "don't simplify away the
kernel / don't fall back to cuBLAS" rules (`docs/optimization_playbook.md`).

**Lineage integrity:** each committed version is reconstructed by applying the
*verified* patch onto the parent-best and tagged `avo-v{N}` (so the next step
resumes from the real best, not the agent's last dirty edit); the worktree is
`git clean`-ed between steps. **Per-shape guard:** set `avo.min_per_shape_speedup`
(e.g. `0.95`) to reject commits that regress any shape even when the geomean
passes. Each step's prompt also injects a **target-hardware** summary so tiling /
occupancy / tensor-core decisions are arch-aware.

---

## How the supervisor prevents stalling

| Layer | Where | Role |
|-------|-------|------|
| Deterministic detector | `stagnation.py` | The "alarm clock" — always fires on stall/cycle/no-improvement counters. Levels: `NUDGE → INTERRUPT → REDIRECT → ESCALATE`. |
| LLM re-planner | `subagents/avo-supervisor/` | On `REDIRECT`, reads lineage + failures + bottleneck and returns a JSON directive (mark-failed, new strategies, optional backtrack). |
| Fallback taxonomy | `supervisor.py` | If the LLM supervisor is unavailable/unparseable, a fixed direction rotation keeps the run progressing. |
| ESCALATE | `controller.py` | After repeated supervisor cycles with no commit, runs one GEAK parallel rescue round. |

The outer controller **never stops because a variation agent stopped** — any
agent termination (`Submitted`, `LimitsExceeded`, exception, deadline) is just
"this step is done", and the loop continues until the budget's `soft_stop`.

---

## Skills used

AVO reuses GEAK's skill system rather than re-encoding domain knowledge:

| Skill | Role | Loaded |
|-------|------|--------|
| `triton` / `hip` | build the harness = scoring `f` | preprocess (automatic) |
| `flydsl` / `fp8-gemm-tuning-sglang-aiter` | optimization knowledge `K` | variation step (when language/task matches) |
| `avo-evolution` | the evolution methodology (lineage, commit gate, self-pivot) | variation step (always) |

`geak-avo` **force-injects** the selected `SKILL.md` bodies into each step's task
(rather than relying on the model to self-load), so guidance is present on every
step of a long run.

## Cross-step memory

Because each variation step runs a fresh `OptimizationAgent` (messages reset),
AVO reconstructs cross-step memory two ways:

- a **run-wide strategy file** (`.optimization_strategies.md`) shared by all
  variation agents and the supervisor, so "tried / failed / pending" persists;
- a **run-wide working notebook** (`avo_state/notebook/`) whose summary
  (best-so-far, what worked, dead ends, recent evidence) is injected into every
  step's prompt.

This is a compact reconstruction, not the paper's full continuous conversation
memory — see the design doc's memory section for the remaining gap.

---

## Stopping & resuming

- **Ctrl-C** once: the controller finishes the current step and finalizes.
- The run ends when the wall-clock budget elapses; `final_report.json` always
  reflects the best committed version.
- Re-running with the **same `-o` directory** resumes from `avo_state/`.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `No run.budgets.<mode>` error | Custom `--config` dropped the `run.budgets` block; keep `geak.yaml` defaults. |
| Preprocess produced no `COMMANDMENT.md` | The `geak --preprocess-only` step failed; run it manually to inspect, or pass an existing prepared dir via `-o`. |
| No commits after many steps | Check `attempts.jsonl` — correctness failures or unverified benchmarks never commit. Confirm the harness emits a parseable speedup. |
| Supervisor never re-plans | Lower the `avo.stagnation` thresholds in a `--config` override. |

---

## See also

- Design doc: [`docs/developer/avo_design.md`](../../../../docs/developer/avo_design.md)
- GEAK quick start: [`docs/quick_start.md`](../../../../docs/quick_start.md)
- Skill authoring: [`skills/README.md`](../../../../skills/README.md)
- Subagent authoring: [`subagents/README.md`](../../../../subagents/README.md)
