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

Pinning GPUs and supplying your own unit-test / eval command:

```bash
geak-avo \
  --repo GEAK/examples/knn \
  --task "Optimize the knn kernel. Metric: latency (lower is better)." \
  --kernel-language hip \
  --mode quick \
  --gpu-ids 2 \
  --test-command "python3 scripts/task_runner.py correctness"
```

The run **resumes automatically**: if the output directory already contains
`avo_state/lineage.json`, AVO continues from the existing best version.

---

## CLI options

**Only two options are required:** `--repo` and `--task`. Everything else is
optional and has a sensible default — a minimal run is just
`geak-avo --repo <path> --task "<goal incl. metric>"`.

| Option | Required? | Default | Meaning |
|--------|-----------|---------|---------|
| `--repo` | **Yes** | — | Target kernel repository root. |
| `-t`, `--task` | **Yes** | — | Optimization task description (include the metric). |
| `-o`, `--output` | No | `optimization_logs/avo_<repo>_<ts>/` | Run output directory. |
| `-m`, `--model` | No | from config | Model name override. |
| `--mode` | No | `full` | Budget mode: `quick` (1 h) or `full` (7 d). |
| `--total-budget-s` | No | — | Override the mode's wall-clock cap, in seconds. |
| `--kernel-language` | No | `python` | `triton` \| `hip` \| `flydsl` \| `python` — selects which skills to inject. Internal to AVO (not forwarded to preprocess, which auto-detects the kernel type). |
| `--gpu-ids` | No | `0` | Comma-separated GPU device indices. Applied end-to-end: preprocess, each variation step's `save_and_test` (via `GEAK_GPU_DEVICE` / `HIP_VISIBLE_DEVICES`), per-step verification, and ESCALATE workers. |
| `--test_command`, `--test-command` | No | — | Manually specify the unit-test / eval command. Forwarded to preprocess so `geak` bakes it into the COMMANDMENT/harness (skips UnitTestAgent auto-discovery); variation steps pick it up from `COMMANDMENT.md`. Only takes effect on a fresh run (preprocess is skipped on resume). |
| `--rag` / `--no-rag` | No | from config (`true`) | Enable/disable RAG tools (`query`/`optimize`). **Overrides** `avo.use_rag`. |
| `--profiling` / `--no-profiling` | No | from config (`true`) | Enable/disable profiling (`profile_kernel` tool + per-step verification `PROFILE`). **Overrides** `avo.use_profiling`. |
| `-c`, `--config` | No | — | Extra YAML merged last over `geak.yaml` + `geak_avo.yaml`. Also forwarded to the preprocess subprocess — note `geak`'s `-c` *replaces* `geak.yaml` as the user layer (it does not merge), so pass a complete config, not a snippet. |

> **Forwarded to preprocess.** `--model`, `--gpu-ids`, `--mode`, `--config`, and
> `--test-command` are passed through to the `geak --preprocess-only` subprocess
> so preprocess and the evolution loop stay consistent (e.g. `--gpu-ids 2` runs
> baseline/profile on GPU 2, not GPU 0).

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
  use_rag: true                  # RAG tools (query/optimize) per step; false disables them
  use_profiling: true            # profiling (profile_kernel tool + per-step verification PROFILE); false skips both
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

### Feature switches (`use_rag`, `use_profiling`)

Both default **on**; set either to `false` to speed up / simplify test runs.
Precedence (highest first): **CLI flag (`--no-rag` / `--no-profiling`) → config
(`avo.use_rag` / `avo.use_profiling`) → default `true`**. The resolved value is
applied **globally and uniformly** — to every variation step, the ESCALATE
rescue workers, and per-step verification:

| Switch | `false` disables |
|--------|------------------|
| `use_rag` | the `query` / `optimize` RAG tools in every step (also skips the RAG postprocessor model) |
| `use_profiling` | the agent's `profile_kernel` tool, the per-step verification `PROFILE` (via `GEAK_SKIP_PROFILE=1`), and the profiling-stage prompt note |

Fast smoke-test — via CLI flags (simplest, highest priority):

```bash
geak-avo --repo <path> --task "<goal>" --mode quick --no-rag --no-profiling
```

…or via config:

```yaml
# avo_fast.yaml  (used with --config)
avo:
  use_rag: false
  use_profiling: false
```

`use_profiling: false` keeps the verified `FULL_BENCHMARK` (so the commit gate
and speedups stay accurate) — it only drops profiling, which AVO uses for
supervisor diagnosis and causal memory, not for the commit decision.

---

## Outputs

```text
optimization_logs/avo_<repo>_<ts>/
├── COMMANDMENT.md            # the test/benchmark contract (= scoring f), from preprocess
├── baseline_metrics.json     # preprocess baseline
├── profile.json              # preprocess profiling
├── avo_repo/                 # isolated clone of --repo (agent edits + lineage git tags live here; original repo untouched)
├── .optimization_strategies.md   # run-wide strategy state (shared by agents + supervisor)
├── avo_state/
│   ├── lineage.json          # committed versions (P_t) + best_id + active tip
│   ├── attempts.jsonl        # every attempt, including failures
│   ├── supervisor_log.jsonl  # supervisor interventions
│   ├── direction.json        # current assigned strategy
│   ├── heartbeat.json        # liveness + resume anchor
│   ├── notebook/             # cross-step working notebook (memory summary source)
│   ├── evolution_log/        # per-step causal memory (rationale + profiling + raw tail) — P-mem-3
│   ├── verify_cache.json     # verified-speedup cache keyed by patch hash (C2)
│   └── patches/v{N}.patch    # committed patches, one per lineage node
├── trajectory.json           # evolution health: best-speedup curve, commit rate, per-strategy, supervisor interventions
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

**Noise-robust progress (B1):** on noisy harnesses set `avo.verify_repeats: 3`
to re-measure each candidate 3× and use the median (suppresses outlier spikes),
and/or `avo.commit_significance_margin: 0.01` to require a commit to clear the
current best by a real >1% margin (a within-noise "tie" is not committed).

Each step's prompt also injects the **current-best diff + its verified metrics**
(`avo.inject_best_exemplar`, default on) so the agent edits from the best
implementation rather than re-deriving it, plus the **top-K other prior
implementations** with their verified + per-shape speedups and truncated diffs
(`avo.lineage_context_k`, default 3) so it can compare approaches across the
lineage (AVO §3.2). Full source of any version: `git show avo-v{id}`.

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

The detector is also **trend-aware** (`stagnation.trend_window`): a non-committing
step that sets a new intra-direction high is "still climbing" and does not count
toward the stall, so slow-but-real directions aren't cut prematurely. The LLM
supervisor's proposed directions are **self-validated** — already-tried/duplicate
directions are dropped and a fallback direction is spliced in if nothing novel
remains. A `trajectory.json` (best-speedup curve, commit rate, per-strategy stats,
supervisor interventions) is written at finalize for at-a-glance run health.

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

On top of these, a bounded **evolution log** (P-mem-3, option C) carries the
*causal* history across steps: each step records the agent's rationale, a short
verbatim raw tail, the verified speedup/commit flag, and profiler metrics; later
steps see recent steps verbatim plus older steps as one-liners with the
bottleneck delta. Knobs: `avo.evolution_log_enabled` / `evolution_log_recent` /
`evolution_log_max_versions`. This approximates the paper's continuous memory
while staying bounded; a literal persistent single agent is out of scope by
design (conflicts with the per-step worktree reset).

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
