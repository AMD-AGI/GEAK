# API reference

Reference for GEAK's public surfaces: the command-line entry points, the
environment variables that tune behavior, the configuration schema, the run
artifacts contract, and the importable Python API.

For task-oriented walkthroughs see **[Quick start](quick_start.md)**; for how
YAML is loaded and merged see **[Configuration files](configuration.md)**.

---

## 1. Command-line entry points

GEAK installs the following console scripts (declared under
`[project.scripts]` in `pyproject.toml`):

| Command | Module | Purpose |
|---------|--------|---------|
| `geak` | `minisweagent.run.mini:app` | Main optimization CLI (preprocess → optimize loop). |
| `geak-preprocess` | `minisweagent.run.preprocess.preprocessor:main` | Run preprocessing only (resolve → context → discover → harness → profile → baseline → commandment). |
| `geak-gemm-tuning` | `minisweagent.run.gemm_tuning:app` | GEMM selection/configuration tuning agent. |
| `kernel-profile` | `minisweagent.run.preprocess.kernel_profile:main` | Standalone kernel profiling. |
| `commandment` | `minisweagent.run.preprocess.commandment:main` | Generate the `COMMANDMENT.md` contract. |
| `validate-commandment` | `minisweagent.tools.validate_commandment:main` | Validate a `COMMANDMENT.md`. |
| `run-harness` | `minisweagent.run.preprocess.run_harness:main` | Execute a test harness in a known mode. |
| `baseline-metrics` | `minisweagent.run.preprocess.baseline:main` | Capture baseline metrics. |
| `codebase-context` | `minisweagent.run.preprocess.codebase_context:main` | Build `CODEBASE_CONTEXT.md`. |
| `resolve-kernel-url` | `minisweagent.run.preprocess.resolve_kernel_url:main` | Resolve a kernel path/URL. |
| `task-generator` | `minisweagent.agents.heterogeneous.task_generator:main` | Generate optimization tasks. |

---

### 1.1 `geak`

Main entry point. Runs preprocessing (unless a harness/test command is
supplied) and then the optimization round loop.

```bash
geak [OPTIONS]
```

| Flag | Alias | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--task` | `-t` | str | — | Task / problem statement (natural language). Paths, GPU IDs, and metrics can be embedded here. |
| `--repo` | | path | — | Target repository root. |
| `--kernel-url` | `--kernel-path` | str | — | Target kernel source (local path or URL). |
| `--test-command` | `--test_command` | str | — | Test command. If omitted, GEAK discovers/generates a harness during preprocess. |
| `--model` | `-m` | str | resolved (§2 / §5) | Model name for this run. |
| `--model-class` | | str | `amd_llm` (advanced) | Backend class shortcut (`litellm`, `amd_llm`, `anthropic_model`, …). |
| `--config` | `-c` | path | `config/geak.yaml` | Config file deep-merged over the base strategy file. |
| `--output` | `-o` | path | auto `optimization_logs/<kernel>_<ts>/` | Output trajectory file or directory. Dotted names are treated as directories. |
| `--num-parallel` | | int | from config | Number of parallel patch agents (isolated git worktrees). |
| `--gpu-ids` | | str | — | Comma-separated GPU IDs, e.g. `0,1,2,3`. |
| `--mode` | | str | `geak.yaml` `run.mode` | Wall-clock budget profile: `quick` (~1h) or `full` (~2h). |
| `--total-budget-s` | | float | from mode | Override the mode's total wall-clock budget (seconds). |
| `--target` | | str | `wall` | Scoring signal: `wall` (end-to-end host latency via `triton.testing.do_bench`) or `kernel` (GPU-only kernel time via `torch.profiler`). The dual-signal harness always reports both; this only picks the scoring signal. |
| `--cost-limit` | `-l` | float | from config | Cost cap. `0` disables. |
| `--yolo` | `-y` | flag | off | Run without per-action confirmation. Parallel workers force this regardless. |
| `--mode` confirm/yolo | | — | | See `agent.mode` in §3. |
| `--debug` / `--no-debug` | | flag | off | Disable post-run patch apply + artifact cleanup; preserve the full run directory. |
| `--preprocess-only` | | flag | off | Run preprocessing only, then exit (advanced). |
| `--exit-immediately` | | flag | off | Exit immediately (advanced). |
| `--visual` | `-v` | flag | off | Toggle the pager-style (Textual) UI. |

**Post-run behavior.** By default (unless `--debug`) two steps run after the
loop completes:

- **Patch apply** — the winning patch is applied to `--repo` on the current
  branch and committed.
- **Cleanup** — per-run artifacts are pruned to `final_report.json`, the
  winning `.diff`, `geak_agent.log`, and `COMMANDMENT.md`.

A hard-kill (wall-clock timeout) always leaves artifacts in place for
forensic analysis, regardless of `--debug`.

**Examples**

```bash
# Natural-language, single agent
geak -t "Optimize aiter/ops/triton/topk.py. Harness: test_topk_harness.py."

# Explicit repo + test command
geak --repo /path/to/repo \
  --test-command "python scripts/task_runner.py compile && python scripts/task_runner.py performance" \
  --task "Optimize the knn kernel. Metric: latency (lower is better)."

# Parallel exploration on 4 GPUs
geak --repo /path/to/repo --task "Optimize the block_reduce kernel" \
  --num-parallel 4 --gpu-ids 0,1,2,3
```

---

### 1.2 `geak-preprocess`

Runs the preprocessing pipeline only and writes intermediate artifacts.

```bash
geak-preprocess <url> [OPTIONS]
```

| Argument / Flag | Alias | Default | Description |
|-----------------|-------|---------|-------------|
| `url` (positional) | | — | GitHub URL or local path to the kernel. |
| `--output` | `-o` | `optimization_logs/…` | Output directory for intermediate artifacts. |
| `--gpu` | | `0` | GPU device ID for profiling. |
| `--model` | `-m` | default | Model for `UnitTestAgent` harness creation. |
| `--harness` | | — | Path to an existing harness (skips LLM generation; must support `--correctness`, `--profile`, `--benchmark`, `--full-benchmark`). |
| `--repo` | | — | Repository root; `url` is resolved relative to this. |
| `--correctness-command` | | — | Compile + correctness command (build folded in). |
| `--performance-command` | | — | Benchmark command (used for profiling + baseline). |
| `--eval-command` | | — | Legacy single command string (prefer the split flags above). |
| `--kernel-type` | | — | Kernel type, e.g. `pytorch2flydsl` (triggers translation). |

---

### 1.3 `geak-gemm-tuning`

Runs a single `GemmTuningAgent` session. Creates
`<cwd>/optimization_logs/gemm_tuning_<timestamp>/` and uses it as the agent
workspace. Agent config comes from `mini_gemm_tuning.yaml`; model/env always
come from `geak.yaml`.

```bash
geak-gemm-tuning run [OPTIONS]
```

| Flag | Alias | Default | Description |
|------|-------|---------|-------------|
| `--task` | `-t` | **required** | Task / instructions for the tuning agent. |
| `--config` | `-c` | — | YAML overlay; only `model_class`, `base_url`, `model_name`, `api_key` from its `model:` section override. |
| `--model` | `-m` | from config | Override `model_name`. |
| `--cwd` | | current dir | Base dir under which `optimization_logs/gemm_tuning_<ts>/` is created. |
| `--log-dir` | | the workspace | Agent log + trajectory directory. |

```bash
geak-gemm-tuning -t "Optimize E2E perf via GEMM tuning. Benchmark: run_sglang_test.sh"
```

---

## 2. Model resolution order

The model **name** is resolved by `get_model_name`
(`src/minisweagent/models/__init__.py`), first hit wins:

1. CLI `-m` / `--model`
2. YAML `model.model_name`
3. env `GEAK_MODEL`
4. env `MSWEA_MODEL_NAME`
5. env `GEAK_MODEL_NAME`
6. env `GEAK_DEFAULT_MODEL` (fallback default `openai/claude-opus-4.8`)

The model **class** comes from YAML `model.model_class` or `--model-class`,
mapped by `get_model_class`:

| Shortcut | Class |
|----------|-------|
| `litellm` | `LitellmModel` — any `provider/model` string supported by LiteLLM (default if unset). |
| `amd_llm` | `AmdLlmModel` — AMD LLM gateway. |
| `anthropic_model` | `AnthropicModel` — direct Anthropic SDK. |
| `deterministic` | `DeterministicModel` — testing. |

A full import path (e.g. `minisweagent.models.anthropic_model.AnthropicModel`)
is also accepted. `MSWEA_MODEL_API_KEY`, when set, is copied into
`model_kwargs.api_key`.

---

## 3. Configuration schema (`geak.yaml`)

Loaded after `mini_kernel_strategy_list.yaml` and deep-merged over it; a
`--config` file is merged on top of both. See
**[Configuration files](configuration.md)** for full details.

```yaml
model:
  model_class: amd_llm      # backend shortcut (see §2)
  model_name: claude-opus-4.6
  api_key: ""               # "" = read AMD_LLM_API_KEY / LLM_GATEWAY_KEY from env
  model_kwargs:             # forwarded to the vendor implementation
    temperature: ...
    max_tokens: ...
    reasoning: { effort: ... }   # GPT-style on the gateway
    text: { verbosity: ... }

agent:
  step_limit: 0             # step cap; 0 = disabled
  cost_limit: 0             # cost cap; 0 = disabled
  mode: confirm             # confirm = interactive; yolo = auto-run

env:
  env: { PAGER: cat, TQDM_DISABLE: "1", ... }   # forwarded to subprocesses
  timeout: 3600             # default command timeout (s)
```

---

## 4. Environment variables

CLI flags and config are usually enough; these tune deeper behavior. Names map
directly to the variables read across `src/minisweagent`.

### Model & run

| Variable | Effect |
|----------|--------|
| `GEAK_MODEL`, `MSWEA_MODEL_NAME`, `GEAK_MODEL_NAME`, `GEAK_DEFAULT_MODEL` | Model name resolution chain (§2). |
| `MSWEA_MODEL_API_KEY` | Override `model_kwargs.api_key` for all backends. |
| `AMD_LLM_API_KEY`, `LLM_GATEWAY_KEY` | AMD LLM gateway credentials (`amd_llm`). |
| `GEAK_CONFIG` | Default config path. |
| `GEAK_PIPELINE_MODE` | Pipeline mode (default `mixed`). |
| `GEAK_MAX_ROUNDS` | Max optimization rounds. |
| `GEAK_AGENT_STEP_LIMIT`, `GEAK_ORCHESTRATOR_STEP_LIMIT`, `GEAK_TASKGEN_STEP_LIMIT` | Step caps per agent role. |
| `GEAK_TASKGEN_COST_LIMIT` | Cost cap for task generation. |
| `MSWEA_GLOBAL_COST_LIMIT`, `MSWEA_GLOBAL_CALL_LIMIT` | Global cost / call caps. |

### Parallelism & GPUs

| Variable | Effect |
|----------|--------|
| `GEAK_GPU_DEVICE` | GPU device for profiling/runs. |
| `GEAK_WORKERS_PER_GPU`, `GEAK_MIN_PARALLEL_WORKERS` | Parallel worker placement. |

### Harness, scoring & timeouts

| Variable | Effect |
|----------|--------|
| `GEAK_HARNESS` | Path to an existing harness. |
| `GEAK_HARNESS_ONLY`, `GEAK_ALLOW_BROKEN_HARNESS`, `GEAK_SKIP_COMMANDMENT_PREFLIGHT` | Harness/preflight relaxations. |
| `GEAK_SCORE_TARGET` | Scoring signal (`wall` / `kernel`); CLI `--target` takes precedence. |
| `GEAK_BENCHMARK_ITERATIONS`, `GEAK_EVAL_BENCHMARK_ITERATIONS`, `GEAK_AGENT_BENCHMARK_ITERATIONS`, `GEAK_BASELINE_REPEATS` | Iteration counts. |
| `GEAK_MAX_BENCHMARK_SHAPES` | Cap benchmark shapes. |
| `GEAK_BASH_TIMEOUT(_S)`, `GEAK_BENCH_TIMEOUT`, `GEAK_CORRECTNESS_TIMEOUT`, `GEAK_PROFILE_TIMEOUT`, `GEAK_EXPLORE_TIMEOUT`, `GEAK_LLM_REQUEST_TIMEOUT` | Various timeouts. |
| `GEAK_SKIP_CORRECTNESS_GATE`, `GEAK_CORRECTNESS_GATE_TIMEOUT` | Correctness gate control. |

### Memory & knowledge base

| Variable | Default | Effect |
|----------|---------|--------|
| `GEAK_MEMORY_DISABLE=1` | off | Turn off all memory (within-session + cross-session). |
| `GEAK_USE_KNOWLEDGE_BASE=0` | on | Stop reading past insights from the knowledge base. |
| `GEAK_SAVE_TO_KNOWLEDGE_BASE=1` | off | Save run insights back to the knowledge base. |
| `GEAK_MEMORY_MIN_SPEEDUP` | `1.10` | Minimum speedup required to save an experience. |
| `GEAK_MEMORY_NO_CROSS_SESSION`, `GEAK_MEMORY_NO_WORKING`, `GEAK_USE_CROSS_SESSION_MEMORY` | | Finer memory toggles. |
| `GEAK_MEMORY_STORE_PATH`, `GEAK_CROSS_SESSION_MEMORY_URL`, `GEAK_MEMORY_API_KEY`, `GEAK_MEMORY_RETRIEVAL_LIMIT` | | Memory backend config. |

### Tools, skills & subagents

| Variable | Effect |
|----------|--------|
| `GEAK_USE_SKILLS=1` | Enable skill loading. |
| `GEAK_DISABLED_TOOLS` | Comma-separated disabled tools. |
| `GEAK_ALLOWED_AGENTS`, `GEAK_EXCLUDED_AGENTS`, `GEAK_FALLBACK_AGENT` | Subagent allow/deny/fallback. |
| `GEAK_USE_KERNEL_ANALYSIS`, `GEAK_PROFILE_EVERY_PATCH` | Analysis/profiling behavior. |
| `GEAK_ROOT`, `GEAK_SUBAGENTS_ROOT`, `GEAK_REPO_ROOT`, `GEAK_WORK_DIR` | Root/path overrides. |

> The list above is representative, not exhaustive. The authoritative set is
> whatever `src/minisweagent` reads at runtime; grep for `GEAK_` / `MSWEA_`.

---

## 5. Run artifacts

Default output base: `optimization_logs/`. Each run gets
`optimization_logs/<kernel_name>_<YYYYmmdd_HHMMSS>/`.

```text
optimization_logs/<kernel>_<timestamp>/
├── CODEBASE_CONTEXT.md          # discovered structure / dependencies
├── COMMANDMENT.md               # single source of truth for the run contract
├── baseline_metrics.json        # captured baseline
├── profile.json                 # bottleneck analysis
├── tasks/round_N/               # canonical + planned-strategy task files
├── results/round_N/<worker>/    # patch_*.patch, patch_*_test.txt, task_*.log
├── round_N_evaluation.json      # per-round best-candidate verification
├── final_report.json            # best verified result across rounds
└── geak_agent.log               # full agent log
```

After cleanup (default, non-`--debug`), a run directory is pruned to
`final_report.json`, the winning `.diff`, `geak_agent.log`, and
`COMMANDMENT.md`.

### Harness result contract

Harnesses report results to GEAK via `GEAK_RESULT_*` environment markers in
their stdout. Key fields:

| Marker | Meaning |
|--------|---------|
| `GEAK_RESULT_LATENCY_MS` | The scoring metric the agent optimizes against (selected by `--target`). |
| `GEAK_RESULT_WALL_MS` | End-to-end host latency (`triton.testing.do_bench`). |
| `GEAK_RESULT_KERNEL_MS` | GPU-only kernel time (`torch.profiler` CUDA events). |
| `GEAK_RESULT_SPEEDUP`, `GEAK_RESULT_GEOMEAN_SPEEDUP` | Speedup vs baseline. |
| `GEAK_RESULT_METRIC`, `GEAK_RESULT_UNIT`, `GEAK_RESULT_DIRECTION` | Custom metric name, unit, and whether higher/lower is better. |
| `GEAK_RESULT_DISPATCH_MS`, `GEAK_RESULT_DISPATCH_FRACTION` | Dispatch/host overhead breakdown. |
| `GEAK_RESULT_TIMING_SOURCE` | Which timer produced the scoring signal. |

---

## 6. Python API

GEAK is primarily a CLI, but a small stable surface is importable.

### `minisweagent`

```python
from minisweagent import (
    get_repo_root, get_data_dir, resolve_entry_script,
    Model, Environment, Agent,   # typing Protocols
)
```

- `get_repo_root() -> Path` — repository root.
- `get_data_dir(name) -> Path` — packaged data directory (e.g. `subagents`, `skills`).
- `resolve_entry_script(entry_script) -> Path | None` — resolve a packaged entry script.
- `Model`, `Environment`, `Agent` — `Protocol` interfaces. `Model.query(messages, **kwargs) -> dict`; `Environment.execute(command, cwd="") -> dict`; `Agent.run(task, **kwargs) -> tuple[str, str]`.

### `minisweagent.config`

```python
from minisweagent.config import (
    builtin_config_dir, get_config_path, load_config, load_agent_config,
)
```

- `get_config_path(config_spec) -> Path` — resolve a config name/path.
- `load_config(config_spec) -> dict` — load a merged config dict.
- `load_agent_config(config_spec) -> tuple[dict, dict]` — agent + model sections.

### `minisweagent.models`

```python
from minisweagent.models import get_model, get_model_name, get_model_class

model = get_model(input_model_name=None, config=None)   # -> Model
name  = get_model_name(input_model_name=None, config=None)   # -> str
cls   = get_model_class(model_name, model_class="")          # -> type
```

`get_model` applies the resolution order in §2, sets Anthropic cache-control
defaults for Claude-family names, and honors `MSWEA_MODEL_API_KEY`.

### Agents

```python
from minisweagent.agents.gemm_tuning_agent import run_gemm_tuning_agent
```

`run_gemm_tuning_agent(...)` powers `geak-gemm-tuning`.

---

## See also

- [Quick start](quick_start.md) — install and first run.
- [Configuration files](configuration.md) — YAML loading and merge rules.
- [Model configuration](model_config.md) — model/backend details.
- [Subagent guide](subagent_guide.md) — skills and subagents.
- [Developer: MCP and native tools](developer/mcp-tools.md) — tool layer.
