# GEAK

GEAK is an agent-driven framework for end-to-end GPU kernel optimization in real codebases, producing reviewable patches backed by profiling, testing, and LLM-guided iteration.

---

## Core Capabilities

- End-to-end optimization
  - Automatically discovers or generates tests and harness scripts
  - Runs a closed loop: profiling → optimization → validation
  - Produces reproducible patches and benchmarks

- Repository-level workflows
  - Works beyond single kernels — supports full-repository optimization (L3)
  - Maintains a consistent test, baseline, and evaluation pipeline

- Parallel exploration
  - Multi-agent search across isolated Git workspaces
  - Improves robustness by exploring multiple optimization strategies

---

## Architecture

![GEAK Framework](docs/assets/GEAK_framework.png)


---

## Getting Started

### Installation

```bash
git clone https://github.com/AMD-AGI/GEAK
cd GEAK
# Docker-based
AMD_LLM_API_KEY=<YOUR_KEY> bash scripts/run-docker.sh

# Local install (recommended)
make install              # core + MCP tools (same as Docker)
make install-full         # + dev tools + swe-rex
make install-dev          # install-full, editable (for developers)

# (or) pip-only install without registering MCP tools as packages
pip install -e .          # core package, including MCP runtime dependencies
pip install -e '.[full]'  # core + dev + langchain + swe-rex

# (optional) RAG index build: if enable RAG, build index after make install
make index

# Set model name and key. In the case of docker-based setup, export the API key before
# running scripts/run-docker.sh.

# Option 1: set a LiteLLM model + provider API key
export MSWEA_MODEL_NAME="openai/gpt-5"
export OPENAI_API_KEY="YOUR_KEY"

# Anthropic example
export MSWEA_MODEL_NAME="anthropic/claude-sonnet-4-5-20250929"
export ANTHROPIC_API_KEY="YOUR_KEY"

# Option 2: If you use AMD LLM Gateway (model_class: amd_llm)
export AMD_LLM_API_KEY="YOUR_KEY"
```

### Usage

#### Basic (single-agent) GPU kernel optimization

```bash
# Typical kernel optimization using natural language input
geak -t "Optimize the kernel from /path/to/aiter, specifically aiter/ops/triton/topk.py. Use the harness at /path/to/test_topk_harness.py. Use four GPUs with IDs 0-3 simultaneously."

# Typical kernel optimization (single agent)
geak --kernel-url /path/to/kernel/file \
  --repo /path/to/kernel/repo \
  --task "Optimize the block_reduce kernel"
```

#### Parallel optimization (multiple agents)

- Each agent works in an isolated git workspace
- Patches and test results are saved separately
- After all runs finish, GEAK automatically selects the best patch based on the specified metric

```bash
geak --repo /path/to/kernel/repo \
  --kernel-url /path/to/kernel/file \
  --task "Optimize the block_reduce kernel. Kernel path: xxx. Metric: bandwidth in GB/s (higher is better)." \
  --num-parallel 4 \
  --gpu-ids 0,1,2,3
```

**Notes:**

- `--repo`: required; the target repository path
- `--kernel-url`: required; path to the target kernel file (local path or URL)
- `--num-parallel`: number of optimization agents
- `--gpu-ids`: comma-separated GPU IDs for agents
- `--apply-best-patch` / `--no-apply-best-patch` (default: on): after the run completes
  *without raising*, apply the winning patch to `--repo` on the current branch and commit
  it with a message pointing at `final_report.json`. Requires a clean working tree on
  `--repo`. A dirty tree is skipped in the log AND printed in yellow on the console so
  the no-op isn't easy to miss. When the repo has no configured git identity (typical
  inside the GEAK container), the commit falls back to `GEAK Agent <geak@amd.com>`;
  override via the `GEAK_GIT_AUTHOR_NAME` / `GEAK_GIT_AUTHOR_EMAIL` environment variables
  (passed through by `entrypoint.sh`).
- `--cleanup` / `--no-cleanup` (default: on): runs at every cooperative exit — success,
  exception during preprocess or run, and the Ctrl-C escalation path — and prunes
  per-run artifacts to the **keep-set**: `final_report.json`, the winning `.diff`,
  `geak_agent.log`, and `COMMANDMENT.md`. A Ctrl-C *during* cleanup leaves it partial
  (cleanup wraps in `try/except Exception` and `KeyboardInterrupt` is a `BaseException`,
  not an `Exception`). Hard-kill (wall-clock timeout) leaves the per-run dir alone for
  forensic analysis regardless of `--cleanup`, and prints a loud red console line
  naming the artifact path. Independent of `--apply-best-patch`.
- `--keep-runs N` / `GEAK_KEEP_RUNS=N` (default: unlimited): after this run completes,
  retain only the N most-recent auto-generated run dirs under output_dir's parent.
  **Scope is `<cwd>/optimization_logs/`** — runs from other working directories are not
  visible. Ignored under `--output` (a warning is logged). Concurrent runs in the same
  parent are protected by a stale-only filter that prefers the agent log's mtime, but
  the safest mode is one `geak` invocation at a time.
- **Budget semantics.** `--mode quick` and `--mode full` are **absolute wall-clock caps**
  of 60 min and 120 min respectively. The hard-kill watchdog `os._exit(124)`s at
  `started_at + total_s` exactly — this anchor enforces the cap. Internally the
  cooperative budget reserves ~60 s for finalize headroom; this is invisible to the
  user and is not load-bearing for the cap promise.
- **Operator log surface.** Every cooperative exit logs `[geak --cleanup] starting` and
  `[geak --cleanup] completed: <status>`. If the completed line is missing, cleanup was
  interrupted mid-flight (Ctrl-C during cleanup, or a rare race with hard-kill).
- **Persistent caches not affected by `--cleanup`.** `~/.cache/amd-ai-devtool/semantic-index`
  (the RAG FAISS index) and the auto-installed `rag-mcp` package. Both are intentional
  caches reused across runs.
- **Cooperative shutdown coverage** is tracked separately: the wall-clock cap is
  enforced absolutely by the hard-kill watchdog, but reaching it should be rare. The
  `subprocess.run` / LLM-client timeouts in the run path should all clamp via
  `deadline.cap(...)`; a focused audit of those call sites is a follow-up.

### Runnable examples

These are **examples** you can test in `examples/`. Replace paths, GPU IDs, and the metric wording as needed.

**Example: HIP kernel `knn`**

```bash
# Repo root containing the kernel file
REPO="/path/to/GEAK/examples/knn"

geak --repo "$REPO" \
  --kernel-url "$REPO/knn_wrapper.py" \
  --test-command "python scripts/task_runner.py compile && python scripts/task_runner.py correctness && python scripts/task_runner.py performance" \
  --task "Optimize the knn kernel. Metric: latency (lower is better)." 
```

**Example: Triton kernel `mla_decode`**

```bash
REPO="/path/to/GEAK/examples/mla_decode"

geak --repo "$REPO" \
  --kernel-url "$REPO/kernel.py" \
  --test-command "python3 -c \"import ast; ast.parse(open('kernel.py').read())\" && python3 'test_kernel_harness.py' --correctness && python3 'test_kernel_harness.py' --full-benchmark" \
  --task "Optimize the MLA decode Triton kernel." 
```

**Example: FlyDSL kernel `preshuffle_gemm`**

```bash
# FlyDSL repo root (requires `pip install flydsl` in the environment)
REPO="/path/to/FlyDSL"

geak --repo "$REPO" \
  --kernel-url "$REPO/kernels/preshuffle_gemm.py" \
  --task "Optimize the preshuffle GEMM kernel." \
  --yolo --exit-immediately
```

For more options and examples, see **[Quick start](docs/quick_start.md)**.


### Configuration

#### Loading Configurations

`geak` loads configuration in layers:

1. base config: `src/minisweagent/config/geak.yaml`
2. template: `src/minisweagent/config/mini_kernel_strategy_list.yaml` (default)
3. user override: `--config xxx.yaml`
4. CLI override: CLI args (final override)

For more options and examples, see [Configuration](docs/configuration.md).


### Output & Artifacts

GEAK saves patches and test logs so the optimization progress and results remain transparent.

- **Default output base**: `optimization_logs/`
- **Auto-generated run directory**: `optimization_logs/<kernel_name>_<YYYYmmdd_HHMMSS>/`

Typical structure (parallel run):

```bash
optimization_logs/<kernel>_<timestamp>/
├── results/round_1/<kernel>-<strategy_0>/
│   ├── patch_0.patch
│   ├── patch_0_test.txt
│   └── task_0.log
├── tasks/round_1/<kernel>-<strategy_1>/
│   └── ...
├── final_report.json
└── geak_agent.log
```

---

## Features

---

### Unit Test Discovery

If `--test-command` is not provided, GEAK will:

- Discover existing tests, or
- Create a validated harness via UnitTestAgent

The resulting test command is fixed for the entire run, ensuring consistent evaluation.

---

### Patch-Driven Optimization

- Patch-based iteration — every step produces a reproducible diff
- Metric-guided — optimize for task-defined or custom metrics
- Strategy-aware — tracks and prioritizes effective optimization directions

---

### Parallel Scaling

Run multiple agents with:

```bash
--num-parallel N
```

- Isolated git worktrees per agent
- Optional GPU pinning via `--gpu-ids`
- Improves exploration and reduces LLM variance

---

### Best Patch Selection

Automatically selects the best result across runs:

- Reads all patch logs under `optimization_logs/`
- Applies user-defined metric
- Outputs `best_results.json`

---

### Tooling Layer

- Kernel Profile — bottleneck analysis (bandwidth, occupancy, etc.)
- RAG — GPU knowledge retrieval (AMD / NVIDIA)
  - See **[RAG MCP Server](mcp_tools/rag-mcp/README.md)** for configuration and usage details.
- In-session strategy tracking — full history, reproducibility, rollback
- Cross-session memory — reuse past optimization insights

  | Flag | Default | What it does |
  |------|---------|--------------|
  | `GEAK_MEMORY_DISABLE=1` | off | Turn off all memory (within-session + cross-session) |
  | `GEAK_USE_KNOWLEDGE_BASE=0` | on | Turn off reading past insights from the knowledge base |
  | `GEAK_SAVE_TO_KNOWLEDGE_BASE=1` | off | Turn on saving run insights to the knowledge base after each run |
  | `GEAK_MEMORY_MIN_SPEEDUP=1.10` | 1.10 | Minimum speedup required to save an experience |

  By default, the knowledge base is **read** (agents see past insights) but **not written** (run results are not saved back). Set `GEAK_SAVE_TO_KNOWLEDGE_BASE=1` to start building the knowledge base from your runs.

## Contributing

We appreciate all contributions. If you are planning to contribute bug fixes, feel free to open a pull request without opening an issue first.

If you plan to contribute **new features, utility functions, or changes to the core**, please **open an issue first** and discuss the design with us. A pull request sent without that discussion may be **closed or not merged** if it diverges from the direction we are taking the project.

For branching, pull requests, code standards, CI expectations, releases, and licensing details, see **[Contribution guidelines](docs/developer/contribution_guidelines.md)**.

## Acknowledgments

GEAK extends [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent) — the agent loop, environment tooling, and SWE-style workflows. For upstream behavior and APIs, see the [mini-SWE-agent documentation](https://mini-swe-agent.com/latest/).

We also thank:

- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM routing used by model backends
- [Typer](https://github.com/tiangolo/typer) & [Rich](https://github.com/Textualize/rich) — CLI and terminal UX
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) ecosystem (e.g., `mcp`, FastMCP) — tool servers for profiling, metrics, and discovery
- [LangChain](https://github.com/langchain-ai/langchain) (optional `[langchain]` extra) — hybrid retrieval for the GPU knowledge path
- AMD Research [IntelliKit](https://github.com/AMDResearch/intellikit) (`metrix`) — GPU profiling metrics integration

Dependencies and versions are listed in `pyproject.toml`; all third-party software remains under their respective licenses.