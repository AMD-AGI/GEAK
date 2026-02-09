# GEAK Unified Branch — Integration Summary

**Branch:** `yab`  
**Base:** `geak_v3_features`  
**Merged from:** `msa`  
**Date:** February 2026

---

## What happened

We had two parallel branches developing different parts of the GEAK system:

- **`geak_v3_features`** — The agent framework. Strategy-based optimization, parallel execution, test generation via a subagent (UnitTestAgent), profiling tools, patch management. All capabilities built as tools inside the agent.

- **`msa`** — The tooling layer. Content-based test discovery pipeline, 6 standalone MCP servers (profiling, evolution, evaluation, etc.), Docker support, runtime environment detection, benchmarking framework, and a database of 50+ GPU optimization strategies.


---

## What's in the unified branch

### From geak_v3_features (kept as-is)

Everything that was in v3 is still there, unchanged:

- **7 agent types** — DefaultAgent, InteractiveAgent, StrategyAgent, ParallelAgent, SelectPatchAgent, UnitTestAgent, TextualAgent
- **Built-in tools** — bash, editor, test_perf, profiling, strategy_manager, submit
- **Config system** — Layered YAML configs (base → template → user override)
- **Parallel execution** — Multiple agents via git worktrees with GPU isolation
- **Strategy management** — Markdown-based tracking of optimization approaches
- **Unit test suite** — ~30 test files for agents, models, environments, configs

### From msa (added)

All of msa's unique capabilities were brought in:

| What | Where it lives | What it does |
|------|---------------|--------------|
| **6 MCP servers** | `mcp_tools/` | Standalone services for test discovery, kernel evolution (generate/mutate/crossover), kernel evaluation/reflection, GPU profiling, MetrixTool, OpenEvolve optimizer |
| **MCP client** | `mcp_tools/mcp-client/` | JSON-RPC client for talking to MCP servers |
| **Docker support** | `Dockerfile`, `entrypoint.sh`, `scripts/run-docker.sh` | Full containerized environment with all tools pre-installed, health checks on startup |
| **Runtime detection** | `src/minisweagent/runtime_env.py` | Auto-detects local vs Docker, checks GPU/ROCm/torch availability |
| **Optimizer** | `src/minisweagent/optimizer/` | Unified interface for OpenEvolve and Autotune |
| **Benchmarking** | `src/minisweagent/benchmark.py` | Standardized metrics (latency, FLOPS, bandwidth) for any kernel |
| **GPU profiling CLI** | `src/minisweagent/kernel_profile.py` | Command-line tool for hardware-level kernel profiling via MetrixTool |
| **MetrixTool** | `src/minisweagent/mcp_tools/metrix.py` | AMD Metrix API integration for detailed GPU metrics |
| **Optimization strategies** | `reference/optimization_strategies.py` | Database of 50+ AMD GPU strategies with code patterns and expected speedups |
| **Test suite** | `test_suite/` | Regression suite with 10 AITER kernels for end-to-end testing |
| **Examples** | `examples/add_kernel/` | Example kernel with runner script |
| **Docs** | `docs/DISCOVERY_PIPELINE.md`, `METRIX_TOOL.md`, `GETTING_STARTED.md`, `RUNTIME_ENV.md`, `RUNTIME_QUICKSTART.md` | Documentation for all ported features |

### Integrated (best-of-both-worlds)

One area where both branches had implementations for the same thing: **test discovery**. Rather than picking one, we combined them:

**How test discovery works now:**

1. When `--create-test` is set (or no `--test-command` is provided), the system first runs **msa's content-based discovery** — a fast regex/keyword pipeline that scans the workspace and scores files as potential tests or benchmarks. This takes ~1 second and costs nothing (no LLM call).

2. Whatever discovery finds (or doesn't find) is formatted and passed as context into **v3's UnitTestAgent** — an LLM-powered subagent that explores the repo and outputs a single test command.

3. The subagent always runs, but it starts informed. If discovery found good candidates, the agent validates and uses them (fewer exploration steps, lower token cost). If discovery found nothing, the agent creates tests from scratch — exactly as it did before.

This means the common case (test exists, discovery finds it) is fast and cheap, while the uncommon case (no tests exist) still works because the LLM can create them.

---

## CLI changes

The `mini` / `geak` command has three new flags:

```
--runtime auto|local|docker    Runtime environment (default: local)
--docker-image IMAGE           Docker image for --runtime=docker
--workspace PATH               Workspace to mount in Docker
```

The `geak` and `kernel-profile` commands are now registered as CLI entry points alongside `mini`.

---

## Docker workflow

```bash
# Build the image (first time)
cd GEAK
docker build --network=host -t geak-agent:yab .

# Run the container
AMD_LLM_API_KEY=<your-key> bash scripts/run-docker.sh

# Inside the container, all tools are available:
geak --help              # Main agent
kernel-profile --help    # GPU profiling
kernel-evolve --help     # Kernel evolution MCP CLI
kernel-ercs --help       # Kernel evaluation MCP CLI
```

The container runs health checks on startup and reports which tools are working.

---

## Using with AIG-Eval

```bash
# Inside the Docker container
cd /path/to/AIG-Eval/tasks/geak_eval

# Test a kernel
./run.sh rope --gpu 0

# Profile a kernel
./run.sh rope --profile --gpu 0

# Run full optimization
./run.sh rope --optimize --gpu 0
```

Note: The `run.sh` in AIG-Eval (branch `geak-eval-kernels`) was updated to call `geak` without specifying a config path — the default strategy-based config is auto-selected.

---

## Package naming

Internally, the package is still `minisweagent` (that's what all the code imports). The user-facing name is `geak-agent` (in pyproject.toml) with CLI commands `geak`, `mini`, and `kernel-profile`. This avoids a massive rename while keeping the public interface clean.

---

## What's NOT changed

- The original GEAK agent code on `main` (the GA/Reflexion-based kernel generation system with TritonBench dataloaders) is untouched. The `yab` branch is a completely separate line of work.
- The `geak_v3_features` branch itself is not modified — `yab` is a new branch based on it.
- The `msa` branch is not modified either.

---

## Known considerations

- **Python version**: The codebase uses Python 3.10+ features (match statements). The Docker base image ships Python 3.10. If running locally, ensure Python >= 3.10.
- **MetrixTool**: Requires the AMD `metrix` package (installed from AMD intellikit in the Dockerfile). Won't be available in non-AMD environments.
- **MCP servers**: These are standalone Python packages. They need to be `pip install -e`'d individually (the Dockerfile handles this). They're not required for basic agent operation — only if you want the kernel-evolve/kernel-ercs/kernel-profiler tools.
- **LLM API key**: Set `AMD_LLM_API_KEY` for LLM features. The UnitTestAgent, kernel-evolve, and kernel-ercs all need this.
