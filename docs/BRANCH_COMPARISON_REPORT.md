# Detailed Branch Comparison Report: `msa` vs `geak_v3_features`

## 1. Overview & Philosophy

### MSA Branch
- **Identity:** "GEAK Agent" — GPU Evolutionary Agent for Kernels.
- **Philosophy:** MCP-first, modular microservices. Every major capability (profiling, evolution, discovery, evaluation) is a standalone MCP server. The agent orchestrates them.
- **Lineage:** Fork of mini-SWE-agent; added `geak_agent/` layer on top of `src/geakagent/` (a rename of minisweagent). Maintains both a legacy discovery pipeline and a newer SWE-agent core.
- **Maturity:** 20 commits focused on discovery pipeline, MCP tool consolidation, Docker support, and runtime detection. Working end-to-end candidate but newer/less polished.

### geak_v3_features Branch
- **Identity:** "GEAK v3" — Built directly on top of mini-SWE-agent (kept as `minisweagent`).
- **Philosophy:** Agent-centric. Capabilities are built as tools *inside* the agent (strategy_manager, profiling_tools, editor, test_perf) rather than external MCP servers. Multi-agent architecture with subagents for specific tasks.
- **Lineage:** Clean fork of mini-SWE-agent; no `geak_agent/` or `mcp_tools/` directories. All new code is under `src/minisweagent/`.
- **Maturity:** Older commit history (original GEAK work), but the v3 features are polished with comprehensive README, tests, and config system.

---

## 2. Repository Structure

### MSA Branch
```
GEAK-agent/
├── geak_agent/                    # Discovery pipeline + metrix + CLI
│   ├── cli.py                     # geak-agent CLI entry point
│   ├── mcp_tools/
│   │   ├── discovery.py           # Content-based test/bench discovery (~1100 lines)
│   │   └── metrix.py              # MetrixTool for GPU profiling (~545 lines)
│   ├── examples/
│   └── tests/
├── mcp_tools/                     # 6 standalone MCP servers
│   ├── automated-test-discovery/  # MCP: discover(kernel_path) tool
│   ├── kernel-profiler/           # MCP: profile_kernel, benchmark_kernel, roofline
│   ├── kernel-evolve/             # MCP: generate_optimization, mutate, crossover
│   ├── kernel-ercs/               # MCP: evaluate_kernel, reflect, check_compatibility
│   ├── metrix-mcp/                # MCP: MetrixTool wrapper
│   ├── openevolve-mcp/            # MCP: OpenEvolve optimizer
│   └── mcp-client/                # JSON-RPC 2.0 client for MCP communication
├── src/geakagent/                 # Core agent (renamed minisweagent)
│   ├── agents/                    # default, interactive, textual, patch_agent
│   ├── config/                    # mini.yaml, mini_patch_agent.yaml, etc.
│   ├── environments/              # local, docker, singularity, bubblewrap, swerex
│   ├── models/                    # amd_llm, anthropic, litellm, openrouter, portkey
│   ├── optimizer/                 # Unified optimizer (OpenEvolve + Autotune)
│   ├── run/                       # mini.py entry point, inspector, swebench
│   ├── benchmark.py               # Standardized benchmarking framework
│   ├── kernel_profile.py          # CLI for GPU profiling
│   └── runtime_env.py             # Runtime environment detection (local/Docker)
├── reference/                     # Legacy reference code
│   ├── optimization_strategies.py # 50+ GPU optimization strategies database
│   └── state.py                   # State machine for optimization pipeline
├── test_suite/                    # Regression test suite (10 AITER kernels)
├── examples/add_kernel/           # Example kernel with runner script
├── Dockerfile                     # Docker image with all MCP tools
├── scripts/run-docker.sh
├── docs/
└── pyproject.toml                 # Scripts: geak, geak-agent, kernel-profile
```

### geak_v3_features Branch
```
GEAK-v3/
├── src/minisweagent/              # Everything lives here
│   ├── agents/                    # 7 agent types
│   │   ├── default.py             # Base agent (bash + tool calls)
│   │   ├── interactive.py         # Human-in-the-loop (confirm/yolo/human modes)
│   │   ├── interactive_textual.py # Textual UI agent
│   │   ├── strategy_agent.py      # Strategy tracking + UI callbacks
│   │   ├── strategy_interactive.py# CLI strategy display
│   │   ├── parallel_agent.py      # Parallel optimization with git worktrees
│   │   ├── select_patch_agent.py  # Best-patch selection from parallel runs
│   │   └── unit_test_agent.py     # Subagent for test discovery/generation
│   ├── tools/                     # 7 built-in tools
│   │   ├── bash_command.py        # Shell execution (with blocklist)
│   │   ├── editor_tool.py         # File editor (view/create/str_replace/insert)
│   │   ├── str_replace_editor.py  # Editor wrapper for tool calls
│   │   ├── profiling_tools.py     # rocprof-compute profiling (~600 lines)
│   │   ├── strategy_manager.py    # Strategy CRUD in markdown format (~600 lines)
│   │   ├── test_perf.py           # Patch save + test execution
│   │   ├── submit.py              # Agent termination
│   │   ├── tools_runtime.py       # Tool dispatcher
│   │   ├── tools.json             # Tool schemas (OpenAI function format)
│   │   └── prompt_for_profiling_analyzer.py
│   ├── config/                    # 10 YAML configs
│   │   ├── mini.yaml              # Base config
│   │   ├── mini_kernel.yaml       # Kernel optimization config
│   │   ├── mini_kernel_strategy_list.yaml # Strategy-based workflow
│   │   ├── geak.yaml              # GEAK model overrides
│   │   ├── mini_unit_test_agent.yaml      # Test discovery/generation subagent
│   │   ├── mini_select_patch.yaml         # Patch selection subagent
│   │   └── ...
│   ├── environments/              # local, docker, singularity, bubblewrap, swerex
│   ├── models/                    # amd_llm, anthropic, litellm, openrouter, portkey
│   ├── run/                       # mini.py entry point, inspector, swebench
│   │   └── utils/task_parser.py   # LLM-powered task parsing
│   └── utils/
├── tests/                         # Comprehensive test suite (~30 test files)
├── test_scripts/                  # Example test scripts
├── rocprim_prompts/               # ROCm-specific prompts
├── docs/                          # MkDocs documentation site (~40 files)
├── mkdocs.yml
└── pyproject.toml                 # Scripts: mini, mini-swe-agent, mini-extra
```

---

## 3. Agent Architecture

### MSA: 3 agents + MCP servers

| Agent | Purpose |
|-------|---------|
| `DefaultAgent` | Base: bash commands + MCP tool calls |
| `InteractiveAgent` | Adds user confirmation modes (confirm/yolo/human) |
| `PatchAgent` | Extends DefaultAgent with patch saving + test running + metric extraction |

The agent delegates heavy lifting to **external MCP servers**:
- Discovery → `automated-test-discovery` MCP
- Profiling → `kernel-profiler` MCP or `metrix-mcp`
- Optimization → `kernel-evolve` MCP or `openevolve-mcp`
- Evaluation → `kernel-ercs` MCP

### geak_v3_features: 7 agents + built-in tools

| Agent | Purpose |
|-------|---------|
| `DefaultAgent` | Base: bash commands + tool calls (test_perf, editor, profiling, strategy_manager, submit) |
| `InteractiveAgent` | Adds confirm/yolo/human modes |
| `StrategyAgent` | Adds strategy tracking with UI notification callbacks |
| `StrategyInteractiveAgent` | CLI strategy display with rich console |
| `ParallelAgent` | Runs N agents in parallel using git worktrees, isolated GPUs |
| `SelectPatchAgent` | LLM picks best patch from parallel runs, writes `best_results.json` |
| `UnitTestAgent` | Subagent: discovers or creates test scripts, returns `TEST_COMMAND` |

**Agent hierarchy:** DefaultAgent → InteractiveAgent → StrategyAgent → StrategyInteractiveAgent

All capabilities are **built-in tools**, not external servers.

---

## 4. Test Discovery

### MSA: Content-based pipeline

| Aspect | Details |
|--------|---------|
| **Implementation** | `geak_agent/mcp_tools/discovery.py` (~1100 lines of Python) |
| **Also as MCP** | `mcp_tools/automated-test-discovery/` (standalone server) |
| **Method** | Regex/keyword scoring: ~17 Python patterns, ~16 C++ patterns |
| **Auto-learning** | Samples 30 test files to learn custom decorators/imports |
| **Kernel matching** | +1.0 for exact name match, +0.3/part for partial |
| **LLM fallback** | Optional: for confidence 0.3–0.6, LLM classifies file |
| **Languages** | Python + C++ (GTest, Catch2, HIP/CUDA) |
| **Output** | Ranked list of TestInfo objects (path, command, confidence, type) |
| **Accuracy** | 90% on 30 AITER kernels |
| **Cost** | Free (unless LLM fallback triggered) |
| **Determinism** | Deterministic |

### geak_v3_features: LLM agent

| Aspect | Details |
|--------|---------|
| **Implementation** | `unit_test_agent.py` + `mini_unit_test_agent.yaml` (~50 lines Python + prompt) |
| **Method** | LLM agent explores repo via bash commands |
| **Prompt workflow** | Part 1: Read README, enumerate test/ and benchmark/, prefer existing |
| **Languages** | Whatever the LLM can read (Python, C++, etc.) |
| **Output** | Single `TEST_COMMAND: <command>` string |
| **Cost** | LLM tokens every run (claude-opus-4.5) |
| **Determinism** | Non-deterministic |

---

## 5. Test Generation

### MSA: Not implemented
- `[c] Create tests (I'll help)` stub exists in user confirmation menu
- No code path generates test files
- Listed as "Future Improvement" in DISCOVERY_PIPELINE.md

### geak_v3_features: Fully implemented (prompt-driven)
- UnitTestAgent's system prompt Part 2: "Creating a new test/benchmark script"
- Creates correctness test validating against torch reference
- Creates benchmark with 20+ iterations
- Covers multiple dtypes, shapes, edge cases
- Handles Type A (normal) vs Type B (fused kernel) scenarios
- Agent creates files via `cat <<'EOF' > test.py ...` in bash, fixes errors iteratively
- No separate Python test-generation code; entirely prompt engineering + agentic loop

---

## 6. Kernel Optimization / Generation

### MSA: MCP-based evolution

| MCP Server | Tools | Purpose |
|------------|-------|---------|
| **kernel-evolve** | `generate_optimization`, `mutate_kernel`, `crossover_kernels`, `get_optimization_strategies`, `suggest_kernel_params` | LLM-guided evolutionary optimization with explicit generate/mutate/crossover |
| **kernel-ercs** | `evaluate_kernel_quality`, `reflect_on_kernel_result`, `get_amd_gpu_specs`, `check_kernel_compatibility` | Quality scoring (9 criteria), reflection, AMD compatibility check |
| **openevolve-mcp** | `optimize_kernel` | Full OpenEvolve integration (population_size=20, num_islands=4) |
| **kernel-profiler** | `profile_kernel`, `benchmark_kernel`, `get_roofline_analysis`, `get_bottleneck_suggestions` | Hardware profiling via rocprof-compute inside Docker |

Additionally:
- `src/geakagent/optimizer/core.py`: Unified optimizer interface (OpenEvolve + Autotune auto-selection)
- `src/geakagent/benchmark.py`: Standardized benchmark framework (latency, FLOPS, bandwidth)
- `reference/optimization_strategies.py`: Database of 50+ AMD GPU optimization strategies with code patterns and expected speedups

### geak_v3_features: In-agent-loop optimization

| Component | Purpose |
|-----------|---------|
| **Strategy Manager tool** | Track strategies in `.optimization_strategies.md` (CRUD: create, add, mark, update, remove, show, next, note, summary) |
| **Profiling tool** | rocprof-compute parsing with roofline, top kernels, L1/L2 cache, compute units analysis |
| **test_perf tool** | Save patch + run test + report pass/fail |
| **Editor tool** | View, create, str_replace, insert with undo + flake8 linting |
| **Parallel Agent** | N agents in parallel via git worktrees, each on isolated GPU |
| **Select Patch Agent** | LLM picks best patch, writes `best_results.json` with speedup |

Workflow (from `mini_kernel_strategy_list.yaml`):
1. Query hardware info (GPU arch, CU count, memory bandwidth)
2. Establish baseline with `test_perf`
3. Profile with `profiling` tool to identify bottlenecks
4. Create strategy list with `strategy_manager`
5. Explore strategies one by one, measuring impact
6. Combine successful strategies

No MCP servers. No explicit generate/mutate/crossover. Optimization is "edit code → run test → check results → iterate."

---

## 7. Profiling

### MSA: Two paths

1. **MetrixTool** (`geak_agent/mcp_tools/metrix.py`, ~545 lines):
   - Uses AMD Metrix Python API directly
   - Multi-GPU support
   - Quick (3 metrics) vs full (14 metrics)
   - Bottleneck classification (latency, memory, compute, LDS, balanced)
   - Auto-detects GPU specs
2. **kernel-profiler MCP** (`mcp_tools/kernel-profiler/`):
   - rocprof-compute profiling inside Docker containers
   - Roofline analysis with AMD MI350X theoretical peaks
   - Bottleneck suggestions by type
3. **metrix-mcp**: Thin MCP wrapper around MetrixTool

### geak_v3_features: Built-in profiling tool

- `src/minisweagent/tools/profiling_tools.py` (~600 lines):
  - Runs rocprof-compute directly (auto-installs if missing)
  - Parses text output: roofline rates, top kernels, system info, speed-of-light, compute units, L1/L2 cache
  - Supports profiling types: roofline, profiling, profiler_analyzer
- `prompt_for_profiling_analyzer.py`: LLM prompt for analyzing profiling output (bottleneck identification, optimization directions)

---

## 8. Runtime & Environment

### MSA
- **`runtime_env.py`** (~316 lines): Auto-detects local vs Docker environment
  - Checks for GPU (ROCm/CUDA), torch, triton availability
  - Interactive prompt with Rich console UI
  - Default Docker image: `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x`
  - Workspace mounting in Docker
- **Dockerfile**: Full image with all MCP tools + Metrix from AMD intellikit
- **`scripts/run-docker.sh`**: Docker run helper
- **Environments**: local, docker, singularity, bubblewrap, swerex_docker

### geak_v3_features
- **No runtime auto-detection** or Docker management code
- **Environments**: Same set (local, docker, singularity, bubblewrap, swerex_docker)
- **No Dockerfile** in this branch
- Relies on user having the correct environment set up

---

## 9. CLI & Entry Points

### MSA: Two CLIs

| Command | Source | Purpose |
|---------|--------|---------|
| `geak-agent` | `geak_agent/cli.py` | Discovery → context building → agent launch |
| `geak` / `mini` | `src/geakagent/run/mini.py` | Main SWE-agent with runtime detection |
| `kernel-profile` | `src/geakagent/kernel_profile.py` | Standalone GPU profiling |

### geak_v3_features: One CLI

| Command | Source | Purpose |
|---------|--------|---------|
| `mini` | `src/minisweagent/run/mini.py` | Main entry point with extensive options |

Key `mini` options:
- `--visual / -v`: Toggle Textual UI
- `--model / -m`: LLM model
- `--task / -t`: Task description (or file path)
- `--yolo / -y`: Auto-execute without confirmation
- `--enable-strategies / --no-enable-strategies`: Strategy management
- `--strategy-file`: Path to strategy markdown
- `--test-command`: Explicit test command
- `--create-test`: Auto-create via UnitTestAgent
- `--patch-output`: Patch save directory
- `--metric`: Metric extraction description
- `--num-parallel`: Number of parallel agents
- `--repo`: Repository for parallel execution
- `--gpu-ids`: Comma-separated GPU IDs

---

## 10. Model Support

Both branches have the same model backends:

| Model Backend | Both Branches |
|---------------|---------------|
| `amd_llm` | AMD LLM Gateway (Claude, GPT, Gemini via AMD proxy) |
| `anthropic` | Direct Anthropic API |
| `litellm` | LiteLLM (multi-provider) |
| `openrouter` | OpenRouter API |
| `portkey` | Portkey AI Gateway |
| `test_models` | Mock models for testing |

v3 additionally has:
- Tool format conversion (OpenAI → Claude/Gemini) in `amd_llm.py`
- Retry logic with exponential backoff
- Cost tracking per model call
- Supports reasoning/text verbosity for GPT models

---

## 11. Parallel Execution

### MSA: Not present
- No parallel agent
- No git worktree isolation
- Single-agent execution only

### geak_v3_features: Full parallel support
- `ParallelAgent` creates N git worktrees (or directory copies for non-git repos)
- Each agent gets isolated GPU via `HIP_VISIBLE_DEVICES`
- Patches saved to `parallel_N/` subdirectories
- `SelectPatchAgent` uses LLM to pick best patch across all parallel runs
- Outputs `best_results.json` with speedup calculation

---

## 12. Strategy Management

### MSA: Reference only
- `reference/optimization_strategies.py`: Static database of 50+ strategies
  - Organized by bottleneck type (latency, memory, compute, LDS, etc.)
  - Includes code patterns and expected speedup ranges
  - AMD-specific (MI300 XCD awareness, waves_per_eu)
  - Not actively used by agent — reference material
- `reference/state.py`: State machine (INIT → DISCOVERING → AUTONOMOUS → CHECKPOINT → ...) — also reference, not integrated

### geak_v3_features: Active strategy tool
- `strategy_manager.py` (~600 lines): Full CRUD tool
  - Markdown-based storage (`.optimization_strategies.md`)
  - Statuses: BASELINE, PENDING, EXPLORING, SUCCESSFUL, FAILED, PARTIAL, SKIPPED, COMBINED
  - Priority: high/normal
  - Commands: create, add, mark, update, remove, show, next, note, summary
  - UI callback mechanism for StrategyAgent
  - Human-readable format (the agent reads/writes markdown)

---

## 13. Patch Management

### MSA: PatchAgent
- `patch_agent.py` (~311 lines)
- Triggered by `SAVE_PATCH_AND_TEST` in command output
- Captures git diff, runs test, saves results
- LLM-based metric extraction from test output
- LLM-based best patch selection from history
- Saves to `results.json`

### geak_v3_features: test_perf tool + SelectPatchAgent
- `test_perf.py` (~200 lines): Tool callable by agent
  - Captures git diff or directory diff
  - Runs test command
  - Saves `.patch` and `_test.txt` files
  - Formats output for agent consumption
- `select_patch_agent.py` (~100 lines): Subagent for multi-run selection
  - Analyzes patches from parallel agents
  - Writes `best_results.json` with speedup, analysis
  - Validates metric alignment

---

## 14. Documentation

### MSA
- `README.md` (~118 lines): Quick start, architecture diagram, installation
- `RUNTIME_ENV.md`, `RUNTIME_QUICKSTART.md`: Runtime/Docker docs
- `docs/DISCOVERY_PIPELINE.md`: Discovery architecture + 90% accuracy results
- `docs/GETTING_STARTED.md`, `docs/METRIX_TOOL.md`: Tool docs
- Individual README per MCP tool

### geak_v3_features
- `README.md` (~600 lines): Comprehensive — features, config system, workflows, best practices
- Full MkDocs site (`mkdocs.yml` + `docs/`): ~40 doc files
  - Quick start, FAQ, contributing guide
  - Reference docs for all agents, environments, models
  - Advanced: control flow, cookbook, environments, YAML config
  - Model troubleshooting
- `rocprim_prompts/`: ROCm-specific optimization prompts

---

## 15. Testing

### MSA
- `geak_agent/tests/test_metrix.py`: MetrixTool test
- `test_suite/`: Regression suite with 10 AITER kernels
  - `config.json`: Kernel definitions (gemm, attention, moe, normalization, etc.)
  - `run_suite.py`: Full pipeline runner (discovery → test gen → benchmark → optimize)

### geak_v3_features
- `tests/`: ~30 test files covering:
  - Agents (default, interactive, multimodal)
  - Config loading
  - Environments (local, docker, singularity, bubblewrap, swerex)
  - Models (litellm, anthropic, openrouter, portkey, cache control)
  - Run (CLI, inspector, hello_world, swebench)
  - Utils
- `test_scripts/test_correctness_benchmark.py`: Example test pattern

---

## 16. Summary: Feature Matrix

| Feature | MSA | v3 |
|---------|:---:|:--:|
| **Test Discovery** | Content-based pipeline (deterministic, free, 90% accurate) | LLM agent (non-deterministic, costs tokens) |
| **Test Generation** | Stub only | Fully implemented (prompt-driven) |
| **Kernel Evolution** | MCP: generate/mutate/crossover | Not present |
| **OpenEvolve** | MCP server integration | Not present |
| **Profiling** | MetrixTool + kernel-profiler MCP | Built-in rocprof parsing |
| **Strategy Tracking** | Reference database (static) | Active markdown-based tool |
| **Parallel Execution** | Not present | Full (git worktrees, GPU isolation) |
| **Patch Management** | PatchAgent (LLM metric extraction) | test_perf tool + SelectPatchAgent |
| **Runtime Detection** | Auto-detect local/Docker | Not present |
| **Dockerfile** | Yes (with all MCP tools) | No |
| **MCP Servers** | 6 servers + client | None |
| **Agent Types** | 3 | 7 |
| **Built-in Tools** | Bash only (delegates to MCP) | 6 tools (bash, editor, test_perf, profiling, strategy_manager, submit) |
| **Config System** | Basic (mini.yaml) | Layered (base → template → user override) |
| **Documentation** | Basic README + per-tool docs | Comprehensive README + MkDocs site |
| **Unit Tests** | 1 test file + test suite | ~30 test files |
| **CLI Entry Points** | 3 (geak, geak-agent, kernel-profile) | 1 (mini, with many options) |

---

## 17. Key Differences in Design Philosophy

| Dimension | MSA | v3 |
|-----------|-----|-----|
| **Tool architecture** | External MCP servers (microservices) | Built-in tools (monolith) |
| **Discovery approach** | Code-based (regex, scoring, deterministic) | LLM-based (agentic, non-deterministic) |
| **Optimization approach** | Explicit evolutionary operations (generate/mutate/crossover via MCP) | Implicit via agent loop (edit → test → iterate) with strategy tracking |
| **Parallelism** | None | First-class (parallel agents, worktrees, patch selection) |
| **Configuration** | Simple | Layered with auto-detection and LLM-powered task parsing |
| **State management** | Reference state machine (not integrated) | Active strategy file read/written by agent |
| **Deployment** | Docker-first with runtime detection | Assumes local environment |
| **Extensibility** | Add new MCP server | Add new tool or agent subclass |

---

## 18. Recommended Integration Path

### Phase 1: Combine test discovery + generation
- Use MSA's content-based discovery (fast, free, 90% accurate)
- Fall back to v3's UnitTestAgent when discovery finds nothing (creates tests)

### Phase 2: Add v3's parallel + strategy to MSA's MCP ecosystem
- Add ParallelAgent + SelectPatchAgent to MSA
- Add strategy_manager tool
- Keep MSA's MCP servers as external tools the agent can call

### Phase 3: Unified CLI
- One `geak` CLI that:
  1. Auto-detects runtime (MSA's runtime_env)
  2. Discovers tests (MSA's pipeline)
  3. Creates tests if needed (v3's UnitTestAgent)
  4. Runs optimization with strategies (v3's strategy system)
  5. Optionally calls MCP tools (MSA's kernel-evolve, kernel-ercs, etc.)
  6. Supports parallel execution (v3's ParallelAgent)
