# Merge Strategy: Unifying `msa` and `geak_v3_features`

## 1. Decision: Which branch is the base?

**Recommendation: Use `geak_v3_features` as the base, port MSA features into it.**

### Guiding principles

1. **Decoupled things go in as-is.** MCP servers, Dockerfile, runtime detection,
   optimizer, benchmark, reference strategies — these are standalone modules with no
   overlap in v3. Copy them directly.

2. **Coupled things get best-of-both-worlds integration.** Where both branches have
   implementations (test discovery, profiling), we combine them so each piece plays
   its strength. Specifically for test discovery: MSA's content-based pipeline runs
   first as a fast pre-scan, and its results are **fed into** v3's UnitTestAgent as
   context — the subagent always runs, but starts informed rather than from scratch.

Rationale:
- v3 has the richer agent architecture (7 agents vs 3), parallel execution, strategy management, and test generation — the hardest pieces to build.
- v3 has comprehensive unit tests (~30 files) and a full MkDocs documentation site.
- v3's config system (layered: base → template → user override) is more mature.
- MSA's unique strengths (content-based discovery, MCP servers, Dockerfile, runtime detection) are modular and can be ported in without touching v3's core.

Renaming: MSA renamed `minisweagent` → `geakagent`. The merge should settle on one name. Since v3 has more code and tests referencing `minisweagent`, and MSA already imports from both, picking one early avoids churn. **Recommendation: keep `minisweagent` as the internal package name** (it's what tests and configs reference) and use `geak` / `geak-agent` as the user-facing CLI names.

---

## 2. Merge phases

### Phase 1: Test Discovery Pipeline (from MSA → v3)

**What to port:**
- `geak_agent/mcp_tools/discovery.py` (~1100 lines) — the `DiscoveryPipeline` class
- `mcp_tools/automated-test-discovery/` — standalone MCP server (optional, for external tool use)

**Where it goes in v3:**
- Create `src/minisweagent/tools/discovery.py` containing the `DiscoveryPipeline` class.
- Alternatively, place it under `src/minisweagent/discovery/` as a subpackage.

**Design: Discovery feeds into UnitTestAgent (not replaces it)**

The approach is: when the user sets `--create-test`, MSA's content-based discovery
runs first (fast, free), and whatever it finds is **fed into** the UnitTestAgent's
task prompt as context. The subagent always runs — but instead of exploring from
scratch, it starts with the discovery results and can:
- Validate and use a high-confidence discovered test directly
- Choose between multiple discovered candidates intelligently
- Create new tests if discovery found nothing (its existing strength)
- Augment a discovered correctness test with a missing benchmark, or vice versa

This gives the LLM the best starting point while preserving its ability to reason,
create, and fix.

**Integration point — modify `mini.py` and `unit_test_agent.py`:**

```python
# In src/minisweagent/run/mini.py

from minisweagent.tools.discovery import DiscoveryPipeline

def resolve_test_command(kernel_path, repo, model, create_test):
    """
    1. Run content-based discovery (fast, free).
    2. Feed results into UnitTestAgent as context.
    3. UnitTestAgent validates, selects, or creates tests.
    """
    discovery_context = ""

    if kernel_path:
        pipeline = DiscoveryPipeline(workspace_path=repo or kernel_path.parent)
        result = pipeline.run(kernel_path=kernel_path, interactive=False)

        if result.tests or result.benchmarks:
            discovery_context = _format_discovery_for_agent(result)

    if create_test:
        from minisweagent.agents.unit_test_agent import run_unit_test_agent
        return run_unit_test_agent(
            model=model,
            repo=repo,
            kernel_name=kernel_path.stem,
            discovery_context=discovery_context,  # NEW: pass discovery results
        )

    return None


def _format_discovery_for_agent(result) -> str:
    """Format discovery results as context for the UnitTestAgent prompt."""
    lines = []
    lines.append("## Pre-Discovery Results (automated content-based scan)")
    lines.append("")

    if result.tests:
        lines.append("### Discovered Test Files (ranked by confidence):")
        for i, t in enumerate(result.tests[:5], 1):
            conf_pct = min(int(t.confidence * 100), 100)
            lines.append(f"  {i}. `{t.file_path}` — {t.test_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{t.command}`")
        lines.append("")

    if result.benchmarks:
        lines.append("### Discovered Benchmark Files (ranked by confidence):")
        for i, b in enumerate(result.benchmarks[:5], 1):
            conf_pct = min(int(b.confidence * 100), 100)
            lines.append(f"  {i}. `{b.file_path}` — {b.bench_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{b.command}`")
        lines.append("")

    if not result.tests and not result.benchmarks:
        lines.append("No existing tests or benchmarks were found by the automated scan.")
        lines.append("You will need to create them from scratch.")
        lines.append("")

    lines.append("Use these results as a starting point. Validate any discovered")
    lines.append("tests/benchmarks before using them. Create new ones if none are suitable.")

    return "\n".join(lines)
```

```python
# In src/minisweagent/agents/unit_test_agent.py — modified run_unit_test_agent

def run_unit_test_agent(
    *,
    model: Model,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None = None,
    discovery_context: str = "",  # NEW parameter
) -> str:
    """Run UnitTestAgent in `repo` and return the extracted test command string."""
    config_path = get_config_path("mini_unit_test_agent")
    config = yaml.safe_load(config_path.read_text())
    agent_config = config.get("agent", {})

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)
    agent = UnitTestAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "unit_test_agent.log"

    # Build task with discovery context injected
    task = f"Find or create unit/benchmark tests for kernel: {kernel_name}\nRepository: {repo}"
    if discovery_context:
        task += f"\n\n{discovery_context}"

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(f"UnitTestAgent did not finish successfully: {exit_status}\n{result}")

    return _extract_test_command(result)
```

**What happens at runtime:**

```
User: mini --create-test --task "Optimize kernel at /path/to/gemm.py"

Step 1: DiscoveryPipeline scans workspace (fast, ~1s, no LLM)
        → Found: test_gemm.py (pytest, 95% confidence)
        → Found: bench_gemm.py (triton.testing.do_bench, 80% confidence)

Step 2: UnitTestAgent launches with task prompt:
        "Find or create unit/benchmark tests for kernel: gemm
         Repository: /path/to/repo

         ## Pre-Discovery Results (automated content-based scan)

         ### Discovered Test Files (ranked by confidence):
           1. `test/test_gemm.py` — pytest, 95% confidence
              Suggested command: `pytest test/test_gemm.py -v`

         ### Discovered Benchmark Files (ranked by confidence):
           1. `benchmark/bench_gemm.py` — triton, 80% confidence
              Suggested command: `python benchmark/bench_gemm.py`

         Use these results as a starting point. Validate any discovered
         tests/benchmarks before using them. Create new ones if none are suitable."

Step 3: UnitTestAgent reads the discovered files, validates them,
        confirms they're suitable, and outputs:
        TEST_COMMAND: pytest test/test_gemm.py -v && python benchmark/bench_gemm.py

        OR if discovery found nothing:
        UnitTestAgent creates tests from scratch (its existing Part 2 behavior).
```

**Why this is better than the previous "fallback" approach:**
- The subagent always runs, so it can *validate* discovery results (e.g. "this test file
  exists but it's for a different kernel variant")
- The subagent can combine discovery results intelligently (e.g. "correctness test found
  but no benchmark — I'll create just the benchmark")
- Discovery gives the LLM a head start: fewer bash exploration steps, lower token cost
- When discovery finds nothing, the subagent still works exactly as before

**Conflicts expected:** None. v3 has no `discovery.py`. Changes to `unit_test_agent.py`
are additive (one new optional parameter).

**Tests to add:**
- Port `geak_agent/tests/test_metrix.py` patterns to `tests/tools/test_discovery.py`.
- Add integration test with a sample kernel directory.
- Test `run_unit_test_agent` with and without `discovery_context`.

---

### Phase 2: MCP Servers (from MSA → v3)

**What to port:**
- `mcp_tools/` directory (all 6 servers + client)
- `geak_agent/mcp_tools/metrix.py` (MetrixTool)

**Where it goes in v3:**
- Add `mcp_tools/` as a top-level directory (same location as MSA).
- Add `src/minisweagent/mcp_tools/` with `metrix.py` and `__init__.py`.

**Integration approach — additive, not replacing:**

The MCP servers are standalone processes. They don't conflict with v3's built-in tools. The agent can call them *in addition to* built-in tools.

Option A (simple): Add MCP server definitions to Cursor/IDE MCP config. The agent calls them as tool calls naturally.

Option B (programmatic): Add an MCP client tool to v3's tool runtime:
```python
# In tools_runtime.py, add optional MCP tool dispatch
if tool_name.startswith("mcp:"):
    server_name, tool_name = tool_name.split(":", 2)[1:]
    return await mcp_client.call_tool(server_name, tool_name, args)
```

**Specific MCP servers and their relationship to v3 built-in tools:**

| MSA MCP Server | v3 Built-in Equivalent | Merge Action |
|----------------|----------------------|--------------|
| `automated-test-discovery` | None (UnitTestAgent is different) | **Add** — complements UnitTestAgent |
| `kernel-profiler` | `profiling_tools.py` (similar) | **Keep both** — MCP version runs in Docker, built-in runs locally |
| `kernel-evolve` | None | **Add** — new capability (generate/mutate/crossover) |
| `kernel-ercs` | None | **Add** — new capability (evaluation/reflection) |
| `metrix-mcp` | `profiling_tools.py` (overlaps) | **Add** — MetrixTool uses AMD API, profiling_tools uses rocprof-compute |
| `openevolve-mcp` | None | **Add** — new capability (OpenEvolve integration) |
| `mcp-client` | None | **Add** — needed to talk to any MCP server |

**Conflicts expected:** Minimal. `mcp_tools/` doesn't exist in v3. The only watch point is `pyproject.toml` dependencies.

---

### Phase 3: Runtime Environment & Docker (from MSA → v3)

**What to port:**
- `src/geakagent/runtime_env.py` → `src/minisweagent/runtime_env.py`
- `Dockerfile`
- `entrypoint.sh`
- `scripts/run-docker.sh`
- `RUNTIME_ENV.md`, `RUNTIME_QUICKSTART.md`

**Integration point — modify `mini.py`:**

Add `--runtime` and `--docker-image` CLI flags (from MSA's cli.py):

```python
# Additional options for mini.py
runtime: str = typer.Option("auto", "--runtime", help="Runtime: local, docker, auto")
docker_image: str = typer.Option(None, "--docker-image", help="Docker image to use")
workspace: Path = typer.Option(None, "--workspace", help="Workspace to mount in Docker")
```

Before agent starts, detect/select runtime:
```python
from minisweagent.runtime_env import detect_runtime_environment, prompt_runtime_environment

if runtime == "auto":
    rt = detect_runtime_environment()
    if not rt.has_gpu:
        rt = prompt_runtime_environment()  # Suggest Docker
elif runtime == "docker":
    rt = RuntimeEnvironment(type=RuntimeType.DOCKER, docker_image=docker_image, ...)
```

**Conflicts expected:** v3 has no `runtime_env.py`. Pure addition. CLI flags are additive.

---

### Phase 4: Optimizer & Benchmark (from MSA → v3)

**What to port:**
- `src/geakagent/optimizer/core.py` → `src/minisweagent/optimizer/core.py`
- `src/geakagent/benchmark.py` → `src/minisweagent/benchmark.py`
- `src/geakagent/kernel_profile.py` → `src/minisweagent/kernel_profile.py`

**Integration with strategy workflow:**

The optimizer can be invoked as a strategy step. In the `mini_kernel_strategy_list.yaml` workflow, after profiling identifies a bottleneck, the agent could call:

```
strategy_manager add "OpenEvolve optimization" --priority high
```

Then the agent (or a wrapper) calls:
```python
from minisweagent.optimizer.core import optimize_kernel
result = optimize_kernel(kernel_code, bottleneck="memory", budget="medium")
```

This is a longer-term integration — for Phase 4, just copy the files and ensure imports work.

**Conflicts expected:** None. These modules don't exist in v3.

---

### Phase 5: Reference & Strategies Database (from MSA → v3)

**What to port:**
- `reference/optimization_strategies.py` (50+ strategies) → `src/minisweagent/reference/optimization_strategies.py`
- `reference/state.py` → `src/minisweagent/reference/state.py` (or integrate into strategy_manager)

**Integration with strategy_manager:**

The strategies database could populate initial strategy lists:
```python
from minisweagent.reference.optimization_strategies import get_strategies_for_bottleneck

strategies = get_strategies_for_bottleneck(BottleneckType.MEMORY_BANDWIDTH, KernelLanguage.TRITON)
for s in strategies:
    strategy_manager.add(s.name, description=s.description, priority="normal")
```

The state machine could be used to add guardrails to the agent loop (e.g., ensure profiling happens before optimization).

**Conflicts expected:** None. `reference/` already exists in MSA as top-level; in v3, nest it under the package.

---

### Phase 6: Test Suite & CI (from MSA → v3)

**What to port:**
- `test_suite/` directory (config.json, run_suite.py)
- `examples/add_kernel/`

**Where it goes in v3:**
- `test_suite/` at top level (same location)
- `examples/` at top level

**Conflicts expected:** None. v3 has `tests/` (unit tests) but no `test_suite/` (integration/regression).

---

## 3. pyproject.toml merge

MSA's `pyproject.toml` defines:
- Package: `geakagent`
- Scripts: `geak`, `geak-agent`, `kernel-profile`
- Dependencies: litellm, anthropic, mcp, rich, typer, etc.

v3's `pyproject.toml` defines:
- Package: `minisweagent`
- Scripts: `mini`, `mini-swe-agent`, `mini-extra`
- Dependencies: similar + pyyaml, requests, jinja2, google-genai, openai

**Merged pyproject.toml should:**
1. Keep package as `minisweagent` (internal), add `geakagent` as alias or thin wrapper.
2. Merge scripts:
   - `geak` → `minisweagent.run.mini:app` (main CLI)
   - `geak-agent` → `geak_agent.cli:main` (discovery-first CLI)
   - `kernel-profile` → `minisweagent.kernel_profile:main`
   - `mini` → `minisweagent.run.mini:app` (keep for backward compat)
3. Union of all dependencies.
4. Add optional `[mcp]` extra for MCP server dependencies.

---

## 4. File-by-file action plan

| File/Directory | Source | Action | Target Location |
|----------------|--------|--------|-----------------|
| `geak_agent/mcp_tools/discovery.py` | MSA | Copy | `src/minisweagent/tools/discovery.py` |
| `geak_agent/mcp_tools/metrix.py` | MSA | Copy | `src/minisweagent/mcp_tools/metrix.py` |
| `geak_agent/cli.py` | MSA | Copy | `src/minisweagent/run/geak_cli.py` |
| `geak_agent/examples/` | MSA | Copy | `examples/` |
| `mcp_tools/` (all 6 servers + client) | MSA | Copy | `mcp_tools/` (top-level) |
| `src/geakagent/runtime_env.py` | MSA | Copy | `src/minisweagent/runtime_env.py` |
| `src/geakagent/optimizer/` | MSA | Copy | `src/minisweagent/optimizer/` |
| `src/geakagent/benchmark.py` | MSA | Copy | `src/minisweagent/benchmark.py` |
| `src/geakagent/kernel_profile.py` | MSA | Copy | `src/minisweagent/kernel_profile.py` |
| `reference/` | MSA | Copy | `src/minisweagent/reference/` |
| `test_suite/` | MSA | Copy | `test_suite/` |
| `Dockerfile` | MSA | Copy | `Dockerfile` |
| `entrypoint.sh` | MSA | Copy | `entrypoint.sh` |
| `scripts/` | MSA | Copy | `scripts/` |
| `RUNTIME_ENV.md` | MSA | Copy | `docs/RUNTIME_ENV.md` |
| `RUNTIME_QUICKSTART.md` | MSA | Copy | `docs/RUNTIME_QUICKSTART.md` |
| `docs/DISCOVERY_PIPELINE.md` | MSA | Copy | `docs/DISCOVERY_PIPELINE.md` |
| `docs/METRIX_TOOL.md` | MSA | Copy | `docs/METRIX_TOOL.md` |
| `src/minisweagent/` (everything) | v3 | Keep | As-is |
| `tests/` | v3 | Keep | As-is |
| `docs/` (MkDocs site) | v3 | Keep | As-is |

---

## 5. Potential conflicts & resolutions

### 5.1 Package naming: `geakagent` vs `minisweagent`
- **Resolution:** Keep `minisweagent` as the internal package. Add a thin `geakagent` wrapper that re-exports for backward compatibility:
  ```python
  # src/geakagent/__init__.py
  from minisweagent import *
  ```

### 5.2 Agent base class differences
- MSA's `DefaultAgent` is in `src/geakagent/agents/default.py`.
- v3's `DefaultAgent` is in `src/minisweagent/agents/default.py`.
- v3's version has more features (ToolRuntime, strategy_manager integration, test_perf context).
- **Resolution:** Keep v3's DefaultAgent. Discard MSA's (it's a simpler version). Port `PatchAgent` from MSA as a new file alongside v3's agents.

### 5.3 `mini.py` entry point
- Both have `src/*/run/mini.py`.
- MSA's has runtime detection; v3's has parallel execution, strategy selection, UnitTestAgent integration.
- **Resolution:** Keep v3's `mini.py` as the base. Add MSA's runtime detection and Docker flags to it.

### 5.4 Profiling tools overlap
- MSA has `MetrixTool` (AMD Metrix API) + `kernel-profiler` MCP (rocprof in Docker).
- v3 has `profiling_tools.py` (rocprof-compute locally).
- **Resolution:** Keep all three. They serve different deployment scenarios:
  - `profiling_tools.py` — local rocprof (no Docker needed)
  - `MetrixTool` — AMD Metrix API (richer metrics, multi-GPU)
  - `kernel-profiler` MCP — rocprof in Docker (sandboxed)

### 5.5 Model files
- Identical structure in both branches (`amd_llm.py`, `anthropic_model.py`, etc.)
- v3's `amd_llm.py` has tool format conversion and retry logic.
- **Resolution:** Keep v3's model files. Diff for any MSA-only fixes and cherry-pick if needed.

### 5.6 Environment files
- Nearly identical in both branches.
- **Resolution:** Keep v3's. Check for MSA-only changes in `docker.py`.

---

## 6. Git merge mechanics

### Option A: Git merge (if branches share common ancestor)

```bash
git checkout geak_v3_features
git merge origin/msa --no-commit

# Resolve conflicts (mostly pyproject.toml, README.md)
# Keep v3's versions of shared files (agents, models, environments)
# Accept MSA's new files (mcp_tools/, discovery.py, runtime_env.py, etc.)

git add .
git commit -m "Merge msa branch: add discovery pipeline, MCP servers, runtime detection, Docker support"
```

### Option B: Cherry-pick approach (cleaner, recommended)

If the branches have diverged significantly:

```bash
git checkout geak_v3_features
git checkout origin/msa -- mcp_tools/
git checkout origin/msa -- geak_agent/mcp_tools/discovery.py
git checkout origin/msa -- geak_agent/mcp_tools/metrix.py
git checkout origin/msa -- geak_agent/cli.py
git checkout origin/msa -- Dockerfile
git checkout origin/msa -- entrypoint.sh
git checkout origin/msa -- scripts/
git checkout origin/msa -- RUNTIME_ENV.md
git checkout origin/msa -- RUNTIME_QUICKSTART.md
git checkout origin/msa -- docs/DISCOVERY_PIPELINE.md
git checkout origin/msa -- docs/METRIX_TOOL.md
git checkout origin/msa -- docs/GETTING_STARTED.md
git checkout origin/msa -- test_suite/
git checkout origin/msa -- examples/
git checkout origin/msa -- reference/

# Then manually copy and adapt:
# - src/geakagent/runtime_env.py → src/minisweagent/runtime_env.py
# - src/geakagent/optimizer/ → src/minisweagent/optimizer/
# - src/geakagent/benchmark.py → src/minisweagent/benchmark.py
# - src/geakagent/kernel_profile.py → src/minisweagent/kernel_profile.py

# Update imports in copied files to use minisweagent instead of geakagent
# Merge pyproject.toml manually
# Update README.md to document new features
```

### Option C: New integration branch

```bash
git checkout -b unified geak_v3_features
# Then apply Phase 1–6 as individual commits
```

---

## 7. Validation checklist

After merge, verify:

- [ ] `mini --help` shows all new flags (--runtime, --docker-image, --workspace)
- [ ] `mini --create-test` runs discovery first, feeds results into UnitTestAgent
- [ ] UnitTestAgent correctly uses discovery context when provided
- [ ] UnitTestAgent still works without discovery context (backward compat)
- [ ] `mini --num-parallel 4` still works (parallel execution)
- [ ] `mini --enable-strategies` still works (strategy management)
- [ ] All 6 MCP servers start and respond to tool calls
- [ ] `pytest tests/` passes (v3's existing tests)
- [ ] Discovery pipeline works on a sample kernel directory
- [ ] Dockerfile builds successfully
- [ ] `kernel-profile` CLI works
- [ ] `geak-agent` CLI works (discovery → context)
- [ ] Profiling tools all work (built-in, MetrixTool, kernel-profiler MCP)
- [ ] `test_suite/run_suite.py` runs on at least one AITER kernel

---

## 8. Timeline estimate

| Phase | Work | Estimate |
|-------|------|----------|
| Phase 1: Discovery pipeline | Copy, integrate, test | 1–2 days |
| Phase 2: MCP servers | Copy, install, verify | 1 day |
| Phase 3: Runtime & Docker | Copy, add CLI flags, test | 1 day |
| Phase 4: Optimizer & benchmark | Copy, fix imports | 0.5 days |
| Phase 5: Reference & strategies | Copy, integrate with strategy_manager | 0.5 days |
| Phase 6: Test suite | Copy, verify | 0.5 days |
| **Integration testing** | End-to-end on real kernels | 1–2 days |
| **Total** | | **5–7 days** |
