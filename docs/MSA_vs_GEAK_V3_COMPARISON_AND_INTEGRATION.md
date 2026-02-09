# MSA vs GEAK v3_features: Test Discovery & Test Generation — Comparison & Integration

## Quick Summary

| | **msa** branch | **geak_v3_features** branch |
|--|----------------|----------------------------|
| **Test Discovery** | Content-based regex/keyword scoring pipeline (`discovery.py`, ~1100 lines) + standalone MCP server | LLM-agent based: UnitTestAgent runs in repo, explores files, picks best test |
| **Test Generation** | Stubbed: `[c] Create tests (I'll help)` prompt exists but is **not implemented** | Fully implemented: UnitTestAgent's system prompt says "Part 2: Create new test/benchmark script if none found" — the LLM writes the test |
| **Benchmark Discovery** | Same content-based pipeline (parallel to test discovery, separate keywords) | Same UnitTestAgent handles both (correctness + benchmark in one command) |

---

## 1. Test Discovery — detailed comparison

### MSA: Deterministic content-based pipeline

**Files:** `geak_agent/mcp_tools/discovery.py` (DiscoveryPipeline class) + `mcp_tools/automated-test-discovery/src/automated_test_discovery/server.py` (MCP wrapper)

**How it works:**
1. **Workspace expansion:** Given a kernel file, walks up to find project root (`.git`, `pyproject.toml`).
2. **Auto-pattern learning:** Samples up to 30 `*test*.py` files and learns custom decorators/assertion functions (e.g. `@perftest()`, `checkAllclose`).
3. **Content scoring:** For every `.py`/`.cpp`/`.cu`/`.hip` file in the workspace:
   - Scores against ~17 Python test keywords (e.g. `import pytest` +0.3, `def test_` +0.4, `.allclose()` +0.3) and ~16 C++ keywords (GTest, Catch2).
   - If `confidence >= 0.3`, it's a candidate.
4. **Kernel-name matching:** Exact name match gives +1.0 confidence; partial (2+ parts match) gives +0.3 per part.
5. **LLM fallback:** For confidence 0.3–0.6, optionally calls LLM (Anthropic) to classify the file and suggest a run command.
6. **Command inference:** Based on content detects `pytest`, `unittest`, `script`, `gtest`, `catch2`, `hipcc`/`nvcc` compile, or Makefile.
7. **User confirmation:** Interactive menu: [y]es / [e]dit / [s]earch more / [c]reate tests.

**Output:** Ranked list of `TestInfo(file_path, test_type, command, confidence)` objects; also benchmark list.

**Strengths:**
- Fast, deterministic, no LLM cost for the common case.
- 90% match rate on 30 AITER kernels.
- Works on C++ too.
- Standalone MCP server for external tool use.

**Weaknesses:**
- Only finds **existing** tests; no creation.
- Can miss tests with unusual patterns (depends on regex coverage).
- The "[c] Create tests" path is a **TODO stub** — it prints a prompt but doesn't actually generate anything.

---

### GEAK v3: LLM-agent-driven discovery

**Files:** `src/minisweagent/agents/unit_test_agent.py` + `src/minisweagent/config/mini_unit_test_agent.yaml`

**How it works:**
1. `run_unit_test_agent(model, repo, kernel_name)` launches a mini SWE agent (DefaultAgent subclass) with a structured system prompt.
2. The agent's **mandatory Part 1 (Discovery):**
   - Explores repo: reads `README.md`, enumerates `benchmark/`, `test/` directories, reads files.
   - Prefers and reuses existing tests/benchmarks.
   - Avoids inventing inputs if existing tests cover them.
3. **Output:** Exactly one shell command: `TEST_COMMAND: <correctness_cmd> && <benchmark_cmd>`.

**Strengths:**
- Can understand project-specific conventions that regex can't catch (e.g. unusual build systems, README-documented commands).
- Naturally handles edge cases because it's an LLM reading the code.

**Weaknesses:**
- Costs LLM tokens every run (model call, usually `claude-opus-4.5`).
- Non-deterministic: different runs may produce different commands.
- Returns only **one** command (no ranked list of candidates).
- No confidence scoring; you either trust the LLM or you don't.

---

## 2. Test Generation — detailed comparison

### MSA: Not implemented (stub only)

In `_prompt_user_confirmation()`:
```python
print("    [c] Create tests (I'll help)")
```
This prints the option but **no code path actually generates tests**. The DISCOVERY_PIPELINE.md doc lists this as a "Future Improvement: Interactive test creation when none found."

The `test_suite/run_suite.py` calls the pipeline and checks for `test_generation_success` in output, but this is about the end-to-end agent flow, not a dedicated test-generation module.

### GEAK v3: Fully implemented via UnitTestAgent

The agent's system prompt has an explicit **Part 2: Creating a new test/benchmark script**:

> (Only if no suitable existing test/benchmark can be found.)
> - Create a minimal correctness test that exercises real kernel code and validates against a trusted reference (torch reference, or non-fused path when applicable) using random inputs across multiple dtypes and shapes.
> - Create a benchmark: For (A) benchmark the kernel with at least 20 iterations per input; print a stable timing metric. For (B) write correctness and benchmark test for original unfused kernels and optimized fused kernel.
> - Cover all supported dtypes, representative sizes, and edge cases.

The agent can run bash commands (including `cat <<'EOF' > newfile.py ...`) to create test files, fix them if they error, and then output the final `TEST_COMMAND`.

**Additional test execution:** The `TestPerfTool` (`src/minisweagent/tools/test_perf.py`) handles running the test command, saving patches, and checking pass/fail — but this is test *execution*, not generation.

---

## 3. Side-by-side

| Feature | MSA | v3 |
|---------|-----|-----|
| **Finds existing tests** | Yes (regex + auto-learning + LLM fallback) | Yes (LLM explores repo) |
| **Ranked candidates** | Yes (confidence 0–1, top 10) | No (one command) |
| **C++ support** | Yes (GTest, Catch2, HIP/CUDA) | Possible (LLM can read C++), but not specialized |
| **Custom pattern learning** | Yes (auto-detects decorators) | Implicit (LLM reads README/code) |
| **MCP server** | Yes (standalone `automated-test-discovery`) | No |
| **Creates tests from scratch** | No (stub only) | Yes (LLM writes correctness + benchmark script) |
| **Handles fused vs unfused** | No | Yes (system prompt has explicit Type A vs Type B logic) |
| **Cost** | Free unless LLM fallback used | LLM call every time |
| **Determinism** | Deterministic (same input → same output) | Non-deterministic |

---

## 4. Integration options

### Option A: MSA discovery first, v3 agent as fallback for generation (recommended)

**Flow:**
1. Run MSA's `DiscoveryPipeline` on the kernel path.
2. **If tests found** (confidence > 0.5): use MSA's top test command; skip UnitTestAgent.
3. **If no tests found** (or all low-confidence): call v3's `run_unit_test_agent()` which will create tests from scratch.
4. Pass the resulting command to the main optimization agent.

**Why this is best:**
- MSA handles the 90% case fast and free (no LLM).
- v3 handles the 10% case where no tests exist — it actually generates them.
- You get ranked candidates when tests exist, and LLM-generated tests when they don't.

**Implementation sketch:**
```python
from geak_agent.mcp_tools.discovery import discover

result = discover(kernel_path=kernel, interactive=False)

if result.tests and result.tests[0].confidence > 0.5:
    test_cmd = result.tests[0].command
    bench_cmd = result.benchmarks[0].command if result.benchmarks else None
else:
    # Fall back to UnitTestAgent (can create tests)
    test_cmd = run_unit_test_agent(model=model, repo=workspace, kernel_name=name)
    bench_cmd = None  # included in test_cmd
```

### Option B: MSA discovery + add test generation to MSA

**Flow:**
- Keep using MSA's discovery for finding.
- Port the "Part 2" test-generation logic from v3's system prompt into MSA as a new method, e.g. `_generate_tests()`.
- When discovery finds nothing, call LLM to generate a correctness + benchmark script (reuse v3's prompt template).

**Why:** Keeps everything in one pipeline; no need for two systems.

### Option C: v3 UnitTestAgent uses MSA's results as context

**Flow:**
- Run MSA discovery first.
- Feed the results into UnitTestAgent's task prompt: "Here are discovered tests: [list with confidence]. If any are suitable, output that. Otherwise create new ones."
- This lets the LLM validate MSA's findings and fill gaps.

**Why:** The LLM can double-check MSA's regex-based findings and create tests when needed.

---

## 5. File reference

| Purpose | MSA branch | GEAK v3_features branch |
|---------|------------|--------------------------|
| Discovery pipeline (full) | `geak_agent/mcp_tools/discovery.py` | — |
| Discovery MCP server | `mcp_tools/automated-test-discovery/.../server.py` | — |
| Test-finding agent | — | `src/minisweagent/agents/unit_test_agent.py` |
| Test-finding config/prompt | — | `src/minisweagent/config/mini_unit_test_agent.yaml` |
| Test generation | Stub only (`[c] Create tests`) | UnitTestAgent Part 2 (in system prompt) |
| Test execution | `test_suite/run_suite.py` | `src/minisweagent/tools/test_perf.py` |
| CLI entry | `geak_agent/cli.py` | `src/minisweagent/run/mini.py` (`--create-test`) |
