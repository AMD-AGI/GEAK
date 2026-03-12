# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Unit test subagent.

This agent searches for (or creates) a fixed test harness for a kernel and
returns a TEST_COMMAND string plus COMMANDMENT-ready commands.  Discovery
results are formatted into an enriched context that includes kernel analysis,
language-specific guidance, and extracted test patterns so the agent can make
informed decisions without re-scanning the repo.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import load_agent_config
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig


@dataclass
class UnitTestAgentConfig(AgentConfig):
    """Config loaded from mini_unit_test_agent.yaml (or provided via kwargs)."""


class UnitTestAgent(DefaultAgent):
    """Agent that creates a fixed test harness and returns TEST_COMMAND."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=UnitTestAgentConfig, **kwargs)


def _extract_test_command(text: str) -> str:
    match = re.search(r"TEST_COMMAND:\s*(.+)\s*$", text.strip(), re.MULTILINE)
    if not match:
        raise ValueError(f"UnitTestAgent did not return TEST_COMMAND. Output was:\n{text}")
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# Language-specific testing guidance (keyed by kernel_type)
# ---------------------------------------------------------------------------

_LANGUAGE_GUIDANCE: dict[str, str] = {
    "triton": (
        "This is a Triton kernel (JIT-compiled Python). No build step needed.\n"
        "- Import the kernel via its Python package path (do NOT use importlib.util).\n"
        "- Use `torch.testing.assert_close` for correctness validation.\n"
        "- Use `triton.testing.do_bench` or `torch.cuda.Event` for benchmarking.\n"
        "- Set `PYTHONPATH` before the process starts if the package is not installed.\n"
        "- Use fixed random seed (`torch.manual_seed(42)`) and fixed tensor sizes."
    ),
    "hip": (
        "This is a HIP kernel (C++ compiled with hipcc).\n"
        "- A build step is REQUIRED before running tests.\n"
        "- Use the project's build system (CMake/Makefile) or compile with `hipcc` directly.\n"
        "- Use host-side validation (compare GPU output against CPU reference).\n"
        "- Use `hipEventElapsedTime` or `torch.cuda.Event` for benchmarking.\n"
        "- NEVER use `sys.path.insert(0, '/absolute/path/...')`. "
        "Rely on PYTHONPATH set by the COMMANDMENT SETUP section."
    ),
    "cuda": (
        "This is a CUDA kernel (C++ compiled with nvcc).\n"
        "- A build step is REQUIRED before running tests.\n"
        "- Use the project's build system (CMake/Makefile) or compile with `nvcc` directly.\n"
        "- Use host-side validation (compare GPU output against CPU reference).\n"
        "- Use `cudaEventElapsedTime` or `torch.cuda.Event` for benchmarking."
    ),
    "ck": (
        "This is a Composable Kernel (CK) kernel (C++ compiled with hipcc via CMake).\n\n"
        "IMPORTANT: Read the 'CK (Composable Kernel) Test Harness' section in INSTRUCTIONS.md.\n\n"
        "Understanding the kernel (mandatory first steps):\n"
        "  - Read any README.md in the kernel example directory to understand the\n"
        "    mathematical operation, inputs, outputs, and parameter space.\n"
        "  - From the README and source code (.cpp, .inc files), identify all input\n"
        "    tensors (names, data types, valid shape ranges) and output tensors.\n"
        "  - Determine the kernel's CLI interface (what arguments it accepts,\n"
        "    what verify/init/time flags mean).\n"
        "  - You may modify the kernel source (without changing the mathematical\n"
        "    operation) to support test harness integration, e.g., adding\n"
        "    output dumping hooks or deterministic initialization.\n\n"
        "Use the CK harness template provided in the discovery context as your starting point.\n"
        "Fill in the placeholder values:\n"
        "  - ORIGINAL_BINARY: path to the saved original binary (pre-optimization)\n"
        "  - BUILD_DIR: cmake build directory for the optimized binary\n"
        "  - CMAKE_SOURCE_DIR: the CK example directory containing CMakeLists.txt\n"
        "  - CMAKE_TARGET: the cmake target name (from add_example_executable)\n\n"
        "Correctness checking (GPU output comparison):\n"
        "  - The preprocessor auto-patches GEAK_DUMP_OUTPUT (binary float32) and\n"
        "    GEAK_RAND_SEED into the C++ source BEFORE building the original binary,\n"
        "    so both original and optimized binaries support these env vars.\n"
        "  - Run both binaries with verify=0 init=1 time=0 and GEAK_DUMP_OUTPUT set\n"
        "    to a temp file.  Set GEAK_RAND_SEED for deterministic random shapes.\n"
        "  - Read dumps with np.fromfile(path, dtype=np.float32) -- NOT np.loadtxt.\n"
        "    The dumps are raw binary float32, not text.\n"
        "  - Compare with np.allclose (FP16 tolerances: rtol=1e-2, atol=1e-2).\n"
        "  - Fallback: if the original binary is missing or dumps are empty, use\n"
        "    the optimized binary's verify=1 (built-in CPU reference check).\n"
        "  - NOTE: verify=1 runs a CPU reference computation which can be slow\n"
        "    (2-37s per shape depending on tensor size).  Prefer GPU dump comparison.\n\n"
        "Generating valid test inputs:\n"
        "  - Use the README and source code to determine valid input shapes and\n"
        "    parameter combinations.  Do NOT invent arbitrary shapes.\n"
        "  - Generate a small deterministic test set: 2-4 shapes for correctness\n"
        "    (total runtime under 60s), 5 shapes for benchmarks.\n"
        "  - Use deterministic initialization (init_method=1 or GEAK_RAND_SEED=42)\n"
        "    so results are reproducible across runs.\n\n"
        "Benchmark considerations:\n"
        "  - CK binaries internally run cold_niters_=5 warmup + nrepeat_=50 timed\n"
        "    iterations when time_kernel=1 (via StreamConfig).  Use exactly 1 external\n"
        "    call per shape.  Do NOT multiply with external iterations (20 external *\n"
        "    50 internal = 1000 kernel launches per shape = guaranteed timeout).\n"
        "  - The --iterations flag should be accepted but ignored for CK harnesses.\n\n"
        "Shape budget guidelines:\n"
        "  - HARNESS_SHAPES: 6-8 shapes for correctness (keep total < 120s)\n"
        "  - PROFILE_SHAPES: 5 shapes (evenly sampled from ALL_SHAPES)\n"
        "  - ALL_SHAPES: all discovered shapes for --full-benchmark\n\n"
        "GEAK_DUMP_OUTPUT auto-patch notes:\n"
        "  - The preprocessor inserts a dump block after the first FromDevice() call.\n"
        "  - For multi-output or grouped kernels (multiple FromDevice calls in a loop),\n"
        "    you may need to adjust the auto-patched dump code to write all output\n"
        "    tensors into a single file.\n\n"
        "Do NOT:\n"
        "  - Use existing C++ gtest files as harnesses\n"
        "  - Try to import CK binaries via pybind11 or ctypes\n"
        "  - Hardcode GPU device IDs\n"
        "  - Skip the cmake rebuild step (template parameters are compile-time)\n"
        "  - Use sys.path.insert(0, '/absolute/path/...')\n"
        "  - Use np.loadtxt to read GEAK_DUMP_OUTPUT files (use np.fromfile)\n"
    ),
    "asm": (
        "This is a precompiled HSACO assembly kernel.\n"
        "- The assembly binary CANNOT be modified or recompiled.\n"
        "- Test ONLY via the Python wrapper that loads and launches it.\n"
        "- Use `torch.testing.assert_close` for correctness against a torch reference.\n"
        "- Benchmark the wrapper launch, not the assembly directly."
    ),
    "unknown": (
        "Kernel type could not be determined automatically.\n"
        "- Inspect the source file to determine if it is Triton, HIP, CUDA, or CK.\n"
        "- Apply the appropriate testing strategy based on your analysis."
    ),
}


def format_discovery_for_agent(result) -> str:
    """Format a ``DiscoveryResult`` into an enriched context string for the UTA.

    Includes kernel analysis, language-specific testing guidance, discovered
    tests/benchmarks with confidence scores, and extracted test patterns.

    Formats an already-available ``DiscoveryResult`` for agent consumption.
    """
    if result is None:
        return ""

    lines: list[str] = []

    # --- Kernel analysis ---
    if result.kernels:
        k = result.kernels[0]
        lines.append("## Kernel Analysis")
        lines.append(f"- **Name**: {k.kernel_name}")
        lines.append(f"- **Type**: {k.kernel_type}")
        lines.append(f"- **Language**: {k.kernel_language}")
        lines.append(f"- **File**: `{k.file_path}`")
        lines.append(f"- **Functions**: {', '.join(k.function_names) if k.function_names else 'N/A'}")
        if k.inner_kernel_path:
            lines.append(f"- **Inner kernel**: `{k.inner_kernel_path}` ({k.inner_kernel_language or 'unknown'})")
        if k.build_info:
            bi = k.build_info
            if bi.compiler:
                lines.append(f"- **Compiler**: {bi.compiler}")
            if bi.build_system:
                lines.append(f"- **Build system**: {bi.build_system}")
            if bi.pybind_module:
                lines.append(f"- **Pybind module**: {bi.pybind_module}")
        lines.append("")

        # Language-specific testing guidance
        guidance = _LANGUAGE_GUIDANCE.get(k.kernel_type, "")
        if guidance:
            lines.append("## Language-Specific Testing Guidance")
            lines.append(guidance)
            lines.append("")

    # --- Discovered tests ---
    if result.tests:
        lines.append("## Discovered Test Files (ranked by confidence)")
        for i, t in enumerate(result.tests[:5], 1):
            conf_pct = min(int(t.confidence * 100), 100)
            lines.append(f"  {i}. `{t.file_path}` — {t.test_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{t.command}`")
        lines.append("")

    # --- Extracted test patterns (from top-confidence tests) ---
    patterns_found = False
    for t in result.tests[:3]:
        p = getattr(t, "patterns", None)
        if p is None:
            continue
        if not patterns_found:
            lines.append("## Extracted Test Patterns (reuse these in your harness)")
            patterns_found = True
        lines.append(f"From `{t.file_path.name}`:")
        if p.tolerances:
            lines.append(f"  Tolerances: {', '.join(p.tolerances)}")
        if p.input_shapes:
            lines.append(f"  Input shapes: {', '.join(p.input_shapes)}")
        if p.dtypes:
            lines.append(f"  Dtypes: {', '.join(p.dtypes)}")
        if p.reference_impls:
            lines.append(f"  Reference implementations: {', '.join(p.reference_impls)}")
        if p.import_patterns:
            lines.append("  Import patterns:")
            for imp in p.import_patterns[:5]:
                lines.append(f"    `{imp}`")
    if patterns_found:
        lines.append("")

    # --- Discovered benchmarks ---
    if result.benchmarks:
        lines.append("## Discovered Benchmark Files (ranked by confidence)")
        for i, b in enumerate(result.benchmarks[:5], 1):
            conf_pct = min(int(b.confidence * 100), 100)
            lines.append(f"  {i}. `{b.file_path}` — {b.bench_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{b.command}`")
        lines.append("")

    # --- Dependency graph summary ---
    if result.kernels and result.dependency_graphs:
        k = result.kernels[0]
        dep_graph = result.dependency_graphs.get(k.kernel_name)
        if dep_graph:
            lines.append("## Dependency Graph")
            lines.append(dep_graph.summary())
            lines.append("")

    if not result.tests and not result.benchmarks:
        lines.append("No existing tests or benchmarks were found by the automated scan.")
        lines.append("You will need to create them from scratch.")
        lines.append("")

    return "\n".join(lines)



def run_unit_test_agent(
    *,
    model: Model,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None = None,
    discovery_context: str = "",
) -> str:
    """Run UnitTestAgent in ``repo`` and return the extracted test command string.

    If *discovery_context* is provided (e.g. from :func:`format_discovery_for_agent`),
    it is appended to the task prompt so the agent starts with pre-scanned results
    instead of exploring from scratch.
    """
    agent_config, _ = load_agent_config("mini_unit_test_agent")

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)
    agent = UnitTestAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "unit_test_agent.log"

    task = (
        f"Create a fixed test harness for kernel: {kernel_name}\n"
        f"Repository: {repo}\n\n"
        f"IMPORTANT:\n"
        f"1. If a README.md exists in or near the kernel directory, read it first to\n"
        f"   understand what the kernel does, its inputs, outputs, and parameter space.\n"
        f"2. Read INSTRUCTIONS.md in the repository for test harness requirements\n"
        f"   and COMMANDMENT format rules before creating the harness.\n"
        f"3. Identify the kernel's inputs and outputs, generate valid test data,\n"
        f"   and keep the test set small enough to run in under 60 seconds."
    )
    if discovery_context:
        task += f"\n\n{discovery_context}"

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(f"UnitTestAgent did not finish successfully: {exit_status}\n{result}")

    return _extract_test_command(result)
