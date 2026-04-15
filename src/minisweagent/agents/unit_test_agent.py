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
        "This is a Composable Kernel (CK) kernel (C++ compiled with hipcc + CK includes).\n"
        "\n"
        "MANDATORY ARCHITECTURE (no exceptions):\n"
        "You MUST follow the CK test harness recipes and create the following files:\n"
        "  1. baseline.cpp  — C ABI wrapper compiled to libbaseline.so. Frozen ground truth. NEVER modified after creation.\n"
        "  2. optimized.cpp — C ABI wrapper compiled to liboptimized.so. The optimizer edits tuning parameters here.\n"
        "  3. Makefile      — builds both .so files with `hipcc -shared -fPIC -O3`.\n"
        "  4. compile.py    — auto-detects GPU arch, calls make.\n"
        "  5. test_harness.py — loads BOTH .so via ctypes, runs correctness/benchmark.\n"
        "\n"
        "RECIPE SELECTION — pick the recipe closest to the target kernel's DeviceOp:\n"
        "\n"
        "  1. softmax-recipe/ — for kernels using VECTOR API (std::vector<index_t> args).\n"
        "     DeviceOps: DeviceSoftmax, DeviceNormalization, DeviceReduce, DevicePool.\n"
        "     Pattern: MakeArgument takes std::vector lengths/strides.\n"
        "\n"
        "  2. gemm-recipe/ — for kernels using SCALAR API (flat M, N, K, Stride args).\n"
        "     DeviceOps: DeviceGemm*, DeviceBatchedGemm*, DeviceGemmMultipleD*.\n"
        "     Pattern: MakeArgument takes individual scalars: M, N, K, StrideA, StrideB, etc.\n"
        "     D tensors use std::array<const void*, NumD> and std::array<index_t, NumD>.\n"
        "\n"
        "  3. conv-recipe/ — for kernels using ARRAY API (std::array<index_t, NDimSpatial+3> args).\n"
        "     DeviceOps: DeviceGroupedConv*, DeviceGroupedConvBwd*.\n"
        "     Pattern: MakeArgument takes std::array lengths/strides for input/weight/output.\n"
        "     CRITICAL: Compute descriptor arrays inline in canonical CK order\n"
        "     (G,N,C,H,W for input; G,K,C,Y,X for weight; G,N,K,Ho,Wo for output).\n"
        "     Strides must reflect the physical layout. See the recipe for the exact pattern.\n"
        "     Do NOT use ConvParam or make_*_packed helpers (they require linking CK library source).\n"
        "\n"
        "All recipes are found in GEAK/knowledge-base/ck-knowledge-base/. Read the selected recipe's README.md for more details.\n"
        "Read the baseline.cpp and test_harness.py of the selected recipe. Adapt the C ABI signature, ctypes bindings, and shape lists for the target kernel.\n"
        "Copy compile.py and Makefile verbatim (only CK_INCLUDE may need adjustment).\n"
        "\n"
        "- TEST_COMMAND MUST include a compile step before running the test harness.\n"
        "- TEST_COMMAND for CK example: `python3 <ABSOLUTE_PATH_TO_WORKDIR>/compile.py && python3 <ABSOLUTE_PATH_TO_HARNESS_DIR>/test_harness.py --correctness && python3 <ABSOLUTE_PATH_TO_HARNESS_DIR>/test_harness.py --benchmark`\n"
        "\n"
        "- DO NOT compile standalone executables. DO NOT use subprocess to invoke kernels.\n"
        "- The test_harness.py MUST load libbaseline.so and liboptimized.so via ctypes.\n"
        "-baseline.cpp and optimized.cpp must expose an `extern \"C\" float run_kernel(...)` function.\n"
        "\n"
        "COMPILATION ERRORS: If hipcc fails, read the error carefully.\n"
        "- Template deduction errors usually mean wrong type aliases or wrong MakeArgument args.\n"
        "- Undefined symbols for ConvParam/HostTensor mean you used CK library utilities — inline instead.\n"
        "- NEVER abandon the .so/ctypes architecture. Fix the C++ wrapper until it compiles.\n"
        "\n"
        "- The harness must work standalone in GEAK_WORK_DIR (a flat copy of the kernel folder, not the full repo).\n"
        "- Test in a temp directory that imitates this layout."
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
    preferred_harness_path: Path | None = None,
    kernel_path: Path | None = None,
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
        f"IMPORTANT: Read src/minisweagent/run/preprocess/INSTRUCTIONS.md in the repository for test harness requirements\n"
        f"and COMMANDMENT format rules before creating the harness."
    )
    if preferred_harness_path is not None:
        task += (
            f"\n\nWrite the harness to this exact path: {preferred_harness_path}"
            "\nDo NOT place the final harness next to the kernel source file."
            "\nReturn TEST_COMMAND using this exact harness path."
        )
    if kernel_path is not None:
        kernel_dir = kernel_path.resolve().parent
        try:
            rel_kernel_dir = kernel_dir.relative_to(repo.resolve())
        except ValueError:
            rel_kernel_dir = None
        task += (
            f"\n\nTarget kernel file: {kernel_path.resolve()}"
            "\nThe harness must remain runnable even when it is outside the kernel directory."
            "\nDo NOT rely on __file__ being in the same directory as kernel.py."
            "\nResolve imports by preferring GEAK_WORK_DIR when available, then "
            "GEAK_REPO_ROOT plus the kernel's repo-relative directory, then the original kernel directory."
        )
        if rel_kernel_dir is not None:
            task += f"\nKernel repo-relative directory: {rel_kernel_dir.as_posix()}"
    if discovery_context:
        task += f"\n\n{discovery_context}"

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(f"UnitTestAgent did not finish successfully: {exit_status}\n{result}")

    return _extract_test_command(result)
