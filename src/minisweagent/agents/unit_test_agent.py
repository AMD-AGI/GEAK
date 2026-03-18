# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

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
        "- A build step is REQUIRED. Needs CK headers and hipcc.\n"
        "- Template parameters (tile sizes, vector widths) are compile-time; test multiple configs.\n"
        "- Use host-side validation against a reference GEMM/convolution.\n"
        "- Use `hipEventElapsedTime` for benchmarking.\n"
        "- NEVER use `sys.path.insert(0, '/absolute/path/...')`. "
        "Rely on PYTHONPATH set by the COMMANDMENT SETUP section."
    ),
    "flydsl": (
        "This is a FlyDSL kernel (Python DSL with @flyc.kernel / @flyc.jit, JIT-compiled via MLIR/ROCm).\n"
        "- No separate kernel build step; the repo must be built once (scripts/build.sh) so that "
        "build-fly/python_packages and MLIR libs exist. The harness runs in the repo root with "
        "PYTHONPATH and LD_LIBRARY_PATH set.\n"
        "- Import the kernel via its Python package path (e.g. `from kernels.softmax_kernel import build_softmax_module`). "
        "Do NOT use importlib.util; use normal imports with PYTHONPATH set in COMMANDMENT SETUP.\n"
        "- Use `torch.testing.assert_close` for correctness against a PyTorch reference (e.g. torch.softmax for softmax). "
        "Reuse tolerances and shapes from existing tests in tests/kernels/ (e.g. test_softmax.py, test_preshuffle_gemm.py).\n"
        "- Use fixed random seed (`torch.manual_seed(42)`) and fixed tensor sizes. "
        "For benchmarking use torch CUDA events or the project's run_perftest/bench_gpu_us_torch from tests/test_common.py and tests/kernels/benchmark_common.py.\n"
        "- The harness MUST support exactly these CLI modes (GEAK contract): --correctness, --profile, --benchmark, --full-benchmark. "
        "The last line of --benchmark and --full-benchmark output MUST be: GEAK_RESULT_LATENCY_MS=<number> (median or geomean latency in ms).\n"
        "- COMMANDMENT SETUP MUST set both PYTHONPATH and LD_LIBRARY_PATH. "
        "Typical values: PYTHONPATH=<REPO_ROOT>/build-fly/python_packages:<REPO_ROOT>; "
        "LD_LIBRARY_PATH=<REPO_ROOT>/build-fly/python_packages/flydsl/_mlir/_mlir_libs. "
        "NEVER use sys.path.insert(0, ...) inside the harness; rely on PYTHONPATH set in SETUP."
    ),
    "asm": (
        "This is a precompiled HSACO assembly kernel.\n"
        "- The assembly binary CANNOT be modified or recompiled.\n"
        "- Test ONLY via the Python wrapper that loads and launches it.\n"
        "- Use `torch.testing.assert_close` for correctness against a torch reference.\n"
        "- Benchmark the wrapper launch, not the assembly directly."
    ),
    "pytorch": (
        "This is a **PyTorch-to-FlyDSL translation** task. The input is pure PyTorch code "
        "(nn.Module) that will be translated into a FlyDSL kernel by the optimizer.\n"
        "- The PyTorch code is the REFERENCE implementation — it is correct by definition.\n"
        "- A FlyDSL kernel will be generated LATER by the optimizer; it does NOT exist yet.\n"
        "- The harness MUST accept `--flydsl-kernel <path>` as an optional CLI argument "
        "to locate the FlyDSL candidate at runtime.\n"
        "- When `--flydsl-kernel` is NOT provided or the file does not exist:\n"
        "  - `--correctness`: validate that the PyTorch reference runs correctly, exit 0.\n"
        "  - `--benchmark` / `--full-benchmark`: benchmark PyTorch only, print GEAK_RESULT_LATENCY_MS.\n"
        "  - `--profile`: profile PyTorch only.\n"
        "- When `--flydsl-kernel` IS provided and the file exists:\n"
        "  - `--correctness`: run BOTH PyTorch and FlyDSL with identical inputs, compare outputs "
        "with `torch.testing.assert_close`. Exit non-zero on mismatch.\n"
        "  - `--benchmark` / `--full-benchmark`: benchmark BOTH, print PyTorch and FlyDSL latencies, "
        "print speedup ratio, and GEAK_RESULT_LATENCY_MS=<FlyDSL latency>.\n"
        "  - `--profile`: profile FlyDSL kernel only (for hardware analysis).\n"
        "- Use `importlib.util` to dynamically load the FlyDSL kernel (path varies per evaluation).\n"
        "- The FlyDSL module should expose a `forward(*args)` function or a `Model` class "
        "matching the PyTorch interface.\n"
        "- Use fixed random seed (`torch.manual_seed(42)`) and fixed tensor sizes.\n"
        "- Generate tensors on CPU, then move to GPU.\n"
        "- No build step needed (both PyTorch and FlyDSL are Python/JIT)."
    ),
    "unknown": (
        "Kernel type could not be determined automatically.\n"
        "- Inspect the source file to determine if it is Triton, HIP, CUDA, CK, or FlyDSL.\n"
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
        f"IMPORTANT: Read INSTRUCTIONS.md in the repository for test harness requirements\n"
        f"and COMMANDMENT format rules before creating the harness."
    )
    if discovery_context:
        task += f"\n\n{discovery_context}"

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(f"UnitTestAgent did not finish successfully: {exit_status}\n{result}")

    return _extract_test_command(result)


# ---------------------------------------------------------------------------
# PyTorch-to-FlyDSL translation support
# ---------------------------------------------------------------------------

_PYTORCH_INDICATORS = (
    "import torch",
    "torch.nn",
    "nn.Module",
)

_GPU_KERNEL_FRAMEWORKS = (
    "@triton.jit",
    "import triton",
    "@flyc.kernel",
    "@flyc.jit",
    "import flyc",
    "#include <hip/",
    "hipcc",
    "__global__",
    "@cuda.jit",
)


def detect_pytorch_translation_task(kernel_path: Path) -> bool:
    """Return True if *kernel_path* is pure PyTorch code suitable for FlyDSL translation.

    A file qualifies when it imports torch **and** defines an ``nn.Module``
    subclass but does NOT use any GPU kernel framework (Triton, FlyDSL, HIP,
    CUDA).  This heuristic covers the standard ``Model(nn.Module)`` pattern
    used by kernel evaluation benchmarks (e.g. ``hingeloss.py``).
    """
    if not kernel_path.exists() or kernel_path.suffix != ".py":
        return False

    try:
        content = kernel_path.read_text(errors="replace")
    except OSError:
        return False

    has_pytorch = all(indicator in content for indicator in _PYTORCH_INDICATORS)
    has_gpu_framework = any(fw in content for fw in _GPU_KERNEL_FRAMEWORKS)

    return has_pytorch and not has_gpu_framework


def format_pytorch_translation_context(
    kernel_path: Path,
    kernel_name: str,
) -> str:
    """Build extra context describing the PyTorch→FlyDSL translation scenario.

    This is appended to the discovery context so the agent understands
    that the PyTorch code is the reference and the FlyDSL kernel will
    appear later.
    """
    lines = [
        "## PyTorch-to-FlyDSL Translation Task",
        "",
        f"The input kernel `{kernel_path.name}` is **pure PyTorch** code.",
        "The optimizer will translate this into a FlyDSL kernel in subsequent steps.",
        "",
        "**Your harness must support two execution modes:**",
        "",
        "1. **Baseline mode** (no `--flydsl-kernel` argument):",
        "   - Runs the PyTorch reference only.",
        "   - `--correctness` validates that PyTorch produces reasonable outputs, exits 0.",
        "   - `--benchmark` reports PyTorch latency as GEAK_RESULT_LATENCY_MS.",
        "",
        "2. **Comparison mode** (`--flydsl-kernel <path>` argument provided):",
        "   - Dynamically loads the FlyDSL kernel from the given path.",
        "   - `--correctness` runs BOTH implementations with identical inputs and compares "
        "outputs using `torch.testing.assert_close`. Exits non-zero on mismatch.",
        "   - `--benchmark` times BOTH implementations, prints per-implementation latencies, "
        "speedup ratio, and GEAK_RESULT_LATENCY_MS=<FlyDSL latency in ms>.",
        "",
        "**FlyDSL kernel loading convention:**",
        "- Use `importlib.util.spec_from_file_location` to load the candidate module "
        "from the path given by `--flydsl-kernel`.",
        "- The FlyDSL module must expose either:",
        "  - A `Model` class with the same `forward()` signature as the PyTorch reference, OR",
        "  - A `forward(*args)` function with the same signature.",
        "- The harness should try `Model` first, then fall back to `forward()`.",
        "",
        f"**PyTorch reference interface** (from `{kernel_path.name}`):",
    ]

    try:
        content = kernel_path.read_text(errors="replace")
        # Extract get_inputs, get_init_inputs, batch_size, input_shape
        for line in content.splitlines():
            stripped = line.strip()
            if any(
                stripped.startswith(prefix)
                for prefix in (
                    "class Model",
                    "def forward",
                    "def get_inputs",
                    "def get_init_inputs",
                    "batch_size",
                    "input_shape",
                    "dim ",
                )
            ):
                lines.append(f"  `{stripped}`")
    except OSError:
        lines.append("  (could not read file)")

    lines.append("")
    return "\n".join(lines)


def run_pytorch_translation_agent(
    *,
    model: Model,
    repo: Path,
    kernel_name: str,
    kernel_path: Path,
    log_dir: Path | None = None,
    discovery_context: str = "",
) -> str:
    """Run UnitTestAgent configured for PyTorch→FlyDSL translation harness creation.

    Uses the ``mini_unit_test_agent_pytorch_translation`` config which has a
    system prompt tailored for creating a two-implementation comparison harness
    where the PyTorch code is the reference and the FlyDSL kernel will be
    generated later.

    Parameters
    ----------
    kernel_path:
        Absolute path to the PyTorch reference file.  Passed to the agent
        so it knows where to import the reference from.
    """
    agent_config, _ = load_agent_config("mini_unit_test_agent_pytorch_translation")

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)
    agent = UnitTestAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "unit_test_agent.log"

    translation_context = format_pytorch_translation_context(kernel_path, kernel_name)

    task = (
        f"Create a PyTorch-to-FlyDSL translation test harness for kernel: {kernel_name}\n"
        f"Repository: {repo}\n"
        f"PyTorch reference file: {kernel_path}\n\n"
        f"The FlyDSL kernel does NOT exist yet. It will be generated by the optimizer.\n"
        f"The harness must accept --flydsl-kernel <path> to load the FlyDSL candidate at runtime.\n\n"
        f"IMPORTANT: Read INSTRUCTIONS.md in the repository for general test harness requirements\n"
        f"and COMMANDMENT format rules before creating the harness.\n"
    )
    if discovery_context:
        task += f"\n{discovery_context}"
    task += f"\n{translation_context}"

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(
            f"UnitTestAgent (pytorch_translation) did not finish successfully: {exit_status}\n{result}"
        )

    return _extract_test_command(result)
