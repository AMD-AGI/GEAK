"""
Automated Test Discovery MCP Server

Single-tool MCP for discovering tests and benchmarks for GPU kernels.
No configuration files needed - uses content-based detection.
"""

import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    name="automated-test-discovery",
    instructions="""
    Automated Test Discovery for GPU Kernels.
    
    Single tool: discover - finds tests, benchmarks, and kernel info.
    
    Provide a kernel file path OR a repository directory and it returns everything:
    - Kernel name and type (triton/hip/cuda)
    - Related test files with confidence scores and run commands
    - Related benchmark files with confidence scores
    - Project workspace path
    
    When a directory is given, all kernels inside it are discovered first,
    then tests and benchmarks are matched against every discovered kernel.
    
    Uses content-based detection (not directory names) and works on any project.
    """
)


# ============================================================================
# Content Detection Patterns
# ============================================================================

KERNEL_PATTERNS = [
    r"@triton\.jit",
    r"@triton\.autotune",
    r"__global__\s+void",
    r"tl\.load|tl\.store",
]

TEST_KEYWORDS = [
    (r"import pytest", 0.3),
    (r"@pytest\.mark", 0.3),
    (r"def test_\w+\s*\(", 0.4),
    (r"assert\s+", 0.2),
    (r"\.allclose\(", 0.3),
    (r"\.assertEqual\(", 0.2),
    (r"torch\.testing\.assert", 0.3),
    (r"@perftest\(\)", 0.35),
    (r"checkAllclose", 0.35),
    (r"from.*test_common import", 0.25),
    (r"correctness", 0.2),
    (r"verify|verification", 0.15),
    (r"class Test\w+", 0.3),
    (r"unittest", 0.2),
    (r"TEST\s*\(\s*\w+\s*,", 0.5),
    (r"EXPECT_TRUE|EXPECT_EQ", 0.35),
    (r"ASSERT_TRUE|ASSERT_EQ", 0.35),
]

BENCH_KEYWORDS = [
    (r"elapsed_time|elapsed", 0.3),
    (r"latency", 0.25),
    (r"throughput", 0.25),
    (r"TFLOPS|GFLOPS", 0.4),
    (r"us/iter|ms/iter", 0.3),
    (r"warmup|warm_up", 0.25),
    (r"benchmark|bench_", 0.3),
    (r"torch\.cuda\.Event\(enable_timing", 0.4),
    (r"triton\.testing\.do_bench", 0.5),
    (r"speedup", 0.25),
    (r"GB/s|TB/s", 0.3),
    (r"hipEventElapsedTime|cudaEventElapsedTime", 0.4),
]

SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    "build", "dist", ".eggs", "site-packages", ".tox", ".pytest_cache"
}


# ============================================================================
# Helper Functions
# ============================================================================

def _should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS or part.endswith(".egg-info"):
            return True
    return False


def _relevance_score(file_path: Path, kernel_path: Path, kernel_name: str, kernel_parts: list[str]) -> float:
    """Score how relevant a test/bench file is to the kernel.

    Returns a multiplier (0.0 - 3.0+) based on:
    - Name match: kernel name appears in the file name or path
    - Path proximity: file is in the same or nearby directory
    - Path component match: kernel path components appear in file path
    """
    score = 0.0
    fname_lower = file_path.name.lower()
    fpath_lower = str(file_path).lower()
    kname_lower = kernel_name.lower()

    # Exact kernel name in filename (strongest signal)
    if kname_lower in fname_lower:
        score += 3.0

    # Kernel name in path (e.g. triton_tests/rope/test_something.py)
    elif kname_lower in fpath_lower:
        score += 2.0

    # Partial name match (kernel parts in filename)
    elif kernel_parts:
        matches = sum(1 for p in kernel_parts if p in fname_lower)
        if matches > 0:
            score += 0.5 * matches

    # Path proximity: same parent directory tree
    try:
        kernel_parents = set(kernel_path.resolve().parents)
        file_parents = set(file_path.resolve().parents)
        shared = kernel_parents & file_parents
        if shared:
            deepest_shared = max(shared, key=lambda p: len(p.parts))
            depth_from_shared = len(file_path.resolve().parts) - len(deepest_shared.parts)
            if depth_from_shared <= 2:
                score += 1.0
            elif depth_from_shared <= 4:
                score += 0.3
    except Exception:
        pass

    return score


def _is_kernel_file(path: Path) -> bool:
    try:
        content = path.read_text()[:3000]
        for pattern in KERNEL_PATTERNS:
            if re.search(pattern, content):
                return True
    except Exception:
        pass
    return False


def _score_as_test(path: Path) -> float:
    try:
        content = path.read_text()
    except Exception:
        return 0.0
    
    score = 0.0
    for pattern, points in TEST_KEYWORDS:
        if re.search(pattern, content, re.IGNORECASE):
            score += points
    
    if "test" in path.name.lower():
        score += 0.1
    
    return score


def _score_as_bench(path: Path) -> float:
    try:
        content = path.read_text()
    except Exception:
        return 0.0
    
    score = 0.0
    for pattern, points in BENCH_KEYWORDS:
        if re.search(pattern, content, re.IGNORECASE):
            score += points
    
    if "bench" in path.name.lower() or "perf" in path.name.lower():
        score += 0.1
    
    return score


def _get_test_command(path: Path) -> str:
    try:
        content = path.read_text()[:2000]
    except Exception:
        content = ""
    
    if path.suffix == ".py":
        if "import pytest" in content or "@pytest" in content:
            return f"pytest {path} -v"
        elif "unittest" in content:
            return f"python -m unittest {path}"
        else:
            return f"python {path}"
    elif path.suffix in [".cpp", ".cc", ".cu", ".hip"]:
        return f"# Build and run: {path.name}"
    else:
        return f"# Unknown: {path}"


def _expand_workspace(kernel_path: Path) -> Path:
    """Find the project root by walking up from *kernel_path*.

    When *kernel_path* is a directory (e.g. a repository root) we start the
    marker search from the directory itself, not its parent.  This ensures
    that ``/path/to/repo/.git`` is found when the caller passes ``/path/to/repo``.
    """
    markers = ["pyproject.toml", "setup.py", ".git", "tests", "op_tests"]

    current = kernel_path if kernel_path.is_dir() else kernel_path.parent
    for _ in range(15):
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return kernel_path if kernel_path.is_dir() else kernel_path.parent


def _get_kernel_type(content: str) -> str:
    if "@triton" in content or "tl." in content:
        return "triton"
    elif "__global__" in content and "hip" in content.lower():
        return "hip"
    elif "__global__" in content:
        return "cuda"
    return "unknown"


# ============================================================================
# Single Monolithic Tool
# ============================================================================

def _find_kernels_in_dir(directory: Path) -> list[dict]:
    """Recursively scan *directory* for kernel files and return info dicts."""
    extensions = {".py", ".cpp", ".cc", ".cu", ".hip"}
    kernels: list[dict] = []
    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix not in extensions:
            continue
        if _should_skip(candidate):
            continue
        if _is_kernel_file(candidate):
            try:
                content = candidate.read_text()[:3000]
            except Exception:
                content = ""
            kernels.append({
                "name": candidate.stem,
                "type": _get_kernel_type(content),
                "file": str(candidate),
            })
    return kernels


@mcp.tool()
def discover(
    kernel_path: str,
    max_tests: int = 5,
    max_benchmarks: int = 5
) -> dict:
    """
    Discover tests and benchmarks for a GPU kernel.
    
    Automatically finds related test and benchmark files using content-based
    detection. No configuration needed - works on any project structure.
    
    Args:
        kernel_path: Path to a kernel file (.py/.cu/.hip) OR a repository
            directory.  When a directory is given, all kernel files inside it
            are discovered first, and tests/benchmarks are matched against each
            discovered kernel.
        max_tests: Maximum number of test results to return (default: 5)
        max_benchmarks: Maximum number of benchmark results to return (default: 5)
    
    Returns:
        Complete discovery result with:
        - kernel: Name, type (triton/hip/cuda), file path  (or list when directory)
        - workspace: Detected project root directory
        - tests: List of {file, name, confidence, command} sorted by relevance
        - benchmarks: List of {file, name, confidence, command} sorted by relevance
        - summary: Human-readable summary of what was found
    
    Example:
        discover("/path/to/gemm_a16w16.py")
        discover("/path/to/repo")  # scans repo recursively for kernels
    """
    path = Path(kernel_path)
    if not path.exists():
        return {
            "error": f"Path not found: {kernel_path}",
            "kernel": None,
            "tests": [],
            "benchmarks": [],
            "summary": "Error: path not found"
        }

    # --- Directory mode: discover kernels first, then find tests for all ---
    if path.is_dir():
        workspace = _expand_workspace(path)
        discovered_kernels = _find_kernels_in_dir(workspace)

        kernel_files = {Path(k["file"]) for k in discovered_kernels}
        all_kernel_names = [k["name"] for k in discovered_kernels]
        all_kernel_parts: list[str] = []
        for kname in all_kernel_names:
            all_kernel_parts.extend(p.lower() for p in kname.split("_") if len(p) > 2)
        all_kernel_parts = list(set(all_kernel_parts))

        tests: list[dict] = []
        benchmarks: list[dict] = []
        extensions = [".py", ".cpp", ".cc", ".cu", ".hip"]

        for ext in extensions:
            for file_path in workspace.rglob(f"*{ext}"):
                if _should_skip(file_path):
                    continue
                if file_path in kernel_files:
                    continue
                if _is_kernel_file(file_path):
                    continue

                fname_lower = file_path.name.lower()

                test_score = _score_as_test(file_path)
                if test_score >= 0.3:
                    for kname in all_kernel_names:
                        if kname.lower() in fname_lower:
                            test_score += 1.0
                            break
                    else:
                        matches = sum(1 for p in all_kernel_parts if p in fname_lower)
                        if matches >= 2:
                            test_score += 0.3 * matches
                    tests.append({
                        "file": str(file_path),
                        "name": file_path.name,
                        "confidence": round(min(test_score, 1.0), 2),
                        "command": _get_test_command(file_path),
                    })

                bench_score = _score_as_bench(file_path)
                if bench_score >= 0.3:
                    for kname in all_kernel_names:
                        if kname.lower() in fname_lower:
                            bench_score += 1.0
                            break
                    else:
                        matches = sum(1 for p in all_kernel_parts if p in fname_lower)
                        if matches >= 2:
                            bench_score += 0.3 * matches
                    benchmarks.append({
                        "file": str(file_path),
                        "name": file_path.name,
                        "confidence": round(min(bench_score, 1.0), 2),
                        "command": f"python {file_path}",
                    })

        tests.sort(key=lambda x: x["confidence"], reverse=True)
        benchmarks.sort(key=lambda x: x["confidence"], reverse=True)

        test_count = len(tests)
        bench_count = len(benchmarks)
        k_count = len(discovered_kernels)
        summary = (
            f"Scanned repository: found {k_count} kernel(s), "
            f"{test_count} test(s), {bench_count} benchmark(s)"
        )
        if tests:
            summary += f". Recommended test: {tests[0]['file']}"

        return {
            "kernel": discovered_kernels if len(discovered_kernels) != 1 else discovered_kernels[0],
            "workspace": str(workspace),
            "tests": tests[:max_tests],
            "benchmarks": benchmarks[:max_benchmarks],
            "total_kernels_found": k_count,
            "total_tests_found": test_count,
            "total_benchmarks_found": bench_count,
            "summary": summary,
        }

    # --- Single-file mode (original behaviour) ---
    workspace = _expand_workspace(path)

    kernel_name = path.stem
    # Use parent dir name if file has a generic name
    _GENERIC_STEMS = {"kernel", "main", "module", "op", "impl"}
    if kernel_name.lower() in _GENERIC_STEMS and path.parent.name:
        kernel_name = path.parent.name

    try:
        content = path.read_text()[:3000]
        kernel_type = _get_kernel_type(content)
    except Exception:
        kernel_type = "unknown"

    kernel_parts = [p.lower() for p in kernel_name.split("_") if len(p) > 2]

    tests = []
    benchmarks = []
    extensions = [".py", ".cpp", ".cc", ".cu", ".hip"]

    for ext in extensions:
        for file_path in workspace.rglob(f"*{ext}"):
            if _should_skip(file_path):
                continue
            if file_path == path:
                continue
            if _is_kernel_file(file_path):
                continue

            relevance = _relevance_score(file_path, path, kernel_name, kernel_parts)

            test_score = _score_as_test(file_path)
            if test_score >= 0.3:
                # Combine: content score (0-1) + relevance bonus (0-3+)
                # Relevant tests score much higher than generic ones
                combined = test_score + relevance
                tests.append({
                    "file": str(file_path),
                    "name": file_path.name,
                    "confidence": round(combined, 2),
                    "command": _get_test_command(file_path)
                })

            bench_score = _score_as_bench(file_path)
            if bench_score >= 0.3:
                combined = bench_score + relevance
                benchmarks.append({
                    "file": str(file_path),
                    "name": file_path.name,
                    "confidence": round(combined, 2),
                    "command": f"python {file_path}"
                })

    tests.sort(key=lambda x: x["confidence"], reverse=True)
    benchmarks.sort(key=lambda x: x["confidence"], reverse=True)
    
    test_count = len(tests)
    bench_count = len(benchmarks)
    
    if test_count > 0 and bench_count > 0:
        summary = f"Found {test_count} test(s) and {bench_count} benchmark(s) for {kernel_name} ({kernel_type} kernel)"
    elif test_count > 0:
        summary = f"Found {test_count} test(s) for {kernel_name} ({kernel_type} kernel), no benchmarks"
    elif bench_count > 0:
        summary = f"Found {bench_count} benchmark(s) for {kernel_name} ({kernel_type} kernel), no tests"
    else:
        summary = f"No tests or benchmarks found for {kernel_name} ({kernel_type} kernel)"
    
    if tests:
        summary += f". Recommended test: {tests[0]['file']}"

    return {
        "kernel": {
            "name": kernel_name,
            "type": kernel_type,
            "file": str(path)
        },
        "workspace": str(workspace),
        "tests": tests[:max_tests],
        "benchmarks": benchmarks[:max_benchmarks],
        "total_tests_found": test_count,
        "total_benchmarks_found": bench_count,
        "summary": summary
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
