"""
Automated Test Discovery MCP Server

Single-tool MCP for discovering tests and benchmarks for GPU kernels.
No configuration files needed - uses content-based detection.
"""

import re
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    name="automated-test-discovery",
    instructions="""
    Automated Test Discovery for GPU Kernels.
    
    Single tool: discover - finds tests, benchmarks, and kernel info.
    
    Just provide a kernel file path and it returns everything:
    - Kernel name and type (triton/hip/cuda)
    - Related test files with confidence scores and run commands
    - Related benchmark files with confidence scores
    - Project workspace path
    
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
    markers = ["pyproject.toml", "setup.py", ".git", "tests", "op_tests"]
    
    current = kernel_path.parent
    for _ in range(15):
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    return kernel_path.parent


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
        kernel_path: Path to the kernel file (.py for Triton, .cu/.hip for CUDA/HIP)
        max_tests: Maximum number of test results to return (default: 5)
        max_benchmarks: Maximum number of benchmark results to return (default: 5)
    
    Returns:
        Complete discovery result with:
        - kernel: Name, type (triton/hip/cuda), file path
        - workspace: Detected project root directory
        - tests: List of {file, name, confidence, command} sorted by relevance
        - benchmarks: List of {file, name, confidence, command} sorted by relevance
        - summary: Human-readable summary of what was found
    
    Example:
        discover("/path/to/gemm_a16w16.py")
        
        Returns:
        {
            "kernel": {"name": "gemm_a16w16", "type": "triton", "file": "..."},
            "workspace": "/path/to/project",
            "tests": [
                {"name": "test_gemm_a16w16.py", "confidence": 1.0, "command": "pytest ..."}
            ],
            "benchmarks": [
                {"name": "bench_gemm_a16w16.py", "confidence": 1.0, "command": "python ..."}
            ],
            "summary": "Found 1 test and 1 benchmark for gemm_a16w16 (triton kernel)"
        }
    """
    path = Path(kernel_path)
    if not path.exists():
        return {
            "error": f"Kernel file not found: {kernel_path}",
            "kernel": None,
            "tests": [],
            "benchmarks": [],
            "summary": "Error: file not found"
        }
    
    # Expand to workspace
    workspace = _expand_workspace(path)
    
    # Get kernel info
    kernel_name = path.stem
    try:
        content = path.read_text()[:3000]
        kernel_type = _get_kernel_type(content)
    except Exception:
        kernel_type = "unknown"
    
    kernel_parts = [p.lower() for p in kernel_name.split("_") if len(p) > 2]
    
    # Scan for tests and benchmarks
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
            
            fname_lower = file_path.name.lower()
            
            # Score as test
            test_score = _score_as_test(file_path)
            if test_score >= 0.3:
                if kernel_name.lower() in fname_lower:
                    test_score += 1.0
                elif kernel_parts:
                    matches = sum(1 for p in kernel_parts if p in fname_lower)
                    if matches >= 2:
                        test_score += 0.3 * matches
                
                tests.append({
                    "file": str(file_path),
                    "name": file_path.name,
                    "confidence": round(min(test_score, 1.0), 2),
                    "command": _get_test_command(file_path)
                })
            
            # Score as benchmark
            bench_score = _score_as_bench(file_path)
            if bench_score >= 0.3:
                if kernel_name.lower() in fname_lower:
                    bench_score += 1.0
                elif kernel_parts:
                    matches = sum(1 for p in kernel_parts if p in fname_lower)
                    if matches >= 2:
                        bench_score += 0.3 * matches
                
                benchmarks.append({
                    "file": str(file_path),
                    "name": file_path.name,
                    "confidence": round(min(bench_score, 1.0), 2),
                    "command": f"python {file_path}"
                })
    
    # Sort by confidence
    tests.sort(key=lambda x: x["confidence"], reverse=True)
    benchmarks.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Build summary
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
    
    # Add top recommendation
    if tests:
        summary += f". Recommended test: {tests[0]['name']}"
    
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
