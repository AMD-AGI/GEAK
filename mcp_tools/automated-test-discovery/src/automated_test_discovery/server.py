"""
Automated Test Discovery MCP Server

Two-phase discovery for GPU kernels:
  Phase 1 (automated): Content-based scan, relevance scoring, ranking
  Phase 2 (LLM finisher, optional): Validates top results, isolates specific
    test functions for the target kernel, or creates a focused test if nothing
    matches.  Writes a focused test script to output_dir.

No configuration files needed - uses content-based detection.
"""

import json
import os
import re
import textwrap
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
    """,
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
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".eggs",
    "site-packages",
    ".tox",
    ".pytest_cache",
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

    Returns a multiplier (0.0 - 4.0+) based on:
    - Exact stem match: test_<kernel_name>.py scores highest
    - Substring match: kernel name appears in filename (lower than exact)
    - Path match: kernel name in file path
    - Partial match: kernel name parts in filename
    - Path proximity: file is in the same or nearby directory
    """
    score = 0.0
    fname_lower = file_path.name.lower()
    fstem_lower = file_path.stem.lower()
    fpath_lower = str(file_path).lower()
    kname_lower = kernel_name.lower()

    # Strip common test/bench prefixes to get the "subject" of the test file
    subject = fstem_lower
    for prefix in ("test_", "bench_", "benchmark_", "test", "bench"):
        if subject.startswith(prefix):
            subject = subject[len(prefix) :]
            break

    # Exact stem match: test_gemm_a8w8.py for kernel gemm_a8w8 (strongest)
    if subject == kname_lower:
        score += 4.0

    # Substring in filename: test_gemm_a8w8_blockscale.py for kernel gemm_a8w8
    elif kname_lower in fname_lower:
        score += 2.5

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
# Phase 2: LLM Finisher
# ============================================================================


def _init_llm_client():
    """Initialize the AMD LLM gateway client. Returns None if unavailable."""
    api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        return anthropic.Anthropic(
            api_key="dummy",
            base_url="https://llm-api.amd.com/Anthropic",
            default_headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "anthropic-version": "2023-10-16",
            },
        )
    except Exception:
        return None


def _verify_focused_test(focused_file: Path, timeout: int = 90) -> tuple[bool, str]:
    """Run a generated focused test and return (passed, output).

    A quick smoke-run to catch import errors, wrong tensor shapes, and
    missing arguments before the rest of the pipeline relies on the harness.
    """
    import subprocess as _sp

    abs_file = Path(focused_file).resolve()
    env = {**os.environ, "HIP_VISIBLE_DEVICES": "0"}
    try:
        result = _sp.run(
            ["python", str(abs_file)],
            capture_output=True, text=True, timeout=timeout,
            env=env, cwd=str(abs_file.parent),
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        passed = result.returncode == 0 and "FAIL" not in result.stdout
        return passed, output
    except _sp.TimeoutExpired:
        return False, f"Test timed out after {timeout}s"
    except Exception as e:
        return False, f"Failed to run test: {e}"


def _analyze_test_file(test_path: str) -> dict:
    """Extract structured facts from a test file for the LLM.

    Returns a dict with:
      - module_name: importable module name (stem)
      - directory: absolute directory containing the file
      - helper_functions: list of non-test function names (importable)
      - test_functions: list of test function names
      - parametrize_values: dict of param_name -> list of values
      - tolerances: list of (atol, rtol) tuples found
      - imports: list of import lines from the file
    """
    p = Path(test_path)
    if not p.is_file():
        return {}

    source = p.read_text()
    lines = source.splitlines()

    # Extract function names
    helpers, tests = [], []
    for line in lines:
        m = re.match(r"^def ([\w]+)\(", line)
        if m:
            name = m.group(1)
            (tests if name.startswith("test_") else helpers).append(name)

    # Extract class/enum definitions
    for line in lines:
        m = re.match(r"^class ([\w]+)", line)
        if m:
            helpers.append(m.group(1))

    # Extract @pytest.mark.parametrize values
    parametrize = {}
    for line in lines:
        m = re.match(r'^@pytest\.mark\.parametrize\("(\w+)",\s*\[(.+)\]\)', line)
        if m:
            param_name, values_str = m.group(1), m.group(2)
            parametrize[param_name] = values_str.strip()
        # Multi-param: @pytest.mark.parametrize("a, b", [...])
        m2 = re.match(r'^@pytest\.mark\.parametrize\(\s*"([\w, ]+)",\s*\[(.+)\]', line)
        if m2:
            param_names = m2.group(1).strip()
            parametrize[param_names] = m2.group(2).strip()

    # Extract tolerance values
    tolerances = set()
    for line in lines:
        m = re.search(r"atol\s*=\s*([\d.eE+-]+)", line)
        m2 = re.search(r"rtol\s*=\s*([\d.eE+-]+)", line)
        if m and m2:
            tolerances.add((m.group(1), m2.group(1)))
        elif m:
            tolerances.add((m.group(1), ""))

    return {
        "module_name": p.stem,
        "directory": str(p.parent.resolve()),
        "helper_functions": helpers,
        "test_functions": tests,
        "parametrize_values": parametrize,
        "tolerances": sorted(tolerances),
    }


def _build_finisher_prompt(
    kernel_file: Path,
    kernel_name: str,
    func_list: str,
    kernel_source: str,
    top_test_path: str,
    top_test_source: str,
    output_path: str,
    test_analysis: dict,
    *,
    verification_error: str | None = None,
) -> str:
    """Build the LLM prompt for focused test generation."""
    retry_block = ""
    if verification_error:
        retry_block = textwrap.dedent(f"""\
        PREVIOUS ATTEMPT FAILED -- The test you generated did not pass:
        ```
        {verification_error[:3000]}
        ```
        Fix the issues. The most common problems are:
        - Wrong import path: use the EXACT sys.path and module name from
          PRE-COMPUTED IMPORTS below
        - Missing function arguments: check the candidate test to see how
          it calls the kernel
        - Wrong tolerances: use the EXACT atol/rtol from PRE-COMPUTED IMPORTS

        """)

    # Build the pre-computed imports section
    imports_block = ""
    if test_analysis and test_analysis.get("helper_functions"):
        ta = test_analysis
        helper_list = ", ".join(ta["helper_functions"])
        tol_lines = ""
        if ta.get("tolerances"):
            tol_lines = "\n".join(
                f"  atol={a}, rtol={r}" for a, r in ta["tolerances"]
            )
        param_lines = ""
        if ta.get("parametrize_values"):
            param_lines = "\n".join(
                f"  {k}: [{v}]" for k, v in ta["parametrize_values"].items()
            )
        imports_block = textwrap.dedent(f"""\

        PRE-COMPUTED IMPORTS (use these EXACTLY -- do not change paths or names):

        The candidate test file has been analyzed. Here is what you MUST import:

        ```python
        import sys
        sys.path.insert(0, "{ta['directory']}")
        from {ta['module_name']} import (
            {helper_list}
        )
        ```

        Helper functions to import: {helper_list}
        Test functions in file: {', '.join(ta.get('test_functions', []))}
        {f"Tolerances found: {chr(10)}{tol_lines}" if tol_lines else "No explicit tolerances found -- use atol=0.1, rtol=0.1"}
        {f"Parametrize values:{chr(10)}{param_lines}" if param_lines else "No parametrize decorators found"}

        IMPORTANT: Use sys.path.insert(0, "{ta['directory']}") and
        from {ta['module_name']} import ... as shown above.
        Do NOT change these paths. Do NOT rewrite these functions.
        """)

    return textwrap.dedent(f"""\
    You are a test isolation agent. Given a kernel and candidate test files,
    produce a focused Python test script that tests ONLY the specified kernel function(s).
    {retry_block}
    TARGET KERNEL:
    - File: {kernel_file}
    - Name: {kernel_name}
    - Functions to test: {func_list}
    - Source:
    ```python
    {kernel_source}
    ```

    TOP CANDIDATE TEST (confidence-ranked):
    - File: {top_test_path}
    - Source:
    ```python
    {top_test_source}
    ```
    {imports_block}
    YOUR TASK:

    Write a focused Python test script. Your #1 priority is to IMPORT and
    REUSE code from the candidate test, NOT rewrite it.

    If a candidate test exists and is relevant:
    1. Use the PRE-COMPUTED IMPORTS above verbatim -- add the sys.path line
       and import all listed helper functions.
    2. Call the imported input generator (e.g., generate_*_inputs) to create
       test inputs. NEVER write your own input generation.
    3. Call the kernel function, call the imported reference function, and
       compare with the exact atol/rtol from the candidate test.
    4. Loop over a representative subset of the parametrize values listed
       above (at least 20-50 combinations covering all parameter axes).

    If no relevant candidate test exists:
    1. Read the kernel function signatures carefully.
    2. Create inputs with at least 3 shape configs (small, medium, large).
    3. Use atol=0.1, rtol=0.1 for GPU kernel tolerance.

    The script MUST:
    - Support `--correctness` (default) and `--profile` modes via argparse
    - Print PASS/FAIL per test case and "X/Y passed" summary
    - Exit 0 if all pass, 1 if any fail
    - Add workspace to sys.path for kernel imports
    - Use torch.manual_seed(42) for reproducibility

    Respond with ONLY a JSON object (no markdown fences):
    {{{{
        "top_test_is_relevant": true/false,
        "reason": "brief explanation",
        "focused_test_code": "the complete Python script as a string",
        "focused_command": "python {output_path}/test_{kernel_name}_focused.py"
    }}}}
    """)


def _llm_finalize_discovery(
    kernel_file: Path,
    kernel_name: str,
    kernel_functions: list[str],
    top_tests: list[dict],
    workspace: Path,
    output_dir: Path | None,
) -> dict | None:
    """Phase 2: Use LLM to validate top test results and generate a focused test.

    Reads the kernel source and the top test candidate, asks the LLM to:
    1. Confirm whether the top test actually tests the target kernel function(s)
    2. If yes: extract the specific test functions and generate a focused script
    3. If no: generate a minimal focused test from scratch

    After generation, runs the test to verify it works. If it fails, retries
    once with the error output. If still failing, falls back to None so the
    caller can use the original test command.

    Returns a dict with 'focused_test_file' and 'focused_command', or None if
    the LLM is unavailable or fails.
    """
    client = _init_llm_client()
    if not client:
        return None

    try:
        kernel_source = kernel_file.read_text()[:8000]
    except Exception:
        return None

    top_test_source = ""
    top_test_path = ""
    test_analysis: dict = {}
    if top_tests:
        top_test_path = top_tests[0]["file"]
        try:
            top_test_source = Path(top_test_path).read_text()[:10000]
        except Exception:
            top_test_source = ""
        try:
            test_analysis = _analyze_test_file(top_test_path)
        except Exception:
            test_analysis = {}

    func_list = ", ".join(kernel_functions[:5]) if kernel_functions else kernel_name
    out_dir = output_dir or (workspace / ".geak_discovery_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    focused_file = out_dir / f"test_{kernel_name}_focused.py"

    verification_error: str | None = None

    for attempt in range(2):
        prompt = _build_finisher_prompt(
            kernel_file=kernel_file,
            kernel_name=kernel_name,
            func_list=func_list,
            kernel_source=kernel_source,
            top_test_path=top_test_path,
            top_test_source=top_test_source,
            output_path=str(out_dir),
            test_analysis=test_analysis,
            verification_error=verification_error,
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4.5",
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            result_text = response.content[0].text.strip()
            if result_text.startswith("```"):
                result_text = re.sub(r"^```\w*\n?", "", result_text)
                result_text = re.sub(r"\n?```$", "", result_text)
            result = json.loads(result_text)
        except Exception:
            return None

        focused_code = result.get("focused_test_code", "")
        if not focused_code:
            return None

        focused_file.write_text(focused_code)

        passed, output = _verify_focused_test(focused_file)
        if passed:
            return {
                "focused_test_file": str(focused_file),
                "focused_command": f"python {focused_file}",
                "top_test_is_relevant": result.get("top_test_is_relevant", False),
                "reason": result.get("reason", ""),
                "verified": True,
            }

        if attempt == 0:
            verification_error = output

    # Both attempts failed -- return the result but mark as unverified so
    # the caller can decide whether to fall back to the original test command.
    return {
        "focused_test_file": str(focused_file),
        "focused_command": f"python {focused_file}",
        "top_test_is_relevant": result.get("top_test_is_relevant", False),
        "reason": result.get("reason", ""),
        "verified": False,
        "verification_error": (verification_error or "")[:1000],
    }


# ============================================================================
# Kernel scanning
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
            kernels.append(
                {
                    "name": candidate.stem,
                    "type": _get_kernel_type(content),
                    "file": str(candidate),
                }
            )
    return kernels


@mcp.tool()
def discover(
    kernel_path: str,
    kernel_function: str = "",
    output_dir: str = "",
    max_tests: int = 5,
    max_benchmarks: int = 5,
    use_llm: bool = True,
) -> dict:
    """
    Discover tests and benchmarks for a GPU kernel.

    Two-phase discovery:
      Phase 1 (automated): Content-based scan with relevance scoring.
      Phase 2 (LLM finisher, optional): Validates top results and generates
        a focused test script that tests ONLY the target kernel function(s).

    Args:
        kernel_path: Path to a kernel file (.py/.cu/.hip) OR a repository
            directory.  When a directory is given, all kernel files inside it
            are discovered first, and tests/benchmarks are matched against each
            discovered kernel.
        kernel_function: Name of the specific kernel function to test
            (e.g. "_rope_fwd" from resolve_kernel_url).  When provided,
            Phase 2 uses this to isolate the exact test functions needed.
        output_dir: Directory where the focused test script is written.
            Defaults to <workspace>/.geak_discovery_output/.
        max_tests: Maximum number of test results to return (default: 5)
        max_benchmarks: Maximum number of benchmark results to return (default: 5)
        use_llm: Whether to run Phase 2 LLM finisher (default: True).
            Set to False for fast automated-only results.

    Returns:
        Complete discovery result with:
        - kernel: Name, type (triton/hip/cuda), file path  (or list when directory)
        - workspace: Detected project root directory
        - tests: List of {file, name, confidence, command} sorted by relevance
        - benchmarks: List of {file, name, confidence, command} sorted by relevance
        - focused_test: (Phase 2) Focused test script path and command, if generated
        - summary: Human-readable summary of what was found

    Example:
        discover("/path/to/rope.py", kernel_function="_rope_fwd")
        discover("/path/to/repo")  # scans repo recursively for kernels
    """
    path = Path(kernel_path)
    if not path.exists():
        return {
            "error": f"Path not found: {kernel_path}",
            "kernel": None,
            "tests": [],
            "benchmarks": [],
            "summary": "Error: path not found",
        }

    # --- Directory mode: discover kernels, then find per-kernel tests ---
    if path.is_dir():
        workspace = _expand_workspace(path)
        discovered_kernels = _find_kernels_in_dir(workspace)

        # Apply generic stem fix (kernel.py -> parent dir name)
        _GENERIC_STEMS = {"kernel", "main", "module", "op", "impl"}
        for k in discovered_kernels:
            kpath = Path(k["file"])
            if k["name"].lower() in _GENERIC_STEMS and kpath.parent.name:
                k["name"] = kpath.parent.name

        kernel_files = {Path(k["file"]) for k in discovered_kernels}

        # Collect all candidate test/benchmark files once
        candidate_files: list[Path] = []
        extensions = [".py", ".cpp", ".cc", ".cu", ".hip"]
        for ext in extensions:
            for file_path in workspace.rglob(f"*{ext}"):
                if _should_skip(file_path):
                    continue
                if file_path in kernel_files:
                    continue
                if _is_kernel_file(file_path):
                    continue
                candidate_files.append(file_path)

        # Score each candidate as test/bench (content-based, computed once)
        candidate_test_scores: dict[Path, float] = {}
        candidate_bench_scores: dict[Path, float] = {}
        for fp in candidate_files:
            ts = _score_as_test(fp)
            if ts >= 0.3:
                candidate_test_scores[fp] = ts
            bs = _score_as_bench(fp)
            if bs >= 0.3:
                candidate_bench_scores[fp] = bs

        # For each kernel, compute per-kernel relevance and find best matches
        global_tests: list[dict] = []
        global_benchmarks: list[dict] = []
        global_test_seen: set[str] = set()
        global_bench_seen: set[str] = set()

        for k in discovered_kernels:
            kname = k["name"]
            kpath = Path(k["file"])
            kparts = [p.lower() for p in kname.split("_") if len(p) > 2]

            per_kernel_tests: list[dict] = []
            per_kernel_benchmarks: list[dict] = []

            for fp in candidate_files:
                relevance = _relevance_score(fp, kpath, kname, kparts)

                if fp in candidate_test_scores:
                    combined = candidate_test_scores[fp] + relevance
                    entry = {
                        "file": str(fp),
                        "name": fp.name,
                        "confidence": round(combined, 2),
                        "command": _get_test_command(fp),
                    }
                    per_kernel_tests.append(entry)
                    if str(fp) not in global_test_seen:
                        global_tests.append(entry)
                        global_test_seen.add(str(fp))

                if fp in candidate_bench_scores:
                    combined = candidate_bench_scores[fp] + relevance
                    entry = {
                        "file": str(fp),
                        "name": fp.name,
                        "confidence": round(combined, 2),
                        "command": f"python {fp}",
                    }
                    per_kernel_benchmarks.append(entry)
                    if str(fp) not in global_bench_seen:
                        global_benchmarks.append(entry)
                        global_bench_seen.add(str(fp))

            per_kernel_tests.sort(key=lambda x: x["confidence"], reverse=True)
            per_kernel_benchmarks.sort(key=lambda x: x["confidence"], reverse=True)

            k["recommended_test"] = per_kernel_tests[0] if per_kernel_tests else None
            k["recommended_benchmark"] = per_kernel_benchmarks[0] if per_kernel_benchmarks else None

        global_tests.sort(key=lambda x: x["confidence"], reverse=True)
        global_benchmarks.sort(key=lambda x: x["confidence"], reverse=True)

        test_count = len(global_tests)
        bench_count = len(global_benchmarks)
        k_count = len(discovered_kernels)

        # Build per-kernel summary
        rec_parts = []
        for k in discovered_kernels:
            rt = k.get("recommended_test")
            if rt:
                rec_parts.append(f"{k['name']} -> {rt['file']}")
            else:
                rec_parts.append(f"{k['name']} -> (no test found)")

        summary = f"Scanned repository: found {k_count} kernel(s), {test_count} test(s), {bench_count} benchmark(s)"
        if rec_parts:
            summary += ". Per-kernel recommendations: " + "; ".join(rec_parts[:10])

        return {
            "kernel": discovered_kernels if len(discovered_kernels) != 1 else discovered_kernels[0],
            "workspace": str(workspace),
            "tests": global_tests[:max_tests],
            "benchmarks": global_benchmarks[:max_benchmarks],
            "total_kernels_found": k_count,
            "total_tests_found": test_count,
            "total_benchmarks_found": bench_count,
            "summary": summary,
        }

    # --- Single-file mode ---
    workspace = _expand_workspace(path)

    kernel_name = path.stem
    _GENERIC_STEMS = {"kernel", "main", "module", "op", "impl"}
    if kernel_name.lower() in _GENERIC_STEMS and path.parent.name:
        kernel_name = path.parent.name

    try:
        content = path.read_text()[:5000]
        kernel_type = _get_kernel_type(content)
    except Exception:
        content = ""
        kernel_type = "unknown"

    # Extract kernel function names from source
    kernel_functions: list[str] = []
    if kernel_function:
        kernel_functions.append(kernel_function)
    # Also extract @triton.jit decorated functions and __global__ functions
    for m in re.finditer(r"@triton\.jit\s*\n\s*def\s+(\w+)", content):
        if m.group(1) not in kernel_functions:
            kernel_functions.append(m.group(1))
    for m in re.finditer(r"__global__\s+void\s+(\w+)", content):
        if m.group(1) not in kernel_functions:
            kernel_functions.append(m.group(1))

    kernel_parts = [p.lower() for p in kernel_name.split("_") if len(p) > 2]
    # Add kernel_function parts for matching too
    if kernel_function:
        kernel_parts.extend(
            p.lower() for p in kernel_function.split("_") if len(p) > 2 and p.lower() not in kernel_parts
        )

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

            # Bonus: if kernel_function name appears inside the test file content
            if kernel_functions and relevance < 2.0:
                try:
                    test_content = file_path.read_text()[:5000]
                    for kf in kernel_functions:
                        if kf in test_content:
                            relevance += 2.0
                            break
                except Exception:
                    pass

            test_score = _score_as_test(file_path)
            if test_score >= 0.3:
                combined = test_score + relevance
                tests.append(
                    {
                        "file": str(file_path),
                        "name": file_path.name,
                        "confidence": round(combined, 2),
                        "command": _get_test_command(file_path),
                    }
                )

            bench_score = _score_as_bench(file_path)
            if bench_score >= 0.3:
                combined = bench_score + relevance
                benchmarks.append(
                    {
                        "file": str(file_path),
                        "name": file_path.name,
                        "confidence": round(combined, 2),
                        "command": f"python {file_path}",
                    }
                )

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

    # --- Phase 2: LLM finisher (optional) ---
    focused_test = None
    if use_llm:
        out_dir = Path(output_dir) if output_dir else None
        focused_test = _llm_finalize_discovery(
            kernel_file=path,
            kernel_name=kernel_name,
            kernel_functions=kernel_functions,
            top_tests=tests[:3],
            workspace=workspace,
            output_dir=out_dir,
        )
        if focused_test:
            summary += f". Focused test: {focused_test['focused_test_file']}"

    result = {
        "kernel": {
            "name": kernel_name,
            "type": kernel_type,
            "file": str(path),
            "functions": kernel_functions,
        },
        "workspace": str(workspace),
        "tests": tests[:max_tests],
        "benchmarks": benchmarks[:max_benchmarks],
        "total_tests_found": test_count,
        "total_benchmarks_found": bench_count,
        "summary": summary,
    }
    if focused_test:
        result["focused_test"] = focused_test

    return result


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
