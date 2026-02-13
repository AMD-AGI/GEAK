"""Discovery Pipeline - Runs before the agent loop.

This module handles:
1. Kernel discovery - Find kernel files in the workspace
2. Test discovery - Find existing tests or prompt to create (content-based)
3. Benchmark discovery - Find performance benchmarks (content-based)
4. User confirmation - Interactive prompts to confirm/edit
5. LLM-assisted analysis - Optional LLM for uncertain cases
6. Configurable patterns - Per-project customization via config files

Discovery Modes:
- User provides: --test "pytest..." --bench "python..."
- Agent discovers: Search repo for test files, confirm with user
- Agent creates: (TODO) Help create tests if none exist

Supported Languages:
- Python: pytest, unittest, custom frameworks
- C++: GTest, Catch2, custom test harnesses
- HIP/CUDA: kernel tests

Content-based detection keywords:
- Tests: assert, pytest, allclose, correctness, assertEqual, TEST(), EXPECT_*
- Benchmarks: elapsed_time, latency, throughput, TFLOPS, benchmark, warmup
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from minisweagent.tools.resolve_kernel_url_impl import RESOLVED_DIR_NAME

# Try to import TOML support
try:
    import tomllib

    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib

        HAS_TOML = True
    except ImportError:
        HAS_TOML = False

# Try to import LLM support
try:
    import anthropic

    HAS_LLM = True
except ImportError:
    HAS_LLM = False


@dataclass
class DiscoveryConfig:
    """
    Configuration for discovery patterns.

    Three-layer resolution (highest to lowest priority):
      1. Per-project:  <repo>/.geak/discovery.toml
      2. Auto-detect:  runtime scan of the codebase (appends to loaded config)
      3. Defaults:     discovery_defaults.toml shipped with GEAK

    The ``_load_config`` helper in ``DiscoveryPipeline`` populates this.
    """

    # --- Kernel detection ---
    kernel_patterns: list[str] = field(default_factory=list)
    wrapper_functions: list[str] = field(default_factory=list)

    # --- Test keywords  (regex, weight) ---
    test_python_keywords: list[tuple] = field(default_factory=list)
    test_cpp_keywords: list[tuple] = field(default_factory=list)

    # --- Benchmark keywords (regex, weight) ---
    bench_keywords: list[tuple] = field(default_factory=list)

    # --- Workspace settings ---
    skip_dirs: list[str] = field(default_factory=list)
    project_markers: list[str] = field(default_factory=list)
    monorepo_markers: list[str] = field(default_factory=list)

    # Whether C++ files should be searched
    include_cpp: bool = True
    # Extra exclude directories (from per-project config)
    exclude_dirs: list[str] = field(default_factory=list)


@dataclass
class KernelInfo:
    """Information about a discovered kernel."""

    file_path: Path
    kernel_name: str
    kernel_type: str  # triton, hip, cuda
    function_names: list[str] = field(default_factory=list)
    has_jit_decorator: bool = False
    has_autotune: bool = False


@dataclass
class TestInfo:
    """Information about a discovered test."""

    file_path: Path
    test_type: str  # pytest, script, makefile
    command: str  # Command to run the test
    confidence: float  # 0-1, how confident we are this is the right test


@dataclass
class BenchmarkInfo:
    """Information about a discovered benchmark."""

    file_path: Path
    bench_type: str  # pytest, script, custom
    command: str
    confidence: float


@dataclass
class DiscoveryResult:
    """Result of the discovery pipeline."""

    kernels: list[KernelInfo] = field(default_factory=list)
    tests: list[TestInfo] = field(default_factory=list)
    benchmarks: list[BenchmarkInfo] = field(default_factory=list)
    workspace_path: Path = None
    needs_user_confirmation: bool = True
    user_provided_test: str | None = None
    user_provided_bench: str | None = None


class DiscoveryPipeline:
    """
    Discovery pipeline that runs before the agent loop.

    Progressive discovery:
    1. Check if user provided explicit commands
    2. If not, search for test/bench files (CONTENT-BASED)
    3. Present findings to user for confirmation
    4. Offer to create tests if none found (TODO)
    5. Use LLM for uncertain cases (confidence 0.3-0.6)
    6. Support configurable patterns per-project via .geak/discovery.toml

    Pattern resolution priority (highest to lowest):
    1. Per-project config:  <repo>/.geak/discovery.toml
    2. Auto-detection:      _auto_detect_patterns()
    3. Built-in defaults:   discovery_defaults.toml shipped with GEAK
    """

    # Path to the built-in defaults TOML (relative to the package config dir)
    _DEFAULTS_TOML = "discovery_defaults.toml"

    def __init__(self, workspace_path: Path = None, use_llm: bool = False):
        self.workspace = Path(workspace_path) if workspace_path else Path.cwd()
        self.result = DiscoveryResult(workspace_path=self.workspace)
        self.use_llm = use_llm and HAS_LLM
        self._llm_client = None
        self._kernel_file = None

        # Load patterns from TOML config (defaults + per-project overrides)
        self.config = self._load_config()

        # Mutable keyword lists -- start from loaded config, auto-detect appends later
        self._test_keywords = list(self.config.test_python_keywords)
        self._bench_keywords = list(self.config.bench_keywords)

        if self.use_llm:
            self._init_llm()

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    @staticmethod
    def _toml_keywords_to_tuples(entries: list[dict]) -> list[tuple]:
        """Convert ``[{pattern: ..., weight: ...}, ...]`` to ``[(pattern, weight), ...]``."""
        return [(e["pattern"], e.get("weight", 0.3)) for e in entries if "pattern" in e]

    def _load_config(self) -> DiscoveryConfig:
        """Load discovery patterns from the defaults TOML and per-project overrides.

        Resolution order:
        1. Load ``discovery_defaults.toml`` shipped with the package.
        2. Deep-merge any ``<workspace>/.geak/discovery.toml`` on top.
        3. Populate a ``DiscoveryConfig`` dataclass from the merged dict.
        """
        defaults = self._load_defaults_toml()
        overrides = self._load_project_toml()

        # Deep-merge: overrides extend defaults for list fields, replace for scalars
        merged = self._deep_merge(defaults, overrides)

        # Build DiscoveryConfig from the merged dict
        kernel_cfg = merged.get("kernel", {})
        test_cfg = merged.get("test", {})
        bench_cfg = merged.get("benchmark", {})
        ws_cfg = merged.get("workspace", {})

        cfg = DiscoveryConfig(
            kernel_patterns=kernel_cfg.get("patterns", []),
            wrapper_functions=kernel_cfg.get("wrapper_functions", []),
            test_python_keywords=self._toml_keywords_to_tuples(test_cfg.get("python_keywords", [])),
            test_cpp_keywords=self._toml_keywords_to_tuples(test_cfg.get("cpp_keywords", [])),
            bench_keywords=self._toml_keywords_to_tuples(bench_cfg.get("keywords", [])),
            skip_dirs=ws_cfg.get("skip_dirs", []),
            project_markers=ws_cfg.get("project_markers", []),
            monorepo_markers=ws_cfg.get("monorepo_markers", []),
        )

        # Handle ``skip_dirs_extra`` from per-project config (additive)
        extra_skip = ws_cfg.get("skip_dirs_extra", [])
        if extra_skip:
            cfg.exclude_dirs = list(extra_skip)

        return cfg

    def _load_defaults_toml(self) -> dict:
        """Load the built-in ``discovery_defaults.toml``."""
        if not HAS_TOML:
            return {}
        defaults_path = Path(__file__).resolve().parent.parent / "config" / self._DEFAULTS_TOML
        if not defaults_path.exists():
            return {}
        try:
            with open(defaults_path, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    def _load_project_toml(self) -> dict:
        """Load per-project ``<workspace>/.geak/discovery.toml`` if it exists."""
        if not HAS_TOML:
            return {}
        project_toml = self.workspace / ".geak" / "discovery.toml"
        if not project_toml.exists():
            return {}
        try:
            with open(project_toml, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge *override* into *base*.

        - Lists are **extended** (override items appended to base).
        - Dicts are merged recursively.
        - Scalars are replaced by the override value.
        """
        merged = dict(base)
        for key, val in override.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(val, dict):
                    merged[key] = DiscoveryPipeline._deep_merge(merged[key], val)
                elif isinstance(merged[key], list) and isinstance(val, list):
                    merged[key] = merged[key] + val
                else:
                    merged[key] = val
            else:
                merged[key] = val
        return merged

    def _auto_detect_patterns(self):
        """
        Auto-detect custom test/benchmark patterns from the codebase.

        Scans a sample of files to learn project-specific patterns:
        - Custom test decorators (@perftest, @benchmark, etc.)
        - Custom assertion functions (checkAllclose, verify_output, etc.)
        - Project-specific imports (from test_common import, etc.)

        This makes discovery work for ANY project structure without config.
        """
        # Patterns to look for in files
        custom_patterns = {
            # Custom decorators
            r"@(\w+test\w*)\s*\(": "test_decorator",
            r"@(\w*bench\w*)\s*\(": "bench_decorator",
            r"@(\w*perf\w*)\s*\(": "bench_decorator",
            # Custom assertion/check functions
            r"(check\w+)\s*\(": "check_func",
            r"(verify\w+)\s*\(": "check_func",
            # Custom imports from test utilities
            r"from\s+(\w*test_common\w*)\s+import": "test_import",
        }

        detected_keywords = set()

        # Sample files that look like tests (by name)
        sample_count = 0
        for py_file in self.workspace.rglob("*test*.py"):
            if sample_count >= 30:
                break
            if self._should_skip_file(py_file):
                continue

            try:
                content = py_file.read_text()[:5000]

                for pattern, ptype in custom_patterns.items():
                    for match in re.finditer(pattern, content):
                        keyword = match.group(1)
                        # Filter out common/generic patterns
                        if len(keyword) > 4 and keyword.lower() not in ["test", "assert", "check", "verify"]:
                            detected_keywords.add((keyword, ptype))

                sample_count += 1
            except Exception:
                continue

        # Add detected patterns to keywords
        for keyword, ptype in detected_keywords:
            if ptype == "test_decorator":
                self._test_keywords.append((rf"@{keyword}\s*\(", 0.35))
            elif ptype == "bench_decorator":
                self._bench_keywords.append((rf"@{keyword}\s*\(", 0.35))
            elif ptype == "check_func":
                self._test_keywords.append((rf"{keyword}\s*\(", 0.3))
            elif ptype == "test_import":
                self._test_keywords.append((rf"from\s+{keyword}\s+import", 0.25))

    def _init_llm(self):
        """Initialize LLM client for smart discovery."""
        api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY")
        if not api_key:
            self.use_llm = False
            return

        try:
            self._llm_client = anthropic.Anthropic(
                api_key="dummy",
                base_url="https://llm-api.amd.com/Anthropic",
                default_headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    "anthropic-version": "2023-10-16",
                },
            )
        except Exception:
            self.use_llm = False

    def _llm_analyze_file(self, file_path: Path, file_type: str) -> dict | None:
        """Use LLM to analyze a file when content-based detection is uncertain."""
        if not self.use_llm or not self._llm_client:
            return None

        try:
            content = file_path.read_text()[:3000]  # Limit content size
        except Exception:
            return None

        prompt = f"""Analyze this Python file and determine if it's a {file_type}.

File: {file_path.name}
Content (first 3000 chars):
```python
{content}
```

Respond with JSON only:
{{
    "is_{file_type}": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "command": "how to run this file (e.g., 'pytest file.py' or 'python file.py')"
}}
"""

        try:
            response = self._llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            import json

            result_text = response.content[0].text
            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text.strip())
        except Exception:
            return None

    def run(
        self,
        kernel_path: Path | None = None,
        test_command: str | None = None,
        bench_command: str | None = None,
        interactive: bool = True,
    ) -> DiscoveryResult:
        """
        Run the full discovery pipeline.

        Args:
            kernel_path: Explicit path to kernel file/directory
            test_command: User-provided test command (skips discovery)
            bench_command: User-provided benchmark command (skips discovery)
            interactive: Whether to prompt user for confirmation

        Returns:
            DiscoveryResult with all discovered information
        """
        print("\n" + "=" * 60)
        print("  DISCOVERY PIPELINE")
        print("=" * 60)

        # Store user-provided commands
        self.result.user_provided_test = test_command
        self.result.user_provided_bench = bench_command

        # If kernel_path is a file, expand workspace to find related tests
        self._kernel_file = None
        if kernel_path and kernel_path.is_file():
            self._kernel_file = kernel_path
            # Search for workspace root (look for common markers)
            self._expand_workspace_for_file(kernel_path)

        # Step 0: Auto-detect patterns from the codebase
        self._auto_detect_patterns()

        # Step 1: Discover kernels
        self._discover_kernels(kernel_path)

        # Step 2: Discover tests (unless user provided)
        if test_command:
            self.result.tests.append(
                TestInfo(file_path=Path("user-provided"), test_type="user", command=test_command, confidence=1.0)
            )
            self.result.needs_user_confirmation = False
        else:
            self._discover_tests()

        # Step 3: Discover benchmarks (unless user provided)
        if bench_command:
            self.result.benchmarks.append(
                BenchmarkInfo(file_path=Path("user-provided"), bench_type="user", command=bench_command, confidence=1.0)
            )
        else:
            self._discover_benchmarks()

        # Step 4: Display findings
        self._display_findings()

        # Step 5: User confirmation (if interactive)
        if interactive and self.result.needs_user_confirmation:
            self._prompt_user_confirmation()

        return self.result

    def _expand_workspace_for_file(self, kernel_file: Path):
        """
        Expand workspace when given a single file to find related tests.

        Smart expansion:
        - Stops at project root markers (pyproject.toml, .git, etc.)
        - Stops at monorepo boundaries (lerna.json, nx.json, etc.)
        - Stops at resolved-clone boundaries (RESOLVED_DIR_NAME) so we never
          walk above a cloned repo into the agent's own workspace.
        - Prefers the closest valid root
        """
        # Project root markers (we want to expand TO these)
        project_markers = self.config.project_markers or [
            "pyproject.toml", "setup.py", "setup.cfg", ".git",
            "op_tests", "tests", "Makefile", "CMakeLists.txt",
        ]

        # Monorepo markers (we want to STOP BEFORE these if they're not the immediate parent)
        monorepo_markers = self.config.monorepo_markers or [
            "lerna.json", "nx.json", "pnpm-workspace.yaml", "rush.json",
        ]

        current = kernel_file.parent
        best_workspace = None

        for depth in range(15):  # Max 15 levels up
            # Hard stop: never walk above a resolved-clone boundary.
            if current.name == RESOLVED_DIR_NAME:
                print(f"      Resolved-clone boundary reached at: {current}")
                break

            # Check for monorepo boundary (stop expansion)
            if depth > 0:  # Allow immediate parent
                for marker in monorepo_markers:
                    if (current / marker).exists():
                        print(f"      Monorepo boundary detected at: {current}")
                        if best_workspace:
                            self.workspace = best_workspace
                            self.result.workspace_path = best_workspace
                            print(f"      Expanded workspace to: {best_workspace}")
                        return

            # Check for project root markers
            for marker in project_markers:
                if (current / marker).exists():
                    best_workspace = current
                    break

            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent

        # Use best found workspace
        if best_workspace:
            self.workspace = best_workspace
            self.result.workspace_path = best_workspace
            print(f"      Expanded workspace to: {best_workspace}")

    def _discover_kernels(self, kernel_path: Path | None = None):
        """Discover kernel files in the workspace."""
        print("\n[1/4] Discovering kernels...")

        search_path = kernel_path or self.workspace

        if search_path.is_file():
            # Single file provided
            kernel_info = self._analyze_kernel_file(search_path)
            if kernel_info:
                self.result.kernels.append(kernel_info)
        else:
            # Search directory
            for py_file in search_path.rglob("*.py"):
                # Skip test files and common non-kernel dirs
                if self._should_skip_file(py_file):
                    continue

                kernel_info = self._analyze_kernel_file(py_file)
                if kernel_info:
                    self.result.kernels.append(kernel_info)

        print(f"      Found {len(self.result.kernels)} kernel(s)")

    def _analyze_kernel_file(self, file_path: Path) -> KernelInfo | None:
        """Analyze a file to determine if it contains kernels.

        Detects both direct kernel definitions (with @triton.jit, __global__, etc.)
        and **wrapper files** that import and launch kernel functions defined elsewhere.
        Wrapper files are common in projects like aiter where the public API lives in
        a separate module from the @triton.jit implementations.
        """
        try:
            content = file_path.read_text()
        except Exception:
            return None

        # Check for kernel patterns
        kernel_type = None
        has_jit = False
        has_autotune = False

        if "@triton.jit" in content or "tl.load" in content:
            kernel_type = "triton"
            has_jit = "@triton.jit" in content
            has_autotune = "@triton.autotune" in content
        elif "__global__" in content and ("__device__" in content or "hipLaunch" in content):
            kernel_type = "hip"
        elif "__global__" in content and "cuda" in content.lower():
            kernel_type = "cuda"

        # Detect Triton wrapper files: they import triton and import/call kernel
        # functions (e.g. ``from ..._triton_kernels.rope.rope import _rope_kernel_*``)
        # but don't define @triton.jit kernels themselves.
        if not kernel_type:
            has_triton_import = bool(re.search(r"import\s+triton", content))
            imports_kernel_funcs = bool(
                re.search(r"from\s+\S+\s+import\s+[^)]*_kernel", content, re.DOTALL)
            )
            calls_kernel_with_grid = bool(re.search(r"_kernel\w*\[", content))
            if has_triton_import and (imports_kernel_funcs or calls_kernel_with_grid):
                kernel_type = "triton"

        if not kernel_type:
            return None

        # Extract function names
        function_names = []

        # Find @triton.jit decorated functions
        jit_pattern = r"@triton\.jit\s*\n\s*def\s+(\w+)"
        for match in re.finditer(jit_pattern, content):
            function_names.append(match.group(1))

        # Find wrapper functions (configurable via discovery.toml)
        wrapper_fns = self.config.wrapper_functions or ["forward", "main"]
        wrapper_alt = "|".join(re.escape(fn) for fn in wrapper_fns)
        wrapper_pattern = rf"def\s+({wrapper_alt})\s*\("
        for match in re.finditer(wrapper_pattern, content):
            if match.group(1) not in function_names:
                function_names.append(match.group(1))

        # For wrapper files with no @triton.jit, also extract public API functions
        # (e.g. rope_fwd, rope_bwd) so they appear in the kernel info.
        if not has_jit:
            public_fn_pattern = r"^def\s+(\w+)\s*\("
            for match in re.finditer(public_fn_pattern, content, re.MULTILINE):
                fname = match.group(1)
                if not fname.startswith("_") and fname not in function_names:
                    function_names.append(fname)

        # Use file stem as kernel name (more reliable for matching tests/benchmarks)
        # Function names may be helpers like "fast_exp" that don't match test names
        kernel_name = file_path.stem

        return KernelInfo(
            file_path=file_path,
            kernel_name=kernel_name,
            kernel_type=kernel_type,
            function_names=function_names,
            has_jit_decorator=has_jit,
            has_autotune=has_autotune,
        )

    def _discover_tests(self):
        """
        Discover test files in the workspace (purely content-based).

        No hardcoded directories - scans everything and scores by content.

        Strategy:
        1. Files matching kernel name get priority boost
        2. All files scored by content keywords
        3. Rank by final confidence score
        """
        print("\n[2/4] Discovering tests (content-based)...")

        seen_paths = set()

        # File extensions to search
        extensions = [".py"]
        if self.config.include_cpp:
            extensions.extend([".cpp", ".cc", ".cu", ".hip", ".cxx"])

        # Get kernel name parts for matching.
        # Fall back to the kernel_file stem when _discover_kernels found nothing
        # (e.g. wrapper files that import @triton.jit kernels from another module).
        kernel_name = None
        kernel_parts = []
        if self.result.kernels:
            kernel_name = self.result.kernels[0].kernel_name
        elif self._kernel_file:
            kernel_name = self._kernel_file.stem
        if kernel_name:
            kernel_parts = [p for p in kernel_name.split("_") if len(p) > 2]

        # Get kernel file paths to exclude
        kernel_files = {k.file_path for k in self.result.kernels}

        # Scan ALL files, score by content
        for ext in extensions:
            for file_path in self.workspace.rglob(f"*{ext}"):
                if self._should_skip_file(file_path) or file_path in seen_paths:
                    continue

                # Skip kernel files themselves
                if file_path in kernel_files:
                    continue

                # Skip files that contain kernel definitions (have @triton.jit, etc.)
                if self._is_kernel_file(file_path):
                    continue

                # Analyze file content - must have minimum content confidence
                test_info = self._analyze_test_file(file_path)
                if not test_info:
                    continue

                # Only boost for kernel name match if content analysis shows it's a test
                # (confidence >= 0.3 means content has test-like patterns)
                fname_lower = file_path.name.lower()
                if kernel_name and kernel_name.lower() in fname_lower:
                    # Exact kernel name match - highest priority
                    test_info.confidence += 1.0
                elif kernel_parts:
                    # Partial match bonus - proportional to how many parts match
                    matches = sum(1 for p in kernel_parts if p.lower() in fname_lower)
                    if matches >= 2:
                        test_info.confidence += 0.3 * matches

                self.result.tests.append(test_info)
                seen_paths.add(file_path)

        # Sort by confidence
        self.result.tests.sort(key=lambda t: t.confidence, reverse=True)

        # Limit results
        if len(self.result.tests) > 10:
            print(f"      Found {len(self.result.tests)} potential test(s), showing top 10")
            self.result.tests = self.result.tests[:10]
        else:
            print(f"      Found {len(self.result.tests)} potential test(s)")

    def _analyze_test_file(self, file_path: Path) -> TestInfo | None:
        """
        Analyze a file to determine if it's a test (content-based).

        Supports:
        - Python: pytest, unittest, custom frameworks
        - C++: GTest, Catch2, custom test harnesses
        - LLM fallback for uncertain cases
        """
        try:
            content = file_path.read_text()
        except Exception:
            return None

        confidence = 0.0
        test_type = "script"
        is_cpp = file_path.suffix in [".cpp", ".cc", ".cu", ".hip", ".cxx"]

        # Select appropriate keywords based on file type
        if is_cpp and self.config.include_cpp:
            keywords = self.config.test_cpp_keywords
        else:
            keywords = self._test_keywords

        # Content-based scoring
        for pattern, score in keywords:
            if re.search(pattern, content, re.IGNORECASE if not is_cpp else 0):
                confidence += score

        # Filename bonus (lower priority than content)
        if "test" in file_path.name.lower():
            confidence += 0.1

        # NOTE: Kernel name matching is done in the main discovery loop, not here
        # This ensures we only boost files that already pass content-based detection

        # LLM fallback for uncertain cases (0.3-0.6 confidence)
        if 0.3 <= confidence <= 0.6 and self.use_llm:
            llm_result = self._llm_analyze_file(file_path, "test")
            if llm_result and llm_result.get("is_test"):
                confidence = max(confidence, llm_result.get("confidence", 0.7))
                if llm_result.get("command"):
                    # Use LLM's suggested command
                    return TestInfo(
                        file_path=file_path,
                        test_type="llm-detected",
                        command=llm_result["command"],
                        confidence=min(confidence, 1.0),
                    )

        # Must have minimum confidence to be considered a test
        if confidence < 0.3:
            return None

        # Determine test type and command for Python files
        if not is_cpp:
            if "import pytest" in content or "@pytest" in content:
                test_type = "pytest"
            elif "unittest" in content:
                test_type = "unittest"

            if test_type == "pytest":
                command = f"pytest {file_path} -v"
            elif test_type == "unittest":
                command = f"python -m unittest {file_path}"
            else:
                command = f"python {file_path}"
        else:
            # C++ test command generation
            if "gtest" in content.lower() or "TEST(" in content:
                test_type = "gtest"
            elif "catch" in content.lower() or "TEST_CASE" in content:
                test_type = "catch2"
            else:
                test_type = "cpp"

            # For C++, we typically need to build first
            # Look for a Makefile or CMakeLists.txt
            parent = file_path.parent
            if (parent / "Makefile").exists():
                command = f"make -C {parent} && ./{file_path.stem}"
            elif (parent / "CMakeLists.txt").exists():
                command = f"cd {parent} && cmake -B build && cmake --build build && ./build/{file_path.stem}"
            else:
                # Assume hipcc/nvcc compilation
                if file_path.suffix in [".cu", ".hip"]:
                    compiler = "hipcc" if file_path.suffix == ".hip" else "nvcc"
                    command = f"{compiler} {file_path} -o /tmp/{file_path.stem} && /tmp/{file_path.stem}"
                else:
                    command = f"g++ {file_path} -lgtest -lgtest_main -o /tmp/{file_path.stem} && /tmp/{file_path.stem}"

        return TestInfo(file_path=file_path, test_type=test_type, command=command, confidence=min(confidence, 1.0))

    def _discover_benchmarks(self):
        """
        Discover benchmark files in the workspace (purely content-based).

        No hardcoded directories - scans everything and scores by content.

        Strategy:
        1. Files matching kernel name get priority boost
        2. All files scored by content keywords (TFLOPS, latency, etc.)
        3. Rank by final confidence score
        """
        print("\n[3/4] Discovering benchmarks (content-based)...")

        seen_paths = set()

        # File extensions to search
        extensions = [".py"]
        if self.config.include_cpp:
            extensions.extend([".cpp", ".cc", ".cu", ".hip", ".cxx"])

        # Get kernel name parts for matching.
        # Fall back to the kernel_file stem when _discover_kernels found nothing
        # (e.g. wrapper files that import @triton.jit kernels from another module).
        kernel_name = None
        kernel_parts = []
        if self.result.kernels:
            kernel_name = self.result.kernels[0].kernel_name
        elif self._kernel_file:
            kernel_name = self._kernel_file.stem
        if kernel_name:
            kernel_parts = [p for p in kernel_name.split("_") if len(p) > 2]

        # Get kernel file paths to exclude
        kernel_files = {k.file_path for k in self.result.kernels}

        # Scan ALL files, score by content
        for ext in extensions:
            for file_path in self.workspace.rglob(f"*{ext}"):
                if self._should_skip_file(file_path) or file_path in seen_paths:
                    continue

                # Skip kernel files themselves
                if file_path in kernel_files:
                    continue

                # Skip files that contain kernel definitions
                if self._is_kernel_file(file_path):
                    continue

                # Analyze file content
                bench_info = self._analyze_bench_file(file_path)
                if not bench_info:
                    continue

                # Boost score for kernel name match (this is what matters most)
                # Allow confidence > 1.0 for ranking, display capped at 100%
                fname_lower = file_path.name.lower()
                if kernel_name and kernel_name.lower() in fname_lower:
                    # Exact kernel name match - highest priority
                    bench_info.confidence += 1.0
                elif kernel_parts:
                    # Partial match bonus - proportional to how many parts match
                    matches = sum(1 for p in kernel_parts if p.lower() in fname_lower)
                    if matches >= 2:
                        bench_info.confidence += 0.3 * matches

                self.result.benchmarks.append(bench_info)
                seen_paths.add(file_path)

        # Sort by confidence
        self.result.benchmarks.sort(key=lambda b: b.confidence, reverse=True)

        # Limit results
        if len(self.result.benchmarks) > 10:
            print(f"      Found {len(self.result.benchmarks)} potential benchmark(s), showing top 10")
            self.result.benchmarks = self.result.benchmarks[:10]
        else:
            print(f"      Found {len(self.result.benchmarks)} potential benchmark(s)")

    def _analyze_bench_file(self, file_path: Path) -> BenchmarkInfo | None:
        """
        Analyze a file to determine if it's a benchmark (content-based).

        Supports:
        - Python: triton.testing.do_bench, torch.cuda.Event, custom
        - C++: hipEventElapsedTime, cudaEventElapsedTime, std::chrono
        - LLM fallback for uncertain cases
        """
        try:
            content = file_path.read_text()
        except Exception:
            return None

        confidence = 0.0
        bench_type = "script"
        is_cpp = file_path.suffix in [".cpp", ".cc", ".cu", ".hip", ".cxx"]

        # Content-based scoring (same keywords work for both Python and C++)
        for pattern, score in self._bench_keywords:
            if re.search(pattern, content, re.IGNORECASE if not is_cpp else 0):
                confidence += score

        # Filename bonus (lower priority than content)
        if "bench" in file_path.name.lower() or "perf" in file_path.name.lower():
            confidence += 0.1

        # NOTE: Kernel name matching is done in the main discovery loop, not here
        # This ensures we only boost files that already pass content-based detection

        # LLM fallback for uncertain cases (0.3-0.6 confidence)
        if 0.3 <= confidence <= 0.6 and self.use_llm:
            llm_result = self._llm_analyze_file(file_path, "benchmark")
            if llm_result and llm_result.get("is_benchmark"):
                confidence = max(confidence, llm_result.get("confidence", 0.7))
                if llm_result.get("command"):
                    return BenchmarkInfo(
                        file_path=file_path,
                        bench_type="llm-detected",
                        command=llm_result["command"],
                        confidence=min(confidence, 1.0),
                    )

        # Must have minimum confidence to be considered a benchmark
        if confidence < 0.3:
            return None

        # Determine benchmark type and command
        if not is_cpp:
            if "pytest" in content and "benchmark" in content:
                bench_type = "pytest"
            elif "triton.testing.do_bench" in content:
                bench_type = "triton"
            command = f"python {file_path}"
        else:
            # C++ benchmark
            if "hipEvent" in content:
                bench_type = "hip"
            elif "cudaEvent" in content:
                bench_type = "cuda"
            else:
                bench_type = "cpp"

            # Look for build system
            parent = file_path.parent
            if (parent / "Makefile").exists():
                command = f"make -C {parent} && ./{file_path.stem}"
            else:
                compiler = "hipcc" if file_path.suffix == ".hip" or "hip" in content.lower() else "nvcc"
                command = f"{compiler} {file_path} -o /tmp/{file_path.stem} && /tmp/{file_path.stem}"

        return BenchmarkInfo(
            file_path=file_path, bench_type=bench_type, command=command, confidence=min(confidence, 1.0)
        )

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during discovery."""
        # Use skip_dirs from loaded config (defaults + per-project overrides)
        skip_dirs = set(self.config.skip_dirs) if self.config.skip_dirs else {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            "build", "dist", ".eggs", "site-packages", ".tox", ".pytest_cache",
        }

        # Add configured exclude directories (from per-project skip_dirs_extra)
        skip_dirs.update(self.config.exclude_dirs)

        for part in file_path.parts:
            if part in skip_dirs or part.endswith(".egg-info"):
                return True

        return False

    def _is_kernel_file(self, file_path: Path) -> bool:
        """Check if a file is a kernel definition file (should not be treated as test/bench)."""
        try:
            content = file_path.read_text()[:2000]
        except Exception:
            return False

        # Check for kernel definition patterns (loaded from config)
        kernel_patterns = self.config.kernel_patterns or [
            r"@triton\.jit", r"@triton\.autotune",
            r"__global__\s+void", r"tl\.load|tl\.store",
        ]
        for pattern in kernel_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _display_findings(self):
        """Display discovery findings to the user."""
        print("\n[4/4] Discovery complete!")
        print("\n" + "-" * 60)

        # Display kernels
        print("\n  KERNELS FOUND:")
        if self.result.kernels:
            for k in self.result.kernels:
                print(f"    - {k.kernel_name} ({k.kernel_type})")
                print(f"      File: {k.file_path}")
                if k.function_names:
                    print(f"      Functions: {', '.join(k.function_names)}")
        else:
            print("    (none found)")

        # Display tests
        print("\n  TESTS FOUND:")
        if self.result.tests:
            for t in self.result.tests:
                conf_pct = min(int(t.confidence * 100), 100)  # Cap at 100% for display
                print(f"    - {t.file_path.name} ({t.test_type}, {conf_pct}% confidence)")
                print(f"      Command: {t.command}")
        else:
            print("    (none found)")

        # Display benchmarks
        print("\n  BENCHMARKS FOUND:")
        if self.result.benchmarks:
            for b in self.result.benchmarks:
                conf_pct = min(int(b.confidence * 100), 100)  # Cap at 100% for display
                print(f"    - {b.file_path.name} ({b.bench_type}, {conf_pct}% confidence)")
                print(f"      Command: {b.command}")
        else:
            print("    (none found)")

        print("\n" + "-" * 60)

    def _prompt_user_confirmation(self):
        """Prompt user to confirm or modify discoveries."""
        if not self.result.tests and not self.result.benchmarks:
            print("\n  No tests or benchmarks found.")
            print("  Options:")
            print("    [c] Create tests (I'll help)")
            print("    [p] Provide test command manually")
            print("    [s] Search in different directory")
            print("    [q] Quit")
        else:
            print("\n  Is this correct?")
            print("    [y] Yes, proceed")
            print("    [e] Edit these paths")
            print("    [s] Search for more")
            print("    [m] Modify these tests")
            print("    [c] Create additional tests")

        # For now, just print the prompt - actual input handling
        # will be done by the agent or CLI
        print("\n  (Awaiting user input...)")

    def get_test_command(self) -> str | None:
        """Get the best test command from discovery."""
        if self.result.user_provided_test:
            return self.result.user_provided_test
        if self.result.tests:
            return self.result.tests[0].command
        return None

    def get_bench_command(self) -> str | None:
        """Get the best benchmark command from discovery."""
        if self.result.user_provided_bench:
            return self.result.user_provided_bench
        if self.result.benchmarks:
            return self.result.benchmarks[0].command
        return None

    def get_kernel_path(self) -> Path | None:
        """Get the primary kernel path."""
        if self.result.kernels:
            return self.result.kernels[0].file_path
        return None

    def to_context(self) -> dict:
        """Convert discovery result to context for agent prompt."""
        return {
            "workspace": str(self.result.workspace_path),
            "kernels": [
                {"name": k.kernel_name, "file": str(k.file_path), "type": k.kernel_type, "functions": k.function_names}
                for k in self.result.kernels
            ],
            "test_command": self.get_test_command(),
            "bench_command": self.get_bench_command(),
            "has_tests": len(self.result.tests) > 0,
            "has_benchmarks": len(self.result.benchmarks) > 0,
        }


# Convenience function
def discover(
    workspace: Path = None,
    kernel_path: Path = None,
    test_command: str = None,
    bench_command: str = None,
    interactive: bool = True,
    use_llm: bool = False,
) -> DiscoveryResult:
    """
    Run discovery pipeline.

    Patterns are auto-detected from the codebase - no configuration needed.

    Args:
        workspace: Workspace directory to search
        kernel_path: Explicit kernel file/directory
        test_command: User-provided test command
        bench_command: User-provided benchmark command
        interactive: Whether to prompt for confirmation
        use_llm: Whether to use LLM for uncertain cases

    Returns:
        DiscoveryResult with all discovered information
    """
    # If workspace is a file, treat it as kernel_path
    if workspace and Path(workspace).is_file():
        kernel_path = Path(workspace)
        workspace = kernel_path.parent

    pipeline = DiscoveryPipeline(workspace, use_llm=use_llm)
    return pipeline.run(
        kernel_path=kernel_path, test_command=test_command, bench_command=bench_command, interactive=interactive
    )


if __name__ == "__main__":
    # Test the discovery pipeline
    import sys

    if len(sys.argv) > 1:
        workspace = Path(sys.argv[1])
    else:
        workspace = Path.cwd()

    result = discover(workspace, interactive=True)

    print("\n\nDiscovery Result:")
    print(f"  Kernels: {len(result.kernels)}")
    print(f"  Tests: {len(result.tests)}")
    print(f"  Benchmarks: {len(result.benchmarks)}")
