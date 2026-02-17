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

# Shared extension constants -- single source of truth
CPP_EXTENSIONS = frozenset((".cpp", ".cc", ".cu", ".hip", ".cxx"))
CPP_HEADER_EXTENSIONS = frozenset((".h", ".hpp"))
ALL_KERNEL_EXTENSIONS = frozenset((".py",)) | CPP_EXTENSIONS

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
    kernel_extensions: list[str] = field(
        default_factory=lambda: list(ALL_KERNEL_EXTENSIONS)
    )
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
class BuildInfo:
    """How to compile/build a kernel."""

    compiler: str | None = None  # "triton" (JIT), "hipcc", "nvcc", "cmake", None (precompiled)
    build_system: str | None = None  # "setup.py", "CMakeLists.txt", "Makefile", None
    build_dir: Path | None = None  # Where compiled artifacts go
    pybind_module: str | None = None  # e.g., "aiter._C" or "torch.ops.aiter"


@dataclass
class KernelInfo:
    """Information about a discovered kernel."""

    file_path: Path
    kernel_name: str
    kernel_type: str  # triton, hip, cuda, ck, asm
    kernel_language: str = "python"  # "python", "cpp", "asm"
    function_names: list[str] = field(default_factory=list)
    has_jit_decorator: bool = False
    has_autotune: bool = False
    inner_kernel_path: Path | None = None
    inner_kernel_language: str | None = None
    build_info: BuildInfo | None = None
    # Populated by the dependency graph builder (Fix 2)
    fusion_opportunities: list[str] = field(default_factory=list)


@dataclass
class KernelNode:
    """A single function/kernel in the dependency graph."""

    name: str
    file_path: Path
    language: str  # "python", "triton", "hip", "ck", "asm"
    node_type: str  # "wrapper", "jit_kernel", "device_func", "asm_module", "torch_op"
    line_range: tuple[int, int] | None = None


@dataclass
class FusionOpportunity:
    """A detected opportunity to fuse operations."""

    description: str  # Human-readable description
    involved_nodes: list[str] = field(default_factory=list)  # Names of kernels/ops
    languages: set[str] = field(default_factory=set)  # Languages involved
    fusion_type: str = ""  # "sequential_launch", "absorb_wrapper_op", "cross_language"
    estimated_benefit: str = "medium"  # "high", "medium", "low"


@dataclass
class KernelDependencyGraph:
    """Cross-language dependency graph for a kernel and its sub-kernels."""

    root_name: str  # Name of the Python wrapper entry point
    nodes: dict[str, KernelNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (caller, callee)
    sequential_launches: list[list[str]] = field(default_factory=list)
    wrapper_ops: list[str] = field(default_factory=list)  # e.g., ["dtype conversion", "reshape"]
    language_boundaries: list[tuple[str, str, str]] = field(default_factory=list)
    fusion_opportunities: list[FusionOpportunity] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for inclusion in agent task prompts."""
        lines = [f"Dependency graph for {self.root_name}:"]
        lines.append(f"  Nodes ({len(self.nodes)}):")
        for name, node in self.nodes.items():
            lines.append(f"    - {name} [{node.language}/{node.node_type}] in {node.file_path.name}")
        if self.edges:
            lines.append(f"  Call edges ({len(self.edges)}):")
            for caller, callee in self.edges:
                lines.append(f"    {caller} -> {callee}")
        if self.sequential_launches:
            lines.append(f"  Sequential kernel launches (potential fusion targets):")
            for group in self.sequential_launches:
                lines.append(f"    [{' -> '.join(group)}]")
        if self.wrapper_ops:
            lines.append(f"  Wrapper operations between launches:")
            for op in self.wrapper_ops:
                lines.append(f"    - {op}")
        if self.language_boundaries:
            lines.append(f"  Language boundaries:")
            for caller, callee, boundary in self.language_boundaries:
                lines.append(f"    {caller} -> {callee} ({boundary})")
        if self.fusion_opportunities:
            lines.append(f"  Fusion opportunities ({len(self.fusion_opportunities)}):")
            for opp in self.fusion_opportunities:
                lines.append(f"    - [{opp.estimated_benefit}] {opp.description}")
        return "\n".join(lines)


@dataclass
class TestPatterns:
    """Patterns extracted from a discovered test file.

    These help the UnitTestAgent (or main agent) create better test
    harnesses by reusing tolerances, input shapes, and import patterns
    from existing tests rather than inventing them from scratch.
    """

    tolerances: list[str] = field(default_factory=list)  # e.g. ["atol=1e-3", "rtol=1e-3"]
    input_shapes: list[str] = field(default_factory=list)  # e.g. ["(1024, 1024)", "(batch, seq_len, hidden)"]
    dtypes: list[str] = field(default_factory=list)  # e.g. ["torch.float16", "torch.bfloat16"]
    reference_impls: list[str] = field(default_factory=list)  # e.g. ["torch.nn.functional.softmax"]
    import_patterns: list[str] = field(default_factory=list)  # e.g. ["from aiter.ops import rope_fwd"]


@dataclass
class TestInfo:
    """Information about a discovered test."""

    file_path: Path
    test_type: str  # pytest, script, makefile
    command: str  # Command to run the test
    confidence: float  # 0-1, how confident we are this is the right test
    patterns: TestPatterns | None = None


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
    dependency_graphs: dict[str, KernelDependencyGraph] = field(default_factory=dict)
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
            kernel_extensions=kernel_cfg.get("extensions", list(ALL_KERNEL_EXTENSIONS)),
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

        # Scope the workspace based on what we were given
        self._kernel_file = None
        if kernel_path and kernel_path.is_file():
            self._kernel_file = kernel_path
            # Search for workspace root (look for common markers)
            self._expand_workspace_for_file(kernel_path)
        elif kernel_path and kernel_path.is_dir():
            # Directory (repository) mode: use the directory as workspace
            # if it looks like a project root, otherwise expand upward.
            self._expand_workspace_for_dir(kernel_path)

        # Step 0: Auto-detect patterns from the codebase
        self._auto_detect_patterns()

        # Step 1: Discover kernels
        self._discover_kernels(kernel_path)

        # Step 1b: Build dependency graphs and detect fusion opportunities
        self._dependency_graphs: dict[str, KernelDependencyGraph] = {}
        for kernel in self.result.kernels:
            dep_graph = self.build_dependency_graph(kernel)
            if dep_graph:
                self._dependency_graphs[kernel.kernel_name] = dep_graph
                if dep_graph.fusion_opportunities:
                    print(f"      {kernel.kernel_name}: {len(dep_graph.fusion_opportunities)} fusion opportunity(ies)")

        self.result.dependency_graphs = self._dependency_graphs

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

    def _expand_workspace_for_dir(self, dir_path: Path):
        """Set the workspace when *kernel_path* is a directory.

        The caller explicitly chose this directory, so we honour it as the
        workspace boundary.  This prevents test discovery from pulling in
        unrelated files from sibling directories (e.g. scanning the entire
        monorepo when only ``geak_eval/`` was requested).

        If the directory itself is a project root (``.git``, ``pyproject.toml``,
        etc.) we note that, but either way the workspace is the directory.
        """
        self.workspace = dir_path
        self.result.workspace_path = dir_path

        project_markers = self.config.project_markers or [
            "pyproject.toml", "setup.py", "setup.cfg", ".git",
            "op_tests", "tests", "Makefile", "CMakeLists.txt",
        ]
        for marker in project_markers:
            if (dir_path / marker).exists():
                print(f"      Directory is a project root: {dir_path}")
                return

        print(f"      Using specified directory as workspace: {dir_path}")

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
            # Search directory -- scan all kernel-relevant extensions
            kernel_exts = set(self.config.kernel_extensions or [".py"])
            for candidate in search_path.rglob("*"):
                if not candidate.is_file():
                    continue
                if candidate.suffix not in kernel_exts:
                    continue
                if self._should_skip_file(candidate):
                    continue

                kernel_info = self._analyze_kernel_file(candidate)
                if kernel_info:
                    self.result.kernels.append(kernel_info)

        print(f"      Found {len(self.result.kernels)} kernel(s)")

    # ------------------------------------------------------------------
    # Kernel language / build info mapping
    # ------------------------------------------------------------------

    _KERNEL_TYPE_TO_LANGUAGE = {
        "triton": "python",
        "hip": "cpp",
        "cuda": "cpp",
        "ck": "cpp",
        "asm": "asm",
    }

    _KERNEL_TYPE_TO_COMPILER = {
        "triton": "triton",
        "hip": "hipcc",
        "cuda": "nvcc",
        "ck": "hipcc",
        "asm": None,
    }

    def _analyze_kernel_file(self, file_path: Path) -> KernelInfo | None:
        """Analyze a file to determine if it contains kernels.

        Detects:
        - Triton: ``@triton.jit``, ``tl.load``/``tl.store``
        - HIP: ``__global__`` + ``__device__``/``hipLaunch``
        - CUDA: ``__global__`` + cuda references
        - Composable Kernel (CK): ``ck::`` namespace, ``#include "ck/..."``
        - Assembly (HSACO): ``hipModuleLoad``, ``hipModuleLaunchKernel``
        - Python wrapper files that import/call kernel functions from other modules
          (including cross-language via ``torch.ops.*`` and ``ctypes``)
        """
        try:
            content = file_path.read_text()
        except Exception:
            return None

        is_cpp = file_path.suffix in CPP_EXTENSIONS | CPP_HEADER_EXTENSIONS

        # --- Detect kernel type ---
        kernel_type = None
        has_jit = False
        has_autotune = False

        # Triton kernels (Python)
        if "@triton.jit" in content or "tl.load" in content:
            kernel_type = "triton"
            has_jit = "@triton.jit" in content
            has_autotune = "@triton.autotune" in content

        # Composable Kernel (CK) -- check BEFORE generic HIP to avoid misclassifying
        elif re.search(r'\bck::', content) or re.search(r'#include\s+[<"]ck/', content):
            kernel_type = "ck"

        # Assembly (HSACO) -- precompiled binaries loaded via HIP runtime
        elif re.search(r'hipModuleLoad|hipModuleLaunchKernel|\.hsaco\b', content):
            kernel_type = "asm"

        # HIP kernels
        elif "__global__" in content and ("__device__" in content or "hipLaunch" in content):
            kernel_type = "hip"

        # CUDA kernels
        elif "__global__" in content and "cuda" in content.lower():
            kernel_type = "cuda"

        # --- Detect Python wrapper files ---
        inner_kernel_path = None
        inner_kernel_language = None
        build_info = None

        if not kernel_type and not is_cpp:
            # Triton wrapper: imports triton and imports/calls _kernel functions
            has_triton_import = bool(re.search(r"import\s+triton", content))
            imports_kernel_funcs = bool(
                re.search(r"from\s+\S+\s+import\s+[^)]*_kernel", content, re.DOTALL)
            )
            calls_kernel_with_grid = bool(re.search(r"_kernel\w*\[", content))
            if has_triton_import and (imports_kernel_funcs or calls_kernel_with_grid):
                kernel_type = "triton"
                # Trace the import to find the inner kernel file
                inner_kernel_path = self._trace_triton_import(content, file_path)
                if inner_kernel_path:
                    inner_kernel_language = "python"

            # Cross-language wrapper: calls torch.ops.* (pybind11 -> C++/HIP/CK)
            if not kernel_type:
                torch_ops_calls = re.findall(r'torch\.ops\.(\w+)\.(\w+)', content)
                if torch_ops_calls:
                    kernel_type = "hip"  # Default for torch.ops wrappers; refine below
                    namespace, op_name = torch_ops_calls[0]
                    pybind_file = self._find_pybind_registration(op_name)
                    if pybind_file:
                        inner_kernel_path = pybind_file
                        # Determine inner language from pybind file content
                        inner_kernel_language, refined_type = self._detect_cpp_kernel_type(pybind_file)
                        if refined_type:
                            kernel_type = refined_type
                    build_info = BuildInfo(
                        compiler="hipcc",
                        pybind_module=f"torch.ops.{namespace}",
                    )

            # ctypes wrapper: loads .so directly
            if not kernel_type:
                ctypes_loads = re.findall(r'ctypes\.CDLL\(["\']([^"\']+)', content)
                if ctypes_loads:
                    kernel_type = "hip"
                    build_info = BuildInfo(compiler="hipcc")

            # HSACO wrapper: uses hipModuleLoad from Python (hip-python)
            if not kernel_type:
                if re.search(r'hipModuleLoadData|HsacoLauncher', content):
                    kernel_type = "asm"

        if not kernel_type:
            return None

        # --- Determine language and build info ---
        kernel_language = self._KERNEL_TYPE_TO_LANGUAGE.get(kernel_type, "python")
        # For Python wrappers around non-Python kernels, the wrapper itself is Python
        if not is_cpp and kernel_type in ("hip", "cuda", "ck", "asm"):
            kernel_language = "python"

        if not build_info:
            compiler = self._KERNEL_TYPE_TO_COMPILER.get(kernel_type)
            build_sys = None
            build_dir = None
            if is_cpp:
                kernel_language = self._KERNEL_TYPE_TO_LANGUAGE.get(kernel_type, "cpp")
                # Try to find build system
                parent = file_path.parent
                if (parent / "CMakeLists.txt").exists():
                    build_sys = "CMakeLists.txt"
                    build_dir = parent / "build"
                elif (parent / "Makefile").exists():
                    build_sys = "Makefile"
            build_info = BuildInfo(compiler=compiler, build_system=build_sys, build_dir=build_dir)

        # --- Extract function names ---
        function_names = []

        if is_cpp:
            # C++: extract __global__ function names
            for match in re.finditer(r'__global__\s+void\s+(\w+)', content):
                function_names.append(match.group(1))
            # CK: extract template instantiation names
            if kernel_type == "ck":
                for match in re.finditer(r'using\s+(\w+)\s*=\s*ck::', content):
                    function_names.append(match.group(1))
        else:
            # Python: @triton.jit decorated functions
            jit_pattern = r"@triton\.jit\s*\n\s*def\s+(\w+)"
            for match in re.finditer(jit_pattern, content):
                function_names.append(match.group(1))

            # Wrapper functions (configurable via discovery.toml)
            wrapper_fns = self.config.wrapper_functions or ["forward", "main"]
            wrapper_alt = "|".join(re.escape(fn) for fn in wrapper_fns)
            wrapper_pattern = rf"def\s+({wrapper_alt})\s*\("
            for match in re.finditer(wrapper_pattern, content):
                if match.group(1) not in function_names:
                    function_names.append(match.group(1))

            # For wrapper files with no @triton.jit, also extract public API functions
            if not has_jit:
                public_fn_pattern = r"^def\s+(\w+)\s*\("
                for match in re.finditer(public_fn_pattern, content, re.MULTILINE):
                    fname = match.group(1)
                    if not fname.startswith("_") and fname not in function_names:
                        function_names.append(fname)

        kernel_name = file_path.stem
        # When the filename is generic (e.g. "kernel.py", "main.py"), the
        # parent directory often carries the real identity (e.g. "rope/kernel.py"
        # → kernel_name = "rope").  Use the parent dir name instead so that
        # test-name matching can work properly.
        _GENERIC_STEMS = {"kernel", "main", "module", "op", "impl"}
        if kernel_name.lower() in _GENERIC_STEMS and file_path.parent.name:
            kernel_name = file_path.parent.name

        return KernelInfo(
            file_path=file_path,
            kernel_name=kernel_name,
            kernel_type=kernel_type,
            kernel_language=kernel_language,
            function_names=function_names,
            has_jit_decorator=has_jit,
            has_autotune=has_autotune,
            inner_kernel_path=inner_kernel_path,
            inner_kernel_language=inner_kernel_language,
            build_info=build_info,
        )

    # ------------------------------------------------------------------
    # Cross-language import tracing helpers
    # ------------------------------------------------------------------

    def _trace_triton_import(self, content: str, wrapper_file: Path) -> Path | None:
        """Trace a Triton wrapper's import to resolve the inner kernel file.

        Given wrapper content like:
            from aiter.ops.triton._triton_kernels.rope.rope import _rope_kernel_sbhd_fwd
        resolves the module path to an actual file on disk.
        """
        import_match = re.search(
            r"from\s+([\w.]+)\s+import\s+[^)]*_kernel", content, re.DOTALL
        )
        if not import_match:
            return None

        module_path = import_match.group(1)
        return self._resolve_module_to_file(module_path, wrapper_file)

    def _resolve_module_to_file(self, module_path: str, reference_file: Path) -> Path | None:
        """Resolve a Python dotted module path to a file on disk.

        Tries multiple strategies:
        1. Convert dots to path separators relative to the workspace root
        2. Try relative to the reference file's package hierarchy
        3. Walk up from the reference file looking for matching directories
        """
        parts = module_path.split(".")
        relative = Path(*parts).with_suffix(".py")

        # Strategy 1: relative to workspace
        candidate = self.workspace / relative
        if candidate.exists():
            return candidate

        # Strategy 2: search within the workspace for the leaf portion
        # e.g., for "aiter.ops.triton._triton_kernels.rope.rope", try finding
        # _triton_kernels/rope/rope.py anywhere under workspace
        for depth in range(1, len(parts)):
            sub_parts = parts[depth:]
            sub_relative = Path(*sub_parts).with_suffix(".py")
            for match in self.workspace.rglob(str(sub_relative)):
                return match

        # Strategy 3: walk up from the reference file
        current = reference_file.parent
        for _ in range(10):
            candidate = current / relative
            if candidate.exists():
                return candidate
            parent = current.parent
            if parent == current:
                break
            current = parent

        return None

    def _find_pybind_registration(self, op_name: str) -> Path | None:
        """Find the C++ file that registers a given torch.ops operation via pybind11.

        Searches for ``m.def("op_name", ...)`` patterns in .cu/.cpp files.
        """
        pybind_pattern = re.compile(
            rf'm\.def\(\s*"{re.escape(op_name)}"', re.IGNORECASE
        )
        # Search common pybind directories first
        search_dirs = ["csrc/pybind", "csrc", "src"]
        for search_dir in search_dirs:
            candidate_dir = self.workspace / search_dir
            if not candidate_dir.is_dir():
                continue
            for f in candidate_dir.rglob("*"):
                if f.suffix not in CPP_EXTENSIONS:
                    continue
                try:
                    text = f.read_text()[:10000]
                    if pybind_pattern.search(text):
                        return f
                except Exception:
                    continue
        return None

    def _detect_cpp_kernel_type(self, cpp_file: Path) -> tuple[str, str | None]:
        """Detect the kernel language and refined type from a C++ file.

        Returns (language, kernel_type) where kernel_type may refine the
        initial guess (e.g., "hip" -> "ck" if file uses Composable Kernel).
        """
        try:
            text = cpp_file.read_text()[:5000]
        except Exception:
            return ("cpp", None)

        if re.search(r'\bck::', text) or re.search(r'#include\s+[<"]ck/', text):
            return ("cpp", "ck")
        if re.search(r'hipModuleLoad|hipModuleLaunchKernel|\.hsaco\b', text):
            return ("asm", "asm")
        if "__global__" in text:
            return ("cpp", "hip")
        return ("cpp", None)

    # ------------------------------------------------------------------
    # Dependency graph and fusion detection
    # ------------------------------------------------------------------

    def build_dependency_graph(self, kernel: KernelInfo) -> KernelDependencyGraph | None:
        """Build a cross-language dependency graph for a kernel.

        Given a KernelInfo (typically a wrapper), traces the call chain from
        the Python entry point down to the lowest-level kernel functions,
        annotating each node with its language. Then detects fusion opportunities.

        Call this after ``_discover_kernels`` has populated ``self.result.kernels``.
        """
        try:
            content = kernel.file_path.read_text()
        except Exception:
            return None

        graph = KernelDependencyGraph(root_name=kernel.kernel_name)
        is_cpp = kernel.file_path.suffix in CPP_EXTENSIONS

        if is_cpp:
            self._build_graph_cpp(graph, kernel, content)
        else:
            self._build_graph_python(graph, kernel, content)

        # Detect fusion opportunities from the completed graph
        graph.fusion_opportunities = self._detect_fusion_opportunities(graph)

        # Store fusion descriptions on the KernelInfo for easy access
        kernel.fusion_opportunities = [opp.description for opp in graph.fusion_opportunities]

        return graph

    def _build_graph_python(
        self, graph: KernelDependencyGraph, kernel: KernelInfo, content: str
    ) -> None:
        """Build dependency graph for a Python wrapper / Triton kernel."""
        # Add the wrapper as the root node
        wrapper_lang = "python"
        wrapper_type = "wrapper"
        graph.nodes[kernel.kernel_name] = KernelNode(
            name=kernel.kernel_name,
            file_path=kernel.file_path,
            language=wrapper_lang,
            node_type=wrapper_type,
        )

        # --- Parse sequential kernel launches in wrapper functions ---
        # Match patterns like: _kernel_name[grid](...) or _kernel_name.run(...)
        launch_pattern = re.compile(r'(\w+_kernel\w*)\s*\[')
        torch_ops_pattern = re.compile(r'torch\.ops\.(\w+)\.(\w+)\s*\(')

        # Track sequential operations in each function body
        current_launches: list[str] = []
        wrapper_ops: list[str] = []

        for line in content.splitlines():
            stripped = line.strip()

            # Detect kernel launches
            launch_match = launch_pattern.search(stripped)
            if launch_match:
                kname = launch_match.group(1)
                current_launches.append(kname)
                if kname not in graph.nodes:
                    graph.nodes[kname] = KernelNode(
                        name=kname,
                        file_path=kernel.inner_kernel_path or kernel.file_path,
                        language="triton",
                        node_type="jit_kernel",
                    )
                graph.edges.append((kernel.kernel_name, kname))
                continue

            # Detect torch.ops calls (cross-language)
            ops_match = torch_ops_pattern.search(stripped)
            if ops_match:
                ns, op = ops_match.group(1), ops_match.group(2)
                full_name = f"torch.ops.{ns}.{op}"
                current_launches.append(full_name)
                if full_name not in graph.nodes:
                    inner_file = kernel.inner_kernel_path or kernel.file_path
                    graph.nodes[full_name] = KernelNode(
                        name=full_name,
                        file_path=inner_file,
                        language="hip",
                        node_type="torch_op",
                    )
                graph.edges.append((kernel.kernel_name, full_name))
                graph.language_boundaries.append(
                    (kernel.kernel_name, full_name, "python->hip_pybind")
                )
                continue

            # Detect wrapper-level torch operations that could be fused
            for op_name, op_label in [
                (r'\.to\(', "dtype conversion"),
                (r'\.reshape\(', "reshape"),
                (r'\.permute\(', "permute"),
                (r'\.contiguous\(', "contiguous"),
                (r'\.view\(', "view/reshape"),
                (r'torch\.empty', "tensor allocation"),
                (r'\.transpose\(', "transpose"),
            ]:
                if re.search(op_name, stripped):
                    wrapper_ops.append(op_label)

        if len(current_launches) >= 2:
            graph.sequential_launches.append(current_launches)
        graph.wrapper_ops = list(dict.fromkeys(wrapper_ops))  # deduplicate, preserve order

        # --- Parse inner kernel file for sub-kernel calls ---
        if kernel.inner_kernel_path and kernel.inner_kernel_path.exists():
            try:
                inner_content = kernel.inner_kernel_path.read_text()
            except Exception:
                inner_content = ""

            if inner_content:
                # Find all @triton.jit functions and their internal calls
                jit_fns: dict[str, list[str]] = {}
                current_fn = None

                for line in inner_content.splitlines():
                    jit_match = re.match(r'\s*@triton\.jit', line)
                    if jit_match:
                        # Next def line is the function
                        current_fn = "__pending__"
                        continue

                    if current_fn == "__pending__":
                        fn_match = re.match(r'\s*def\s+(\w+)\s*\(', line)
                        if fn_match:
                            current_fn = fn_match.group(1)
                            jit_fns[current_fn] = []
                            if current_fn not in graph.nodes:
                                graph.nodes[current_fn] = KernelNode(
                                    name=current_fn,
                                    file_path=kernel.inner_kernel_path,
                                    language="triton",
                                    node_type="jit_kernel",
                                )
                        else:
                            current_fn = None
                        continue

                    if current_fn and current_fn != "__pending__":
                        # Look for calls to other known jit functions
                        for called_fn in jit_fns:
                            if called_fn != current_fn and re.search(
                                rf'\b{re.escape(called_fn)}\s*\(', line
                            ):
                                if called_fn not in jit_fns[current_fn]:
                                    jit_fns[current_fn].append(called_fn)

                    # Detect end of function (next def or decorator at same/lower indentation)
                    if current_fn and current_fn != "__pending__":
                        if re.match(r'(def |class |@)', line) and not line.startswith(' '):
                            current_fn = None

                # Add edges for internal calls
                for fn, calls in jit_fns.items():
                    for called in calls:
                        graph.edges.append((fn, called))

                # Add language boundary for wrapper -> inner kernel
                for launch in current_launches:
                    if launch in jit_fns:
                        graph.language_boundaries.append(
                            (kernel.kernel_name, launch, "python->triton")
                        )

    def _build_graph_cpp(
        self, graph: KernelDependencyGraph, kernel: KernelInfo, content: str
    ) -> None:
        """Build dependency graph for a C++/HIP/CK kernel file."""
        graph.nodes[kernel.kernel_name] = KernelNode(
            name=kernel.kernel_name,
            file_path=kernel.file_path,
            language=kernel.kernel_language,
            node_type="jit_kernel" if kernel.kernel_type == "triton" else "device_func",
        )

        # Extract __global__ functions
        for match in re.finditer(r'__global__\s+void\s+(\w+)', content):
            fname = match.group(1)
            if fname not in graph.nodes:
                graph.nodes[fname] = KernelNode(
                    name=fname,
                    file_path=kernel.file_path,
                    language="hip",
                    node_type="device_func",
                )
                graph.edges.append((kernel.kernel_name, fname))

        # Extract CK template instantiations
        if kernel.kernel_type == "ck":
            for match in re.finditer(r'using\s+(\w+)\s*=\s*ck::', content):
                tname = match.group(1)
                if tname not in graph.nodes:
                    graph.nodes[tname] = KernelNode(
                        name=tname,
                        file_path=kernel.file_path,
                        language="ck",
                        node_type="device_func",
                    )
                    graph.edges.append((kernel.kernel_name, tname))

    def _detect_fusion_opportunities(
        self, graph: KernelDependencyGraph
    ) -> list[FusionOpportunity]:
        """Detect fusion opportunities from the dependency graph.

        Patterns detected:
        1. Sequential same-language kernel launches in the wrapper
        2. Wrapper-level torch operations that could be absorbed into adjacent kernels
        3. Cross-language boundaries where rewriting in one language could help
        """
        opportunities: list[FusionOpportunity] = []

        # 1. Sequential same-language launches
        for launch_group in graph.sequential_launches:
            if len(launch_group) < 2:
                continue
            langs = set()
            for n in launch_group:
                if n in graph.nodes:
                    langs.add(graph.nodes[n].language)
            if len(langs) == 1 and "asm" not in langs:
                lang = next(iter(langs))
                opportunities.append(FusionOpportunity(
                    description=(
                        f"Fuse {len(launch_group)} sequential {lang} kernel launches "
                        f"({', '.join(launch_group)}) into a single kernel to eliminate "
                        f"intermediate memory round-trips and launch overhead"
                    ),
                    involved_nodes=list(launch_group),
                    languages=langs,
                    fusion_type="sequential_launch",
                    estimated_benefit="high",
                ))
            elif len(langs) > 1 and "asm" not in langs:
                opportunities.append(FusionOpportunity(
                    description=(
                        f"Cross-language sequential launches ({', '.join(launch_group)}): "
                        f"languages {langs}. Rewriting in a single language could "
                        f"eliminate intermediate memory transfers"
                    ),
                    involved_nodes=list(launch_group),
                    languages=langs,
                    fusion_type="cross_language",
                    estimated_benefit="medium",
                ))

        # 2. Wrapper operations absorbable into adjacent kernel
        fusible_ops = {"dtype conversion", "reshape", "permute", "contiguous",
                       "view/reshape", "transpose"}
        for op in graph.wrapper_ops:
            if op in fusible_ops:
                # Find the nearest kernel launch this op sits between
                adjacent_kernels = [
                    n for n in graph.nodes.values()
                    if n.node_type in ("jit_kernel", "device_func", "torch_op")
                    and n.language != "asm"
                ]
                if adjacent_kernels:
                    adj = adjacent_kernels[0]
                    opportunities.append(FusionOpportunity(
                        description=(
                            f"Absorb wrapper-level '{op}' operation into "
                            f"kernel '{adj.name}' to avoid an extra memory pass"
                        ),
                        involved_nodes=[adj.name],
                        languages={adj.language, "python"},
                        fusion_type="absorb_wrapper_op",
                        estimated_benefit="medium",
                    ))

        return opportunities

    def _discover_scored_files(self, analyze_fn, label: str) -> list:
        """Shared scanner for test and benchmark discovery.

        Scans all files in the workspace, scores by content using *analyze_fn*,
        boosts by kernel-name match, and returns a sorted list.
        """
        seen_paths: set[Path] = set()

        extensions = [".py"]
        if self.config.include_cpp:
            extensions.extend(sorted(CPP_EXTENSIONS))

        kernel_name = None
        kernel_parts: list[str] = []
        if self.result.kernels:
            kernel_name = self.result.kernels[0].kernel_name
        elif self._kernel_file:
            kernel_name = self._kernel_file.stem
        if kernel_name:
            kernel_parts = [p for p in kernel_name.split("_") if len(p) > 2]

        kernel_files = {k.file_path for k in self.result.kernels}
        results: list = []

        for ext in extensions:
            for file_path in self.workspace.rglob(f"*{ext}"):
                if self._should_skip_file(file_path) or file_path in seen_paths:
                    continue
                if file_path in kernel_files:
                    continue
                if self._is_kernel_file(file_path):
                    continue

                info = analyze_fn(file_path)
                if not info:
                    continue

                fname_lower = file_path.name.lower()
                if kernel_name and kernel_name.lower() in fname_lower:
                    info.confidence += 1.0
                elif kernel_parts:
                    matches = sum(1 for p in kernel_parts if p.lower() in fname_lower)
                    if matches >= 2:
                        info.confidence += 0.3 * matches

                results.append(info)
                seen_paths.add(file_path)

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def _discover_tests(self):
        """
        Discover test files in the workspace (purely content-based).

        No hardcoded directories - scans everything and scores by content.
        """
        print("\n[2/4] Discovering tests (content-based)...")

        seen_paths = set()

        self.result.tests = self._discover_scored_files(
            analyze_fn=self._analyze_test_file,
            label="test",
        )
        if len(self.result.tests) > 10:
            print(f"      Found {len(self.result.tests)} potential test(s), showing top 10")
            self.result.tests = self.result.tests[:10]
        else:
            print(f"      Found {len(self.result.tests)} potential test(s)")

        # Extract patterns from top-confidence test files
        self._extract_test_patterns()

    def _extract_test_patterns(self):
        """Extract reusable patterns from high-confidence test files.

        Reads the top test files and pulls out tolerances, input shapes,
        dtypes, reference implementations, and import patterns so the
        UnitTestAgent can reuse them instead of inventing from scratch.
        """
        for test in self.result.tests[:3]:  # Top 3 by confidence
            if test.confidence < 0.5:
                continue
            try:
                content = test.file_path.read_text()
            except Exception:
                continue

            patterns = TestPatterns()

            # Tolerances: atol=X, rtol=X, assert_close(..., atol=, rtol=)
            for m in re.finditer(r'[ar]tol\s*=\s*([0-9eE.\-+]+)', content):
                tok = f"{'atol' if 'atol' in content[m.start()-4:m.start()] else 'rtol'}={m.group(1)}"
                if tok not in patterns.tolerances:
                    patterns.tolerances.append(tok)

            # Input shapes: tuples of ints like (1024, 1024) or named like (batch, seq_len)
            for m in re.finditer(r'\(\s*\d+(?:\s*,\s*\d+)+\s*\)', content):
                shape = m.group(0)
                if shape not in patterns.input_shapes and len(shape) < 60:
                    patterns.input_shapes.append(shape)

            # Dtypes: torch.float16, torch.bfloat16, etc.
            for m in re.finditer(r'torch\.(float16|float32|float64|bfloat16|int32|int64|half|float|double)', content):
                dtype = f"torch.{m.group(1)}"
                if dtype not in patterns.dtypes:
                    patterns.dtypes.append(dtype)

            # Reference implementations: torch.nn.functional.X, torch.X
            for m in re.finditer(r'torch\.nn\.functional\.(\w+)', content):
                ref = f"torch.nn.functional.{m.group(1)}"
                if ref not in patterns.reference_impls:
                    patterns.reference_impls.append(ref)

            # Import patterns: from X import Y (kernel-related)
            for m in re.finditer(r'^(from\s+\S+\s+import\s+.+)$', content, re.MULTILINE):
                imp = m.group(1).strip()
                if len(imp) < 120 and imp not in patterns.import_patterns:
                    patterns.import_patterns.append(imp)

            # Only attach if we found something useful
            if any([patterns.tolerances, patterns.input_shapes, patterns.dtypes,
                     patterns.reference_impls, patterns.import_patterns]):
                test.patterns = patterns

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
        is_cpp = file_path.suffix in CPP_EXTENSIONS

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
        """
        print("\n[3/4] Discovering benchmarks (content-based)...")

        self.result.benchmarks = self._discover_scored_files(
            analyze_fn=self._analyze_bench_file,
            label="benchmark",
        )
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
        is_cpp = file_path.suffix in CPP_EXTENSIONS

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

        # Check for kernel definition patterns (loaded from config, includes CK/ASM)
        kernel_patterns = self.config.kernel_patterns or [
            r"@triton\.jit", r"@triton\.autotune",
            r"__global__\s+void", r"tl\.load|tl\.store",
            r"\bck::", r'#include\s+[<"]ck/',
            r"hipModuleLoad|hipModuleLaunchKernel",
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
                lang_label = f"{k.kernel_type}/{k.kernel_language}"
                print(f"    - {k.kernel_name} ({lang_label})")
                print(f"      File: {k.file_path}")
                if k.function_names:
                    print(f"      Functions: {', '.join(k.function_names)}")
                if k.inner_kernel_path:
                    inner_lang = k.inner_kernel_language or "unknown"
                    print(f"      Inner kernel: {k.inner_kernel_path} ({inner_lang})")
                if k.build_info and k.build_info.compiler:
                    print(f"      Build: {k.build_info.compiler}"
                          + (f" ({k.build_info.build_system})" if k.build_info.build_system else ""))
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

    # When kernel_path is a directory and no explicit workspace was given,
    # let the pipeline determine the workspace from the directory itself
    # rather than defaulting to cwd.
    if kernel_path and Path(kernel_path).is_dir() and workspace is None:
        workspace = kernel_path

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
