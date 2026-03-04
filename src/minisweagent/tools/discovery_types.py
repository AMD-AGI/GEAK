# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Shared dataclasses and constants for kernel/test/benchmark discovery.

These types are used across the GEAK pipeline (orchestrator, task generator,
task planner, preprocessor, etc.).  The actual discovery logic lives in
``automated_test_discovery``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Shared extension constants -- single source of truth
CPP_EXTENSIONS = frozenset((".cpp", ".cc", ".cu", ".hip", ".cxx"))
CPP_HEADER_EXTENSIONS = frozenset((".h", ".hpp"))
ALL_KERNEL_EXTENSIONS = frozenset((".py",)) | CPP_EXTENSIONS


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

    description: str
    involved_nodes: list[str] = field(default_factory=list)
    languages: set[str] = field(default_factory=set)
    fusion_type: str = ""  # "sequential_launch", "absorb_wrapper_op", "cross_language"
    estimated_benefit: str = "medium"  # "high", "medium", "low"


@dataclass
class KernelDependencyGraph:
    """Cross-language dependency graph for a kernel and its sub-kernels."""

    root_name: str
    nodes: dict[str, KernelNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    sequential_launches: list[list[str]] = field(default_factory=list)
    wrapper_ops: list[str] = field(default_factory=list)
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
            lines.append("  Sequential kernel launches (potential fusion targets):")
            for group in self.sequential_launches:
                lines.append(f"    [{' -> '.join(group)}]")
        if self.wrapper_ops:
            lines.append("  Wrapper operations between launches:")
            for op in self.wrapper_ops:
                lines.append(f"    - {op}")
        if self.language_boundaries:
            lines.append("  Language boundaries:")
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

    tolerances: list[str] = field(default_factory=list)
    input_shapes: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    reference_impls: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)


def _parse_shape_size(shape_str: str) -> int | None:
    """Parse a shape string like '(1024, 1024)' and return the product of its dimensions."""
    nums = re.findall(r"\d+", shape_str)
    if not nums:
        return None
    size = 1
    for n in nums:
        size *= int(n)
    return size


def select_shapes_uniform(shapes: list[str], count: int) -> list[str]:
    """Select *count* shapes uniformly spread from smallest to largest.

    Deduplicates, sorts by total element count (product of dimensions),
    then picks evenly-spaced indices so the result spans the full
    small-to-large range.  Returns up to *count* shapes (fewer if the
    input list is shorter).
    """
    seen: set[str] = set()
    sized: list[tuple[int, str]] = []
    for s in shapes:
        if s in seen:
            continue
        seen.add(s)
        sz = _parse_shape_size(s)
        if sz is not None:
            sized.append((sz, s))

    if not sized:
        return []

    sized.sort(key=lambda t: t[0])

    if count <= 0:
        return []
    if count == 1:
        return [sized[len(sized) // 2][1]]

    n = len(sized)
    if n <= count:
        return [s for _, s in sized]

    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    seen_idx: set[int] = set()
    unique_indices: list[int] = []
    for idx in indices:
        if idx not in seen_idx:
            seen_idx.add(idx)
            unique_indices.append(idx)

    return [sized[i][1] for i in unique_indices]


@dataclass
class TestInfo:
    """Information about a discovered test."""

    file_path: Path
    test_type: str  # pytest, script, makefile
    command: str
    confidence: float  # 0-1
    patterns: TestPatterns | None = None


@dataclass
class BenchmarkInfo:
    """Information about a discovered benchmark."""

    file_path: Path
    bench_type: str  # pytest, script, custom
    command: str
    confidence: float


def _infer_kernel_language(kernel_path: Path, kernel_type: str) -> str:
    """Derive ``kernel_language`` from file extension and discovery ``type``."""
    if kernel_type == "asm":
        return "asm"
    if kernel_path.suffix == ".py":
        return "python"
    return "cpp"


@dataclass
class DiscoveryResult:
    """Result of the discovery pipeline."""

    kernels: list[KernelInfo] = field(default_factory=list)
    tests: list[TestInfo] = field(default_factory=list)
    benchmarks: list[BenchmarkInfo] = field(default_factory=list)
    dependency_graphs: dict[str, KernelDependencyGraph] = field(default_factory=dict)
    workspace_path: Path | None = None
    needs_user_confirmation: bool = True
    user_provided_test: str | None = None
    user_provided_bench: str | None = None

    @classmethod
    def from_dict(cls, disc_dict: dict, kernel_path: str | Path) -> DiscoveryResult:
        """Build a ``DiscoveryResult`` from a raw discovery JSON dict.

        This is the single canonical conversion path -- all CLI entry
        points should use this instead of inline dict unpacking.
        """
        kp = Path(kernel_path)
        kernel_info = disc_dict.get("kernel") or {}
        kernels: list[KernelInfo] = []
        if kernel_info.get("file"):
            ktype = kernel_info.get("type", "unknown")
            klang = _infer_kernel_language(kp, ktype)

            _build_info: BuildInfo | None = None
            if klang == "cpp":
                _repo = kp.parent
                while _repo != _repo.parent:
                    if (_repo / "setup.py").exists():
                        _build_info = BuildInfo(compiler="hipcc", build_system="setup.py", build_dir=_repo)
                        break
                    if (_repo / "CMakeLists.txt").exists():
                        _build_info = BuildInfo(compiler="hipcc", build_system="CMakeLists.txt", build_dir=_repo)
                        break
                    if (_repo / "Makefile").exists():
                        _build_info = BuildInfo(compiler="hipcc", build_system="Makefile", build_dir=_repo)
                        break
                    _repo = _repo.parent

            kernels.append(
                KernelInfo(
                    file_path=Path(kernel_info["file"]),
                    kernel_name=kernel_info.get("name", kp.stem),
                    kernel_type=ktype,
                    kernel_language=klang,
                    function_names=kernel_info.get("functions", []),
                    build_info=_build_info,
                )
            )
        tests = [
            TestInfo(
                file_path=Path(t["file"]),
                test_type=t.get("type", "script"),
                command=t.get("command", ""),
                confidence=t.get("confidence", 0.5),
            )
            for t in (disc_dict.get("tests") or [])
        ]
        benchmarks = [
            BenchmarkInfo(
                file_path=Path(b["file"]),
                bench_type=b.get("type", "script"),
                command=b.get("command", ""),
                confidence=b.get("confidence", 0.5),
            )
            for b in (disc_dict.get("benchmarks") or [])
        ]
        return cls(
            kernels=kernels,
            tests=tests,
            benchmarks=benchmarks,
            workspace_path=Path(disc_dict.get("workspace", kp.parent)),
        )
