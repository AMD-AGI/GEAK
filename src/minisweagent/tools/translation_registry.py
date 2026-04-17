"""Translation registry: data-driven lookup for source -> target language pairs.

Each ``TranslationPair`` describes everything needed to translate a kernel from
one language to another: detection heuristic, agent configs, KB files, harness
flags, environment setup, and output filename conventions.

Adding a new pair (e.g. Triton -> FlyDSL) requires only a new
``TranslationPair`` entry—zero changes to the pipeline code.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TranslationPair dataclass
# ---------------------------------------------------------------------------


@dataclass
class TranslationPair:
    """Describes a single source -> target translation."""

    source: str
    target: str
    detect_source: Callable[[Path], bool]
    config_name: str
    harness_config_name: str
    harness_candidate_flag: str
    candidate_filename_fn: Callable[[str], str]
    kb_base_files: list[str]
    kb_translation_files: list[str]
    kb_category_files: dict[str, str] = field(default_factory=dict)
    env_setup: Callable[[Path], dict[str, str]] = field(default=lambda: _noop_env_setup)
    max_attempts: int = 3
    perf_fail_threshold: float = 0.5
    perf_warn_threshold: float = 0.8
    supported: bool = True
    self_review: bool = False
    review_triggers_retry: bool = False
    review_retry_on_efficiency: bool = False


def _noop_env_setup(_repo_root: Path) -> dict[str, str]:
    return {}


# ---------------------------------------------------------------------------
# PyTorch source detection
# ---------------------------------------------------------------------------


def _detect_pytorch_module(kernel_path: Path) -> bool:
    """Return True if *kernel_path* contains a PyTorch nn.Module kernel.

    Heuristic: file must import torch and define a class that inherits from
    nn.Module, matching the KernelBench ``Model(nn.Module)`` pattern.
    """
    try:
        text = kernel_path.read_text(errors="replace")
    except OSError:
        return False
    has_torch = "import torch" in text
    has_module = bool(re.search(r"class\s+\w+\s*\(\s*(?:nn\.Module|torch\.nn\.Module)\s*\)", text))
    return has_torch and has_module


# ---------------------------------------------------------------------------
# Kernel category detection (for tiered KB loading)
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "gemm": [
        r"torch\.matmul",
        r"torch\.mm\b",
        r"torch\.bmm\b",
        r"@\s",
        r"F\.linear",
        r"nn\.Linear",
    ],
    "attention": [
        r"scaled_dot_product_attention",
        r"MultiheadAttention",
        r"multi_head_attention",
        r"flash_attn",
        r"\w\s+@\s+\w.*transpose",
    ],
    "reductions": [
        r"torch\.sum\b",
        r"torch\.mean\b",
        r"torch\.norm\b",
        r"torch\.softmax\b",
        r"F\.softmax",
        r"F\.layer_norm",
        r"nn\.LayerNorm",
        r"F\.normalize",
    ],
}


def detect_kernel_categories(source_path: Path) -> list[str]:
    """Detect kernel categories by pattern matching the source file."""
    try:
        text = source_path.read_text(errors="replace")
    except OSError:
        return []
    categories: list[str] = []
    for cat, patterns in _CATEGORY_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            categories.append(cat)
    if "attention" in categories and "reductions" not in categories:
        categories.append("reductions")
    return categories


# ---------------------------------------------------------------------------
# FlyDSL environment setup
# ---------------------------------------------------------------------------


def _flydsl_env_setup(repo_root: Path, flydsl_repo: Path | None = None) -> dict[str, str]:
    """Discover FlyDSL build artifacts and return env overrides.

    Scans for ``build-fly/python_packages`` and MLIR shared libs under
    *flydsl_repo* (if given), then *repo_root* and its parent, then
    common installation paths and ``FLYDSL_HOME`` env var.

    Returns PYTHONPATH and LD_LIBRARY_PATH additions suitable for
    ``run_harness(env_overrides=...)``.
    """
    overrides: dict[str, str] = {}
    search_roots: list[Path] = []
    if flydsl_repo:
        search_roots.append(flydsl_repo)
    search_roots.extend([repo_root, repo_root.parent])
    flydsl_home = os.environ.get("FLYDSL_HOME")
    if flydsl_home:
        search_roots.append(Path(flydsl_home))
    search_roots.append(Path("/workspace/FlyDSL"))

    for root in search_roots:
        fly_python = root / "build-fly" / "python_packages"
        if fly_python.is_dir():
            tests_dir = root / "tests"
            paths = [str(fly_python), str(root)]
            if tests_dir.is_dir():
                paths.append(str(tests_dir))
            existing = os.environ.get("PYTHONPATH", "")
            if existing:
                paths.append(existing)
            overrides["PYTHONPATH"] = ":".join(paths)

            mlir_lib = fly_python / "flydsl" / "_mlir" / "_mlir_libs"
            if mlir_lib.is_dir():
                existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
                overrides["LD_LIBRARY_PATH"] = f"{mlir_lib}:{existing_ld}" if existing_ld else str(mlir_lib)

            break

    return overrides


# ---------------------------------------------------------------------------
# KB content loading
# ---------------------------------------------------------------------------

_FLYDSL_REPO_DOCS = [
    "docs/kernel_authoring_guide.md",
    "docs/layout_system_guide.md",
    "docs/prebuilt_kernels_guide.md",
    "docs/cute_layout_algebra_guide.md",
]


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (``---`` delimited) from markdown content."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return content


def load_translation_kb(
    pair: TranslationPair,
    categories: list[str],
    flydsl_repo: Path | None = None,
) -> str:
    """Load KB content for translation agent prompt injection.

    Two content types are concatenated:

    1. **FlyDSL reference** (API, patterns, kernels):
       - Default: from ``knowledge-base/`` (our authored API reference)
       - With ``flydsl_repo``: from FlyDSL repo ``docs/`` directory

    2. **Translation content** (always from ``knowledge-base/``):
       - Translation guide (PyTorch op mapping, structural patterns, pitfalls)
       - Category-specific guides (reductions, GEMM, attention)
    """
    kb_root = Path(__file__).resolve().parents[3] / "knowledge-base"
    sections: list[str] = []

    if flydsl_repo:
        for doc_path in _FLYDSL_REPO_DOCS:
            full_path = flydsl_repo / doc_path
            if full_path.exists():
                sections.append(full_path.read_text())
    else:
        for f in pair.kb_base_files:
            path = kb_root / f
            if path.exists():
                sections.append(_strip_frontmatter(path.read_text()))
            else:
                logger.warning("KB base file not found: %s", path)

    for f in pair.kb_translation_files:
        path = kb_root / f
        if path.exists():
            sections.append(_strip_frontmatter(path.read_text()))
        else:
            logger.warning("KB translation file not found: %s", path)

    for cat in categories:
        if cat in pair.kb_category_files:
            path = kb_root / pair.kb_category_files[cat]
            if path.exists():
                sections.append(_strip_frontmatter(path.read_text()))

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Registry: built-in pairs
# ---------------------------------------------------------------------------

_PYTORCH_TO_FLYDSL = TranslationPair(
    source="pytorch",
    target="flydsl",
    detect_source=_detect_pytorch_module,
    config_name="mini_kernel_pytorch_to_flydsl",
    harness_config_name="mini_unit_test_agent_pytorch_translation",
    harness_candidate_flag="--flydsl-kernel",
    candidate_filename_fn=lambda stem: f"{stem}_flydsl.py",
    kb_base_files=["flydsl/flydsl_translation_api_reference.md"],
    kb_translation_files=["flydsl/flydsl_translation_guide.md"],
    kb_category_files={
        "gemm": "flydsl/flydsl_translation_gemm.md",
        "reductions": "flydsl/flydsl_translation_reductions.md",
        "attention": "flydsl/flydsl_translation_attention.md",
    },
    env_setup=_flydsl_env_setup,
    max_attempts=3,
)


class TranslationRegistry:
    """Registry of supported translation pairs."""

    def __init__(self) -> None:
        self._pairs: list[TranslationPair] = [_PYTORCH_TO_FLYDSL]

    def detect(
        self,
        kernel_path: Path,
        target_language: str | None = None,
    ) -> TranslationPair | None:
        """Find matching pair for *kernel_path*.

        If *target_language* is given, only pairs matching that target are
        considered.  Returns ``None`` if no pair matches.
        """
        for pair in self._pairs:
            if not pair.supported:
                continue
            if target_language and pair.target != target_language:
                continue
            if pair.detect_source(kernel_path):
                return pair
        return None

    def get_pair(self, source: str, target: str) -> TranslationPair | None:
        """Direct lookup by source/target names."""
        for pair in self._pairs:
            if pair.source == source and pair.target == target:
                return pair
        return None

    def register(self, pair: TranslationPair) -> None:
        """Register a new translation pair."""
        self._pairs.append(pair)


REGISTRY = TranslationRegistry()
