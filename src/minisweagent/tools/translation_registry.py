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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    max_rounds: int = 10
    supported: bool = True


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
        r"F\.scaled_dot_product_attention",
        r"MultiheadAttention",
        r"multi_head_attention",
        r"flash_attn",
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


def _flydsl_env_setup(repo_root: Path) -> dict[str, str]:
    """Discover FlyDSL build artifacts and return env overrides.

    Scans for ``build-fly/python_packages`` and MLIR shared libs under
    *repo_root* and its parent.  Returns PYTHONPATH and LD_LIBRARY_PATH
    additions suitable for ``run_harness(env_overrides=...)``.
    """
    overrides: dict[str, str] = {}
    search_roots = [repo_root, repo_root.parent]

    for root in search_roots:
        fly_python = root / "build-fly" / "python_packages"
        if fly_python.is_dir():
            existing = os.environ.get("PYTHONPATH", "")
            overrides["PYTHONPATH"] = f"{fly_python}:{existing}" if existing else str(fly_python)

        mlir_lib = root / "build-fly" / "lib"
        if mlir_lib.is_dir():
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            overrides["LD_LIBRARY_PATH"] = f"{mlir_lib}:{existing}" if existing else str(mlir_lib)

        if overrides:
            break

    return overrides


# ---------------------------------------------------------------------------
# GPU auto-detection via rocminfo
# ---------------------------------------------------------------------------

_GFX_TO_GPU: dict[str, str] = {
    "gfx942": "MI300X",
    "gfx940": "MI300A",
    "gfx90a": "MI250X",
    "gfx908": "MI100",
}


def detect_target_gpu() -> tuple[str | None, str | None]:
    """Detect AMD GPU architecture via ``rocminfo``.

    Returns (gfx_name, gpu_model) or (None, None) if unavailable.
    """
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None, None
        for line in result.stdout.splitlines():
            if "gfx" in line.lower():
                match = re.search(r"(gfx\d+\w*)", line)
                if match:
                    gfx = match.group(1)
                    return gfx, _GFX_TO_GPU.get(gfx, gfx)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None, None


def _extract_gpu_specs_section(gpu_model: str) -> str | None:
    """Extract the section for *gpu_model* from ``cdna-architecture.md``."""
    kb_root = Path(__file__).resolve().parents[3] / "knowledge-base"
    cdna_path = kb_root / "amd-knowledge-base" / "layer-1-hardware" / "amd-gpu-arch" / "cdna-architecture.md"
    if not cdna_path.exists():
        logger.warning("cdna-architecture.md not found at %s", cdna_path)
        return None

    text = cdna_path.read_text()
    pattern = re.compile(
        rf"(#{1,3}\s+.*{re.escape(gpu_model)}.*?\n)(.*?)(?=\n#{1,3}\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return (match.group(1) + match.group(2)).strip()

    logger.info("No section found for GPU model %s in cdna-architecture.md", gpu_model)
    return None


def get_gpu_specs() -> str | None:
    """Detect GPU and return the relevant specs section, or None."""
    gfx, model = detect_target_gpu()
    if not model:
        logger.info("GPU auto-detection unavailable; no hardware specs will be injected")
        return None
    logger.info("Detected GPU: %s (%s)", model, gfx)
    return _extract_gpu_specs_section(model)


# ---------------------------------------------------------------------------
# KB content loading
# ---------------------------------------------------------------------------

_FLYDSL_REPO_DOCS = [
    "docs/kernel_authoring_guide.md",
    "docs/layout_system_guide.md",
    "docs/prebuilt_kernels_guide.md",
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
    gpu_specs: str | None,
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

    if gpu_specs:
        sections.append(f"<hardware_specs>\n{gpu_specs}\n</hardware_specs>")

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
    max_rounds=10,
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
