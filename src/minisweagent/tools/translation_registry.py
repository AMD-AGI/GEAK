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

from minisweagent import get_data_dir

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
    perf_fail_threshold: float = 0.1
    perf_warn_threshold: float = 0.8
    supported: bool = True
    self_review: bool = False
    review_triggers_retry: bool = False
    review_retry_on_efficiency: bool = False
    # Skill-docs directory (under skills/) that holds this pair's KB files.
    # Defaults to the pytorch->flydsl skill for back-compat.
    kb_skill_dir: str = "pytorch2flydsl-translation"


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


def _detect_triton(kernel_path: Path) -> bool:
    """Return True if *kernel_path* is a Triton kernel (``@triton.jit``)."""
    if kernel_path.suffix.lower() != ".py":
        return False
    try:
        text = kernel_path.read_text(errors="replace")
    except OSError:
        return False
    has_triton = "import triton" in text or "from triton" in text
    has_jit = "@triton.jit" in text or "tl.program_id" in text or "triton.language" in text
    return has_triton and has_jit


def _detect_tilelang(kernel_path: Path) -> bool:
    """Return True if *kernel_path* is a TileLang kernel."""
    if kernel_path.suffix.lower() != ".py":
        return False
    try:
        text = kernel_path.read_text(errors="replace")
    except OSError:
        return False
    has_tl = "import tilelang" in text or "from tilelang" in text
    has_prim = "T.Kernel" in text or "tilelang.language" in text or "@tilelang.jit" in text
    return has_tl and has_prim


def _detect_ck(kernel_path: Path) -> bool:
    """Return True if *kernel_path* is a Composable Kernel (CK) source (.cu/.cpp/.hip).

    CK usage often lives in ``.cuh`` headers while the top-level ``.cu`` only
    ``#include``s them, so we also key off CK include names and the canonical
    ``ck_gemm_*`` / ``ck_*`` path segments, not just inline ``ck::`` tokens.
    """
    if kernel_path.suffix.lower() not in {".cu", ".cpp", ".hip", ".cc", ".cxx", ".cuh", ".hpp", ".h"}:
        return False
    pstr = str(kernel_path).lower()
    if "/ck_gemm" in pstr or "/ck_" in pstr or "composable_kernel" in pstr or "/ck/" in pstr:
        return True
    try:
        text = kernel_path.read_text(errors="replace")
    except OSError:
        return False
    markers = ("ck::", "ck_tile", "composable_kernel", '"ck/', "<ck/", "ck_gemm", "blockscale_common.cuh")
    return any(m in text for m in markers)


def _detect_hip(kernel_path: Path) -> bool:
    """Return True if *kernel_path* is a raw HIP/C++ device kernel (not CK)."""
    if kernel_path.suffix.lower() not in {".cu", ".cpp", ".hip", ".cc", ".cxx"}:
        return False
    try:
        text = kernel_path.read_text(errors="replace")
    except OSError:
        return False
    has_hip = "__global__" in text or "hip_runtime" in text or "hipLaunchKernel" in text
    return has_hip and not _detect_ck(kernel_path)


def _detect_flydsl(kernel_path: Path) -> bool:
    """Return True if *kernel_path* is a FlyDSL kernel."""
    if kernel_path.suffix.lower() != ".py":
        return False
    try:
        text = kernel_path.read_text(errors="replace")
    except OSError:
        return False
    return "flydsl" in text and ("@flyc.kernel" in text or "flydsl.compiler" in text or "@flyc.jit" in text)


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
    "conv_pool_bn": [
        r"nn\.Conv2d",
        r"nn\.Conv1d",
        r"nn\.Conv3d",
        r"F\.conv2d",
        r"F\.max_pool2d",
        r"nn\.MaxPool2d",
        r"nn\.BatchNorm2d",
        r"F\.batch_norm",
        r"nn\.AvgPool2d",
        r"F\.avg_pool2d",
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


def _tilelang_env_setup(repo_root: Path, flydsl_repo: Path | None = None) -> dict[str, str]:
    """Env overrides for running TileLang kernels.

    TileLang ships as an installed package (``import tilelang``) with a dev
    build root under ``/opt/tilelang/build`` on the standard image. We surface
    that build root on PYTHONPATH when present; otherwise rely on the installed
    package. ``flydsl_repo`` is accepted (and ignored) so the registry can call
    every ``env_setup`` with a uniform signature.
    """
    overrides: dict[str, str] = {}
    candidates = [
        os.environ.get("TILELANG_HOME"),
        "/opt/tilelang/build",
        "/opt/tilelang",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.is_dir():
            existing = os.environ.get("PYTHONPATH", "")
            overrides["PYTHONPATH"] = f"{p}:{existing}" if existing else str(p)
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
       - Default: from ``skills/pytorch2flydsl-translation/docs/``
       - With ``flydsl_repo``: from FlyDSL repo ``docs/`` directory

    2. **Translation content** (always from skill docs):
       - Translation guide (PyTorch op mapping, structural patterns, pitfalls)
       - Category-specific guides (reductions, GEMM, attention)
    """
    kb_root = get_data_dir("skills") / pair.kb_skill_dir / "docs"
    native_pure_root = kb_root / "native-pure"
    native_root = kb_root / "native"
    sections: list[str] = []

    def _resolve_kb_path(filename: str) -> Path:
        """Prefer native-pure/ or native/ version based on env vars."""
        if _native_pure_mode:
            pure_path = native_pure_root / filename
            if pure_path.exists():
                return pure_path
        if _native_mode:
            native_path = native_root / filename
            if native_path.exists():
                return native_path
        return kb_root / filename

    if flydsl_repo:
        for doc_path in _FLYDSL_REPO_DOCS:
            full_path = flydsl_repo / doc_path
            if full_path.exists():
                sections.append(full_path.read_text())
    else:
        for f in pair.kb_base_files:
            path = _resolve_kb_path(f)
            if path.exists():
                sections.append(_strip_frontmatter(path.read_text()))
            else:
                logger.warning("KB base file not found: %s", path)

    for f in pair.kb_translation_files:
        path = _resolve_kb_path(f)
        if path.exists():
            sections.append(_strip_frontmatter(path.read_text()))
        else:
            logger.warning("KB translation file not found: %s", path)

    for cat in categories:
        if cat in pair.kb_category_files:
            path = _resolve_kb_path(pair.kb_category_files[cat])
            if path.exists():
                sections.append(_strip_frontmatter(path.read_text()))

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Registry: built-in pairs
# ---------------------------------------------------------------------------

_native_mode = os.environ.get("GEAK_NATIVE_PATTERN", "") == "1"
_native_pure_mode = os.environ.get("GEAK_NATIVE_PURE", "") == "1"

if _native_pure_mode:
    _config_name = "mini_kernel_pytorch_to_flydsl_native_pure"
elif _native_mode:
    _config_name = "mini_kernel_pytorch_to_flydsl_native"
else:
    _config_name = "mini_kernel_pytorch_to_flydsl"

_PYTORCH_TO_FLYDSL = TranslationPair(
    source="pytorch",
    target="flydsl",
    detect_source=_detect_pytorch_module,
    config_name=_config_name,
    harness_config_name="mini_unit_test_agent_pytorch_translation",
    harness_candidate_flag="--flydsl-kernel",
    candidate_filename_fn=lambda stem: f"{stem}_flydsl.py",
    kb_base_files=["flydsl_translation_api_reference.md"],
    kb_translation_files=(
        ["flydsl_translation_guide.md", "flydsl_translation_im2col_pad.md"]
        if _native_pure_mode
        else ["flydsl_translation_guide.md"]
    ),
    kb_category_files={
        "gemm": "flydsl_translation_gemm.md",
        "reductions": "flydsl_translation_reductions.md",
        "attention": "flydsl_translation_attention.md",
        "conv_pool_bn": "flydsl_translation_conv_pool_bn.md",
    },
    env_setup=_flydsl_env_setup,
    max_attempts=3,
)


# ---------------------------------------------------------------------------
# Additional pairs — targets are the two OPTIMIZED DSLs (FlyDSL, TileLang).
# We convert the less-optimized sources (pytorch, triton, ck, hip) TOWARD these
# targets; FlyDSL and TileLang are always >= CK/Triton/HIP in achievable perf on
# gfx942, so the valuable rewrite direction is *into* them.
# Adding a pair requires only a TranslationPair entry (zero pipeline changes).
# ---------------------------------------------------------------------------

# FlyDSL-target pairs reuse the FlyDSL skill KB (target API is identical; only
# the source language differs, which the per-pair config/subagent prompt covers).
_FLYDSL_KB_BASE = ["flydsl_translation_api_reference.md"]
_FLYDSL_KB_TRANSLATION = ["flydsl_translation_guide.md"]
_FLYDSL_KB_CATEGORY = {
    "gemm": "flydsl_translation_gemm.md",
    "reductions": "flydsl_translation_reductions.md",
    "attention": "flydsl_translation_attention.md",
    "conv_pool_bn": "flydsl_translation_conv_pool_bn.md",
}

# TileLang-target pairs share the tilelang-translation skill KB.
_TILELANG_KB_BASE = ["tilelang_translation_api_reference.md"]
_TILELANG_KB_TRANSLATION = ["tilelang_translation_guide.md"]


def _flydsl_pair(source: str, detect: Callable[[Path], bool]) -> TranslationPair:
    return TranslationPair(
        source=source,
        target="flydsl",
        detect_source=detect,
        config_name="mini_kernel_pytorch_to_flydsl",
        harness_config_name="mini_unit_test_agent_pytorch_translation",
        harness_candidate_flag="--flydsl-kernel",
        candidate_filename_fn=lambda stem: f"{stem}_flydsl.py",
        kb_base_files=list(_FLYDSL_KB_BASE),
        kb_translation_files=list(_FLYDSL_KB_TRANSLATION),
        kb_category_files=dict(_FLYDSL_KB_CATEGORY),
        env_setup=_flydsl_env_setup,
        kb_skill_dir="pytorch2flydsl-translation",
        max_attempts=3,
    )


def _tilelang_pair(source: str, detect: Callable[[Path], bool]) -> TranslationPair:
    return TranslationPair(
        source=source,
        target="tilelang",
        detect_source=detect,
        config_name="mini_kernel_to_tilelang",
        harness_config_name="mini_unit_test_agent_pytorch_translation",
        harness_candidate_flag="--tilelang-kernel",
        candidate_filename_fn=lambda stem: f"{stem}_tilelang.py",
        kb_base_files=list(_TILELANG_KB_BASE),
        kb_translation_files=list(_TILELANG_KB_TRANSLATION),
        kb_category_files={},
        env_setup=_tilelang_env_setup,
        kb_skill_dir="tilelang-translation",
        max_attempts=3,
    )


# Sources we convert toward FlyDSL.
_TRITON_TO_FLYDSL = _flydsl_pair("triton", _detect_triton)
_CK_TO_FLYDSL = _flydsl_pair("ck", _detect_ck)
_HIP_TO_FLYDSL = _flydsl_pair("hip", _detect_hip)

# Sources we convert toward TileLang.
_PYTORCH_TO_TILELANG = _tilelang_pair("pytorch", _detect_pytorch_module)
_TRITON_TO_TILELANG = _tilelang_pair("triton", _detect_triton)
_CK_TO_TILELANG = _tilelang_pair("ck", _detect_ck)
_HIP_TO_TILELANG = _tilelang_pair("hip", _detect_hip)

# Cross-conversion between the two top-tier DSLs (one may win per-op / per-arch).
_TILELANG_TO_FLYDSL = _flydsl_pair("tilelang", _detect_tilelang)
_FLYDSL_TO_TILELANG = _tilelang_pair("flydsl", _detect_flydsl)


class TranslationRegistry:
    """Registry of supported translation pairs."""

    def __init__(self) -> None:
        self._pairs: list[TranslationPair] = [
            # pytorch -> flydsl is the original, detection-priority-first entry.
            _PYTORCH_TO_FLYDSL,
            # -> FlyDSL (optimized target)
            _TRITON_TO_FLYDSL,
            _CK_TO_FLYDSL,
            _HIP_TO_FLYDSL,
            _TILELANG_TO_FLYDSL,
            # -> TileLang (optimized target)
            _PYTORCH_TO_TILELANG,
            _TRITON_TO_TILELANG,
            _CK_TO_TILELANG,
            _HIP_TO_TILELANG,
            _FLYDSL_TO_TILELANG,
        ]

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
