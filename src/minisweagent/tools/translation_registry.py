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
    perf_fail_threshold: float = 0.1
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


def _is_paged_decode_attention(text: str, source_path: Path) -> bool:
    """Detect paged-decode attention kernels (MLA & PagedAttention) for KB loading.

    These ``seqlen_q == 1`` paged-KV kernels are written with ``torch.matmul`` +
    paged-cache indexing (not SDPA/MHA/``@``-transpose), so they slip past the
    generic ``attention`` regexes. Detect them explicitly so the attention KB
    (which holds § Decode Attention) and the gemm KB both load. Matches explicit
    naming (docstrings, classes, fused APIs), the structural paged-cache combo
    (paged table + cache seqlens + a KV cache), and filename stems such as
    ``MultiHeadLatentAttention.py`` / ``PagedAttentionKVCache.py``.
    """
    if re.search(
        r"MultiHeadLatent|Multi-head\s+Latent|multihead\s+latent|"
        r"PagedAttention|Paged\s+(KV\s+Cache\s+)?Attention",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"mla_fwd_decode|flydsl_mla_fwd_decode|get_mla_metadata|mla_reduce",
        text,
    ):
        return True
    # Structural combo: a paged block/page table + cache seqlens + a KV cache.
    # Covers MLA (block_table + kv_cache + headdim_qk/headdim_v) and
    # PagedAttention (page_table + k_cache/v_cache + symmetric headdim).
    if (
        re.search(r"block_table|page_table", text)
        and re.search(r"cache_seqlen", text)
        and re.search(r"kv_cache|k_cache|v_cache", text)
    ):
        return True
    if re.search(r"MultiHeadLatent|\bmla\b|PagedAttention", source_path.stem, re.IGNORECASE):
        return True
    return False


def _is_manual_softmax_attention(text: str) -> bool:
    """Detect attention written manually with ``matmul`` + ``softmax``.

    Kernels like MHA/SDPA compute ``softmax(Q @ K^T) @ V`` using ``torch.matmul``
    (or ``bmm``) with a transposed operand rather than
    ``F.scaled_dot_product_attention``, ``nn.MultiheadAttention``, or the ``@``
    operator, so they slip past the generic ``attention`` regexes. Require BOTH a
    matmul against a transposed operand (the ``Q @ K^T`` score step) AND a softmax,
    so plain GEMM kernels that merely transpose an operand are not misflagged.
    """
    has_qkt = bool(re.search(r"(?:matmul|bmm)\s*\([^)]*\.(?:transpose|mT|permute)", text))
    has_softmax = bool(re.search(r"softmax", text))
    return has_qkt and has_softmax


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
    # Paged-decode attention kernels (MLA, PagedAttention) don't match the generic
    # ``attention`` regexes (they use ``torch.matmul`` + paged-cache logic, not
    # SDPA/MHA/@-transpose), so detect them explicitly and load the attention KB
    # (which contains the § Decode Attention section) plus the gemm KB (Split-K
    # GEMM, needed for the decomposed path).
    if _is_paged_decode_attention(text, source_path):
        for cat in ("attention", "gemm"):
            if cat not in categories:
                categories.append(cat)
    # Manually-implemented attention (MHA, SDPA: softmax(Q@K^T)@V via torch.matmul +
    # transpose) also misses the generic ``attention`` regexes — force the KB.
    if _is_manual_softmax_attention(text) and "attention" not in categories:
        categories.append("attention")
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
       - Default: from ``skills/pytorch2flydsl-translation/docs/``
       - With ``flydsl_repo``: from FlyDSL repo ``docs/`` directory

    2. **Translation content** (always from skill docs):
       - Translation guide (PyTorch op mapping, structural patterns, pitfalls)
       - Category-specific guides (reductions, GEMM, attention)
    """
    kb_root = Path(__file__).resolve().parents[3] / "skills" / "pytorch2flydsl-translation" / "docs"
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
