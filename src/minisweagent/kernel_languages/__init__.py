"""`KernelLanguage` registry — one detection site for all language routing.

Replaces the 3 scattered detection functions in today's codebase:
  - src/minisweagent/cli.py::_normalize_kernel_type
  - src/minisweagent/agents/heterogeneous/task_generator.py::_infer_kernel_type
  - src/minisweagent/run/preprocess/discovery_types.py::_infer_kernel_language

Per docs/refactor/EXECUTION_PLAN.md §1 (audit §9 inventory) + §16.1.

Usage:
    from minisweagent.kernel_languages import registry

    lang = registry.detect_best(Path("/path/to/kernel.py"))
    # -> KernelLanguage(name='triton', ...)

    lang = registry.get("hip")
    # -> KernelLanguage(name='hip', ...)

The registry is populated at import time with the built-in languages (Triton,
HIP). Adding a new language means dropping a folder under `kernel_languages/<name>/`
and calling `registry.register(...)` from that language's module (or using
the `geak add-language` scaffolder, which does it automatically — lands in
a later commit).
"""

from __future__ import annotations

import re
from pathlib import Path

from minisweagent.kernel_languages.base import KernelLanguage


class _Registry:
    """Singleton registry. Access via module-level `registry` instance."""

    def __init__(self) -> None:
        self._langs: dict[str, KernelLanguage] = {}

    def register(self, lang: KernelLanguage) -> None:
        """Register a language. Idempotent — re-registration overwrites."""
        self._langs[lang.name] = lang

    def get(self, name: str) -> KernelLanguage | None:
        """Look up a language by its canonical `name`. None if not found."""
        return self._langs.get(name)

    def all(self) -> list[KernelLanguage]:
        return list(self._langs.values())

    def detect_best(self, path: Path) -> KernelLanguage | None:
        """Find the most likely language for a kernel file.

        Score = file_extension match + detect_hints regex hits in file content.
        Returns the highest-scoring language, or None if no language scores > 0.
        """
        if not path.exists():
            # Degrade to extension-only match
            return self._detect_by_extension(path)

        ext = path.suffix.lower()
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return self._detect_by_extension(path)

        scored: list[tuple[float, KernelLanguage]] = []
        for lang in self._langs.values():
            score = 0.0
            if ext in lang.file_extensions:
                score += 1.0
            for pat in lang.detect_hints:
                if re.search(pat, content, re.MULTILINE):
                    score += 1.0
            if score > 0:
                scored.append((score, lang))

        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        return scored[0][1]

    def detect_best_by_name(self, name: str) -> KernelLanguage | None:
        """Legacy shim for code that passes a string like 'triton' instead of a Path.

        Replaces ``_normalize_kernel_type`` from today's ``cli.py``.
        """
        n = (name or "").strip().lower()
        # Existing codebase normalizations: both 'triton' and 'rocm' sometimes
        # appear; 'hip' and 'rocblas' are HIP-family. Apply the same aliases.
        aliases = {
            "rocm": "hip",
            "rocblas": "hip",
            "cuda": "triton",  # legacy compatibility; GEAK's Triton supports CUDA-style wrappers too
        }
        canonical = aliases.get(n, n)
        return self.get(canonical)

    def _detect_by_extension(self, path: Path) -> KernelLanguage | None:
        ext = path.suffix.lower()
        for lang in self._langs.values():
            if ext in lang.file_extensions:
                return lang
        return None


# Module-level singleton — import this, not a new _Registry().
registry = _Registry()


# ---------------------------------------------------------------------------
# Auto-register built-in languages at import time
# ---------------------------------------------------------------------------


def _bootstrap_builtin_languages() -> None:
    """Import built-in language modules so they register themselves.

    Each language's `kernel_language.py` calls `registry.register(...)` at
    module import time. This function triggers those imports.
    """
    # Import order matters only for determinism in logging — all languages are
    # registered by the time this function returns.
    try:
        from minisweagent.kernel_languages.triton import kernel_language as _  # noqa: F401
    except ImportError:
        pass  # triton bundle not yet populated — safe during PR-1 staging
    try:
        from minisweagent.kernel_languages.hip import kernel_language as _  # noqa: F401
    except ImportError:
        pass


_bootstrap_builtin_languages()


__all__ = ["KernelLanguage", "registry"]
