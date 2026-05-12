"""Deterministic helpers for evaluation-contract freezing.

Used by ``ContractResolutionPhase`` after Discovery.  Extracts
structured fields (notably ``compile_command``) from messy
``eval_command`` strings so per-language ``commandment.j2`` templates
receive a proper ``compile_command`` for ``## Setup``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fragments that usually start the *evaluation* half of a merged shell
# command — everything before the first match is treated as environment
# + build (compile) when a compile-like token is present.
_STOP_SUBSTRINGS = (
    "correctness",
    "pytest",
    "--correctness",
    "--benchmark",
    "--full-benchmark",
    "performance",  # e.g. task_runner.py performance
    " run_benchmark",
)

# At least one of these must appear in the candidate prefix for us to
# call it a compile/build command block.
_BUILD_SUBSTRINGS = (
    "compile",
    "cmake",
    "ninja",
    " meson ",
    "make ",
    "make\t",
    "hipcc",
    "nvcc",
    "setup.py build",
    "pip install",
)


_MAKE_WORD_RE = re.compile(r"(?<![A-Za-z0-9_])(?:g?make)\b")


def _segment_has_build_token(low: str) -> bool:
    if any(tok in low for tok in _BUILD_SUBSTRINGS):
        return True
    # Tuple uses ``"make "`` with trailing space — catch bare ``make`` / ``gmake``.
    return bool(_MAKE_WORD_RE.search(low))


def _segment_is_env_or_cd(part: str) -> bool:
    s = part.strip()
    sl = s.lower()
    return sl.startswith("export ") or sl.startswith("cd ") or sl.startswith("source ")


def infer_compile_command_from_eval(eval_command: str | None) -> str | None:
    """Split ``eval_command`` on ``&&`` and return the compile/build prefix.

    Stops before obvious correctness/benchmark segments, and — once a
    build-like token has been collected — before unrelated trailing
    commands (e.g. ``make && ./run_tests.sh`` → ``make``).

    Returns ``None`` when no confident split exists (caller may invoke
    the optional LLM normalizer).
    """
    if not eval_command or not str(eval_command).strip():
        return None
    text = str(eval_command).strip()
    parts = [p.strip() for p in text.split("&&") if p.strip()]
    if not parts:
        return None

    acc: list[str] = []
    saw_build = False
    for part in parts:
        low = part.lower()
        if any(stop in low for stop in _STOP_SUBSTRINGS):
            if "compile" in low and all(s not in low for s in ("correctness", "performance")):
                acc.append(part)
            break
        has_build = _segment_has_build_token(low)
        if saw_build and not has_build and not _segment_is_env_or_cd(part):
            break
        acc.append(part)
        if has_build:
            saw_build = True

    joined = " && ".join(acc).strip() if acc else ""
    if not joined:
        return None
    lowj = joined.lower()
    if _segment_has_build_token(lowj):
        return joined
    return None


def discovery_digest(discovery: dict[str, Any] | None, *, max_chars: int = 6000) -> dict[str, Any]:
    """Return a JSON-serializable, size-capped snapshot for contract.json."""
    if not discovery:
        return {}
    raw = json.dumps(discovery, indent=2, default=str)
    if len(raw) <= max_chars:
        return {"truncated": False, "json": discovery}
    return {"truncated": True, "preview": raw[:max_chars] + "\n…"}


def codebase_context_excerpt(codebase_context_path: str | None, *, max_chars: int = 8000) -> str:
    if not codebase_context_path:
        return ""
    p = Path(codebase_context_path)
    if not p.is_file():
        return ""
    try:
        body = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("codebase_context excerpt unreadable: %s", exc)
        return ""
    if len(body) <= max_chars:
        return body
    return body[:max_chars] + "\n…"


def build_evaluation_contract(ctx: Any) -> dict[str, Any]:
    """Assemble the v1 ``evaluation_contract`` dict from ``PhaseContext``."""
    lang = getattr(ctx.language, "name", None) if ctx.language is not None else None
    eval_cmd = getattr(ctx, "eval_command", None)
    tier0_compile = infer_compile_command_from_eval(
        eval_cmd if isinstance(eval_cmd, str) else None
    )
    return {
        "version": 1,
        "kernel_language": lang,
        "kernel_path": ctx.kernel_path or "",
        "repo_root": ctx.repo_root or "",
        "harness_cli": getattr(ctx, "harness", None),
        "eval_command": eval_cmd,
        "correctness_command": ctx.correctness_command,
        "performance_command": ctx.performance_command,
        "compile_command": tier0_compile,
        "tier0_deterministic_compile": tier0_compile is not None,
        "discovery_digest": discovery_digest(ctx.discovery if isinstance(ctx.discovery, dict) else None),
        "codebase_context_excerpt": codebase_context_excerpt(ctx.codebase_context_path),
    }


__all__ = [
    "build_evaluation_contract",
    "codebase_context_excerpt",
    "discovery_digest",
    "infer_compile_command_from_eval",
]
