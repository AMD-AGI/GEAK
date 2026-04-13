"""Path helpers for the owned preprocess stage."""

from __future__ import annotations

import sys
from pathlib import Path

from minisweagent import get_repo_root


def get_preprocess_repo_root() -> Path:
    """Return the repository root for files living under ``run/preprocess``."""

    return get_repo_root()


def ensure_preprocess_mcp_importable(*subdirs: str) -> None:
    """Prepend preprocess-owned MCP source roots to ``sys.path`` if needed."""

    repo_root = get_preprocess_repo_root()
    for subdir in subdirs:
        path = str(repo_root / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)


__all__ = ["ensure_preprocess_mcp_importable", "get_preprocess_repo_root"]
