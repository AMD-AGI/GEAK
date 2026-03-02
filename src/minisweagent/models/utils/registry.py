"""Shared litellm model-registry helpers."""

from __future__ import annotations

import json
from pathlib import Path

import litellm


def register_litellm_models(registry_path: str | Path | None) -> None:
    """Load a JSON model-registry file and register models with litellm."""
    if not registry_path:
        return
    path = Path(registry_path)
    if path.is_file():
        litellm.utils.register_model(json.loads(path.read_text()))
