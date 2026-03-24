"""Preprocess-local config loader helpers.

Preprocess-owned agents should prefer prompt/config assets that live under
``run/preprocess/config/`` so the stage can be moved as a unit. When a local
config is absent, these helpers fall back to the shared global config loader.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from minisweagent.config import get_config_path

_LOCAL_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_preprocess_agent_config(config_spec: str | Path) -> tuple[dict, dict]:
    """Load a preprocess-owned agent config, falling back to shared config.

    ``config_spec`` may be given with or without a ``.yaml`` suffix.
    """

    config_path = Path(config_spec)
    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")

    local_path = _LOCAL_CONFIG_DIR / config_path.name
    if local_path.exists():
        cfg = yaml.safe_load(local_path.read_text()) or {}
        return cfg.get("agent", {}), cfg.get("model", {})

    cfg = yaml.safe_load(get_config_path(config_path).read_text()) or {}
    return cfg.get("agent", {}), cfg.get("model", {})


__all__ = ["load_preprocess_agent_config"]
