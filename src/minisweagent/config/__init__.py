# Modifications Copyright(C)[2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration files and utilities for mini-SWE-agent."""

import os
from pathlib import Path

import yaml

builtin_config_dir = Path(__file__).parent


def get_config_path(config_spec: str | Path) -> Path:
    """Get the path to a config file."""
    config_spec = Path(config_spec)
    if config_spec.suffix != ".yaml":
        config_spec = config_spec.with_suffix(".yaml")
    candidates = [
        Path(config_spec),
        Path(os.getenv("MSWEA_CONFIG_DIR", ".")) / config_spec,
        builtin_config_dir / config_spec,
        builtin_config_dir / "extra" / config_spec,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find config file for {config_spec} (tried: {candidates})")


def load_config(config_spec: str | Path) -> dict:
    """Resolve *config_spec* and return the parsed YAML as a dict."""
    return yaml.safe_load(get_config_path(config_spec).read_text()) or {}


def load_agent_config(config_spec: str | Path) -> tuple[dict, dict]:
    """Convenience wrapper: return ``(agent_config, model_config)`` from a config file."""
    cfg = load_config(config_spec)
    return cfg.get("agent", {}), cfg.get("model", {})


__all__ = ["builtin_config_dir", "get_config_path", "load_config", "load_agent_config"]
