"""Tests for ``minisweagent.run.preprocess_v3.registry``.

All tests use ``tmp_path`` to build a synthetic ``subagents/preprocess``
tree — the registry never reads from the real repo on disk. This keeps
the suite isolated from whatever YAML files PR 3 eventually drops in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.run.preprocess_v3.registry import (
    SubagentRegistry,
    SubagentSpec,
    SubagentSpecError,
)

_ALPHA_YAML = """\
name: alpha
description: First synthetic subagent for testing.
system_prompt: |
  You are alpha. Answer concisely.
model: claude-opus-4.6
tools:
  - bash
  - read_file
max_steps: 12
"""

_BETA_YAML = """\
name: beta
description: Second synthetic subagent for testing.
system_prompt: "You are beta."
"""

_BETA_WITH_EXTRAS_YAML = """\
name: beta
description: Second synthetic subagent for testing.
system_prompt: "You are beta."
custom_routing_key: pipeline-step-3b
unknown_future_knob:
  retries: 4
  backoff: exponential
"""

_MISSING_FIELD_YAML = """\
name: gamma
description: Missing system_prompt — should fail validation.
"""

_BAD_TOOLS_YAML = """\
name: delta
description: Has a non-list tools field.
system_prompt: "You are delta."
tools: bash,read_file
"""

_INVALID_YAML = """\
name: bad
description: This file has invalid YAML
system_prompt: "you are bad"
  this_is: : not yaml :
"""


def _write_subagent(root: Path, name: str, body: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    yaml_path = folder / "SUBAGENT.yaml"
    yaml_path.write_text(body, encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------------
# discover()
# ---------------------------------------------------------------------------


def test_discover_loads_two_subagents(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "alpha", _ALPHA_YAML)
    _write_subagent(tmp_path, "beta", _BETA_YAML)

    registry = SubagentRegistry(root=tmp_path)
    specs = registry.discover()

    assert set(specs) == {"alpha", "beta"}
    assert isinstance(specs["alpha"], SubagentSpec)
    assert isinstance(specs["beta"], SubagentSpec)


def test_discover_returns_empty_for_missing_root(tmp_path: Path) -> None:
    registry = SubagentRegistry(root=tmp_path / "no-such-dir")
    assert registry.discover() == {}


def test_discover_returns_empty_when_root_has_no_yamls(tmp_path: Path) -> None:
    """A folder without a SUBAGENT.yaml is silently skipped, not an error."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "README.md").write_text("placeholder", encoding="utf-8")

    registry = SubagentRegistry(root=tmp_path)
    assert registry.discover() == {}


def test_discover_populates_field_values(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "alpha", _ALPHA_YAML)

    spec = SubagentRegistry(root=tmp_path).discover()["alpha"]

    assert spec.name == "alpha"
    assert spec.description == "First synthetic subagent for testing."
    assert spec.system_prompt.strip() == "You are alpha. Answer concisely."
    assert spec.model == "claude-opus-4.6"
    assert spec.tools == ["bash", "read_file"]
    assert spec.max_steps == 12


def test_discover_applies_defaults(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "beta", _BETA_YAML)

    spec = SubagentRegistry(root=tmp_path).discover()["beta"]

    assert spec.model is None
    assert spec.tools == []
    assert spec.max_steps == 30
    assert spec.extras == {}


# ---------------------------------------------------------------------------
# get() / names()
# ---------------------------------------------------------------------------


def test_get_returns_spec_for_known_name(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "alpha", _ALPHA_YAML)
    _write_subagent(tmp_path, "beta", _BETA_YAML)

    registry = SubagentRegistry(root=tmp_path)

    assert registry.get("alpha").name == "alpha"
    assert registry.get("beta").name == "beta"


def test_get_raises_keyerror_for_unknown_name(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "alpha", _ALPHA_YAML)
    registry = SubagentRegistry(root=tmp_path)

    with pytest.raises(KeyError, match="no subagent named 'does-not-exist'"):
        registry.get("does-not-exist")


def test_names_returns_sorted_list(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "beta", _BETA_YAML)
    _write_subagent(tmp_path, "alpha", _ALPHA_YAML)

    assert SubagentRegistry(root=tmp_path).names() == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Error / extras / forward-compat
# ---------------------------------------------------------------------------


def test_discover_raises_on_missing_required_field(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "gamma", _MISSING_FIELD_YAML)
    registry = SubagentRegistry(root=tmp_path)

    with pytest.raises(SubagentSpecError, match="missing required field"):
        registry.discover()


def test_discover_raises_on_bad_tools_field(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "delta", _BAD_TOOLS_YAML)
    registry = SubagentRegistry(root=tmp_path)

    with pytest.raises(SubagentSpecError, match="'tools' must be a list"):
        registry.discover()


def test_discover_raises_on_invalid_yaml(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "bad", _INVALID_YAML)
    registry = SubagentRegistry(root=tmp_path)

    with pytest.raises(SubagentSpecError, match="invalid YAML"):
        registry.discover()


def test_extras_captures_unknown_keys(tmp_path: Path) -> None:
    _write_subagent(tmp_path, "beta", _BETA_WITH_EXTRAS_YAML)

    spec = SubagentRegistry(root=tmp_path).discover()["beta"]

    assert spec.extras == {
        "custom_routing_key": "pipeline-step-3b",
        "unknown_future_knob": {"retries": 4, "backoff": "exponential"},
    }


def test_discover_raises_on_duplicate_name(tmp_path: Path) -> None:
    """Two folders whose YAMLs declare the same `name:` are an error."""
    _write_subagent(tmp_path, "alpha", _ALPHA_YAML)
    # Drop a second folder whose YAML reuses name: alpha.
    _write_subagent(tmp_path, "alpha-alias", _ALPHA_YAML)

    registry = SubagentRegistry(root=tmp_path)

    with pytest.raises(SubagentSpecError, match="duplicate subagent name"):
        registry.discover()


def test_default_root_resolves_to_repo_subagents_preprocess() -> None:
    """The no-arg constructor points at the in-repo placeholder dir.

    This test pins the resolution rule (walk parents looking for
    ``pyproject.toml`` + ``subagents/``) so a future refactor can't
    silently break the default-root contract.
    """
    registry = SubagentRegistry()
    assert registry.root.name == "preprocess"
    assert registry.root.parent.name == "subagents"
