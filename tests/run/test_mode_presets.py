"""Unit tests for ``apply_mode_presets`` and ``resolve_max_rounds`` precedence."""

from __future__ import annotations

import pytest

from minisweagent.run.pipeline_helpers import apply_mode_presets, resolve_max_rounds


def _config_with_presets(quick_rounds: int = 2, full_rounds: int = 5) -> dict:
    return {
        "agent": {"step_limit": 100, "cost_limit": 0.0},
        "run": {
            "mode": "full",
            "presets": {
                "quick": {"orchestrator": {"max_rounds": quick_rounds}},
                "full": {"orchestrator": {"max_rounds": full_rounds}},
            },
        },
    }


def test_apply_mode_presets_quick_sets_max_rounds():
    cfg = _config_with_presets(quick_rounds=2)
    apply_mode_presets(cfg, "quick")
    assert cfg["orchestrator"]["max_rounds"] == 2


def test_apply_mode_presets_full_sets_max_rounds():
    cfg = _config_with_presets(full_rounds=5)
    apply_mode_presets(cfg, "full")
    assert cfg["orchestrator"]["max_rounds"] == 5


def test_apply_mode_presets_does_not_touch_step_or_cost_limits():
    cfg = _config_with_presets()
    cfg["agent"]["step_limit"] = 999
    cfg["agent"]["cost_limit"] = 42.0
    apply_mode_presets(cfg, "quick")
    # Mode must NOT modify these.
    assert cfg["agent"]["step_limit"] == 999
    assert cfg["agent"]["cost_limit"] == 42.0


def test_apply_mode_presets_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown run mode"):
        apply_mode_presets(_config_with_presets(), "blazing")


def test_apply_mode_presets_idempotent_when_no_block():
    cfg = {"agent": {"step_limit": 0}}
    # No run.presets block at all -- should be a no-op (with debug log).
    apply_mode_presets(cfg, "quick")
    assert cfg == {"agent": {"step_limit": 0}}


# ---------------------------------------------------------------------------
# resolve_max_rounds precedence: CLI > mode preset > GEAK_MAX_ROUNDS env > default
# ---------------------------------------------------------------------------


def test_resolve_max_rounds_cli_wins_over_everything(monkeypatch):
    monkeypatch.setenv("GEAK_MAX_ROUNDS", "8")
    cfg = {"orchestrator": {"max_rounds": 4}}
    value, source = resolve_max_rounds(cli_max_rounds=7, config=cfg, default=5)
    assert (value, source) == (7, "cli")


def test_resolve_max_rounds_mode_preset_wins_over_env(monkeypatch):
    monkeypatch.setenv("GEAK_MAX_ROUNDS", "8")
    cfg = {"orchestrator": {"max_rounds": 2}}
    value, source = resolve_max_rounds(cli_max_rounds=None, config=cfg, default=5)
    assert (value, source) == (2, "mode")


def test_resolve_max_rounds_env_used_when_no_mode_preset(monkeypatch):
    monkeypatch.setenv("GEAK_MAX_ROUNDS", "8")
    value, source = resolve_max_rounds(cli_max_rounds=None, config={}, default=5)
    assert (value, source) == (8, "env")


def test_resolve_max_rounds_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("GEAK_MAX_ROUNDS", raising=False)
    value, source = resolve_max_rounds(cli_max_rounds=None, config={}, default=5)
    assert (value, source) == (5, "default")


def test_resolve_max_rounds_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("GEAK_MAX_ROUNDS", "not-an-int")
    value, source = resolve_max_rounds(cli_max_rounds=None, config={}, default=5)
    assert (value, source) == (5, "default")
