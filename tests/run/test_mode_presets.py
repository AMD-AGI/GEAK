"""Unit tests for ``resolve_max_rounds`` precedence."""

from __future__ import annotations

from minisweagent.run.pipeline_helpers import resolve_max_rounds


# ---------------------------------------------------------------------------
# resolve_max_rounds precedence: CLI > yaml run.max_rounds > GEAK_MAX_ROUNDS env > default
# ---------------------------------------------------------------------------


def test_resolve_max_rounds_cli_wins_over_everything(monkeypatch):
    monkeypatch.setenv("GEAK_MAX_ROUNDS", "8")
    cfg = {"run": {"max_rounds": 4}}
    value, source = resolve_max_rounds(cli_max_rounds=7, config=cfg, default=5)
    assert (value, source) == (7, "cli")


def test_resolve_max_rounds_yaml_wins_over_env(monkeypatch):
    monkeypatch.setenv("GEAK_MAX_ROUNDS", "8")
    cfg = {"run": {"max_rounds": 2}}
    value, source = resolve_max_rounds(cli_max_rounds=None, config=cfg, default=5)
    assert (value, source) == (2, "yaml")


def test_resolve_max_rounds_env_used_when_no_yaml(monkeypatch):
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
