#!/usr/bin/env python3
"""Tests for run_e2e's swappable-agent-backend selection and dispatch.

CONTRACT under test:
  * A provider key on its own selects the CLI that key belongs to, so a
    key-only setup never additionally has to set GEAK_AGENT_BACKEND.
  * An AMBIGUOUS credential environment (two backends configured) keeps the
    native Claude path rather than hijacking it — the same shape rule the JS
    side applies, so the two halves cannot disagree about which keys mean what.
  * Once a backend is selected, the workflow's top-level return is recovered
    from the runtime's --result-file, then its stdout, then the on-disk
    workflow_return.json — in that order, because each is a weaker witness
    than the one before it.

Selection is decided at IMPORT time (module constants), so these tests reload
the module under a controlled environment rather than poking the constants.

Run: python3 -m pytest GEAK/interface/test_run_e2e_runtime_dispatch.py -v
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve().parent

# Every variable that can influence selection. Cleared before each load so the
# developer's own shell (a stray GEAK_AMDKEY is enough) cannot change the verdict.
_SELECTION_ENV = (
    "GEAK_AMDKEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "GEAK_AGENT_BACKEND", "GEAK_AGENT_PROFILE", "GEAK_MODEL", "GEAK_AGENT_AUTO",
)


def _fresh(monkeypatch, **env):
    for name in _SELECTION_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("run_e2e_rt", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Proc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


# ── selection ───────────────────────────────────────────────────────────────

def test_no_credentials_stays_on_the_native_path(monkeypatch):
    rx = _fresh(monkeypatch)
    assert rx.AUTO_BACKEND == ""
    assert rx.EFFECTIVE_BACKEND == ""
    assert rx.USE_RUNTIME is False
    assert rx.runtime_combo_label() == "native (claude/Workflow)"
    assert rx._runtime_selection_args() == []


@pytest.mark.parametrize("key", ["GEAK_AMDKEY", "OPENAI_API_KEY"])
def test_a_provider_key_alone_selects_codex(monkeypatch, key):
    """Both codex provider_autoselect triggers must select codex by themselves;
    the label says "(from key)" so an operator can see it was not explicit."""
    rx = _fresh(monkeypatch, **{key: "x" * 32})
    assert rx.AUTO_BACKEND == "codex"
    assert rx.USE_RUNTIME is True
    assert rx._runtime_selection_args() == ["--agent", "codex"]
    assert rx.runtime_combo_label() == "agent=codex (from key)"


def test_an_ambiguous_credential_environment_keeps_native(monkeypatch):
    """A gateway key sitting next to an Anthropic-side variable is ambiguous.
    Silently moving that run onto codex would change which model answered."""
    rx = _fresh(monkeypatch, GEAK_AMDKEY="x" * 32, ANTHROPIC_API_KEY="sk-ant-x")
    assert rx.AUTO_BACKEND == ""
    assert rx.USE_RUNTIME is False


def test_geak_agent_auto_off_disables_key_based_selection(monkeypatch):
    rx = _fresh(monkeypatch, GEAK_AMDKEY="x" * 32, GEAK_AGENT_AUTO="0")
    assert rx.AUTO_BACKEND == ""
    assert rx.USE_RUNTIME is False


def test_explicit_backend_wins_and_is_not_labelled_as_derived(monkeypatch):
    """An explicit backend short-circuits derivation entirely: AUTO_BACKEND
    stays empty, so the label must not claim the key chose it."""
    rx = _fresh(monkeypatch, GEAK_AGENT_BACKEND="codex", GEAK_AMDKEY="x" * 32)
    assert rx.AUTO_BACKEND == ""
    assert rx.EFFECTIVE_BACKEND == "codex"
    assert rx.runtime_combo_label() == "agent=codex"


def test_a_profile_selects_the_runtime_and_carries_a_model_override(monkeypatch):
    rx = _fresh(monkeypatch, GEAK_AGENT_PROFILE="codex-gpt56", GEAK_MODEL="openai_gpt56")
    assert rx.USE_RUNTIME is True
    assert rx._runtime_selection_args() == [
        "--profile", "codex-gpt56", "--model", "openai_gpt56",
    ]
    assert rx.runtime_combo_label() == "profile=codex-gpt56 model=openai_gpt56"


def test_an_unreadable_registry_degrades_to_native(monkeypatch, tmp_path):
    """Selection reads registry.json for the credential names. If it cannot be
    read there is no basis to reroute the run, so it must stay native."""
    rx = _fresh(monkeypatch, GEAK_AMDKEY="x" * 32)
    monkeypatch.setattr(rx, "RUNTIME_REGISTRY", tmp_path / "missing.json")
    assert rx._derive_agent_from_env() == ""


# ── dispatch ────────────────────────────────────────────────────────────────

def _runtime_mod(monkeypatch, captured):
    rx = _fresh(monkeypatch, GEAK_AGENT_BACKEND="codex")

    def fake_run(cmd, **kw):
        captured["cmd"], captured["kw"] = cmd, kw
        return captured["proc"]

    monkeypatch.setattr(rx, "subprocess", SimpleNamespace(run=fake_run))
    return rx


def test_the_result_file_is_the_authoritative_return(monkeypatch, tmp_path):
    captured = {"proc": _Proc(stdout="WORKFLOW_RESULT {\"eval_dir\": \"/from/stdout\"}")}
    rx = _runtime_mod(monkeypatch, captured)
    (tmp_path / "runtime_result.json").write_text(
        json.dumps({"eval_dir": str(tmp_path), "throughput_speedup": 1.21}), encoding="utf-8")

    out = rx._invoke_via_runtime({"model_path": "/m"}, 60, str(tmp_path))
    assert out["throughput_speedup"] == 1.21
    assert out["eval_dir"] == str(tmp_path)


def test_the_command_carries_the_selection_the_files_and_a_timeout_margin(monkeypatch, tmp_path):
    """--agent is passed explicitly so the JS never re-derives the backend, and
    the wrapper timeout must exceed the workflow's own budget or a graceful
    finalize gets killed and the run reports nothing."""
    captured = {"proc": _Proc(stdout='{"eval_dir": "/e"}')}
    rx = _runtime_mod(monkeypatch, captured)

    rx._invoke_via_runtime({"model_path": "/m"}, 3600, str(tmp_path))

    cmd = captured["cmd"]
    assert cmd[0] == rx.NODE_BIN and cmd[1] == str(rx.RUNTIME_SCRIPT)
    assert cmd[cmd.index("--args") + 1] == json.dumps({"model_path": "/m"})
    assert cmd[cmd.index("--agent") + 1] == "codex"
    assert cmd[cmd.index("--result-file") + 1] == str(tmp_path / "runtime_result.json")
    assert cmd[cmd.index("--metrics-file") + 1] == str(tmp_path / "runtime_metrics.json")
    assert captured["kw"]["timeout"] == 3600 + 900


def test_without_an_eval_dir_no_result_file_is_requested(monkeypatch):
    captured = {"proc": _Proc(stdout='{"eval_dir": "/e"}')}
    rx = _runtime_mod(monkeypatch, captured)

    assert rx._invoke_via_runtime({"model_path": "/m"}, 0)["eval_dir"] == "/e"
    assert "--result-file" not in captured["cmd"]
    assert captured["kw"]["timeout"] is None


def test_stdout_is_used_when_the_result_file_is_absent(monkeypatch, tmp_path):
    captured = {"proc": _Proc(stdout='WORKFLOW_RESULT {"eval_dir": "/from/stdout"}')}
    rx = _runtime_mod(monkeypatch, captured)

    assert rx._invoke_via_runtime({}, 60, str(tmp_path))["eval_dir"] == "/from/stdout"


def test_a_corrupt_result_file_falls_through_to_stdout(monkeypatch, tmp_path):
    """A truncated result-file must not mask a perfectly good stdout return."""
    captured = {"proc": _Proc(stdout='WORKFLOW_RESULT {"eval_dir": "/from/stdout"}')}
    rx = _runtime_mod(monkeypatch, captured)
    (tmp_path / "runtime_result.json").write_text("{not json", encoding="utf-8")

    assert rx._invoke_via_runtime({}, 60, str(tmp_path))["eval_dir"] == "/from/stdout"


def test_the_on_disk_workflow_return_is_the_last_resort(monkeypatch, tmp_path):
    captured = {"proc": _Proc(stdout="no json here at all")}
    rx = _runtime_mod(monkeypatch, captured)
    (tmp_path / "workflow_return.json").write_text(
        json.dumps({"eval_dir": str(tmp_path), "status": "ok"}), encoding="utf-8")

    assert rx._invoke_via_runtime({}, 60, str(tmp_path))["status"] == "ok"


def test_a_nonzero_exit_names_the_combo_and_keeps_the_stderr_tail(monkeypatch, tmp_path):
    captured = {"proc": _Proc(rc=7, stderr="model not found")}
    rx = _runtime_mod(monkeypatch, captured)

    with pytest.raises(RuntimeError) as e:
        rx._invoke_via_runtime({}, 60, str(tmp_path))
    assert "rc=7" in str(e.value)
    assert "agent=codex" in str(e.value)
    assert "model not found" in str(e.value)


def test_no_recoverable_return_anywhere_raises_workflow_parse_error(monkeypatch, tmp_path):
    """Distinct from a crash: the runtime exited 0 but produced nothing usable,
    which main() must classify as a scrape failure rather than a runner error."""
    captured = {"proc": _Proc(stdout="finished, no return")}
    rx = _runtime_mod(monkeypatch, captured)

    with pytest.raises(rx.WorkflowParseError):
        rx._invoke_via_runtime({}, 60, str(tmp_path))


def test_invoke_workflow_routes_to_the_runtime_only_with_ps_args(monkeypatch):
    """invoke_workflow is the single dispatch point: with a backend selected AND
    structured args it takes the runtime; without ps_args there is nothing to
    hand the runtime, so it must fall back to the native path."""
    rx = _fresh(monkeypatch, GEAK_AGENT_BACKEND="codex")
    seen = {}

    def fake_runtime(ps_args, timeout_s, eval_dir=None):
        seen["runtime"] = (ps_args, timeout_s, eval_dir)
        return {"ok": 1}

    monkeypatch.setattr(rx, "_invoke_via_runtime", fake_runtime)
    monkeypatch.setattr(rx, "_invoke_via_cli", lambda p, t: '{"eval_dir": "/native"}')

    assert rx.invoke_workflow("prompt", 60, "/e", ps_args={"a": 1}) == {"ok": 1}
    assert seen["runtime"] == ({"a": 1}, 60, "/e")
    assert rx.invoke_workflow("prompt", 60, "/e")["eval_dir"] == "/native"
