"""SDK preparation: CPU-only; mocked package downloads, real import checks."""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from interface import sdk_bootstrap as sb


def _missing():
    raise ImportError("missing SDK")


@pytest.fixture
def missing_sdk(monkeypatch, tmp_path):
    monkeypatch.setenv("GEAK_SDK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(sb, "_load_sdk", _missing)
    # A missed mock must never reach a package index in CI.
    monkeypatch.setattr(sb, "_run", lambda *a: pytest.fail("unexpected subprocess"))
    return tmp_path / "cache"


@pytest.fixture
def fake_install(monkeypatch, missing_sdk):
    calls = []

    def install(target, staging, deadline, log):
        calls.append("install")
        target.mkdir()
        (target / "ready").touch()

    def validate(target, deadline, log):
        calls.append("validate")
        if not (target / "ready").exists():
            raise subprocess.CalledProcessError(1, "validate")

    def activate(target):
        calls.append("activate")
        assert (target / "ready").exists()
        return "SDK"

    monkeypatch.setattr(sb, "_install", install)
    monkeypatch.setattr(sb, "_validate", validate)
    monkeypatch.setattr(sb, "_activate", activate)
    return calls


def test_installed_sdk_needs_no_cache_or_installer(monkeypatch):
    sdk = SimpleNamespace(ClaudeAgentOptions=lambda: None, ClaudeSDKClient=lambda: None)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    monkeypatch.setattr(sb, "_cache_root", lambda: pytest.fail("must reuse installed SDK"))
    assert sb.ensure_claude_sdk() is sdk


@pytest.mark.parametrize("sdk", [None, SimpleNamespace(), SimpleNamespace(ClaudeSDKClient=1)])
def test_missing_or_obsolete_sdk_requires_preparation(monkeypatch, sdk):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    with pytest.raises(ImportError):
        sb._load_sdk()


def test_first_install_and_offline_cache_reuse(missing_sdk, fake_install):
    assert sb.ensure_claude_sdk() == "SDK"
    assert fake_install == ["install", "validate", "activate"]
    assert not list(missing_sdk.glob("bootstrap-*.log"))
    assert not list(missing_sdk.glob(".install-*"))
    fake_install.clear()
    assert sb.ensure_claude_sdk() == "SDK"
    assert fake_install == ["validate", "activate"]


def test_invalid_cache_is_replaced_only_after_validation(missing_sdk, fake_install):
    sb.ensure_claude_sdk()
    ready = next(missing_sdk.glob("*/ready"))
    ready.unlink()
    fake_install.clear()
    sb.ensure_claude_sdk()
    assert fake_install == ["validate", "install", "validate", "activate"]
    assert ready.exists()


@pytest.mark.parametrize("failure_at", ["_install", "_validate", "_activate"])
def test_failures_keep_private_logs_and_never_report_success(
    monkeypatch, missing_sdk, fake_install, failure_at,
):
    def fail(*args):
        if failure_at != "_activate":
            args[-1].write("private-index-credential\n")
        raise RuntimeError("private-index-credential")

    monkeypatch.setattr(sb, failure_at, fail)
    with pytest.raises(sb.SdkBootstrapError) as caught:
        sb.ensure_claude_sdk()
    assert "private-index-credential" not in str(caught.value)
    assert "not started" in str(caught.value)
    log = next(missing_sdk.glob("bootstrap-*.log"))
    assert str(log) in str(caught.value)
    assert log.stat().st_mode & 0o777 == 0o600
    assert not list(missing_sdk.glob(".install-*"))
    if failure_at != "_activate":
        assert not list(missing_sdk.glob("*/ready"))


def test_unwritable_cache_is_actionable(monkeypatch, missing_sdk):
    missing_sdk.write_text("not a directory")
    with pytest.raises(sb.SdkBootstrapError, match="writable cache"):
        sb.ensure_claude_sdk()


def test_exhausted_budget_does_not_install(missing_sdk):
    with pytest.raises(sb.SdkBootstrapError, match="TimeoutError"):
        sb.ensure_claude_sdk(timeout_s=0)
    assert not missing_sdk.exists()


def test_installer_budget_is_capped(monkeypatch, missing_sdk, fake_install):
    def validate(target, deadline, log):
        assert 299 < deadline - time.monotonic() <= 300
    monkeypatch.setattr(sb, "_validate", validate)
    sb.ensure_claude_sdk(timeout_s=99999)


def test_cache_location_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("GEAK_SDK_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert sb._cache_root() == tmp_path / ".cache/geak/sdk"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert sb._cache_root() == tmp_path / "xdg/geak/sdk"
    monkeypatch.setenv("GEAK_SDK_CACHE_DIR", str(tmp_path / "custom"))
    assert sb._cache_root() == tmp_path / "custom"


@pytest.mark.parametrize("has_pip", [True, False])
def test_install_targets_private_directory_even_without_pip(monkeypatch, tmp_path, has_pip):
    calls = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if has_pip else None)
    monkeypatch.setattr(sb, "_run", lambda command, *a: calls.append(command))
    target = tmp_path / "packages"
    sb._install(target, tmp_path, 123, None)
    assert len(calls) == (1 if has_pip else 2)
    installer = calls[-1]
    assert installer[0] == sys.executable
    assert installer[installer.index("--target") + 1] == str(target)
    assert "--no-user" in installer
    assert installer[-1] == sb.SDK_REQUIREMENT
    if has_pip:
        assert installer[1:3] == ["-m", "pip"]
    else:
        assert sb.PIP_URL in calls[0]
        assert installer[1] == str(tmp_path / "pip.pyz")


def test_run_uses_remaining_budget_and_private_output(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    monkeypatch.setattr(sb.time, "monotonic", lambda: 7)
    log = object()
    sb._run(["installer"], 10, log)
    assert calls == [(["installer"], {
        "check": True, "stdin": subprocess.DEVNULL, "stdout": log,
        "stderr": subprocess.STDOUT, "timeout": 3,
    })]


def test_contending_installer_waits_for_lock_then_reuses_cache(missing_sdk, fake_install):
    # Holding the same lock through another file descriptor simulates a second
    # process. The contender cannot publish or activate while it is held.
    sb.ensure_claude_sdk()
    fake_install.clear()
    lock_path = next(missing_sdk.glob("*.lock"))
    with sb._install_lock(lock_path, time.monotonic() + 5):
        with pytest.raises(sb.SdkBootstrapError, match="TimeoutError"):
            sb.ensure_claude_sdk(timeout_s=0.03)
        assert fake_install == []
    sb.ensure_claude_sdk()
    assert fake_install == ["validate", "activate"]


def _write_sdk(target, code):
    pkg = target / "claude_agent_sdk"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(code)


def test_validation_imports_target_in_fresh_interpreter(tmp_path):
    _write_sdk(tmp_path, "class ClaudeSDKClient: pass\nclass ClaudeAgentOptions: pass\n")
    with (tmp_path / "validate.log").open("w") as log:
        sb._validate(tmp_path, time.monotonic() + 10, log)
        (tmp_path / "claude_agent_sdk/__init__.py").write_text("# incomplete SDK\n")
        with pytest.raises(subprocess.CalledProcessError):
            sb._validate(tmp_path, time.monotonic() + 10, log)


@pytest.mark.parametrize("broken", [False, True])
def test_activation_replaces_old_sdk_without_exporting_pythonpath(monkeypatch, tmp_path, broken):
    old = SimpleNamespace(old=True)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", old)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk.stale", old)
    monkeypatch.setattr(sys, "path", sys.path[:])
    original_path = sys.path[:]
    env = dict(os.environ)
    code = "class ClaudeSDKClient: pass\nclass ClaudeAgentOptions: pass\n"
    if broken:
        code = "from . import partial\nraise ImportError('incompatible dependency')\n"
    _write_sdk(tmp_path, code)
    (tmp_path / "claude_agent_sdk/partial.py").write_text("")
    if broken:
        with pytest.raises(ImportError):
            sb._activate(tmp_path)
        assert sys.path == original_path
        assert sys.modules["claude_agent_sdk"] is old
        assert sys.modules["claude_agent_sdk.stale"] is old
        assert "claude_agent_sdk.partial" not in sys.modules
    else:
        sdk = sb._activate(tmp_path)
        assert Path(sdk.__file__).is_relative_to(tmp_path)
        assert "claude_agent_sdk.stale" not in sys.modules
        assert sys.path[0] == str(tmp_path)
    assert dict(os.environ) == env
