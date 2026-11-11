"""Prepare the persistent Claude controller without changing serving packages.

Only the native Workflow dispatcher calls this module. Dependencies are installed
with --target and activated in this process, never exported through PYTHONPATH.
"""
from __future__ import annotations

import fcntl
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

SDK_REQUIREMENT = "claude-agent-sdk==0.2.180"
INSTALL_TIMEOUT_S = 300
PIP_URL = "https://bootstrap.pypa.io/pip/pip.pyz"


class SdkBootstrapError(ImportError):
    """Native controller preparation failed (result error: sdk_import_failed)."""


def _load_sdk():
    sdk = importlib.import_module("claude_agent_sdk")
    if not callable(getattr(sdk, "ClaudeAgentOptions", None)) or not callable(
        getattr(sdk, "ClaudeSDKClient", None)
    ):
        # This is a missing dependency capability, not a caller argument error.
        raise ImportError("claude_agent_sdk lacks the persistent ClaudeSDKClient API")  # noqa: TRY004
    return sdk


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("SDK preparation budget exhausted")
    return seconds


def _cache_root():
    override = os.environ.get("GEAK_SDK_CACHE_DIR")
    return Path(override).expanduser() if override else (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "geak" / "sdk"
    )


@contextmanager
def _install_lock(path, deadline):
    # flock is released by the OS even if an interrupted controller is killed.
    with path.open("a") as lock:
        while True:
            _remaining(deadline)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(0.1, _remaining(deadline)))
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run(command, deadline, log):
    subprocess.run(
        command, check=True, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, timeout=_remaining(deadline),
    )


def _validate(target, deadline, log):
    # A fresh interpreter must load the target, not an SDK elsewhere on sys.path.
    code = (
        "import pathlib, sys; sys.path.insert(0, sys.argv[1]); "
        "import claude_agent_sdk as sdk; "
        "assert pathlib.Path(sdk.__file__).resolve().is_relative_to("
        "pathlib.Path(sys.argv[1]).resolve()); "
        "assert callable(sdk.ClaudeSDKClient) and callable(sdk.ClaudeAgentOptions)"
    )
    _run([sys.executable, "-I", "-c", code, str(target)], deadline, log)


def _install(target, staging, deadline, log):
    if importlib.util.find_spec("pip") is not None:
        installer = [sys.executable, "-m", "pip"]
    else:
        # Minimal cluster Python images can lack both pip and ensurepip. Run the
        # official standalone pip archive without installing pip into Python.
        archive = staging / "pip.pyz"
        _run([
            sys.executable, "-c",
            "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])",
            PIP_URL, str(archive),
        ], deadline, log)
        installer = [sys.executable, str(archive)]
    _run(installer + [
        "--isolated", "--disable-pip-version-check", "--no-input", "install",
        "--no-user", "--target", str(target), SDK_REQUIREMENT,
    ], deadline, log)


def _activate(target):
    original_path = sys.path[:]
    # A failed import / older SDK may have left modules from the wrong SDK.
    previous = {
        name: module for name, module in sys.modules.copy().items()
        if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk.")
    }
    for name in previous:
        del sys.modules[name]
    sys.path.insert(0, str(target))
    importlib.invalidate_caches()
    try:
        return _load_sdk()
    except Exception:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
                del sys.modules[name]
        sys.modules.update(previous)
        raise


def ensure_claude_sdk(timeout_s=INSTALL_TIMEOUT_S):
    """Reuse a usable SDK, or install the tested version in a private cache.

    Installation, cache validation and lock waits share a bounded budget. The
    caller must deduct preparation time from its total workflow timeout.
    """
    try:
        return _load_sdk()
    except ImportError:
        pass

    deadline = time.monotonic() + min(timeout_s, INSTALL_TIMEOUT_S)
    log_path = None
    try:
        _remaining(deadline)
        root = _cache_root().resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = f"{SDK_REQUIREMENT.split('==')[1]}-{sys.implementation.cache_tag}-{sysconfig.get_platform()}"
        target = root / key
        with _install_lock(root / f"{key}.lock", deadline):
            # Private logs may contain package-index URLs; never echo pip output
            # (or subprocess exceptions containing it) into the public result.
            fd, name = tempfile.mkstemp(prefix="bootstrap-", suffix=".log", dir=root)
            log_path = Path(name)
            with os.fdopen(fd, "w") as log:
                valid = False
                if target.is_dir():
                    try:
                        _validate(target, deadline, log)
                        valid = True
                    except subprocess.CalledProcessError:
                        pass
                if not valid:
                    print(f"[GEAK] Preparing {SDK_REQUIREMENT} in {target}", file=sys.stderr)
                    with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as tmp:
                        staging = Path(tmp)
                        packages = staging / "packages"
                        _install(packages, staging, deadline, log)
                        _validate(packages, deadline, log)
                        if target.exists():
                            shutil.rmtree(target)
                        packages.rename(target)
                _remaining(deadline)
                try:
                    sdk = _activate(target)
                except Exception:
                    traceback.print_exc(file=log)
                    raise
            log_path.unlink()
            return sdk
    except Exception as exc:  # noqa: BLE001 -- sanitize installer errors for result.json
        detail = f" See private installer log: {log_path}." if log_path else ""
        raise SdkBootstrapError(
            f"Could not prepare the persistent Claude SDK ({type(exc).__name__})."
            f"{detail} Install {SDK_REQUIREMENT} in {sys.executable} or set "
            "GEAK_SDK_CACHE_DIR to a writable cache and retry. "
            "The native workflow was not started."
        ) from None
