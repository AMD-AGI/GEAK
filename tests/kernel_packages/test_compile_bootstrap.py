"""Tests for the compile-mode bootstrap injected into harness subprocesses.

Coverage:

  * ``bootstrap_dir()`` resolves to a directory that contains
    ``sitecustomize.py``.
  * The sitecustomize, when imported in a fresh subprocess, sets
    ``AITER_REBUILD=2`` (defensive ``setdefault`` semantics).
  * The aiter-jit-core post-exec hook clears ``rebuilded_list`` when
    that module is loaded.

We do NOT spawn a real Python with ``site``; instead we exec the
sitecustomize source in a controlled namespace and then import a
synthetic ``aiter.jit.core`` module to verify the hook fires.
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
from pathlib import Path

import pytest


def test_bootstrap_dir_returns_path_with_sitecustomize():
    from minisweagent._compile_bootstrap import bootstrap_dir

    bd = Path(bootstrap_dir())
    assert bd.is_dir(), f"bootstrap_dir() = {bd} is not a directory"
    assert (bd / "sitecustomize.py").is_file(), (
        f"{bd}/sitecustomize.py missing — bootstrap won't be auto-loaded"
    )


def test_sitecustomize_sets_aiter_rebuild_env(monkeypatch, tmp_path):
    """Loading sitecustomize must default AITER_REBUILD when unset."""
    monkeypatch.delenv("AITER_REBUILD", raising=False)
    # Re-execute sitecustomize source in a controlled namespace.
    sc_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "minisweagent"
        / "_compile_bootstrap"
        / "sitecustomize.py"
    )
    src = sc_path.read_text(encoding="utf-8")
    # Use a fresh ns so previous imports don't pollute.
    ns: dict = {"__name__": "sitecustomize", "__file__": str(sc_path)}
    exec(compile(src, str(sc_path), "exec"), ns)
    assert os.environ.get("AITER_REBUILD") == "2"


def test_sitecustomize_does_not_override_explicit_aiter_rebuild(monkeypatch):
    """``setdefault`` semantics — explicit env wins over the bootstrap."""
    monkeypatch.setenv("AITER_REBUILD", "0")
    sc_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "minisweagent"
        / "_compile_bootstrap"
        / "sitecustomize.py"
    )
    src = sc_path.read_text(encoding="utf-8")
    ns: dict = {"__name__": "sitecustomize", "__file__": str(sc_path)}
    exec(compile(src, str(sc_path), "exec"), ns)
    assert os.environ.get("AITER_REBUILD") == "0"


def test_aiter_jit_core_hook_clears_rebuilded_list(monkeypatch, tmp_path):
    """Synthesise ``aiter.jit.core`` on disk, install hook, import, verify."""
    # Build a fake aiter package on disk.
    fake = tmp_path / "fake_pkgs"
    aiter_dir = fake / "aiter" / "jit"
    aiter_dir.mkdir(parents=True)
    (fake / "aiter" / "__init__.py").write_text("")
    (fake / "aiter" / "jit" / "__init__.py").write_text("")
    (aiter_dir / "core.py").write_text(
        textwrap.dedent(
            """
            # Mimic aiter's shipped allowlist semantics.
            rebuilded_list = ["module_aiter_core"]
            """
        ).lstrip()
    )

    # Pre-emptively clear AITER_REBUILD so sitecustomize sets it (assert later).
    monkeypatch.delenv("AITER_REBUILD", raising=False)

    # Make our fake package importable, evict any cached real aiter.
    monkeypatch.syspath_prepend(str(fake))
    for mod in list(sys.modules):
        if mod == "aiter" or mod.startswith("aiter."):
            sys.modules.pop(mod, None)

    # Load sitecustomize fresh into its own namespace; this also installs
    # the meta_path finder.
    sc_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "minisweagent"
        / "_compile_bootstrap"
        / "sitecustomize.py"
    )
    src = sc_path.read_text(encoding="utf-8")
    ns: dict = {"__name__": "sitecustomize", "__file__": str(sc_path)}
    try:
        exec(compile(src, str(sc_path), "exec"), ns)

        # Hook is now on sys.meta_path.  Import aiter.jit.core and verify
        # the post-exec hook cleared rebuilded_list.
        mod = importlib.import_module("aiter.jit.core")
        assert getattr(mod, "rebuilded_list", None) == [], (
            "rebuilded_list should be empty after sitecustomize hook fires; "
            f"got {mod.rebuilded_list!r}"
        )
    finally:
        # Best-effort cleanup of meta_path entries we added.
        finders = ns.get("_AiterJitCoreFinder")
        if finders is not None:
            sys.meta_path[:] = [
                f for f in sys.meta_path if not isinstance(f, finders)
            ]
        for mod_name in list(sys.modules):
            if mod_name == "aiter" or mod_name.startswith("aiter."):
                sys.modules.pop(mod_name, None)


def test_bootstrap_is_dependency_free():
    """The bootstrap MUST only use stdlib (no minisweagent imports), so
    it can load in any subprocess env including dependency-poor ones."""
    sc_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "minisweagent"
        / "_compile_bootstrap"
        / "sitecustomize.py"
    )
    src = sc_path.read_text(encoding="utf-8")
    # Heuristic but precise enough: forbid any `minisweagent` reference.
    assert "minisweagent" not in src, (
        "sitecustomize.py imports minisweagent; bootstrap must stay stdlib-only"
    )
