"""End-to-end integration of the kernel_packages plumbing.

We don't spawn a real harness here — we exercise the wiring:

  * ``create_worktree`` dispatches to a profile's ``make_worktree``
    when one matches.
  * ``ensure_worktree_installed`` short-circuits on a shadow worktree
    (``skip_install`` profile) without invoking pip.
  * ``SaveAndTestTool._inject_compile_bootstrap`` injects the bootstrap
    PYTHONPATH and AITER_REBUILD into a subprocess env dict.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def fake_vllm_source(tmp_path):
    src = tmp_path / "vllm"
    src.mkdir()
    (src / "__init__.py").write_text("# vllm\n")
    (src / "_C.abi3.so").write_bytes(b"\x7fELF")
    return src


def test_create_worktree_dispatches_to_vllm_profile(fake_vllm_source, tmp_path):
    """``create_worktree`` should produce a shadow tree (not git worktree)
    when the source matches the vllm profile."""
    from minisweagent.kernel_packages.shadow_worktree import is_shadow_worktree
    from minisweagent.run.task_file import create_worktree

    dst = tmp_path / "slot_0"
    result = create_worktree(fake_vllm_source, dst)
    assert result == dst
    assert is_shadow_worktree(dst), (
        "create_worktree should have routed through shadow_worktree for "
        "wheel-only vllm-shaped source"
    )


def test_ensure_worktree_installed_skips_shadow_profile(fake_vllm_source, tmp_path, monkeypatch):
    """The install pipeline must short-circuit cleanly on shadow trees."""
    from minisweagent.run.preprocess.worktree_install import ensure_worktree_installed
    from minisweagent.run.task_file import create_worktree

    dst = tmp_path / "slot_0"
    create_worktree(fake_vllm_source, dst)

    # Sentinel: any pip subprocess invocation would explode this.
    def _no_pip(*args, **kwargs):
        raise AssertionError(
            "ensure_worktree_installed must NOT invoke pip on a shadow worktree"
        )

    monkeypatch.setattr("subprocess.run", _no_pip)
    result = ensure_worktree_installed(dst)
    assert result.get("skipped_for_profile") == "vllm"


def test_inject_compile_bootstrap_sets_env_and_pythonpath():
    from minisweagent._compile_bootstrap import bootstrap_dir
    from minisweagent.tools.save_and_test import SaveAndTestTool

    env = {"PYTHONPATH": "/some/existing/path"}
    SaveAndTestTool._inject_compile_bootstrap(env)
    assert env["AITER_REBUILD"] == "2"
    assert bootstrap_dir() in env["PYTHONPATH"]
    # Existing PYTHONPATH preserved (just prefixed).
    assert "/some/existing/path" in env["PYTHONPATH"]


def test_inject_compile_bootstrap_respects_explicit_aiter_rebuild():
    """Caller-supplied AITER_REBUILD wins (debug overrides allowed)."""
    from minisweagent.tools.save_and_test import SaveAndTestTool

    env = {"AITER_REBUILD": "0"}
    SaveAndTestTool._inject_compile_bootstrap(env)
    assert env["AITER_REBUILD"] == "0"


def test_profile_runtime_env_applied_for_shadow_vllm(fake_vllm_source, tmp_path):
    """Shadow-tree vLLM worktree must surface runtime_env + PYTHONPATH prefix."""
    from minisweagent.run.task_file import create_worktree
    from minisweagent.tools.save_and_test import SaveAndTestTool

    dst = tmp_path / "slot_0"
    create_worktree(fake_vllm_source, dst)

    env: dict[str, str] = {}
    SaveAndTestTool._inject_profile_runtime_env(env, dst)
    assert env.get("VLLM_USE_PRECOMPILED") == "1"
    # Shadow worktree root prefixed onto PYTHONPATH so ``import vllm``
    # resolves to the agent's edited copy.
    assert str(dst) in env.get("PYTHONPATH", "")


def test_profile_runtime_env_noop_for_unrelated_repo(tmp_path):
    """Plain repos without a profile match: no env mutation."""
    from minisweagent.tools.save_and_test import SaveAndTestTool

    plain = tmp_path / "plain_repo"
    plain.mkdir()
    (plain / "setup.py").write_text("")
    env: dict[str, str] = {}
    SaveAndTestTool._inject_profile_runtime_env(env, plain)
    assert env == {}
