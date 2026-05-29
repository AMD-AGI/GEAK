"""Tests for the package-profile registry and detection plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minisweagent.kernel_packages import REGISTRY, PackageProfile, detect_packages


def test_vllm_profile_is_registered():
    """``vllm_profile`` registers via side-effect on package import."""
    names = {p.name for p in REGISTRY}
    assert "vllm" in names, f"vllm profile missing from REGISTRY: {names}"


def test_detect_returns_empty_for_unrelated_repo(tmp_path):
    """A plain Python package without vllm signatures matches nothing."""
    pkg = tmp_path / "myrepo"
    pkg.mkdir()
    (pkg / "setup.py").write_text("")
    (pkg / "__init__.py").write_text("")
    assert detect_packages(pkg) == []


def test_detect_matches_wheel_only_vllm_layout(tmp_path):
    """Detect succeeds on a path mimicking site-packages/vllm/."""
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    (vllm / "__init__.py").write_text(
        '"""vllm.\n\nA high-throughput and memory-efficient inference engine."""\n'
        "__version__ = '0.0.0'\n"
    )
    # Fake the binary extension so detect's heuristic fires.
    (vllm / "_C.abi3.so").write_bytes(b"\x7fELF" + b"\0" * 64)
    assert any(p.name == "vllm" for p in detect_packages(vllm))


def test_detect_skips_vllm_when_csrc_present(tmp_path):
    """Source-built vllm checkouts (with csrc/) should NOT match the
    wheel-only profile — they fall through to git-worktree + pip -e."""
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    (vllm / "__init__.py").write_text("# vllm")
    (vllm / "_C.abi3.so").write_bytes(b"\x7fELF" + b"\0" * 64)
    (vllm / "csrc").mkdir()  # source tree present → editable install path
    assert all(p.name != "vllm" for p in detect_packages(vllm))


def test_detect_safe_against_profile_exceptions(monkeypatch, tmp_path):
    """A profile whose ``detect`` raises must not poison detection of peers."""
    from minisweagent.kernel_packages import profile as profile_mod

    def boom(_):
        raise RuntimeError("boom")

    bad = PackageProfile(name="exploding", detect=boom)
    monkeypatch.setattr(profile_mod, "REGISTRY", [bad, *profile_mod.REGISTRY])

    # Should not raise, and should still return matching real profiles
    # for a legitimate vllm-shaped path.
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    (vllm / "__init__.py").write_text("# vllm")
    (vllm / "_C.abi3.so").write_bytes(b"\x7fELF" + b"\0" * 64)
    matched = detect_packages(vllm)
    assert any(p.name == "vllm" for p in matched)


def test_runtime_env_present_on_vllm_profile():
    """vLLM profile must inject VLLM_USE_PRECOMPILED to disable rebuild paths."""
    [vllm] = [p for p in REGISTRY if p.name == "vllm"]
    assert vllm.runtime_env.get("VLLM_USE_PRECOMPILED") == "1"
    assert vllm.skip_install is True
    assert vllm.make_worktree is not None
