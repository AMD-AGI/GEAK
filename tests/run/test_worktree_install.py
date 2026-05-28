"""Tests for the editable-install hook that keeps worktree .so files in sync.

Coverage:

  * ``_discover_install_targets`` finds every installable sub-project
    under a worktree and skips vendored / build / test sub-trees.
  * ``_detect_layout`` (legacy single-target shim) recognises aiter
    (root ``setup.py``), sglang's ``python/`` subdir layout, and
    PEP-517 only (root ``pyproject.toml``); returns ``None`` for
    unrelated repos.
  * ``ensure_worktree_installed`` is a no-op on non-installable
    worktrees and on missing paths.
  * Multi-package monorepos (sglang-style ``python/`` +
    ``sgl-kernel/``) get every sub-project installed in a single call.
  * The install subprocess is gated by per-(worktree, target) dedup:
    calling it twice with the same worktree only invokes pip once
    PER sub-project; ``force=True`` bypasses dedup.
  * The snapshot is one-shot per package across runs (don't overwrite
    an existing ``info.json``).
  * ``restore_original_packages`` runs pip uninstall + reinstall and
    deletes the snapshot on success; failures don't raise.

We mock out :func:`subprocess.run` so no real pip / setup.py
invocation happens during tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import minisweagent.run.preprocess.worktree_install as wi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path: Path):
    """Redirect snapshot root + reset per-process dedup state between tests."""
    monkeypatch.setattr(wi, "_SNAPSHOT_ROOT", tmp_path / "backup")
    wi._INSTALLED_IN_PROCESS.clear()
    wi._TIER2_UNINSTALLED_PKGS.clear()
    yield
    wi._INSTALLED_IN_PROCESS.clear()
    wi._TIER2_UNINSTALLED_PKGS.clear()


# ---------------------------------------------------------------------------
# _detect_layout
# ---------------------------------------------------------------------------


class TestDetectLayout:
    def test_aiter_layout(self, tmp_path: Path):
        (tmp_path / "setup.py").write_text("# stub\n", encoding="utf-8")
        marker, argv, cwd_rel = wi._detect_layout(tmp_path)
        assert marker == "setup.py"
        # Modern unified install path: `pip install -e <abs target>` for
        # both setup.py and pyproject.toml. The legacy ``setup.py
        # develop`` invocation was deprecated upstream.
        assert argv[:5] == [argv[0], "-m", "pip", "install", "-e"]
        assert str(tmp_path) in argv  # absolute path of target dir
        assert cwd_rel == "."

    def test_sglang_layout(self, tmp_path: Path):
        (tmp_path / "python").mkdir()
        (tmp_path / "python" / "setup.py").write_text("# stub\n", encoding="utf-8")
        marker, argv, cwd_rel = wi._detect_layout(tmp_path)
        assert marker == "python/setup.py"
        assert argv[:5] == [argv[0], "-m", "pip", "install", "-e"]
        # The target is the sglang python/ dir, passed as an absolute path.
        assert str(tmp_path / "python") in argv
        assert cwd_rel == "."

    def test_root_setup_wins_over_subdir(self, tmp_path: Path):
        # New generic discovery walks depth-first: a root-level
        # installable marker wins because we don't recurse into a
        # sub-tree that's already installable. This is the right
        # behavior for repos whose root ``setup.py`` already declares
        # ``packages = find_namespace_packages()`` covering the
        # ``python/`` sub-tree.
        (tmp_path / "setup.py").write_text("# stub\n", encoding="utf-8")
        (tmp_path / "python").mkdir()
        (tmp_path / "python" / "setup.py").write_text("# stub\n", encoding="utf-8")
        marker, _, _ = wi._detect_layout(tmp_path)
        assert marker == "setup.py"

    def test_pep517_layout(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        marker, argv, _ = wi._detect_layout(tmp_path)
        assert marker == "pyproject.toml"
        assert argv[:5] == [argv[0], "-m", "pip", "install", "-e"]

    def test_no_layout(self, tmp_path: Path):
        # Empty dir → not installable → returns None.
        assert wi._detect_layout(tmp_path) is None

    def test_missing_dir(self, tmp_path: Path):
        # ensure_worktree_installed must tolerate non-existent paths.
        result = wi.ensure_worktree_installed(tmp_path / "no-such-thing")
        # Backward-compat fields stay; new ``targets`` key defaults to [].
        assert result["installed"] is False
        assert result["layout"] is None
        assert result["package"] is None
        assert result["returncode"] == 0
        assert result["stderr_tail"] == ""
        assert result["duration_s"] == 0.0
        assert result["targets"] == []

    def test_none_path(self):
        # None / empty path is a no-op.
        result = wi.ensure_worktree_installed(None)
        assert result["installed"] is False
        assert result["targets"] == []


# ---------------------------------------------------------------------------
# _discover_install_targets — multi-package monorepo discovery
# ---------------------------------------------------------------------------


class TestDiscoverInstallTargets:
    """Generic discovery: any depth / count / shape of monorepo."""

    def test_sglang_real_layout_finds_both_subprojects(self, tmp_path: Path):
        # Real sglang layout: NO root marker; PEP-517-only sub-projects
        # at python/ and sgl-kernel/. Both must be discovered so the
        # kernel header edits actually rebuild sgl_kernel.
        (tmp_path / "python").mkdir()
        (tmp_path / "python" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (tmp_path / "sgl-kernel").mkdir()
        (tmp_path / "sgl-kernel" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        targets = wi._discover_install_targets(tmp_path)
        markers = sorted(t[0] for t in targets)
        assert markers == ["python/pyproject.toml", "sgl-kernel/pyproject.toml"]
        # Each install argv targets the correct ABSOLUTE sub-project dir.
        target_dirs = sorted(str(t[2]) for t in targets)
        assert target_dirs == [str(tmp_path / "python"), str(tmp_path / "sgl-kernel")]

    def test_skips_vendored_directories(self, tmp_path: Path):
        # 3rdparty / third_party / vendor / submodules / external are
        # never descended into, even when they contain installable
        # markers.
        (tmp_path / "python").mkdir()
        (tmp_path / "python" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        for vendored in ("3rdparty", "third_party", "vendor", "submodules", "external", "deps"):
            (tmp_path / vendored).mkdir()
            (tmp_path / vendored / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        targets = wi._discover_install_targets(tmp_path)
        markers = [t[0] for t in targets]
        assert markers == ["python/pyproject.toml"]

    def test_skips_build_and_cache_directories(self, tmp_path: Path):
        # build/, dist/, .cache/, .pytest_cache/, etc must be skipped
        # even when (incorrectly) containing installable markers.
        (tmp_path / "kernel").mkdir()
        (tmp_path / "kernel" / "setup.py").write_text("# stub\n", encoding="utf-8")
        for noisy in ("build", "dist", ".cache", ".pytest_cache", "__pycache__", "_build"):
            (tmp_path / noisy).mkdir()
            (tmp_path / noisy / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        targets = wi._discover_install_targets(tmp_path)
        assert [t[0] for t in targets] == ["kernel/setup.py"]

    def test_skips_test_and_example_directories(self, tmp_path: Path):
        # tests/, e2e_test/, examples/, samples/, docs/, benchmarks/,
        # scripts/, tools/ are skipped — they're rarely shippable
        # distributions and pulling their build deps would be
        # expensive / break the install loop.
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        for skip_name in ("tests", "test", "e2e_test", "e2e-tests", "examples",
                          "samples", "docs", "benchmarks", "scripts", "tools"):
            (tmp_path / skip_name).mkdir()
            (tmp_path / skip_name / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        targets = wi._discover_install_targets(tmp_path)
        assert [t[0] for t in targets] == ["core/pyproject.toml"]

    def test_does_not_recurse_into_installable_subtree(self, tmp_path: Path):
        # Once a directory is recognised as a target, its sub-tree is NOT
        # descended further. Otherwise we'd install every nested package
        # and conflict with the parent's install_requires graph.
        (tmp_path / "python").mkdir()
        (tmp_path / "python" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (tmp_path / "python" / "subpkg").mkdir()
        (tmp_path / "python" / "subpkg" / "setup.py").write_text("# stub\n", encoding="utf-8")
        targets = wi._discover_install_targets(tmp_path)
        markers = [t[0] for t in targets]
        assert markers == ["python/pyproject.toml"]
        assert not any("subpkg" in m for m in markers)

    def test_max_targets_bound(self, tmp_path: Path):
        # A pathological repo with many top-level installables must
        # still terminate. ``max_targets`` caps the total count.
        for i in range(20):
            d = tmp_path / f"pkg{i:02d}"
            d.mkdir()
            (d / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        targets = wi._discover_install_targets(tmp_path, max_targets=3)
        assert len(targets) == 3

    def test_max_depth_bound(self, tmp_path: Path):
        # A deeply-nested installable must not be reached when depth
        # exceeds ``max_depth``.
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        targets = wi._discover_install_targets(tmp_path, max_depth=2)
        assert targets == []

    def test_root_target_does_not_self_skip_by_name(self, tmp_path: Path):
        # The walk's skip rules must NOT apply to the root itself —
        # otherwise pytest's tmp_path (basename like "test_xxx0") would
        # be filtered out and a perfectly valid repo would be considered
        # uninstallable.
        (tmp_path / "setup.py").write_text("# stub\n", encoding="utf-8")
        # Pretend tmp_path is named "test_repo" (already starts with "test"):
        targets = wi._discover_install_targets(tmp_path)
        assert len(targets) == 1
        assert targets[0][0] == "setup.py"


# ---------------------------------------------------------------------------
# ensure_worktree_installed — mocking subprocess
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _is_pip_install_e(argv: list[str]) -> bool:
    """Helper: is this argv a ``python -m pip install -e <dir>`` invocation?"""
    return (
        len(argv) >= 5
        and argv[1] == "-m"
        and argv[2] == "pip"
        and argv[3] == "install"
        and argv[4] == "-e"
    )


class TestEnsureWorktreeInstalled:
    def test_install_runs_pip_install_e(self, tmp_path: Path, monkeypatch):
        (tmp_path / "setup.py").write_text("# stub\n", encoding="utf-8")
        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            # First call is `pip show` snapshot probe — return "not installed".
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1, stdout="")
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        assert result["installed"] is True
        assert result["returncode"] == 0
        # We should see at least one ``pip install -e`` call.
        install_calls = [c for c in calls if _is_pip_install_e(c)]
        assert install_calls, f"expected pip install -e in calls: {calls}"
        # And it must target the worktree (absolute path).
        assert any(str(tmp_path) in c for c in install_calls)

    def test_install_dedup_in_process(self, tmp_path: Path, monkeypatch):
        # Calling ensure_worktree_installed twice with the same
        # (worktree, target) must only invoke the install subprocess
        # once.
        (tmp_path / "setup.py").write_text("# stub\n", encoding="utf-8")
        run_calls = {"count": 0}

        def fake_run(argv, **kwargs):
            if _is_pip_install_e(argv):
                run_calls["count"] += 1
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        wi.ensure_worktree_installed(tmp_path)
        wi.ensure_worktree_installed(tmp_path)
        assert run_calls["count"] == 1

    def test_install_force_bypasses_dedup(self, tmp_path: Path, monkeypatch):
        (tmp_path / "setup.py").write_text("# stub\n", encoding="utf-8")
        run_calls = {"count": 0}

        def fake_run(argv, **kwargs):
            if _is_pip_install_e(argv):
                run_calls["count"] += 1
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        wi.ensure_worktree_installed(tmp_path)
        wi.ensure_worktree_installed(tmp_path, force=True)
        assert run_calls["count"] == 2

    def test_install_failure_is_non_blocking(self, tmp_path: Path, monkeypatch):
        # When install exits non-zero we record stderr_tail in the
        # returned dict but DO NOT raise — the harness call must
        # proceed so the user sees the real error.
        (tmp_path / "setup.py").write_text("# stub\n", encoding="utf-8")

        def fake_run(argv, **kwargs):
            return _FakeCompleted(returncode=1, stderr="boom\nninja failed\n")

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        # No exception must escape.
        result = wi.ensure_worktree_installed(tmp_path)
        assert result["installed"] is False
        assert result["returncode"] == 1
        assert "boom" in result["stderr_tail"]
        assert "ninja failed" in result["stderr_tail"]

    def test_multi_package_monorepo_installs_every_subproject(
        self, tmp_path: Path, monkeypatch
    ):
        # The CRITICAL regression test for the sglang-style layout:
        # when a worktree contains multiple installable sub-projects
        # (e.g. ``python/`` + ``sgl-kernel/``), every one must be
        # editable-installed in a single ``ensure_worktree_installed``
        # call. Without this, kernel edits in ``sgl-kernel/`` are
        # silently bypassed by the wheel-installed binary in
        # site-packages.
        (tmp_path / "python").mkdir()
        (tmp_path / "python" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (tmp_path / "sgl-kernel").mkdir()
        (tmp_path / "sgl-kernel" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

        installed_targets: list[str] = []

        def fake_run(argv, **kwargs):
            if _is_pip_install_e(argv):
                # The 6th arg (index 5) is the target dir.
                installed_targets.append(argv[5])
            elif argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1, stdout="")
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        assert result["installed"] is True
        assert result["returncode"] == 0
        # BOTH sub-projects must have been installed.
        assert sorted(installed_targets) == sorted([
            str(tmp_path / "python"),
            str(tmp_path / "sgl-kernel"),
        ])
        # Per-target results are exposed for debugging / multi-target log.
        assert len(result["targets"]) == 2
        assert all(t["installed"] for t in result["targets"])

    def test_multi_package_partial_failure_records_first_error(
        self, tmp_path: Path, monkeypatch
    ):
        # When one sub-project's install fails, the OTHER targets must
        # still be attempted (best-effort), and the aggregated result
        # must surface the failing rc / stderr_tail prominently.
        (tmp_path / "python").mkdir()
        (tmp_path / "python" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (tmp_path / "sgl-kernel").mkdir()
        (tmp_path / "sgl-kernel" / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

        attempted: list[str] = []

        def fake_run(argv, **kwargs):
            if _is_pip_install_e(argv):
                target = argv[5]
                attempted.append(target)
                if "sgl-kernel" in target:
                    return _FakeCompleted(returncode=1, stderr="hipify exploded\n")
                return _FakeCompleted(returncode=0)
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1, stdout="")
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        # Both sub-projects were attempted (best-effort).
        assert len(attempted) == 2
        # Aggregate result reflects the failure.
        assert result["installed"] is False
        assert result["returncode"] == 1
        assert "hipify exploded" in result["stderr_tail"]
        # Per-target detail lets callers identify WHICH one failed.
        per_kernel = [t for t in result["targets"] if "sgl-kernel" in t["layout"]]
        per_python = [t for t in result["targets"] if t["layout"].startswith("python/")]
        assert per_kernel and per_kernel[0]["returncode"] == 1
        assert per_python and per_python[0]["returncode"] == 0


# ---------------------------------------------------------------------------
# Post-install verification (C-fix): pip-show location must be under target
# ---------------------------------------------------------------------------


class TestPostInstallVerification:
    """The verification step catches the silent-failure mode where pip
    reports rc=0 but a physical site-packages/<pkg>/ directory shadows
    the egg-link. Without this check, save_and_test would happily run
    the wheel binary while logging "editable-install OK" — exactly the
    bug that turned the entire round_1 sgl-kernel run into baseline
    measurements.
    """

    def test_install_succeeds_when_pip_show_points_to_target(
        self, tmp_path: Path, monkeypatch
    ):
        # Standard happy path: editable install registered, pip show
        # reports a location under the worktree → verification passes.
        worktree = tmp_path / "sgl-kernel"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub", encoding="utf-8")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(
                    returncode=0,
                    stdout=(
                        "Name: sglang-kernel\n"
                        "Version: 0.0.1\n"
                        f"Location: {worktree}\n"
                        f"Editable project location: {worktree}\n"
                    ),
                )
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        # Note: snapshot's pip-show says "this is editable" so it
        # refuses to snapshot, but the post-install verification still
        # passes (location IS under target).
        assert per["installed"] is True
        assert per["returncode"] == 0

    def test_install_demoted_when_pip_show_points_outside_target(
        self, tmp_path: Path, monkeypatch
    ):
        # pip install -e returns rc=0 but the package's pip show points
        # at site-packages — the silent-failure mode. Verification must
        # demote the install to "failed" with a clear diagnostic.
        worktree = tmp_path / "sgl-kernel"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub", encoding="utf-8")

        site_packages = "/opt/venv/lib/python3.10/site-packages"

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                # Note: NO ``Editable project location`` field — the
                # wheel install would not have it.
                return _FakeCompleted(
                    returncode=0,
                    stdout=(
                        f"Name: sglang-kernel\n"
                        f"Version: 0.0.1\n"
                        f"Location: {site_packages}\n"
                    ),
                )
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        # Tier 1 said rc=0 but verification stripped the success.
        assert per["installed"] is False
        assert per["returncode"] != 0
        assert "verification failed" in per["stderr_tail"]
        assert "site-packages" in per["stderr_tail"]
        # Aggregate result reflects the failure too.
        assert result["installed"] is False
        assert result["returncode"] == per["returncode"]

    def test_verification_skipped_when_envvar_set(
        self, tmp_path: Path, monkeypatch
    ):
        # The env-var escape hatch lets a power-user disable the check
        # (e.g. when intentionally testing the wheel path). With the
        # var set, the same shadowed-install scenario must NOT demote.
        monkeypatch.setenv("GEAK_ALLOW_WHEEL_FALLBACK", "1")
        worktree = tmp_path / "sgl-kernel"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub", encoding="utf-8")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(
                    returncode=0,
                    stdout="Name: sglang-kernel\nVersion: 0.0.1\nLocation: /opt/venv/lib\n",
                )
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        assert per["installed"] is True
        assert per["returncode"] == 0

    def test_verification_no_op_when_pip_show_returns_nothing(
        self, tmp_path: Path, monkeypatch
    ):
        # Non-standard distribution names where our heuristic guesses
        # wrong → pip show returns nothing. Don't false-positive: skip
        # the check rather than refuse the install (better to under-
        # warn than fail repos with weird naming).
        worktree = tmp_path / "weird"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub", encoding="utf-8")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        assert per["installed"] is True

    def test_verification_accepts_subdir_location(self, tmp_path: Path, monkeypatch):
        # ``setup.py develop`` typically registers the egg-link with
        # location pointing at <target>/python (not <target> itself).
        # The verification must accept any path UNDER the target dir.
        worktree = tmp_path / "sgl-kernel"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub", encoding="utf-8")
        (worktree / "python").mkdir()

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(
                    returncode=0,
                    stdout=(
                        "Name: sglang-kernel\n"
                        "Version: 0.0.1\n"
                        f"Location: {worktree / 'python'}\n"
                        f"Editable project location: {worktree / 'python'}\n"
                    ),
                )
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        assert per["installed"] is True


# ---------------------------------------------------------------------------
# Tier 2 fallback: ``setup_<arch>.py develop`` when pip install -e fails
# ---------------------------------------------------------------------------


def _is_setup_py_develop(argv: list[str]) -> bool:
    """Helper: is this argv a ``python <something>setup*.py develop`` call?"""
    return (
        len(argv) >= 3
        and argv[1].endswith(".py")
        and argv[1].rsplit("/", 1)[-1].startswith("setup")
        and argv[2] == "develop"
    )


def _is_pip_uninstall(argv: list[str]) -> bool:
    return (
        len(argv) >= 4
        and argv[1] == "-m"
        and argv[2] == "pip"
        and argv[3] == "uninstall"
    )


class TestTier2Fallback:
    """When upstream's pyproject.toml is broken (sgl-kernel ROCm being the
    canonical case), pip install -e fails with auto-discovery errors but
    a ``setup_<arch>.py develop`` invocation succeeds. The fallback must
    only fire on tier1 failure, must pick the correct arch-specific
    setup file, and must uninstall the wheel exactly once per process
    so the egg-link wins on import.
    """

    def test_tier2_fires_when_pip_fails_and_setup_rocm_exists(
        self, tmp_path: Path, monkeypatch
    ):
        # Simulate the sgl-kernel ROCm scenario: pyproject.toml + setup_rocm.py
        # both exist, pip install -e blows up (broken backend), but the
        # arch-specific setup file is there to save the day.
        kernel = tmp_path / "sgl-kernel"
        kernel.mkdir()
        (kernel / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (kernel / "setup_rocm.py").write_text("# stub\n", encoding="utf-8")

        # Force arch detection to ROCm regardless of test host's torch.
        monkeypatch.setattr(wi, "_detect_torch_arch", lambda: "rocm")

        seen: list[list[str]] = []

        def fake_run(argv, **kwargs):
            seen.append(list(argv))
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1, stdout="")  # not installed
            if _is_pip_install_e(argv):
                # Tier 1 fails — same error sgl-kernel actually produces.
                return _FakeCompleted(
                    returncode=1,
                    stderr="error: Multiple top-level packages discovered in a flat-layout\n",
                )
            if _is_pip_uninstall(argv):
                return _FakeCompleted(returncode=0)
            if _is_setup_py_develop(argv):
                # Tier 2 succeeds.
                return _FakeCompleted(returncode=0)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)

        # Final outcome: install succeeded via Tier 2.
        assert result["installed"] is True, result
        assert result["returncode"] == 0
        assert len(result["targets"]) == 1
        per = result["targets"][0]
        assert per["installed"] is True
        assert per["tier_used"] == "tier2:setup_rocm.py"
        # The argv recorded for the target must be the setup_rocm.py one.
        assert per["argv"][1] == "setup_rocm.py"
        assert "develop" in per["argv"]

        # Verify the actual call sequence: pip install -e (fails) →
        # pip uninstall (one-shot) → setup_rocm.py develop.
        install_e_calls = [c for c in seen if _is_pip_install_e(c)]
        uninstall_calls = [c for c in seen if _is_pip_uninstall(c)]
        develop_calls = [c for c in seen if _is_setup_py_develop(c)]
        assert len(install_e_calls) == 1
        assert len(uninstall_calls) == 1
        assert len(develop_calls) == 1
        # The uninstall targets the GUESSED package name (sgl-kernel → sglang-kernel).
        assert "sglang-kernel" in uninstall_calls[0]
        # And the develop subprocess must run with cwd = the sub-project dir.
        # We check by inspecting the kwargs passed into fake_run is not
        # straightforward through a mutable list, so instead verify via
        # the recorded argv: it's a relative filename, hence requires the
        # subprocess to be started from the kernel dir.
        assert develop_calls[0][1] == "setup_rocm.py"

    def test_tier2_picks_cuda_setup_on_cuda_host(self, tmp_path: Path, monkeypatch):
        # Same logic as above but verifying that arch-specific filename
        # selection is honored: a ``setup_cuda.py`` is tried on a CUDA
        # host.
        kernel = tmp_path / "kernel"
        kernel.mkdir()
        (kernel / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (kernel / "setup_cuda.py").write_text("# stub\n", encoding="utf-8")
        # ALSO ship a setup.py to confirm arch-specific wins over generic.
        (kernel / "setup.py").write_text("# stub\n", encoding="utf-8")

        monkeypatch.setattr(wi, "_detect_torch_arch", lambda: "cuda")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1)
            if _is_pip_install_e(argv):
                return _FakeCompleted(returncode=1, stderr="boom\n")
            if _is_pip_uninstall(argv):
                return _FakeCompleted(returncode=0)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)

        # Tier 2 fired and picked the CUDA-specific file.
        assert result["installed"] is True, result
        per = result["targets"][0]
        # NOTE: tier2 was reached because tier1 (pip install -e on root)
        # failed; root has neither setup nor pyproject so the discovered
        # target must be ``kernel/``. Discovery does walk into sub-dirs.
        assert per["tier_used"] == "tier2:setup_cuda.py", per
        assert per["argv"][1] == "setup_cuda.py"

    def test_tier2_falls_back_to_plain_setup_py(self, tmp_path: Path, monkeypatch):
        # When the arch-specific filename does NOT exist but a plain
        # ``setup.py`` does, Tier 2 should use it. Picks up legacy repos
        # that have a single setup.py for all platforms.
        kernel = tmp_path / "kernel"
        kernel.mkdir()
        (kernel / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (kernel / "setup.py").write_text("# stub\n", encoding="utf-8")

        monkeypatch.setattr(wi, "_detect_torch_arch", lambda: "rocm")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1)
            if _is_pip_install_e(argv):
                return _FakeCompleted(returncode=1, stderr="boom\n")
            if _is_pip_uninstall(argv):
                return _FakeCompleted(returncode=0)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        assert per["installed"] is True
        assert per["tier_used"] == "tier2:setup.py"
        assert per["argv"][1] == "setup.py"

    def test_tier2_skipped_when_pip_succeeds(self, tmp_path: Path, monkeypatch):
        # The fallback exists for emergencies — when tier1 already wins
        # we must NOT spend time on tier2 even if a setup_rocm.py is
        # present.
        kernel = tmp_path / "sgl-kernel"
        kernel.mkdir()
        (kernel / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (kernel / "setup_rocm.py").write_text("# stub\n", encoding="utf-8")

        monkeypatch.setattr(wi, "_detect_torch_arch", lambda: "rocm")
        seen: list[list[str]] = []

        def fake_run(argv, **kwargs):
            seen.append(list(argv))
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        assert per["installed"] is True
        assert per["tier_used"] == "tier1:pip"
        # Crucially, neither the uninstall nor the develop fallback ran.
        assert not any(_is_pip_uninstall(c) for c in seen)
        assert not any(_is_setup_py_develop(c) for c in seen)

    def test_tier2_skipped_when_no_setup_arch_file(self, tmp_path: Path, monkeypatch):
        # If pip install -e fails AND there's no setup_<arch>.py / setup.py
        # to fall back on, we must surface tier1's failure unchanged
        # (no false-positive "installed" claim).
        kernel = tmp_path / "kernel"
        kernel.mkdir()
        # Only a pyproject.toml — no setup files at all.
        (kernel / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

        monkeypatch.setattr(wi, "_detect_torch_arch", lambda: "rocm")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1)
            if _is_pip_install_e(argv):
                return _FakeCompleted(returncode=1, stderr="kaboom\n")
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        result = wi.ensure_worktree_installed(tmp_path)
        per = result["targets"][0]
        assert per["installed"] is False
        assert per["tier_used"] == "tier1:pip"
        assert per["returncode"] == 1
        assert "kaboom" in per["stderr_tail"]

    def test_tier2_uninstall_is_one_shot_per_pkg(self, tmp_path: Path, monkeypatch):
        # Two consecutive ensure_worktree_installed(force=True) calls on
        # the same worktree where tier1 keeps failing must only run
        # ``pip uninstall`` ONCE for the same distribution name (the
        # second develop call just overwrites the egg-link).
        kernel = tmp_path / "sgl-kernel"
        kernel.mkdir()
        (kernel / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (kernel / "setup_rocm.py").write_text("# stub\n", encoding="utf-8")

        monkeypatch.setattr(wi, "_detect_torch_arch", lambda: "rocm")
        uninstall_count = {"n": 0}
        develop_count = {"n": 0}

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1)
            if _is_pip_install_e(argv):
                return _FakeCompleted(returncode=1, stderr="boom\n")
            if _is_pip_uninstall(argv):
                uninstall_count["n"] += 1
                return _FakeCompleted(returncode=0)
            if _is_setup_py_develop(argv):
                develop_count["n"] += 1
                return _FakeCompleted(returncode=0)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        wi.ensure_worktree_installed(tmp_path)
        wi.ensure_worktree_installed(tmp_path, force=True)

        # Two develop calls (one per ensure_*), but only ONE uninstall.
        assert develop_count["n"] == 2
        assert uninstall_count["n"] == 1


class TestArchDetection:
    def test_returns_one_of_known_strings(self):
        # The detector must always return a string from the keys of the
        # arch→setup-name table, otherwise ``_find_arch_setup_py`` would
        # silently fall through to "cpu" defaults.
        arch = wi._detect_torch_arch()
        assert arch in {"rocm", "cuda", "musa", "cpu"}

    def test_find_arch_setup_py_prefers_arch_specific(self, tmp_path: Path):
        # Both setup_rocm.py and setup.py present → ROCm wins on rocm host.
        (tmp_path / "setup_rocm.py").write_text("# stub", encoding="utf-8")
        (tmp_path / "setup.py").write_text("# stub", encoding="utf-8")
        chosen = wi._find_arch_setup_py(tmp_path, "rocm")
        assert chosen is not None and chosen.name == "setup_rocm.py"

    def test_find_arch_setup_py_falls_back_to_plain_setup(self, tmp_path: Path):
        # Only setup.py → use it for any arch.
        (tmp_path / "setup.py").write_text("# stub", encoding="utf-8")
        for arch in ("rocm", "cuda", "cpu", "musa"):
            chosen = wi._find_arch_setup_py(tmp_path, arch)
            assert chosen is not None and chosen.name == "setup.py", arch

    def test_find_arch_setup_py_returns_none_when_absent(self, tmp_path: Path):
        # Empty dir → nothing to fall back to.
        assert wi._find_arch_setup_py(tmp_path, "rocm") is None


# ---------------------------------------------------------------------------
# Snapshot semantics
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_records_pip_show_output(self, tmp_path: Path, monkeypatch):
        # Use a real-looking worktree name (aiter) so the snapshot dir
        # name matches the normalised package name produced by
        # ``_guess_package_name``.
        worktree = tmp_path / "aiter"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub\n", encoding="utf-8")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(
                    returncode=0,
                    stdout="Name: aiter\nVersion: 0.1.5\nLocation: /opt/venv/lib\n",
                )
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        wi.ensure_worktree_installed(worktree)
        snap = wi._SNAPSHOT_ROOT / "aiter" / "info.json"
        assert snap.is_file(), "snapshot info.json should have been written"
        data = json.loads(snap.read_text())
        assert data["name"] == "aiter"
        assert data["version"] == "0.1.5"

    def test_snapshot_skips_existing(self, tmp_path: Path, monkeypatch):
        # Pre-create a snapshot that says aiter==9.9.9 (representing a
        # prior crashed-run's record of the real original).  A new
        # ensure_worktree_installed call must NOT overwrite it, even
        # though pip show would now report a different version.
        worktree = tmp_path / "aiter"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub\n", encoding="utf-8")
        snap_dir = wi._SNAPSHOT_ROOT / "aiter"
        snap_dir.mkdir(parents=True)
        (snap_dir / "info.json").write_text(
            json.dumps({"name": "aiter", "version": "9.9.9", "location": "/orig"}),
            encoding="utf-8",
        )

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                # This would be the "currently editable" view; must NOT
                # overwrite the genuine original.
                return _FakeCompleted(
                    returncode=0,
                    stdout="Name: aiter\nVersion: 0.0.0\nLocation: /tmp/worktree\nEditable project location: /tmp/worktree\n",
                )
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        wi.ensure_worktree_installed(worktree)
        # Original snapshot must be preserved verbatim.
        data = json.loads((snap_dir / "info.json").read_text())
        assert data["version"] == "9.9.9"

    def test_snapshot_refuses_editable_install(self, tmp_path: Path, monkeypatch):
        # When the only thing pip show reports is an editable install,
        # we can't snapshot it as "original" — that would record the
        # worktree itself as the restore target.  Refuse and log.
        worktree = tmp_path / "aiter"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub\n", encoding="utf-8")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(
                    returncode=0,
                    stdout="Name: aiter\nVersion: 0.0.0\nLocation: /tmp/worktree\n"
                           "Editable project location: /tmp/worktree\n",
                )
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        wi.ensure_worktree_installed(worktree)
        snap = wi._SNAPSHOT_ROOT / "aiter" / "info.json"
        assert not snap.exists()

    def test_snapshot_skips_unknown_package(self, tmp_path: Path, monkeypatch):
        # When pip show returns rc!=0 (package not installed), no
        # snapshot is recorded — there's nothing to restore.
        worktree = tmp_path / "aiter"
        worktree.mkdir()
        (worktree / "setup.py").write_text("# stub\n", encoding="utf-8")

        def fake_run(argv, **kwargs):
            if argv[:4] == [argv[0], "-m", "pip", "show"]:
                return _FakeCompleted(returncode=1)
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        wi.ensure_worktree_installed(worktree)
        assert not (wi._SNAPSHOT_ROOT / "aiter" / "info.json").exists()


# ---------------------------------------------------------------------------
# restore_original_packages
# ---------------------------------------------------------------------------


class TestRestore:
    def test_restore_uninstalls_and_reinstalls(self, tmp_path: Path, monkeypatch):
        # Pre-seed a snapshot.
        snap_dir = wi._SNAPSHOT_ROOT / "aiter"
        snap_dir.mkdir(parents=True)
        (snap_dir / "info.json").write_text(
            json.dumps({"name": "aiter", "version": "0.1.5"}),
            encoding="utf-8",
        )

        calls: list[list[str]] = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        outcome = wi.restore_original_packages()
        assert outcome == {"restored": ["aiter"], "failed": []}
        # Both uninstall and reinstall must have been called.
        assert any("uninstall" in c for c in calls), calls
        assert any("install" in c and "--force-reinstall" in c for c in calls), calls
        # Snapshot cleaned up after successful restore.
        assert not snap_dir.exists()

    def test_restore_reports_failure_when_reinstall_fails(self, tmp_path: Path, monkeypatch):
        snap_dir = wi._SNAPSHOT_ROOT / "aiter"
        snap_dir.mkdir(parents=True)
        (snap_dir / "info.json").write_text(
            json.dumps({"name": "aiter", "version": "0.1.5"}),
            encoding="utf-8",
        )

        def fake_run(argv, **kwargs):
            # uninstall succeeds, reinstall fails (e.g. PyPI unreachable)
            if "--force-reinstall" in argv:
                return _FakeCompleted(returncode=1, stderr="No matching distribution\n")
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        outcome = wi.restore_original_packages()
        assert outcome == {"restored": [], "failed": ["aiter"]}
        # Snapshot left in place on failure so a manual retry has the info.
        assert snap_dir.exists()

    def test_restore_no_snapshot_root_is_noop(self, tmp_path: Path, monkeypatch):
        # restore must not error when the backup root doesn't exist
        # (clean-install case — no editable was ever performed).
        monkeypatch.setattr(wi, "_SNAPSHOT_ROOT", tmp_path / "definitely-missing")
        outcome = wi.restore_original_packages()
        assert outcome == {"restored": [], "failed": []}

    def test_restore_ignores_unreadable_info_json(self, tmp_path: Path, monkeypatch):
        snap_dir = wi._SNAPSHOT_ROOT / "broken"
        snap_dir.mkdir(parents=True)
        (snap_dir / "info.json").write_text("not json", encoding="utf-8")

        def fake_run(argv, **kwargs):
            return _FakeCompleted(returncode=0)

        monkeypatch.setattr(wi.subprocess, "run", fake_run)
        outcome = wi.restore_original_packages()
        # Bad snapshot is skipped, doesn't blow up.
        assert outcome == {"restored": [], "failed": []}
