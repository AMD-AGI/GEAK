"""Tests for ``minisweagent.run.preprocess_v3.clone``.

URL canonicalisation is exercised exhaustively via parametrisation (pure
function, no I/O). The clone path mocks out ``subprocess.run`` so the
suite stays fully offline — the only network-touching test is marked
``pytest.mark.network`` and is skipped by default. The split helper is
covered with a small fake repo tree under ``tmp_path``.
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from unittest import mock

import pytest


def _network_available(host: str = "github.com", port: int = 443, timeout: float = 2.0) -> bool:
    """Return True when ``host:port`` is reachable (used by the opt-in network test)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


from minisweagent.run.preprocess_v3 import clone as clone_mod
from minisweagent.run.preprocess_v3.clone import (
    CloneError,
    clone_repo,
    resolve_repo_url,
    split_repo_for_baseline_and_eval,
)

# ---------------------------------------------------------------------------
# resolve_repo_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AMD-AGI/GEAK", "https://github.com/AMD-AGI/GEAK.git"),
        ("org/repo", "https://github.com/org/repo.git"),
        ("https://github.com/org/repo", "https://github.com/org/repo.git"),
        ("https://github.com/org/repo.git", "https://github.com/org/repo.git"),
        ("https://github.com/org/repo/", "https://github.com/org/repo.git"),
        ("https://github.com/org/repo/tree/main", "https://github.com/org/repo.git"),
        ("git@github.com:org/repo.git", "https://github.com/org/repo.git"),
        ("git@github.com:org/repo", "https://github.com/org/repo.git"),
        ("  org/repo  ", "https://github.com/org/repo.git"),
    ],
)
def test_resolve_repo_url_canonicalises(raw: str, expected: str) -> None:
    assert resolve_repo_url(raw) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "not-a-repo",
        "https://gitlab.com/org/repo",
        "https://github.com/just-an-owner",
        "ftp://github.com/org/repo",
    ],
)
def test_resolve_repo_url_rejects_unsupported(bad: str) -> None:
    with pytest.raises(ValueError):
        resolve_repo_url(bad)


def test_resolve_repo_url_rejects_none() -> None:
    with pytest.raises(ValueError):
        resolve_repo_url(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# clone_repo — pure command-construction layer, subprocess.run mocked out
# ---------------------------------------------------------------------------


def _ok_proc(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _fail_proc(stderr: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr=stderr)


def test_clone_repo_invokes_git_clone(tmp_path: Path) -> None:
    """Happy-path: ``git clone <url> <dest>`` is invoked with the expected args."""
    dest = tmp_path / "checkout"
    captured_calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        captured_calls.append(list(cmd))
        # Simulate git creating the destination directory.
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return _ok_proc()

    with mock.patch.object(clone_mod, "_run_git", side_effect=_fake_run):
        result = clone_repo("https://github.com/org/repo.git", dest)

    assert result == dest.resolve()
    assert captured_calls == [["git", "clone", "https://github.com/org/repo.git", str(dest)]]


def test_clone_repo_invokes_checkout_when_ref_given(tmp_path: Path) -> None:
    """When ``ref`` is set, a follow-up ``git -C <dest> checkout <ref>`` runs."""
    dest = tmp_path / "checkout"
    captured_calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        captured_calls.append(list(cmd))
        if cmd[1] == "clone":
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return _ok_proc()

    with mock.patch.object(clone_mod, "_run_git", side_effect=_fake_run):
        clone_repo("https://github.com/org/repo.git", dest, ref="v1.2.3")

    assert captured_calls == [
        ["git", "clone", "https://github.com/org/repo.git", str(dest)],
        ["git", "-C", str(dest), "checkout", "v1.2.3"],
    ]


def test_clone_repo_raises_when_dest_already_exists(tmp_path: Path) -> None:
    dest = tmp_path / "existing"
    dest.mkdir()

    with pytest.raises(CloneError, match="destination already exists"):
        clone_repo("https://github.com/org/repo.git", dest)


def test_clone_repo_raises_clone_error_on_git_failure(tmp_path: Path) -> None:
    dest = tmp_path / "checkout"

    with mock.patch.object(clone_mod, "_run_git", return_value=_fail_proc("fatal: repo not found")):
        with pytest.raises(CloneError, match="git clone failed"):
            clone_repo("https://github.com/org/missing.git", dest)

    assert not dest.exists(), "failed clone should leave no partial directory"


def test_clone_repo_raises_clone_error_on_checkout_failure(tmp_path: Path) -> None:
    dest = tmp_path / "checkout"

    def _fake_run(cmd, **_kwargs):
        if cmd[1] == "clone":
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
            return _ok_proc()
        return _fail_proc("error: pathspec 'nonexistent' did not match any file(s)")

    with mock.patch.object(clone_mod, "_run_git", side_effect=_fake_run):
        with pytest.raises(CloneError, match="git checkout"):
            clone_repo("https://github.com/org/repo.git", dest, ref="nonexistent")


# ---------------------------------------------------------------------------
# split_repo_for_baseline_and_eval
# ---------------------------------------------------------------------------


def _make_fake_repo(root: Path) -> Path:
    """Build a tiny repo-like tree with a fake ``.git`` directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.py").write_text("# kernel source\n", encoding="utf-8")
    (root / "README.md").write_text("# repo\n", encoding="utf-8")
    sub = root / "src" / "pkg"
    sub.mkdir(parents=True)
    (sub / "module.py").write_text("def f(): return 1\n", encoding="utf-8")
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return root


def test_split_creates_baseline_and_eval_with_identical_contents(tmp_path: Path) -> None:
    clone = _make_fake_repo(tmp_path / "clone")
    work_root = tmp_path / "work"

    baseline, eval_dir = split_repo_for_baseline_and_eval(clone, work_root)

    assert baseline == (work_root / "baseline").resolve()
    assert eval_dir == (work_root / "eval").resolve()
    assert baseline.is_dir()
    assert eval_dir.is_dir()

    for tree in (baseline, eval_dir):
        assert (tree / "kernel.py").read_text() == "# kernel source\n"
        assert (tree / "README.md").read_text() == "# repo\n"
        assert (tree / "src" / "pkg" / "module.py").read_text() == "def f(): return 1\n"


def test_split_omits_git_metadata(tmp_path: Path) -> None:
    """The ``.git`` directory is intentionally not copied into baseline/eval."""
    clone = _make_fake_repo(tmp_path / "clone")
    work_root = tmp_path / "work"

    baseline, eval_dir = split_repo_for_baseline_and_eval(clone, work_root)

    assert not (baseline / ".git").exists()
    assert not (eval_dir / ".git").exists()


def test_split_is_idempotent_across_repeated_runs(tmp_path: Path) -> None:
    """A stale ``baseline/`` or ``eval/`` is replaced cleanly on a repeat call."""
    clone = _make_fake_repo(tmp_path / "clone")
    work_root = tmp_path / "work"

    baseline, eval_dir = split_repo_for_baseline_and_eval(clone, work_root)
    # Drop a stale file inside the eval tree, then re-split — the stale
    # file should be gone after the second call.
    (eval_dir / "stale.txt").write_text("garbage\n", encoding="utf-8")

    baseline2, eval2 = split_repo_for_baseline_and_eval(clone, work_root)

    assert baseline2 == baseline
    assert eval2 == eval_dir
    assert not (eval2 / "stale.txt").exists()


def test_split_raises_for_missing_clone(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        split_repo_for_baseline_and_eval(tmp_path / "nope", tmp_path / "work")


# ---------------------------------------------------------------------------
# Network integration test — opt-in
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("GEAK_TEST_NETWORK", "").strip().lower() not in {"1", "true", "yes"},
    reason="Opt-in network test; set GEAK_TEST_NETWORK=1 to run.",
)
def test_clone_repo_real_network(tmp_path: Path) -> None:  # pragma: no cover
    """End-to-end clone against a real public repo. Off by default.

    Gated by both ``GEAK_TEST_NETWORK=1`` (opt-in) AND a live socket
    probe against github.com:443, so the test never produces a spurious
    failure on sandboxed CI without the env var.
    """
    if not _network_available():
        pytest.skip("github.com:443 not reachable; skipping live clone test")
    dest = tmp_path / "checkout"
    clone_repo("https://github.com/octocat/Hello-World.git", dest)
    assert (dest / "README").exists() or (dest / "README.md").exists()
