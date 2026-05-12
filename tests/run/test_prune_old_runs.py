"""Tests for ``prune_old_runs`` -- the ``--keep-runs`` retention helper.

Key invariants covered:

- Only auto-generated ``<kernel>_<YYYYmmdd_HHMMSS>`` dirs are touched.
  Real kernel names contain ``.``, ``-``, and uppercase, so the regex
  must match those too.
- The stale filter uses ``geak_agent.log`` mtime first so a long-running
  but actively-logging sibling looks fresh to a concurrent ``--keep-runs``
  from another geak invocation.
- Returns 0 cleanly when the parent directory doesn't exist (or isn't a
  directory).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from minisweagent.run.postprocess.finalize_apply import prune_old_runs
from minisweagent.utils.log import DEFAULT_LOG_FILENAME

# Stale threshold tighter than the default 600s so tests run fast.
_STALE = 0.5


def _set_mtime(p: Path, age_s: float) -> None:
    """Backdate `p`'s mtime by `age_s` seconds."""
    now = time.time()
    os.utime(p, (now - age_s, now - age_s))


def test_prune_old_runs_keeps_most_recent_n_and_skips_exclude(tmp_path: Path) -> None:
    """Happy path: ``exclude`` is removed from the candidate pool first, then
    we keep the N most recent of what remains.

    Setup: 4 stale matching dirs. exclude=newest. So candidates = the other
    three. keep=2 means we keep the two most recent of those three, prune
    the oldest one. The excluded newest survives regardless.
    """
    parent = tmp_path
    names = [
        "kernel_20260101_120000",  # oldest
        "kernel_20260101_120100",
        "kernel_20260101_120200",
        "kernel_20260101_120300",  # newest
    ]
    dirs = [parent / n for n in names]
    for i, d in enumerate(dirs):
        d.mkdir()
        # Older index -> larger age so newest dir has the freshest mtime.
        _set_mtime(d, age_s=10.0 + (3 - i))

    removed = prune_old_runs(parent, keep=2, exclude=dirs[-1], stale_after_s=_STALE)
    # candidates (after exclusion) = [dirs[2], dirs[1], dirs[0]] newest-first.
    # keep=2 keeps dirs[2] and dirs[1]; prunes dirs[0].
    assert removed == 1
    surviving = sorted(d.name for d in parent.iterdir())
    assert surviving == sorted([dirs[-1].name, dirs[-2].name, dirs[-3].name])
    assert not dirs[0].exists()


def test_prune_old_runs_exclude_is_protected_even_when_oldest(tmp_path: Path) -> None:
    """exclude= must survive even if it's the oldest dir."""
    parent = tmp_path
    old = parent / "kernel_20260101_120000"
    mid = parent / "kernel_20260101_120100"
    new = parent / "kernel_20260101_120200"
    for d in (old, mid, new):
        d.mkdir()
    _set_mtime(old, age_s=30)
    _set_mtime(mid, age_s=20)
    _set_mtime(new, age_s=10)

    removed = prune_old_runs(parent, keep=1, exclude=old, stale_after_s=_STALE)
    # keep=1 means only ``new`` would be kept; but ``old`` is excluded so
    # it survives anyway. Only ``mid`` is pruned.
    assert removed == 1
    assert old.is_dir()
    assert new.is_dir()
    assert not mid.exists()


def test_prune_old_runs_matches_real_kernel_names(tmp_path: Path) -> None:
    """The regex must match kernel names containing `.`, `-`, uppercase."""
    parent = tmp_path
    matching = [
        "flash_attn_v2_20260101_120000",
        "Topk-Triton.kernel_20260101_120030",
        "kernel_20260101_120100",
    ]
    not_matching = [
        "random_dir",
        "logs",
        "kernel_20260101",  # missing time
        "kernel_2026.01.01_120000",  # wrong date format
    ]
    for n in matching:
        d = parent / n
        d.mkdir()
        _set_mtime(d, age_s=30)
    for n in not_matching:
        if n == "logs":
            (parent / n).mkdir()
            _set_mtime(parent / n, age_s=30)
            continue
        if "." in n or n == "kernel_20260101":
            d = parent / n
            d.mkdir()
            _set_mtime(d, age_s=30)
        else:
            d = parent / n
            d.mkdir()
            _set_mtime(d, age_s=30)
    # A bare file -- must never be touched.
    (parent / "notes.txt").write_text("free-form notes")
    _set_mtime(parent / "notes.txt", age_s=30)

    removed = prune_old_runs(parent, keep=0, exclude=None, stale_after_s=_STALE)

    # All three timestamped dirs are pruned.
    assert removed == 3
    for n in matching:
        assert not (parent / n).exists(), n
    # Everything else stays.
    for n in not_matching:
        assert (parent / n).exists(), n
    assert (parent / "notes.txt").is_file()


def test_prune_old_runs_stale_filter_uses_log_mtime(tmp_path: Path) -> None:
    """Three sub-cases for the freshness key:
    1. Stale dir mtime but ``geak_agent.log`` was just touched -> NOT pruned.
    2. Stale dir mtime, no log, but a non-log child was just touched -> NOT pruned.
    3. Stale dir mtime, no recent children at all -> pruned (last-resort fallback).
    """
    parent = tmp_path

    # Case 1: dir mtime backdated; log file fresh.
    case1 = parent / "kernel_20260101_120000"
    case1.mkdir()
    log = case1 / DEFAULT_LOG_FILENAME
    log.write_text("agent log")
    _set_mtime(log, age_s=0.0)  # fresh now
    _set_mtime(case1, age_s=3600)  # stale dir mtime

    # Case 2: no log; non-log child fresh.
    case2 = parent / "kernel_20260101_120100"
    case2.mkdir()
    other = case2 / "best_patch.diff"
    other.write_text("diff content")
    _set_mtime(other, age_s=0.0)
    _set_mtime(case2, age_s=3600)

    # Case 3: stale, no recent children.
    case3 = parent / "kernel_20260101_120200"
    case3.mkdir()
    (case3 / "final_report.json").write_text("{}")
    _set_mtime(case3 / "final_report.json", age_s=3600)
    _set_mtime(case3, age_s=3600)

    removed = prune_old_runs(parent, keep=0, exclude=None, stale_after_s=600.0)

    # Only case 3 was stale by every freshness key.
    assert removed == 1
    assert case1.is_dir(), "case 1 (fresh log) must not be pruned"
    assert case2.is_dir(), "case 2 (fresh non-log child) must not be pruned"
    assert not case3.exists(), "case 3 (everything stale) must be pruned"


def test_prune_old_runs_returns_zero_when_parent_missing(tmp_path: Path) -> None:
    """Nonexistent parent returns 0 cleanly without raising."""
    assert prune_old_runs(tmp_path / "does_not_exist", keep=1) == 0


def test_prune_old_runs_returns_zero_when_parent_is_a_file(tmp_path: Path) -> None:
    """A file in place of a directory returns 0, not raise."""
    f = tmp_path / "not_a_dir"
    f.write_text("oops")
    assert prune_old_runs(f, keep=1) == 0


def test_prune_old_runs_no_op_when_nothing_matches_regex(tmp_path: Path) -> None:
    """Empty parent (no matching dirs) returns 0 even with keep=0."""
    (tmp_path / "logs").mkdir()
    _set_mtime(tmp_path / "logs", age_s=3600)
    assert prune_old_runs(tmp_path, keep=0, stale_after_s=_STALE) == 0
    assert (tmp_path / "logs").is_dir()


def test_prune_old_runs_negative_keep_treated_as_zero(tmp_path: Path) -> None:
    """Defensive: negative keep behaves like keep=0 (prune everything stale)."""
    d = tmp_path / "kernel_20260101_120000"
    d.mkdir()
    _set_mtime(d, age_s=3600)
    assert prune_old_runs(tmp_path, keep=-3, stale_after_s=_STALE) == 1
    assert not d.exists()
