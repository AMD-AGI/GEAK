"""Every tracked test file is either run by CI or exempt WITH A REASON.

CI's pytest step is an explicit allowlist of file paths, not a directory scan. That is a deliberate
choice -- some `test_*.py` in this tree are task fixtures that need a GPU, a built kernel or a
working directory CI does not have, and a bare `pytest` would collect them and fail. The cost is
that the allowlist is invisible: adding a test file is the normal way to add a test, nothing points
out that it was never wired in, and the file passes locally forever while contributing nothing.

Measured: when this check was written, 7 test files holding 66 passing tests were in no CI list at
all -- including `test_check_flydsl_compat.py`, whose sibling `test_upstream_knobs.py` had just
gained five new cases that CI would never have executed.

So the same rule the knowledge base already holds itself to applies here: a missing entry must be
an explicit exemption with a stated reason, never a blank. A blank reads as "nobody looked".

This file is itself in the allowlist it guards.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / ".github" / "workflows"

# Tracked test files CI deliberately does not run. Key: repo-relative path. Value: why.
# A file listed here must still exist -- a stale exemption is how a real gap hides.
EXEMPT = {
    "examples/tasks/knn/test_knn.py":
        "Task fixture, not a repo test: it is the oracle shipped with the knn example and reads "
        "`src/knn_cuda.hip` relative to its own task directory, so collection raises "
        "FileNotFoundError from any other cwd. It is exercised by running that example, not by "
        "the unit suite.",
}


def tracked_test_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted(f for f in out if re.search(r"(^|/)test_[^/]*\.py$", f))


def ci_text():
    assert WORKFLOWS.is_dir(), f"no workflow directory at {WORKFLOWS}"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(WORKFLOWS.glob("*.yml")))


def test_every_tracked_test_file_is_run_or_exempt():
    ci = ci_text()
    unwired = [f for f in tracked_test_files() if f not in ci and f not in EXEMPT]
    assert not unwired, (
        "these test files are tracked by git but appear in no workflow in .github/workflows/, so "
        "CI never runs them:\n  " + "\n  ".join(unwired)
        + "\n\nAdd each to the pytest allowlist in ci-l0-checks.yml (or, if it asserts at import "
          "time and exits non-zero, to the standalone contract-check step), or add it to EXEMPT in "
          "this file with the reason it cannot run in CI."
    )


def test_no_stale_exemptions():
    """An exemption for a file that no longer exists silently widens the next real gap."""
    gone = [f for f in EXEMPT if not (REPO / f).exists()]
    assert not gone, ("EXEMPT names files that are not in the tree any more; drop them:\n  "
                      + "\n  ".join(gone))


def test_exemptions_are_not_also_wired():
    """A file that is both exempt and in CI means the stated reason is wrong or obsolete."""
    ci = ci_text()
    both = [f for f in EXEMPT if f in ci]
    assert not both, ("these files are exempt AND referenced by a workflow -- remove the "
                      "exemption:\n  " + "\n  ".join(both))


def test_exemptions_state_a_reason():
    thin = [f for f, why in EXEMPT.items() if len(why.strip()) < 40]
    assert not thin, ("an exemption needs a reason someone can check, not a placeholder:\n  "
                      + "\n  ".join(thin))


def test_ci_does_not_invoke_pytest_as_a_module_while_unittest_py_is_tracked():
    """`python -m pytest` + a root-level `unittest.py` = stdlib `unittest` is shadowed.

    `-m` puts the invocation directory on `sys.path[0]`. This repo tracks `unittest.py` at the
    root -- the task-harness entry-point convention, which the oracle task dirs share, so it is
    not renameable on a whim. Any import of `unittest.mock` under that shadow dies with
    "No module named 'unittest.mock'; 'unittest' is not a package": pytest's own startup on some
    builds, and torch unconditionally (`torch/_guards.py`). The `pytest` console script does not
    prepend cwd and is unaffected, so the fix is the invocation, not the filename.

    Reproduced on box 2026-10-13 from the repo root: `python -m pytest` failed before collection;
    `pytest` passed. The L0 job is green on `-m` today only because its pytest build happens not
    to take that import path and torch is deliberately not installed there -- it would start
    failing the moment either changes, which is exactly the kind of latent break a contract
    check exists to catch.
    """
    if not (REPO / "unittest.py").exists():
        pytest.skip("no root-level unittest.py to shadow stdlib unittest")
    offenders = []
    for p in sorted(WORKFLOWS.glob("*.yml")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # a comment explaining the rule is not a violation of it
            if re.search(r"\bpython[0-9.]*\s+-m\s+pytest\b", line):
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "a workflow invokes pytest as a module while `unittest.py` is tracked at the repo root, "
        "which shadows stdlib `unittest` for everything the run imports:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse the `pytest` console script instead -- it does not put cwd on sys.path."
    )


if __name__ == "__main__":  # convenience; CI runs this through pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
