"""Run the script-style contract checks under pytest, so root `pytest` covers them.

Four of these files are ignored by `conftest.py` because their module-level `sys.exit`
aborts pytest collection (see that file). Ignoring them alone would turn a loud failure
into a quiet coverage hole, so they are executed here the way they are meant to be run --
as programs, asserting a zero exit.

This mirrors the `KB and workflow contract checks` step in
`.github/workflows/ci-l0-checks.yml`, which stays the authoritative gate: it invokes each
program directly, so it still fails loudly if pytest itself cannot start.

It is in CI's pytest allowlist anyway, because `test_script_style_ignores_are_covered_here`
guards conftest/wrapper drift and has no equivalent in the direct invocations -- otherwise
the one check that notices a silently-dropped contract check is the one CI never runs. The
six subprocesses cost ~5s.

The two lists are synced by hand: they are short and rarely change, and parsing the
workflow YAML instead would make a green run depend on the shape of a CI file.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def root_conftest():
    """The repo-root conftest, loaded by path rather than by name.

    `perf_knowledge/corpus/conftest.py` shares the basename and neither directory is a
    package, so in a run collecting both trees whichever pytest imported last owns
    `sys.modules["conftest"]`. The tests below mean the ROOT one, and failed with
    `AttributeError: ... has no attribute 'SCRIPT_STYLE'` in exactly the combined run CI
    performs while passing when `ci/` ran alone.
    """
    spec = importlib.util.spec_from_file_location("_root_conftest", REPO / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# The programs CI runs under "KB and workflow contract checks".
CONTRACT_CHECKS = [
    "kernel_workflow/scripts/test_kb.py",
    "kernel_workflow/scripts/test_lane_refs.py",
    "kernel_workflow/scripts/test_kb_switch.py",
    "kernel_workflow/scripts/test_knowledge_reachable.py",
    "perf_knowledge/test_docs_contract.py",
    "e2e_workflow/scripts/tests/test_e2e_lane_defaults.py",
    "e2e_workflow/scripts/tests/test_kb_identity_flags.py",
]

# The generated-artifact gates from the same CI step: the KB must lint clean and its
# index must match the cards it is generated from.
KB_DIR = "kernel_workflow/knowledge/learned"
KB_GATES = [
    ["kernel_workflow/scripts/kb.py", "--kb-dir", KB_DIR, "lint", "--cards"],
    ["kernel_workflow/scripts/kb.py", "--kb-dir", KB_DIR, "index", "--check"],
]


def _run(argv):
    return subprocess.run(
        [sys.executable, *argv], cwd=REPO, capture_output=True, text=True, timeout=600
    )


def _explain(proc, what):
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
    hint = ""
    if "'unittest' is not a package" in (proc.stderr or ""):
        # A trailing ':' in PYTHONPATH adds an empty entry, which puts the invocation
        # directory on sys.path, and this repo's root `unittest.py` then shadows the
        # stdlib module. That is an environment fault, not a repo one -- it has been
        # mistaken for a repo regression before. Re-run with `env -u PYTHONPATH`.
        hint = (
            "\n\nHINT: stdlib `unittest` is being shadowed. Check PYTHONPATH for a "
            "trailing ':' (an empty entry puts the cwd on sys.path, and this repo has a "
            "root `unittest.py`). Re-run with `env -u PYTHONPATH pytest ...`."
        )
    return f"{what} exited {proc.returncode}\n" + "\n".join(tail) + hint


@pytest.mark.parametrize("script", CONTRACT_CHECKS)
def test_contract_check_exits_zero(script):
    assert (REPO / script).is_file(), f"{script} is listed here but does not exist"
    proc = _run([script])
    assert proc.returncode == 0, _explain(proc, script)


@pytest.mark.parametrize("argv", KB_GATES, ids=["kb_lint_cards", "kb_index_check"])
def test_kb_gate_exits_zero(argv):
    proc = _run(argv)
    assert proc.returncode == 0, _explain(proc, " ".join(argv[1:]))


def test_script_style_ignores_are_covered_here():
    """Every script-style file conftest.py hides from collection must be run by this module.

    Without this, dropping a script from CONTRACT_CHECKS while leaving it in
    `SCRIPT_STYLE` would remove it from the suite with nothing to say so -- the exact
    silent hole the ignore list was written to avoid.

    `conftest.EXAMPLE_TASKS` is deliberately NOT asserted on: those need a GPU and belong
    behind `gpu_lock.sh`, so being ignored and unrun is their correct state.
    """
    conftest = root_conftest()

    missing = sorted(set(conftest.SCRIPT_STYLE) - set(CONTRACT_CHECKS))
    assert not missing, (
        "conftest.py ignores these script-style files but nothing runs them: "
        + ", ".join(missing)
    )


def test_example_task_ignores_agree_with_the_ci_inventory():
    """`conftest.EXAMPLE_TASKS` and the inventory's EXEMPT name the same files.

    Both say "this test file is not part of the unit suite", for the same reason, in two
    places: conftest so local collection skips it, the inventory so CI's allowlist gate
    accepts its absence. Nothing linked them, so exempting a file in one and not the
    other would leave either a collection crash or a gate failure for someone else to
    discover.
    """
    conftest = root_conftest()

    from test_ci_test_inventory import EXEMPT

    unexempt = sorted(set(conftest.EXAMPLE_TASKS) - set(EXEMPT))
    assert not unexempt, (
        "conftest.py skips these but ci/test_ci_test_inventory.py does not exempt them, "
        "so CI's allowlist gate will fail on them: " + ", ".join(unexempt)
    )
