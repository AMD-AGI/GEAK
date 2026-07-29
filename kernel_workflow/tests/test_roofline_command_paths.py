"""Regression tests for layout-agnostic roofline manifest command-path resolution.

Guards the bug where a manifest `command` referenced a script by a path relative to the
task/eval root while `workdir` pointed at a sibling subdir, so rocprof-compute launched it
from the wrong CWD -> "No such file or directory" -> no counters -> roofline degraded to
"unknown". The resolver must repair this WITHOUT assuming any project-specific directory
name (no hard-coded "workspace"/"generated") or absolute prefix, and must be a strict no-op
when the manifest is already correct.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_RK_PATH = os.path.join(_HERE, "..", "scripts", "roofline_kernel.py")
_spec = importlib.util.spec_from_file_location("roofline_kernel_under_test", _RK_PATH)
rk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rk)


def _make_task(tmp_path):
    task = tmp_path
    gen = os.path.join(task, "generated")
    ws = os.path.join(task, "workspace")
    os.makedirs(gen)
    os.makedirs(ws)
    driver = os.path.join(gen, "roofline_driver.py")
    with open(driver, "w", encoding="utf-8") as handle:
        handle.write("# driver\n")
    out_dir = os.path.join(task, "roofline")
    return task, gen, ws, driver, out_dir


def test_relocates_relative_path_when_workdir_is_sibling(tmp_path):
    """The exact production failure: workdir=<task>/workspace, command references
    generated/roofline_driver.py which actually lives at <task>/generated/."""
    task, gen, ws, driver, out_dir = _make_task(str(tmp_path))
    cmd = ["python3", "generated/roofline_driver.py", "--case-id", "m1", "--leg", "candidate"]
    resolved, notes = rk._resolve_command_paths(cmd, ws, out_dir)
    assert resolved[1] == driver
    assert os.path.isfile(resolved[1])
    assert resolved[0] == "python3" and resolved[2:] == cmd[2:]  # only the path changed
    assert any("relocated" in note for note in notes)


def test_noop_when_command_resolves_from_workdir(tmp_path):
    """Already-correct manifest -> byte-identical output, no notes."""
    task, gen, ws, driver, out_dir = _make_task(str(tmp_path))
    cmd = ["python3", "generated/roofline_driver.py", "--case-id", "m1"]
    resolved, notes = rk._resolve_command_paths(cmd, task, out_dir)
    assert resolved == cmd
    assert notes == []


def test_noop_when_absolute_and_exists(tmp_path):
    task, gen, ws, driver, out_dir = _make_task(str(tmp_path))
    cmd = ["python3", driver, "--case-id", "m1"]
    resolved, notes = rk._resolve_command_paths(cmd, ws, out_dir)
    assert resolved == cmd
    assert notes == []


def test_basename_search_finds_file_in_sibling(tmp_path):
    """A bare script name resolvable only under a sibling of workdir is located."""
    task, gen, ws, driver, out_dir = _make_task(str(tmp_path))
    unittest_py = os.path.join(ws, "unittest.py")
    with open(unittest_py, "w", encoding="utf-8") as handle:
        handle.write("#\n")
    cmd = ["python3", "unittest.py", "--profile"]
    resolved, notes = rk._resolve_command_paths(cmd, task, out_dir)
    assert resolved[1] == unittest_py
    assert any("relocated" in note for note in notes)


def test_unresolvable_leaves_command_and_warns(tmp_path):
    """Nothing is fabricated: unknown file -> command unchanged + actionable warning."""
    task, gen, ws, driver, out_dir = _make_task(str(tmp_path))
    cmd = ["python3", "definitely_absent_driver.py"]
    resolved, notes = rk._resolve_command_paths(cmd, ws, out_dir)
    assert resolved == cmd
    assert any("not found" in note and "likely fail" in note for note in notes)


def test_flags_and_opaque_values_are_untouched(tmp_path):
    """Only path-like tokens are considered; flags/values are never relocated."""
    task, gen, ws, driver, out_dir = _make_task(str(tmp_path))
    cmd = ["python3", driver, "--case-id", "down_proj_decode_M1", "--iters", "200"]
    resolved, notes = rk._resolve_command_paths(cmd, ws, out_dir)
    assert resolved == cmd
    assert notes == []
