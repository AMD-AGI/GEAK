#!/usr/bin/env python3
"""tools/hip_ep_e2e/integrate.py: a kernel_workflow patch becomes a hip-ep overlay; a patch
that touches anything but the editable files (anywhere in the task, deletions included) or
does not apply is refused, and a new file an agent left outside src/ is left out.

    python3 -m pytest tools/hip_ep_e2e/tests/test_integrate.py
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import integrate  # noqa: E402

KERNEL = "src/lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"
FROZEN = "src/lib/Runtime/Kernels/unit_test/matmul_nbits/Xfp16_Wu4_Yfp16/test_matmul_nbits.cpp"


def _task(tmp_path):
    t = tmp_path / "task"
    for rel, text in ((KERNEL, "int tile = 128;\n"), (FROZEN, "// oracle\n")):
        (t / rel).parent.mkdir(parents=True, exist_ok=True)
        (t / rel).write_text(text)
    (t / "config.yaml").write_text(f"source_file_path:\n- {KERNEL}\n")
    return t


def _patch(tmp_path, task, rel, new_text):
    """A git diff from the task root, as kernel_workflow's TechLead writes it."""
    work = tmp_path / "ws"
    subprocess.run(["cp", "-r", str(task), str(work)], check=True)
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "base"],
                   cwd=work, check=True)
    if new_text is None:
        (work / rel).unlink()
    else:
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(new_text)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)  # as kernel_lane commits a round
    diff = subprocess.run(["git", "diff", "--cached"], cwd=work, capture_output=True, text=True,
                          check=True)
    p = tmp_path / "final_patch.diff"
    p.write_text(diff.stdout)
    return p


def test_patch_becomes_overlay(tmp_path):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, KERNEL, "int tile = 256;\n")
    base = tmp_path / "lut_overlay" / "lib" / "Runtime" / "Kernels" / "hip" / "autotune"
    base.mkdir(parents=True)
    (base / "gfx1151.fb").write_bytes(b"lut")
    out = tmp_path / "overlay"
    info = integrate.integrate(str(task), str(patch), str(out), str(tmp_path / "lut_overlay"))
    assert info["files"] == ["lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"]
    assert (out / "lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip").read_text() == "int tile = 256;\n"
    assert (out / "lib/Runtime/Kernels/hip/autotune/gfx1151.fb").read_bytes() == b"lut"
    assert (task / KERNEL).read_text() == "int tile = 128;\n"
    assert json.loads((tmp_path / "overlay.json").read_text())["files"] == info["files"]


def test_frozen_edit_is_refused(tmp_path):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, FROZEN, "// weakened oracle\n")
    with pytest.raises(RuntimeError, match="outside the task's editable set"):
        integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))


def test_stale_patch_is_refused(tmp_path):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, KERNEL, "int tile = 256;\n")
    (task / KERNEL).write_text("int tile = 64;\n")
    with pytest.raises(RuntimeError, match="does not apply"):
        integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))


def test_edit_outside_src_is_refused(tmp_path):
    task = _task(tmp_path)
    (task / "scripts").mkdir()
    (task / "scripts" / "task_runner.py").write_text("ok = check()\n")
    patch = _patch(tmp_path, task, "scripts/task_runner.py", "ok = True\n")
    with pytest.raises(RuntimeError, match=r"editable set: \['scripts/task_runner.py'\]"):
        integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))


def test_deleting_a_frozen_file_is_refused(tmp_path):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, FROZEN, None)
    with pytest.raises(RuntimeError, match="outside the task's editable set"):
        integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))
    assert not (tmp_path / "overlay").exists()


def test_deleting_an_editable_file_is_refused(tmp_path):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, KERNEL, None)
    with pytest.raises(RuntimeError, match="deletes editable files"):
        integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))


def _two_file_patch(tmp_path, task, extra_rel):
    """The kernel edit plus one new file, in one patch."""
    work = tmp_path / "ws2"
    subprocess.run(["cp", "-r", str(task), str(work)], check=True)
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=work, check=True)
    (work / KERNEL).write_text("int tile = 256;\n")
    (work / extra_rel).parent.mkdir(parents=True, exist_ok=True)
    (work / extra_rel).write_text("# scratch\n")
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    p = tmp_path / "final_patch2.diff"
    p.write_text(subprocess.run(["git", "diff", "--cached"], cwd=work, capture_output=True,
                                text=True, check=True).stdout)
    return p


def test_a_scratch_file_outside_src_is_left_out(tmp_path):
    task = _task(tmp_path)
    patch = _two_file_patch(tmp_path, task, "test_harness.py")
    info = integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))
    assert info["files"] == ["lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"]
    assert info["ignored"] == ["test_harness.py"]
    assert not (tmp_path / "overlay" / "test_harness.py").exists()


def test_a_new_file_inside_src_is_refused(tmp_path):
    task = _task(tmp_path)
    patch = _two_file_patch(tmp_path, task, "src/lib/Runtime/Kernels/include/extra.h")
    with pytest.raises(RuntimeError, match="outside the task's editable set"):
        integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))


def test_a_rerun_replaces_the_earlier_overlay(tmp_path):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, KERNEL, "int tile = 256;\n")
    out = tmp_path / "overlay"
    integrate.integrate(str(task), str(patch), str(out))
    (out / "leftover").write_text("from an earlier attempt")
    info = integrate.integrate(str(task), str(patch), str(out))
    assert info["files"] == ["lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"]
    assert not (out / "leftover").exists()


def test_the_task_build_dir_is_not_compared(tmp_path):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, KERNEL, "int tile = 256;\n")
    (task / "build").mkdir()
    (task / "build" / "kernel.o").write_bytes(b"object")
    info = integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))
    assert info["files"] == ["lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"]


def test_a_patch_applies_with_a_git_work_tree_around_the_temp_dir(tmp_path, monkeypatch):
    task = _task(tmp_path)
    patch = _patch(tmp_path, task, KERNEL, "int tile = 256;\n")
    repo = tmp_path / "repo"
    (repo / "tmp").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.setattr(tempfile, "tempdir", str(repo / "tmp"))
    info = integrate.integrate(str(task), str(patch), str(tmp_path / "overlay"))
    assert info["files"] == ["lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"]
