#!/usr/bin/env python3
"""tools/hip_ep_e2e/make_pkg.sh on a fake package: the candidate gets the new
kernel library, the base package is never written through the hard links, a base on
another filesystem is copied, a failed run leaves nothing behind, and an existing
output is refused.

    python3 -m pytest tools/hip_ep_e2e/tests/test_make_pkg.py
"""
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "make_pkg.sh")
LIB = "libcustom_kernels_gfx11-generic.so"


def _base(tmp_path):
    base = tmp_path / "base"
    (base / "lib").mkdir(parents=True)
    (base / "bin").mkdir()
    (base / "lib" / LIB).write_bytes(b"old kernels")
    (base / "lib" / "libamdgpu-ep.so").write_bytes(b"ep")
    (base / "bin" / "model_benchmark").write_bytes(b"mb")
    (base / "release.zip").write_bytes(b"zip")
    (tmp_path / "new").mkdir()
    new = tmp_path / "new" / LIB
    new.write_bytes(b"new kernels")
    return base, new


def test_swaps_kernel_lib_only(tmp_path):
    base, new = _base(tmp_path)
    out = tmp_path / "cand"
    subprocess.run(["bash", SCRIPT, str(base), str(new), str(out)], check=True,
                   capture_output=True)
    assert (out / "lib" / LIB).read_bytes() == b"new kernels"
    assert (base / "lib" / LIB).read_bytes() == b"old kernels"
    assert os.stat(out / "lib" / "libamdgpu-ep.so").st_ino == \
        os.stat(base / "lib" / "libamdgpu-ep.so").st_ino
    assert not (out / "release.zip").exists()
    info = json.loads((out / "PKG_INFO.json").read_text())
    assert info["kernel_lib_md5"] != info["base_kernel_lib_md5"]


def test_chained_package_leaves_its_base_record_alone(tmp_path):
    base, new = _base(tmp_path)
    c1, c2 = tmp_path / "c1", tmp_path / "c2"
    subprocess.run(["bash", SCRIPT, str(base), str(new), str(c1)], check=True, capture_output=True)
    before = (c1 / "PKG_INFO.json").read_text()
    (tmp_path / "newer").mkdir()
    newer = tmp_path / "newer" / LIB
    newer.write_bytes(b"newer kernels")
    subprocess.run(["bash", SCRIPT, str(c1), str(newer), str(c2)], check=True, capture_output=True)
    assert (c1 / "PKG_INFO.json").read_text() == before
    assert json.loads((c2 / "PKG_INFO.json").read_text())["base"] == str(c1)


def test_refuses_existing_output(tmp_path):
    base, new = _base(tmp_path)
    out = tmp_path / "cand"
    out.mkdir()
    p = subprocess.run(["bash", SCRIPT, str(base), str(new), str(out)], capture_output=True,
                       text=True, check=False)
    assert p.returncode != 0 and "refusing" in p.stderr


def test_a_base_on_another_filesystem_is_copied(tmp_path):
    if not os.path.isdir("/dev/shm") or os.stat("/dev/shm").st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("needs a second filesystem")
    root = pathlib.Path(tempfile.mkdtemp(dir="/dev/shm"))
    try:
        base, new = _base(root)
        out = tmp_path / "cand"
        subprocess.run(["bash", SCRIPT, str(base), str(new), str(out)], check=True, capture_output=True)
        assert (out / "lib" / LIB).read_bytes() == b"new kernels"
        assert (out / "bin" / "model_benchmark").read_bytes() == b"mb"
        assert (base / "lib" / LIB).read_bytes() == b"old kernels"
    finally:
        shutil.rmtree(root)


def test_a_failed_run_leaves_no_package(tmp_path):
    base, new = _base(tmp_path)
    fakebin = tmp_path / "bin-fake"
    fakebin.mkdir()
    # Fails the library copy (the one two-argument cp) after the package dir exists.
    (fakebin / "cp").write_text('#!/bin/bash\n[ "$#" -eq 2 ] && exit 1\nexec /bin/cp "$@"\n')
    (fakebin / "cp").chmod(0o755)
    out = tmp_path / "cand"
    p = subprocess.run(["bash", SCRIPT, str(base), str(new), str(out)], capture_output=True, text=True,
                       env={**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}, check=False)
    assert p.returncode != 0 and not out.exists()


def test_a_library_for_another_arch_is_refused(tmp_path):
    base, _ = _base(tmp_path)
    other = tmp_path / "libcustom_kernels_gfx1201.so"
    other.write_bytes(b"other arch")
    p = subprocess.run(["bash", SCRIPT, str(base), str(other), str(tmp_path / "cand")], capture_output=True,
                       text=True, check=False)
    assert p.returncode != 0 and "not the arch" in p.stderr and not (tmp_path / "cand").exists()
