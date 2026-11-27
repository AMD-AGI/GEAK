#!/usr/bin/env python3
"""tools/hip_ep_e2e/build_kernels.sh on a fake hip-ep checkout, with stand-ins for
ninja and the compiler: the library is built from the tag plus the overlay and
nothing else, the tree is back at the tag afterwards, a failed build leaves no
library, and the build lock is the one tools/ep_task_runner.py takes.

    python3 -m pytest tools/hip_ep_e2e/tests/test_build_kernels.py
"""
import fcntl
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "build_kernels.sh")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import ep_task_runner  # noqa: E402

KERNEL = "lib/Runtime/Kernels/hip/k.hip"
LIB = "libcustom_kernels_gfx11-generic.so"
# "Builds" the library by copying the kernel source into it, so a test can read
# which source a build saw.
FAKE_NINJA = """#!/bin/bash
[ "$1" = -C ] || exit 2
mkdir -p "$2/lib/Runtime/Kernels"
cp "$HIPEP_SRC/lib/Runtime/Kernels/hip/k.hip" "$2/lib/Runtime/Kernels/lib$3.so"
exit "${FAKE_NINJA_RC:-0}"
"""


def _git(src, *args):
    subprocess.run(["git", "-C", str(src), *args], check=True, capture_output=True)


@pytest.fixture
def box(tmp_path):
    src, build, therock, fakebin, pkg = (tmp_path / d for d in ("src", "build", "therock", "bin", "pkg"))
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / LIB).write_bytes(b"shipped")
    (src / "lib/Runtime/Kernels/hip").mkdir(parents=True)
    (src / KERNEL).write_text("orig")
    _git(src, "init", "-q")
    _git(src, "add", "-A")
    _git(src, "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "release")
    _git(src, "tag", "v0.5.3")
    (therock / "bin").mkdir(parents=True)
    cc = therock / "bin" / "amdclang++"
    cc.write_text("#!/bin/bash\necho 'AMD clang version 24.0.0 (fake)'\n")
    (build / "CMakeFiles" / "4.0.0").mkdir(parents=True)
    (build / "build.ninja").write_text("")
    (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
    (build / "CMakeFiles" / "4.0.0" / "CMakeHIPCompiler.cmake").write_text(
        f'set(CMAKE_HIP_COMPILER "{cc}")\n')
    fakebin.mkdir()
    (fakebin / "ninja").write_text(FAKE_NINJA)
    for exe in (cc, fakebin / "ninja"):
        exe.chmod(0o755)
    env = {**os.environ, "HIPEP_PKG": str(pkg), "HIPEP_SRC": str(src), "HIPEP_BUILD": str(build),
           "HIPEP_THEROCK": str(therock), "TMPDIR": str(tmp_path),
           "PATH": f"{fakebin}:{os.environ['PATH']}"}
    return {"src": src, "build": build, "pkg": pkg, "env": env, "tmp": tmp_path}


def _build(box, overlay, out, **extra):
    return subprocess.run(["bash", SCRIPT, str(overlay), str(out)], capture_output=True,
                          text=True, env={**box["env"], **extra}, check=False)


def _overlay(tmp_path, files):
    ov = tmp_path / "overlay"
    for rel, text in files.items():
        (ov / rel).parent.mkdir(parents=True, exist_ok=True)
        (ov / rel).write_text(text)
    return ov


def test_overlay_is_built_and_the_tree_goes_back_to_the_tag(box):
    out = box["tmp"] / "out"
    p = _build(box, _overlay(box["tmp"], {KERNEL: "edited"}), out)
    assert p.returncode == 0, p.stderr
    assert (out / LIB).read_text() == "edited"
    assert (box["src"] / KERNEL).read_text() == "orig"
    assert f"1 overlay files: {KERNEL}" in p.stdout


def test_a_staged_edit_is_not_built(box):
    (box["src"] / KERNEL).write_text("staged")
    _git(box["src"], "add", KERNEL)
    out = box["tmp"] / "out"
    p = _build(box, "-", out)
    assert p.returncode == 0, p.stderr
    assert (out / LIB).read_text() == "orig"


def test_overlay_outside_lib_is_refused(box):
    p = _build(box, _overlay(box["tmp"], {"CMakeLists.txt": "x"}), box["tmp"] / "out")
    assert p.returncode != 0 and "outside lib/" in p.stderr
    assert not (box["src"] / "CMakeLists.txt").exists()


def test_failed_build_leaves_no_library(box):
    out = box["tmp"] / "out"
    assert _build(box, "-", out).returncode == 0
    p = _build(box, _overlay(box["tmp"], {KERNEL: "broken"}), out, FAKE_NINJA_RC="1")
    assert p.returncode != 0 and "build failed" in p.stderr
    assert not (out / LIB).exists()
    assert (box["src"] / KERNEL).read_text() == "orig"


def _lock_name(src):
    return os.path.basename(ep_task_runner._ep_build_lock_path(str(src)))


def test_debug_build_dir_is_refused(box):
    (box["build"] / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Debug\n")
    p = _build(box, "-", box["tmp"] / "out")
    assert p.returncode != 0 and "not a Release build" in p.stderr


def test_a_build_waits_for_the_lock(box):
    with open(box["tmp"] / _lock_name(box["src"]), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        p = subprocess.Popen(["bash", SCRIPT, "-", str(box["tmp"] / "out")], env=box["env"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        assert p.poll() is None, "the build went ahead while another build held the lock"
    assert p.wait(timeout=30) == 0


def test_a_symlinked_checkout_takes_the_same_lock(box):
    link = box["tmp"] / "src-link"
    link.symlink_to(box["src"])
    assert _build(box, "-", box["tmp"] / "out", HIPEP_SRC=str(link)).returncode == 0
    assert (box["tmp"] / _lock_name(box["src"])).exists()


def test_a_checkout_off_the_tag_is_refused(box):
    _git(box["src"], "tag", "-d", "v0.5.3")
    p = _build(box, "-", box["tmp"] / "out")
    assert p.returncode != 0 and "is not at v0.5.3" in p.stderr


def test_a_build_dir_from_another_compiler_is_refused(box):
    (box["build"] / "CMakeFiles" / "4.0.0" / "CMakeHIPCompiler.cmake").write_text(
        'set(CMAKE_HIP_COMPILER "/opt/rocm/llvm/bin/clang++")\n')
    p = _build(box, "-", box["tmp"] / "out")
    assert p.returncode != 0 and "/opt/rocm/llvm/bin/clang++" in p.stderr


def test_unset_paths_are_named(box):
    env = {k: v for k, v in box["env"].items() if k != "HIPEP_SRC"}
    p = subprocess.run(["bash", SCRIPT, "-", str(box["tmp"] / "out")], capture_output=True, text=True,
                       env=env, check=False)
    assert p.returncode != 0 and "HIPEP_SRC is not set" in p.stderr


def test_the_arch_is_the_one_the_package_ships(box):
    (box["pkg"] / "lib" / LIB).rename(box["pkg"] / "lib" / "libcustom_kernels_gfx1201.so")
    out = box["tmp"] / "out"
    p = _build(box, "-", out)
    assert p.returncode == 0, p.stderr
    assert (out / "libcustom_kernels_gfx1201.so").read_text() == "orig"


def test_a_package_without_exactly_one_kernel_library_is_refused(box):
    (box["pkg"] / "lib" / "libcustom_kernels_gfx1151.so").write_bytes(b"second")
    p = _build(box, "-", box["tmp"] / "out")
    assert p.returncode != 0 and "expected one" in p.stderr
