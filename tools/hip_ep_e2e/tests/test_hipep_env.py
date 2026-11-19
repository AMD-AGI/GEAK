#!/usr/bin/env python3
"""tools/hip_ep_e2e/hipep_env.py: machine paths come from the environment, each
package run gets its own HOME and temp dir, and a verdict file is written whole and
never outlives a failed rerun.

    python3 -m pytest tools/hip_ep_e2e/tests/test_hipep_env.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench  # noqa: E402
import hipep_env  # noqa: E402
import kprof  # noqa: E402
import parity  # noqa: E402


def test_every_machine_path_is_required(monkeypatch):
    assert set(hipep_env.PATHS) == {"HIPEP_PKG", "HIPEP_THEROCK", "HIPEP_SYSROOT", "HIPEP_PY",
                                    "HIPEP_SRC", "HIPEP_BUILD"}
    for name in hipep_env.PATHS:
        monkeypatch.delenv(name, raising=False)
        with pytest.raises(SystemExit, match=f"{name} is not set"):
            hipep_env.require(name)
    monkeypatch.setenv("HIPEP_THEROCK", "/tr")
    with pytest.raises(SystemExit, match="HIPEP_SYSROOT is not set"):
        hipep_env.Env.from_environ("/pkg")
    monkeypatch.setenv("HIPEP_SYSROOT", "/s")
    with pytest.raises(SystemExit, match="HIPEP_PY is not set"):
        _ = hipep_env.Env.from_environ("/pkg").python


def test_a_failed_write_leaves_the_previous_verdict(tmp_path):
    path = tmp_path / "ab.json"
    hipep_env.write_json(str(path), {"accepted": True})
    with pytest.raises(TypeError):
        hipep_env.write_json(str(path), {"accepted": object()})
    assert path.read_text() == '{\n  "accepted": true\n}'


def test_package_comes_from_the_argument_or_the_environment(monkeypatch):
    monkeypatch.setenv("HIPEP_THEROCK", "/tr")
    monkeypatch.setenv("HIPEP_SYSROOT", "/s")
    monkeypatch.setenv("HIPEP_PKG", "/pkg/env")
    assert hipep_env.Env.from_environ().pkg == "/pkg/env"
    assert hipep_env.Env.from_environ("/pkg/arg").pkg == "/pkg/arg"


def test_runs_of_same_named_packages_do_not_share_a_home(tmp_path):
    a = hipep_env.pkg_home(str(tmp_path), "/run/lanes/0_tuning/pkg")
    b = hipep_env.pkg_home(str(tmp_path), "/run/lanes/1_kernel/pkg/")
    assert a != b
    assert os.path.basename(a).startswith("home-pkg-")
    assert a == hipep_env.pkg_home(str(tmp_path), "/run/lanes/0_tuning/pkg/")


def test_package_runs_get_their_own_home_and_temp_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HIPDNN_MATMUL_AUTOTUNE_MODE", "online")
    monkeypatch.setenv("LD_PRELOAD", "/x.so")
    env = hipep_env.Env(pkg="/p", therock="/tr", sysroot="/s")
    home = str(tmp_path / "home")
    e = hipep_env.pkg_env(env, home)
    assert e["HOME"] == home
    assert e["TMPDIR"] == os.path.join(home, "tmp") and os.path.isdir(e["TMPDIR"])
    assert e["THEROCK_DIST"] == "/tr"
    assert "HIPDNN_MATMUL_AUTOTUNE_MODE" not in e and "LD_PRELOAD" not in e


def test_the_kernel_library_is_the_one_the_package_ships(tmp_path):
    (tmp_path / "lib").mkdir()
    with pytest.raises(RuntimeError, match="expected one"):
        hipep_env.kernel_lib(str(tmp_path))
    (tmp_path / "lib" / "libcustom_kernels_gfx1201.so").write_bytes(b"k")
    assert hipep_env.kernel_lib(str(tmp_path)).endswith("lib/libcustom_kernels_gfx1201.so")


@pytest.mark.parametrize("tool, broken, argv, verdict", [
    (bench, "collect", ["ab", "--model", "m", "--base", "a", "--cand", "b"], "ab.json"),
    (parity, "dump", ["--model", "m", "--base", "a", "--cand", "b"], "parity.json"),
    (kprof, "record", ["record", "--model", "m", "--pkg", "p"], "heads.json"),
])
def test_a_failed_rerun_leaves_no_earlier_verdict(tmp_path, monkeypatch, tool, broken, argv, verdict):
    (tmp_path / verdict).write_text('{"accepted": true}')

    def fail(*args, **kwargs):
        raise RuntimeError("the tool exited 1")

    monkeypatch.setattr(hipep_env.Env, "from_environ", staticmethod(lambda pkg=None: pkg))
    monkeypatch.setattr(tool, broken, fail)
    monkeypatch.setattr(sys, "argv", ["tool", *argv, "--out", str(tmp_path)])
    with pytest.raises(RuntimeError):
        tool.main()
    assert not (tmp_path / verdict).exists(), "an earlier run's verdict survived a failed rerun"


def test_package_runs_wait_for_the_serving_lock(tmp_path, monkeypatch):
    import fcntl
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("SERVING_GPU_LOCK", str(lock))
    monkeypatch.setenv("SERVING_LOCK_WAIT", "0")
    monkeypatch.setattr(hipep_env.subprocess, "run", lambda *a, **k: pytest.fail("ran without the lock"))
    env = hipep_env.Env(pkg=str(tmp_path), therock="/tr", sysroot="/s")
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="busy"):
            hipep_env.run_logged("x", env, ["x"], str(tmp_path / "home"), str(tmp_path / "log"))
