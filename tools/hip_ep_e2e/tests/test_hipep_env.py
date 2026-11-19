#!/usr/bin/env python3
"""tools/hip_ep_e2e/hipep_env.py: machine paths come from the environment, each
package run gets its own HOME and temp dir, and a verdict file is written whole.

    python3 -m pytest tools/hip_ep_e2e/tests/test_hipep_env.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hipep_env  # noqa: E402


def test_unset_path_is_named(monkeypatch):
    monkeypatch.setenv("HIPEP_SYSROOT", "/s")
    monkeypatch.delenv("HIPEP_THEROCK", raising=False)
    with pytest.raises(SystemExit, match="HIPEP_THEROCK is not set"):
        hipep_env.Env.from_environ("/pkg")


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
    env = hipep_env.Env(pkg="/p", therock="/tr", sysroot="/s", lock_path="/l", lock_wait=1)
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
