# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kernel_tools/rocprofv3_safe.sh under GEAK's gpu_lock.sh, with a fake rocprofv3 (CPU-only).

The device rule: rocprofv3 aborts only when HIP_VISIBLE_DEVICES sits INSIDE ROCR_VISIBLE_DEVICES, so
the wrapper may touch HIP_VISIBLE_DEVICES only when BOTH are set (and then collapses the pair to the
same device). With only HIP set -- exactly what gpu_lock.sh exports -- it must leave it alone, or the
collection would see every GPU and profile device 0.
"""

import os
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
GPU_LOCK = SCRIPTS / "gpu_lock.sh"
SAFE = SCRIPTS / "kernel_tools" / "rocprofv3_safe.sh"

FAKE = """#!/usr/bin/env bash
out=""; while [ $# -gt 0 ]; do [ "$1" = "-d" ] && out="$2"; [ "$1" = "--" ] && { shift; break; }; shift; done
mkdir -p "$out"; echo "HIP=${HIP_VISIBLE_DEVICES-unset} ROCR=${ROCR_VISIBLE_DEVICES-unset}" > "$out/env_seen.txt"
[ -n "${FAKE_HANG:-}" ] && sleep 30
"$@"
"""


def _setup(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "rocprofv3").write_text(FAKE)
    (bindir / "rocprofv3").chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", GEAK_GPU_LOCK_DIR=str(tmp_path / "locks"),
               GEAK_GPU_REQUIRE_IDLE="0", KERNEL_ENV_KEEP_ARCH="1", KERNEL_ENV_SKIP_ENUM_REAP="1")
    for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GEAK_GPU_LOCK_HELD", "GEAK_GPU_BROKER",
              "KT_ALLOW_UNLOCKED"):
        env.pop(k, None)
    return env


def _seen(out):
    return (out / "env_seen.txt").read_text().strip()


def test_hip_only_is_not_unset_under_gpu_lock(tmp_path):
    env = _setup(tmp_path)
    out = tmp_path / "o1"
    r = subprocess.run(["bash", str(GPU_LOCK), "5", "bash", str(SAFE), "--kernel", "k", "--out", str(out),
                        "--kernel-trace", "--cmd", "true"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert _seen(out) == "HIP=5 ROCR=unset"


def test_self_locks_with_gpu_id_and_keeps_hip(tmp_path):
    env = _setup(tmp_path)
    out = tmp_path / "o2"
    r = subprocess.run(["bash", str(SAFE), "--gpu", "6", "--kernel", "k", "--out", str(out),
                        "--pmc", "SQ_WAVES", "--", "true"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert _seen(out) == "HIP=6 ROCR=unset"


def test_hip_inside_rocr_collapses_to_the_same_device(tmp_path):
    env = _setup(tmp_path)
    env["ROCR_VISIBLE_DEVICES"] = "4,6"
    out = tmp_path / "o3"
    # gpu_lock.sh keeps an inherited ROCR mask and exports HIP=1 (logical index 1 -> physical 6)
    r = subprocess.run(["bash", str(GPU_LOCK), "1", "bash", str(SAFE), "--kernel", "k", "--out", str(out),
                        "--kernel-trace", "--", "true"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert _seen(out) == "HIP=unset ROCR=6"


def test_refuses_outside_the_lock_without_an_id(tmp_path):
    env = _setup(tmp_path)
    r = subprocess.run(["bash", str(SAFE), "--kernel", "k", "--out", str(tmp_path / "o4"),
                        "--kernel-trace", "--", "true"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "gpu_lock.sh" in r.stderr


def test_timeout_degrades_and_no_timeout_is_allowed(tmp_path):
    env = _setup(tmp_path)
    env["FAKE_HANG"] = "1"
    out = tmp_path / "o5"
    r = subprocess.run(["bash", str(GPU_LOCK), "2", "bash", str(SAFE), "--kernel", "k", "--out", str(out),
                        "--pmc", "SQ_WAVES", "--timeout", "1", "--", "true"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 124 and '"state":"timeout"' in (out / "rocprofv3_safe.json").read_text()
    env.pop("FAKE_HANG")
    r = subprocess.run(["bash", str(GPU_LOCK), "2", "bash", str(SAFE), "--kernel", "k", "--out", str(out),
                        "--kernel-trace", "--no-timeout", "--cmd", "true"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.glob("o5.old_*")), "the previous --out must be moved aside, not deleted"
