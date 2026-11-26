# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kernel_tools detect a GEAK lock held by a parent process (CPU only, fake rocprofv3, hermetic locks).

`gpu_lock.sh <ids> bash rocprofv3_safe.sh ...` WITHOUT GEAK_KERNEL_TOOLS=1 runs inside the lock but gets
no GEAK_GPU_LOCK_HELD; the tools must still see the held lock (held_lock.py) and must keep refusing when
no parent holds it -- an exported HIP_VISIBLE_DEVICES or GEAK_KERNEL_TOOLS=1 alone is not a lock.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
GPU_LOCK = SCRIPTS / "gpu_lock.sh"
KT = SCRIPTS / "kernel_tools"
SAFE = KT / "rocprofv3_safe.sh"
sys.path.insert(0, str(KT))
import held_lock  # noqa: E402

FAKE = """#!/usr/bin/env bash
out=""; while [ $# -gt 0 ]; do [ "$1" = "-d" ] && out="$2"; [ "$1" = "--" ] && { shift; break; }; shift; done
mkdir -p "$out"; echo "HIP=${HIP_VISIBLE_DEVICES-unset}" > "$out/env_seen.txt"
printf 'Kernel_Name,Start_Timestamp,End_Timestamp\\nk,1,2\\n' > "$out/test_kernel_trace.csv"
"$@"
"""


def _env(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "rocprofv3").write_text(FAKE)
    (bindir / "rocprofv3").chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GEAK_") and k not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                                                      "KT_ALLOW_UNLOCKED", "KERNEL_TOOLS_PROFILE")}
    # The DEFAULT gpu_lock.sh environment: no GEAK_KERNEL_TOOLS, so no GEAK_GPU_LOCK_HELD is exported.
    # KERNEL_EXTRACT_LOCK_DIR keeps the flock files hermetic on that default path.
    env.update(PATH=f"{bindir}:{os.environ['PATH']}", KERNEL_EXTRACT_LOCK_DIR=str(tmp_path / "locks"),
               GEAK_GPU_REQUIRE_IDLE="0", KERNEL_ENV_KEEP_ARCH="1", KERNEL_ENV_SKIP_ENUM_REAP="1")
    return env


def _safe(*extra):
    return ["bash", str(SAFE), *extra, "--kernel", "k", "--kernel-trace", "--", "true"]


def test_tool_under_default_gpu_lock_runs_without_the_env_flag(tmp_path):
    env = _env(tmp_path)
    for spec in ("5", "3,5"):   # single-GPU and pool mode
        out = tmp_path / f"o{spec.replace(',', '_')}"
        r = subprocess.run(["bash", str(GPU_LOCK), spec, *_safe("--out", str(out))], cwd=tmp_path, env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        assert "REFUSING" not in r.stderr
        assert (out / "env_seen.txt").read_text().strip() in ("HIP=3", "HIP=5")


def test_flag_or_exported_device_alone_is_not_a_lock(tmp_path):
    env = _env(tmp_path)
    env.update(GEAK_KERNEL_TOOLS="1", HIP_VISIBLE_DEVICES="5")
    r = subprocess.run(_safe("--out", str(tmp_path / "o")), cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "REFUSING" in r.stderr and "GEAK_KERNEL_TOOLS=1 alone is not a lock" in r.stderr
    assert not (tmp_path / "o" / "env_seen.txt").exists()


def test_a_lock_held_by_an_unrelated_process_is_not_adopted(tmp_path):
    env = _env(tmp_path)
    locks = tmp_path / "locks"
    locks.mkdir()
    holder = subprocess.Popen(["flock", str(locks / "gpu_5.lock"), "sleep", "30"])
    try:
        for _ in range(50):
            if held_lock._flock_held(str(locks / "gpu_5.lock")):
                break
            time.sleep(0.1)
        env["HIP_VISIBLE_DEVICES"] = "5"
        assert held_lock.held_gpu(env) is None   # held, but not by an ancestor of this process
        r = subprocess.run(_safe("--out", str(tmp_path / "o")), cwd=tmp_path, env=env,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "REFUSING" in r.stderr
    finally:
        holder.kill()
        holder.wait()


def test_held_lock_with_another_dev_id_runs_on_the_held_gpu(tmp_path):
    env = _env(tmp_path)
    out = tmp_path / "o"
    r = subprocess.run(["bash", str(GPU_LOCK), "5", *_safe("--gpu", "4", "--out", str(out))], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "already under gpu_lock.sh holding GPU 5" in r.stderr
    assert (out / "env_seen.txt").read_text().strip() == "HIP=5"


def test_held_gpu_needs_one_visible_id(tmp_path):
    assert held_lock.held_gpu({}) is None
    assert held_lock.held_gpu({"HIP_VISIBLE_DEVICES": "2,3"}) is None
