# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpu_lock.sh must not deadlock when a command already holding a GPU calls it again.

profile_kernel.sh locks the GPU internally and agents wrap it in gpu_lock.sh, so a nested call for
the same GPU is the normal profiling path. The nested call may skip the lock only while the
process that holds it still has the lock file open. These tests run without a GPU: the idle check
is off and the compile-arch pin is skipped.
"""

import fcntl
import os
import shlex
import signal
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
GPU_LOCK_WRAPPER = SCRIPTS / "gpu_lock.sh"
WRAP = f"bash {shlex.quote(str(GPU_LOCK_WRAPPER))}"


def _env(lock_dir, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEAK_GPU_")}
    env.update(
        KERNEL_EXTRACT_LOCK_DIR=str(lock_dir),
        GEAK_GPU_REQUIRE_IDLE="0",
        KERNEL_ENV_KEEP_ARCH="1",
    )
    env.update(extra)
    return env


def _run(cmd, cwd, env, timeout=60):
    return subprocess.run(
        ["bash", "-c", cmd],
        check=False,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_blocks(cmd, cwd, env, seconds=3):
    """The command must still be waiting for a lock after `seconds`."""
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=seconds)
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    assert "should-not-run" not in out


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def test_nested_call_for_held_gpu_does_not_block(tmp_path, workspace):
    # Before the fix the inner flock waited on the outer one (up to 1200 s).
    result = _run(
        f"{WRAP} 0 {WRAP} 0 echo nested-ok", workspace, _env(tmp_path / "locks")
    )
    assert result.returncode == 0, result.stderr
    assert "nested-ok" in result.stdout


def test_nested_pool_call_reuses_the_held_lane(tmp_path, workspace):
    inner = f"{WRAP} 0,1,2,3 bash -c 'echo inner=$HIP_VISIBLE_DEVICES'"
    outer = f"{WRAP} 0,1,2,3 bash -c {shlex.quote('echo outer=$HIP_VISIBLE_DEVICES; ' + inner)}"
    result = _run(outer, workspace, _env(tmp_path / "locks"))
    assert result.returncode == 0, result.stderr
    lanes = dict(line.split("=") for line in result.stdout.split())
    assert lanes["inner"] == lanes["outer"]


def test_nested_call_propagates_exit_status(tmp_path, workspace):
    result = _run(
        f"{WRAP} 0 {WRAP} 0 bash -c 'exit 3'", workspace, _env(tmp_path / "locks")
    )
    assert result.returncode == 3


def test_nested_call_still_runs_source_check_on_exit(tmp_path, workspace):
    # A source-validation failure recorded during a nested command must invalidate it (exit 86).
    (workspace / ".geak").mkdir()
    record = "echo '{}' >> .geak/invalid_measurements.jsonl"
    inner = f"{WRAP} 0 bash -c {shlex.quote(record)}; echo inner=$?"
    result = _run(
        f"{WRAP} 0 bash -c {shlex.quote(inner)}", workspace, _env(tmp_path / "locks")
    )
    assert "inner=86" in result.stdout
    assert "GEAK_SOURCE_INVALID" in result.stderr


def test_holding_one_gpu_does_not_bypass_the_lock_of_another(tmp_path, workspace):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    with open(lock_dir / "gpu_1.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # someone else holds GPU 1
        _assert_blocks(
            f"{WRAP} 0 {WRAP} 1 echo should-not-run", workspace, _env(lock_dir)
        )


def test_marker_from_a_finished_holder_does_not_bypass_the_lock(tmp_path, workspace):
    # A descendant that outlived its wrapper still carries GEAK_GPU_LOCK_HELD, but its holder is
    # gone; it must queue for the GPU like anyone else.
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    gone = subprocess.Popen(["true"])
    gone.wait()
    with open(lock_dir / "gpu_0.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another job holds GPU 0
        env = _env(lock_dir, GEAK_GPU_LOCK_HELD=f"0:{gone.pid}:200")
        _assert_blocks(f"{WRAP} 0 echo should-not-run", workspace, env)
