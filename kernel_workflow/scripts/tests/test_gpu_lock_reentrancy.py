# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpu_lock.sh must not deadlock when a command already holding a GPU calls it again.

profile_kernel.sh locks the GPU internally and agents wrap it in gpu_lock.sh, so a nested call for
the same GPU is the normal profiling path. These tests run without a GPU: the idle check is off and
rocminfo is not needed for the re-entrant path.
"""

import fcntl
import os
import signal
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
GPU_LOCK_WRAPPER = SCRIPTS / "gpu_lock.sh"


def _env(lock_dir, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEAK_GPU_")}
    env.update(
        KERNEL_EXTRACT_LOCK_DIR=str(lock_dir),
        GEAK_GPU_REQUIRE_IDLE="0",
        KERNEL_ENV_KEEP_ARCH="1",
    )
    env.update(extra)
    return env


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def test_nested_call_for_held_gpu_does_not_block(tmp_path, workspace):
    # Outer call takes GPU 0, inner call asks for GPU 0 again. Before the fix the inner flock
    # waited on the outer one (up to 1200 s); now it runs directly.
    nested = f"bash {GPU_LOCK_WRAPPER} 0 echo nested-ok"
    result = subprocess.run(
        ["bash", str(GPU_LOCK_WRAPPER), "0", "bash", "-c", nested],
        check=False,
        cwd=workspace,
        env=_env(tmp_path / "locks"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "nested-ok" in result.stdout


def test_nested_call_propagates_exit_status(tmp_path, workspace):
    result = subprocess.run(
        ["bash", str(GPU_LOCK_WRAPPER), "0", "bash", "-c", "exit 3"],
        check=False,
        cwd=workspace,
        env=_env(tmp_path / "locks", GEAK_GPU_LOCK_HELD="0"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 3


def test_nested_call_still_runs_source_check_on_exit(tmp_path, workspace):
    # A source-validation failure recorded during the command must still invalidate it (exit 86).
    (workspace / ".geak").mkdir()
    record = "echo '{}' >> .geak/invalid_measurements.jsonl; echo PASS"
    result = subprocess.run(
        ["bash", str(GPU_LOCK_WRAPPER), "0", "bash", "-c", record],
        check=False,
        cwd=workspace,
        env=_env(tmp_path / "locks", GEAK_GPU_LOCK_HELD="0"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 86
    assert "GEAK_SOURCE_INVALID" in result.stderr


def test_holding_one_gpu_does_not_bypass_the_lock_of_another(tmp_path, workspace):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    with open(lock_dir / "gpu_1.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # someone else holds GPU 1
        proc = subprocess.Popen(
            ["bash", str(GPU_LOCK_WRAPPER), "1", "echo", "should-not-run"],
            cwd=workspace,
            env=_env(lock_dir, GEAK_GPU_LOCK_HELD="0"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                proc.wait(timeout=3)
        finally:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
    assert "should-not-run" not in out
