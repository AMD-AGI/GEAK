# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEAK's shared lock namespace, inherited ownership and compatibility shim.

CPU-only. Two processes asking for the same GPU id must serialize, because both flock the same file
in the same namespace. No GPU work is performed.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parents[1]
GEAK_LOCK = SCRIPTS / "gpu_lock.sh"
SHIM = REPO / "perf_knowledge/expert_skills/skills/gluon_authoring/scripts/gpu_lock.sh"
DEFAULT_DIR = "/tmp/team_gpu_locks"


def _env(lock_dir=None, **extra):
    env = dict(os.environ, GEAK_GPU_REQUIRE_IDLE="0", KERNEL_ENV_KEEP_ARCH="1",
               KERNEL_ENV_SKIP_ENUM_REAP="1")
    for k in ("GEAK_GPU_BROKER", "GEAK_GPU_ALLOWED", "GEAK_GPU_LOCK_HELD", "GEAK_GPU_LOCK_DIR",
              "GEAK_GPU_LOCKED_SET", "GEAK_GPU_LEASE_ACTIVE", "GEAK_GPU_LEASE_IDS"):
        env.pop(k, None)
    if lock_dir is not None:
        env["GEAK_GPU_LOCK_DIR"] = str(lock_dir)
    env.update(extra)
    return env


HOLD = 'echo "start $1 $(date +%s.%N)" >> "$2"; sleep 1.5; echo "end $1 $(date +%s.%N)" >> "$2"'


def _intervals(log):
    t = {}
    for line in log.read_text().splitlines():
        what, who, ts = line.split()
        t.setdefault(who, {})[what] = float(ts)
    return t


def _assert_serialized(log):
    t = _intervals(log)
    assert set(t) == {"first", "geak"}, log.read_text()
    a, b = sorted(t.values(), key=lambda x: x["start"])
    assert b["start"] >= a["end"], f"holders of one GPU id overlapped: {t}"


def test_lock_dir_defaults_to_the_shared_namespace():
    geak = GEAK_LOCK.read_text()
    assert f'LOCK_DIR="${{GEAK_GPU_LOCK_DIR:-${{KERNEL_EXTRACT_LOCK_DIR:-{DEFAULT_DIR}}}}}"' in geak


def _race(tmp_path, gpu_id, lock_dir):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    log = tmp_path / f"order_{gpu_id}.log"
    log.write_text("")
    env = _env(lock_dir)
    a = subprocess.Popen(["bash", str(SHIM), str(gpu_id), "bash", "-c", HOLD, "_", "first", str(log)],
                         cwd=ws, env=env)
    time.sleep(0.4)
    b = subprocess.Popen(["bash", str(GEAK_LOCK), str(gpu_id), "bash", "-c", HOLD, "_", "geak", str(log)],
                         cwd=ws, env=env)
    assert a.wait(timeout=120) == 0 and b.wait(timeout=120) == 0
    return log


def test_two_holders_serialize_on_one_gpu_id(tmp_path):
    # Private namespace (test-only override): both wrappers resolve it identically, and a test must
    # never queue behind -- or block -- a real tenant of /tmp/team_gpu_locks.
    _assert_serialized(_race(tmp_path, 3, tmp_path / "locks"))


def test_two_holders_serialize_in_the_default_namespace(tmp_path):
    gpu_id = 40000 + os.getpid() % 10000      # an id no real run uses
    lock = Path(DEFAULT_DIR) / f"gpu_{gpu_id}.lock"
    if not lock.exists() and not (os.access(DEFAULT_DIR, os.W_OK) or not Path(DEFAULT_DIR).exists()):
        pytest.skip(f"{DEFAULT_DIR} is owned by another uid on this box; cannot create {lock.name}")
    existed = lock.exists()
    try:
        _assert_serialized(_race(tmp_path, gpu_id, None))
        assert lock.exists(), "the default-namespace lock file was not created where both wrappers look"
    finally:
        if not existed and lock.exists():
            lock.unlink()


def test_help_and_selftest(tmp_path):
    r = subprocess.run(["bash", str(GEAK_LOCK), "--help"], check=False, capture_output=True, text=True, env=_env())
    assert r.returncode == 0 and "gpu_lock.sh" in r.stdout and "GEAK_GPU_USE_LOG" in r.stdout
    assert not (Path.cwd() / "gpu_--help.lock").exists()
    r = subprocess.run(["bash", str(GEAK_LOCK), "--selftest"], check=False, capture_output=True, text=True,
                       env=_env(), cwd=tmp_path, timeout=120)
    assert r.returncode == 0 and "GPU_LOCK SELFTEST PASS" in r.stdout, r.stdout + r.stderr


def test_use_log_mode_and_lock_env(tmp_path):
    use = tmp_path / "use.log"
    r = subprocess.run(["bash", str(GEAK_LOCK), "2", "bash", "-c",
                        'echo "$HIP_VISIBLE_DEVICES/$GEAK_GPU_LOCK_HELD"'],
                       check=False, capture_output=True, text=True, cwd=tmp_path, timeout=120,
                       env=_env(tmp_path / "locks", GEAK_GPU_USE_LOG=str(use)))
    assert r.returncode == 0 and r.stdout.strip() == "2/2", r.stderr
    assert '"mode":"pin"' in use.read_text()


def test_public_lock_has_no_optional_skill_runtime_dependency(tmp_path):
    # Even an explicit legacy broker setting cannot redirect the shared lock to a skill package.
    scheduler = tmp_path / "scheduler"
    scheduler.mkdir()
    marker = tmp_path / "unexpected-client"
    (scheduler / "gpu_client.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).touch(); raise SystemExit(99)\n")
    import socket
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock = tmp_path / "broker.sock"
    listener.bind(str(sock))
    env = _env(tmp_path / "locks", GEAK_GPU_BROKER="1", GEAK_GPU_BROKER_SCHED_DIR=str(scheduler),
               GEAK_GPU_BROKER_SOCK=str(sock))
    try:
        r = subprocess.run(["bash", str(GEAK_LOCK), "1", "echo", "public-lock"],
                           check=False, capture_output=True, text=True, cwd=tmp_path, env=env, timeout=10)
    finally:
        listener.close()
    assert r.returncode == 0 and r.stdout.strip() == "public-lock", r.stderr
    assert not marker.exists()


def test_command_exit_code_is_preserved(tmp_path):
    r = subprocess.run(["bash", str(GEAK_LOCK), "1", "bash", "-c", "exit 7"], check=False, cwd=tmp_path,
                       env=_env(tmp_path / "locks"), timeout=120)
    assert r.returncode == 7


@pytest.mark.parametrize("requested", ["2", "1", "1,3", "group:3,4", "pool:1:3,4"])
def test_nested_single_card_requests_fail_before_deadlocking(tmp_path, requested):
    r = subprocess.run(["bash", str(SHIM), "2", "bash", str(GEAK_LOCK), requested,
                        "echo", "unexpected-run"], check=False, capture_output=True, text=True, cwd=tmp_path,
                       env=_env(tmp_path / "locks"), timeout=10)
    assert r.returncode == 2 and "nested gpu_lock.sh call" in r.stderr, r.stdout + r.stderr
    assert "unexpected-run" not in r.stdout


@pytest.mark.parametrize("marker", [
    {"GEAK_GPU_LEASE_ACTIVE": "1", "GEAK_GPU_LEASE_IDS": "1,2"},
    {"GEAK_GPU_LEASE_IDS": "1,2"},
    {"GEAK_GPU_LOCKED_SET": "1,2"},
])
def test_complete_set_marker_refuses_any_new_allocation(tmp_path, marker):
    r = subprocess.run(["bash", str(GEAK_LOCK), "3", "echo", "unexpected-run"],
                       check=False, capture_output=True, text=True, cwd=tmp_path,
                       env=_env(tmp_path / "locks", **marker), timeout=10)
    assert r.returncode == 2 and "nested gpu_lock.sh call" in r.stderr
    assert not r.stdout


def test_ascending_nested_single_card_path_preserves_e2e_contract(tmp_path):
    r = subprocess.run(["bash", str(GEAK_LOCK), "1", "bash", str(GEAK_LOCK), "3", "bash", "-c",
                        'echo "$HIP_VISIBLE_DEVICES/$GEAK_GPU_LOCK_HELD"'],
                       check=False, capture_output=True, text=True, cwd=tmp_path,
                       env=_env(tmp_path / "locks"), timeout=10)
    assert r.returncode == 0 and r.stdout.strip() == "3/3", r.stderr


def test_shim_honors_the_public_allocation_fence(tmp_path):
    r = subprocess.run(["bash", str(SHIM), "2", "echo", "unexpected-run"],
                       check=False, capture_output=True, text=True, cwd=tmp_path,
                       env=_env(tmp_path / "locks", GEAK_GPU_ALLOWED="1"), timeout=10)
    assert r.returncode == 1 and "allocated only [1]" in r.stderr
    assert not r.stdout


def test_selftest_isolates_inherited_ownership_and_allocation(tmp_path):
    r = subprocess.run(["bash", str(SHIM), "--selftest"], check=False, capture_output=True, text=True,
                       cwd=tmp_path, timeout=15,
                       env=_env(tmp_path / "locks", GEAK_GPU_ALLOWED="99",
                                GEAK_GPU_LEASE_ACTIVE="1", GEAK_GPU_LEASE_IDS="99"))
    assert r.returncode == 0 and "GPU_LOCK SELFTEST PASS" in r.stdout, r.stdout + r.stderr
