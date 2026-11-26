# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEAK's shared lock namespace and inherited ownership; the Gluon skill uses the same lock path.

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
SKILL = REPO / "perf_knowledge/expert_skills/skills/gluon_authoring"
DEFAULT_DIR = "/tmp/team_gpu_locks"


def _env(lock_dir=None, kernel_tools=True, **extra):
    # kernel_tools: GEAK_KERNEL_TOOLS=1, the opt-in under which gpu_lock.sh honours GEAK_GPU_LOCK_DIR
    # and exports GEAK_GPU_LOCK_HELD (the path these tests exercise unless they say otherwise)
    env = dict(os.environ, GEAK_GPU_REQUIRE_IDLE="0", KERNEL_ENV_KEEP_ARCH="1",
               KERNEL_ENV_SKIP_ENUM_REAP="1")
    for k in ("GEAK_GPU_BROKER", "GEAK_GPU_ALLOWED", "GEAK_GPU_LOCK_HELD", "GEAK_GPU_LOCK_DIR",
              "GEAK_GPU_LOCKED_SET", "GEAK_GPU_LEASE_ACTIVE", "GEAK_GPU_LEASE_IDS",
              "GEAK_KERNEL_TOOLS", "KERNEL_TOOLS_PROFILE"):
        env.pop(k, None)
    if kernel_tools:
        env["GEAK_KERNEL_TOOLS"] = "1"
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
    assert f'LOCK_DIR="${{KERNEL_EXTRACT_LOCK_DIR:-{DEFAULT_DIR}}}"' in geak   # the default-run form


def test_default_run_ignores_the_override_and_exports_no_held_marker(tmp_path):
    # Without GEAK_KERNEL_TOOLS / KERNEL_TOOLS_PROFILE the lock behaves as before kernel tools existed.
    env = _env(tmp_path / "override", kernel_tools=False, KERNEL_EXTRACT_LOCK_DIR=str(tmp_path / "default"))
    r = subprocess.run(["bash", str(GEAK_LOCK), "2", "bash", "-c", 'echo "held=${GEAK_GPU_LOCK_HELD-unset}"'],
                       check=False, capture_output=True, text=True, cwd=tmp_path, timeout=60, env=env)
    assert r.returncode == 0 and r.stdout.strip() == "held=unset", r.stdout + r.stderr
    assert (tmp_path / "default" / "gpu_2.lock").exists()
    assert not (tmp_path / "override" / "gpu_2.lock").exists()


def _race(tmp_path, gpu_id, lock_dir):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    log = tmp_path / f"order_{gpu_id}.log"
    log.write_text("")
    env = _env(lock_dir)
    a = subprocess.Popen(["bash", str(GEAK_LOCK), str(gpu_id), "bash", "-c", HOLD, "_", "first", str(log)],
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


def test_ascending_nested_single_card_path_preserves_e2e_contract(tmp_path):
    r = subprocess.run(["bash", str(GEAK_LOCK), "1", "bash", str(GEAK_LOCK), "3", "bash", "-c",
                        'echo "$HIP_VISIBLE_DEVICES/$GEAK_GPU_LOCK_HELD"'],
                       check=False, capture_output=True, text=True, cwd=tmp_path,
                       env=_env(tmp_path / "locks"), timeout=10)
    assert r.returncode == 0 and r.stdout.strip() == "3/3", r.stderr


def test_skill_uses_the_shared_lock_and_carries_no_copy():
    assert not (SKILL / "scripts/gpu_lock.sh").exists()
    refs = [p for p in SKILL.rglob("*") if p.is_file() and p.suffix in (".md", ".py", ".sh")
            and "gpu_lock.sh" in p.read_text(errors="replace")]
    assert refs
    for p in refs:
        for line in p.read_text(errors="replace").splitlines():
            for tok in line.split():
                if tok.strip("`'\"(),").endswith("/gpu_lock.sh"):
                    assert "kernel_workflow/scripts/gpu_lock.sh" in tok, (p, line)


def test_lock_honors_the_public_allocation_fence(tmp_path):
    r = subprocess.run(["bash", str(GEAK_LOCK), "2", "echo", "unexpected-run"],
                       check=False, capture_output=True, text=True, cwd=tmp_path,
                       env=_env(tmp_path / "locks", GEAK_GPU_ALLOWED="1"), timeout=10)
    assert r.returncode == 1 and "allocated only [1]" in r.stderr
    assert not r.stdout


