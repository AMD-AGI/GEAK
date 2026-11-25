import contextlib
import fcntl
import importlib.util
import json
import multiprocessing
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPTS / "gpu_lease.py"
GPU_LOCK_WRAPPER = SCRIPTS / "gpu_lock.sh"
PROFILE_WRAPPER = SCRIPTS / "profile_kernel.sh"


def load_gpu_lease():
    spec = importlib.util.spec_from_file_location("gpu_lease", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_request_preserves_visibility_order_and_sorts_lock_order():
    gpu_lease = load_gpu_lease()

    request = gpu_lease.GpuRequest.from_fixed_ids("3,1,2", count=3)

    assert request.visible_ids == (3, 1, 2)
    assert request.lock_ids == (1, 2, 3)


@pytest.mark.parametrize(
    ("gpu_ids", "count", "message"),
    [
        ("", 1, "at least one GPU"),
        ("0,0", 2, "duplicate GPU"),
        ("0,a", 2, "invalid GPU"),
        ("0,1", 1, "count=1"),
    ],
)
def test_fixed_request_rejects_invalid_input(gpu_ids, count, message):
    gpu_lease = load_gpu_lease()

    with pytest.raises(ValueError, match=message):
        gpu_lease.GpuRequest.from_fixed_ids(gpu_ids, count=count)


def test_dynamic_request_rejects_count_larger_than_pool():
    gpu_lease = load_gpu_lease()

    with pytest.raises(ValueError, match="count=3"):
        gpu_lease.GpuRequest.from_pool("0,1", count=3)


def attempt_lease(lock_dir, gpu_ids, count, wait_timeout_s, result_queue):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids(gpu_ids, count=count)
    try:
        with gpu_lease.GpuLease(
            request,
            lock_dir=Path(lock_dir),
            wait_timeout_s=wait_timeout_s,
            poll_interval_s=0.01,
        ):
            result_queue.put("acquired")
    except gpu_lease.LeaseTimeout:
        result_queue.put("timeout")


def hold_lease(lock_dir, gpu_ids, count, ready_queue, release_queue):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids(gpu_ids, count=count)
    with gpu_lease.GpuLease(
        request,
        lock_dir=Path(lock_dir),
        wait_timeout_s=1.0,
        poll_interval_s=0.01,
    ):
        ready_queue.put("ready")
        release_queue.get(timeout=5)


def hold_dynamic_lease(lock_dir, pool_ids, count, ready_queue, release_queue):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_pool(pool_ids, count=count)
    with gpu_lease.GpuLease(
        request,
        lock_dir=Path(lock_dir),
        wait_timeout_s=2.0,
        poll_interval_s=0.01,
    ):
        ready_queue.put("ready")
        release_queue.get(timeout=5)


def hold_allocator_lock(lock_dir, ready_queue, duration_s):
    lock_path = Path(lock_dir)
    lock_path.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path / "allocator.lock", os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        ready_queue.put("ready")
        time.sleep(duration_s)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def run_attempt(lock_dir, gpu_ids, count, wait_timeout_s=0.15):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=attempt_lease,
        args=(str(lock_dir), gpu_ids, count, wait_timeout_s, result_queue),
    )
    proc.start()
    result = result_queue.get(timeout=5)
    proc.join(timeout=5)
    assert proc.exitcode == 0
    return result


def test_group_lease_blocks_overlapping_single_and_group_requests(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("1,0", count=2)

    with gpu_lease.GpuLease(
        request, lock_dir=tmp_path, wait_timeout_s=1.0, poll_interval_s=0.01
    ):
        assert run_attempt(tmp_path, "0", 1) == "timeout"
        assert run_attempt(tmp_path, "1,2", 2) == "timeout"
        assert run_attempt(tmp_path, "2,3", 2) == "acquired"


def test_dynamic_pool_selects_first_complete_available_group(tmp_path):
    gpu_lease = load_gpu_lease()
    ctx = multiprocessing.get_context("spawn")
    ready_queue = ctx.Queue()
    release_queue = ctx.Queue()
    holder = ctx.Process(
        target=hold_lease,
        args=(str(tmp_path), "0", 1, ready_queue, release_queue),
    )
    holder.start()
    assert ready_queue.get(timeout=5) == "ready"

    try:
        request = gpu_lease.GpuRequest.from_pool("0,1,2", count=2)
        with gpu_lease.GpuLease(
            request,
            lock_dir=tmp_path,
            wait_timeout_s=0.2,
            poll_interval_s=0.005,
        ) as lease:
            assert lease.selected_ids == (1, 2)
    finally:
        release_queue.put("release")
        holder.join(timeout=5)
        assert holder.exitcode == 0


def test_dynamic_pool_skips_lockable_but_externally_busy_group(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_pool("0,1,2", count=2)

    with gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.1,
        poll_interval_s=0.005,
        idle_checker=lambda gpu_id: gpu_id != 0,
    ) as lease:
        assert lease.selected_ids == (1, 2)


def test_older_group_request_blocks_younger_single_backfill(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    holder_ready = ctx.Queue()
    holder_release = ctx.Queue()
    holder = ctx.Process(
        target=hold_lease,
        args=(str(tmp_path), "0", 1, holder_ready, holder_release),
    )
    holder.start()
    assert holder_ready.get(timeout=5) == "ready"

    group_ready = ctx.Queue()
    group_release = ctx.Queue()
    group = ctx.Process(
        target=hold_dynamic_lease,
        args=(str(tmp_path), "0,1", 2, group_ready, group_release),
    )
    group.start()
    deadline = time.monotonic() + 1
    while (
        not list(tmp_path.glob("request_*.json"))
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert list(tmp_path.glob("request_*.json"))

    try:
        assert run_attempt(
            tmp_path, "1", 1, wait_timeout_s=0.05
        ) == "timeout"
        holder_release.put("release")
        holder.join(timeout=5)
        assert holder.exitcode == 0
        assert group_ready.get(timeout=5) == "ready"
    finally:
        group_release.put("release")
        group.join(timeout=5)
        if holder.is_alive():
            holder_release.put("release")
            holder.join(timeout=5)

    assert group.exitcode == 0


def test_dead_pending_request_is_removed_without_blocking(tmp_path):
    stale = tmp_path / "request_dead.json"
    stale.write_text(
        json.dumps(
            {
                "request_id": "dead",
                "manager_pid": 999999999,
                "pid_namespace": os.readlink("/proc/self/ns/pid"),
                "pool_ids": [0, 1],
                "count": 2,
                "started_at_ns": 1,
            }
        )
    )

    assert run_attempt(tmp_path, "0", 1) == "acquired"
    assert not stale.exists()


def test_host_scope_names_every_file_after_the_host_and_leaves_other_hosts_alone(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)
    # Another machine's files on the shared lease dir: its pids do not exist on this host.
    other = {
        "request": tmp_path / "hostB.request_x.json",
        "lease": tmp_path / "hostB.lease_x.json",
    }
    other["request"].write_text(json.dumps({
        "request_id": "x", "manager_pid": 999999999,
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "pool_ids": [0], "count": 1, "started_at_ns": 1,
    }))
    other["lease"].write_text(json.dumps({"gpu_ids": [0], "child_pgid": 999999999}))

    with gpu_lease.GpuLease(
        request, lock_dir=tmp_path, wait_timeout_s=1.0, poll_interval_s=0.01,
        host_scope="hostA",
    ) as lease:
        assert lease.metadata_path.name.startswith("hostA.lease_")
        # _probe_request_lock finds a waiter's lock as its request json's .lock sibling.
        assert lease.request_path.name.startswith("hostA.request_")
        assert lease.request_lock_path == lease.request_path.with_suffix(".lock")
        assert lease.gpu_lock_path(0) == tmp_path / "hostA.gpu0.lock"
        assert (tmp_path / "hostA.allocator.lock").exists()
        assert not (tmp_path / "gpu_0.lock").exists()
        assert not (tmp_path / "allocator.lock").exists()

    assert other["request"].exists() and other["lease"].exists()
    unscoped = gpu_lease.GpuLease(request, lock_dir=tmp_path, wait_timeout_s=0.1)
    assert unscoped.gpu_lock_path(0) == tmp_path / "gpu_0.lock"


def test_host_scoped_lease_waits_for_the_card_lease_a_serving_leg_holds(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("0,1", count=2)
    # mn.py's lease_file: <host>.gpu<N>.lock on the shared mount.
    with open(tmp_path / "hostA.gpu1.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(gpu_lease.LeaseTimeout), gpu_lease.GpuLease(
            request, lock_dir=tmp_path, wait_timeout_s=0.1, poll_interval_s=0.01,
            host_scope="hostA",
        ):
            pass
        with gpu_lease.GpuLease(
            request, lock_dir=tmp_path, wait_timeout_s=0.1, poll_interval_s=0.01,
            host_scope="hostB",
        ) as lease:
            assert lease.selected_ids == (0, 1)


def test_failed_group_acquire_releases_partial_locks(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready_queue = ctx.Queue()
    release_queue = ctx.Queue()
    holder = ctx.Process(
        target=hold_lease,
        args=(str(tmp_path), "1", 1, ready_queue, release_queue),
    )
    holder.start()
    assert ready_queue.get(timeout=5) == "ready"

    try:
        assert run_attempt(tmp_path, "0,1", 2) == "timeout"
        assert run_attempt(tmp_path, "0", 1) == "acquired"
    finally:
        release_queue.put("release")
        holder.join(timeout=5)
        assert holder.exitcode == 0


def test_wait_timeout_is_hard_limit_when_allocator_is_busy(tmp_path):
    gpu_lease = load_gpu_lease()
    ctx = multiprocessing.get_context("spawn")
    ready_queue = ctx.Queue()
    holder = ctx.Process(
        target=hold_allocator_lock,
        args=(str(tmp_path), ready_queue, 0.5),
    )
    holder.start()
    assert ready_queue.get(timeout=5) == "ready"

    started = time.monotonic()
    try:
        request = gpu_lease.GpuRequest.from_fixed_ids("0,1", count=2)
        with pytest.raises(gpu_lease.LeaseTimeout), gpu_lease.GpuLease(
            request,
            lock_dir=tmp_path,
            wait_timeout_s=0.05,
            poll_interval_s=0.005,
        ):
            raise AssertionError("busy allocator must not be acquired")
        elapsed = time.monotonic() - started
    finally:
        holder.join(timeout=5)
        assert holder.exitcode == 0

    assert elapsed < 0.3


def test_idle_check_rejects_entire_group_and_releases_locks(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("0,1", count=2)
    busy = {1}

    with pytest.raises(gpu_lease.LeaseTimeout), gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.02,
        poll_interval_s=0.005,
        idle_checker=lambda gpu_id: gpu_id not in busy,
    ):
        raise AssertionError("busy group must not be acquired")

    assert run_attempt(tmp_path, "0", 1) == "acquired"
    busy.clear()
    with gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.1,
        poll_interval_s=0.005,
        idle_checker=lambda gpu_id: gpu_id not in busy,
    ):
        pass


def test_lease_metadata_exists_while_held_and_is_removed_on_release(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("2,0", count=2)

    with gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.1,
    ) as lease:
        metadata = json.loads(lease.metadata_path.read_text())
        assert metadata["manager_pid"] == os.getpid()
        assert metadata["gpu_ids"] == [2, 0]
        assert metadata["lock_ids"] == [0, 2]

    assert not lease.metadata_path.exists()


def test_live_stale_lease_metadata_blocks_reallocation(tmp_path):
    gpu_lease = load_gpu_lease()
    stale_process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    stale_path = tmp_path / "lease_stale.json"
    stale_path.write_text(
        json.dumps(
            {
                "lease_id": "stale",
                "manager_pid": 999999,
                "child_pgid": stale_process.pid,
                "gpu_ids": [0, 1],
                "lock_ids": [0, 1],
            }
        )
    )
    request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)

    try:
        with pytest.raises(gpu_lease.LeaseTimeout), gpu_lease.GpuLease(
            request,
            lock_dir=tmp_path,
            wait_timeout_s=0.02,
            poll_interval_s=0.005,
        ):
            raise AssertionError("live stale lease must block reuse")
    finally:
        os.killpg(stale_process.pid, signal.SIGKILL)
        stale_process.wait(timeout=5)

    with gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.1,
    ):
        pass
    assert not stale_path.exists()


def test_zombie_only_process_group_does_not_keep_stale_lease_live(tmp_path):
    gpu_lease = load_gpu_lease()
    child = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(0)"],
        start_new_session=True,
    )
    deadline = time.monotonic() + 2
    while process_is_live(child.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert Path(f"/proc/{child.pid}").exists()
    stale_path = tmp_path / "lease_zombie.json"
    stale_path.write_text(
        json.dumps(
            {
                "lease_id": "zombie",
                "manager_pid": 999999999,
                "pid_namespace": os.readlink("/proc/self/ns/pid"),
                "gpu_ids": [0],
                "lock_ids": [0],
                "child_pgid": child.pid,
                "started_at_ns": 1,
            }
        )
    )

    try:
        request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)
        with gpu_lease.GpuLease(
            request,
            lock_dir=tmp_path,
            wait_timeout_s=0.05,
            poll_interval_s=0.005,
        ):
            pass
    finally:
        child.wait(timeout=5)

    assert not stale_path.exists()


def test_stale_lease_from_other_pid_namespace_fails_closed(tmp_path):
    gpu_lease = load_gpu_lease()
    stale_path = tmp_path / "lease_other_namespace.json"
    stale_path.write_text(
        json.dumps(
            {
                "lease_id": "other-namespace",
                "manager_pid": 1,
                "child_pgid": 999999,
                "pid_namespace": "pid:[different]",
                "gpu_ids": [0],
                "lock_ids": [0],
            }
        )
    )
    request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)

    with pytest.raises(gpu_lease.LeaseTimeout), gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.02,
        poll_interval_s=0.005,
    ):
        raise AssertionError("unverifiable cross-namespace lease must block reuse")
    assert stale_path.exists()


def test_sysfs_idle_checker_uses_busy_and_vram_thresholds(tmp_path):
    gpu_lease = load_gpu_lease()
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text("4\n")
    (device / "mem_info_vram_used").write_text(str(512 * 1024 * 1024))
    checker = gpu_lease.SysfsIdleChecker(
        drm_root=tmp_path,
        max_busy_pct=5,
        max_vram_mb=1024,
    )

    assert checker(0)

    (device / "gpu_busy_percent").write_text("6\n")
    assert not checker(0)

    (device / "gpu_busy_percent").write_text("0\n")
    (device / "mem_info_vram_used").write_text(str(2048 * 1024 * 1024))
    assert not checker(0)

    checker_without_vram_gate = gpu_lease.SysfsIdleChecker(
        drm_root=tmp_path,
        max_busy_pct=5,
        max_vram_mb=-1,
    )
    assert checker_without_vram_gate(0)


def test_amd_smi_mapping_resolves_logical_gpu_to_pci_sysfs(tmp_path):
    gpu_lease = load_gpu_lease()
    payload = json.dumps(
        [
            {"gpu": 0, "bdf": "0000:05:00.0"},
            {"gpu": 1, "bdf": "0000:75:00.0"},
        ]
    )

    mapping = gpu_lease.parse_amd_smi_device_map(
        payload,
        pci_root=tmp_path,
    )

    assert mapping == {
        0: tmp_path / "0000:05:00.0",
        1: tmp_path / "0000:75:00.0",
    }


def test_required_idle_check_fails_closed_when_telemetry_is_missing(tmp_path):
    gpu_lease = load_gpu_lease()
    checker = gpu_lease.SysfsIdleChecker(
        drm_root=tmp_path,
        max_busy_pct=5,
        max_vram_mb=-1,
        fail_open=False,
    )

    assert not checker(0)


def test_dynamic_pool_allows_partial_mapping_when_enough_candidates_remain(
    tmp_path, monkeypatch
):
    gpu_lease = load_gpu_lease()
    devices = {}
    for gpu_id in (1, 2):
        device = tmp_path / f"card{gpu_id}" / "device"
        device.mkdir(parents=True)
        (device / "gpu_busy_percent").write_text("0\n")
        devices[gpu_id] = device
    monkeypatch.setattr(
        gpu_lease, "discover_amd_smi_device_map", lambda: devices
    )
    captured = {}

    def fake_run(request, command, **kwargs):
        captured["idle_checker"] = kwargs["idle_checker"]
        return 0

    monkeypatch.setattr(gpu_lease, "run_command", fake_run)

    result = gpu_lease.main(
        [
            "run",
            "--pool",
            "0,1,2",
            "--count",
            "2",
            "--require-idle",
            "--",
            "true",
        ]
    )

    assert result == 0
    assert captured["idle_checker"](0) is False
    assert captured["idle_checker"](1) is True
    assert captured["idle_checker"](2) is True


def test_run_command_sets_group_environment_and_preserves_exit_code(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("3,1", count=2)
    output = tmp_path / "env.json"
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, pathlib, sys; "
            "data = dict("
            "hip=os.environ.get('HIP_VISIBLE_DEVICES'), "
            "cuda=os.environ.get('CUDA_VISIBLE_DEVICES'), "
            "group=os.environ.get('GEAK_GPU_GROUP'), "
            "active=os.environ.get('GEAK_GPU_LEASE_ACTIVE')); "
            f"pathlib.Path({str(output)!r}).write_text(json.dumps(data)); "
            "sys.exit(7)"
        ),
    ]

    return_code = gpu_lease.run_command(
        request,
        command,
        lock_dir=tmp_path / "locks",
        wait_timeout_s=1.0,
        run_timeout_s=5.0,
        term_grace_s=0.1,
    )

    assert return_code == 7
    assert json.loads(output.read_text()) == {
        "hip": "3,1",
        "cuda": "3,1",
        "group": "3,1",
        "active": "1",
    }
    assert run_attempt(tmp_path / "locks", "1", 1) == "acquired"


def test_run_command_maps_child_signal_to_shell_exit_code(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)

    return_code = gpu_lease.run_command(
        request,
        [
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
        lock_dir=tmp_path / "locks",
        wait_timeout_s=1.0,
        run_timeout_s=5.0,
        term_grace_s=0.1,
    )

    assert return_code == 128 + signal.SIGTERM


def test_cli_maps_missing_command_to_127_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "run",
            "--fixed-ids",
            "0",
            "--lock-dir",
            str(tmp_path / "locks"),
            "--",
            "definitely-not-a-real-command-geak",
        ],
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 127
    assert "Traceback" not in result.stderr


def process_is_live(pid):
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    fields = stat_path.read_text().split()
    return len(fields) > 2 and fields[2] != "Z"


def test_run_command_timeout_kills_entire_process_group_and_releases_locks(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("0,1", count=2)
    pids_path = tmp_path / "pids.json"
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, pathlib, subprocess, time; "
            "child = subprocess.Popen(['sleep', '60']); "
            f"pathlib.Path({str(pids_path)!r}).write_text(json.dumps("
            "{'parent': os.getpid(), 'child': child.pid})); "
            "time.sleep(60)"
        ),
    ]

    return_code = gpu_lease.run_command(
        request,
        command,
        lock_dir=tmp_path / "locks",
        wait_timeout_s=1.0,
        run_timeout_s=0.2,
        term_grace_s=0.1,
    )

    assert return_code == gpu_lease.TIMEOUT_EXIT_CODE
    pids = json.loads(pids_path.read_text())
    deadline = time.monotonic() + 2.0
    while any(process_is_live(pid) for pid in pids.values()) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any(process_is_live(pid) for pid in pids.values())
    assert run_attempt(tmp_path / "locks", "0", 1) == "acquired"


def test_timeout_kills_descendant_that_ignores_sigterm(tmp_path):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("0,1", count=2)
    child_ready = tmp_path / "child-ready"
    pids_path = tmp_path / "ignore-term-pids.json"
    child_code = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(child_ready)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent_code = (
        "import json, os, pathlib, subprocess, sys, time; "
        f"ready = pathlib.Path({str(child_ready)!r}); "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "\nwhile not ready.exists(): time.sleep(0.01)\n"
        f"pathlib.Path({str(pids_path)!r}).write_text(json.dumps("
        "{'parent': os.getpid(), 'child': child.pid})); "
        "time.sleep(60)"
    )

    return_code = gpu_lease.run_command(
        request,
        [sys.executable, "-c", parent_code],
        lock_dir=tmp_path / "locks",
        wait_timeout_s=1.0,
        run_timeout_s=0.5,
        term_grace_s=0.1,
    )

    assert return_code == gpu_lease.TIMEOUT_EXIT_CODE
    pids = json.loads(pids_path.read_text())
    try:
        deadline = time.monotonic() + 2
        while any(process_is_live(pid) for pid in pids.values()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(process_is_live(pid) for pid in pids.values())
    finally:
        for pid in pids.values():
            if process_is_live(pid):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("parent_exit_code", [0, 7])
def test_parent_exit_does_not_leave_background_descendant(
    tmp_path, parent_exit_code
):
    gpu_lease = load_gpu_lease()
    request = gpu_lease.GpuRequest.from_fixed_ids("0,1", count=2)
    pids_path = tmp_path / f"background-{parent_exit_code}.json"
    parent_code = (
        "import json, os, pathlib, subprocess, sys; "
        "child = subprocess.Popen(['sleep', '60']); "
        f"pathlib.Path({str(pids_path)!r}).write_text(json.dumps("
        "{'parent': os.getpid(), 'child': child.pid})); "
        f"sys.exit({parent_exit_code})"
    )

    return_code = gpu_lease.run_command(
        request,
        [sys.executable, "-c", parent_code],
        lock_dir=tmp_path / "locks",
        wait_timeout_s=1.0,
        run_timeout_s=5.0,
        term_grace_s=0.1,
    )

    assert return_code == parent_exit_code
    pids = json.loads(pids_path.read_text())
    try:
        deadline = time.monotonic() + 2
        while process_is_live(pids["child"]) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not process_is_live(pids["child"])
        assert run_attempt(tmp_path / "locks", "0", 1) == "acquired"
    finally:
        if process_is_live(pids["child"]):
            os.kill(pids["child"], signal.SIGKILL)


def test_metadata_update_failure_terminates_started_process_group(tmp_path):
    gpu_lease = load_gpu_lease()
    original_lease = gpu_lease.GpuLease
    pids_path = tmp_path / "metadata-failure-pids.json"

    class FailingMetadataLease(original_lease):
        def update_metadata(self, **values):
            deadline = time.monotonic() + 1
            while not pids_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise OSError("simulated metadata write failure")

    gpu_lease.GpuLease = FailingMetadataLease
    request = gpu_lease.GpuRequest.from_fixed_ids("0,1", count=2)
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, pathlib, subprocess, time; "
            "child = subprocess.Popen(['sleep', '60']); "
            f"pathlib.Path({str(pids_path)!r}).write_text(json.dumps("
            "{'parent': os.getpid(), 'child': child.pid})); "
            "time.sleep(60)"
        ),
    ]

    try:
        with pytest.raises(OSError, match="metadata"):
            gpu_lease.run_command(
                request,
                command,
                lock_dir=tmp_path / "locks",
                wait_timeout_s=1,
                run_timeout_s=60,
                term_grace_s=0.1,
            )
        pids = json.loads(pids_path.read_text())
        deadline = time.monotonic() + 2
        while any(process_is_live(pid) for pid in pids.values()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(process_is_live(pid) for pid in pids.values())
    finally:
        gpu_lease.GpuLease = original_lease
        if pids_path.exists():
            for pid in json.loads(pids_path.read_text()).values():
                if process_is_live(pid):
                    os.kill(pid, signal.SIGKILL)


def test_signal_during_popen_does_not_leak_started_child(tmp_path):
    gpu_lease = load_gpu_lease()
    original_popen = gpu_lease.subprocess.Popen
    child_pid = None

    def racing_popen(*args, **kwargs):
        nonlocal child_pid
        process = original_popen(*args, **kwargs)
        child_pid = process.pid
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    gpu_lease.subprocess.Popen = racing_popen
    request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)
    try:
        with pytest.raises(SystemExit) as raised:
            gpu_lease.run_command(
                request,
                ["sleep", "60"],
                lock_dir=tmp_path / "locks",
                wait_timeout_s=1,
                run_timeout_s=60,
                term_grace_s=0.1,
            )
        assert raised.value.code == 128 + signal.SIGTERM
        assert child_pid is not None
        deadline = time.monotonic() + 2
        while process_is_live(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not process_is_live(child_pid)
    finally:
        gpu_lease.subprocess.Popen = original_popen
        if child_pid is not None and process_is_live(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def wrapper_env(tmp_path, **extra):
    return {
        **os.environ,
        "KERNEL_EXTRACT_LOCK_DIR": str(tmp_path / "locks"),
        "GEAK_GPU_REQUIRE_IDLE": "0",
        "GEAK_GPU_POOL_WAIT": "1",
        "GEAK_GPU_RUN_TIMEOUT": "5",
        "KERNEL_ENV_SKIP_ENUM_REAP": "1",
        "KERNEL_ENV_KEEP_ARCH": "1",
        **extra,
    }


def run_wrapper(tmp_path, gpu_spec, *command, cwd=None, env=None, timeout=10):
    return subprocess.run(
        ["bash", str(GPU_LOCK_WRAPPER), gpu_spec, *command],
        cwd=cwd or tmp_path,
        env=env or wrapper_env(tmp_path),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def write_visible_devices(path):
    return (
        sys.executable,
        "-c",
        (
            "import os; from pathlib import Path; "
            f"Path({str(path)!r}).write_text(os.environ['HIP_VISIBLE_DEVICES'])"
        ),
    )


def start_holder(tmp_path, gpu_spec, ready, hold_s):
    holder = subprocess.Popen(
        [
            "bash",
            str(GPU_LOCK_WRAPPER),
            gpu_spec,
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import time; "
                f"Path({str(ready)!r}).write_text('ready'); time.sleep({hold_s})"
            ),
        ],
        cwd=tmp_path,
        env=wrapper_env(tmp_path),
    )
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    return holder


def test_group_wrapper_preserves_command_output_exit_code_and_build_environment(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "wrapper.json"

    result = run_wrapper(
        tmp_path,
        "group:2,0",
        "--",
        sys.executable,
        "-c",
        (
            "import json, os, pathlib, sys; "
            "data = dict("
            "hip=os.environ.get('HIP_VISIBLE_DEVICES'), "
            "group=os.environ.get('GEAK_GPU_GROUP'), "
            "torch_ext=os.environ.get('TORCH_EXTENSIONS_DIR')); "
            f"pathlib.Path({str(output)!r}).write_text(json.dumps(data)); "
            "print('CHILD_STDOUT'); "
            "sys.exit(9)"
        ),
        cwd=workspace,
    )

    assert result.returncode == 9
    assert "CHILD_STDOUT" in result.stdout
    assert json.loads(output.read_text()) == {
        "hip": "2,0",
        "group": "2,0",
        "torch_ext": str(workspace / ".torch_ext"),
    }


def test_group_wrapper_rejects_busy_gpu_from_configured_sysfs(tmp_path):
    sysfs_root = tmp_path / "drm"
    for gpu_id, busy in ((0, 0), (1, 100)):
        device = sysfs_root / f"card{gpu_id}" / "device"
        device.mkdir(parents=True)
        (device / "gpu_busy_percent").write_text(f"{busy}\n")
        (device / "mem_info_vram_used").write_text("0\n")
    marker = tmp_path / "ran"
    env = wrapper_env(
        tmp_path,
        GEAK_GPU_SYSFS_ROOT=str(sysfs_root),
        GEAK_GPU_REQUIRE_IDLE="1",
        GEAK_GPU_POOL_WAIT="0.05",
    )

    result = run_wrapper(
        tmp_path,
        "group:0,1",
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        env=env,
    )

    assert result.returncode == 2
    assert not marker.exists()
    assert "failed to acquire 2 GPU(s)" in result.stderr


def test_group_lease_blocks_single_gpu_wrapper_on_shared_lock(tmp_path):
    holder = start_holder(tmp_path, "group:0,1", tmp_path / "ready", 0.4)
    started = time.monotonic()
    single = run_wrapper(
        tmp_path, "0", sys.executable, "-c", "print('single-acquired')"
    )
    elapsed = time.monotonic() - started
    holder.wait(timeout=5)

    assert holder.returncode == 0
    assert single.returncode == 0
    assert "single-acquired" in single.stdout
    assert elapsed >= 0.25


def test_single_gpu_holder_blocks_group_until_released(tmp_path):
    holder = start_holder(tmp_path, "1", tmp_path / "ready", 0.4)
    output = tmp_path / "group-visible"
    started = time.monotonic()
    result = run_wrapper(tmp_path, "group:0,1", *write_visible_devices(output))
    elapsed = time.monotonic() - started
    holder.wait(timeout=5)

    assert result.returncode == 0
    assert output.read_text() == "0,1"
    assert elapsed >= 0.25


@pytest.mark.parametrize("gpu_spec", ["0", "0,1"])
def test_a_descendant_that_outlives_the_command_does_not_keep_the_lane(tmp_path, gpu_spec):
    pid_file = tmp_path / "kid.pid"
    run_wrapper(tmp_path, gpu_spec, "bash", "-c",
                f"sleep 120 >/dev/null 2>&1 & echo $! > {shlex.quote(str(pid_file))}")
    kid = int(pid_file.read_text())
    try:
        assert process_is_live(kid)
        with open(tmp_path / "locks" / "gpu_0.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # raises if the descendant holds it
    finally:
        os.kill(kid, signal.SIGKILL)


@pytest.mark.parametrize("gpu_spec", ["0", "0,1"])
def test_a_blocked_waiter_names_the_process_holding_the_lock(tmp_path, gpu_spec):
    (tmp_path / "locks").mkdir()
    lanes = [str(tmp_path / "locks" / f"gpu_{g}.lock") for g in gpu_spec.split(",")]
    script = (f"import fcntl, time\nfs = [open(p, 'a') for p in {lanes!r}]\n"
              "[fcntl.flock(f, fcntl.LOCK_EX) for f in fs]\nprint(flush=True); time.sleep(60)")
    holder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    holder.stdout.readline()                                    # every lane is held
    waiter = subprocess.Popen(["bash", str(GPU_LOCK_WRAPPER), gpu_spec, "true"], cwd=tmp_path,
                              env=wrapper_env(tmp_path, GEAK_GPU_POOL_WAIT="30"),
                              stderr=subprocess.PIPE, text=True)
    try:
        lines = [waiter.stderr.readline() for _ in range(1 + len(lanes))]
    finally:
        holder.kill()
        holder.wait()
    _, err = waiter.communicate(timeout=30)
    assert waiter.returncode == 0, err                          # the wait ends once the lock frees
    held = [line for line in lines + err.splitlines() if "lock is open in pid" in line]
    assert lines[0].startswith("NOTE: ") and len(held) == len(lanes), lines + [err]
    for g, line in enumerate(held):                             # the holder, never the waiter
        assert line.startswith(f"  GPU {g} lock is open in pid {holder.pid} (ppid {os.getpid()}): ")


def test_pool_wrapper_selects_available_n_gpu_group(tmp_path):
    holder = start_holder(tmp_path, "group:0", tmp_path / "ready", 0.5)
    selected = tmp_path / "selected"
    try:
        result = run_wrapper(tmp_path, "pool:2:0,1,2", *write_visible_devices(selected))
    finally:
        holder.wait(timeout=5)

    assert result.returncode == 0
    assert selected.read_text() == "1,2"


@pytest.mark.parametrize(
    ("gpu_spec", "expected"),
    [
        ("group:2,0", "2,0"),
        ("pool:2:0,1,2", "0,1"),
    ],
)
def test_compact_gpu_spec_sets_visible_devices(tmp_path, gpu_spec, expected):
    output = tmp_path / "compact-spec"
    result = run_wrapper(tmp_path, gpu_spec, *write_visible_devices(output))

    assert result.returncode == 0
    assert output.read_text() == expected


def test_group_spec_respects_allocation_fence(tmp_path):
    marker = tmp_path / "ran"
    env = wrapper_env(tmp_path, GEAK_GPU_ALLOWED="0,1")

    result = run_wrapper(
        tmp_path,
        "pool:2:0,1,2",
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        env=env,
    )

    assert result.returncode == 1
    assert not marker.exists()
    assert "GPU(s) 2" in result.stderr


def test_group_wrapper_run_timeout_kills_command(tmp_path):
    env = wrapper_env(tmp_path, GEAK_GPU_RUN_TIMEOUT="0.2")
    started = time.monotonic()
    result = run_wrapper(
        tmp_path, "group:0", sys.executable, "-c", "import time; time.sleep(30)", env=env
    )

    assert result.returncode == 124
    assert time.monotonic() - started < 10


def test_group_wrapper_writes_use_log(tmp_path):
    log = tmp_path / "use.jsonl"
    env = wrapper_env(tmp_path, GEAK_GPU_USE_LOG=str(log))
    result = run_wrapper(tmp_path, "pool:2:0,1,2", "true", env=env)

    assert result.returncode == 0
    record = json.loads(log.read_text().splitlines()[0])
    assert record["mode"] == "group"
    assert record["gpus"] == [0, 1]
    assert record["pool"] == "0,1,2"


def test_profile_wrapper_accepts_compact_group_spec(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "profile-visible"
    profile_dir = tmp_path / "profile"
    benchmark = (
        "python -c \"import os; from pathlib import Path; "
        f"Path({str(output)!r}).write_text(os.environ['HIP_VISIBLE_DEVICES'])\""
    )
    # A pinned arch skips the rocminfo identity probe, so this runs on CPU-only CI.
    env = wrapper_env(
        tmp_path,
        PROFILER_PRIORITY="definitely-not-a-profiler",
        WARMUP_RUNS="1",
        PYTORCH_ROCM_ARCH="gfx950",
    )

    result = subprocess.run(
        ["bash", str(PROFILE_WRAPPER), "group:1,0", benchmark, str(profile_dir)],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert output.read_text() == "1,0"
    assert (profile_dir / "profile_report.txt").exists()


@pytest.mark.parametrize("gpu_spec", ["0", "0,1", "group:0,1"])
def test_wrapper_rejects_nested_request_inside_active_lease(tmp_path, gpu_spec):
    marker = tmp_path / "nested-ran"
    env = wrapper_env(tmp_path, GEAK_GPU_LEASE_ACTIVE="1", GEAK_GPU_LEASE_IDS="0,1")

    result = run_wrapper(
        tmp_path,
        gpu_spec,
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        env=env,
    )

    assert result.returncode == 2
    assert not marker.exists()
    assert "nested gpu_lock.sh call" in result.stderr


def test_cli_signal_terminates_child_process_group_and_releases_locks(tmp_path):
    pids_path = tmp_path / "signal-pids.json"
    command = [
        sys.executable,
        str(MODULE_PATH),
        "run",
        "--fixed-ids",
        "0,1",
        "--lock-dir",
        str(tmp_path / "locks"),
        "--wait-timeout",
        "1",
        "--run-timeout",
        "60",
        "--term-grace",
        "0.1",
        "--",
        sys.executable,
        "-c",
        (
            "import json, os, pathlib, subprocess, time; "
            "child = subprocess.Popen(['sleep', '60']); "
            f"pathlib.Path({str(pids_path)!r}).write_text(json.dumps("
            "{'parent': os.getpid(), 'child': child.pid})); "
            "time.sleep(60)"
        ),
    ]
    manager = subprocess.Popen(command)
    deadline = time.monotonic() + 5
    while not pids_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pids_path.exists()

    manager.send_signal(signal.SIGTERM)
    manager.wait(timeout=5)

    pids = json.loads(pids_path.read_text())
    try:
        deadline = time.monotonic() + 2
        while any(process_is_live(pid) for pid in pids.values()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(process_is_live(pid) for pid in pids.values())
        assert run_attempt(tmp_path / "locks", "0", 1) == "acquired"
    finally:
        for pid in pids.values():
            if process_is_live(pid):
                os.kill(pid, signal.SIGKILL)


def test_manager_sigkill_does_not_release_locks_while_launcher_survives(tmp_path):
    child_pid_path = tmp_path / "sigkill-child"
    command = [
        sys.executable,
        str(MODULE_PATH),
        "run",
        "--fixed-ids",
        "0,1",
        "--lock-dir",
        str(tmp_path / "locks"),
        "--wait-timeout",
        "1",
        "--run-timeout",
        "60",
        "--",
        sys.executable,
        "-c",
        (
            "import os, time; from pathlib import Path; "
            f"Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        ),
    ]
    manager = subprocess.Popen(command)
    deadline = time.monotonic() + 5
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text())

    os.kill(manager.pid, signal.SIGKILL)
    manager.wait(timeout=5)
    for metadata in (tmp_path / "locks").glob("lease_*.json"):
        metadata.unlink()
    try:
        assert process_is_live(child_pid)
        assert run_attempt(
            tmp_path / "locks", "0", 1, wait_timeout_s=0.05
        ) == "timeout"
    finally:
        if process_is_live(child_pid):
            os.killpg(child_pid, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while process_is_live(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)

    assert run_attempt(tmp_path / "locks", "0", 1) == "acquired"


def test_reap_own_orphans_only_kills_our_reparented_leaks(tmp_path, monkeypatch):
    gpu_lease = load_gpu_lease()

    root = tmp_path / "workspace"
    under = str(root / "build")
    outside = str(tmp_path / "somewhere_else")
    my_ns = "pid:[4026531836]"

    # pid -> (ppid, cwd, cmdline, namespace, has lease marker)
    fake = {
        11: (1, under, "python bench.py --bs 8192", my_ns, True),  # leaked rank -> reap
        12: (1, under, "torchrun --nproc 8 bench.py", my_ns, False),  # launcher -> reap
        21: (4321, under, "python bench.py", my_ns, True),  # live parent -> spare
        22: (1, outside, "python bench.py", my_ns, True),  # not under root -> spare
        23: (1, under, "python unrelated_service.py", my_ns, False),  # unattributed -> spare
        24: (1, under, "python bench.py", "pid:[4026999999]", True),  # other ns -> spare
    }

    monkeypatch.setattr(gpu_lease.os, "listdir", lambda path: [str(p) for p in fake])
    monkeypatch.setattr(gpu_lease, "_proc_ppid", lambda pid: fake.get(pid, (None,))[0])
    monkeypatch.setattr(gpu_lease, "_proc_cwd", lambda pid: fake[pid][1])
    monkeypatch.setattr(gpu_lease, "_proc_cmdline", lambda pid: fake[pid][2])
    monkeypatch.setattr(gpu_lease, "_proc_has_lease_marker", lambda pid: fake[pid][4])
    monkeypatch.setattr(gpu_lease, "_pid_namespace", lambda: my_ns)
    monkeypatch.setattr(gpu_lease, "_pid_namespace_for_pid", lambda pid: fake[pid][3])

    killed = []
    reaped = gpu_lease.reap_own_orphans(
        root, killer=lambda pid, sig: killed.append((pid, sig))
    )

    assert sorted(reaped) == [11, 12]
    assert sorted(pid for pid, _ in killed) == [11, 12]
    assert all(sig == signal.SIGKILL for _, sig in killed)


def _write_other_namespace_lease(path):
    path.write_text(
        json.dumps(
            {
                "lease_id": "other-namespace",
                "manager_pid": 1,
                "child_pgid": 999999,
                "pid_namespace": "pid:[different]",
                "gpu_ids": [0],
                "lock_ids": [0],
            }
        )
    )


def test_other_namespace_lease_reclaimed_when_gpus_verifiably_idle(tmp_path):
    gpu_lease = load_gpu_lease()
    stale_path = tmp_path / "lease_other_namespace.json"
    _write_other_namespace_lease(stale_path)
    request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)

    with gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.5,
        poll_interval_s=0.005,
        stale_idle_checker=lambda gpu_id: True,
    ) as lease:
        assert lease.selected_ids == (0,)
    assert not stale_path.exists()


def test_other_namespace_lease_kept_when_gpu_not_idle(tmp_path):
    gpu_lease = load_gpu_lease()
    stale_path = tmp_path / "lease_other_namespace.json"
    _write_other_namespace_lease(stale_path)
    request = gpu_lease.GpuRequest.from_fixed_ids("0", count=1)

    with pytest.raises(gpu_lease.LeaseTimeout), gpu_lease.GpuLease(
        request,
        lock_dir=tmp_path,
        wait_timeout_s=0.02,
        poll_interval_s=0.005,
        stale_idle_checker=lambda gpu_id: False,
    ):
        raise AssertionError("a busy GPU keeps the cross-namespace lease blocking")
    assert stale_path.exists()


def write_env(path, *names):
    return (
        sys.executable,
        "-c",
        (
            "import json, os, pathlib; "
            f"pathlib.Path({str(path)!r}).write_text(json.dumps("
            f"{{n: os.environ.get(n) for n in {list(names)!r}}}))"
        ),
    )


def test_group_lease_gets_its_own_rendezvous_port(tmp_path):
    out = tmp_path / "env.json"
    result = run_wrapper(
        tmp_path, "group:2,3", *write_env(out, "MASTER_PORT", "PET_MASTER_PORT")
    )

    assert result.returncode == 0
    env = json.loads(out.read_text())
    assert env["MASTER_PORT"] == env["PET_MASTER_PORT"]
    assert 29632 <= int(env["MASTER_PORT"]) < 29648


def test_caller_master_port_is_kept(tmp_path):
    out = tmp_path / "env.json"
    env = wrapper_env(tmp_path, MASTER_PORT="31234")
    result = run_wrapper(tmp_path, "group:0", *write_env(out, "MASTER_PORT"), env=env)

    assert result.returncode == 0
    assert json.loads(out.read_text())["MASTER_PORT"] == "31234"


def test_concurrent_pool_leases_get_distinct_ports(tmp_path):
    outs = [tmp_path / f"env{i}.json" for i in range(2)]
    procs = [
        subprocess.Popen(
            [
                "bash",
                str(GPU_LOCK_WRAPPER),
                "pool:2:0,1,2,3",
                "bash",
                "-c",
                shlex.join(write_env(out, "MASTER_PORT")) + "; sleep 0.5",
            ],
            cwd=tmp_path,
            env=wrapper_env(tmp_path),
        )
        for out in outs
    ]
    for proc in procs:
        assert proc.wait(timeout=10) == 0
    ports = {json.loads(out.read_text())["MASTER_PORT"] for out in outs}
    assert len(ports) == 2


def test_sweep_matches_its_eval_dir_as_a_whole_path(tmp_path):
    # A sibling run whose eval dir only starts with the same name is not this run's, and a narrow
    # COLUMNS must not cut the eval dir off the command lines it reads.
    ours, sibling = tmp_path / "team_a", tmp_path / "team_ab"
    jobs = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", "gpu_lock.sh", f"{d}/round_1"],
            start_new_session=True,
        )
        for d in (ours, sibling)
    ]
    try:
        deadline = time.monotonic() + 10
        while any(b"gpu_lock.sh" not in Path(f"/proc/{job.pid}/cmdline").read_bytes() for job in jobs):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        results = [
            subprocess.run(
                ["bash", str(SCRIPTS / "sweep_orphan_leases.sh"), "--eval-dir", eval_dir],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "COLUMNS": "80"},
            )
            for eval_dir in (str(ours), f"{ours}/")
        ]
    finally:
        for job in jobs:
            job.kill()
            job.wait()
    for result in results:
        assert result.returncode == 3, result.stdout + result.stderr
        assert f"pid={jobs[0].pid} " in result.stdout
        assert f"pid={jobs[1].pid} " not in result.stdout


# ---- Queued group requests from several PID namespaces (amendment A09) ---------------------
# Containers sharing one lock dir cannot probe each other's pids. A waiting manager therefore
# holds a flock on request_<id>.lock; a request without that lock file comes from a
# gpu_lease.py that predates it ("old version") and must keep queueing as before.

OTHER_NAMESPACE = "pid:[other-container]"


def own_pid_namespace():
    return os.readlink("/proc/self/ns/pid")


def write_request(
    lock_dir, request_id, pool_ids, *, manager_pid, namespace=None, started_at_ns=None
):
    """A request_<id>.json with exactly the fields every gpu_lease.py version writes."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"request_{request_id}.json"
    path.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "manager_pid": manager_pid,
                "pid_namespace": namespace or own_pid_namespace(),
                "pool_ids": list(pool_ids),
                "count": len(pool_ids),
                "started_at_ns": time.time_ns() if started_at_ns is None else started_at_ns,
            }
        )
    )
    return path


def other_namespace_request(lock_dir, request_id, pool_ids, **kwargs):
    """A request from another container. Its manager_pid, 1, is this container's init here."""
    return write_request(
        lock_dir, request_id, pool_ids, manager_pid=1, namespace=OTHER_NAMESPACE, **kwargs
    )


def hold_request_lock(lock_dir, request_id):
    """The flock a live manager holds while its request waits."""
    fd = os.open(lock_dir / f"request_{request_id}.lock", os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def leftover_request_lock(lock_dir, request_id):
    """The lock file of a manager that died: still there, but nobody holds it."""
    path = lock_dir / f"request_{request_id}.lock"
    os.close(os.open(path, os.O_CREAT | os.O_RDWR, 0o666))
    return path


def lease_blocked(gpu_lease, lock_dir, gpu_ids, wait_timeout_s=0.05):
    request = gpu_lease.GpuRequest.from_fixed_ids(gpu_ids, count=len(gpu_ids.split(",")))
    try:
        with gpu_lease.GpuLease(
            request, lock_dir=lock_dir, wait_timeout_s=wait_timeout_s, poll_interval_s=0.005
        ):
            return False
    except gpu_lease.LeaseTimeout:
        return True


def wait_for_requests(lock_dir, count, timeout_s=10):
    deadline = time.monotonic() + timeout_s
    while len(list(lock_dir.glob("request_*.json"))) < count:
        assert time.monotonic() < deadline, f"fewer than {count} request(s) in {lock_dir}"
        time.sleep(0.01)
    return sorted(lock_dir.glob("request_*.json"))


def hold_gpu_lock(lock_dir, gpu_id):
    lock_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_dir / f"gpu_{gpu_id}.lock", os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


@contextlib.contextmanager
def queued_group_lease(gpu_lease, lock_dir):
    """A lease for GPUs 0,1 waiting in this process while GPU 0 is busy; yields its request."""
    gpu0 = hold_gpu_lock(lock_dir, 0)
    result = {}
    waiter = threading.Thread(
        target=lambda: result.update(
            blocked=lease_blocked(gpu_lease, lock_dir, "0,1", wait_timeout_s=10.0)
        )
    )
    waiter.start()
    try:
        [request] = wait_for_requests(lock_dir, 1)
        yield request
    finally:
        os.close(gpu0)
        waiter.join(timeout=15)
    assert result == {"blocked": False}


@pytest.fixture
def live_process():
    """A live process of this PID namespace, standing in for another lease manager."""
    process = subprocess.Popen(["sleep", "60"])
    yield process
    process.kill()
    process.wait(timeout=5)


def test_request_for_other_gpus_is_never_touched(tmp_path):
    gpu_lease = load_gpu_lease()
    # Each of these would be removed if it overlapped the lease below; for GPUs 1,2 none may be.
    dead_here = write_request(tmp_path, "dead-here", [1, 2], manager_pid=999999999)
    dead_there = other_namespace_request(tmp_path, "dead-there", [1, 2])
    dead_there_lock = leftover_request_lock(tmp_path, "dead-there")
    stale_there = other_namespace_request(tmp_path, "stale-there", [1, 2], started_at_ns=1)

    assert not lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=1.0)
    assert all(path.exists() for path in (dead_here, dead_there, dead_there_lock, stale_there))


def test_requests_from_other_pid_namespace_keep_their_place_in_the_queue(tmp_path):
    gpu_lease = load_gpu_lease()
    # Oldest: an old-version manager in another container (no request lock). Its pid, 1, is
    # this container's init here -- the "namespace mismatch" main deleted on sight.
    old_version = other_namespace_request(tmp_path, "old-version", [0, 1])
    # Next: a new-version manager there, holding its request lock while it waits.
    new_version = other_namespace_request(tmp_path, "new-version", [1, 2])
    new_version_fd = hold_request_lock(tmp_path, "new-version")
    younger_fd = None
    try:
        assert lease_blocked(gpu_lease, tmp_path, "1", wait_timeout_s=0.2)
        assert old_version.exists() and new_version.exists()

        old_version.unlink()  # that manager got its GPUs; the next oldest still goes first
        assert lease_blocked(gpu_lease, tmp_path, "1", wait_timeout_s=0.2)
        assert new_version.exists()

        # A waiter there that is younger than us neither blocks us nor loses its request.
        younger = other_namespace_request(
            tmp_path, "younger", [1], started_at_ns=time.time_ns() + 10**12
        )
        younger_fd = hold_request_lock(tmp_path, "younger")
        new_version.unlink()  # that manager is done: json first, then its lock
        (tmp_path / "request_new-version.lock").unlink()
        os.close(new_version_fd)
        new_version_fd = None
        assert not lease_blocked(gpu_lease, tmp_path, "1", wait_timeout_s=1.0)
        assert younger.exists() and (tmp_path / "request_younger.lock").exists()
    finally:
        for fd in (new_version_fd, younger_fd):
            if fd is not None:
                os.close(fd)


def test_leftover_request_lock_of_dead_manager_is_cleared(tmp_path):
    gpu_lease = load_gpu_lease()
    # A new-version manager in another container was killed while waiting: no cleanup ran, but
    # the kernel dropped its flock. The request is recent, so only the free lock proves death.
    dead = other_namespace_request(tmp_path, "dead", [0, 1])
    dead_lock = leftover_request_lock(tmp_path, "dead")
    # Lock files without a json: one left by a dead manager, and one of a live manager (between
    # taking its lock and writing its json, or its json was just deleted by an old version).
    orphan_lock = leftover_request_lock(tmp_path, "orphan")
    live_fd = hold_request_lock(tmp_path, "live")
    try:
        assert not lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=1.0)
        assert not dead.exists() and not dead_lock.exists()
        assert not orphan_lock.exists()
        assert (tmp_path / "request_live.lock").exists()
    finally:
        os.close(live_fd)


def test_old_version_request_from_other_pid_namespace_is_kept_until_stale(tmp_path):
    gpu_lease = load_gpu_lease()
    one_second_ago = time.time_ns() - 1_000_000_000
    request = other_namespace_request(tmp_path, "old-version", [0], started_at_ns=one_second_ago)
    # No request lock, so it cannot be proven dead. For a manager whose wait limit is 10 s it is
    # stale only after 2 x 10 s, so it queues ahead of that manager.
    result = {}
    waiter = threading.Thread(
        target=lambda: result.update(
            blocked=lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=10.0)
        )
    )
    waiter.start()
    try:
        wait_for_requests(tmp_path, 2)
        time.sleep(0.3)
        assert waiter.is_alive() and request.exists()
        request.unlink()  # that manager is done
    finally:
        waiter.join(timeout=15)
    assert result == {"blocked": False}

    # For a manager whose wait limit is 0.2 s the same request (older than 2 x 0.2 s) is stale.
    other_namespace_request(tmp_path, "old-version", [0], started_at_ns=one_second_ago)
    assert not lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=0.2)
    assert not request.exists()


def test_same_namespace_request_is_judged_by_its_pid(tmp_path, live_process):
    gpu_lease = load_gpu_lease()
    # An old-version manager in this container: live pid, no request lock. However old, it is
    # a live contender (no staleness rule here), exactly as with main.
    old_version = write_request(
        tmp_path, "old-version", [0, 1], manager_pid=live_process.pid, started_at_ns=1
    )
    assert lease_blocked(gpu_lease, tmp_path, "1")
    assert old_version.exists()

    live_process.kill()
    live_process.wait(timeout=5)
    # Its manager is gone; so is a new-version one that left its unlocked lock file behind.
    dead_new_version = write_request(tmp_path, "dead-new-version", [1], manager_pid=999999999)
    dead_lock = leftover_request_lock(tmp_path, "dead-new-version")
    assert not lease_blocked(gpu_lease, tmp_path, "1", wait_timeout_s=1.0)
    assert not old_version.exists()
    assert not dead_new_version.exists() and not dead_lock.exists()


def test_unreadable_pid_namespace_is_not_proof_of_death(tmp_path, monkeypatch, live_process):
    gpu_lease = load_gpu_lease()
    real_probe = gpu_lease._pid_namespace_for_pid
    # Another user's manager in this container: reading its /proc/<pid>/ns/pid fails (EACCES).
    monkeypatch.setattr(
        gpu_lease,
        "_pid_namespace_for_pid",
        lambda pid: None if pid == live_process.pid else real_probe(pid),
    )
    pid = live_process.pid

    # Old version (no request lock): a contender, like one from another namespace ...
    request = write_request(tmp_path, "other-user", [0], manager_pid=pid)
    assert lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=0.2)
    assert request.exists()
    # ... until it is older than twice our wait limit.
    write_request(tmp_path, "other-user", [0], manager_pid=pid, started_at_ns=1)
    assert not lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=1.0)
    assert not request.exists()

    # New version: its held lock keeps it however old it is.
    write_request(tmp_path, "other-user", [0], manager_pid=pid, started_at_ns=1)
    fd = hold_request_lock(tmp_path, "other-user")
    try:
        assert lease_blocked(gpu_lease, tmp_path, "0")
        assert request.exists()
    finally:
        os.close(fd)
    (tmp_path / "request_other-user.lock").unlink()

    # A missing /proc/<pid> or a zombie is proof of death, readable or not.
    live_process.kill()
    deadline = time.monotonic() + 5
    while process_is_live(pid):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert Path(f"/proc/{pid}").exists()  # a zombie: not reaped yet
    assert not lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=1.0)
    assert not request.exists()
    live_process.wait(timeout=5)
    write_request(tmp_path, "other-user", [0], manager_pid=pid)
    assert not lease_blocked(gpu_lease, tmp_path, "0", wait_timeout_s=1.0)
    assert not request.exists()


def test_waiting_request_keeps_the_format_older_versions_read(tmp_path):
    gpu_lease = load_gpu_lease()
    with queued_group_lease(gpu_lease, tmp_path) as request:
        data = json.loads(request.read_text())
        assert set(data) == {
            "request_id", "manager_pid", "pid_namespace", "pool_ids", "count", "started_at_ns"
        }
        assert request.name == f"request_{data['request_id']}.json"
        assert (data["manager_pid"], data["pid_namespace"]) == (os.getpid(), own_pid_namespace())
        assert (data["pool_ids"], data["count"]) == ([0, 1], 2)
        # main's liveness rule keeps it: same namespace as its pid, which is alive ...
        assert gpu_lease._pid_namespace_for_pid(data["manager_pid"]) == data["pid_namespace"]
        assert gpu_lease._pid_is_live(data["manager_pid"])
        # ... and main never globs the lock file, which is held for as long as the lease waits.
        fd = os.open(request.with_suffix(".lock"), os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    assert not list(tmp_path.glob("request_*"))


def test_deleted_request_is_restored_with_its_original_priority(tmp_path):
    gpu_lease = load_gpu_lease()
    with queued_group_lease(gpu_lease, tmp_path) as request:
        original = json.loads(request.read_text())
        request.unlink()  # what main's gpu_lease.py in another PID namespace does
        deadline = time.monotonic() + 5
        while not request.exists():
            assert time.monotonic() < deadline, "the deleted request was not restored"
            time.sleep(0.005)
        assert json.loads(request.read_text()) == original
        # Still the oldest request: a younger one for GPU 1 (free) has to wait behind it.
        assert lease_blocked(gpu_lease, tmp_path, "1", wait_timeout_s=0.2)


# Real PID namespaces: a waiter started through unshare(1) records a namespace of its own and
# a pid that means nothing here, like a manager in another container sharing the lock dir. It
# still sees this /proc (mounting its own needs privileges), so the lease must not rely on
# /proc for its own request or for another namespace's request.
NAMESPACE_WAITER = """
import importlib.util, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("gpu_lease", sys.argv[1])
gpu_lease = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_lease)
lock_dir, gpu_ids, name, events = Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
request = gpu_lease.GpuRequest.from_fixed_ids(gpu_ids, count=len(gpu_ids.split(",")))
with gpu_lease.GpuLease(request, lock_dir=lock_dir, wait_timeout_s=30, poll_interval_s=0.01):
    with events.open("a") as log:
        print(name, "acquired", file=log)
    time.sleep(0.3)
    with events.open("a") as log:
        print(name, "released", file=log)
"""


@pytest.fixture(scope="module")
def other_pid_namespace():
    """Command prefix that runs its command in a new PID namespace, if this host allows it."""
    probe = [sys.executable, "-c", "import os; print(os.readlink('/proc/self/ns/pid'))"]
    for prefix in (
        ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--kill-child"],
        ["unshare", "--pid", "--fork", "--kill-child"],
    ):
        try:
            result = subprocess.run(
                prefix + probe, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout.strip() not in ("", own_pid_namespace()):
            return prefix
    pytest.skip("cannot create a PID namespace here (unshare)")


def start_namespace_waiter(prefix, lock_dir, gpu_ids, name, events):
    arguments = [str(MODULE_PATH), str(lock_dir), gpu_ids, name, str(events)]
    return subprocess.Popen([*prefix, sys.executable, "-c", NAMESPACE_WAITER, *arguments])


def test_group_waiters_in_two_pid_namespaces_are_served_in_order(tmp_path, other_pid_namespace):
    lock_dir = tmp_path / "locks"
    events = tmp_path / "events"
    gpu0 = hold_gpu_lock(lock_dir, 0)  # GPU 0 busy: both waiters queue
    waiters = []
    try:
        # Older: a group waiter in another PID namespace for GPUs 0,1.
        waiters.append(
            start_namespace_waiter(other_pid_namespace, lock_dir, "0,1", "other-namespace", events)
        )
        [foreign] = wait_for_requests(lock_dir, 1)
        assert json.loads(foreign.read_text())["pid_namespace"] != own_pid_namespace()
        # Younger: a waiter here for GPU 1. GPU 1 is free, but the older request wants it too.
        waiters.append(start_namespace_waiter([], lock_dir, "1", "this-namespace", events))
        requests = wait_for_requests(lock_dir, 2)
        time.sleep(0.5)  # dozens of polls by each
        # Each sees the other's request alive: none is deleted and nobody jumps the queue.
        assert all(path.exists() for path in requests)
        assert not events.exists()
        os.close(gpu0)
        gpu0 = None
        for waiter in waiters:
            assert waiter.wait(timeout=30) == 0
    finally:
        if gpu0 is not None:
            os.close(gpu0)
        for waiter in waiters:
            if waiter.poll() is None:
                waiter.kill()
                waiter.wait(timeout=5)
    assert events.read_text().splitlines() == [
        "other-namespace acquired",
        "other-namespace released",
        "this-namespace acquired",
        "this-namespace released",
    ]
    assert not list(lock_dir.glob("request_*"))


def test_request_of_killed_waiter_in_other_pid_namespace_is_cleared(tmp_path, other_pid_namespace):
    gpu_lease = load_gpu_lease()
    lock_dir = tmp_path / "locks"
    gpu0 = hold_gpu_lock(lock_dir, 0)  # stays busy: the waiter can never get its group
    try:
        waiter = start_namespace_waiter(
            other_pid_namespace, lock_dir, "0,1", "killed", tmp_path / "events"
        )
        try:
            [request] = wait_for_requests(lock_dir, 1)
            assert json.loads(request.read_text())["pid_namespace"] != own_pid_namespace()
            assert lease_blocked(gpu_lease, lock_dir, "1")  # alive: its older request goes first
        finally:
            waiter.kill()  # unshare dies; --kill-child SIGKILLs the waiter, so no cleanup runs
            waiter.wait(timeout=5)
        assert request.exists() and request.with_suffix(".lock").exists()
        # The kernel released the dead waiter's flock: its request no longer holds us up.
        assert not lease_blocked(gpu_lease, lock_dir, "1", wait_timeout_s=2.0)
        assert not request.exists() and not request.with_suffix(".lock").exists()
    finally:
        os.close(gpu0)
