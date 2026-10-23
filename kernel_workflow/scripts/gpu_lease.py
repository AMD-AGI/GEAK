#!/usr/bin/env python3
"""Acquire a fixed group of GPU locks and run one command under the lease."""

# Python 3.8 is supported, so retain typing.Optional/Tuple instead of PEP 604/585.
# ruff: noqa: UP006, UP045, PYI034

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

TIMEOUT_EXIT_CODE = 124

# Command-line tokens that identify a leaked child of one of our dead leases. A process is
# reaped only if it is reparented (ppid==1), in our pid namespace, has its cwd under the
# lease's reap root, AND carries the lease marker in its environment or matches one of
# these. A live lease's workers have a live parent, so they are never eligible; anything
# that cannot be attributed is left alone.
# GEAK_GPU_REAP_SIGNATURES (comma list) adds workload-specific tokens.
_ORPHAN_CMD_SIGNATURES = (
    "torchrun",
    "gpu_lease",
)


def _orphan_signatures() -> Tuple[str, ...]:
    extra = os.environ.get("GEAK_GPU_REAP_SIGNATURES", "")
    return _ORPHAN_CMD_SIGNATURES + tuple(
        token.strip() for token in extra.split(",") if token.strip()
    )


class GpuRequest:
    def __init__(
        self,
        pool_ids: Tuple[int, ...],
        *,
        count: int,
        fixed_ids: Optional[Tuple[int, ...]] = None,
    ):
        self.pool_ids = pool_ids
        self.count = count
        self.fixed_ids = fixed_ids

    @property
    def visible_ids(self) -> Tuple[int, ...]:
        return self.fixed_ids if self.fixed_ids is not None else self.pool_ids

    @property
    def lock_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.visible_ids))

    @classmethod
    def from_fixed_ids(cls, value: str, *, count: int) -> GpuRequest:
        ids = _parse_gpu_ids(value)
        if count != len(ids):
            raise ValueError(
                f"count={count} does not match fixed GPU group size {len(ids)}"
            )
        fixed_ids = tuple(ids)
        return cls(fixed_ids, count=count, fixed_ids=fixed_ids)

    @classmethod
    def from_pool(cls, value: str, *, count: int) -> GpuRequest:
        ids = tuple(_parse_gpu_ids(value))
        if count < 1 or count > len(ids):
            raise ValueError(
                f"count={count} must be between 1 and pool size {len(ids)}"
            )
        return cls(ids, count=count)

    def candidate_groups(self) -> Sequence[Tuple[int, ...]]:
        if self.fixed_ids is not None:
            return (self.fixed_ids,)
        return tuple(
            tuple(self.pool_ids[index] for index in indexes)
            for indexes in combinations(range(len(self.pool_ids)), self.count)
        )


def _parse_gpu_ids(value: str) -> Sequence[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("at least one GPU is required")
    ids = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(f"invalid GPU id: {part!r}")
        ids.append(int(part))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate GPU ids are not allowed")
    return ids


class LeaseTimeout(TimeoutError):
    """Raised when a complete GPU group cannot be acquired before the deadline."""


class CommandStartError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


class SysfsIdleChecker:
    def __init__(
        self,
        *,
        drm_root: Path = Path("/sys/class/drm"),
        max_busy_pct: int = 5,
        max_vram_mb: int = 1024,
        device_map: Optional[Mapping[int, Path]] = None,
        fail_open: bool = True,
    ):
        self.drm_root = Path(drm_root)
        self.max_busy_pct = int(max_busy_pct)
        self.max_vram_bytes = (
            None if int(max_vram_mb) < 0 else int(max_vram_mb) * 1024 * 1024
        )
        self.device_map = dict(device_map or {})
        self.fail_open = bool(fail_open)

    def __call__(self, gpu_id: int) -> bool:
        device = self._device_path(gpu_id)
        if device is None:
            return self.fail_open
        busy = _read_int(device / "gpu_busy_percent")
        if busy is None:
            return self.fail_open
        vram = _read_int(device / "mem_info_vram_used")
        if self.max_vram_bytes is not None and vram is None:
            return self.fail_open
        vram_ok = self.max_vram_bytes is None or vram <= self.max_vram_bytes
        return busy <= self.max_busy_pct and vram_ok

    def _device_path(self, gpu_id: int) -> Optional[Path]:
        mapped = self.device_map.get(gpu_id)
        if mapped is not None:
            return Path(mapped)
        card_device = self.drm_root / f"card{gpu_id}" / "device"
        if card_device.exists():
            return card_device
        render_devices = sorted(
            self.drm_root.glob("renderD*/device"),
            key=lambda path: int(path.parent.name[len("renderD") :]),
        )
        if 0 <= gpu_id < len(render_devices):
            return render_devices[gpu_id]
        return None


class GpuLease:
    def __init__(
        self,
        request: GpuRequest,
        *,
        lock_dir: Path,
        wait_timeout_s: float,
        poll_interval_s: float = 0.2,
        idle_checker: Optional[Callable[[int], bool]] = None,
        reap_root: Optional[Path] = None,
        reap_interval_s: float = 10.0,
        stale_idle_checker: Optional[Callable[[int], bool]] = None,
    ):
        self.request = request
        self.lock_dir = Path(lock_dir)
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.poll_interval_s = max(0.001, float(poll_interval_s))
        self.idle_checker = idle_checker
        # Fail-closed by default. When set, a lease left by another PID namespace (e.g. a
        # restarted container) is reclaimed only if every overlapping GPU is verifiably idle.
        self.stale_idle_checker = stale_idle_checker
        # Directory our own leaked workers run under. When set, a busy pool first reaps
        # those leaks instead of waiting for them to time out. None disables reaping.
        self.reap_root = Path(reap_root) if reap_root is not None else None
        self.reap_interval_s = max(0.0, float(reap_interval_s))
        self._last_reap_monotonic = None
        self._gpu_fds = []
        self._selected_ids = None
        self.lease_id = f"{os.getpid()}-{time.time_ns()}"
        self.metadata_path = self.lock_dir / f"lease_{self.lease_id}.json"
        self.request_path = self.lock_dir / f"request_{self.lease_id}.json"
        self._metadata = {
            "lease_id": self.lease_id,
            "manager_pid": os.getpid(),
            "pid_namespace": _pid_namespace(),
            "started_at_ns": time.time_ns(),
        }

    @property
    def selected_ids(self) -> Tuple[int, ...]:
        if self._selected_ids is None:
            raise RuntimeError("GPU lease has not been acquired")
        return self._selected_ids

    @property
    def lock_fds(self) -> Tuple[int, ...]:
        if not self._gpu_fds:
            raise RuntimeError("GPU lease has not been acquired")
        return tuple(self._gpu_fds)

    def __enter__(self) -> GpuLease:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def acquire(self) -> None:
        if self._gpu_fds:
            raise RuntimeError("GPU lease is already acquired")

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.wait_timeout_s
        attempted = False
        rejected_groups = set()
        self._write_request()
        # Clear OUR OWN leaked orphans up front so a self-inflicted busy pool does not
        # turn into a dead-wait to timeout. Foreign/unattributable procs are untouched.
        self._maybe_reap()
        try:
            while True:
                if attempted and time.monotonic() >= deadline:
                    self._raise_timeout()
                attempted = True
                acquired = self._try_acquire_group(excluded=rejected_groups)
                if acquired:
                    if (
                        self._has_live_stale_overlap()
                        or self.idle_checker is not None
                        and not all(
                            self.idle_checker(gpu_id) for gpu_id in self.selected_ids
                        )
                    ):
                        rejected_groups.add(self.selected_ids)
                        self.release()
                        if len(rejected_groups) < len(self.request.candidate_groups()):
                            continue
                    else:
                        self._write_metadata()
                        return
                rejected_groups.clear()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._raise_timeout()
                # Pool still busy: retry our own-orphan reap (throttled) before waiting,
                # in case a leak appeared or survived the first pass.
                self._maybe_reap()
                time.sleep(min(self.poll_interval_s, remaining))
        except BaseException:
            self.release()
            raise
        finally:
            self.request_path.unlink(missing_ok=True)

    def _maybe_reap(self) -> None:
        if self.reap_root is None:
            return
        now = time.monotonic()
        if (
            self._last_reap_monotonic is not None
            and (now - self._last_reap_monotonic) < self.reap_interval_s
        ):
            return
        self._last_reap_monotonic = now
        try:
            reaped = reap_own_orphans(self.reap_root)
        except OSError:
            return
        if reaped:
            sys.stderr.write(
                f"[gpu_lease] reaped {len(reaped)} own orphan(s) under "
                f"{self.reap_root}: {','.join(str(pid) for pid in reaped)}\n"
            )
            sys.stderr.flush()

    def _raise_timeout(self) -> None:
        ids = ",".join(str(gpu_id) for gpu_id in self.request.pool_ids)
        raise LeaseTimeout(
            f"failed to acquire {self.request.count} GPU(s) from [{ids}] after "
            f"{self.wait_timeout_s:g}s"
        )

    def release(self) -> None:
        try:
            self.metadata_path.unlink()
        except FileNotFoundError:
            pass
        for fd in reversed(self._gpu_fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self._gpu_fds.clear()
        self._selected_ids = None

    def update_metadata(self, **values) -> None:
        self._metadata.update(values)
        self._write_metadata()

    def _write_metadata(self) -> None:
        self._metadata.update(
            {
                "gpu_ids": list(self.selected_ids),
                "lock_ids": sorted(self.selected_ids),
            }
        )
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._metadata, sort_keys=True))
        os.replace(temporary, self.metadata_path)

    def _write_request(self) -> None:
        request = {
            "request_id": self.lease_id,
            "manager_pid": os.getpid(),
            "pid_namespace": _pid_namespace(),
            "pool_ids": list(self.request.pool_ids),
            "count": self.request.count,
            "started_at_ns": time.time_ns(),
        }
        temporary = self.request_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(request, sort_keys=True))
        os.replace(temporary, self.request_path)

    def _has_live_stale_overlap(self) -> bool:
        requested = set(self.selected_ids)
        for path in self.lock_dir.glob("lease_*.json"):
            if path == self.metadata_path:
                continue
            try:
                metadata = json.loads(path.read_text())
                stale_ids = {int(gpu_id) for gpu_id in metadata.get("gpu_ids", [])}
                child_pgid = int(metadata.get("child_pgid", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return True
            if not requested.intersection(stale_ids):
                continue
            stale_namespace = metadata.get("pid_namespace")
            if stale_namespace and stale_namespace != _pid_namespace():
                # Its pids cannot be probed from here. We already hold the GPU flocks, so its
                # manager is gone; orphaned children could still run, hence the idle proof.
                overlap = requested.intersection(stale_ids)
                if self.stale_idle_checker is None or not all(
                    self.stale_idle_checker(gpu_id) for gpu_id in overlap
                ):
                    return True
            if child_pgid > 0 and _process_group_exists(child_pgid):
                return True
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return False

    def _try_acquire_group(self, *, excluded: Optional[set] = None) -> bool:
        excluded = excluded or set()
        allocator_fd = os.open(
            self.lock_dir / "allocator.lock",
            os.O_CREAT | os.O_RDWR,
            0o666,
        )
        allocator_locked = False
        try:
            try:
                fcntl.flock(allocator_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                allocator_locked = True
            except BlockingIOError:
                return False
            if not self._is_request_turn():
                return False
            for candidate in self.request.candidate_groups():
                if candidate in excluded:
                    continue
                self._selected_ids = candidate
                for gpu_id in sorted(candidate):
                    fd = os.open(
                        self.lock_dir / f"gpu_{gpu_id}.lock",
                        os.O_CREAT | os.O_RDWR,
                        0o666,
                    )
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(fd)
                        self.release()
                        break
                    self._gpu_fds.append(fd)
                else:
                    return True
            return False
        finally:
            if allocator_locked:
                fcntl.flock(allocator_fd, fcntl.LOCK_UN)
            os.close(allocator_fd)

    def _is_request_turn(self) -> bool:
        requested_pool = set(self.request.pool_ids)
        contenders = []
        for path in self.lock_dir.glob("request_*.json"):
            try:
                data = json.loads(path.read_text())
                manager_pid = int(data["manager_pid"])
                namespace = str(data["pid_namespace"])
                pool_ids = {int(gpu_id) for gpu_id in data["pool_ids"]}
                request_id = str(data["request_id"])
                priority = (int(data["started_at_ns"]), request_id)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
                continue
            if _pid_namespace_for_pid(manager_pid) != namespace or not _pid_is_live(
                manager_pid
            ):
                path.unlink(missing_ok=True)
                continue
            if requested_pool & pool_ids:
                contenders.append((priority, request_id))
        if not contenders:
            return False
        _, winner = min(contenders)
        return winner == self.lease_id


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_namespace() -> str:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return "unknown"


def _pid_namespace_for_pid(pid: int) -> Optional[str]:
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except OSError:
        return None


def _pid_is_live(pid: int) -> bool:
    try:
        suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1]
        return suffix.strip().split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _proc_ppid(pid: int) -> Optional[int]:
    try:
        suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1]
        # After the ")" the fields are: state ppid pgrp ... -> ppid is index 1.
        return int(suffix.strip().split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _proc_cwd(pid: int) -> Optional[str]:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def _proc_has_lease_marker(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    return b"GEAK_GPU_LEASE_ACTIVE=1" in raw.split(b"\x00")


def reap_own_orphans(
    reap_root: Path,
    *,
    signatures: Optional[Sequence[str]] = None,
    killer: Callable[[int, int], None] = os.kill,
) -> Sequence[int]:
    """Kill our own reparented leaks under ``reap_root``; return the pids reaped.

    A pid is eligible only if it is visible in this /proc, is an orphan (ppid==1), lives
    in our pid namespace, has its cwd under ``reap_root``, and its cmdline matches a
    leaked-worker signature or carries the lease marker. A live lease's workers have a live parent, so they are never
    touched.
    """
    if signatures is None:
        signatures = _orphan_signatures()
    root = str(Path(reap_root).resolve())
    my_namespace = _pid_namespace()
    reaped = []
    try:
        pids = [int(entry) for entry in os.listdir("/proc") if entry.isdigit()]
    except OSError:
        return reaped
    for pid in pids:
        if pid == os.getpid():
            continue
        if _proc_ppid(pid) != 1:
            continue
        cwd = _proc_cwd(pid)
        if cwd is None:
            continue
        resolved = str(Path(cwd))
        if resolved != root and not resolved.startswith(root + os.sep):
            continue
        cmdline = _proc_cmdline(pid)
        if not _proc_has_lease_marker(pid) and not any(
            token in cmdline for token in signatures
        ):
            continue
        # Only kill what we could actually kill: same pid namespace as us.
        if _pid_namespace_for_pid(pid) != my_namespace:
            continue
        try:
            killer(pid, signal.SIGKILL)
            reaped.append(pid)
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return reaped


def parse_amd_smi_device_map(
    payload: str, *, pci_root: Path = Path("/sys/bus/pci/devices")
) -> Mapping[int, Path]:
    records = json.loads(payload)
    mapping = {}
    for record in records:
        mapping[int(record["gpu"])] = Path(pci_root) / str(record["bdf"])
    return mapping


def discover_amd_smi_device_map() -> Mapping[int, Path]:
    try:
        result = subprocess.run(
            ["amd-smi", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
            # amd-smi takes about 3.1 s on an 8 x MI355X node; a 3 s limit returned an empty map
            # and every lock with the idle check on failed with "cannot resolve sysfs devices".
            timeout=30,
        )
        return parse_amd_smi_device_map(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return {}


def run_command(
    request: GpuRequest,
    command: Sequence[str],
    *,
    lock_dir: Path,
    wait_timeout_s: float,
    run_timeout_s: float,
    term_grace_s: float,
    env: Optional[Mapping[str, str]] = None,
    idle_checker: Optional[Callable[[int], bool]] = None,
    reap_root: Optional[Path] = None,
    stale_idle_checker: Optional[Callable[[int], bool]] = None,
    use_log: Optional[Path] = None,
) -> int:
    if not command:
        raise ValueError("command must not be empty")

    child_env = os.environ.copy()
    if env is not None:
        child_env.update(env)
    process = None
    previous_handlers = {}

    def forward_signal(signum, _frame):
        if process is not None:
            _terminate_process_group(process, term_grace_s=term_grace_s)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)

    wait_started = time.monotonic()
    try:
        with GpuLease(
            request,
            lock_dir=lock_dir,
            wait_timeout_s=wait_timeout_s,
            idle_checker=idle_checker,
            reap_root=reap_root,
            stale_idle_checker=stale_idle_checker,
        ) as lease:
            visible = ",".join(str(gpu_id) for gpu_id in lease.selected_ids)
            if use_log is not None:
                _append_use_log(
                    use_log,
                    request,
                    lease.selected_ids,
                    wait_s=time.monotonic() - wait_started,
                )
            child_env.update(
                {
                    "HIP_VISIBLE_DEVICES": visible,
                    "CUDA_VISIBLE_DEVICES": visible,
                    "GEAK_GPU_GROUP": visible,
                    "GEAK_GPU_LEASE_ACTIVE": "1",
                    "GEAK_GPU_LEASE_IDS": visible,
                }
            )
            if "MASTER_PORT" not in child_env:
                # Concurrent group leases would all rendezvous on torch's default 29500.
                port = str(_rendezvous_port(lease.selected_ids))
                child_env.update({"MASTER_PORT": port, "PET_MASTER_PORT": port})
            try:
                managed_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                previous_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, managed_signals
                )

                def restore_child_signal_mask():
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

                try:
                    try:
                        process = subprocess.Popen(
                            list(command),
                            start_new_session=True,
                            env=child_env,
                            close_fds=True,
                            pass_fds=lease.lock_fds,
                            # gpu_lease.py is a dedicated, single-threaded CLI process.
                            # The child must not inherit the short signal mask used to
                            # close the Popen-before-assignment race in the parent.
                            preexec_fn=restore_child_signal_mask,  # noqa: PLW1509
                        )
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                except FileNotFoundError as error:
                    raise CommandStartError(
                        f"command not found: {command[0]}", exit_code=127
                    ) from error
                except PermissionError as error:
                    raise CommandStartError(
                        f"command is not executable: {command[0]}", exit_code=126
                    ) from error
                lease.update_metadata(child_pgid=process.pid)
                if run_timeout_s is not None and float(run_timeout_s) < 0:
                    return_code = process.wait()
                else:
                    return_code = process.wait(timeout=max(0.0, float(run_timeout_s)))
                _terminate_and_reap(
                    process, term_grace_s=term_grace_s, reap_root=reap_root
                )
                return _shell_exit_code(return_code)
            except subprocess.TimeoutExpired:
                # Deadlocked / runaway bench that blew --run-timeout: killpg the child
                # group, then reap the reparented ranks so a hang cannot pin the pool.
                _terminate_and_reap(
                    process, term_grace_s=term_grace_s, reap_root=reap_root
                )
                return TIMEOUT_EXIT_CODE
            except BaseException:
                if process is not None:
                    _terminate_and_reap(
                        process, term_grace_s=term_grace_s, reap_root=reap_root
                    )
                raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _rendezvous_port(gpu_ids: Sequence[int]) -> int:
    """A free TCP port derived from the lease: concurrent leases hold disjoint GPUs."""
    base = 29600 + 16 * min(gpu_ids)
    for port in range(base, base + 16):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _append_use_log(
    path: Path, request: GpuRequest, gpu_ids: Sequence[int], *, wait_s: float
) -> None:
    # Same one-line-per-acquisition log as gpu_lock.sh's single-GPU paths; best effort.
    record = {
        "t": int(time.time()),
        "gpus": list(gpu_ids),
        "pool": ",".join(str(gpu_id) for gpu_id in request.pool_ids),
        "pid": os.getpid(),
        "mode": "group",
        "wait_s": int(wait_s),
    }
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _terminate_process_group(process: subprocess.Popen, *, term_grace_s: float) -> None:
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    deadline = time.monotonic() + max(0.0, float(term_grace_s))
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(pgid):
            break
        time.sleep(0.01)

    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _terminate_and_reap(
    process: subprocess.Popen,
    *,
    term_grace_s: float,
    reap_root: Optional[Path] = None,
) -> None:
    """Kill the child's process group, then sweep its reparented workers.

    Launchers such as torchrun may start ranks in sessions of their own, so
    ``killpg(child_pgid)`` can leave GPU-holding ranks alive; they reparent to init
    and keep the cards busy. The sweep uses the rules of reap_own_orphans(), so
    unattributable processes are never touched. Reparenting is not instantaneous,
    so poll for a few seconds until two consecutive sweeps come back empty.
    """
    _terminate_process_group(process, term_grace_s=term_grace_s)
    if reap_root is None:
        return
    deadline = time.monotonic() + 10.0
    empty_sweeps = 0
    while time.monotonic() < deadline and empty_sweeps < 2:
        try:
            reaped = reap_own_orphans(reap_root)
        except OSError:
            reaped = []
        if reaped:
            sys.stderr.write(
                f"[gpu_lease] post-terminate reaped {len(reaped)} reparented "
                f"orphan(s) under {reap_root}: "
                f"{','.join(str(pid) for pid in reaped)}\n"
            )
            sys.stderr.flush()
            empty_sweeps = 0
        else:
            empty_sweeps += 1
        time.sleep(0.5)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            fields = (process_dir / "stat").read_text().rsplit(")", 1)[1]
            parts = fields.strip().split()
            state = parts[0]
            process_group = int(parts[2])
        except (OSError, IndexError, ValueError):
            continue
        if process_group == pgid and state != "Z":
            return True
    return False


def _shell_exit_code(return_code: int) -> int:
    return 128 + (-return_code) if return_code < 0 else return_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="run one command under a GPU group lease")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixed-ids")
    source.add_argument("--pool")
    run.add_argument("--count", type=int)
    run.add_argument(
        "--lock-dir",
        default=os.environ.get("GEAK_GPU_LOCK_DIR", "/tmp/team_gpu_locks"),
    )
    run.add_argument(
        "--reap-root",
        default=os.environ.get("GEAK_GPU_REAP_ROOT"),
        help=(
            "directory whose reparented (ppid==1) lease workers may be reaped; "
            "unset disables reaping"
        ),
    )
    run.add_argument("--wait-timeout", type=float, default=1200.0)
    run.add_argument("--run-timeout", type=float, default=900.0)
    run.add_argument("--term-grace", type=float, default=5.0)
    run.add_argument("--require-idle", action="store_true")
    run.add_argument(
        "--sysfs-root",
        default=os.environ.get("GEAK_GPU_SYSFS_ROOT", "/sys/class/drm"),
    )
    run.add_argument(
        "--max-busy-pct",
        type=int,
        default=int(os.environ.get("GEAK_GPU_MAX_BUSY_PCT", "5")),
    )
    run.add_argument(
        "--max-vram-mb",
        type=int,
        default=int(os.environ.get("GEAK_GPU_MAX_VRAM_MB", "-1")),
    )
    run.add_argument("--use-log", default=os.environ.get("GEAK_GPU_USE_LOG"))
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if os.environ.get("GEAK_GPU_LEASE_ACTIVE") == "1":
        parser.error("nested GPU lease requests are not supported")

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        if args.fixed_ids is not None:
            raw_ids = [
                part.strip() for part in args.fixed_ids.split(",") if part.strip()
            ]
            count = args.count if args.count is not None else len(raw_ids)
            request = GpuRequest.from_fixed_ids(args.fixed_ids, count=count)
        else:
            if args.count is None:
                raise ValueError("--count is required with --pool")
            request = GpuRequest.from_pool(args.pool, count=args.count)
        idle_checker = None
        if args.require_idle:
            device_map = {}
            if Path(args.sysfs_root) == Path("/sys/class/drm"):
                device_map = discover_amd_smi_device_map()
                missing = set(request.visible_ids) - set(device_map)
                insufficient_dynamic_pool = (
                    request.fixed_ids is None
                    and len(set(request.pool_ids) & set(device_map)) < request.count
                )
                if (
                    request.fixed_ids is not None
                    and missing
                    or insufficient_dynamic_pool
                ):
                    missing_text = ",".join(str(gpu_id) for gpu_id in sorted(missing))
                    raise ValueError(
                        "cannot resolve sysfs devices for required GPU idle check: "
                        f"{missing_text}; set GEAK_GPU_REQUIRE_IDLE=0 to bypass"
                    )
            idle_checker = SysfsIdleChecker(
                drm_root=Path(args.sysfs_root),
                max_busy_pct=args.max_busy_pct,
                max_vram_mb=args.max_vram_mb,
                device_map=device_map,
                fail_open=False,
            )
        stale_checker = []

        def stale_idle_checker(gpu_id):
            # Built on first use: only a lease left by another PID namespace needs it, and the
            # amd-smi device map costs seconds per lock on a large node.
            if not stale_checker:
                stale_checker.append(
                    SysfsIdleChecker(
                        drm_root=Path(args.sysfs_root),
                        max_busy_pct=args.max_busy_pct,
                        # orphaned children hold VRAM, so the stale proof always checks it
                        max_vram_mb=args.max_vram_mb if args.max_vram_mb >= 0 else 1024,
                        device_map=(
                            idle_checker.device_map
                            if idle_checker is not None
                            else discover_amd_smi_device_map()
                            if Path(args.sysfs_root) == Path("/sys/class/drm")
                            else {}
                        ),
                        fail_open=False,
                    )
                )
            return stale_checker[0](gpu_id)

        return run_command(
            request,
            command,
            lock_dir=Path(args.lock_dir),
            wait_timeout_s=args.wait_timeout,
            run_timeout_s=args.run_timeout,
            term_grace_s=args.term_grace,
            idle_checker=idle_checker,
            reap_root=Path(args.reap_root) if args.reap_root else None,
            stale_idle_checker=stale_idle_checker,
            use_log=Path(args.use_log) if args.use_log else None,
        )
    except CommandStartError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return error.exit_code
    except (ValueError, LeaseTimeout) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
