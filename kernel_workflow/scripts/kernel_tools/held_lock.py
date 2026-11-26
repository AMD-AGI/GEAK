#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect a GEAK GPU lock already held by a parent process (no GPU work; never keeps a lock).

gpu_lock.sh exports GEAK_GPU_LOCK_HELD only with GEAK_KERNEL_TOOLS=1 set BEFORE it (its default
environment is unchanged). A kernel tool started as `gpu_lock.sh <ids> bash <tool> ...` without that
variable still runs inside the lock: gpu_lock.sh exported HIP_VISIBLE_DEVICES=<id>, and an ancestor
process (gpu_lock.sh, or the subshell that took the flock) keeps <lock dir>/gpu_<id>.lock open while
the command runs. This helper recognises exactly that case and prints the held GPU id:

  * HIP_VISIBLE_DEVICES is one GPU id;
  * an ANCESTOR of this process has <lock dir>/gpu_<id>.lock open (lock dir = GEAK_GPU_LOCK_DIR when
    set, then KERNEL_EXTRACT_LOCK_DIR, then /tmp/team_gpu_locks: the directories gpu_lock.sh uses);
  * that file is flock'ed exclusively (a non-blocking shared flock on it fails).

Anything else (no ancestor holds the file, a sibling or an unrelated process holds it, several
visible GPUs) is "not held": the caller then takes the lock itself or refuses, as before.

Exit 0 + the id on stdout: held by an ancestor. Exit 1: not detected.
"""
from __future__ import annotations

import fcntl
import os
import sys

DEFAULT_LOCK_DIR = "/tmp/team_gpu_locks"


def _lock_dirs(environ):
    dirs = []
    for key in ("GEAK_GPU_LOCK_DIR", "KERNEL_EXTRACT_LOCK_DIR"):
        if environ.get(key):
            dirs.append(environ[key])
    dirs.append(DEFAULT_LOCK_DIR)
    out = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def _ancestors(pid, proc="/proc"):
    """Parent, grandparent, ... of pid (pid itself excluded), stopping at pid 1."""
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            with open(f"{proc}/{pid}/stat") as fh:
                stat = fh.read()
            pid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            return
        if pid > 0:
            yield pid


def _holds_open(pid, dev_ino, proc="/proc"):
    fd_dir = f"{proc}/{pid}/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return False
    for fd in fds:
        try:
            st = os.stat(f"{fd_dir}/{fd}")
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == dev_ino:
            return True
    return False


def _flock_held(path):
    """True when someone holds an exclusive flock on path: a non-blocking SHARED flock on a fresh
    open fails. (/proc/locks is not usable: inside a container it may not list the flock.) When the
    file is free the shared lock is taken and dropped at once."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)   # closing drops a shared lock taken above
    return False


def held_gpu(environ=None, pid=None, proc="/proc"):
    """The GPU id an ancestor holds the GEAK lock for, or None."""
    env = os.environ if environ is None else environ
    visible = (env.get("HIP_VISIBLE_DEVICES") or "").strip()
    if not visible.isdigit():
        return None
    pid = os.getpid() if pid is None else pid
    for lock_dir in _lock_dirs(env):
        path = os.path.join(lock_dir, f"gpu_{int(visible)}.lock")
        try:
            st = os.stat(path)
        except OSError:
            continue
        dev_ino = (st.st_dev, st.st_ino)
        if not _flock_held(path):
            continue
        if any(_holds_open(a, dev_ino, proc) for a in _ancestors(pid, proc)):
            return str(int(visible))
    return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    gpu = held_gpu()
    if gpu is None:
        return 1
    print(gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
