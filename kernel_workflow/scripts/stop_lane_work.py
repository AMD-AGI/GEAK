#!/usr/bin/env python3
"""Stop the GPU work a kernel lane left behind when it gave up on one of its agents.

kernel_lane.js bounds every agent with AGENT_TIMEOUT_MS, but a workflow cannot cancel an agent: on a
timeout the lane only stops waiting. The agent keeps going -- its UT runs keep the GPU lock, its
polling loops start the next run -- and the next lane queues behind it. On a 4-GPU AgentX run one
abandoned benchmark_engineer held GPU0 for 70 min after its lane had failed, so the retried lane's
own benchmark_engineer timed out while still waiting for the GPU (GEAK #606).

When a lane ends because an agent returned nothing, it runs this script on its EVAL_DIR:

  1. with --block, write <eval_dir>/.geak_stop. gpu_lock.sh refuses to start a command whose working
     directory is inside a stopped dir, so the abandoned agent cannot take a GPU again;
  2. stop every process whose working directory is inside <eval_dir>, or whose command line names
     <eval_dir> or a path under it, with all its descendants: SIGTERM, then SIGKILL after a grace.

A process is re-identified by its /proc start time before each signal, so a recycled pid is never hit.
Never selected: PID 1, this process and its ancestors (the workflow host and the agent running this).
Signals go to one pid at a time, never a process group: a tool's shell can share its group with the
agent host, and killing that group would end the whole run.

Prints one JSON object: {"ok", "eval_dir", "blocked", "stopped": [...], "survivors": [...]}.
Exit 0 when nothing selected is still running (finding nothing is success), 1 otherwise, 2 on bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

STOP_FILE = ".geak_stop"
# An eval dir this shallow is not a lane's own directory; matching under it would select unrelated work.
_MIN_ROOT_PARTS = 3
# A path starts an argument or follows whitespace, a quote or shell punctuation, and ends at one of
# these; so /run/team_a never matches /run/team_ab or /backup/run/team_a.
_PATH_START = frozenset("\0\t\n \"'=:;,|&()<>`")
_PATH_END = frozenset("/\"' :;,=)\0\t\n")
_POLL_S = 0.25


def _stat(pid: int):
    """``(ppid, start_ticks)`` from /proc/<pid>/stat, or None once the process is gone or a zombie."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # comm (field 2) may hold spaces and parentheses; everything after the LAST ')' is fixed-format.
    fields = raw[raw.rfind(")") + 2:].split()
    try:
        if fields[0] in ("Z", "X"):
            return None
        return int(fields[1]), int(fields[19])
    except (IndexError, ValueError):
        return None


def _argv(pid: int) -> list:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def _name(argv: list) -> str:
    """The program and the script it runs, never the flags or a ``-c`` body (they can carry a credential)."""
    if not argv:
        return "?"
    words = [os.path.basename(argv[0])]
    script = next((a for a in argv[1:] if not a.startswith("-")), "")
    if script.endswith((".py", ".sh")):
        words.append(os.path.basename(script))
    return " ".join(words)[:80]


def mentions(text: str, roots: tuple) -> bool:
    for root in roots:
        at = text.find(root)
        while at != -1:
            end = at + len(root)
            if (at == 0 or text[at - 1] in _PATH_START) and (end == len(text) or text[end] in _PATH_END):
                return True
            at = text.find(root, at + 1)
    return False


def _cwd_in(pid: int, roots: tuple) -> bool:
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return any(cwd == root or cwd.startswith(root + "/") for root in roots)


def roots_of(eval_dir: str) -> tuple:
    roots = []
    for root in (os.path.normpath(os.path.abspath(eval_dir)), os.path.realpath(eval_dir)):
        if root not in roots and len(Path(root).parts) >= _MIN_ROOT_PARTS:
            roots.append(root)
    return tuple(roots)


def _ancestors(pid: int) -> set:
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        st = _stat(pid)
        if st is None:
            break
        pid = st[0]
    return seen


def select(roots: tuple) -> dict:
    """``{pid: (start_ticks, name)}`` for every live process inside ``roots`` and all its descendants."""
    procs = {}
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        st = _stat(pid)
        if st is None:
            continue
        procs[pid] = st
        children.setdefault(st[0], []).append(pid)
    spare = _ancestors(os.getpid()) | {1, os.getpid()}
    seeds = [pid for pid in procs
             if pid not in spare and (_cwd_in(pid, roots) or mentions("\0".join(_argv(pid)), roots))]
    chosen = {}
    stack = list(seeds)
    while stack:
        pid = stack.pop()
        if pid in chosen or pid in spare or pid not in procs:
            continue
        chosen[pid] = (procs[pid][1], _name(_argv(pid)))
        stack.extend(children.get(pid, ()))
    return chosen


def _same(pid: int, start: int) -> bool:
    st = _stat(pid)
    return st is not None and st[1] == start


def _signal_all(chosen: dict, sig: int) -> None:
    for pid, (start, _) in chosen.items():
        if _same(pid, start):
            try:
                os.kill(pid, sig)
            except OSError:
                pass


def _wait_gone(chosen: dict, seconds: float) -> dict:
    deadline = time.monotonic() + seconds
    while True:
        alive = {pid: v for pid, v in chosen.items() if _same(pid, v[0])}
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(_POLL_S)


def stop(eval_dir: str, block: bool, reason: str, grace: float) -> dict:
    roots = roots_of(eval_dir)
    if block:
        Path(eval_dir, STOP_FILE).write_text((reason or "stopped by the kernel lane") + "\n")
    stopped = {}
    # A second round catches what a polling loop started while the first round was being stopped.
    for _ in range(2):
        chosen = select(roots)
        if not chosen:
            break
        stopped.update(chosen)
        _signal_all(chosen, signal.SIGTERM)
        alive = _wait_gone(chosen, grace)
        if alive:
            _signal_all(alive, signal.SIGKILL)
            _wait_gone(alive, 5.0)
    survivors = {pid: v for pid, v in stopped.items() if _same(pid, v[0])}
    return {
        "ok": not survivors,
        "eval_dir": eval_dir,
        "blocked": block,
        "stopped": [{"pid": pid, "name": name} for pid, (_, name) in sorted(stopped.items())],
        "survivors": [{"pid": pid, "name": name} for pid, (_, name) in sorted(survivors.items())],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", required=True, help="the lane's EVAL_DIR")
    ap.add_argument("--block", action="store_true",
                    help=f"write <eval_dir>/{STOP_FILE} so gpu_lock.sh refuses new commands under it")
    ap.add_argument("--reason", default="", help="recorded in the stop file")
    ap.add_argument("--grace", type=float, default=float(os.environ.get("STOP_LANE_GRACE_SEC", "10")),
                    help="seconds between SIGTERM and SIGKILL (default 10)")
    a = ap.parse_args(argv)
    eval_dir = os.path.abspath(a.eval_dir)
    if not os.path.isdir(eval_dir) or not roots_of(eval_dir):
        print(f"stop_lane_work.py: --eval-dir must be an existing lane directory, got {a.eval_dir!r}",
              file=sys.stderr)
        return 2
    # Run from / so this process never sits inside the directory it is clearing.
    os.chdir("/")
    result = stop(eval_dir, a.block, a.reason, a.grace)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
