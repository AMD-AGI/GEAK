#!/usr/bin/env python3
"""Stop what a GEAK run left running, choosing processes by what they belong to rather than by name.

A run hands the GPUs over twice: to its own final phase at the reserve boundary, and to its caller when it
exits. Both times something it started can still be up -- a leg an abandoned search agent launched with
``setsid nohup``, a tuner sweep, a script polling for idle GPUs so it can start the next leg -- and nothing
else will stop it: detached work is reparented to PID 1, out of every agent's and every tool's reach. The
next server booted onto those GPUs then fails its memory-balance check, or measures beside a neighbour.

A process belongs to the run when it carries the run's marker (``GEAK_E2E_RUN_MARK``, set by run_e2e.py
before the workflow starts and inherited by everything under it), or when it started after the run did and
its command line or working directory is inside the run's eval dir. Every descendant of a selected process
is selected with it. Never selected: PID 1, this process, and its ancestors -- the caller's orchestrator,
the runner, and for a fence the workflow and the agent running it.

  fence  The final phase is starting. Stop what holds a GPU (/dev/kfd) or the serving-GPU lock and runs
         under one of the run's processes -- a server renaming its workers with setproctitle can wipe the
         marker from their environment, never their ancestry -- and the run's processes running detached
         from the live workflow; then write EVAL_DIR/.final_phase_fence: the final phase owns the GPUs.
  exit   The run is over. Stop every process that belongs to it.

A process is re-identified by its /proc start time before every signal, so a recycled pid is never hit:
SIGTERM first, SIGKILL for whatever is still up after the grace period.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

MARK_ENV = "GEAK_E2E_RUN_MARK"
RUN_FILE = ".geak_run.json"
FENCE_FILE = ".final_phase_fence"
LOG_FILE = "logs/run_reaper.jsonl"
KFD = "/dev/kfd"
# bench_e2e.sh: SERVING_GPU_LOCK defaults to /tmp/geak_serving_gpu_<gpu list>.lock
SERVING_LOCK_PREFIX = "/tmp/geak_serving_gpu_"
TERM_GRACE_S = 20.0
KILL_WAIT_S = 10.0
ROUNDS = 3
_POLL_S = 0.25
# An eval dir this shallow is not a run's own directory; matching under it would select unrelated work.
_MIN_ROOT_PARTS = 3
# A path starts an argument or follows whitespace, a quote or shell punctuation; after "/" or a name character
# the root is only the tail of a longer path (/backup/run_ws/... for /run_ws/...).
_PATH_START = frozenset("\0\t\n \"'=:;,|&()<>`")
_PATH_END = frozenset("/\"' :;,=)\0")
_LAUNCHERS = frozenset({"bash", "sh", "dash", "zsh", "env", "nohup", "setsid", "timeout", "stdbuf", "nice"})


class Proc(NamedTuple):
    pid: int
    ppid: int
    start: int
    zombie: bool
    marked: bool
    in_eval_dir: bool
    # Read only for the run's processes and what runs under them: another job's server is not this run's.
    holds_gpu: bool
    holds_lock: bool
    name: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stat(pid: int) -> tuple[int, int, bool] | None:
    """``(ppid, start_ticks, zombie)`` from /proc/<pid>/stat, or None once the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # comm (field 2) may hold spaces and parentheses; everything after the LAST ')' is fixed-format.
    fields = raw[raw.rfind(")") + 2:].split()
    try:
        return int(fields[1]), int(fields[19]), fields[0] in ("Z", "X")
    except (IndexError, ValueError):
        return None


def _carries(pid: int, mark: str) -> bool:
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return False
    prefix = (MARK_ENV + "=").encode()
    for entry in env.split(b"\0"):
        if entry.startswith(prefix):
            return mark.encode() in entry[len(prefix):].split(b":")
    return False


def _argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def _word(arg: str) -> str:
    """The basename of ``arg``'s first word, or "" unless it is a plain program, module or script name."""
    head = arg.split(maxsplit=1)[0] if arg.strip() else ""
    base = os.path.basename(head)
    return base if base and all(c.isalnum() or c in "._+-" for c in base) else ""


def _name(argv: list[str]) -> str:
    """The program, and the script it runs -- never the flags, their values or a ``-c`` command: any of
    those can carry a credential."""
    if not argv:
        return "?"
    words = [_word(argv[0]) or "?"]
    if words[0] in _LAUNCHERS or words[0].startswith("python"):
        script = _word(next((a for a in argv[1:] if not a.startswith("-")), ""))
        if script:
            words.append(script)
    return " ".join(words)[:80]


def _mentions(text: str, roots: tuple[str, ...]) -> bool:
    for root in roots:
        at = text.find(root)
        while at != -1:
            end = at + len(root)
            if (at == 0 or text[at - 1] in _PATH_START) and (end == len(text) or text[end] in _PATH_END):
                return True
            at = text.find(root, at + 1)
    return False


def _cwd_in(pid: int, roots: tuple[str, ...]) -> bool:
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return any(cwd == root or cwd.startswith(root + "/") for root in roots)


def _open_files(pid: int) -> tuple[bool, bool]:
    gpu = lock = False
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False, False
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target == KFD:
            gpu = True
        elif target.startswith(SERVING_LOCK_PREFIX) and target.endswith(".lock"):
            lock = True
        if gpu and lock:
            break
    return gpu, lock


def eval_dir_roots(eval_dir: str) -> tuple[str, ...]:
    roots: list[str] = []
    for root in (os.path.normpath(eval_dir), os.path.realpath(eval_dir)):
        if root not in roots and len(Path(root).parts) >= _MIN_ROOT_PARTS:
            roots.append(root)
    return tuple(roots)


def snapshot(mark: str, roots: tuple[str, ...], since_ticks: int | None) -> dict[int, Proc]:
    procs: dict[int, Proc] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return procs
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        st = _stat(pid)
        if st is None:
            continue
        ppid, start, zombie = st
        argv = _argv(pid)
        marked = bool(mark) and _carries(pid, mark)
        in_dir = (
            bool(roots)
            and since_ticks is not None
            and start >= since_ticks
            and (_mentions("\0".join(argv), roots) or _cwd_in(pid, roots))
        )
        procs[pid] = Proc(pid, ppid, start, zombie, marked, in_dir, False, False, _name(argv))
    for pid in _with_descendants(procs, {pid for pid, p in procs.items() if p.marked or p.in_eval_dir}):
        if not procs[pid].zombie:
            gpu, lock = _open_files(pid)
            if gpu or lock:
                procs[pid] = procs[pid]._replace(holds_gpu=gpu, holds_lock=lock)
    return procs


def _with_descendants(procs: dict[int, Proc], roots: set[int], keep: frozenset[int] = frozenset()) -> list[int]:
    """``roots`` and every process under them, stopping at ``keep``."""
    children: dict[int, list[int]] = defaultdict(list)
    for p in procs.values():
        children[p.ppid].append(p.pid)
    seen = set(roots)
    stack = sorted(roots)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in seen and child not in keep:
                seen.add(child)
                stack.append(child)
    return sorted(seen)


def protected_pids() -> list[int]:
    """This process and every ancestor of it, nearest first, ending with PID 1: never signalled."""
    chain: list[int] = []
    pid = os.getpid()
    while pid > 1 and pid not in chain:
        chain.append(pid)
        st = _stat(pid)
        if st is None:
            break
        pid = st[0]
    return chain + [1]


def workflow_pids(chain: list[int], runner_pid: object) -> frozenset[int]:
    """The protected processes the live workflow's own tools run under: this process up to the runner.

    An orphan is adopted by PID 1 or by a subreaper further up -- the caller's orchestrator may be one -- and
    either way it has left the workflow.
    """
    if isinstance(runner_pid, int) and runner_pid in chain:
        return frozenset(chain[: chain.index(runner_pid) + 1])
    return frozenset(chain) - {1}


def _detached(pid: int, procs: dict[int, Proc], workflow: frozenset[int]) -> bool:
    """Whether ``pid`` runs outside the live workflow: none of the workflow's processes is above it."""
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid in workflow:
            return False
        seen.add(pid)
        p = procs.get(pid)
        if p is None:
            return True
        pid = p.ppid
    return True


def select(
    procs: dict[int, Proc], mode: str, keep: set[int], workflow: frozenset[int] | None = None
) -> list[tuple[Proc, str]]:
    """The processes ``mode`` stops, each with the rule that chose it."""
    if workflow is None:
        workflow = frozenset(keep) - {1}
    why: dict[int, str] = {}
    for pid, p in procs.items():
        if p.zombie or pid in keep:
            continue
        owned = p.marked or p.in_eval_dir
        if mode == "exit":
            if owned:
                why[pid] = "run"
        elif p.holds_gpu:
            why[pid] = "gpu"
        elif p.holds_lock:
            why[pid] = "serving-lock"
        elif owned and _detached(pid, procs, workflow):
            why[pid] = "detached"
    for pid in _with_descendants(procs, set(why), frozenset(keep)):
        if pid not in why and not procs[pid].zombie:
            why[pid] = "descendant"
    return [(procs[pid], why[pid]) for pid in sorted(why)]


def _alive(p: Proc) -> bool:
    st = _stat(p.pid)
    return st is not None and st[1] == p.start and not st[2]


def _signal(p: Proc, sig: int) -> bool:
    if not _alive(p):
        return False
    try:
        os.kill(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _wait_gone(procs: list[Proc], seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while any(_alive(p) for p in procs) and time.monotonic() < deadline:
        time.sleep(_POLL_S)


def stop(chosen: list[tuple[Proc, str]], *, term_grace_s: float, kill_wait_s: float) -> tuple[list[dict], list[dict]]:
    procs = [p for p, _ in chosen]
    termed = {p.pid for p in procs if _signal(p, signal.SIGTERM)}
    _wait_gone(procs, term_grace_s)
    killed = {p.pid for p in procs if _signal(p, signal.SIGKILL)}
    _wait_gone(procs, kill_wait_s)
    stopped, survivors = [], []
    for p, why in chosen:
        row = {"pid": p.pid, "name": p.name, "why": why}
        if _alive(p):
            survivors.append(row)
        elif p.pid in termed or p.pid in killed:
            stopped.append({**row, "signal": "KILL" if p.pid in killed else "TERM"})
    return stopped, survivors


def mark_run(eval_dir: str) -> dict:
    """Mark everything this process starts from here on as this run's, and record the mark in ``eval_dir``.

    The mark is appended to an inherited one rather than replacing it, so an outer run still recognises
    what this one starts. Returns the run record the reaper needs: the mark and the runner's start time.
    """
    mark = uuid.uuid4().hex
    inherited = os.environ.get(MARK_ENV, "")
    os.environ[MARK_ENV] = f"{inherited}:{mark}" if inherited else mark
    st = _stat(os.getpid())
    run = {
        "mark": mark,
        "runner_pid": os.getpid(),
        "start_ticks": st[1] if st else None,
        "eval_dir": str(eval_dir),
        "marked_at": _now(),
    }
    path = Path(eval_dir) / RUN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(run, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return run


def load_run(eval_dir: str) -> dict | None:
    try:
        run = json.loads((Path(eval_dir) / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return run if isinstance(run, dict) and run.get("mark") else None


def _record(eval_dir: str, receipt: dict) -> None:
    try:
        path = Path(eval_dir) / LOG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(receipt) + "\n")
    except OSError:
        pass


def _write_fence(eval_dir: str, receipt: dict) -> str:
    path = Path(eval_dir) / FENCE_FILE
    fence = {"stage": receipt["stage"], "at": receipt["at"], "stopped": len(receipt["stopped"])}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(fence, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return str(path)


def reap(
    eval_dir: str,
    *,
    mode: str,
    stage: str = "",
    run: dict | None = None,
    term_grace_s: float = TERM_GRACE_S,
    kill_wait_s: float = KILL_WAIT_S,
) -> dict:
    """Stop what ``mode`` selects for the run that owns ``eval_dir``; return what was done. Never raises."""
    receipt: dict = {"mode": mode, "stage": stage, "at": _now(), "stopped": [], "survivors": [], "note": ""}
    try:
        if mode not in ("fence", "exit"):
            raise ValueError(f"unknown mode {mode!r}")
        run = run or load_run(eval_dir)
        if not run:
            receipt["note"] = f"no {RUN_FILE} in the eval dir, so this run's processes cannot be told apart; none stopped"
        else:
            roots = eval_dir_roots(eval_dir)
            since = run.get("start_ticks")
            chain = protected_pids()
            keep, workflow = set(chain), workflow_pids(chain, run.get("runner_pid"))
            for _ in range(ROUNDS):
                procs = snapshot(str(run["mark"]), roots, since if isinstance(since, int) else None)
                chosen = select(procs, mode, keep, workflow)
                if not chosen:
                    break
                stopped, survivors = stop(chosen, term_grace_s=term_grace_s, kill_wait_s=kill_wait_s)
                receipt["stopped"].extend(stopped)
                receipt["survivors"] = survivors
                if survivors:
                    break
        if mode == "fence":
            receipt["fence"] = _write_fence(eval_dir, receipt)
    except Exception as exc:  # the caller is handing GPUs over; a reaper fault must not stop that
        receipt["note"] = f"{receipt['note']}; " if receipt["note"] else ""
        receipt["note"] += f"reaper error: {type(exc).__name__}: {exc}"
        receipt["error"] = True
    receipt["ok"] = not receipt["survivors"] and not receipt.get("error")
    _record(eval_dir, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("mode", choices=("fence", "exit"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--stage", default="")
    ap.add_argument("--term-grace-s", type=float, default=TERM_GRACE_S)
    ap.add_argument("--kill-wait-s", type=float, default=KILL_WAIT_S)
    a = ap.parse_args(argv)
    receipt = reap(a.eval_dir, mode=a.mode, stage=a.stage, term_grace_s=a.term_grace_s, kill_wait_s=a.kill_wait_s)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
