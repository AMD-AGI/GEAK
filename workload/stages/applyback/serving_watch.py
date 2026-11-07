#!/usr/bin/env python3
"""Is the serving container still serving? Fail a leg in minutes when it dies, not at its timeout.

A server can die without its container exiting: on 2026-11-06 every scheduler rank aborted on
a GPU fault three minutes into a CI-length leg, the HTTP front end kept answering /metrics,
the client waited with nine requests in flight, and the leg only gave up at its 3 h timeout.
`/health` and "the container is running" both kept saying yes.

    serving_watch.py scan  --container NAME [--since RFC3339]
        one look: exit 0 alive, 3 dead (reason as JSON on stdout)
    serving_watch.py watch --container NAME --client-pid PID --out FILE [--interval S]
        poll until the client exits; if the server dies first, stop the client's process
        group and write {"dead": true, ...} to FILE (exit 3), else {"dead": false} (exit 0)

Dead means: the container is not running, or its log carries a fatal line. The patterns are
the ones a dead GPU server leaves regardless of which kernel killed it (a fault in the HSA/HIP
runtime, a Python abort, a scheduler subprocess reported crashed, a collective timeout).
Stdlib only.
"""
import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time

FATAL = [re.compile(p) for p in (
    r"Fatal Python error",
    r"HSA_STATUS_ERROR",
    r"Memory access fault",
    r"hipError(?!Success)\w+",
    r"CUDA error: ",
    r"Segmentation fault",
    r"core dumped",
    r"[Ss]ubprocess \S+ .*crashed with exit code",
    r"Watchdog caught collective operation timeout",
    r"(NCCL|RCCL) error",
)]


def fatal_line(text):
    """The first fatal line in a log text, or None."""
    for line in text.splitlines():
        for pat in FATAL:
            if pat.search(line):
                return line.strip()[:500]
    return None


def _docker(*args, timeout=60):
    p = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def scan(container, since=None):
    """{"dead": bool, "reason": ...} from one look at the container and its log."""
    rc, out = _docker("inspect", "-f", "{{.State.Running}}", container)
    if rc != 0 or out.strip() != "true":
        return {"dead": True, "reason": "container not running", "detail": out.strip()[:300]}
    args = ["logs"] + (["--since", since] if since else []) + [container]
    _, log = _docker(*args, timeout=120)
    line = fatal_line(log)
    if line:
        return {"dead": True, "reason": "fatal line in the server log", "line": line}
    return {"dead": False}


def _alive(pid):
    """Running, not merely unreaped: the client's parent (the leg's shell) only reaps it after
    the watch returns, so an exited client stays a zombie that os.kill(pid, 0) still finds."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return True


def _stop_group(pid):
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        end = time.time() + wait
        while time.time() < end and _alive(pid):
            time.sleep(1)


def watch(container, client_pid, out, interval=15):
    since, checks = None, 0
    while _alive(client_pid):
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = scan(container, since)
        checks += 1
        if r["dead"]:
            r.update(time=now, checks=checks, action="stopped the client")
            _stop_group(client_pid)
            json.dump(r, open(out, "w"), indent=1)
            return 3
        since = now
        time.sleep(interval)
    r = scan(container, since)          # a death in the last interval still counts
    r.update(checks=checks + 1, time=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(r, open(out, "w"), indent=1)
    return 3 if r["dead"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--container", required=True)
    s.add_argument("--since")
    w = sub.add_parser("watch")
    w.add_argument("--container", required=True)
    w.add_argument("--client-pid", type=int, required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--interval", type=float, default=15)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        r = scan(a.container, a.since)
        print(json.dumps(r))
        return 3 if r["dead"] else 0
    return watch(a.container, a.client_pid, a.out, a.interval)


if __name__ == "__main__":
    sys.exit(main())
