#!/usr/bin/env python3
"""Serving-GPU lease + process registry for KernelFusion apply-back.

WHY THIS EXISTS
---------------
Qwen3.5-397B-A17B-FP8 (20261010_test_e2e_qwen35): the e03 apply-back integrator
launched its A/B driver with nohup/setsid, returned "not finished", and the driver was
reparented to PID 1. Every later call (e04..e08) then saw all eight serving GPUs held
by a process nobody owned, refused -- correctly, under PROCESS SAFETY -- to kill a
process it had not started, and returned "blocked" without measuring anything. Five
of eight Top-K entries ended "blocked" on a GPU that the workflow itself had occupied.

The fix is ownership, not courage. The WORKFLOW (JS layer) owns the serving GPUs: it
acquires a lease before each apply-back call, and the agent must register every
process it starts (A/B driver, bench, server) under that lease -- pid, pgid and the
/proc start time, frozen at launch. When the agent returns, the workflow waits for
registered processes (bounded) and then reaps ONLY those, by the identity it recorded.
That is "a process this call started", which PROCESS SAFETY has always allowed; it is
never a pattern match, never PID 1, never a group that contains PID 1 or one of our
own ancestors, and never a pid whose start time changed (a recycled pid is a stranger).

Registry file: <lease-dir>/<lease-id>.json (one per apply_one / stacking step / unit job):

    {"schema_version": 1, "lease_id": "apply_e03", "state": "held|released",
     "serving_gpu": "0,1,...", "acquired_at": "...", "released_at": "...",
     "processes": [{"pid", "pgid", "start_ticks", "own_group", "role", "cmd",
                    "registered_at", "adopted_from"}],
     "events": [...]}

Descendants of a registered process are ADOPTED into the registry whenever the
registry is inspected (status / wait / reap), before anything is signalled: a server
that bench_e2e.sh launched in its own session is still a descendant of the registered
driver while the driver lives, and adoption is what lets the workflow find it after the
driver is gone and the server has been reparented to PID 1.

Subcommands (every one prints ONE JSON object on stdout):

  acquire  --lease-id ID --serving-gpu G [--wait-s S]   take the serving lease
  register --lease-id ID --pid P [--role R] [--cmd C]   record a process you started
  launch   --lease-id ID --log F [--role R] -- CMD...   start CMD in its own session,
                                                        register it, return at once
  status   --lease-id ID | --all                        live registered processes
  wait     --lease-id ID | --all --max-s S [--release]  poll until none alive (bounded)
  reap     --lease-id ID | --all [--grace-s S]          TERM->KILL registered survivors
  release  --lease-id ID                                mark released (refused if alive)
  reap-recorded --eval-dir D [--grace-s S]              reap servers bench_e2e.sh recorded
                                                        under D (server_identity.json)

Exit codes: 0 ok / nothing alive; 2 busy / still alive; 1 usage or safety refusal.

`wait --max-s` is capped at 570 s so ONE call fits inside an agent's 10-minute Bash
tool limit. A long A/B is polled by repeating that call until it reports alive=0;
never return from the agent while it reports anything alive.
"""
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

SCHEMA_VERSION = 1
WAIT_CAP_S = 570
PROC_ROOT = os.environ.get("GEAK_LEASE_PROC_ROOT", "/proc")


# --------------------------------------------------------------------------- /proc
def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _stat_fields(pid):
    """Fields of /proc/PID/stat from `state` (field 3) on, or None when gone.

    comm (field 2) can contain spaces AND parens, so cut through the LAST ') '."""
    try:
        with open(os.path.join(PROC_ROOT, str(int(pid)), "stat")) as fh:
            raw = fh.read()
    except (OSError, ValueError):
        return None
    tail = raw.rsplit(") ", 1)
    if len(tail) != 2:
        return None
    return tail[1].split()


def start_ticks(pid):
    fields = _stat_fields(pid)
    # field 22 (starttime) is index 19 counting from field 3
    return fields[19] if fields and len(fields) > 19 else None


def ppid_of(pid):
    fields = _stat_fields(pid)
    try:
        return int(fields[1]) if fields else None
    except (ValueError, IndexError):
        return None


def pgid_of(pid):
    fields = _stat_fields(pid)
    try:
        return int(fields[2]) if fields else None
    except (ValueError, IndexError):
        return None


def is_zombie(pid):
    fields = _stat_fields(pid)
    return bool(fields) and fields[0] in ("Z", "X")


def cmdline_of(pid):
    try:
        with open(os.path.join(PROC_ROOT, str(int(pid)), "cmdline"), "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (OSError, ValueError):
        return ""


def all_pids():
    out = []
    try:
        names = os.listdir(PROC_ROOT)
    except OSError:
        return out
    for name in names:
        if name.isdigit():
            out.append(int(name))
    return out


def ancestors_of_self():
    """Our own pid and every ancestor up to (and including) PID 1."""
    chain = []
    pid = os.getpid()
    for _ in range(128):
        chain.append(pid)
        if pid <= 1:
            break
        parent = ppid_of(pid)
        if parent is None or parent == pid:
            break
        pid = parent
    if 1 not in chain:
        chain.append(1)
    return chain


def protected_pgids():
    """Groups we must never signal: PID 1's, ours, and every ancestor's."""
    out = {0, 1}
    for pid in ancestors_of_self():
        g = pgid_of(pid)
        if g is not None:
            out.add(g)
    return out


def identity_alive(rec):
    """The recorded process still exists AND is the same process (start time)."""
    pid = rec.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    now = start_ticks(pid)
    if now is None or is_zombie(pid):
        return False
    recorded = rec.get("start_ticks")
    return recorded in (None, "") or str(recorded) == str(now)


def descendants(root_pids):
    """Transitive descendants of root_pids (excluding the roots, PID 1 and us)."""
    parent = {}
    for pid in all_pids():
        p = ppid_of(pid)
        if p is not None:
            parent[pid] = p
    marked = set(root_pids)
    changed = True
    while changed:
        changed = False
        for pid, p in parent.items():
            if pid not in marked and p in marked:
                marked.add(pid)
                changed = True
    skip = set(root_pids) | set(ancestors_of_self())
    return sorted(p for p in marked if p not in skip and p > 1)


# --------------------------------------------------------------------------- registry
def _path(lease_dir, lease_id):
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(lease_id))
    if not safe:
        raise SystemExit("empty --lease-id")
    return os.path.join(lease_dir, safe + ".json")


class _Locked(object):
    """Directory-wide flock so concurrent agents never interleave read-modify-write."""

    def __init__(self, lease_dir):
        os.makedirs(lease_dir, exist_ok=True)
        self.fh = open(os.path.join(lease_dir, ".registry.lock"), "a+")

    def __enter__(self):
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        self.fh.close()


def load(lease_dir, lease_id):
    path = _path(lease_dir, lease_id)
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def save(lease_dir, doc):
    path = _path(lease_dir, doc["lease_id"])
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def new_doc(lease_id, serving_gpu=""):
    return {"schema_version": SCHEMA_VERSION, "lease_id": str(lease_id),
            "serving_gpu": str(serving_gpu or ""), "state": "held",
            "acquired_at": _now(), "released_at": "", "processes": [], "events": []}


def all_leases(lease_dir):
    out = []
    try:
        names = sorted(os.listdir(lease_dir))
    except OSError:
        return out
    for name in names:
        if name.endswith(".json") and not name.startswith("."):
            try:
                with open(os.path.join(lease_dir, name)) as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and doc.get("lease_id"):
                out.append(doc)
    return out


def _record(pid, role="", cmd="", adopted_from=None):
    pgid = pgid_of(pid)
    return {"pid": int(pid), "pgid": pgid, "start_ticks": start_ticks(pid),
            # pgid == pid is the ONLY structural proof that a group belongs to this
            # launch: a group led by our own process can hold nothing but its own
            # descendants, so it cannot contain PID 1 or the caller's tree.
            "own_group": bool(pgid is not None and pgid == int(pid)),
            "role": role or ("adopted" if adopted_from else ""),
            "cmd": (cmd or cmdline_of(pid))[:400],
            "registered_at": _now(), "adopted_from": adopted_from}


def adopt(doc):
    """Register live descendants of live registered processes. Returns new records."""
    live_roots = [r["pid"] for r in doc.get("processes", []) if identity_alive(r)]
    if not live_roots:
        return []
    known = {r["pid"] for r in doc.get("processes", [])}
    added = []
    for pid in descendants(live_roots):
        if pid in known or start_ticks(pid) is None:
            continue
        parent = ppid_of(pid)
        rec = _record(pid, adopted_from=parent)
        doc.setdefault("processes", []).append(rec)
        added.append(rec)
    return added


def live(doc):
    return [r for r in doc.get("processes", []) if identity_alive(r)]


def _summary(doc):
    alive = live(doc)
    return {"lease_id": doc.get("lease_id"), "state": doc.get("state"),
            "alive_count": len(alive),
            "alive": [{k: r.get(k) for k in ("pid", "pgid", "role", "cmd")} for r in alive],
            "registered": len(doc.get("processes", []))}


# --------------------------------------------------------------------------- serving lock
def serving_lock_path(serving_gpu, override=""):
    if override:
        return override
    key = str(serving_gpu or "0").replace(",", "_")
    return "/tmp/geak_serving_gpu_%s.lock" % key


def serving_lock_free(path):
    """Non-blocking probe of bench_e2e.sh's serving-GPU flock. Holds nothing after."""
    try:
        fh = open(path, "a+")
    except OSError:
        return True   # cannot even open it -> nobody can hold it through this path
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()
    return True


# --------------------------------------------------------------------------- commands
def cmd_acquire(a):
    deadline = time.time() + max(0, min(float(a.wait_s or 0), WAIT_CAP_S))
    lock_path = serving_lock_path(a.serving_gpu, a.serving_lock)
    while True:
        with _Locked(a.lease_dir):
            holders = []
            for doc in all_leases(a.lease_dir):
                if doc.get("state") != "held":
                    continue
                adopt(doc)
                alive = live(doc)
                if alive:
                    holders.append(_summary(doc))
                    save(a.lease_dir, doc)
                elif doc.get("lease_id") != a.lease_id:
                    # Held with nothing alive: its call returned cleanly but nobody
                    # released it. Stale, not busy.
                    doc["state"] = "released"
                    doc["released_at"] = _now()
                    doc.setdefault("events", []).append(
                        {"at": _now(), "event": "stale_released_by_acquire",
                         "by": a.lease_id})
                    save(a.lease_dir, doc)
            lock_free = serving_lock_free(lock_path)
            if not holders and lock_free:
                doc = load(a.lease_dir, a.lease_id)
                if doc is None or doc.get("state") == "released":
                    doc = new_doc(a.lease_id, a.serving_gpu)
                doc["state"] = "held"
                doc["serving_gpu"] = str(a.serving_gpu or doc.get("serving_gpu") or "")
                doc.setdefault("events", []).append({"at": _now(), "event": "acquired"})
                path = save(a.lease_dir, doc)
                return 0, {"ok": True, "action": "acquire", "lease_id": a.lease_id,
                           "lease_path": path, "serving_lock": lock_path,
                           "serving_lock_held": False, "busy_holders": []}
        if time.time() >= deadline:
            return 2, {"ok": False, "action": "acquire", "lease_id": a.lease_id,
                       "lease_path": _path(a.lease_dir, a.lease_id),
                       "serving_lock": lock_path, "serving_lock_held": not lock_free,
                       # Workflow-owned leases with live processes: the workflow may
                       # wait for / reap these (it started them). A held serving lock
                       # with NO such holder belongs to someone else: wait, never kill.
                       "busy_holders": holders,
                       "reason": ("gpu_busy: %s" % (
                           ", ".join(h["lease_id"] for h in holders) if holders
                           else "serving-GPU lock %s held by a process outside the "
                                "lease registry" % lock_path))}
        time.sleep(max(1.0, float(a.poll_s)))


def _refuse_pid(pid):
    if pid <= 1:
        return "pid %d is container init / invalid" % pid
    if pid in ancestors_of_self():
        return "pid %d is an ancestor of this helper (your own shell / the orchestrator)" % pid
    if start_ticks(pid) is None:
        return "pid %d is not running" % pid
    return ""


def cmd_register(a):
    pid = int(a.pid)
    why = _refuse_pid(pid)
    if why:
        return 1, {"ok": False, "action": "register", "lease_id": a.lease_id,
                   "pid": pid, "reason": why}
    with _Locked(a.lease_dir):
        doc = load(a.lease_dir, a.lease_id) or new_doc(a.lease_id)
        if doc.get("state") == "released":
            doc["state"] = "held"
        rec = None
        for r in doc.get("processes", []):
            if r.get("pid") == pid and str(r.get("start_ticks")) == str(start_ticks(pid)):
                rec = r
        if rec is None:
            rec = _record(pid, a.role, a.cmd)
            doc.setdefault("processes", []).append(rec)
        path = save(a.lease_dir, doc)
    return 0, {"ok": True, "action": "register", "lease_id": a.lease_id,
               "lease_path": path, "process": rec}


def cmd_launch(a):
    argv = list(a.command or [])
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        return 1, {"ok": False, "action": "launch", "reason": "no command after --"}
    os.makedirs(os.path.dirname(os.path.abspath(a.log)) or ".", exist_ok=True)
    with open(a.log, "ab") as log_fh:
        # Own session => pgid == pid, so the workflow can reap the whole group by the
        # identity recorded here, and nothing else.
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log_fh,
                                stderr=subprocess.STDOUT, cwd=a.cwd or None,
                                start_new_session=True)
    # Give /proc a moment so the start time is readable.
    for _ in range(50):
        if start_ticks(proc.pid) is not None:
            break
        time.sleep(0.02)
    with _Locked(a.lease_dir):
        doc = load(a.lease_dir, a.lease_id) or new_doc(a.lease_id)
        if doc.get("state") == "released":
            doc["state"] = "held"
        rec = _record(proc.pid, a.role or "launched", " ".join(argv))
        doc.setdefault("processes", []).append(rec)
        path = save(a.lease_dir, doc)
    return 0, {"ok": True, "action": "launch", "lease_id": a.lease_id,
               "lease_path": path, "pid": proc.pid, "pgid": rec.get("pgid"),
               "log": os.path.abspath(a.log),
               "next": "poll with: wait --lease-id %s --max-s 540 (repeat until alive_count==0)"
                       % a.lease_id}


def _targets(a):
    if a.all:
        return [d for d in all_leases(a.lease_dir) if d.get("state") == "held"]
    doc = load(a.lease_dir, a.lease_id)
    return [doc] if doc else []


def cmd_status(a):
    with _Locked(a.lease_dir):
        docs = _targets(a)
        for doc in docs:
            adopt(doc)
            save(a.lease_dir, doc)
        leases = [_summary(d) for d in docs]
    alive = sum(x["alive_count"] for x in leases)
    return (2 if alive else 0), {"ok": True, "action": "status", "alive_count": alive,
                                 "leases": leases}


def cmd_wait(a):
    max_s = max(0.0, min(float(a.max_s), WAIT_CAP_S))
    deadline = time.time() + max_s
    while True:
        code, out = cmd_status(a)
        if code == 0 or time.time() >= deadline:
            break
        time.sleep(max(1.0, float(a.poll_s)))
    out["action"] = "wait"
    out["waited_max_s"] = max_s
    if code == 0 and a.release:
        with _Locked(a.lease_dir):
            released = []
            for doc in _targets(a):
                if not live(doc):
                    doc["state"] = "released"
                    doc["released_at"] = _now()
                    doc.setdefault("events", []).append({"at": _now(), "event": "released_clean"})
                    save(a.lease_dir, doc)
                    released.append(doc["lease_id"])
        out["released"] = released
    return code, out


def _signal_record(rec, sig, protected, refused, sent):
    """Signal ONE registered process by its recorded identity. Never a pattern."""
    pid = rec.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        refused.append({"pid": pid, "reason": "pid <= 1"})
        return
    if not identity_alive(rec):
        return
    if pid in ancestors_of_self():
        refused.append({"pid": pid, "reason": "ancestor of this helper"})
        return
    pgid = rec.get("pgid")
    group_ok = bool(rec.get("own_group") and isinstance(pgid, int) and pgid > 1
                    and pgid not in protected and pgid_of(pid) == pgid)
    if group_ok:
        # Defence in depth: refuse a live group that somehow holds PID 1 or an ancestor.
        anc = set(ancestors_of_self())
        members = [p for p in all_pids() if pgid_of(p) == pgid]
        if any(m in anc or m == 1 for m in members):
            group_ok = False
            refused.append({"pid": pid, "reason": "group %s contains pid 1 / an ancestor; "
                                                  "falling back to pid-only" % pgid})
    try:
        if group_ok:
            os.killpg(pgid, sig)
            sent.append({"pid": pid, "pgid": pgid, "signal": int(sig), "mode": "group"})
        else:
            os.kill(pid, sig)
            sent.append({"pid": pid, "signal": int(sig), "mode": "pid"})
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        refused.append({"pid": pid, "reason": "permission: %s" % exc})


def cmd_reap(a):
    grace = max(0.0, min(float(a.grace_s), 120.0))
    protected = protected_pgids()
    report = {"ok": True, "action": "reap", "leases": [], "signalled": [], "refused": []}
    with _Locked(a.lease_dir):
        docs = _targets(a)
        for doc in docs:
            adopt(doc)                       # map the tree BEFORE signalling anything
            save(a.lease_dir, doc)
        for doc in docs:
            for rec in live(doc):
                _signal_record(rec, signal.SIGTERM, protected, report["refused"],
                               report["signalled"])
    end = time.time() + grace
    while time.time() < end and any(live(d) for d in docs):
        time.sleep(0.25)
    with _Locked(a.lease_dir):
        for doc in docs:
            for rec in live(doc):
                # identity_alive() re-checked the start time: a pid recycled during the
                # grace window is a stranger and is not escalated against.
                _signal_record(rec, signal.SIGKILL, protected, report["refused"],
                               report["signalled"])
        time.sleep(0.2)
        for doc in docs:
            still = live(doc)
            doc.setdefault("events", []).append(
                {"at": _now(), "event": "reaped", "still_alive": [r["pid"] for r in still]})
            if not still:
                doc["state"] = "released"
                doc["released_at"] = _now()
            save(a.lease_dir, doc)
            report["leases"].append(_summary(doc))
    alive = sum(x["alive_count"] for x in report["leases"])
    report["alive_count"] = alive
    return (2 if alive else 0), report


def cmd_release(a):
    with _Locked(a.lease_dir):
        doc = load(a.lease_dir, a.lease_id)
        if doc is None:
            return 0, {"ok": True, "action": "release", "lease_id": a.lease_id,
                       "note": "no such lease"}
        adopt(doc)
        alive = live(doc)
        if alive:
            save(a.lease_dir, doc)
            return 2, {"ok": False, "action": "release", "lease_id": a.lease_id,
                       "reason": "registered processes still alive; wait or reap first",
                       **_summary(doc)}
        doc["state"] = "released"
        doc["released_at"] = _now()
        doc.setdefault("events", []).append({"at": _now(), "event": "released"})
        save(a.lease_dir, doc)
    return 0, {"ok": True, "action": "release", "lease_id": a.lease_id}


RECORDED_LEASE_ID = "recorded-servers"


def _recorded_servers(eval_dir):
    """Server identities bench_e2e.sh froze at launch, anywhere under eval_dir."""
    out = []
    for root, _dirs, files in os.walk(eval_dir):
        if "server_identity.json" not in files:
            continue
        path = os.path.join(root, "server_identity.json")
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        pid = doc.get("pid")
        ticks = str(doc.get("start_ticks") or "")
        # No start time, no identity: a bare pid may already belong to a stranger.
        if not isinstance(pid, int) or pid <= 1 or not ticks:
            continue
        pgid = doc.get("pgid")
        out.append({"pid": pid, "pgid": pgid if isinstance(pgid, int) else None,
                    "start_ticks": ticks,
                    "own_group": isinstance(pgid, int) and pgid == pid,
                    "role": "recorded_server", "cmd": path,
                    "registered_at": _now(), "adopted_from": None})
    return out


def cmd_reap_recorded(a):
    """Reap servers this run launched and left behind, by their recorded identity.

    Qwen3.5-397B-A17B-FP8 (20261010_test_e2e_qwen35, fusion-only run): the Setup
    director reused a server an interrupted baseline attempt had left up and never
    stopped it. The server had inherited bench_e2e.sh's serving-GPU lock fd, so the
    KernelFusion shape-capture replay waited 7200 s on the lock, timed out, and the
    decode layer table was never built -- the screen then ran on prefill alone.
    Every server bench_e2e.sh launches writes server_identity.json (pid, pgid, /proc
    start time) next to its output; those are processes this run started, so the
    workflow may stop them at a phase boundary. A record whose start time no longer
    matches is a recycled pid and is left alone.
    """
    eval_dir = os.path.abspath(a.eval_dir)
    found = [r for r in _recorded_servers(eval_dir) if identity_alive(r)]
    if not found:
        return 0, {"ok": True, "action": "reap-recorded", "eval_dir": eval_dir,
                   "alive_count": 0, "signalled": [], "refused": []}
    os.makedirs(a.lease_dir, exist_ok=True)
    with _Locked(a.lease_dir):
        doc = load(a.lease_dir, RECORDED_LEASE_ID) or {
            "schema_version": SCHEMA_VERSION, "lease_id": RECORDED_LEASE_ID,
            "acquired_at": _now(), "processes": [], "events": []}
        known = {(r.get("pid"), str(r.get("start_ticks"))) for r in doc["processes"]}
        for rec in found:
            if (rec["pid"], rec["start_ticks"]) not in known:
                doc["processes"].append(rec)
        doc["state"] = "held"
        doc.setdefault("events", []).append(
            {"at": _now(), "event": "recorded_servers_found",
             "pids": [r["pid"] for r in found]})
        save(a.lease_dir, doc)
    a.lease_id, a.all = RECORDED_LEASE_ID, False
    code, report = cmd_reap(a)
    report["action"] = "reap-recorded"
    report["eval_dir"] = eval_dir
    return code, report


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, allow_all=False):
        p.add_argument("--lease-dir", required=True)
        if allow_all:
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--lease-id")
            g.add_argument("--all", action="store_true")
        else:
            p.add_argument("--lease-id", required=True)
            p.set_defaults(all=False)

    p = sub.add_parser("acquire")
    common(p)
    p.add_argument("--serving-gpu", default="")
    p.add_argument("--serving-lock", default="")
    p.add_argument("--wait-s", type=float, default=0)
    p.add_argument("--poll-s", type=float, default=5)
    p.set_defaults(fn=cmd_acquire)

    p = sub.add_parser("register")
    common(p)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--role", default="")
    p.add_argument("--cmd", default="")
    p.set_defaults(fn=cmd_register)

    p = sub.add_parser("launch")
    common(p)
    p.add_argument("--log", required=True)
    p.add_argument("--role", default="")
    p.add_argument("--cwd", default="")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_launch)

    p = sub.add_parser("status")
    common(p, allow_all=True)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("wait")
    common(p, allow_all=True)
    p.add_argument("--max-s", type=float, default=540)
    p.add_argument("--poll-s", type=float, default=10)
    p.add_argument("--release", action="store_true")
    p.set_defaults(fn=cmd_wait)

    p = sub.add_parser("reap")
    common(p, allow_all=True)
    p.add_argument("--grace-s", type=float, default=15)
    p.set_defaults(fn=cmd_reap)

    p = sub.add_parser("release")
    common(p)
    p.set_defaults(fn=cmd_release)

    p = sub.add_parser("reap-recorded")
    p.add_argument("--lease-dir", required=True)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--grace-s", type=float, default=15)
    p.set_defaults(fn=cmd_reap_recorded)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    code, out = a.fn(a)
    print(json.dumps(out, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
