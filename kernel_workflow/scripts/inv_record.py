#!/usr/bin/env python3
"""Invocation recorder for engineer commands (Eikos phase 2a: observation only).

Wraps ONE command from the outside -- put it in front of gpu_lock.sh, not inside it, so the ids it
allocates exist before the lock is taken and reach gpu_lock's use-log line:

    python3 inv_record.py run --rec-dir R --workspace W --mode benchmark --domain D.json \\
        [--engineer-id r1_d2] [--identity-file commandment=EVAL/COMMANDMENT.md ...] \\
        -- bash gpu_lock.sh $GPU_ID python3 test_harness.py --benchmark

What it records (append-only, under R, which must lie outside the source domain):
  invocations.jsonl  a `start` event before the child runs and a `completion` event after it.
                     A start with no completion is an incomplete observation, cause unknown.
  raw/<id>.stdout|.stderr  the child's streams, byte for byte.
It observes; it does not decide. It never infers keep/revert/switch from source hashes -- those
exist only as agent declarations (`declare`), stored apart and joined later as consistency evidence.

Child semantics are preserved: stdout/stderr stream through unchanged, the exit status is passed
back (a signal death is re-raised), and no recorder failure changes either. Recorder failures go
to `recorder_error` in the completion event (or recorder_errors.jsonl), never to the child streams.

Source identity: a tree hash of the DECLARED source domain, built with a private git index AND a
private object directory (real objects reached read-only through alternates), so the repository's
index, objects and refs are never written. Before == after shows no NET change during the command,
not that no transient edit happened. Anything the domain cannot cover marks identity incomplete.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

SCHEMA = "inv_record.v1"
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
LATENCY_RE = re.compile(rb"GEAK_RESULT_LATENCY_MS=([-+0-9.eE]+)(.*)")
MODES = ("correctness", "benchmark", "full_benchmark", "profile", "other")
NEXT_CHOICES = ("continue", "switch", "submit")
DECLARE_KINDS = ("keep", "revert", "line", "next", "submit")
STDOUT_PARSE_CAP = 32 * 1024 * 1024


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: str):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _inside(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


# --------------------------------------------------------------------------- source identity
def load_domain(path):
    """The declared source domain. Without one, identity is incomplete by definition."""
    if not path:
        return None, "source domain not declared"
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, "domain config unreadable: %s" % exc
    if not isinstance(d, dict):
        return None, "domain config is not an object"
    out = {"exclude": d.get("exclude", []), "include_ignored": d.get("include_ignored", []),
           "external_deps": d.get("external_deps", [])}
    for k, v in out.items():
        if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            return None, "domain field %s must be a list of non-empty strings" % k
    return out, None


def _git(args, cwd, env=None):
    return subprocess.run(["git"] + args, cwd=cwd, env=env, capture_output=True, timeout=300)


def fingerprint(ws: str, baseline: str, domain, domain_err, rec_dir: str) -> dict:
    """Identity of the measured source. `complete` is False whenever any part is uncovered."""
    gaps = [domain_err] if domain_err else []
    domain = domain or {"exclude": [], "include_ignored": [], "external_deps": []}
    out = {"baseline": None, "tree": None, "external": {}, "complete": False, "gaps": gaps}
    r = _git(["rev-parse", "--verify", baseline + "^{commit}"], ws)
    if r.returncode != 0:
        gaps.append("baseline %r is not a commit" % baseline)
        return out
    out["baseline"] = r.stdout.decode().strip()
    top = _git(["rev-parse", "--show-toplevel"], ws).stdout.decode().strip()
    objects = _git(["rev-parse", "--git-path", "objects"], ws).stdout.decode().strip()
    objects = os.path.abspath(os.path.join(ws, objects))
    excl = [os.path.normpath(e) for e in domain["exclude"]]
    if _inside(rec_dir, ws) and not any(_inside(rec_dir, os.path.join(ws, e)) for e in excl):
        gaps.append("recorder output lies inside the source domain")
    tmp = tempfile.mkdtemp(prefix="inv_fp_")
    try:
        os.makedirs(os.path.join(tmp, "objects"))
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"),
                   GIT_OBJECT_DIRECTORY=os.path.join(tmp, "objects"),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=objects, GIT_OPTIONAL_LOCKS="0")
        # Excluded paths leave the private index entirely (an exclude pathspec on `add` errors when the
        # path is also gitignored). --cached: the working tree is never touched.
        steps = [["read-tree", out["baseline"]], ["add", "-A", "--", "."]]
        steps += [["rm", "-r", "-q", "--cached", "--ignore-unmatch", "--", e] for e in excl]
        for p in domain["include_ignored"]:
            if not os.path.lexists(os.path.join(ws, p)):
                gaps.append("declared ignored path missing: %s" % p)
            else:
                steps.append(["add", "-f", "--", p])
        for args in steps:
            r = _git(args, top, env)
            if r.returncode != 0:
                gaps.append("git %s failed: %s" % (args[0], r.stderr.decode(errors="replace").strip()[:200]))
                return out
        r = _git(["write-tree"], top, env)
        if r.returncode != 0:
            gaps.append("git write-tree failed")
            return out
        out["tree"] = r.stdout.decode().strip()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # Links are hashed as links (git stores the target string). What a link points at outside the
    # workspace is runtime-read content the tree does not cover: it must be a declared external dep.
    ext = [os.path.realpath(p) for p in domain["external_deps"]]
    for dirpath, dirnames, filenames in os.walk(ws):
        rel = os.path.relpath(dirpath, ws)
        dirnames[:] = [d for d in dirnames if d != ".git"
                       and not any(_inside(os.path.join(dirpath, d), os.path.join(ws, e)) for e in excl)]
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                tgt = os.path.realpath(p)
                covered = _inside(tgt, ws) and not any(_inside(tgt, os.path.join(ws, e)) for e in excl)
                if not covered and not any(_inside(tgt, e) for e in ext):
                    gaps.append("symlink %s points outside the hashed domain to an undeclared target"
                                % os.path.normpath(os.path.join(rel, name)))
    for dep in domain["external_deps"]:
        if os.path.isfile(dep):
            out["external"][dep] = file_sha256(dep)
        elif os.path.isdir(dep):
            h = hashlib.sha256()
            for dirpath, dirnames, filenames in os.walk(dep):
                dirnames.sort()
                for name in sorted(filenames):
                    p = os.path.join(dirpath, name)
                    h.update(os.path.relpath(p, dep).encode() + b"\0" + (file_sha256(p) or "?").encode() + b"\0")
            out["external"][dep] = h.hexdigest()
        else:
            gaps.append("declared external dependency missing: %s" % dep)
    out["complete"] = not gaps
    out["identity"] = sha256_bytes(canonical(
        {"baseline": out["baseline"], "tree": out["tree"], "external": out["external"]}).encode())
    return out


# --------------------------------------------------------------------------- recording helpers
class Recorder:
    def __init__(self, rec_dir):
        self.rec_dir, self.errors = rec_dir, []

    def append(self, name, obj) -> bool:
        try:
            os.makedirs(self.rec_dir, exist_ok=True)
            with open(os.path.join(self.rec_dir, name), "a", encoding="utf-8") as fh:
                fh.write(canonical(obj) + "\n")
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.errors.append("append %s: %s" % (name, exc))
            return False


def parse_latencies(stdout: bytes) -> dict:
    cases = []
    for line in stdout.splitlines():
        m = LATENCY_RE.search(line)
        if not m:
            continue
        try:
            ms = float(m.group(1))
        except ValueError:
            continue
        cases.append({"case": m.group(2).decode(errors="replace").strip(), "latency_ms": ms})
    return {"parsed": bool(cases), "format": "GEAK_RESULT_LATENCY_MS.v1", "cases": cases}


def _tee(src, sink, raw_path, keep, rec):
    raw = None
    try:
        raw = open(raw_path, "wb")
    except OSError as exc:
        rec.errors.append("raw open %s: %s" % (raw_path, exc))
    kept = 0
    while True:
        chunk = os.read(src.fileno(), 65536)
        if not chunk:
            break
        try:
            sink.write(chunk)
            sink.flush()
        except (OSError, ValueError):
            pass                                    # our own reader went away; keep draining the child
        if raw is not None:
            try:
                raw.write(chunk)
            except OSError as exc:
                rec.errors.append("raw write %s: %s" % (raw_path, exc))
                raw.close()
                raw = None
        if keep is not None and kept < STDOUT_PARSE_CAP:
            keep.append(chunk)
            kept += len(chunk)
    if raw is not None:
        raw.close()


def cmd_run(a) -> int:
    rec = Recorder(a.rec_dir)
    inv = uuid.uuid4().hex
    engineer = a.engineer_id or os.environ.get("GEAK_ENGINEER_ID") or None
    if engineer is not None and not ID_RE.match(engineer):
        rec.errors.append("engineer id rejected (charset/length)")
        engineer = None
    child_env = dict(os.environ, GEAK_RECORDER_INV_ID=inv)
    if engineer:
        child_env["GEAK_ENGINEER_ID"] = engineer
    try:
        domain, derr = load_domain(a.domain)
        before = fingerprint(a.workspace, a.baseline, domain, derr, a.rec_dir)
    except Exception as exc:  # noqa: BLE001 - never block the child on a recorder fault
        before = {"complete": False, "gaps": ["fingerprint error: %s" % exc]}
        domain, derr = None, "fingerprint error"
    idfiles = {}
    for spec in a.identity_file:
        name, _, path = spec.partition("=")
        idfiles[name or path] = file_sha256(path) if path else None
    measurement = {"mode": a.mode, "argv_sha256": sha256_bytes(canonical(a.cmd).encode()),
                   "identity_files": idfiles, "gpu_spec": a.gpu_spec,
                   "gpu_actual": "join gpu_lock use-log on recorder_inv_id"}
    start = {"schema": SCHEMA, "event": "start", "inv": inv, "engineer_id": engineer, "t": time.time(),
             "workspace": os.path.abspath(a.workspace), "cmd": a.cmd, "measurement": measurement,
             "source_before": before}
    rec.append("invocations.jsonl", start)
    raw_dir = os.path.join(a.rec_dir, "raw")
    try:
        os.makedirs(raw_dir, exist_ok=True)
    except OSError as exc:
        rec.errors.append("raw dir: %s" % exc)
    t0 = time.monotonic()
    out_keep = []
    try:
        proc = subprocess.Popen(a.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_env)
    except OSError as exc:
        sys.stderr.write("inv_record: cannot execute %s: %s\n" % (a.cmd[0], exc))
        rec.append("invocations.jsonl", {"schema": SCHEMA, "event": "completion", "inv": inv, "t": time.time(),
                                         "child": {"exec_error": str(exc), "exit": 127, "signal": None},
                                         "recorder_error": rec.errors or None})
        return 127
    forwarded = []

    def forward(signum, _frame):
        forwarded.append(signum)
        try:
            proc.send_signal(signum)
        except OSError:
            pass
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    threads = [threading.Thread(target=_tee, args=(proc.stdout, sys.stdout.buffer,
                                                  os.path.join(raw_dir, inv + ".stdout"), out_keep, rec)),
               threading.Thread(target=_tee, args=(proc.stderr, sys.stderr.buffer,
                                                  os.path.join(raw_dir, inv + ".stderr"), None, rec))]
    for t in threads:
        t.start()
    rc = proc.wait()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    sig = -rc if rc < 0 else None
    completion = {"schema": SCHEMA, "event": "completion", "inv": inv, "t": time.time(),
                  "child": {"exit": None if sig else rc, "signal": sig, "forwarded_signals": forwarded,
                            "wall_s": round(wall, 3)}}
    try:
        after = fingerprint(a.workspace, a.baseline, domain, derr, a.rec_dir)
        completion["source_after"] = after
        b, f = before.get("identity"), after.get("identity")
        completion["source_status"] = ("incomplete" if not (before.get("complete") and after.get("complete"))
                                       else "no_net_change" if b == f else "source_changed_during_command")
    except Exception as exc:  # noqa: BLE001
        rec.errors.append("fingerprint after: %s" % exc)
        completion["source_status"] = "incomplete"
    completion["measurement"] = parse_latencies(b"".join(out_keep)) if a.mode in (
        "benchmark", "full_benchmark") else {"parsed": False, "format": None, "cases": []}
    # Correctness is evidence only from a correctness-mode command, on this source + oracle identity.
    completion["correctness"] = ({"evidence": "correctness_command", "exit": None if sig else rc,
                                  "signal": sig, "source_identity": before.get("identity"),
                                  "identity_files": idfiles}
                                 if a.mode == "correctness" else {"evidence": "none"})
    completion["recorder_error"] = rec.errors or None
    if not rec.append("invocations.jsonl", completion):
        rec.append("recorder_errors.jsonl", {"inv": inv, "errors": rec.errors})
    if sig:
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        return 128 + sig
    return rc


# --------------------------------------------------------------------------- declarations
def _last_inv(rec_dir):
    try:
        with open(os.path.join(rec_dir, "invocations.jsonl"), encoding="utf-8") as fh:
            starts = [json.loads(l)["inv"] for l in fh if '"event":"start"' in l]
        return starts[-1] if starts else None
    except (OSError, ValueError, KeyError):
        return None


def cmd_declare(a) -> int:
    """Agent-reported events, stored as declarations. `next` is a commitment: first one wins."""
    if not ID_RE.match(a.engineer_id or ""):
        print(canonical({"status": "rejected", "error": "bad engineer id"}))
        return 0
    if a.kind not in DECLARE_KINDS or (a.kind == "next" and (a.value not in NEXT_CHOICES or a.seq is None)):
        print(canonical({"status": "rejected", "error": "next needs --seq and a value in %s" % (NEXT_CHOICES,)}))
        return 0
    rec = Recorder(a.rec_dir)
    did = uuid.uuid4().hex
    d = {"schema": SCHEMA, "decl": did, "engineer_id": a.engineer_id, "kind": a.kind, "value": a.value,
         "seq": a.seq, "t": time.time(), "latest_inv": _last_inv(a.rec_dir), "reported_by": "agent"}
    status = "recorded"
    if a.kind == "next":
        cdir = os.path.join(a.rec_dir, "commitments")
        try:
            os.makedirs(cdir, exist_ok=True)
            fd = os.open(os.path.join(cdir, "%s__%d.json" % (a.engineer_id, a.seq)),
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(canonical(d) + "\n")
        except FileExistsError:
            status, d["conflict"] = "conflict_not_overwritten", True
        except OSError as exc:
            status = "write_failed"
            rec.errors.append(str(exc))
    if not rec.append("declarations.jsonl", d) and status == "recorded":
        status = "write_failed"
    print(canonical({"status": status, "decl": did}))
    return 0


# --------------------------------------------------------------------------- summary (observed vs declared)
def _read_jsonl(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    except OSError:
        return []


def summarize(rec_dir) -> dict:
    events = _read_jsonl(os.path.join(rec_dir, "invocations.jsonl"))
    starts = [e for e in events if e.get("event") == "start"]
    done = {e["inv"]: e for e in events if e.get("event") == "completion"}
    order = [s["inv"] for s in starts]
    src = {s["inv"]: (s.get("source_before") or {}).get("identity") if (s.get("source_before") or {}).get("complete") else None
           for s in starts}
    joins = []
    for d in _read_jsonl(os.path.join(rec_dir, "declarations.jsonl")):
        j = {"decl": d.get("decl"), "kind": d.get("kind"), "value": d.get("value"), "consistency": "unknown"}
        if d.get("kind") in ("keep", "revert") and d.get("latest_inv") in order:
            i = order.index(d["latest_inv"])
            cand, nxt = src.get(order[i]), (src.get(order[i + 1]) if i + 1 < len(order) else None)
            prev = next((src[x] for x in reversed(order[:i]) if src.get(x) and src[x] != cand), None)
            if cand and nxt and prev:
                if d["kind"] == "revert":
                    j["consistency"] = "consistent" if nxt == prev else "inconsistent" if nxt == cand else "unknown"
                else:
                    j["consistency"] = "inconsistent" if nxt == prev else "unknown"
        joins.append(j)
    return {"invocations": len(starts), "completed": sum(1 for i in order if i in done),
            "incomplete_cause_unknown": sum(1 for i in order if i not in done),
            "distinct_complete_sources": len({v for v in src.values() if v}),
            "declarations": joins,
            "note": "actions are never inferred from source hashes; consistency is evidence, not proof"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rec-dir", required=True)
    r.add_argument("--workspace", required=True)
    r.add_argument("--mode", required=True, choices=MODES)
    r.add_argument("--domain")
    r.add_argument("--baseline", default="HEAD")
    r.add_argument("--engineer-id")
    r.add_argument("--gpu-spec")
    r.add_argument("--identity-file", action="append", default=[])
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    d = sub.add_parser("declare")
    d.add_argument("--rec-dir", required=True)
    d.add_argument("--engineer-id", required=True)
    d.add_argument("--kind", required=True)
    d.add_argument("--value", default=None)
    d.add_argument("--seq", type=int)
    s = sub.add_parser("summarize")
    s.add_argument("--rec-dir", required=True)
    a = ap.parse_args(argv)
    if a.sub == "run":
        a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not a.cmd:
            ap.error("run needs a command after --")
        return cmd_run(a)
    if a.sub == "declare":
        return cmd_declare(a)
    print(json.dumps(summarize(a.rec_dir), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
