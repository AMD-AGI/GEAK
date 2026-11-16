#!/usr/bin/env python3
"""Invocation recorder for engineer commands (Eikos phase 2a: observation only).

Wraps ONE command from the outside -- put it in front of gpu_lock.sh, not inside it, so the ids it
allocates exist before the lock is taken and reach gpu_lock's use-log line:

    python3 inv_record.py run --rec-dir R --workspace W --mode benchmark --domain D.json \\
        [--engineer-id r1_d2] [--identity-file commandment=EVAL/COMMANDMENT.md ...] \\
        -- bash gpu_lock.sh $GPU_ID python3 test_harness.py --benchmark

What it records (append-only, under R, which must lie outside the source domain):
  invocations.jsonl  a `start` event before the child runs and a `completion` event after it.
                     A start with no completion -- or a torn completion line -- is an incomplete
                     observation, cause unknown.
  raw/<id>.stdout|.stderr  the child's streams, byte for byte.
It observes; it does not decide. It never infers keep/revert/switch from source hashes -- those
exist only as agent declarations (`declare`), stored apart and joined later as consistency evidence.

Child semantics are preserved: stdout/stderr stream through unchanged and the exit status is passed
back (a signal death is re-raised). The child runs in its own process group; a termination signal
the recorder receives is forwarded to that whole group, as a terminal would deliver it. If
descendants still hold the child's streams after the child exits, the recorder stops reading after
a short grace period and says so; it does not kill them. No recorder failure (open, write, flush,
close, read) reaches the child's streams: it goes to `recorder_error` (or recorder_errors.jsonl).

Source identity: a tree hash of the DECLARED source domain -- the workspace subtree only, even when
the workspace is a subdirectory of its repository -- built with a private git index AND a private
object directory (real objects reached read-only through alternates), so the repository's index,
objects and refs are never written. Ignored files present in the domain must be declared
(`include_ignored`) or excluded; otherwise identity is incomplete. A symlink counts as covered only
if its target is content the tree actually captured, or a declared external dependency.
Before == after shows no NET change during the command, not that no transient edit happened.

Scope of measurement identity in this version: command mode, argv hash, hashes of caller-named
identity files (e.g. COMMANDMENT, harness, oracle), and the parsed case ids. Metric, weights and
input regime are covered only through those files' hashes; the actual GPU comes from joining the
gpu_lock use-log on recorder_inv_id (`summarize --gpu-use-log`). Whether two invocations are
comparable is decided later (stage 2b), not here.
"""
import argparse
import hashlib
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

SCHEMA = "inv_record.v5"
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
LATENCY_RE = re.compile(rb"GEAK_RESULT_LATENCY_MS=(\S*)(.*)")
MODES = ("correctness", "benchmark", "full_benchmark", "profile", "other")
NEXT_CHOICES = ("continue", "switch", "submit")
DECLARE_KINDS = ("keep", "revert", "line", "next", "submit")
STDOUT_PARSE_CAP = 32 * 1024 * 1024
DRAIN_GRACE_S = float(os.environ.get("GEAK_INV_RECORD_DRAIN_GRACE_S", "2.0"))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


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


def _excluded(rel: str, excl) -> bool:
    """rel is workspace-relative with '/' separators. An entry with a '/' is a path prefix; a bare
    name (e.g. __pycache__) matches that component anywhere."""
    rel = rel.strip("/")
    parts = rel.split("/")
    for e in excl:
        e = e.strip("/")
        if "/" in e:
            if rel == e or rel.startswith(e + "/"):
                return True
        elif e in parts:
            return True
    return False


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
    out = {"baseline": None, "tree": None, "workspace_prefix": None, "external": {},
           "complete": False, "gaps": gaps}
    r = _git(["rev-parse", "--verify", baseline + "^{commit}"], ws)
    if r.returncode != 0:
        gaps.append("baseline %r is not a commit" % baseline)
        return out
    out["baseline"] = r.stdout.decode().strip()
    top = _git(["rev-parse", "--show-toplevel"], ws).stdout.decode().strip()
    prefix = _git(["rev-parse", "--show-prefix"], ws).stdout.decode().strip()   # "" or "sub/dir/"
    out["workspace_prefix"] = prefix
    objects = os.path.abspath(os.path.join(ws, _git(["rev-parse", "--git-path", "objects"], ws).stdout.decode().strip()))
    excl = domain["exclude"]
    if _inside(rec_dir, ws) and not _excluded(os.path.relpath(os.path.realpath(rec_dir), os.path.realpath(ws)), excl):
        gaps.append("recorder output lies inside the source domain")
    spec = prefix.rstrip("/") or "."
    tmp = tempfile.mkdtemp(prefix="inv_fp_")
    index_paths = set()
    try:
        os.makedirs(os.path.join(tmp, "objects"))
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"),
                   GIT_OBJECT_DIRECTORY=os.path.join(tmp, "objects"),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=objects, GIT_OPTIONAL_LOCKS="0")
        # Only the workspace subtree is refreshed; the rest of the repo keeps its baseline content in
        # the private index and is not part of the hashed subtree anyway. Excluded paths leave the
        # private index entirely (an exclude pathspec on `add` errors on gitignored paths).
        steps = [["read-tree", out["baseline"]], ["add", "-A", "--", spec]]
        for e in excl:
            e = e.strip("/")
            specs = [prefix + e] if "/" in e else [":(glob)%s**/%s" % (prefix, e), ":(glob)%s**/%s/**" % (prefix, e)]
            steps.append(["rm", "-r", "-q", "--cached", "--ignore-unmatch", "--"] + specs)
        for p in domain["include_ignored"]:
            if not os.path.lexists(os.path.join(ws, p)):
                gaps.append("declared ignored path missing: %s" % p)
            else:
                steps.append(["add", "-f", "--", prefix + p.strip("/")])
        for args in steps:
            r = _git(args, top, env)
            if r.returncode != 0:
                gaps.append("git %s failed: %s" % (args[0], r.stderr.decode(errors="replace").strip()[:200]))
                return out
        r = _git(["write-tree"] + (["--prefix=" + prefix] if prefix else []), top, env)
        if r.returncode != 0:
            gaps.append("git write-tree failed: %s" % r.stderr.decode(errors="replace").strip()[:200])
            return out
        out["tree"] = r.stdout.decode().strip()
        # Ignored content still present in the domain: the build may read it, the tree does not hold it.
        r = _git(["ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory", "--", spec], top, env)
        if r.returncode != 0:
            gaps.append("git ls-files (ignored) failed")
        else:
            undeclared = [p for p in (x.decode(errors="replace") for x in r.stdout.split(b"\0") if x)
                          if not _excluded(p[len(prefix):], excl)
                          and not _inside(os.path.join(top, p), rec_dir)]
            if undeclared:
                gaps.append("ignored content in the domain is neither declared nor excluded: %s"
                            % ", ".join(sorted(undeclared)[:5]) + (" ..." if len(undeclared) > 5 else ""))
        r = _git(["ls-files", "-z", "--", spec], top, env)
        index_paths = {x.decode(errors="replace") for x in r.stdout.split(b"\0") if x}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # Links are hashed as links (git stores the target string). A link is covered only if what it
    # points at is content the tree captured, or a declared external dependency.
    ext = [os.path.realpath(p) for p in domain["external_deps"]]
    rws = os.path.realpath(ws)
    for dirpath, dirnames, filenames in os.walk(ws):
        rel_dir = os.path.relpath(dirpath, ws)
        keep = []
        for d in dirnames:
            rel = os.path.normpath(os.path.join(rel_dir, d)).replace(os.sep, "/")
            if d != ".git" and not _excluded(rel, excl):
                keep.append(d)
        dirnames[:] = keep
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            p = os.path.join(dirpath, name)
            rel = os.path.normpath(os.path.join(rel_dir, name)).replace(os.sep, "/")
            if not os.path.islink(p) or _excluded(rel, excl):
                continue
            tgt = os.path.realpath(p)
            if any(_inside(tgt, e) for e in ext):
                continue
            covered = False
            if _inside(tgt, rws) and os.path.exists(tgt):
                rel_t = os.path.relpath(tgt, rws).replace(os.sep, "/")
                if not _excluded(rel_t, excl):
                    covered = os.path.isdir(tgt) or (prefix + rel_t) in index_paths
            if not covered:
                gaps.append("symlink %s points at content the tree did not capture and no declared "
                            "external dependency covers" % rel)
    for dep in domain["external_deps"]:
        if os.path.isfile(dep):
            digest = file_sha256(dep)
            if digest is None:
                gaps.append("declared external dependency unreadable: %s" % dep)
            out["external"][dep] = digest
        elif os.path.isdir(dep):
            out["external"][dep] = _hash_dep_tree(dep, gaps)
        else:
            gaps.append("declared external dependency missing: %s" % dep)
    out["complete"] = not gaps
    out["identity"] = sha256_bytes(canonical(
        {"baseline": out["baseline"], "prefix": prefix, "tree": out["tree"], "external": out["external"]}).encode())
    return out


def _hash_dep_tree(root: str, gaps: list) -> str:
    """Content hash of a declared dependency directory AS IT IS READ: symlinks are recorded (link
    text) AND followed, so content reached through a nested link is covered. A directory already
    visited on the current path is a cycle: recorded, not re-entered. Unreadable or dangling
    entries, and special files (fifo/socket/device, never opened), are gaps (identity incomplete),
    never silently skipped."""
    h = hashlib.sha256()

    def walk(path, rel, ancestors):
        real = os.path.realpath(path)
        if real in ancestors:
            h.update(b"C" + rel.encode() + b"\0")
            return
        try:
            names = sorted(os.listdir(path))
        except OSError as exc:
            gaps.append("external dependency unreadable: %s (%s)" % (path, exc.strerror))
            return
        for name in names:
            p, r = os.path.join(path, name), (rel + "/" + name).lstrip("/")
            if os.path.islink(p):
                try:
                    h.update(b"L" + r.encode() + b"\0" + os.readlink(p).encode() + b"\0")
                except OSError as exc:
                    gaps.append("external dependency link unreadable: %s (%s)" % (p, exc.strerror))
                    continue
            if os.path.isdir(p):
                h.update(b"D" + r.encode() + b"\0")
                walk(p, r, ancestors | {real})
            elif os.path.isfile(p):
                digest = file_sha256(p)
                if digest is None:
                    gaps.append("external dependency file unreadable: %s" % p)
                h.update(b"F" + r.encode() + b"\0" + (digest or "?").encode() + b"\0")
            elif not os.path.exists(p):
                gaps.append("external dependency link dangles: %s" % p)
            else:
                # FIFO/socket/device: presence says nothing about what a run reads from it. Not
                # opened (a FIFO read could block or consume data) -- just an identity gap.
                gaps.append("external dependency holds a special file (fifo/socket/device): %s" % p)
    walk(root, "", frozenset())
    return h.hexdigest()


# --------------------------------------------------------------------------- recording helpers
class Recorder:
    def __init__(self, rec_dir):
        self.rec_dir, self.errors, self._lock = rec_dir, [], threading.Lock()

    def error(self, msg):
        with self._lock:
            self.errors.append(msg)

    def append(self, name, obj) -> bool:
        try:
            os.makedirs(self.rec_dir, exist_ok=True)
            with open(os.path.join(self.rec_dir, name), "a", encoding="utf-8") as fh:
                fh.write(canonical(obj) + "\n")
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.error("append %s: %s" % (name, exc))
            return False


def parse_latencies(stdout: bytes, truncated: bool = False, stream_cutoff: bool = False,
                    read_failed: bool = False) -> dict:
    """Usable only if every GEAK_RESULT_LATENCY_MS line has a finite value > 0 and a distinct,
    non-empty case id, and the whole output was seen: neither cut by the parse-memory cap
    (`truncated`), by the drain deadline (`stream_cutoff`) nor by a failed read (`read_failed`).
    Anything else is `partial` (the bad lines are kept as raw text) or `none`."""
    cases, invalid, seen, dup = [], [], set(), []
    for line in stdout.splitlines():
        m = LATENCY_RE.search(line)
        if not m:
            continue
        raw_v, case = m.group(1).decode(errors="replace"), m.group(2).decode(errors="replace").strip()
        try:
            ms = float(raw_v)
        except ValueError:
            ms = None
        if ms is None or not math.isfinite(ms) or ms <= 0 or not case:
            invalid.append({"raw": line.decode(errors="replace")[:200],
                            "why": "value" if (ms is None or not math.isfinite(ms) or ms <= 0) else "no case id"})
            continue
        if case in seen:
            dup.append(case)
        seen.add(case)
        cases.append({"case": case, "latency_ms": ms})
    status = "none" if not cases and not invalid else (
        "usable" if cases and not invalid and not dup and not truncated and not stream_cutoff
        and not read_failed else "partial")
    if status == "none" and (stream_cutoff or read_failed):
        status = "partial"                          # nothing parsed in the prefix; the rest was never seen
    return {"status": status, "parsed": status == "usable", "format": "GEAK_RESULT_LATENCY_MS.v1",
            "cases": cases, "invalid": invalid, "duplicate_cases": sorted(set(dup)),
            "stdout_truncated_for_parse": truncated, "stream_cutoff": stream_cutoff,
            "stream_read_failed": read_failed}


def _write_all(raw, chunk, raw_path, rec, problems) -> bool:
    """Write every byte of chunk to the unbuffered raw file. An unbuffered write may store fewer
    bytes than asked without raising (e.g. at a file-size limit): keep writing the remainder, and
    treat an error or a write that stores nothing as a failed raw capture."""
    view = memoryview(chunk)
    while view:
        try:
            n = raw.write(view)
        except OSError as exc:
            rec.error("raw write %s: %s" % (raw_path, exc))
            problems.append("raw_write_failed")
            return False
        if not n:
            rec.error("raw write %s: no progress (%d bytes unwritten)" % (raw_path, len(view)))
            problems.append("raw_write_short")
            return False
        view = view[n:]
    return True


def _tee(fd, sink, raw_path, keep, rec, stop, problems):
    """Copy the child's stream to our own stream and the raw file. Every recorder-side failure is
    recorded, never raised: an uncaught thread exception would print into our stderr. `problems`
    collects what makes this stream's capture incomplete (raw_* = the raw file only, including a
    short write; read_failed = the stream itself was not fully read)."""
    raw = None
    try:
        raw = open(raw_path, "wb", buffering=0)        # unbuffered: a full disk fails at write, not close
    except OSError as exc:
        rec.error("raw open %s: %s" % (raw_path, exc))
        problems.append("raw_open_failed")
    kept, truncated = 0, False
    try:
        while not stop.is_set():                            # checked every pass, data or not
            try:
                ready, _, _ = select.select([fd], [], [], 0.2)
            except (OSError, ValueError) as exc:
                rec.error("select: %s" % exc)
                problems.append("read_failed")
                break
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError as exc:
                rec.error("read: %s" % exc)
                problems.append("read_failed")
                break
            if not chunk:
                break
            try:
                sink.write(chunk)
                sink.flush()
            except (OSError, ValueError):
                pass                                        # our reader went away; keep draining the child
            if raw is not None and not _write_all(raw, chunk, raw_path, rec, problems):
                try:
                    raw.close()
                except OSError:
                    pass
                raw = None                                  # raw record stops here; the stream goes on
            if keep is not None:
                room = max(0, STDOUT_PARSE_CAP - kept)
                if room:
                    keep.append(chunk[:room])
                    kept += min(room, len(chunk))
                if len(chunk) > room:
                    truncated = True                        # raw file still has every byte
    except Exception as exc:  # noqa: BLE001 - last resort: record, never print
        rec.error("tee: %s: %s" % (type(exc).__name__, exc))
        problems.append("read_failed")
    finally:
        if raw is not None:
            try:
                raw.close()
            except OSError as exc:
                rec.error("raw close %s: %s" % (raw_path, exc))
                problems.append("raw_close_failed")
        if keep is not None and truncated:
            keep.append(None)                               # marker: parse input was truncated


def cmd_run(a) -> int:
    rec = Recorder(a.rec_dir)
    inv = uuid.uuid4().hex
    engineer = a.engineer_id or os.environ.get("GEAK_ENGINEER_ID") or None
    if engineer is not None and not ID_RE.match(engineer):
        rec.error("engineer id rejected (charset/length)")
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
    rec.append("invocations.jsonl", {"schema": SCHEMA, "event": "start", "inv": inv, "engineer_id": engineer,
                                     "t": time.time(), "workspace": os.path.abspath(a.workspace), "cmd": a.cmd,
                                     "measurement": measurement, "source_before": before})
    raw_dir = os.path.join(a.rec_dir, "raw")
    try:
        os.makedirs(raw_dir, exist_ok=True)
    except OSError as exc:
        rec.error("raw dir: %s" % exc)
    t0 = time.monotonic()
    # Own process group so a forwarded signal reaches the whole managed tree. process_group= only
    # exists from Python 3.11 and GEAK supports 3.8+, so older interpreters use os.setpgrp in the
    # child (no other thread exists yet at this point). If even that is refused, run the child
    # without a group rather than not at all, and record why.
    group = {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}
    grouped = True
    try:
        try:
            proc = subprocess.Popen(a.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_env, **group)
        except (TypeError, ValueError, subprocess.SubprocessError) as exc:
            rec.error("process group unavailable, child runs ungrouped: %s: %s" % (type(exc).__name__, exc))
            grouped = False
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
            if grouped:
                os.killpg(proc.pid, signum)                 # the whole managed tree, as a terminal would
            else:
                proc.send_signal(signum)
        except OSError:
            pass
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    stop, out_keep = threading.Event(), []
    # Daemon threads: a thread stuck writing to our own (unread) stdout must not keep us alive.
    problems = {"stdout": [], "stderr": []}
    threads = [threading.Thread(target=_tee, daemon=True, args=(proc.stdout.fileno(), sys.stdout.buffer,
                                                  os.path.join(raw_dir, inv + ".stdout"), out_keep, rec, stop,
                                                  problems["stdout"])),
               threading.Thread(target=_tee, daemon=True, args=(proc.stderr.fileno(), sys.stderr.buffer,
                                                  os.path.join(raw_dir, inv + ".stderr"), None, rec, stop,
                                                  problems["stderr"]))]
    for t in threads:
        t.start()
    rc = proc.wait()
    wall = time.monotonic() - t0
    deadline = time.monotonic() + DRAIN_GRACE_S
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    still = {"stdout": threads[0].is_alive(), "stderr": threads[1].is_alive()}
    held = any(still.values())                    # descendants still hold or feed the streams
    for name, alive in still.items():
        if alive:
            problems[name].append("drain_cutoff")
    stop.set()
    for t in threads:
        t.join(2.0)                                 # each pass re-checks `stop` within ~0.2 s
    stuck = any(t.is_alive() for t in threads)
    if stuck:
        rec.error("a stream copier did not stop within 2 s (blocked writing to our own output)")
    sig = -rc if rc < 0 else None
    completion = {"schema": SCHEMA, "event": "completion", "inv": inv, "t": time.time(),
                  "child": {"exit": None if sig else rc, "signal": sig, "forwarded_signals": forwarded,
                            "pgid": proc.pid, "wall_s": round(wall, 3),
                            "descendants_held_streams": held, "grouped": grouped}}
    # Capture status per stream, apart from the child's outcome: a drain cutoff, an unread stream or
    # a failed raw-file open/write/close each make that stream's raw record incomplete.
    capture = {n: {"status": "incomplete" if p else "complete", "problems": sorted(set(p))}
               for n, p in problems.items()}
    completion["raw_capture"] = capture
    completion["raw_streams"] = "complete" if all(c["status"] == "complete" for c in capture.values()) else "incomplete"
    try:
        after = fingerprint(a.workspace, a.baseline, domain, derr, a.rec_dir)
        completion["source_after"] = after
        b, f = before.get("identity"), after.get("identity")
        completion["source_status"] = ("incomplete" if not (before.get("complete") and after.get("complete"))
                                       else "no_net_change" if b == f else "source_changed_during_command")
    except Exception as exc:  # noqa: BLE001
        rec.error("fingerprint after: %s" % exc)
        completion["source_status"] = "incomplete"
    truncated = bool(out_keep) and out_keep[-1] is None
    data = b"".join(c for c in out_keep if c is not None)
    # The parse sees what was READ: a cutoff or read failure on stdout means it saw a prefix. A
    # failed raw-file write alone does not: the in-memory copy is still the whole stream.
    out_problems = set(problems["stdout"])
    completion["measurement"] = (parse_latencies(data, truncated, stream_cutoff="drain_cutoff" in out_problems,
                                                 read_failed="read_failed" in out_problems)
                                 if a.mode in ("benchmark", "full_benchmark")
                                 else {"status": "not_parsed", "parsed": False, "cases": []})
    # A caller-labelled correctness command's exit status, bound to this source and the named
    # identity files. Evidence about that command, not a standalone correctness-pass claim.
    completion["correctness"] = ({"evidence": "correctness_mode_command_exit", "exit": None if sig else rc,
                                  "signal": sig, "source_identity": before.get("identity"),
                                  "source_complete": bool(before.get("complete")), "identity_files": idfiles}
                                 if a.mode == "correctness" else {"evidence": "none"})
    completion["recorder_error"] = rec.errors or None
    if not rec.append("invocations.jsonl", completion):
        rec.append("recorder_errors.jsonl", {"inv": inv, "errors": rec.errors})
    if sig:
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        return 128 + sig
    return rc


# --------------------------------------------------------------------------- reading records
def _read_jsonl(path):
    """Valid records plus a count of torn/malformed lines (kept as a fact, never repaired)."""
    good, bad = [], 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(obj, dict):
                    good.append(obj)
                else:
                    bad += 1
    except OSError:
        pass
    return good, bad


def _starts(rec_dir, engineer):
    events, _ = _read_jsonl(os.path.join(rec_dir, "invocations.jsonl"))
    return [e for e in events if e.get("event") == "start" and e.get("engineer_id") == engineer
            and isinstance(e.get("inv"), str)]


# --------------------------------------------------------------------------- declarations
def cmd_declare(a) -> int:
    """Agent-reported events, stored as declarations. `next` is a commitment: first one wins.
    A declaration attaches only to an invocation of the SAME engineer."""
    def reject(why):
        print(canonical({"status": "rejected", "error": why}))
        return 0
    if not ID_RE.match(a.engineer_id or ""):
        return reject("bad engineer id")
    if a.kind not in DECLARE_KINDS or (a.kind == "next" and (a.value not in NEXT_CHOICES or a.seq is None)):
        return reject("next needs --seq and a value in %s" % (NEXT_CHOICES,))
    mine = [s["inv"] for s in _starts(a.rec_dir, a.engineer_id)]
    if a.inv is not None and a.inv not in mine:
        return reject("invocation %s is not one of this engineer's" % a.inv)
    rec = Recorder(a.rec_dir)
    did = uuid.uuid4().hex
    d = {"schema": SCHEMA, "decl": did, "engineer_id": a.engineer_id, "kind": a.kind, "value": a.value,
         "seq": a.seq, "t": time.time(), "inv": a.inv if a.inv is not None else (mine[-1] if mine else None),
         "inv_source": "explicit" if a.inv is not None else ("latest_of_engineer" if mine else "none"),
         "reported_by": "agent"}
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
            rec.error(str(exc))
    if not rec.append("declarations.jsonl", d) and status == "recorded":
        status = "write_failed"
    print(canonical({"status": status, "decl": did}))
    return 0


# --------------------------------------------------------------------------- summary (observed vs declared)
def summarize(rec_dir, gpu_use_log=None) -> dict:
    events, bad_inv = _read_jsonl(os.path.join(rec_dir, "invocations.jsonl"))
    decls, bad_decl = _read_jsonl(os.path.join(rec_dir, "declarations.jsonl"))
    starts = [e for e in events if e.get("event") == "start" and isinstance(e.get("inv"), str)]
    done = {e.get("inv") for e in events if e.get("event") == "completion"}
    per = {}
    for s in starts:
        per.setdefault(s.get("engineer_id") or "unattributed", []).append(s)
    engineers = {}
    for eng, ss in per.items():
        order = [s["inv"] for s in ss]
        src = {s["inv"]: (s.get("source_before") or {}).get("identity")
               if (s.get("source_before") or {}).get("complete") else None for s in ss}
        joins = []
        for d in (x for x in decls if x.get("engineer_id") == eng):
            j = {"decl": d.get("decl"), "kind": d.get("kind"), "value": d.get("value"), "consistency": "unknown"}
            if d.get("kind") in ("keep", "revert") and d.get("inv") in order:
                i = order.index(d["inv"])
                cand, nxt = src.get(order[i]), (src.get(order[i + 1]) if i + 1 < len(order) else None)
                prev = next((src[x] for x in reversed(order[:i]) if src.get(x) and src[x] != cand), None)
                if cand and nxt and prev:
                    if d["kind"] == "revert":
                        j["consistency"] = "consistent" if nxt == prev else "inconsistent" if nxt == cand else "unknown"
                    else:
                        j["consistency"] = "inconsistent" if nxt == prev else "unknown"
            joins.append(j)
        engineers[eng] = {"invocations": len(order), "completed": sum(1 for i in order if i in done),
                          "incomplete_cause_unknown": sum(1 for i in order if i not in done),
                          "distinct_complete_sources": len({v for v in src.values() if v}),
                          "declarations": joins}
    out = {"engineers": engineers, "malformed_records": {"invocations": bad_inv, "declarations": bad_decl},
           "note": "actions are never inferred from source hashes; consistency is evidence, not proof"}
    if gpu_use_log:
        lines, bad = _read_jsonl(gpu_use_log)
        ours = {s["inv"] for s in starts}
        matched = {}
        for l in lines:
            if l.get("recorder_inv_id") in ours:
                matched.setdefault(l["recorder_inv_id"], []).append(l.get("gpu"))
        out["gpu_lock"] = {
            "recorder_invocations_with_lock_line": len(matched),
            "gpu_by_invocation": matched,
            "lock_lines_with_engineer_but_no_recorder_id": sum(1 for l in lines if l.get("engineer_id")
                                                               and not l.get("recorder_inv_id")),
            "lock_lines_without_ids": sum(1 for l in lines if not l.get("engineer_id") and not l.get("recorder_inv_id")),
            "malformed_lines": bad,
            "outside_observed_population": "unknown"}
    return out


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
    d.add_argument("--inv", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--rec-dir", required=True)
    s.add_argument("--gpu-use-log")
    a = ap.parse_args(argv)
    if a.sub == "run":
        a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not a.cmd:
            ap.error("run needs a command after --")
        return cmd_run(a)
    if a.sub == "declare":
        return cmd_declare(a)
    print(json.dumps(summarize(a.rec_dir, a.gpu_use_log), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
