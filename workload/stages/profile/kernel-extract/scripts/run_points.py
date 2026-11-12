#!/usr/bin/env python3
"""Capture planned points on the supplied server; census and collect dispatch evidence.

    run_points.py --server-state probe/probe_server.json \
        --points sampling_points.json --observed evidence/observed_workload.json \
        --eval-dir <pool-directory> --all-evidence

Prefill runs in token order, decode in batch/context order. Each point gets at most
one recapture. A graph-hidden decode window requests eager fallback from
probe_server.py; a dead server stops capture. Signed completed censuses are reused
unless --redo is specified. Census work overlaps capture of the next point.

Incremental signing: while points are captured, profile_run.json is a checkpoint with
status "running" that holds every signed record (the resumed and carried-over ones, and
each point of this invocation as soon as its census is written and verified), rewritten
atomically (temporary file + os.replace). A hard kill or a preemption therefore keeps
every finished census signed: rerunning the same command resumes them, and recaptures,
under a fresh tag, only the points the killed invocation left without a signed record
(a census in flight, or a window's traces). The finished file is written exactly as
before.

Environment knobs (default off; they never change what a window captures, and they are
environment rather than flags so a frozen capture argv stays byte-identical):
  GEAK_CENSUS_SERIAL=1        wait for each point's census before opening the next window
                              (the census and the next window otherwise overlap in host RAM)
  GEAK_CENSUS_WORKERS=N       pass --workers N to point_census.py (default min(files, CPUs))
  GEAK_BOUNDARY_RESTART_GB=G  at a point boundary (previous census finished, next window not
                              yet open, at least one point captured by this invocation), stop
                              capturing when MemAvailable < G GB: status "memory restart
                              requested", exit 1; rerunning the same command resumes from the
                              signed censuses (the server restart is the caller's job)
All three are recorded in profile_run.json.

Outputs: per-point probe/census/shape evidence, coverage, ranking, selection,
source map, adjacency and profile_run.json. UT declarations are generated separately
by draft_task_decls.py after the profile handoff. Stdlib only.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# The eager evidence window's tag: trace subdirectory, probe/shapes file stem, and the
# vLLM profile_prefix. vLLM's validate_profile_prefix accepts only [A-Za-z0-9._-] (first
# character alphanumeric), so the old "@eager" tag was refused with HTTP 400.
EAGER_SUFFIX = ".eager"
LEGACY_EAGER_SUFFIXES = ("@eager",)       # sglang-era files; still recognised when read
MEMORY_RESTART = "memory restart requested"
# profile_run.json while points are captured (the incremental checkpoint); a file that
# still says so was left by an invocation that did not finish (killed or preempted).
RUNNING = "running"


def eager_tag(pid):
    return pid + EAGER_SUFFIX


def eager_shapes_names(pid):
    """Every shapes-file name an eager evidence window of `pid` may have written."""
    return [s + ".json" for s in [eager_tag(pid)] + [pid + x for x in LEGACY_EAGER_SUFFIXES]]


def is_eager_shapes_file(name):
    return any(name.endswith(s + ".json") for s in (EAGER_SUFFIX,) + LEGACY_EAGER_SUFFIXES)


def env_knobs(environ=None):
    """GEAK_CENSUS_SERIAL / GEAK_CENSUS_WORKERS / GEAK_BOUNDARY_RESTART_GB, validated."""
    env = os.environ if environ is None else environ
    serial = env.get("GEAK_CENSUS_SERIAL", "").strip()
    if serial not in ("", "0", "1"):
        raise SystemExit("run_points: GEAK_CENSUS_SERIAL must be 0 or 1, not %r" % serial)
    workers = env.get("GEAK_CENSUS_WORKERS", "").strip()
    if workers and not (workers.isdigit() and int(workers) > 0):
        raise SystemExit("run_points: GEAK_CENSUS_WORKERS must be a positive integer, "
                         "not %r" % workers)
    restart = env.get("GEAK_BOUNDARY_RESTART_GB", "").strip()
    try:
        restart_gb = float(restart) if restart else None
    except ValueError:
        restart_gb = -1.0
    if restart_gb is not None and not restart_gb > 0:
        raise SystemExit("run_points: GEAK_BOUNDARY_RESTART_GB must be a positive number, "
                         "not %r" % restart)
    return {"census_serial": serial == "1", "census_workers": int(workers) if workers else None,
            "boundary_restart_gb": restart_gb}


def mem_available_gb(path=None):
    """MemAvailable in GB (kB / 1048576, the unit the watchdogs use), or None if unreadable."""
    path = path or os.environ.get("GEAK_MEMINFO") or "/proc/meminfo"
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def now():
    return time.strftime("%H:%M:%S")


def run(cmd, log):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(log, "a") as fh:
        fh.write("[%s] $ %s\n%s%s\n" % (now(), " ".join(cmd), p.stdout, p.stderr))
    return p.returncode, round(time.time() - t0, 1), (p.stdout + p.stderr).strip()


def write_json_atomic(path, doc):
    """Write `doc` as `path` (indent=1) so that a reader or a kill sees the old or the new
    file, never a partial one."""
    tmp = "%s.tmp.%d.%d" % (path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def census_complete(c, pid, expect_files):
    """None if census result `c` of point `pid` may be signed, else why not.

    point_census.py exited 0 and was not a graph-hidden refusal; its output is a readable
    census of this point over this capture's trace directory, with at least one instance
    and every expected rank; the directory still holds that many trace files.
    """
    if c.get("rc") != 0 or not c.get("out") or c.get("graph_hidden"):
        return "census rc=%s" % c.get("rc")
    try:
        with open(c["out"]) as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        return "census unreadable: %s" % e
    if d.get("point_id") != pid:
        return "census is of %r" % d.get("point_id")
    if os.path.abspath(d.get("trace_dir") or "") != os.path.abspath(c.get("trace_dir") or ""):
        return "census trace_dir %s is not the capture's" % d.get("trace_dir")
    if not (d.get("instances") or 0) > 0:
        return "census has no instance"
    ranks = len(d.get("per_rank") or {})
    if ranks < (expect_files or 1) or (d.get("ranks") or 0) < (expect_files or 1):
        return "census has %d of %d ranks" % (ranks, expect_files)
    traces = [f for f in os.listdir(c["trace_dir"]) if ".trace.json" in f] \
        if os.path.isdir(c["trace_dir"]) else []
    if len(traces) < (expect_files or 1):
        return "trace directory holds %d of %d files" % (len(traces), expect_files)
    return None


def resume_plan(points, prev_doc, census_dir, trace_root, redo=()):
    """(resumed, recapture): the points whose census is reused, and those a killed
    invocation left unsigned.

    A finished profile_run.json keeps the old contract: a census with a missing or
    different measurement_signature is refused (SystemExit). A checkpoint left by a
    killed invocation (status "running") holds every record it signed, so a point it
    does not hold, yet has a census file (in flight at the kill, possibly partial) or a
    non-empty trace directory (a window in flight), is recaptured under a fresh tag; a
    point it holds with a different signature is still refused.
    """
    prev = (prev_doc or {}).get("points") or {}
    killed = (prev_doc or {}).get("status") == RUNNING
    resumed, recapture = [], []
    for pt in points:
        pid = pt["point_id"]
        if pid in redo:
            continue
        done = os.path.exists(os.path.join(census_dir, pid + ".json"))
        if killed and pid not in prev:
            dirs = [os.path.join(trace_root, pid + x) for x in ("", "_r2")]
            if done or any(os.path.isdir(d) and os.listdir(d) for d in dirs):
                recapture.append(pid)
            continue
        if done:
            signature = pt.get("measurement_signature")
            if signature and prev.get(pid, {}).get("measurement_signature") != signature:
                raise SystemExit("run_points: incompatible or unsigned shared census for %s; "
                                 "use --redo %s or a fresh capture pool" % (pid, pid))
            resumed.append(pid)
    return resumed, recapture


def census_graph_hidden_reason(phase, rc, message):
    """Only the census's explicit missing-GPU-anchor diagnosis requests eager retry.

    A probe may verify one rank while the all-rank census rejects another. Other
    census failures (missing files, unequal rank counts, etc.) are not evidence of
    graph-hidden kernels and must keep their original failure classification.
    """
    if phase != "DECODE" or rc == 0:
        return None
    for line in message.splitlines():
        line = line.strip()
        if (line.startswith("point_census:")
                and "fewer than 2 decode anchors on the GPU timeline" in line
                and "replayed kernels are not traced" in line):
            return line
    return None


def ordered(points, wanted):
    pts = [p for p in points if not wanted or p["point_id"] in wanted]
    missing = set(wanted or ()) - {p["point_id"] for p in pts}
    if missing:
        raise SystemExit("run_points: unknown point id(s): %s" % ", ".join(sorted(missing)))
    pre = sorted((p for p in pts if p["phase"] == "PREFILL"),
                 key=lambda p: (p.get("new_tokens") or 0, p["point_id"]))
    dec = sorted((p for p in pts if p["phase"] == "DECODE"),
                 key=lambda p: (p.get("batch_size") or 0, p.get("context_tokens") or 0,
                                p["point_id"]))
    other = [p for p in pts if p["phase"] not in ("PREFILL", "DECODE")]
    if other:
        raise SystemExit("run_points: phases other than PREFILL/DECODE are not driven here: "
                         "%s" % ", ".join(p["point_id"] for p in other))
    return pre + dec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", required=True, help="shared sampling pool sampling_points.json")
    ap.add_argument("--point-id", nargs="*", default=None, help="subset; default all")
    ap.add_argument("--server-state", default=None,
                    help="probe_server.py's probe_server.json: supplies --base-url, "
                         "--trace-root, --expect-files, --decode-eager and --vocab-size")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="the model's vocab_size: serving_probe.py keeps prompt ids below it")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--trace-root", default=None,
                    help="host-visible profiler output directory; one subdir per point")
    ap.add_argument("--server-trace-root", default=None,
                    help="the same directory as the server sees it, if it differs")
    ap.add_argument("--expect-files", type=int, default=None,
                    help="trace files one window writes (= TP ranks)")
    ap.add_argument("--decode-eager", action="store_true",
                    help="decode is graph-deployed but this server runs eagerly: decode "
                         "collective wait is excluded from the ranking denominator")
    ap.add_argument("--decode-on-graph", action="store_true",
                    help="try DECODE points on this graph-mode server too (old behaviour). "
                         "Default: when --server-state says the server keeps the decode CUDA "
                         "graph, it captures PREFILL only and every DECODE point is deferred "
                         "to the eager server without a capture")
    ap.add_argument("--draft-shares-name", action="store_true",
                    help="passed to point_census.py for PREFILL points")
    ap.add_argument("--observed", default=None,
                    help="observed_workload.json: relabel points to their achieved shapes "
                         "(relabel_points.py) before the ledger")
    ap.add_argument("--redo", nargs="*", default=[],
                    help="recapture these points even if they already have a census")
    ap.add_argument("--evidence-capture", action="store_true",
                    help="the server is eager now: capture the decode point named in "
                         "evidence/eager_evidence_request.json for launch evidence only "
                         "(frames, launching ops), then redo the evidence")
    ap.add_argument("--no-evidence", action="store_true",
                    help="stop after selection: no shapes, source map or adjacency")
    ap.add_argument("--all-evidence", action="store_true",
                    help="shared capture pool: resolve every censused kernel before per-config selection")
    ap.add_argument("--probe-arg", action="append", default=[],
                    help="extra serving_probe.py argument, e.g. --probe-arg=--decode-tokens=20000")
    ap.add_argument("--select-arg", action="append", default=[],
                    help="extra select_targets.py argument (both selections)")
    ap.add_argument("--select-by-kernel", action="store_true",
                    help="keep the selection by kernel name: no candidate stacks, no "
                         "operator aggregation")
    ap.add_argument("--eval-dir", required=True)
    a = ap.parse_args()
    st = {}
    if a.server_state:
        st = json.load(open(a.server_state))
        a.base_url = a.base_url or st["base_url"]
        a.trace_root = a.trace_root or st["trace_root"]
        a.expect_files = a.expect_files or st["expect_files"]
        a.decode_eager = a.decode_eager or bool(st.get("decode_eager"))
        a.vocab_size = a.vocab_size or st.get("vocab_size")
        if st.get("framework") == "vllm":
            a.probe_arg += ["--framework", "vllm", "--served-model", st["model"]]
            if st.get("iteration_log_dir"):
                a.probe_arg += ["--iteration-log-dir", st["iteration_log_dir"]]
        server_capture = st.get("decode_capture")
    else:
        server_capture = None
    if not (a.base_url and a.trace_root and a.expect_files):
        ap.error("--base-url, --trace-root and --expect-files (or --server-state) are required")
    if a.vocab_size:
        a.probe_arg = ["--vocab-size", str(a.vocab_size)] + a.probe_arg
    else:
        print("run_points: vocab size unknown; serving_probe.py keeps prompt ids small",
              flush=True)

    ev = os.path.abspath(a.eval_dir)
    probe_dir = os.path.join(ev, "evidence", "probe")
    census_dir = os.path.join(ev, "evidence", "census")
    for d in (probe_dir, census_dir, a.trace_root):
        os.makedirs(d, exist_ok=True)
    log = os.path.join(ev, "run_points.log")
    plan = json.load(open(a.points))
    points = ordered(plan["points"], a.point_id)
    py = sys.executable
    run_path = os.path.join(ev, "profile_run.json")
    prev_doc = {}
    try:
        prev_doc = json.load(open(run_path))
    except (OSError, ValueError):
        pass
    if not isinstance(prev_doc, dict):
        prev_doc = {}
    prev = prev_doc.get("points") or {}
    knobs = env_knobs()
    timeline = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "points": {},
                "server": a.base_url, "decode_eager": a.decode_eager}
    timeline.update(knobs)
    t_all = time.time()
    # Decode on a graph-mode server: under spec decoding the profiler drops most VERIFY
    # replays (sglang 0.5.21/ROCm 7.2: 1 of 8 kept on most ranks), so every decode point
    # was refused as graph-hidden and recaptured eagerly anyway -- after a failed attempt
    # that cost up to the window timeout. Defer them to the eager server up front.
    # Only when the server state says the decode graph is on: without --server-state the
    # server's mode is unknown, and an "as_declared" deployment may not be graph-launched.
    defer_decode = (st.get("framework", "sglang") == "sglang" and server_capture == "profile_cuda_graph" and not a.decode_eager
                    and not a.decode_on_graph)
    timeline["decode_deferred_to_eager"] = defer_decode
    deferred = []
    # Before any capture: refuse an unsigned or incompatible census up front (the loop
    # used to find it only when it got there), and take the points a killed invocation
    # left unsigned as redo points.
    _resumed, recapture = resume_plan(points, prev_doc, census_dir, a.trace_root, a.redo)
    redo = list(a.redo) + recapture
    if recapture:
        timeline["recaptured_after_kill"] = recapture
        print("[%s] profile_run.json is the checkpoint of an invocation that did not finish; "
              "recapturing %d point(s) it left unsigned: %s"
              % (now(), len(recapture), ", ".join(recapture)), flush=True)

    def census_job(pt, trace_dir):
        out = os.path.join(census_dir, pt["point_id"] + ".json")
        cmd = [py, os.path.join(HERE, "point_census.py"), "--trace-dir", trace_dir,
               "--phase", pt["phase"], "--point-id", pt["point_id"], "--out", out]
        planned = pt.get("new_tokens") if pt["phase"] == "PREFILL" else pt.get("batch_size")
        if planned:
            cmd += ["--planned-shape", str(planned)]
        if pt["phase"] == "PREFILL" and a.draft_shares_name:
            cmd.append("--draft-shares-name")
        if knobs["census_workers"]:
            cmd += ["--workers", str(knobs["census_workers"])]
        rc, took, msg = run(cmd, log)
        result = {"rc": rc, "wall_s": took, "out": out if rc == 0 else None,
                  "trace_dir": trace_dir, "message": msg.splitlines()[-1] if msg else ""}
        hidden_reason = census_graph_hidden_reason(pt["phase"], rc, msg)
        if hidden_reason:
            result.update(graph_hidden=True, graph_hidden_reason=hidden_reason)
        return result

    if a.evidence_capture:
        eager_evidence(a, ev, plan, py, log, timeline, knobs["census_workers"])

    # The checkpoint: every record signed so far, under ck_lock (census callbacks run on
    # the census thread). Earlier signed records are carried over as the resume path
    # writes them; a redo point's old record is not (its census file is about to change).
    ck_lock = threading.Lock()
    ck = {"open": True, "points": {
        pid: dict(r, resumed=True) for pid, r in prev.items()
        if pid not in redo and isinstance(r, dict)
        and os.path.exists(os.path.join(census_dir, pid + ".json"))}}
    ck_header = {k: v for k, v in timeline.items() if k != "points"}

    def checkpoint():
        # caller holds ck_lock
        if not ck["open"]:
            return
        doc = dict(ck_header, status=RUNNING, points=ck["points"],
                   signed_points=len(ck["points"]))
        try:
            write_json_atomic(run_path, doc)
        except OSError as e:
            print("[%s] run_points: checkpoint not written: %s" % (now(), e), flush=True)

    def sign(pid, record, fut):
        """Add `pid`'s record, with its census, to the checkpoint once that census is
        complete; an incomplete one stays unsigned (a rerun recaptures it)."""
        try:
            c = fut.result()
            why = census_complete(c, pid, a.expect_files)
        except Exception as e:          # noqa: BLE001 -- never break the census thread
            c, why = None, repr(e)
        with ck_lock:
            if why:
                if ck["open"]:
                    print("[%s] %s not signed: %s" % (now(), pid, why), flush=True)
                return
            ck["points"][pid] = dict(record, census=c)
            checkpoint()

    with ck_lock:
        checkpoint()
    futures, censuses = {}, {}
    down = mem_stop = False
    captured_here = 0
    with ThreadPoolExecutor(max_workers=1) as census_pool:
        for pt in points:
            if down:
                break
            pid = pt["point_id"]
            done = os.path.join(census_dir, pid + ".json")
            if os.path.exists(done) and pid not in redo:
                signature = pt.get("measurement_signature")
                if signature and prev.get(pid, {}).get("measurement_signature") != signature:
                    raise SystemExit("run_points: incompatible or unsigned shared census for %s; "
                                     "use --redo %s or a fresh capture pool" % (pid, pid))
                # Resume: a censused point is evidence already paid for.
                timeline["points"][pid] = dict(prev.get(pid, {"phase": pt["phase"]}),
                                               resumed=True)
                censuses[pid] = done
                continue
            rec = timeline["points"].setdefault(pid, {"phase": pt["phase"], "attempts": [],
                                                     "measurement_signature": pt.get("measurement_signature")})
            if defer_decode and pt["phase"] == "DECODE":
                # graph_hidden is the contract probe_server.py (and any group driver) reads
                # to restart eagerly and --redo exactly these points.
                rec["attempts"].append({"tag": pid, "status": "deferred", "graph_hidden": True,
                                        "reason": "decode is captured on the eager server "
                                                  "(--decode-on-graph tries it here)",
                                        "probe_wall_s": 0})
                deferred.append(pid)
                continue
            if knobs["boundary_restart_gb"] and captured_here:
                # A point boundary: no window open. Measure after the previous census, so
                # its transient memory is not mistaken for the server's residual growth.
                for f in futures.values():
                    f.result()
                gb = mem_available_gb()
                if gb is not None and gb < knobs["boundary_restart_gb"]:
                    timeline["boundary_stop"] = {
                        "before_point": pid, "mem_available_gb": round(gb, 1),
                        "threshold_gb": knobs["boundary_restart_gb"],
                        "captured_this_run": captured_here,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    print("[%s] MemAvailable %.0f GB < %.0f GB at the boundary before %s: "
                          "stopping for a server restart" % (now(), gb,
                          knobs["boundary_restart_gb"], pid), flush=True)
                    del timeline["points"][pid]
                    mem_stop = True
                    break
            stamp = time.strftime("%H%M%S")
            for attempt in ("", "_r2"):
                tag = "%s%s_%s" % (pid, attempt, stamp) if pid in redo else pid + attempt
                trace_dir = os.path.join(a.trace_root, tag)
                server_dir = os.path.join(a.server_trace_root or a.trace_root, tag)
                os.makedirs(trace_dir, exist_ok=True)
                out = os.path.join(probe_dir, pid + attempt + ".json")
                cmd = [py, os.path.join(HERE, "serving_probe.py"), "--points", a.points,
                       "--point-id", pid, "--base-url", a.base_url,
                       "--trace-dir", trace_dir, "--server-trace-dir", server_dir,
                       "--expect-files", str(a.expect_files), "--out", out] + a.probe_arg
                print("[%s] %s%s %s ..." % (now(), pid, attempt, pt["phase"]), flush=True)
                rc, took, msg = run(cmd, log)
                try:
                    res = json.load(open(out))
                except (OSError, ValueError):
                    res = {"status": "failed", "verify_reason": msg[-300:]}
                hidden = any(e.get("graph_hidden") for e in res.get("journal") or []
                             if e.get("event") == "verify")
                rec["attempts"].append({"tag": tag, "status": res.get("status"),
                                        "reason": res.get("verify_reason"),
                                        "graph_hidden": hidden,
                                        "probe_wall_s": took, "result": out,
                                        "trace_dir": trace_dir})
                print("[%s]   %s in %.0fs: %s" % (now(), res.get("status"), took,
                                                  res.get("verify_reason")), flush=True)
                if res.get("status") in ("ok", "partial"):
                    rec["capture_mode"] = ("eager" if a.decode_eager else "profile_cuda_graph") \
                        if pt["phase"] == "DECODE" else "eager_prefill"
                    record = json.loads(json.dumps(rec))     # as finish() writes it
                    futures[pid] = census_pool.submit(census_job, pt, trace_dir)
                    captured_here += 1
                    if knobs["census_serial"]:
                        t_wait = time.time()
                        futures[pid].result()
                        rec["census_serial_wait_s"] = round(time.time() - t_wait, 1)
                        sign(pid, dict(record, census_serial_wait_s=rec["census_serial_wait_s"]),
                             futures[pid])
                    else:
                        # signed on the census thread the moment the census completes
                        futures[pid].add_done_callback(
                            lambda f, pid=pid, record=record: sign(pid, record, f))
                    break
                if res.get("server_down"):
                    # Every later point would fail the same way; a rerun resumes from here.
                    print("[%s]   the server is down; stopping the captures (its log says "
                          "why)" % now(), flush=True)
                    down = True
                    break
                if hidden:
                    # Recapturing on the same graph-mode server would fail the same way.
                    print("[%s]   graph-hidden decode; needs an eager server "
                          "(probe_server.py run does that itself)" % now(), flush=True)
                    break
        print("[%s] captures done in %.0fs; waiting for %d census job(s)"
              % (now(), time.time() - t_all, sum(not f.done() for f in futures.values())),
              flush=True)
        for pid, fut in futures.items():
            c = fut.result()
            rec = timeline["points"][pid]
            if c.get("graph_hidden"):
                # probe_server.graph_hidden_points treats a `census` entry as a
                # completed census. Retain the rejected census separately so the
                # existing fallback sees this point as still needing a capture.
                rec["census_failure"] = c
                attempt = next(a for a in reversed(rec["attempts"])
                               if a["trace_dir"] == c["trace_dir"])
                attempt.update(graph_hidden=True, graph_hidden_source="point_census",
                               graph_hidden_reason=c["graph_hidden_reason"])
            else:
                rec["census"] = c
            if c["out"]:
                censuses[pid] = c["out"]
            else:
                print("[%s] census %s FAILED: %s" % (now(), pid, c["message"]), flush=True)
    timeline["capture_and_census_s"] = round(time.time() - t_all, 1)

    def finish(status):
        timeline["status"] = status
        timeline["total_s"] = round(time.time() - t_all, 1)
        timeline["points_without_census"] = sorted(set(p["point_id"] for p in points)
                                                   - set(censuses))
        with ck_lock:
            ck["open"] = False              # no checkpoint may overwrite the final file
            write_json_atomic(run_path, timeline)

    if mem_stop or down:
        # The loop stopped early. Points after the stop that an earlier invocation already
        # censused were never visited; keep their records (as the resume path would have
        # written them), or the next invocation's signature check and the multi-config
        # projection find a census without a measurement_signature and refuse it.
        for pt in points:
            pid = pt["point_id"]
            done = os.path.join(census_dir, pid + ".json")
            if (pid not in timeline["points"] and pid not in redo and pid in prev
                    and os.path.exists(done)):
                timeline["points"][pid] = dict(prev[pid], resumed=True)
                censuses[pid] = done
    if mem_stop:
        finish(MEMORY_RESTART)
        print("run_points: %s after %d point(s) captured here (%d/%d censused); rerun the "
              "same command on a fresh server -- censused points are kept"
              % (MEMORY_RESTART, captured_here, len(censuses), len(points)), flush=True)
        return 1
    if down:
        # A ranking over the points captured before the crash would misweight the phases.
        finish("server down")
        print("run_points: the server went down after %d/%d point(s) censused; fix the cause "
              "and rerun -- censused points are kept" % (len(censuses), len(points)))
        return 1
    if deferred:
        # A ranking without the decode phase would be thrown away by the eager pass.
        finish("decode deferred to eager")
        print("[%s] %d decode point(s) deferred to the eager server: %s; %d/%d censused "
              "here, no ranking yet" % (now(), len(deferred), ", ".join(deferred),
                                        len(censuses), len(points)), flush=True)
        return 1
    if not censuses:
        finish("no point produced a census")
        print("run_points: no point produced a census; nothing to rank")
        return 1

    points_file = a.points
    steps = []
    if a.observed:
        points_file = os.path.join(ev, "sampling_points_final.json")
        steps.append(("relabel_points", [py, os.path.join(HERE, "relabel_points.py"),
                                         "--observed", a.observed, "--points", a.points,
                                         "--census-dir", census_dir, "--probe-dir", probe_dir,
                                         "--out", points_file]))
    cov = os.path.join(ev, "coverage.json")
    cmd = [py, os.path.join(HERE, "coverage_ledger.py"), "--points", points_file, "--out", cov]
    for pid, path in censuses.items():
        cmd += ["--census", "%s=%s" % (pid, path)]
    steps.append(("coverage_ledger", cmd))
    rank = os.path.join(ev, "kernel_ranking.json")
    cmd = [py, os.path.join(HERE, "build_ranking.py"), "--coverage", cov, "--out", rank,
           "--probe-result", os.path.join(probe_dir, "*.json")]
    for pid in sorted(censuses):
        if timeline["points"].get(pid, {}).get("capture_mode") == "eager":
            cmd += ["--eager-point", pid]
    steps.append(("build_ranking", cmd))
    sel = os.path.join(ev, "selection_targets.json")
    steps.append(("select_targets", [py, os.path.join(HERE, "select_targets.py"), rank,
                                     "--output", sel] + a.select_arg))
    timeline["post"] = {}

    def run_steps(steps):
        for name, cmd in steps:
            rc, took, msg = run(cmd, log)
            timeline["post"][name] = {"rc": rc, "wall_s": took}
            print("[%s] %s rc=%d (%.0fs)\n%s" % (now(), name, rc, took,
                                                 "\n".join(msg.splitlines()[-12:])), flush=True)
            if rc:
                return False
        return True

    ok = run_steps(steps)
    if ok and not a.no_evidence:
        ok = evidence(a, ev, censuses, points, sel, py, log, timeline, run_steps)
    status = "ok" if ok and len(censuses) == len(points) else "incomplete"
    if a.evidence_capture:
        ee = timeline.get("eager_evidence") or {}
        missing = timeline.get("eager_evidence_needed")
        if ee.get("status") != "ok" or missing:
            # This invocation WAS the eager evidence pass; nothing retries it. Graph-replayed
            # decode kernels keep no launching op and no source frame, so say so loudly.
            note = ("EAGER EVIDENCE FAILED: the eager window for %s %s; %d graph-replayed "
                    "decode kernel(s) have no launch evidence (see evidence/"
                    "eager_evidence_request.json and run_points.log)"
                    % (ee.get("point"), ee.get("reason") or "did not cover every kernel",
                       len(missing or [])))
            timeline.setdefault("notes", []).append(note)
            print("[%s] %s" % (now(), note), flush=True)
            if status == "ok":
                status = "eager evidence failed"
    finish(status)
    print("[%s] %s in %.0fs; %d/%d points censused%s"
          % (now(), timeline["status"], timeline["total_s"], len(censuses), len(points),
             (" -- missing: " + ", ".join(timeline["points_without_census"]))
             if timeline["points_without_census"] else ""))
    return 0 if timeline["status"] == "ok" else 1


def eager_evidence(a, ev, plan, py, log, timeline, census_workers=None):
    """One decode window on the eager server, for frames and launching ops only. Its
    census lives apart (evidence/census_eager) and never enters the ledger: under eager
    launch every rank spins in the all-reduce, so its costs are not the deployment's."""
    req = json.load(open(os.path.join(ev, "evidence", "eager_evidence_request.json")))
    pid = req["point"]
    tag = eager_tag(pid)
    pt = next(p for p in plan["points"] if p["point_id"] == pid)
    trace_dir = os.path.join(a.trace_root, tag)
    os.makedirs(trace_dir, exist_ok=True)
    out = os.path.join(ev, "evidence", "probe", tag + ".json")
    rc, took, msg = run([py, os.path.join(HERE, "serving_probe.py"), "--points", a.points,
                         "--point-id", pid, "--base-url", a.base_url, "--trace-dir", trace_dir,
                         "--server-trace-dir", os.path.join(a.server_trace_root or a.trace_root, tag),
                         "--expect-files", str(a.expect_files), "--out", out] + a.probe_arg, log)
    try:
        probe_status = json.load(open(out)).get("status")
    except (OSError, ValueError):
        probe_status = None
    # serving_probe exits 1 on "partial" too (shape off plan); launch evidence only needs
    # the window's ops and frames, exactly as a partial capture is accepted for a point.
    reason = None
    if probe_status not in ("ok", "partial"):
        reason = "probe %s (rc=%d): %s" % (probe_status or "wrote no result", rc,
                                           msg.splitlines()[-1] if msg else "")
        rc = rc or 1
    else:
        rc = 0
    cdir = os.path.join(ev, "evidence", "census_eager")
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, pid + ".json")
    steps = [[py, os.path.join(HERE, "point_census.py"), "--trace-dir", trace_dir, "--phase",
              "DECODE", "--point-id", tag, "--planned-shape", str(pt.get("batch_size")),
              "--out", cpath] + (["--workers", str(census_workers)] if census_workers else []),
             [py, os.path.join(HERE, "kernel_shapes.py"), "--trace-dir", trace_dir, "--phase",
              "DECODE", "--census", cpath, "--point-id", tag, "--out",
              os.path.join(ev, "evidence", "shapes", tag + ".json")]
             + [x for k in req["kernels"] for x in ("--kernel", k)]]
    for cmd in steps if rc == 0 else []:
        rc, t2, msg = run(cmd, log)
        took += t2
        if rc:
            reason = "%s rc=%d: %s" % (os.path.basename(cmd[1]), rc,
                                       msg.splitlines()[-1] if msg else "")
            break
    timeline["eager_evidence"] = {"point": pid, "tag": tag, "rc": rc, "wall_s": took,
                                  "status": "ok" if rc == 0 else "failed",
                                  "probe_status": probe_status, "reason": reason,
                                  "message": msg.splitlines()[-1] if msg else ""}
    print("[%s] eager evidence window %s: %s rc=%d in %.0fs%s"
          % (now(), tag, "ok" if rc == 0 else "FAILED", rc, took,
             " -- " + reason if reason else ""), flush=True)


def same_targets(shapes_file, targets_file):
    """A manifest was made for exactly these kernels (it lists what it was asked for)."""
    try:
        d = json.load(open(shapes_file))
        want = {r.get("id") or r.get("name") for r in json.load(open(targets_file))["targets"]}
    except (OSError, ValueError, KeyError):
        return False
    asked = d.get("requested")
    if asked is None:
        asked = set(d.get("kernels", {})) | set(d.get("not_launched_here", []))
    return set(asked) == want


def evidence(a, ev, censuses, points, sel, py, log, timeline, run_steps):
    """After the ranking: launch stacks for the attribution candidates, selection by
    operator over them, then collect shape/source and companion-launch evidence."""
    phase = {p["point_id"]: p["phase"] for p in points}
    rank = os.path.join(ev, "kernel_ranking.json")
    cand = sel
    if not a.select_by_kernel:
        cand = os.path.join(ev, "attribution_candidates.json")
        if not run_steps([("attribution_candidates",
                           [py, os.path.join(HERE, "select_targets.py"), rank, "--candidates",
                            "--output", cand] + [x for x in a.select_arg
                                                  if x.split("=")[0] in ("--phases",)])]):
            return False
    if getattr(a, "all_evidence", False):
        # A kernel small in the pooled log can dominate one config. Selection happens
        # after projection, so discovery here must not discard its source evidence.
        from coverage_ledger import load_census
        names = set()
        for pid, path in censuses.items():
            kernels, _, _ = load_census(path, phase[pid])
            names.update(k for k in kernels if k not in ("__collective_wait__", "__untracked_tail__"))
        cand = os.path.join(ev, "attribution_candidates.shared.json")
        json.dump({"scope": "shared_pool_all_censused_kernels",
                   "targets": [{"id": k, "name": k} for k in sorted(names)]}, open(cand, "w"), indent=1)
    shapes_dir = os.path.join(ev, "evidence", "shapes")
    os.makedirs(shapes_dir, exist_ok=True)
    trace_dirs = {}
    for pid, path in censuses.items():
        trace_dirs[pid] = json.load(open(path))["trace_dir"]

    def shape_job(pid):
        out = os.path.join(shapes_dir, pid + ".json")
        if (os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(censuses[pid])
                and same_targets(out, cand)):
            return pid, (0, 0.0, "kept")      # same census, same targets: nothing to redo
        cmd = [py, os.path.join(HERE, "kernel_shapes.py"), "--trace-dir", trace_dirs[pid],
               "--phase", phase[pid], "--census", censuses[pid], "--targets", cand,
               "--point-id", pid, "--out", os.path.join(shapes_dir, pid + ".json")]
        if phase[pid] == "PREFILL" and a.draft_shares_name:
            cmd.append("--draft-shares-name")
        return pid, run(cmd, log)
    t0 = time.time()
    # Two at a time: a decode trace with Python stacks holds ~6 GB while it is scanned.
    with ThreadPoolExecutor(max_workers=2) as pool:
        res = list(pool.map(shape_job, sorted(censuses)))
    bad = [pid for pid, (rc, _t, _m) in res if rc]
    # Decode kernels replayed from a CUDA graph carry no launching op and no source frame.
    # Their costs are right (that is the deployment); their evidence needs one eager window.
    replayed, req = {}, os.path.join(ev, "evidence", "eager_evidence_request.json")
    for pid in sorted(censuses):
        f = os.path.join(shapes_dir, pid + ".json")
        if phase[pid] == "DECODE" and os.path.exists(f):
            for k in json.load(open(f)).get("graph_replayed_kernels", []):
                replayed.setdefault(k, []).append(pid)
    covered = set()
    for f in os.listdir(shapes_dir):
        if is_eager_shapes_file(f):
            d = json.load(open(os.path.join(shapes_dir, f)))
            covered |= set(d.get("requested") or d.get("kernels", {}))
    if set(replayed) - covered:
        dec = sorted((p for p in points if p["phase"] == "DECODE" and p["point_id"] in censuses),
                     key=lambda p: p.get("batch_size") or 0)
        json.dump({"point": dec[0]["point_id"], "kernels": sorted(replayed),
                   "why": "graph-replayed decode launches name no op and no source frame"},
                  open(req, "w"), indent=1)
        timeline["eager_evidence_needed"] = sorted(replayed)
        print("[%s] %d decode kernel(s) are graph-replayed: requested one eager evidence "
              "window at %s (%s)" % (now(), len(replayed), dec[0]["point_id"], req), flush=True)
    timeline["post"]["kernel_shapes"] = {"rc": 1 if bad else 0, "wall_s": round(time.time() - t0, 1),
                                         "failed_points": bad}
    print("[%s] kernel_shapes %d point(s) in %.0fs%s" % (now(), len(res), time.time() - t0,
          "; FAILED: %s" % bad if bad else ""), flush=True)
    smap = os.path.join(ev, "evidence", "source_map.json")
    steps = [("source_map", [py, os.path.join(HERE, "source_map.py"), "--shapes-dir", shapes_dir,
                             "--out", smap])]
    if not a.select_by_kernel:
        steps.append(("select_operators", [py, os.path.join(HERE, "select_targets.py"), rank,
                                           "--source-map", smap, "--output", sel] + a.select_arg))
    steps += [("adjacency", [py, os.path.join(HERE, "adjacency.py"), "--targets",
                            cand if getattr(a, "all_evidence", False) else sel,
                            "--out", os.path.join(ev, "evidence", "adjacency.json")]
              + [x for pid in sorted(trace_dirs) for x in ("--trace-dir", trace_dirs[pid])])]
    return run_steps(steps) and not bad


if __name__ == "__main__":
    sys.exit(main())
