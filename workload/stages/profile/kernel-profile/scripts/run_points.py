#!/usr/bin/env python3
"""Capture planned points on the supplied server; census and collect dispatch evidence.

    run_points.py --server-state probe/probe_server.json \
        --points sampling_points.json --observed evidence/observed_workload.json \
        --eval-dir "$EVAL_DIR"

Prefill runs in token order, decode in batch/context order. Each point gets at most
one recapture. A graph-hidden decode window requests eager fallback from
probe_server.py; a dead server stops capture. Signed completed censuses are reused
unless --redo is specified. Census work overlaps capture of the next point.

Outputs: per-point probe/census/shape evidence, coverage, ranking, selection,
source map, adjacency and profile_run.json. Task boundaries are not decided here.
Stdlib only.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def now():
    return time.strftime("%H:%M:%S")


def run(cmd, log):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(log, "a") as fh:
        fh.write("[%s] $ %s\n%s%s\n" % (now(), " ".join(cmd), p.stdout, p.stderr))
    return p.returncode, round(time.time() - t0, 1), (p.stdout + p.stderr).strip()


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
    ap.add_argument("--points", required=True, help="sampling_points.json from plan_points.py")
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
        a.server_command = st.get("command") or []
    else:
        a.server_command = []
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
    prev = {}
    try:
        prev = json.load(open(run_path)).get("points", {})
    except (OSError, ValueError):
        pass
    timeline = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "points": {},
                "server": a.base_url, "decode_eager": a.decode_eager}
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

    def census_job(pt, trace_dir):
        out = os.path.join(census_dir, pt["point_id"] + ".json")
        cmd = [py, os.path.join(HERE, "point_census.py"), "--trace-dir", trace_dir,
               "--phase", pt["phase"], "--point-id", pt["point_id"], "--out", out]
        planned = pt.get("new_tokens") if pt["phase"] == "PREFILL" else pt.get("batch_size")
        if planned:
            cmd += ["--planned-shape", str(planned)]
        if pt["phase"] == "PREFILL" and a.draft_shares_name:
            cmd.append("--draft-shares-name")
        rc, took, msg = run(cmd, log)
        result = {"rc": rc, "wall_s": took, "out": out if rc == 0 else None,
                  "trace_dir": trace_dir, "message": msg.splitlines()[-1] if msg else ""}
        hidden_reason = census_graph_hidden_reason(pt["phase"], rc, msg)
        if hidden_reason:
            result.update(graph_hidden=True, graph_hidden_reason=hidden_reason)
        return result

    if a.evidence_capture:
        eager_evidence(a, ev, plan, py, log, timeline)

    futures, censuses = {}, {}
    down = False
    with ThreadPoolExecutor(max_workers=1) as census_pool:
        for pt in points:
            if down:
                break
            pid = pt["point_id"]
            done = os.path.join(census_dir, pid + ".json")
            if os.path.exists(done) and pid not in a.redo:
                signature = pt.get("measurement_signature")
                if signature and prev.get(pid, {}).get("measurement_signature") != signature:
                    raise SystemExit("run_points: incompatible or unsigned shared census for %s; "
                                     "use --redo %s or a fresh plan" % (pid, pid))
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
            stamp = time.strftime("%H%M%S")
            for attempt in ("", "_r2"):
                tag = "%s%s_%s" % (pid, attempt, stamp) if pid in a.redo else pid + attempt
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
                    futures[pid] = census_pool.submit(census_job, pt, trace_dir)
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
        json.dump(timeline, open(run_path, "w"), indent=1)

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
    finish("ok" if ok and len(censuses) == len(points) else "incomplete")
    print("[%s] %s in %.0fs; %d/%d points censused%s"
          % (now(), timeline["status"], timeline["total_s"], len(censuses), len(points),
             (" -- missing: " + ", ".join(timeline["points_without_census"]))
             if timeline["points_without_census"] else ""))
    return 0 if timeline["status"] == "ok" else 1


def eager_windows(dec, replayed, missing, taken=()):
    """Decode points whose eager windows together launch every missing kernel: each time
    the point that covers most of the rest, ties to the smaller batch. A kernel graph-
    replayed only at batch 2 (vLLM DSpark: the id-rows MoE GEMMs) is absent from a batch-1
    window. Points that already have a window (`taken`) are not reused: a second window
    would overwrite its manifest."""
    windows, left = [], set(missing)
    order = [p["point_id"] for p in dec if p["point_id"] not in taken]
    while left:
        best = max(order, key=lambda pid: (len(left & {k for k, ps in replayed.items()
                                                      if pid in ps}), -order.index(pid)))
        got = sorted(k for k in left if best in replayed.get(k, []))
        if not got:
            break
        windows.append({"point": best, "kernels": got})
        left -= set(got)
    return windows


def eager_evidence(a, ev, plan, py, log, timeline):
    """Decode windows on the eager server, for frames and launching ops only. Their
    censuses live apart (evidence/census_eager) and never enter the ledger: under eager
    launch every rank spins in the all-reduce, so its costs are not the deployment's."""
    req = json.load(open(os.path.join(ev, "evidence", "eager_evidence_request.json")))
    windows = req.get("windows") or [{"point": req["point"], "kernels": req["kernels"]}]
    for w in windows:
        eager_window(a, ev, plan, py, log, timeline, w["point"], w["kernels"])


def eager_window(a, ev, plan, py, log, timeline, pid, kernels):
    tag = pid + "@eager"
    pt = next(p for p in plan["points"] if p["point_id"] == pid)
    trace_dir = os.path.join(a.trace_root, tag)
    os.makedirs(trace_dir, exist_ok=True)
    out = os.path.join(ev, "evidence", "probe", tag + ".json")
    rc, took, msg = run([py, os.path.join(HERE, "serving_probe.py"), "--points", a.points,
                         "--point-id", pid, "--base-url", a.base_url, "--trace-dir", trace_dir,
                         "--server-trace-dir", os.path.join(a.server_trace_root or a.trace_root, tag),
                         "--expect-files", str(a.expect_files), "--out", out] + a.probe_arg, log)
    cdir = os.path.join(ev, "evidence", "census_eager")
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, pid + ".json")
    steps = [[py, os.path.join(HERE, "point_census.py"), "--trace-dir", trace_dir, "--phase",
              "DECODE", "--point-id", tag, "--planned-shape", str(pt.get("batch_size")),
              "--out", cpath],
             [py, os.path.join(HERE, "kernel_shapes.py"), "--trace-dir", trace_dir, "--phase",
              "DECODE", "--census", cpath, "--point-id", tag, "--out",
              os.path.join(ev, "evidence", "shapes", tag + ".json")]
             + [x for k in kernels for x in ("--kernel", k)]]
    for cmd in steps if rc == 0 else []:
        rc, t2, msg = run(cmd, log)
        took += t2
        if rc:
            break
    timeline.setdefault("eager_evidence_windows", []).append(
        {"point": pid, "rc": rc, "wall_s": took, "message": msg.splitlines()[-1] if msg else ""})
    timeline["eager_evidence"] = timeline["eager_evidence_windows"][0]
    print("[%s] eager evidence window %s: rc=%d in %.0fs" % (now(), tag, rc, took), flush=True)


def flag_value(cmd, flag):
    for i, x in enumerate(cmd):
        if x == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if x.startswith(flag + "="):
            return x.split("=", 1)[1]
    return None


def frameless_prefill_kernels(shapes_dir, phase, wanted):
    """Candidate kernels launched in PREFILL whose samples, in every manifest, carry no
    Python frame and were not graph-replayed: launched from a thread the profiler's Python
    tracer never saw."""
    seen, framed = set(), set()
    for f in sorted(os.listdir(shapes_dir)):
        pid = f[:-len(".json")]
        try:
            d = json.load(open(os.path.join(shapes_dir, f)))
        except (OSError, ValueError):
            continue
        for k, v in (d.get("kernels") or {}).items():
            samples = [r for var in v.get("variants", []) for r in var.get("samples", [])]
            if any(r.get("py_frames") for r in samples):
                framed.add(k)
            elif (phase.get(pid) == "PREFILL" and k in wanted
                  and not any(r.get("graph_replayed") for r in samples)):
                seen.add(k)
    return sorted(seen - framed)


def serial_evidence(a, ev, phase, shapes_dir, censuses, cand, py, log, timeline):
    """vLLM --enable-dbo runs a prefill of at least --dbo-prefill-token-threshold tokens as
    micro-batches, each in a thread started for that step. The profiler records no Python
    frames for those threads, so the prefill kernels have no source (MiniMax-M3.1-Flash,
    vLLM 0.33: the MoE and indexer prefill kernels). One extra window on the same server, at
    the largest observed single-sequence prefill below the threshold, runs the forward on
    the worker thread. Like the eager window it supplies frames and launching ops only: its
    census stays in evidence/census_serial and never enters the ledger."""
    try:
        wanted = {r.get("id") or r.get("name") for r in json.load(open(cand))["targets"]}
    except (OSError, ValueError, KeyError):
        return
    missing = frameless_prefill_kernels(shapes_dir, phase, wanted)
    if not missing:
        return
    cmd = a.server_command
    limit = flag_value(cmd, "--dbo-prefill-token-threshold")
    if "--enable-dbo" not in cmd or not limit:
        timeline["frameless_prefill_kernels"] = {"kernels": missing, "evidence": None,
                                                 "why": "no serial-prefill window is known "
                                                        "for this server"}
        return
    plan = json.load(open(os.path.join(ev, "sampling_plan.json")))
    cells = [c["point"] for c in plan.get("captures", []) + plan.get("unselected", [])
             if c["point"]["phase"] == "PREFILL" and c["partition"][2] == 1
             and c["point"].get("new_tokens", 0) < int(limit)]
    if not cells:
        timeline["frameless_prefill_kernels"] = {"kernels": missing, "evidence": None,
                                                 "why": "no observed prefill below %s" % limit}
        return
    pt = dict(max(cells, key=lambda p: p["new_tokens"]))
    pid, tag = pt["point_id"], pt["point_id"] + "@serial"
    shapes = os.path.join(shapes_dir, tag + ".json")
    if os.path.exists(shapes) and set(missing) <= set(json.load(open(shapes)).get("requested", [])):
        return
    points = os.path.join(ev, "evidence", "serial_evidence_points.json")
    json.dump({"contract": "serial prefill evidence window (frames only)",
               "context_tolerance": plan.get("context_tolerance"), "points": [pt]},
              open(points, "w"), indent=1)
    trace_dir = os.path.join(a.trace_root, tag)
    os.makedirs(trace_dir, exist_ok=True)
    out = os.path.join(ev, "evidence", "probe", tag + ".json")
    t0 = time.time()
    rc, _t, msg = run([py, os.path.join(HERE, "serving_probe.py"), "--points", points,
                       "--point-id", pid, "--base-url", a.base_url, "--trace-dir", trace_dir,
                       "--server-trace-dir", os.path.join(a.server_trace_root or a.trace_root, tag),
                       "--expect-files", str(a.expect_files), "--out", out] + a.probe_arg, log)
    cdir = os.path.join(ev, "evidence", "census_serial")
    os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, pid + ".json")
    steps = [[py, os.path.join(HERE, "point_census.py"), "--trace-dir", trace_dir, "--phase",
              "PREFILL", "--point-id", tag, "--planned-shape", str(pt["new_tokens"]),
              "--out", cpath],
             [py, os.path.join(HERE, "kernel_shapes.py"), "--trace-dir", trace_dir, "--phase",
              "PREFILL", "--census", cpath, "--point-id", tag, "--out", shapes]
             + [x for k in missing for x in ("--kernel", k)]]
    for step in steps if rc == 0 else []:
        rc, _t, msg = run(step, log)
        if rc:
            break
    timeline["frameless_prefill_kernels"] = {
        "kernels": missing, "evidence": tag, "rc": rc, "dbo_prefill_token_threshold": int(limit),
        "new_tokens": pt["new_tokens"], "wall_s": round(time.time() - t0, 1),
        "message": msg.splitlines()[-1] if msg else ""}
    print("[%s] %d prefill kernel(s) had no frames (micro-batch threads): serial evidence "
          "window %s at %d tokens: rc=%d" % (now(), len(missing), tag, pt["new_tokens"], rc),
          flush=True)


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
    serial_evidence(a, ev, phase, shapes_dir, censuses, cand, py, log, timeline)
    covered, taken = set(), set()
    for f in os.listdir(shapes_dir):
        if f.endswith("@eager.json"):
            taken.add(f[:-len("@eager.json")])
            d = json.load(open(os.path.join(shapes_dir, f)))
            covered |= set(d.get("requested") or d.get("kernels", {}))
    if set(replayed) - covered:
        dec = sorted((p for p in points if p["phase"] == "DECODE" and p["point_id"] in censuses),
                     key=lambda p: p.get("batch_size") or 0)
        windows = eager_windows(dec, replayed, set(replayed) - covered, taken)
    if set(replayed) - covered and not windows:
        timeline["eager_evidence_unavailable"] = sorted(set(replayed) - covered)
    elif set(replayed) - covered:
        json.dump({"point": windows[0]["point"], "kernels": sorted(replayed), "windows": windows,
                   "why": "graph-replayed decode launches name no op and no source frame"},
                  open(req, "w"), indent=1)
        timeline["eager_evidence_needed"] = sorted(replayed)
        print("[%s] %d decode kernel(s) are graph-replayed: requested %d eager evidence "
              "window(s) at %s (%s)" % (now(), len(replayed), len(windows),
                                        ", ".join(w["point"] for w in windows), req), flush=True)
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
                            sel,
                            "--out", os.path.join(ev, "evidence", "adjacency.json")]
              + [x for pid in sorted(trace_dirs) for x in ("--trace-dir", trace_dirs[pid])])]
    return run_steps(steps) and not bad


if __name__ == "__main__":
    sys.exit(main())
