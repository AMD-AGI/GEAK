#!/usr/bin/env python3
"""Capture every sampling point on ONE live server and census each one while the next
is being captured; finish with the ledger, the ranking and the target selection.

    python3 run_points.py --points sampling_points.json \
        --base-url http://127.0.0.1:PORT --trace-root <SGLANG_TORCH_PROFILER_DIR> \
        --expect-files <TP ranks> --decode-eager --eval-dir "$EVAL_DIR"

Produces, under --eval-dir:
    evidence/probe/<point>.json     serving_probe.py result per point (and <point>_r2)
    evidence/census/<point>.json    point_census.py output per verified point
    coverage.json                   coverage_ledger.py
    kernel_ranking.json             build_ranking.py
    selection_targets.json          select_targets.py: by kernel name first, then (after
                                    the evidence below) by operator, which replaces it
    attribution_candidates.json     select_targets.py --candidates: kernels whose launch
                                    stacks are resolved so operators can be ranked
    evidence/shapes, evidence/source_map.json, evidence/adjacency.json, task_decls.json
    profile_run.json                what ran, in what order, and how long each step took

Why one script
--------------
The last two runs spent the profile hour like this: one probe server per pass (five
restarts at ~2 min each), the prompt cache flushed before every decode point so each
re-prefilled 137-179k tokens, a fixed post-arm wait of the whole window timeout, every
trace parsed with json.load, the census only after the last capture, and then 30-60
GPU-idle minutes of hand-written census/merge/ranking scripts. Here the server is
started once by the caller and never restarted or flushed, points run back to back,
and a point's census runs in the background as soon as its trace verifies.

The server
----------
Starting it stays the caller's job: the launcher, flags and environment belong to the
deployment (see references/capture.md). Start it ONCE, with the deployment's flags plus
`--disable-cuda-graph` when decode is graph-deployed, and pass --decode-eager: graph
replay hides decode kernels from the profiler, and verifying that only after a graph
pass cost one run a second server and ~35 minutes. Prefill is normally eager anyway,
so the same server serves both phases.

Order: PREFILL points (ascending tokens), then DECODE (ascending batch). Decode request
i uses the same prompt at every point, so moving to a larger batch only prefills the
requests that are new. A point whose capture fails gets one recapture (<point>_r2), as
the skill allows, and is then reported and skipped. If the server stops answering, the
captures stop at once and nothing is ranked; a rerun resumes.

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
    ap.add_argument("--points", required=True, help="plan_sampling_points.py output")
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
                         "(frames, launching ops), then redo the evidence and the draft")
    ap.add_argument("--no-evidence", action="store_true",
                    help="stop after selection: no shapes, source map, adjacency or draft")
    ap.add_argument("--ut-cases", default=None, help="passed to coverage_ledger.py")
    ap.add_argument("--probe-arg", action="append", default=[],
                    help="extra serving_probe.py argument, e.g. --probe-arg=--decode-tokens=20000")
    ap.add_argument("--select-arg", action="append", default=[],
                    help="extra select_targets.py argument (both selections)")
    ap.add_argument("--select-by-kernel", action="store_true",
                    help="keep the selection by kernel name: no candidate stacks, no "
                         "operator aggregation")
    ap.add_argument("--eval-dir", required=True)
    a = ap.parse_args()
    if a.server_state:
        st = json.load(open(a.server_state))
        a.base_url = a.base_url or st["base_url"]
        a.trace_root = a.trace_root or st["trace_root"]
        a.expect_files = a.expect_files or st["expect_files"]
        a.decode_eager = a.decode_eager or bool(st.get("decode_eager"))
        a.vocab_size = a.vocab_size or st.get("vocab_size")
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
        return {"rc": rc, "wall_s": took, "out": out if rc == 0 else None,
                "trace_dir": trace_dir, "message": msg.splitlines()[-1] if msg else ""}

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
                # Resume: a censused point is evidence already paid for.
                timeline["points"][pid] = dict(prev.get(pid, {"phase": pt["phase"]}),
                                               resumed=True)
                censuses[pid] = done
                continue
            rec = timeline["points"].setdefault(pid, {"phase": pt["phase"], "attempts": []})
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
            timeline["points"][pid]["census"] = c
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
    if a.ut_cases:
        cmd += ["--ut-cases", a.ut_cases]
    steps.append(("coverage_ledger", cmd))
    rank = os.path.join(ev, "kernel_ranking.json")
    cmd = [py, os.path.join(HERE, "build_ranking.py"), "--coverage", cov, "--out", rank,
           "--probe-result", os.path.join(probe_dir, "*.json")]
    if a.decode_eager:
        cmd += ["--eager-captured", "DECODE"]
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


def eager_evidence(a, ev, plan, py, log, timeline):
    """One decode window on the eager server, for frames and launching ops only. Its
    census lives apart (evidence/census_eager) and never enters the ledger: under eager
    launch every rank spins in the all-reduce, so its costs are not the deployment's."""
    req = json.load(open(os.path.join(ev, "evidence", "eager_evidence_request.json")))
    pid, tag = req["point"], req["point"] + "@eager"
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
             + [x for k in req["kernels"] for x in ("--kernel", k)]]
    for cmd in steps if rc == 0 else []:
        rc, t2, msg = run(cmd, log)
        took += t2
        if rc:
            break
    timeline["eager_evidence"] = {"point": pid, "rc": rc, "wall_s": took,
                                  "message": msg.splitlines()[-1] if msg else ""}
    print("[%s] eager evidence window %s: rc=%d in %.0fs" % (now(), tag, rc, took), flush=True)


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
    operator over them, then what a task needs that only the traces hold, and the draft."""
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
    covered = set()
    for f in os.listdir(shapes_dir):
        if f.endswith("@eager.json"):
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
    steps += [("adjacency", [py, os.path.join(HERE, "adjacency.py"), "--targets", sel,
                            "--out", os.path.join(ev, "evidence", "adjacency.json")]
              + [x for pid in sorted(trace_dirs) for x in ("--trace-dir", trace_dirs[pid])])]
    decls = os.path.join(ev, "task_decls.json")
    unreviewed = False
    if os.path.exists(decls):
        d = json.load(open(decls))
        unreviewed = all(t.get("needs_review") for t in d.get("tasks", []) + d.get("excluded", []))
    if not os.path.exists(decls) or unreviewed:
        # A draft nobody has touched is regenerated from the newer evidence; a reviewed
        # declaration is never overwritten.
        steps.append(("draft_task_decls", [py, os.path.join(HERE, "draft_task_decls.py"),
                                           "--eval-dir", ev, "--out", decls, "--force"]))
    return run_steps(steps) and not bad


if __name__ == "__main__":
    sys.exit(main())
