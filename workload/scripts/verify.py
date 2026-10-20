#!/usr/bin/env python3
"""The cold re-verification gate: no speedup may enter a report or be applied back until
it passes here.

Numbers measured by the worker during optimization **do not count**. In the previous
campaign this gate caught two things, neither hypothetical:

1. A kernel that measured PASS while running and failed CORRECTNESS on a cold rerun — it
   skipped part of its output, and because the unit-test leg allocates from a virgin
   allocator whose pages read back as zero, it sailed through every warm check.
2. A genuine 1.51x result that a node loss had caused the worker to revert; it survived
   in best_kernel_src, and re-verification recovered it.

So this file does three things:
  * Code identity: best source whose md5 equals the original while claiming a speedup is
    pure noise, judged `noise` outright.
  * N cold reruns: a fresh process each time, taking the **minimum** aggregate speedup
    (not the mean, and certainly not the maximum).
  * Recovery of best_kernel_src: if it is better than the current kernel_src, verify that
    instead.

Usage:
    python3 verify.py --work-root <dir> --tasks-dir <dir> [--repeats 2]
                      [--threshold 1.045] [--out verified.json]
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time

SPEEDUP_RE = re.compile(r"GEAK_GEOMEAN_SPEEDUP\s+([0-9.]+)")
GATE_KEYS = {"legs_differ", "baseline_recording", "leaf_execution",
             "negative_control", "correctness", "timing"}
WEIGHTED_RE = re.compile(r"GEAK_WEIGHTED_SPEEDUP\s+([0-9.]+)")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


RUN_ENTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_task_entry.py")


def editable_src(ksrc_dir):
    """The one editable file in kernel_src/, chosen by what it is.

    Sorting the directory and taking the first entry picks `__pycache__` as soon as one
    attempt has run -- a directory, which every md5 and copy then trips over."""
    names = [n for n in sorted(os.listdir(ksrc_dir))
             if os.path.isfile(os.path.join(ksrc_dir, n)) and not n.startswith("__")]
    py = [n for n in names if n.endswith(".py")]
    return (py or names or [None])[0]


def _objects(out):
    """Every JSON object in the output, pretty-printed ones included.

    json.loads per line only sees contract B. The extraction skill's harness prints its
    verdict across many lines, so decode from each `{` and keep what parses."""
    dec = json.JSONDecoder()
    found = []
    for i, ch in enumerate(out):
        if ch != "{":
            continue
        if found and i < found[-1][1]:
            continue
        try:
            obj, end = dec.raw_decode(out[i:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            found.append((obj, i + end))
    return [o for o, _ in found]


def _aggregate(timing):
    """The serving-weighted aggregate out of a gates.timing block.

    aggregate.primary when it exists; otherwise the px_causal / px_alloc pair, which is
    an interval, and the low end is the figure of record."""
    if not isinstance(timing, dict):
        return None
    agg = timing.get("aggregate")
    if isinstance(agg, dict) and isinstance(agg.get("primary"), (int, float)):
        return float(agg["primary"])
    iv = [float(timing[k]) for k in ("serving_weighted_speedup_px_causal",
                                     "serving_weighted_speedup_px_alloc")
          if isinstance(timing.get(k), (int, float))]
    if iv:
        return min(iv)
    for k in ("serving_weighted_speedup", "weighted", "speedup"):
        if isinstance(timing.get(k), (int, float)):
            return float(timing[k])
    return None


def parse_entry_output(out, task_dir=None):
    """(correct, speedup) from any of the three harness contracts.

    Contract A: `CORRECTNESS: PASS` plus `GEAK_GEOMEAN_SPEEDUP <x>`.
    Contract B: a JSON line {"exit": 0, "weighted": <x>} -- exit 0 means every gate
    passed and `weighted` is the serving-weighted speedup across the sweep.
    Contract C: a pretty-printed {"ut_result": .., "ok": true, "gates": {..}} or a bare
    gate map followed by `UT PASS`, with the aggregate under gates.timing and the
    numbers in the task's own ut_result.json. Treating C as a failure is how a sweep
    that passed all six gates got recorded as a correctness failure."""
    m = SPEEDUP_RE.findall(out)
    if m:
        return bool(re.search(r"^CORRECTNESS: PASS", out, re.M)), float(m[-1])

    objs = _objects(out)
    for d in objs:
        if "exit" in d:
            w = d.get("weighted")
            return d.get("exit") == 0, (None if w in (None, "") else float(w))

    # --- Contract C.
    gates, ok = None, None
    for d in objs:
        if isinstance(d.get("gates"), dict):
            gates, ok = d["gates"], d.get("ok")
        elif GATE_KEYS & set(d):            # a bare gate map, no wrapper
            gates = d
    if gates is None and not re.search(r"^UT (PASS|FAIL)", out, re.M):
        return False, None

    passed = all(v is True for k, v in gates.items()
                 if isinstance(v, bool)) if gates else None
    if re.search(r"^UT FAIL", out, re.M):
        passed = False
    elif re.search(r"^UT PASS", out, re.M) and passed is not False:
        passed = True
    if ok is False:
        passed = False
    if passed is None:
        return False, None

    sp = _aggregate((gates or {}).get("timing"))
    if sp is None and task_dir:
        # C puts the numbers in a file, not on stdout, and the file has lived at both
        # the task root and under run/ depending on the harness revision.
        for rel in ("ut_result.json", os.path.join("run", "ut_result.json")):
            p = os.path.join(task_dir, rel)
            if not os.path.exists(p):
                continue
            try:
                d = json.load(open(p))
            except (OSError, ValueError):
                continue
            sp = _aggregate((d.get("gates") or {}).get("timing")) or _aggregate(d)
            if sp is not None:
                break
    return passed, sp


def run_once(task_dir, timeout, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    try:
        # Through run_task_entry.py, never `cd task_dir && python3 unittest.py`: the task
        # directory shadows the stdlib `unittest` package and the failure surfaces as an
        # empty measurement far from its cause.
        p = subprocess.run([sys.executable, RUN_ENTRY, task_dir],
                           capture_output=True, text=True, timeout=timeout, env=env)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out += "\n[verify] TIMEOUT"
    ok, sp = parse_entry_output(out, task_dir)
    w = WEIGHTED_RE.findall(out)
    return {"correct": ok,
            "speedup": sp,
            "weighted": float(w[-1]) if w else sp,
            "log": out}


def original_src(task_dir, name, run_dir=None):
    """Where the original implementation lives. The extraction skill leaves a pristine
    copy; failing that, fall back to the worker's first snapshot.

    The snapshot matters: the extraction skill copies aside only the files it considers
    the kernel, and for a kernel spread over a .py launcher plus .cu/.h device code that
    is a strict subset. snapshots/before_attempt_1 is the whole tree as optimization
    found it, so it is a reference for every file."""
    for cand in (os.path.join(task_dir, "baseline_ref", name),
                 os.path.join(task_dir, "kernel_src_original", name),
                 os.path.join(task_dir, "original", name),
                 os.path.join(task_dir, ".original", name)):
        if os.path.exists(cand):
            return cand
    if run_dir:
        snaps = os.path.join(run_dir, "snapshots")
        if os.path.isdir(snaps):
            for d in sorted(os.listdir(snaps),
                            key=lambda n: (len(n), n)):   # before_attempt_1 first
                cand = os.path.join(snaps, d, name)
                if os.path.isfile(cand):
                    return cand
    return None


def verify_kernel(work_root, tasks_dir, kernel, args):
    task_dir = os.path.join(tasks_dir, kernel)
    run_dir = os.path.join(work_root, "runs", kernel)
    ksrc_dir = os.path.join(task_dir, "kernel_src")
    rec = {"kernel": kernel, "verdict": "fail", "notes": []}

    if not os.path.isdir(ksrc_dir):
        rec["notes"].append("the task directory has no kernel_src/")
        return rec
    name = editable_src(ksrc_dir)
    if not name:
        rec["notes"].append("kernel_src/ holds no editable file")
        return rec
    cur = os.path.join(ksrc_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    # --- Recover a result that was reverted.
    # Recovery is by filename, across the whole directory. A kernel is not always one
    # file -- opus_mla is a .py launcher plus a .cu and a .h -- and copying the
    # alphabetically-first entry onto the single editable file writes C++ into the
    # Python module, which fails in seconds in a way that reads like a bad kernel.
    best_dir = os.path.join(run_dir, "best_kernel_src")
    pairs = []
    if os.path.isfile(best_dir):
        pairs = [(best_dir, cur)]
    elif os.path.isdir(best_dir):
        for n in sorted(os.listdir(best_dir)):
            src = os.path.join(best_dir, n)
            if not os.path.isfile(src) or n.startswith("__"):
                continue
            dst = os.path.join(ksrc_dir, n)
            if not os.path.exists(dst):
                rec["notes"].append("best_kernel_src/%s has no counterpart in "
                                    "kernel_src/; not restoring it" % n)
                continue
            pairs.append((src, dst))
    changed = [(s, d) for s, d in pairs if md5(s) != md5(d)]
    if changed:
        rec["notes"].append("best_kernel_src differs from the current kernel_src in %s; "
                            "verifying best instead (a node loss may have reverted a "
                            "real result)"
                            % ", ".join(os.path.basename(d) for _, d in changed))
        saved = os.path.join(run_dir, "kernel_src_at_verify_time")
        os.makedirs(saved, exist_ok=True)
        for s, d in changed:
            shutil.copyfile(d, os.path.join(saved, os.path.basename(d)))
            shutil.copyfile(s, d)

    # --- Code identity: claiming a speedup without changing anything is noise.
    # Every file, not just the editable one: opus_mla's edit lives in a .h while its .py
    # is untouched, so a single-file check calls the whole kernel unchanged and dismisses
    # a real result in zero seconds.
    compared, differing, uncompared = [], [], []
    for fn in sorted(os.listdir(ksrc_dir)):
        fp = os.path.join(ksrc_dir, fn)
        if not os.path.isfile(fp) or fn.startswith("__") or fn == "source_manifest.json":
            continue
        o = original_src(task_dir, fn, run_dir)
        if not o:
            uncompared.append(fn)
            continue
        compared.append(fn)
        if md5(o) != md5(fp):
            differing.append(fn)
    # Only when every file was compared. A file nobody could look at may hold the whole
    # edit -- that is exactly how opus_mla's .h-resident KEEP was dismissed as noise
    # without being run. Uncomparable means unknown, and unknown goes to the GPU.
    if compared and not differing and not uncompared:
        rec.update(verdict="noise", speedup=1.0)
        rec["notes"].append("every source compared (%s) is md5-identical to the original "
                            "implementation; any speedup measured earlier was noise"
                            % ", ".join(compared))
        return rec
    if uncompared:
        rec["notes"].append("no original was found for %s; identity is therefore "
                            "unproven and the sweep runs" % ", ".join(uncompared))
    if not compared:
        rec["notes"].append("no copy of the original implementation was found; skipping "
                            "the code-identity check")
    elif differing:
        rec["notes"].append("changed against the original: %s" % ", ".join(differing))

    # --- Cold reruns. A fresh process each time means a fresh allocator each time.
    runs = []
    for i in range(args.repeats):
        r = run_once(task_dir, args.timeout)
        runs.append({k: r[k] for k in ("correct", "speedup", "weighted")})
        with open(os.path.join(run_dir, "verify_run_%d.log" % (i + 1)), "w") as f:
            f.write(r["log"])
        if not r["correct"]:
            rec.update(verdict="incorrect", runs=runs, speedup=None)
            rec["notes"].append("cold rerun %d failed CORRECTNESS, having passed during "
                                "optimization. The usual cause: part of the output is "
                                "never written and the warm run happened to read a "
                                "leftover value." % (i + 1))
            return rec
        if r["speedup"] is None:
            rec.update(verdict="fail", runs=runs)
            rec["notes"].append("rerun %d printed neither GEAK_GEOMEAN_SPEEDUP nor a "
                                "{\"exit\": .., \"weighted\": ..} line" % (i + 1))
            return rec

    sps = [r["speedup"] for r in runs]
    # Take the minimum. Only what holds across every cold start counts; a mean gets
    # pulled up by one lucky run.
    sp = min(sps)
    rec["runs"] = runs
    rec["speedup"] = sp
    rec["speedup_spread"] = round(max(sps) - min(sps), 4)
    if rec["speedup_spread"] > args.spread_warn:
        rec["notes"].append("spread across reruns is %.4f, which is wide; this number is "
                            "unstable" % rec["speedup_spread"])
    band = getattr(args, "thresholds", {}).get(kernel, args.threshold)
    rec["threshold"] = band
    if band != args.threshold:
        rec["notes"].append("this seam carries its own noise band %.4f (the campaign "
                            "default is %.4f)" % (band, args.threshold))
    rec["verdict"] = "pass" if sp >= band else "noise"
    if rec["verdict"] == "noise":
        rec["notes"].append("the cold minimum %.4f does not clear the noise band %.4f"
                            % (sp, band))
    return rec


def parse_overrides(spec):
    """"task=1.35,other=1.08" -> {"task": 1.35, "other": 1.08}. A malformed entry is a
    configuration error and must not silently fall back to the global band."""
    out = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--kernels", help="comma-separated; by default verifies every kernel "
                                      "that has a status.json")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=1.045)
    ap.add_argument("--threshold-override", default="", help='comma-separated task=threshold overrides. The noise band belongs to the seam: a collective kernel that barriers across 8 ranks has a far wider null spread than a single-rank one, and one global band is wrong for both.')
    ap.add_argument("--spread-warn", type=float, default=0.03)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    args.thresholds = parse_overrides(args.threshold_override)

    work_root = os.path.abspath(args.work_root)
    tasks_dir = os.path.abspath(args.tasks_dir)
    sdir = os.path.join(work_root, "state")

    if args.kernels:
        kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    else:
        kernels = sorted(f[: -len(".status.json")]
                         for f in (os.listdir(sdir) if os.path.isdir(sdir) else [])
                         if f.endswith(".status.json"))
    if not kernels:
        print("there is no kernel to verify", file=sys.stderr)
        return 2

    results = []
    for k in kernels:
        print("[verify] %s ..." % k, flush=True)
        t0 = time.time()
        r = verify_kernel(work_root, tasks_dir, k, args)
        r["verify_seconds"] = round(time.time() - t0, 1)
        try:
            claimed = json.load(open(os.path.join(sdir, "%s.status.json" % k))).get("best_speedup")
        except (OSError, ValueError):
            claimed = None
        r["claimed_speedup"] = claimed
        if claimed and r.get("speedup") and claimed - r["speedup"] > args.spread_warn:
            r.setdefault("notes", []).append(
                "optimization claimed %.4f; the cold rerun only reproduced %.4f — the "
                "rerun is what counts" % (claimed, r["speedup"]))
        results.append(r)
        print("    %-9s speedup=%s %s"
              % (r["verdict"], r.get("speedup"), "; ".join(r.get("notes", []))))

    out = args.out or os.path.join(work_root, "verified.json")
    payload = {"threshold": args.threshold, "threshold_overrides": args.thresholds,
               "repeats": args.repeats,
               "kernels": results,
               "passed": [r["kernel"] for r in results if r["verdict"] == "pass"]}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print("\nverified: %d / %d -> %s" % (len(payload["passed"]), len(results), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
