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


def parse_entry_output(out):
    """(correct, speedup) from either harness contract.

    Contract A: `CORRECTNESS: PASS` plus `GEAK_GEOMEAN_SPEEDUP <x>`.
    Contract B: a JSON line {"exit": 0, "weighted": <x>} -- what the extraction skill's
    frozen tests print, where exit 0 means every gate passed and `weighted` is the
    serving-weighted speedup across the whole sweep."""
    m = SPEEDUP_RE.findall(out)
    if m:
        return bool(re.search(r"^CORRECTNESS: PASS", out, re.M)), float(m[-1])
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if "exit" not in d:
            continue
        w = d.get("weighted")
        return d.get("exit") == 0, (None if w in (None, "") else float(w))
    return False, None


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
    ok, sp = parse_entry_output(out)
    w = WEIGHTED_RE.findall(out)
    return {"correct": ok,
            "speedup": sp,
            "weighted": float(w[-1]) if w else sp,
            "log": out}


def original_src(task_dir, name):
    """Where the original implementation lives. The extraction skill leaves a pristine
    copy; failing that, fall back to the snapshots."""
    for cand in (os.path.join(task_dir, "baseline_ref", name),
                 os.path.join(task_dir, "kernel_src_original", name),
                 os.path.join(task_dir, "original", name),
                 os.path.join(task_dir, ".original", name)):
        if os.path.exists(cand):
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
    best_dir = os.path.join(run_dir, "best_kernel_src")
    best_file = None
    if os.path.isfile(best_dir):
        best_file = best_dir
    elif os.path.isdir(best_dir) and os.listdir(best_dir):
        best_file = os.path.join(best_dir, sorted(os.listdir(best_dir))[0])
    if best_file and md5(best_file) != md5(cur):
        rec["notes"].append("best_kernel_src differs from the current kernel_src; "
                            "verifying best instead (a node loss may have reverted a real "
                            "result)")
        shutil.copyfile(cur, os.path.join(run_dir, "kernel_src_at_verify_time.py"))
        shutil.copyfile(best_file, cur)

    # --- Code identity: claiming a speedup without changing anything is noise.
    orig = original_src(task_dir, name)
    if orig and md5(orig) == md5(cur):
        rec.update(verdict="noise", speedup=1.0)
        rec["notes"].append("the source is md5-identical to the original implementation; "
                            "any speedup measured earlier was noise")
        return rec
    if not orig:
        rec["notes"].append("no copy of the original implementation was found; skipping "
                            "the code-identity check")

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
    rec["verdict"] = "pass" if sp >= args.threshold else "noise"
    if rec["verdict"] == "noise":
        rec["notes"].append("the cold minimum %.4f does not clear the noise band %.4f"
                            % (sp, args.threshold))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--kernels", help="comma-separated; by default verifies every kernel "
                                      "that has a status.json")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=1.045)
    ap.add_argument("--spread-warn", type=float, default=0.03)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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
    payload = {"threshold": args.threshold, "repeats": args.repeats,
               "kernels": results,
               "passed": [r["kernel"] for r in results if r["verdict"] == "pass"]}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print("\nverified: %d / %d -> %s" % (len(payload["passed"]), len(results), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
