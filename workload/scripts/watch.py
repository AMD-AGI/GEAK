#!/usr/bin/env python3
"""Report on an unattended run every half hour; intervene only if asked to.

A campaign is a day and a half of GPU time started by one command. Two things can go
wrong quietly, and both have:

  * **Nobody looks until the end.** A stage that went wrong in its first ten minutes
    spends the rest of the window going further wrong.
  * **The session stops working but does not exit.** An agent waiting on a prompt that
    will never be answered, or on a command that will never return, looks exactly like an
    agent thinking hard. The time budget notices eventually -- after hours.

Reporting is therefore the default and the point: every tick writes `PROGRESS.md` (one
block per tick, appended), `progress.json` (the latest snapshot) and a refreshed
`REPORT.md` -- the run's real report, rendered by `report.py` from whatever the stages
have finished so far, rather than a liveness ping. A campaign that is killed on its
fourth stage still leaves a readable report of the first three.

Acting on what it sees is a separate decision, off unless `on_stall` says
`restart-attempt`. A monitor that reports is useful to everyone; a monitor that kills
things is useful only when its stall test is right, so the killing is opt-in and the
test below is deliberately conservative.

WHAT COUNTS AS STUCK, and why it takes two signals rather than one. A stage is called
stalled only when, across the whole stall window, **nothing under `$OUT` changed** *and*
**the agent's process tree burned almost no CPU**. Either signal alone lies: the long
stages legitimately spend forty minutes building an image or loading 892 GB of weights
without touching `$OUT`, and a process can spin without producing anything. Requiring
both means a stall is a stall.

WHAT INTERVENTION MEANS. The stalled attempt is killed -- and that is all. It is not a
rollback and not a restart of the stage: `agent_stage.dispatch` is already a retry loop
whose next attempt opens with a continuation prompt listing the artifacts on disk, so
killing a wedged attempt is how the work *resumes*. Nothing on disk is deleted or
rewritten. There is a cap (`max_interventions`, default 3 per stage), because a stage
that stalls three times is not a stage a monitor can fix, and saying so is more useful
than killing it forever.

Everything else it finds is reported, not acted on. A dead driver, a GPU held by another
job, a stage that reported `blocked` -- those need a decision, and a monitor inventing
one would be worse than a monitor that says clearly what it saw.

    python3 scripts/watch.py --config <cfg> [--interval 1800] [--once]

`stage.py all` starts one of these itself, so a plain campaign already reports on a
cadence without being asked; `--no-report` there turns it off.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report as report_mod  # noqa: E402
import stage as stage_mod  # noqa: E402

CLOCK_TICKS = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0


# ----------------------------------------------------------------- processes

def proc_cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return f.read().decode("utf-8", "replace").replace("\0", " ").strip()
    except OSError:
        return ""


def proc_stat(pid):
    """(ppid, cpu_seconds) for one pid, or None if it is gone."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            raw = f.read()
    except OSError:
        return None
    # The command field is parenthesised and may itself contain spaces.
    try:
        rest = raw[raw.rindex(")") + 2:].split()
        ppid = int(rest[1])
        cpu = (int(rest[11]) + int(rest[12])) / CLOCK_TICKS   # utime + stime
        return ppid, cpu
    except (ValueError, IndexError):
        return None


def all_pids():
    return [int(d) for d in os.listdir("/proc") if d.isdigit()]


def tree_for(marker):
    """Every pid whose command line mentions `marker`, plus all their descendants.

    The runner is invoked as `timeout N bash runners/<x>.sh <prompt> ...` and the prompt
    path lives under `$OUT`, so the stage's own output directory identifies its session
    without the dispatcher having to write a pid file that could outlive the process.
    """
    pids = all_pids()
    parents = {}
    for pid in pids:
        st = proc_stat(pid)
        if st:
            parents[pid] = st[0]
    roots = [p for p in pids if marker in proc_cmdline(p)]
    seen = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in seen and pid not in seen:
                seen.add(pid)
                changed = True
    return sorted(seen)


def tree_cpu(pids):
    total = 0.0
    for pid in pids:
        st = proc_stat(pid)
        if st:
            total += st[1]
    return total


def kill_tree(pids, log):
    """Kill children before parents, so nothing is re-parented and left running."""
    for pid in sorted(pids, reverse=True):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                log("  cannot signal pid %d (permission denied)" % pid)
                break
            time.sleep(0.2)
            if proc_stat(pid) is None:
                break


# ----------------------------------------------------------------- filesystem

def scan(out_dir):
    """Newest mtime, file count and total size under a stage's output directory."""
    newest, count, size = 0.0, 0, 0
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            count += 1
            size += st.st_size
            newest = max(newest, st.st_mtime)
    return {"files": count, "bytes": size, "newest": newest}


def changed_since(out_dir, since, limit=8):
    rows = []
    for root, dirs, files in os.walk(out_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            p = os.path.join(root, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_mtime > since:
                rows.append((st.st_mtime, os.path.relpath(p, out_dir), st.st_size))
    rows.sort(reverse=True)
    return rows[:limit]


def human_age(seconds):
    if seconds < 0:
        return "-"
    if seconds < 90:
        return "%ds" % int(seconds)
    if seconds < 5400:
        return "%dm" % int(seconds / 60)
    return "%.1fh" % (seconds / 3600.0)


def human_size(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return "%.0f%s" % (n, unit)
        n /= 1024.0


# ----------------------------------------------------------------- gpus

def gpu_usage_sysfs():
    """VRAM in use per card, straight from the driver.

    The monitor usually runs inside a container that has the render nodes but not
    `rocm-smi`, which is the one place a GPU report is actually wanted. sysfs is there
    either way.
    """
    used = []
    import glob
    for p in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")):
        if "-" in os.path.basename(os.path.dirname(os.path.dirname(p))):
            continue                                  # a partition, not a card
        try:
            with open(p) as f:
                used.append(int(f.read().strip()))
        except (OSError, ValueError):
            pass
    return used or None


def gpu_usage():
    """Best effort. A monitor that cannot read the GPUs says so rather than guessing."""
    try:
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                           capture_output=True, text=True, timeout=60)
        doc = json.loads(r.stdout)
    except Exception:
        return gpu_usage_sysfs()
    used = []
    for card in sorted(doc):
        for k, v in (doc[card] or {}).items():
            if "Used" in k and "VRAM" in k:
                try:
                    used.append(int(v))
                except (TypeError, ValueError):
                    pass
    return used or None


# ----------------------------------------------------------------- the tick

def snapshot(cfg, prev, driver_pid):
    now = time.time()
    state = stage_mod.load_state(cfg)
    stages = []
    for name in cfg["order"]:
        lay = stage_mod.stage_layout(cfg, name)
        out = lay["out"]
        rec = state["stages"].get(name) or {}
        s = {"stage": name, "skill": lay["skill"], "mode": lay["mode"],
             "status": rec.get("status") or ("-" if not os.path.isdir(out) else "running"),
             "out": out, "notes": rec.get("notes") or ""}
        s.update(scan(out) if os.path.isdir(out) else {"files": 0, "bytes": 0, "newest": 0})
        s["pids"] = tree_for(out) if os.path.isdir(out) else []
        s["cpu"] = tree_cpu(s["pids"]) if s["pids"] else 0.0
        p = (prev or {}).get("stages", {}).get(name) or {}
        s["new_files"] = s["files"] - p.get("files", s["files"])
        s["cpu_delta"] = s["cpu"] - p.get("cpu", s["cpu"]) if p else 0.0
        s["quiet_since"] = (p.get("quiet_since", now)
                            if p and s["newest"] <= p.get("newest", 0) else now)
        s["recent"] = changed_since(out, (prev or {}).get("at", now - 1800)) \
            if os.path.isdir(out) else []
        stages.append(s)
    return {"at": now, "driver_alive": proc_stat(driver_pid) is not None if driver_pid else None,
            "driver_pid": driver_pid, "gpu_vram_used": gpu_usage(),
            "stages": {s["stage"]: s for s in stages}, "order": list(cfg["order"])}


def findings(snap, prev, stall_sec, min_cpu_sec):
    """What is wrong, separated into what can be acted on and what cannot."""
    out = []
    if snap["driver_alive"] is False:
        done = all((snap["stages"][s]["status"] == "ok") for s in snap["order"])
        if not done:
            out.append(("report", None,
                        "the driver process (pid %s) is gone but the campaign is not "
                        "finished. Nothing is running. Re-issue `stage.py ... all` to "
                        "resume; completed stages are skipped."
                        % snap["driver_pid"]))
    for name in snap["order"]:
        s = snap["stages"][name]
        if s["status"] == "blocked":
            out.append(("report", name, "reported blocked: %s" % (s["notes"] or "(no note)")))
        if not s["pids"] or s["status"] == "ok":
            continue
        quiet = snap["at"] - s["quiet_since"]
        if quiet >= stall_sec and s["cpu_delta"] < min_cpu_sec:
            out.append(("stall", name,
                        "nothing written under $OUT for %s and the session used %.1fs of "
                        "CPU over that window (threshold %ss). Both signals agree it is "
                        "wedged rather than working."
                        % (human_age(quiet), s["cpu_delta"], min_cpu_sec)))
        elif quiet >= stall_sec:
            out.append(("report", name,
                        "nothing written under $OUT for %s, but the session is burning "
                        "CPU (%.0fs this window). Left alone: it is working on something "
                        "that does not touch $OUT."
                        % (human_age(quiet), s["cpu_delta"])))
    return out


def render(snap, prev, acts, tick, rows=None):
    at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(snap["at"]))
    gap = (" (+%s)" % human_age(snap["at"] - prev["at"])) if prev else ""
    L = ["", "## %s — tick %d%s" % (at, tick, gap), "",
         "| stage | state | files | size | newest write | this window |",
         "| --- | --- | ---: | ---: | --- | --- |"]
    for name in snap["order"]:
        s = snap["stages"][name]
        age = ("%s ago" % human_age(snap["at"] - s["newest"])) if s["newest"] else "never"
        win = ("+%d files, %.0fs CPU" % (s["new_files"], s["cpu_delta"])) if s["pids"] \
            else ("+%d files" % s["new_files"] if s["new_files"] else "-")
        L.append("| %s | %s | %d | %s | %s | %s |"
                 % (name, s["status"], s["files"], human_size(s["bytes"]), age, win))
    live = [n for n in snap["order"] if snap["stages"][n]["pids"]]
    if live:
        s = snap["stages"][live[0]]
        L += ["", "Written in the last window (%s):" % live[0]]
        if s["recent"]:
            for mt, rel, size in s["recent"]:
                L.append("  - `%s` (%s, %s)"
                         % (rel, human_size(size),
                            time.strftime("%H:%M:%S", time.gmtime(mt))))
        else:
            L.append("  - nothing")
    # --- What the run has produced so far, not merely that it is alive. Rendered from
    # the same rows as the final REPORT.md, so a reader of a tick and a reader of the
    # finished report are looking at one set of numbers.
    done = [r for r in (rows or []) if r["status"] not in ("-", "running")]
    if done:
        L += ["", "Results so far:"]
        for r in done:
            keys = list(r["metrics"])[:4]
            L.append("  - **%s** `%s` — %s%s"
                     % (r["stage"], r["skill"], r["status"],
                        (": " + ", ".join("%s=%s" % (k, r["metrics"][k]) for k in keys))
                        if keys else ""))
    vram = snap.get("gpu_vram_used")
    if vram:
        busy = [i for i, v in enumerate(vram) if v > 2 << 30]
        L += ["", "GPUs holding more than 2 GiB: %s"
              % (", ".join(str(i) for i in busy) if busy else "none")]
    else:
        L += ["", "GPU state: unproven (rocm-smi did not answer here)."]
    if acts:
        L += ["", "**Findings**"]
        for kind, name, msg in acts:
            L.append("  - [%s]%s %s" % (kind, (" %s:" % name) if name else "", msg))
    return "\n".join(L) + "\n"


def tick_once(cfg, prev, args, tick, counts, log):
    snap = snapshot(cfg, prev, args.driver_pid)
    acts = findings(snap, prev, args.stall_sec, args.min_cpu_sec)

    acted = []
    for kind, name, msg in acts:
        if kind != "stall" or args.on_stall == "report":
            continue
        if counts.get(name, 0) >= args.max_interventions:
            acted.append(("report", name,
                          "stalled again, but %d interventions have already been spent "
                          "on this stage. Not killing it again -- a stage that wedges "
                          "this often needs a person." % counts[name]))
            continue
        pids = snap["stages"][name]["pids"]
        counts[name] = counts.get(name, 0) + 1
        log("[watch] %s is stalled; killing attempt (pids %s). The dispatcher's next "
            "attempt resumes from the artifacts already on disk."
            % (name, ",".join(str(p) for p in pids)))
        kill_tree(pids, log)
        acted.append(("acted", name,
                      "killed the stalled attempt (%d pids, intervention %d of %d). "
                      "Nothing on disk was touched; the next attempt opens with a "
                      "continuation prompt listing what is already there."
                      % (len(pids), counts[name], args.max_interventions)))
        snap["stages"][name]["quiet_since"] = snap["at"]

    run_root = stage_mod.resolve_path(cfg["run_root"])
    os.makedirs(run_root, exist_ok=True)
    report = render(snap, prev, acts + acted, tick, report_mod.stage_rows(run_root))
    with open(os.path.join(run_root, "PROGRESS.md"), "a") as f:
        f.write(report)
    with open(os.path.join(run_root, "progress.json"), "w") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False, default=str)
    refresh_report(run_root, log)
    log(report.rstrip())
    return snap


def refresh_report(run_root, log):
    """Re-render REPORT.md from what has finished, so it is never only a final artifact.

    A failure here is logged and nothing else. Rendering a report is the least important
    thing happening on this machine, and a reporter that can take down the run it reports
    on would be a worse bargain than no reporter at all.
    """
    try:
        report_mod.main(["--run-root", run_root])
    except Exception as exc:
        log("[watch] REPORT.md not refreshed this tick: %s: %s"
            % (type(exc).__name__, exc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append")
    ap.add_argument("--set", action="append")
    ap.add_argument("--run-root")
    ap.add_argument("--interval", type=int, default=None, help="seconds between reports")
    ap.add_argument("--stall-sec", type=int, default=None,
                    help="silence under $OUT before a live stage counts as stalled")
    ap.add_argument("--min-cpu-sec", type=float, default=None,
                    help="CPU seconds in the window below which a quiet stage is wedged")
    ap.add_argument("--on-stall", choices=("restart-attempt", "report"), default=None)
    ap.add_argument("--max-interventions", type=int, default=None)
    ap.add_argument("--driver-pid", type=int, default=0,
                    help="pid of the stage.py that owns this run; 0 means do not check")
    ap.add_argument("--once", action="store_true", help="one tick, then exit")
    args = ap.parse_args(argv)
    args.dry_run = False
    args.cmd = "plan"
    cfg = stage_mod.load_config(args)

    w = stage_mod.report_settings(cfg)
    if args.interval is None:
        args.interval = int(w.get("interval_sec") or 1800)
    if args.stall_sec is None:
        args.stall_sec = int(w.get("stall_sec") or 5400)
    if args.min_cpu_sec is None:
        args.min_cpu_sec = float(w.get("min_cpu_sec") or 5.0)
    if args.on_stall is None:
        args.on_stall = w.get("on_stall") or "report"
    if args.max_interventions is None:
        args.max_interventions = int(w.get("max_interventions") or 3)

    def log(msg):
        print(msg, flush=True)

    log("[watch] reporting every %ds into PROGRESS.md + REPORT.md; stall = %ds quiet AND "
        "< %.0fs CPU; on stall: %s (max %d/stage)"
        % (args.interval, args.stall_sec, args.min_cpu_sec, args.on_stall,
           args.max_interventions))
    prev, counts, tick = None, {}, 0
    while True:
        tick += 1
        try:
            prev = tick_once(cfg, prev, args, tick, counts, log)
        except Exception as exc:                      # a monitor must not kill the run
            log("[watch] tick %d failed: %s: %s" % (tick, type(exc).__name__, exc))
        if args.once:
            return 0
        if args.driver_pid and proc_stat(args.driver_pid) is None:
            log("[watch] the driver is gone; one final report was written. Exiting.")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
