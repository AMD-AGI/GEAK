#!/usr/bin/env python3
"""Write, validate and fix every task's UT in parallel, one agent lane per task, spread over
the deployment's GPUs.

    python3 ut_wave.py plan   --eval-dir "$EVAL_DIR" [--gpus 0,...,7]
    python3 ut_wave.py start  --eval-dir "$EVAL_DIR" --runner "$RUNNER" \
        [--backend docker --image <serving image>] [--gpus ...] [--lane-sec 7200]
    python3 ut_wave.py wait   --eval-dir "$EVAL_DIR" [--timeout 540]   # 0 done, 2 running
    python3 ut_wave.py status --eval-dir "$EVAL_DIR"
    python3 ut_wave.py stop   --eval-dir "$EVAL_DIR"
    python3 ut_wave.py block  --task-dir <task> --reason "..."          # called by a lane agent

Input is what build_task_plan.py wrote: task_plan.json and each task's scaffold,
requested_kernel.json, case plan and server evidence. One lane = one task directory, one
runner session at a time (runners/CONTRACT.md: `bash <runner> <prompt> <workdir> <log>`),
pinned to its own GPU(s), until the task's own UT writes a passing receipt or the agent
declares a blocker. A writing agent never shares a task directory, so lanes cannot trample
each other's kernel_src.

The GPUs are the deployment's (serving.json `gpus`): the probe server that held them has
stopped by now, so the profile's whole allocation is free. Measured on the 2026-10-20 GLM-5.2
run, four sequential-in-spirit lanes took 1h24m, 1h29m, 2h03m and 3h20m -- wall clock is the
slowest lane, and four of eight GPUs sat idle. So:

  * lanes = min(GPUs, tasks), the most expensive task first (cases x family weight);
  * GPUs left over when tasks < GPUs go to the most expensive tasks (up to
    --max-gpus-per-task): ut_driver shards its correctness legs by case and spreads its
    timing buckets over every GPU a lane holds, which is where a 4186-case attention UT
    spends its hour;
  * a task whose harness needs several GPUs at once (requested_kernel required_gpu_count,
    e.g. a TP8 collective) starts once no single-GPU task is waiting and that many GPUs
    are free;
  * timing (--timing tail, the default) is not run inside the lanes, where it would time
    one task while seven others compile and run beside it. Lanes finish every other gate;
    the wave then times each task alone with `unittest.py --reuse` (the passed gates are
    kept, only timing runs) spread over all GPUs.

Backends: `docker` (production) starts one container per lane with exactly that lane's
/dev/dri/renderD<128+8*i> nodes mounted, so an agent cannot reach another lane's GPU;
`local` pins with ROCR/HIP_VISIBLE_DEVICES, which an agent can override -- for tests and
single-lane runs. Inside a lane the GPUs are always 0..k-1 (GPU_IDS).

`start` detaches the controller (it outlives the calling agent's tool timeout); `wait`
blocks up to --timeout seconds. All state lives in <eval-dir>/ut_wave/, so a restarted
controller resumes: passed and blocked tasks stay so, others are dispatched again.
Stdlib only.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.abspath(__file__)
SKILL = os.path.dirname(os.path.dirname(HERE))
LANE_DOC = os.path.join(SKILL, "references", "ut-lane.md")
# Relative effort per contract family, from the 2026-10-20 lanes: whole MoE took 2.4x a
# GEMM lane, attention with paged state and graph replay ~1.5x.
FAMILY_WEIGHT = {"fused_moe": 3.0, "sparse_mla": 2.0, "paged_attention_decode": 2.0,
                 "varlen_attention": 2.0, "dsa_indexer": 2.0, "all_reduce": 2.0}
FINAL = ("passed", "blocked", "failed")


def now():
    return time.time()


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp-%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def log(wave, msg):
    line = "[%s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    with open(os.path.join(wave, "wave.log"), "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- receipts
def receipt(tdir):
    """(verdict, fresh, report) of a task's last full UT run. `fresh` = written after the
    last change to anything the UT reads, so an old receipt never passes a changed task."""
    res = load(os.path.join(tdir, "run", "ut_result.json"))
    if not res:
        return None, False, None
    newest = 0.0
    for root, dirs, files in os.walk(tdir):
        rel = os.path.relpath(root, tdir)
        if rel.startswith(("run", "_cand_overlay", "baseline_overlay")) or "__pycache__" in rel:
            continue
        for f in files:
            newest = max(newest, os.path.getmtime(os.path.join(root, f)))
    fresh = os.path.getmtime(os.path.join(tdir, "run", "ut_result.json")) >= newest
    return res.get("verdict"), fresh, load(os.path.join(tdir, "run", "ut_report.json"), {})


def passed(tdir):
    verdict, fresh, _ = receipt(tdir)
    return verdict == "pass" and fresh


def timing_ready(tdir):
    """Every gate but timing passed on the current task (what a lane owes under --timing tail)."""
    verdict, fresh, rep = receipt(tdir)
    if not fresh or not rep:
        return False
    gates = rep.get("gates") or {}
    return all(r.get("status") in ("pass", "not_required")
               for g, r in gates.items() if g != "timing") and "correctness" in gates


# ----------------------------------------------------------------------------- planning
def task_rows(eval_dir, only=None):
    plan = load(os.path.join(eval_dir, "task_plan.json"))
    if not plan:
        raise SystemExit("ut_wave: %s/task_plan.json is missing (build_task_plan.py)" % eval_dir)
    rows = []
    for t in plan["tasks"]:
        if only and t["label"] not in only:
            continue
        req = load(os.path.join(t["task_dir"], "run", "requested_kernel.json"), {})
        cases = (t.get("planned_cases") or {}).get("total") or 1
        calls = ((t.get("server_evidence") or {}).get("calls") or 0)
        cost = (cases + calls / 4.0) * FAMILY_WEIGHT.get(t.get("ut_family"), 1.0)
        rows.append({"label": t["label"], "task_dir": t["task_dir"], "family": t.get("ut_family"),
                     "need": int(req.get("required_gpu_count") or 1), "cost": round(cost, 1)})
    rows.sort(key=lambda r: -r["cost"])
    return rows


def deployment_gpus(eval_dir):
    for path in (os.path.join(eval_dir, "serving.json"),
                 os.path.join(os.path.dirname(eval_dir.rstrip("/")), "serving.json")):
        cfg = load(path)
        if cfg and cfg.get("gpus"):
            return [int(g) for g in str(cfg["gpus"]).split(",") if g.strip() != ""]
    return None


def allocate(rows, n_gpus, cap):
    """GPUs per single-GPU task: one each, and when tasks < GPUs the rest go one at a time
    to the task with the most cost per GPU (at most `cap` each)."""
    single = [r for r in rows if r["need"] == 1]
    share = {r["label"]: 1 for r in single}
    spare = n_gpus - len(single)
    while spare > 0 and single:
        r = max(single, key=lambda r: (r["cost"] / share[r["label"]]
                                       if share[r["label"]] < cap else -1))
        if share[r["label"]] >= cap:
            break
        share[r["label"]] += 1
        spare -= 1
    for r in rows:
        if r["need"] > 1:
            share[r["label"]] = r["need"]
    return share


# ----------------------------------------------------------------------------- backends
class Local(object):
    def __init__(self, wave):
        self.wave, self.procs = wave, {}

    def start(self, name, gpus, argv, env):
        e = dict(os.environ, **env)
        if env.get("PYTHONPATH") and os.environ.get("PYTHONPATH"):
            e["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + os.environ["PYTHONPATH"]
        e["ROCR_VISIBLE_DEVICES"] = e["HIP_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        e["GPU_IDS"] = ",".join(str(i) for i in range(len(gpus)))
        out = open(os.path.join(self.wave, "logs", name + ".spawn.log"), "a")
        p = subprocess.Popen([sys.executable] + argv, env=e, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        self.procs[name] = p
        with open(os.path.join(self.wave, "state", name + ".pid"), "w") as f:
            f.write(str(p.pid))
        return True

    def alive(self, name):
        p = self.procs.get(name)
        if p is not None:
            return p.poll() is None
        try:
            pid = int(open(os.path.join(self.wave, "state", name + ".pid")).read())
            with open("/proc/%d/stat" % pid) as f:
                return f.read().split(") ", 1)[1].split()[0] not in ("Z", "X")
        except (OSError, ValueError, IndexError):
            return False

    def stop(self, name):
        try:
            pid = int(open(os.path.join(self.wave, "state", name + ".pid")).read())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        p = self.procs.pop(name, None)
        if p is not None:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                pass


class Docker(object):
    """One container per lane, mounting exactly the lane's render nodes."""

    def __init__(self, wave, image, render_base=128, render_step=8):
        self.wave, self.image = wave, image
        self.base, self.step = render_base, render_step

    def cname(self, name):
        return ("ut_lane_%s" % name)[:120]

    def start(self, name, gpus, argv, env):
        c = self.cname(name)
        subprocess.run(["docker", "rm", "-f", c], capture_output=True)
        cmd = ["docker", "run", "-d", "--name", c, "--network", "host", "--ipc", "host",
               "--device", "/dev/kfd", "--group-add", "video", "--cap-add", "SYS_PTRACE",
               "--security-opt", "seccomp=unconfined", "--shm-size", "32g",
               "-v", "/shared_nfs:/shared_nfs"]
        for g in gpus:
            node = "/dev/dri/renderD%d" % (self.base + self.step * g)
            if not os.path.exists(node):
                log(self.wave, "%s does not exist (renderD numbering is not %d+%d*i here)"
                    % (node, self.base, self.step))
                return False
            cmd += ["--device", node]
        env = dict(env, GPU_IDS=",".join(str(i) for i in range(len(gpus))))
        for k, v in env.items():
            cmd += ["-e", "%s=%s" % (k, v)]
        cmd += [self.image, "python3"] + argv
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if p.returncode:
            log(self.wave, "container for %s did not start: %s" % (name, (p.stderr or p.stdout)[-300:]))
        return p.returncode == 0

    def alive(self, name):
        p = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.cname(name)],
                           capture_output=True, text=True)
        return p.returncode == 0 and p.stdout.strip() == "true"

    def stop(self, name):
        subprocess.run(["docker", "rm", "-f", self.cname(name)], capture_output=True)


# ----------------------------------------------------------------------------- controller
PASS_ENV = ("WORKLOAD_AGENT_ENV_FILE", "WORKLOAD_AGENT_STATE_BASE", "WORKLOAD_AGENT_MODEL",
            "CLAUDE_BIN", "GEAK_ROOT", "WORKLOAD_ROOT", "SHARED_TOOLS", "HF_HOME")


def lane_env(eval_dir, extra):
    """What a lane runs under: the deployment's environment from serving.json (kernel
    dispatch reads it -- SGLANG_USE_AITER selects whole code paths), its PYTHONPATH prefix,
    the runner's credentials pointers, then --env overrides."""
    env = {}
    for path in (os.path.join(eval_dir, "serving.json"),
                 os.path.join(os.path.dirname(eval_dir.rstrip("/")), "serving.json")):
        cfg = load(path)
        if cfg:
            env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
            if cfg.get("pythonpath_prepend"):
                env["PYTHONPATH"] = cfg["pythonpath_prepend"]
            break
    env.update({k: os.environ[k] for k in PASS_ENV if os.environ.get(k)})
    for item in extra or []:
        k, _, v = item.partition("=")
        env[k] = v
    return env


def heartbeat(wave, name):
    """Refresh state/<name>.hb every 20 s until the returned event is set."""
    done = threading.Event()
    path = os.path.join(wave, "state", name + ".hb")

    def beat():
        while not done.is_set():
            save(path, {"t": now(), "pid": os.getpid()})
            done.wait(20)
    threading.Thread(target=beat, daemon=True).start()
    return done


def wave_dir(eval_dir):
    return os.path.join(os.path.abspath(eval_dir), "ut_wave")


def status_of(wave, label):
    return load(os.path.join(wave, "state", label + ".json"), {"status": "pending"})


def set_status(wave, label, **kw):
    st = status_of(wave, label)
    st.update(kw, updated=now())
    save(os.path.join(wave, "state", label + ".json"), st)
    return st


def controller(a):
    eval_dir = os.path.abspath(a.eval_dir)
    wave = wave_dir(eval_dir)
    for d in ("state", "logs", "prompts"):
        os.makedirs(os.path.join(wave, d), exist_ok=True)
    cfg = load(os.path.join(wave, "wave.json"))
    rows = cfg["rows"]
    gpus = cfg["gpus"]
    backend = (Docker(wave, cfg["image"], cfg["render_base"], cfg["render_step"])
               if cfg["backend"] == "docker" else Local(wave))
    with open(os.path.join(wave, "controller.pid"), "w") as f:
        f.write(str(os.getpid()))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    running = {}                          # job name -> {"gpus", "kind", "label", "started"}
    for r in rows:                        # resume: a lane that was running is dispatched again
        st = status_of(wave, r["label"])
        if st.get("status") not in FINAL + ("timing",):
            set_status(wave, r["label"], status="pending")
    t0 = now()
    log(wave, "wave: %d task(s) on GPUs %s, backend %s, timing %s; order %s"
        % (len(rows), gpus, cfg["backend"], cfg["timing"], [r["label"] for r in rows]))
    while not stop["flag"]:
        # 1) reap
        for name, job in list(running.items()):
            hb = load(os.path.join(wave, "state", name + ".hb"), {}).get("t", job["started"])
            alive = backend.alive(name)
            stale = now() - max(hb, job["started"]) > cfg["stale_sec"]
            if alive and not stale:
                continue
            backend.stop(name)
            del running[name]
            st = status_of(wave, job["label"])
            if job["kind"] == "lane" and st.get("status") not in FINAL + ("timing",):
                lost = int(st.get("lost") or 0) + 1
                set_status(wave, job["label"], status="pending" if lost <= cfg["relaunch"] else "failed",
                           lost=lost, why="lane %s (%s)" % ("went stale" if stale else "exited",
                                                             "relaunching" if lost <= cfg["relaunch"]
                                                             else "out of relaunches"))
            elif job["kind"] == "timing" and st.get("status") == "timing":
                ok = passed(st["task_dir"])
                set_status(wave, job["label"], status="passed" if ok else "failed",
                           why=None if ok else "the timing tail did not produce a passing receipt",
                           ended=now())
            log(wave, "%s %s finished on %s -> %s" % (job["kind"], job["label"], job["gpus"],
                                                      status_of(wave, job["label"]).get("status")))
        busy = {g for j in running.values() for g in j["gpus"]}
        free = [g for g in gpus if g not in busy]
        states = {r["label"]: status_of(wave, r["label"]) for r in rows}
        pending = [r for r in rows if states[r["label"]].get("status") == "pending"
                   and r["label"] not in {j["label"] for j in running.values()}]
        lanes_left = any(j["kind"] == "lane" for j in running.values()) or pending
        # 2) dispatch lanes: single-GPU tasks first, most expensive first
        for r in [p for p in pending if p["need"] == 1] + [p for p in pending if p["need"] > 1]:
            if r["need"] > 1 and any(p["need"] == 1 for p in pending if p is not r):
                continue
            want = cfg["share"].get(r["label"], 1)
            if r["need"] > 1 and len(free) < r["need"]:
                continue
            if not free:
                break
            take = free[:max(r["need"], min(want, len(free)))]
            free = free[len(take):]
            argv = [HERE, "lane", "--wave-dir", wave, "--task", r["label"]]
            if backend.start(r["label"], take, argv, cfg["env"]):
                running[r["label"]] = {"gpus": take, "kind": "lane", "label": r["label"],
                                       "started": now()}
                set_status(wave, r["label"], status="running", gpus=take, task_dir=r["task_dir"],
                           started=states[r["label"]].get("started") or now())
                log(wave, "lane %s <- GPUs %s (cost %s)" % (r["label"], take, r["cost"]))
            else:
                free = take + free
        # 3) timing tail: one task at a time, on every GPU, once no lane is left
        if cfg["timing"] == "tail" and not lanes_left and not running:
            ready = [r for r in rows if states[r["label"]].get("status") == "timing"]
            if ready:
                r = ready[0]
                name = "timing_" + r["label"]
                argv = [HERE, "tail", "--wave-dir", wave, "--task", r["label"]]
                if backend.start(name, gpus, argv, cfg["env"]):
                    running[name] = {"gpus": list(gpus), "kind": "timing", "label": r["label"],
                                     "started": now()}
                    log(wave, "timing %s <- all GPUs %s" % (r["label"], gpus))
        states = {r["label"]: status_of(wave, r["label"]).get("status") for r in rows}
        if not running and all(s in FINAL for s in states.values()):
            break
        if cfg["max_wall_sec"] and now() - t0 > cfg["max_wall_sec"]:
            log(wave, "wall-clock ceiling reached; stopping lanes (state is resumable)")
            break
        time.sleep(cfg["tick_sec"])
    for name in list(running):
        backend.stop(name)
    summary = {r["label"]: status_of(wave, r["label"]) for r in rows}
    save(os.path.join(wave, "summary.json"), {"finished": now(), "seconds": round(now() - t0),
                                              "tasks": summary})
    log(wave, "wave done in %ds: %s" % (now() - t0, {k: v.get("status") for k, v in summary.items()}))
    try:
        os.remove(os.path.join(wave, "controller.pid"))
    except OSError:
        pass
    return 0


# ----------------------------------------------------------------------------- lane
def lane_prompt(wave, row, st, attempt, cfg, deadline):
    tdir = row["task_dir"]
    req = os.path.join(tdir, "run", "requested_kernel.json")
    gpu_ids = os.environ.get("GPU_IDS", "0")
    lines = [
        "You are one lane of a parallel UT wave. You own exactly one task directory and the "
        "GPU(s) of this lane; other agents own the other tasks, at the same time. Nobody will "
        "answer questions: decide, act, record.", "",
        "TASK            %s" % row["label"],
        "TASK_DIR        %s" % tdir,
        "REQUEST         %s" % req,
        "EVAL_DIR        %s" % cfg["eval_dir"],
        "SKILL_ROOT      %s" % SKILL,
        "SHARED_TOOLS    %s" % os.path.join(SKILL, "shared"),
        "GPUS            logical %s (GPU_IDS; the UT driver spreads its legs over all of them)"
        % gpu_ids,
        "DEADLINE        %s (epoch %d)" % (time.strftime("%F %T", time.localtime(deadline)), deadline),
        "TIMING          %s" % ("do NOT run the timing gate: finish with `unittest.py --skip timing`; "
                                "the wave times every task alone afterwards"
                                if cfg["timing"] == "tail" else "run every gate, timing included"),
        "",
    ]
    if attempt > 1:
        verdict, fresh, rep = receipt(tdir)
        quick = load(os.path.join(tdir, "run", "ut_report.quick.json"), {})
        last = rep if rep else quick
        bad = {g: {k: v for k, v in r.items() if k in ("status", "why", "missing", "env_differs",
                                                       "fix_missing", "fix_env")}
               for g, r in (last.get("gates") or {}).items()
               if r.get("status") not in ("pass", "not_required", "skipped")}
        lines += ["THIS IS ATTEMPT %d. The previous session ended without a passing receipt%s. "
                  "Do not start over: read the task directory and continue. The last UT report "
                  "(%s) said:" % (attempt, " (%s)" % st.get("why") if st.get("why") else "",
                                  "run/ut_report.json" if rep else "run/ut_report.quick.json"),
                  "", json.dumps(bad, indent=1, default=str)[:4000] or "(no report yet)", ""]
    with open(LANE_DOC) as f:
        lines += ["=" * 78, "THE LANE PROCEDURE -- %s" % LANE_DOC, "=" * 78, "", f.read()]
    return "\n".join(lines)


def lane(a):
    wave = os.path.abspath(a.wave_dir)
    cfg = load(os.path.join(wave, "wave.json"))
    row = next(r for r in cfg["rows"] if r["label"] == a.task)
    tdir = row["task_dir"]
    done = heartbeat(wave, row["label"])
    st = status_of(wave, row["label"])
    deadline = (st.get("started") or now()) + cfg["lane_sec"] * cfg["attempts"]
    try:
        for attempt in range(int(st.get("attempts") or 0) + 1, cfg["attempts"] + 1):
            left = int(min(cfg["lane_sec"], deadline - now()))
            if left < 120:
                set_status(wave, row["label"], status="failed", why="lane budget spent")
                return 1
            prompt = os.path.join(wave, "prompts", "%s.%d.txt" % (row["label"], attempt))
            with open(prompt, "w") as f:
                f.write(lane_prompt(wave, row, st, attempt, cfg, now() + left))
            logf = os.path.join(wave, "logs", "%s.%d.log" % (row["label"], attempt))
            set_status(wave, row["label"], attempts=attempt, last_prompt=prompt, last_log=logf)
            env = dict(os.environ, TASK_DIR=tdir, SKILL_ROOT=SKILL, EVAL_DIR=cfg["eval_dir"],
                       SHARED_TOOLS=os.path.join(SKILL, "shared"))
            rc = subprocess.call(["timeout", str(left), "bash", cfg["runner"], prompt, tdir, logf],
                                 env=env)
            blocker = load(os.path.join(tdir, "run", "lane_blocked.json"))
            if rc == 78:
                set_status(wave, row["label"], status="blocked",
                           why="runner precondition unmet (exit 78); see %s" % logf)
                return 0
            if blocker:
                set_status(wave, row["label"], status="blocked", why=blocker.get("reason"))
                return 0
            if passed(tdir):
                set_status(wave, row["label"], status="passed", ended=now(), why=None)
                return 0
            if cfg["timing"] == "tail" and timing_ready(tdir):
                set_status(wave, row["label"], status="timing", why=None)
                return 0
            verdict, fresh, _ = receipt(tdir)
            st = set_status(wave, row["label"],
                            why="session exited %d; receipt %s%s" % (
                                rc, verdict or "absent", "" if fresh or not verdict else " (stale)"))
        set_status(wave, row["label"], status="failed", ended=now())
        return 1
    finally:
        done.set()


def tail(a):
    """Time one task alone over every GPU: `unittest.py --reuse` keeps the gates the lane
    passed under an unchanged task and runs only what is left (timing)."""
    wave = os.path.abspath(a.wave_dir)
    cfg = load(os.path.join(wave, "wave.json"))
    row = next(r for r in cfg["rows"] if r["label"] == a.task)
    logf = os.path.join(wave, "logs", "%s.timing.log" % row["label"])
    done = heartbeat(wave, "timing_" + row["label"])
    try:
        with open(logf, "a") as out:
            rc = subprocess.call([sys.executable, os.path.join(row["task_dir"], "unittest.py"),
                                  "--reuse"], cwd=row["task_dir"], stdout=out,
                                 stderr=subprocess.STDOUT)
    finally:
        done.set()
    set_status(wave, row["label"], status="passed" if passed(row["task_dir"]) else "failed",
               ended=now(), why=None if rc == 0 else "timing tail exited %d; see %s" % (rc, logf))
    return 0


# ----------------------------------------------------------------------------- CLI
def cmd_plan(a, quiet=False):
    eval_dir = os.path.abspath(a.eval_dir)
    gpus = ([int(g) for g in a.gpus.split(",") if g.strip() != ""] if a.gpus
            else deployment_gpus(eval_dir))
    if not gpus:
        raise SystemExit("ut_wave: no --gpus and no `gpus` in serving.json")
    only = set(a.tasks.split(",")) if a.tasks else None
    rows = task_rows(eval_dir, only)
    rows = [r for r in rows if a.force or not passed(r["task_dir"])]
    big = [r for r in rows if r["need"] > len(gpus)]
    if big:
        raise SystemExit("ut_wave: %s need more GPUs than the wave has (%d)"
                         % ([(r["label"], r["need"]) for r in big], len(gpus)))
    share = allocate(rows, len(gpus), a.max_gpus_per_task)
    if not quiet:
        print("%d GPU(s): %s" % (len(gpus), gpus))
        for r in rows:
            print("  %-48s %-22s cost %8.1f  gpus %d%s" % (r["label"], r["family"], r["cost"],
                                                           share[r["label"]],
                                                           "  (exclusive, after single-GPU tasks)"
                                                           if r["need"] > 1 else ""))
    return eval_dir, gpus, rows, share


def cmd_start(a):
    eval_dir, gpus, rows, share = cmd_plan(a)
    wave = wave_dir(eval_dir)
    if os.path.exists(os.path.join(wave, "controller.pid")):
        try:
            os.kill(int(open(os.path.join(wave, "controller.pid")).read()), 0)
            raise SystemExit("ut_wave: a controller is already running for %s" % wave)
        except (OSError, ValueError):
            pass
    if not rows:
        print("ut_wave: every task already has a passing receipt")
        return 0
    if a.backend == "docker" and not a.image:
        raise SystemExit("ut_wave: --backend docker needs --image (the serving image)")
    runner = os.path.abspath(a.runner)
    if not os.path.exists(runner):
        raise SystemExit("ut_wave: runner %s does not exist" % runner)
    for d in ("state", "logs", "prompts"):
        os.makedirs(os.path.join(wave, d), exist_ok=True)
    save(os.path.join(wave, "wave.json"), {
        "eval_dir": eval_dir, "gpus": gpus, "rows": rows, "share": share,
        "backend": a.backend, "image": a.image, "render_base": a.render_base,
        "render_step": a.render_step, "runner": runner, "timing": a.timing,
        "lane_sec": a.lane_sec, "attempts": a.attempts, "relaunch": a.relaunch,
        "stale_sec": a.stale_sec, "tick_sec": a.tick_sec, "max_wall_sec": a.max_wall_sec,
        "env": lane_env(eval_dir, a.env),
        "created": now()})
    if a.foreground:
        return controller(argparse.Namespace(eval_dir=eval_dir))
    out = open(os.path.join(wave, "controller.log"), "a")
    p = subprocess.Popen([sys.executable, HERE, "_controller", "--eval-dir", eval_dir],
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print("ut_wave: controller pid %d; `ut_wave.py wait --eval-dir %s` to follow" % (p.pid, eval_dir))
    return 0


def cmd_status(a, quiet=False):
    wave = wave_dir(a.eval_dir)
    cfg = load(os.path.join(wave, "wave.json"))
    if not cfg:
        print("ut_wave: no wave under %s" % wave)
        return 1, {}
    rows = {r["label"]: status_of(wave, r["label"]) for r in cfg["rows"]}
    if not quiet:
        for label, st in rows.items():
            print("  %-48s %-8s gpus %-12s attempts %s %s" % (
                label, st.get("status"), st.get("gpus") or "-", st.get("attempts") or 0,
                ("-- " + st["why"]) if st.get("why") else ""))
    return 0, rows


def controller_alive(wave):
    try:
        os.kill(int(open(os.path.join(wave, "controller.pid")).read()), 0)
        return True
    except (OSError, ValueError):
        return False


def cmd_wait(a):
    wave = wave_dir(a.eval_dir)
    t_end = now() + a.timeout
    while True:
        _, rows = cmd_status(a, quiet=True)
        if rows and all(st.get("status") in FINAL for st in rows.values()) \
                and not controller_alive(wave):
            cmd_status(a)
            return 0
        if not controller_alive(wave) and os.path.exists(os.path.join(wave, "wave.json")) \
                and not os.path.exists(os.path.join(wave, "summary.json")):
            print("ut_wave: the controller is not running and the wave is unfinished; "
                  "`ut_wave.py start` resumes it")
            cmd_status(a)
            return 3
        if now() >= t_end:
            cmd_status(a)
            return 2
        time.sleep(min(30, max(1, t_end - now())))


def cmd_stop(a):
    wave = wave_dir(a.eval_dir)
    try:
        os.kill(int(open(os.path.join(wave, "controller.pid")).read()), signal.SIGTERM)
        print("ut_wave: controller asked to stop; lanes are stopped with it")
    except (OSError, ValueError):
        print("ut_wave: no controller running")
    return 0


def cmd_block(a):
    path = os.path.join(os.path.abspath(a.task_dir), "run", "lane_blocked.json")
    if not a.reason.strip():
        raise SystemExit("ut_wave block: --reason must say what is missing and how it shows")
    save(path, {"reason": a.reason.strip(), "at": now()})
    print("recorded %s" % path)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "start"):
        p = sub.add_parser(name)
        p.add_argument("--eval-dir", required=True)
        p.add_argument("--gpus", default=None, help="physical GPU ids; default serving.json `gpus`")
        p.add_argument("--tasks", default=None, help="comma list of task labels (default all)")
        p.add_argument("--force", action="store_true", help="include tasks that already pass")
        p.add_argument("--max-gpus-per-task", type=int, default=4)
        if name == "start":
            p.add_argument("--runner", required=True, help="runners/CONTRACT.md runner script")
            p.add_argument("--backend", choices=("docker", "local"), default="docker")
            p.add_argument("--image", default=None)
            p.add_argument("--render-base", type=int, default=128)
            p.add_argument("--render-step", type=int, default=8)
            p.add_argument("--timing", choices=("tail", "lane"), default="tail")
            p.add_argument("--lane-sec", type=int, default=7200, help="per runner session")
            p.add_argument("--attempts", type=int, default=3, help="runner sessions per lane")
            p.add_argument("--relaunch", type=int, default=1,
                           help="relaunches of a lane whose process died or went stale")
            p.add_argument("--stale-sec", type=int, default=900)
            p.add_argument("--tick-sec", type=int, default=15)
            p.add_argument("--max-wall-sec", type=int, default=0)
            p.add_argument("--env", action="append", default=[],
                           help="K=V added to every lane (after serving.json env)")
            p.add_argument("--foreground", action="store_true")
    p = sub.add_parser("_controller")
    p.add_argument("--eval-dir", required=True)
    for name in ("status", "stop", "wait"):
        p = sub.add_parser(name)
        p.add_argument("--eval-dir", required=True)
        if name == "wait":
            p.add_argument("--timeout", type=float, default=540)
    for name in ("lane", "tail"):
        p = sub.add_parser(name)
        p.add_argument("--wave-dir", required=True)
        p.add_argument("--task", required=True)
    p = sub.add_parser("block")
    p.add_argument("--task-dir", required=True)
    p.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cmd_plan(a)
        return 0
    return {"start": cmd_start, "_controller": controller, "status": lambda a: cmd_status(a)[0],
            "stop": cmd_stop, "wait": cmd_wait, "lane": lane, "tail": tail,
            "block": cmd_block}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
