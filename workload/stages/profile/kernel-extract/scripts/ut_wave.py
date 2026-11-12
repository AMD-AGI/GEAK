#!/usr/bin/env python3
"""Parallel UT authors with command-scoped GPU leases (Linux/ROCm, stdlib only).

    ut_wave.py plan/start --eval-dir DIR --gpus <physical-ids>
    ut_wave.py start --eval-dir DIR --runner RUNNER --backend docker --image IMAGE
    ut_wave.py status/wait/stop --eval-dir DIR
    ut_wave.py gpu [--count N] [--timeout SEC] [--env K=V] -- <foreground-command>

The wave starts up to --author-workers CPU-only lanes, one
writer per task. Every GPU operation goes through `gpu`: a host broker acquires
required_gpu_count devices together from the allowed pool, launches a short-lived
validation container/process, drains its workers and returns the devices. CPU
source work/lint holds no GPU. The broker shares gpu_lock.sh's host flock files;
do not hold an outer pool lease around this wave. Queue metadata and logs live in
<eval-dir>/ut_wave/gpu_queue; input tensors are created by the UT, never transported
through the queue. Host/container task and skill paths must match.

Every task qualifies via unittest.py --validation-mode correctness --reuse.
Performance measurements run later through the optimizer's task entrypoint.

Docker CPU lanes mount no GPU devices; GPU jobs mount only their leased render
nodes in the pinned image. Local mode uses physical ROCR and logical HIP/GPU_IDS
masks and is intended for tests/trusted runners. `start` detaches the controller;
retries/resume drain the previous attempt before starting another task writer.
"""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import ut_gpu_queue as GPU

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
    # UT_WAVE_ORDER: an explicit priority (comma list of labels, e.g. by workload share) that
    # replaces the cost order for the labels it names; unnamed tasks keep cost order after them
    order = [x for x in os.environ.get("UT_WAVE_ORDER", "").split(",") if x]
    if order:
        rows.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else len(order))
    return rows


def parse_gpu_ids(value, source):
    values = value if isinstance(value, list) else str(value).split(',')
    if not values or any(not str(v).strip().isascii() or not str(v).strip().isdigit() for v in values):
        raise SystemExit('ut_wave: %s must contain nonnegative physical GPU IDs' % source)
    ids = [int(v) for v in values]
    if len(set(ids)) != len(ids):
        raise SystemExit('ut_wave: %s contains duplicate GPU IDs' % source)
    return ids


def deployment_gpus(eval_dir):
    for path in (os.path.join(eval_dir, "serving.json"),
                 os.path.join(os.path.dirname(eval_dir.rstrip("/")), "serving.json")):
        if not os.path.exists(path):
            continue
        cfg = load(path)
        if not isinstance(cfg, dict):
            raise SystemExit('ut_wave: cannot read workload GPU allocation from %s' % path)
        if 'gpus' in cfg:
            return parse_gpu_ids(cfg['gpus'], path + ':gpus')
    return None


# ----------------------------------------------------------------------------- backends
class Local(object):
    def __init__(self, wave):
        self.wave, self.procs = wave, {}

    def start(self, name, gpus, argv, env):
        e = dict(os.environ, **env)
        if env.get("PYTHONPATH") and os.environ.get("PYTHONPATH"):
            e["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + os.environ["PYTHONPATH"]
        e["ROCR_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        e["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(gpus)))
        e["CUDA_VISIBLE_DEVICES"] = e["HIP_VISIBLE_DEVICES"]
        e["GPU_IDS"] = ",".join(str(i) for i in range(len(gpus)))
        out = open(os.path.join(self.wave, "logs", name + ".spawn.log"), "a")
        p = subprocess.Popen([sys.executable] + argv, env=e, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        out.close()
        self.procs[name] = p
        with open(os.path.join(self.wave, "state", name + ".pid"), "w") as f:
            f.write(str(p.pid))
        save(os.path.join(self.wave, 'state', name + '.process.json'),
             {'pid': p.pid, 'identity': GPU.identity(p.pid)})
        return True

    def alive(self, name):
        p = self.procs.get(name)
        if p is not None:
            return p.poll() is None
        try:
            pid = int(open(os.path.join(self.wave, "state", name + ".pid")).read())
            saved = load(os.path.join(self.wave, 'state', name + '.process.json'))
            if saved and GPU.identity(pid) != saved['identity']:
                return False
            with open("/proc/%d/stat" % pid) as f:
                return f.read().split(") ", 1)[1].split()[0] not in ("Z", "X")
        except (OSError, ValueError, IndexError):
            return False

    def stop(self, name):
        pid = None
        try:
            with open(os.path.join(self.wave, "state", name + ".pid")) as f:
                pid = int(f.read())
            saved = load(os.path.join(self.wave, 'state', name + '.process.json'))
            current = GPU.identity(pid)
            if current and saved and current != saved['identity']:
                return  # PID reuse must not kill another workload
            # start_new_session makes the saved pid the group id even if its
            # leader has already exited. Drain orphan group members as well.
            os.killpg(pid, signal.SIGTERM)
        except (OSError, ValueError):
            pass
        p = self.procs.pop(name, None)
        if p is not None:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                # GPU supervisors must finish their subreaper cleanup before
                # their owner releases the host lock. Never kill that supervisor.
                if name.startswith("gpu_"):
                    p.wait()
                else:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        if pid is not None:
            import glob
            deadline = now() + 2
            while True:
                live = []
                for path in glob.glob('/proc/[0-9]*/stat'):
                    try:
                        with open(path) as f:
                            fields = f.read().split(') ', 1)[1].split()
                        if int(fields[2]) == pid and fields[0] not in ('Z', 'X'):
                            live.append(int(path.split('/')[2]))
                    except (OSError, ValueError, IndexError):
                        pass
                if not live:
                    break
                # For an adopted GPU supervisor, allow it to drain all workers.
                if now() > deadline and not name.startswith("gpu_"):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(.05)


class Docker(object):
    """One container per lane, mounting exactly the lane's render nodes."""

    def __init__(self, wave, image, render_base=128, render_step=8, container_prefix=None):
        self.wave, self.image = wave, image
        self.base, self.step = render_base, render_step
        self.prefix = container_prefix or default_container_prefix(wave)

    def cname(self, name):
        return (self.prefix + name)[:120]

    def start(self, name, gpus, argv, env):
        c = self.cname(name)
        subprocess.run(["docker", "rm", "-f", c], capture_output=True)
        cmd = ["docker", "run", "-d", "--name", c, "--network", "host", "--ipc", "host",
               "--group-add", "video", "--cap-add", "SYS_PTRACE",
               "--security-opt", "seccomp=unconfined", "--shm-size", "32g",
               "-v", "/shared_nfs:/shared_nfs"]
        # UT_WAVE_DOCKER_EXTRA: extra docker args (mounts the runner needs, e.g. its binary and
        # credential file outside /shared_nfs, model caches); never device or GPU options
        import shlex
        extra = shlex.split(os.environ.get("UT_WAVE_DOCKER_EXTRA", ""))
        if any(x.split('=', 1)[0] in ('--device', '--device-cgroup-rule', '--gpus', '--privileged')
               or '/dev/' in x or 'docker.sock' in x for x in extra):
            raise ValueError("UT_WAVE_DOCKER_EXTRA must not grant GPU devices or a Docker socket")
        cmd += extra
        if gpus:
            cmd += ["--device", "/dev/kfd"]
        for g in gpus:
            node = "/dev/dri/renderD%d" % (self.base + self.step * g)
            if not os.path.exists(node):
                log(self.wave, "%s does not exist (renderD numbering is not %d+%d*i here)"
                    % (node, self.base, self.step))
                return False
            cmd += ["--device", node]
        env = dict(env, GPU_IDS=",".join(str(i) for i in range(len(gpus))))
        # Mounted render nodes define the physical set. HIP indices are local
        # to that set; a serving.json mask must not override the new allocation.
        env.pop('ROCR_VISIBLE_DEVICES', None)
        env['HIP_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES'] = env['GPU_IDS']
        if not gpus:
            env['ROCR_VISIBLE_DEVICES'] = ''
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
        p = subprocess.run(["docker", "rm", "-f", self.cname(name)], capture_output=True)
        if p.returncode:
            check = subprocess.run(["docker", "inspect", self.cname(name)], capture_output=True)
            if check.returncode == 0 or b'No such' not in check.stderr:
                raise RuntimeError("cannot confirm container drained: %s" % self.cname(name))


# ----------------------------------------------------------------------------- controller
def default_container_prefix(wave):
    # Multiple campaigns may share one host pool. Identical task labels must not
    # cause one campaign's docker rm -f to remove another's author.
    return os.environ.get('UT_WAVE_CONTAINER_PREFIX') or (
        'ut_lane_' + hashlib.sha256(os.path.abspath(wave).encode()).hexdigest()[:12] + '_')


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
    """The independent broker retains leases and drains GPU jobs if we disappear."""
    wave = wave_dir(a.eval_dir)
    cfg = load(os.path.join(wave, 'wave.json'))
    broker = None
    backend = GPU.backend_for(wave, cfg)
    try:
        os.makedirs(os.path.join(wave, 'gpu_queue'), exist_ok=True)
        try:
            os.unlink(os.path.join(wave, 'broker.json'))
        except FileNotFoundError:
            pass
        with open(os.path.join(wave, 'broker.log'), 'a') as out:
            broker = subprocess.Popen([sys.executable, GPU.__file__, 'broker', wave, str(os.getpid())],
                                      stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        while not load(os.path.join(wave, 'broker.json')):
            if broker.poll() is not None:
                raise RuntimeError('GPU broker failed to start; see broker.log')
            time.sleep(.05)
        return controller_run(a)
    finally:
        try:
            for row in cfg['rows']:
                backend.stop(row['label'])
        finally:
            if broker is not None:
                if broker.poll() is None:
                    broker.terminate()
                broker.wait()  # no unlock until all GPU workers/containers drain
            try:
                os.remove(os.path.join(wave, 'controller.pid'))
            except FileNotFoundError:
                pass


def controller_run(a):
    eval_dir = os.path.abspath(a.eval_dir)
    wave = wave_dir(eval_dir)
    cfg = load(os.path.join(wave, "wave.json"))
    rows = cfg["rows"]
    backend = GPU.backend_for(wave, cfg)
    with open(os.path.join(wave, "controller.pid"), "w") as f:
        f.write(str(os.getpid()))
    save(os.path.join(wave, 'controller.process.json'),
         {'pid': os.getpid(), 'identity': GPU.identity(os.getpid())})
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    running = {}
    for row in rows:
        st = status_of(wave, row["label"])
        if st.get("status") not in FINAL or st.get("status") == "passed" and not passed(row["task_dir"]):
            set_status(wave, row["label"], status="pending")
    t0 = now()
    log(wave, "wave: %d CPU authors, GPU pool %s, backend %s" %
        (len(rows), cfg["gpus"], cfg["backend"]))
    while not stop["flag"]:
        broker = load(os.path.join(wave, 'broker.json'), {})
        if GPU.identity(broker.get('pid', -1)) != broker.get('identity'):
            raise RuntimeError('GPU broker exited; stopping author lanes')
        for name, started in list(running.items()):
            hb = load(os.path.join(wave, "state", name + ".hb"), {}).get("t", started)
            stale = now() - max(hb, started) > cfg["stale_sec"]
            if backend.alive(name) and not stale:
                continue
            backend.stop(name)
            GPU.drain_owner(wave, name)
            del running[name]
            st = status_of(wave, name)
            if st.get("status") not in FINAL:
                lost = int(st.get("lost") or 0) + 1
                set_status(wave, name, status="pending" if lost <= cfg["relaunch"] else "failed",
                           lost=lost, why="lane %s; %s" %
                           ("went stale" if stale else "exited",
                            "relaunching" if lost <= cfg["relaunch"] else "out of relaunches"))
            log(wave, "lane %s finished -> %s" % (name, status_of(wave, name).get("status")))
        for row in rows:
            name = row["label"]
            st = status_of(wave, name)
            if name in running or st.get("status") != "pending":
                continue
            if len(running) >= cfg["author_workers"]:
                break
            argv = [HERE, "lane", "--wave-dir", wave, "--task", name]
            if backend.start(name, [], argv, cfg["env"]):
                running[name] = now()
                set_status(wave, name, status="running", gpus=[], task_dir=row["task_dir"],
                           started=st.get("started") or now())
                log(wave, "lane %s <- CPU; GPU on demand (cost %s)" % (name, row["cost"]))
        states = {r["label"]: status_of(wave, r["label"]).get("status") for r in rows}
        if not running and all(s in FINAL for s in states.values()):
            break
        if cfg["max_wall_sec"] and now() - t0 > cfg["max_wall_sec"]:
            log(wave, "wall-clock ceiling reached; stopping lanes (state is resumable)")
            break
        time.sleep(cfg["tick_sec"])
    for name in running:
        backend.stop(name)
        GPU.drain_owner(wave, name)
    summary = {r["label"]: status_of(wave, r["label"]) for r in rows}
    save(os.path.join(wave, "summary.json"), {"finished": now(), "seconds": round(now() - t0),
                                             "tasks": summary})
    log(wave, "wave done in %ds: %s" % (now() - t0, {k: v.get("status") for k, v in summary.items()}))
    return 0


# ----------------------------------------------------------------------------- lane
def lane_prompt(wave, row, st, attempt, cfg, deadline):
    tdir = row["task_dir"]
    req = os.path.join(tdir, "run", "requested_kernel.json")
    lines = [
        "You are one lane of a parallel UT wave. You own exactly one task directory and the "
        "writing session for it; other agents own the other tasks, at the same time. Nobody will "
        "answer questions: decide, act, record.", "",
        "TASK            %s" % row["label"],
        "TASK_DIR        %s" % tdir,
        "REQUEST         %s" % req,
        "EVAL_DIR        %s" % cfg["eval_dir"],
        "SKILL_ROOT      %s" % SKILL,
        "SHARED_TOOLS    %s" % os.path.join(SKILL, "shared"),
        "GPUS            allocated only inside GPU commands; authoring is CPU-only",
        "DEADLINE        %s (epoch %d)" % (time.strftime("%F %T", time.localtime(deadline)), deadline),
        "VALIDATION      unittest.py --validation-mode correctness --reuse; no timing measurements",
        "",
    ]
    lines += ["GPU MODE: on-demand. This author container is CPU-only and holds no GPU lease.",
              "For EVERY GPU operation (imports needing GPU, compile, quick, full validation), use:",
              'python3 "$SKILL_ROOT/scripts/ut_wave.py" gpu -- python3 "$TASK_DIR/unittest.py" <flags>',
              "The default request is required_gpu_count=%d; --count N may request more for sharding."
              % row['need'],
              "Use --env K=V for command-specific environment; cwd defaults to TASK_DIR.",
              "Do not nest gpu_lock wrappers; the host broker holds the locks until workers drain.", ""]
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
    cfg = load(os.path.join(wave, 'wave.json'))
    row = next(r for r in cfg['rows'] if r['label'] == a.task)
    with open(os.path.join(row['task_dir'], 'run', '.ut_writer.lock'), 'a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            set_status(wave, row['label'], status='blocked', why='another UT writer owns this task directory')
            return 1
        return lane_run(a)


def lane_run(a):
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
            token = uuid.uuid4().hex
            save(os.path.join(wave, 'state', row['label'] + '.token'), {'token': token, 'task': row['label']})
            env.update(UT_WAVE_DIR=wave, UT_WAVE_OWNER=row['label'], UT_WAVE_TASK=row['label'], UT_WAVE_TOKEN=token)
            try:
                rc = subprocess.call(["timeout", str(left), "bash", cfg["runner"], prompt, tdir, logf],
                                     env=env)
            finally:
                GPU.drain_owner(wave, row['label'])
            blocker = load(os.path.join(tdir, "run", "lane_blocked.json"))
            if rc == 78:
                set_status(wave, row["label"], status="blocked",
                           why="runner precondition unmet (exit 78); see %s" % logf)
                return 0
            if blocker:
                set_status(wave, row["label"], status="blocked", why=blocker.get("reason"))
                return 0
            gpu_ok = GPU.last_command_passed(wave, row['label'])
            if gpu_ok and passed(tdir):
                set_status(wave, row["label"], status="passed", ended=now(), why=None)
                return 0
            verdict, fresh, _ = receipt(tdir)
            st = set_status(wave, row["label"],
                            why="session exited %d; receipt %s%s" % (
                                rc, verdict or "absent", "" if fresh or not verdict else " (stale)"))
        set_status(wave, row["label"], status="failed", ended=now())
        return 1
    finally:
        done.set()


# ----------------------------------------------------------------------------- CLI
def cmd_plan(a, quiet=False):
    eval_dir = os.path.abspath(a.eval_dir)
    workload_gpus = deployment_gpus(eval_dir)
    gpus = parse_gpu_ids(a.gpus, '--gpus') if a.gpus is not None else workload_gpus
    if not gpus:
        raise SystemExit("ut_wave: no --gpus and no `gpus` in serving.json")
    if workload_gpus is not None and not set(gpus).issubset(workload_gpus):
        raise SystemExit('ut_wave: requested GPUs %s exceed workload serving.json allocation %s; '
                         '--gpus may only select a subset' % (gpus, workload_gpus))
    allowed = os.environ.get('GEAK_GPU_ALLOWED')
    if allowed and not set(gpus).issubset(parse_gpu_ids(allowed, 'GEAK_GPU_ALLOWED')):
        raise SystemExit('ut_wave: --gpus exceeds GEAK_GPU_ALLOWED')
    mask = os.environ.get('ROCR_VISIBLE_DEVICES')
    if mask:
        try:
            permitted = {int(x) for x in mask.split(',')}
        except ValueError:
            raise SystemExit('ut_wave: resolve inherited ROCR UUID mask to host physical IDs first')
        if not set(gpus).issubset(permitted):
            raise SystemExit('ut_wave: physical GPU pool exceeds inherited ROCR allocation')
    only = set(a.tasks.split(",")) if a.tasks else None
    rows = task_rows(eval_dir, only)
    previous = load(os.path.join(wave_dir(eval_dir), 'wave.json'), {})
    rows = [r for r in rows if a.force or not passed(r["task_dir"])
            or previous.get('gpu_mode') == 'on-demand'
            and not GPU.last_command_passed(wave_dir(eval_dir), r['label'])]
    big = [r for r in rows if r["need"] > len(gpus)]
    if big:
        raise SystemExit("ut_wave: %s need more GPUs than the wave has (%d)"
                         % ([(r["label"], r["need"]) for r in big], len(gpus)))
    if not quiet:
        print("%d GPU(s): %s" % (len(gpus), gpus))
        for row in rows:
            print("  %-48s %-22s cost %8.1f  GPUs per command %d" %
                  (row["label"], row["family"], row["cost"], row["need"]))
    return eval_dir, gpus, rows


def cmd_start(a):
    if a.author_workers < 1:
        raise SystemExit('ut_wave: --author-workers must be positive')
    if os.environ.get('GEAK_GPU_LOCKED_SET'):
        raise SystemExit('ut_wave: release the outer pool lease before starting an on-demand wave')
    eval_dir, gpus, rows = cmd_plan(a)
    wave = wave_dir(eval_dir)
    os.makedirs(wave, exist_ok=True)
    guard = open(os.path.join(wave, 'controller.lock'), 'a')
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('ut_wave: a controller is already running')
    if os.path.exists(os.path.join(wave, "controller.pid")):
        if controller_alive(wave):
            raise SystemExit("ut_wave: a controller is already running for %s" % wave)
    old_broker = load(os.path.join(wave, 'broker.json'), {})
    if old_broker and GPU.identity(old_broker['pid']) == old_broker['identity']:
        os.kill(old_broker['pid'], signal.SIGTERM)
        while GPU.identity(old_broker['pid']) == old_broker['identity']:
            time.sleep(.2)
    old = load(os.path.join(wave, 'wave.json'))
    if old:
        old_backend = GPU.backend_for(wave, old)
        for row in old['rows']:
            old_backend.stop(row['label'])
            old_backend.stop('timing_' + row['label'])
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
    if a.force:
        for row in rows:
            set_status(wave, row['label'], status='pending', attempts=0, lost=0, started=None, why=None)
    save(os.path.join(wave, "wave.json"), {
        "eval_dir": eval_dir, "gpus": gpus, "rows": rows,
        "backend": a.backend, "image": a.image, "render_base": a.render_base,
        "render_step": a.render_step, "runner": runner, "validation_mode": "correctness",
        "gpu_mode": "on-demand", "author_workers": a.author_workers,
        "container_prefix": default_container_prefix(wave),
        "lock_dir": os.path.abspath(os.environ.get('KERNEL_EXTRACT_LOCK_DIR', '/tmp/team_gpu_locks')),
        "lane_sec": a.lane_sec, "attempts": a.attempts, "relaunch": a.relaunch,
        "stale_sec": a.stale_sec, "tick_sec": a.tick_sec, "max_wall_sec": a.max_wall_sec,
        "env": lane_env(eval_dir, a.env),
        "created": now()})
    if a.foreground:
        return controller(argparse.Namespace(eval_dir=eval_dir))
    out = open(os.path.join(wave, "controller.log"), "a")
    p = subprocess.Popen([sys.executable, HERE, "_controller", "--eval-dir", eval_dir],
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                         pass_fds=(guard.fileno(),))
    out.close()
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
        with open(os.path.join(wave, 'controller.pid')) as f:
            pid = int(f.read())
        current = GPU.identity(pid)
        saved = load(os.path.join(wave, 'controller.process.json'))
        return bool(current and (not saved or current == saved['identity']))
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
        if not controller_alive(wave):
            raise ProcessLookupError()
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
        p.add_argument("--gpus", default=None, help="physical GPU IDs; default workload serving.json `gpus`; may only narrow that set")
        p.add_argument("--tasks", default=None, help="comma list of task labels (default all)")
        p.add_argument("--force", action="store_true", help="include tasks that already pass")
        if name == "start":
            p.add_argument("--runner", required=True, help="runners/CONTRACT.md runner script")
            p.add_argument('--author-workers', type=int, default=8, help='maximum concurrent CPU authors')
            p.add_argument("--backend", choices=("docker", "local"), default="docker")
            p.add_argument("--image", default=None)
            p.add_argument("--render-base", type=int, default=128)
            p.add_argument("--render-step", type=int, default=8)
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
    p = sub.add_parser("lane")
    p.add_argument("--wave-dir", required=True)
    p.add_argument("--task", required=True)
    p = sub.add_parser("block")
    p.add_argument("--task-dir", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser('gpu', help='submit a GPU command from an on-demand lane')
    p.add_argument('--count', type=int, default=None, help='default required_gpu_count (usually 1)')
    p.add_argument('--timeout', type=float, default=7200, help='seconds including queue wait')
    p.add_argument('--cwd', default=None, help='defaults to TASK_DIR; must stay inside it')
    p.add_argument('--env', action='append', default=[], help='explicit command environment K=V')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cmd_plan(a)
        return 0
    return {"start": cmd_start, "_controller": controller, "status": lambda a: cmd_status(a)[0],
            "stop": cmd_stop, "wait": cmd_wait, "lane": lane,
            "block": cmd_block, 'gpu': GPU.submit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
