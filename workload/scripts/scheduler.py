#!/usr/bin/env python3
"""The optimize stage's scheduler: spread N kernel tasks across M GPUs until the time
budget is exhausted.

It knows nothing about *which optimization method* is used or *which agent* runs it — the
method comes from the skill's prompt.sh and the agent from the runner. All this file
decides is who runs on which GPU, what happens when they die, and how time is accounted.

Every design constraint below was learned the hard way in a previous 200 GPU-hour
campaign:

* **One container per GPU.** Mount exactly one /dev/dri/renderD<128+8*i> into the
  container. An env-only ROCR_VISIBLE_DEVICES does not hold — the agent resets it in its
  own subprocesses, two workers land on the same physical GPU, and both sets of numbers
  are worthless. Physical isolation is the only isolation.
* **The node is the source of truth, not memory.** The supervisor may restart, so each
  tick first **checks** who is still alive (is the container up, is the heartbeat fresh)
  before dispatching anything. No "who I think is running" is maintained.
* **Stale reclamation needs a startup grace period.** Otherwise a worker that was just
  dispatched, and has not yet written its first heartbeat, gets killed every tick and
  never starts.
* **Credit lives on shared storage.** Losing a node costs only the current attempt; the
  task resumes on another GPU from its accumulated seconds rather than from zero.
* **Blacklist bad nodes.** A GPU that fails to start a container N times in a row leaves
  the rotation, so one bad GPU does not eat every task.

Usage:
    python3 scheduler.py --work-root <dir> --tasks-dir <dir> --skill-root <dir> \
                         --runner <runners/x.sh> [--gpus 0,1,...] [--target 172800]
    python3 scheduler.py --work-root <dir> status
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "worker.sh")


def log(work_root, msg):
    line = "[%s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    try:
        os.makedirs(work_root, exist_ok=True)
        with open(os.path.join(work_root, "scheduler.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def sh(cmd, timeout=60):
    """Returns (rc, stdout+stderr). Never raises — the scheduler must not fall over on
    any single point of failure."""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except OSError as e:
        return 127, str(e)


# ---------------------------------------------------------------- tasks and state

def discover_tasks(tasks_dir, order_file=None, max_tasks=None):
    """Task directories are the profile stage's output. Ordering prefers roofline's
    ranking (descending expected end-to-end gain).

    The order matters: with fewer GPUs than kernels, which kernels run first decides
    whether this campaign moves the end-to-end number at all.
    """
    present = sorted(d for d in os.listdir(tasks_dir)
                     if os.path.isdir(os.path.join(tasks_dir, d, "kernel_src")))
    ordered = []
    if order_file and os.path.exists(order_file):
        try:
            data = json.load(open(order_file))
            rows = data.get("kernels") if isinstance(data, dict) else data
            for r in rows or []:
                name = r.get("task") or r.get("kernel") if isinstance(r, dict) else r
                # roofline keys may carry a module prefix; task directories are named
                # after the kernel alone.
                for cand in (name, str(name).rpartition("/")[2] if name else None):
                    if cand in present and cand not in ordered:
                        ordered.append(cand)
                        break
        except (ValueError, OSError, AttributeError):
            pass
    ordered += [t for t in present if t not in ordered]
    if max_tasks:
        ordered = ordered[:max_tasks]
    return ordered


def state_path(work_root, kernel, suffix):
    return os.path.join(work_root, "state", "%s.%s" % (kernel, suffix))


def read_credit(work_root, kernel):
    try:
        return int(open(state_path(work_root, kernel, "credit")).read().strip() or 0)
    except (OSError, ValueError):
        return 0


def read_hb(work_root, kernel):
    try:
        return int(open(state_path(work_root, kernel, "hb")).read().strip() or 0)
    except (OSError, ValueError):
        return 0


def is_done(work_root, kernel, target):
    if os.path.exists(state_path(work_root, kernel, "done")):
        return True
    return read_credit(work_root, kernel) >= target


def is_blocked(work_root, kernel):
    """A runner returning 78 means an unmet precondition. Ten thousand retries give the
    same answer; do not burn budget on them."""
    p = state_path(work_root, kernel, "status.json")
    try:
        return json.load(open(p)).get("status") == "blocked"
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- execution backends

class Backend(object):
    """Abstracts starting, stopping and checking one worker on GPU i.

    The docker backend provides real isolation (use it in production); the local backend
    only sets env vars (for smoke tests, or where docker is unavailable). local is not
    safe concurrent isolation, and says so at startup.
    """

    def __init__(self, work_root, image=None):
        self.work_root = work_root
        self.image = image

    def name_of(self, kernel, gpu):
        return ("wl_opt_g%d_%s" % (gpu, kernel))[:200]

    def alive(self, kernel, gpu):
        raise NotImplementedError

    def start(self, kernel, gpu, env):
        raise NotImplementedError

    def stop(self, kernel, gpu):
        raise NotImplementedError


class LocalBackend(Backend):
    def __init__(self, work_root, image=None):
        Backend.__init__(self, work_root, image)
        self.procs = {}

    def _pidfile(self, kernel, gpu):
        return state_path(self.work_root, kernel, "pid")

    def alive(self, kernel, gpu):
        # Prefer the Popen handle. poll() also reaps the child — without reaping it
        # becomes a zombie, and a zombie is **alive** as far as `kill -0` is concerned, so
        # a worker that was -9'd would be judged still running.
        p = self.procs.get((gpu, kernel))
        if p is not None:
            return p.poll() is None
        # No handle means the scheduler itself restarted and the worker was left behind
        # by its previous life. Fall back to /proc, again counting zombies as dead.
        try:
            pid = int(open(self._pidfile(kernel, gpu)).read().strip())
        except (OSError, ValueError):
            return False
        try:
            with open("/proc/%d/stat" % pid) as f:
                return f.read().split(") ", 1)[1].split()[0] not in ("Z", "X")
        except (OSError, IndexError):
            return False

    def start(self, kernel, gpu, env):
        e = dict(os.environ, **env)
        # The local backend can only pin via env. The agent can override it — which is
        # exactly why production uses docker.
        e["ROCR_VISIBLE_DEVICES"] = str(gpu)
        e["HIP_VISIBLE_DEVICES"] = str(gpu)
        e["GPU_IDS"] = "0"
        logf = os.path.join(self.work_root, "logs", kernel)
        os.makedirs(logf, exist_ok=True)
        out = open(os.path.join(logf, "spawn.log"), "a")
        p = subprocess.Popen(["bash", WORKER, kernel, "gpu%d" % gpu],
                             env=e, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        os.makedirs(os.path.dirname(self._pidfile(kernel, gpu)), exist_ok=True)
        with open(self._pidfile(kernel, gpu), "w") as f:
            f.write(str(p.pid))
        self.procs[(gpu, kernel)] = p
        return True

    def stop(self, kernel, gpu):
        # Kill the whole process group: the worker has a ticker subshell and a runner
        # session beneath it, and killing only the worker leaves orphans that keep
        # crediting time.
        try:
            pid = int(open(self._pidfile(kernel, gpu)).read().strip())
            os.killpg(os.getpgid(pid), 15)
        except (OSError, ValueError):
            pass
        p = self.procs.pop((gpu, kernel), None)
        if p is not None:
            try:
                p.wait(timeout=10)          # reap it; leave no zombie
            except (subprocess.TimeoutExpired, OSError):
                pass


class DockerBackend(Backend):
    def alive(self, kernel, gpu):
        rc, out = sh(["docker", "inspect", "-f", "{{.State.Running}}",
                      self.name_of(kernel, gpu)])
        return rc == 0 and out.strip() == "true"

    def start(self, kernel, gpu, env):
        name = self.name_of(kernel, gpu)
        sh(["docker", "rm", "-f", name])
        render = "/dev/dri/renderD%d" % (128 + 8 * gpu)
        if not os.path.exists(render):
            log(self.work_root, "%s for GPU %d does not exist — this node does not number "
                                "renderD as 128+8i; specify it with "
                                "--render-base/--render-step" % (render, gpu))
            return False
        cmd = ["docker", "run", "-d", "--name", name,
               "--device", "/dev/kfd", "--device", render,
               "--group-add", "video", "--ipc", "host",
               "--security-opt", "seccomp=unconfined",
               "--shm-size", "16g", "-v", "/shared_nfs:/shared_nfs"]
        for k, v in env.items():
            cmd += ["-e", "%s=%s" % (k, v)]
        # The container sees exactly one GPU, so the worker always addresses it as 0.
        cmd += ["-e", "GPU_IDS=0", self.image,
                "bash", WORKER, kernel, "gpu%d" % gpu]
        rc, out = sh(cmd, timeout=180)
        if rc != 0:
            log(self.work_root, "container failed to start gpu=%d kernel=%s: %s"
                % (gpu, kernel, out.strip()[:300]))
            return False
        return True

    def stop(self, kernel, gpu):
        sh(["docker", "rm", "-f", self.name_of(kernel, gpu)])


# ---------------------------------------------------------------- main loop

def tick(args, backend, tasks, assign, bad):
    now = int(time.time())
    target = args.target

    # 1) Check reality: which GPUs still carry a live worker.
    for gpu in list(assign):
        kernel = assign[gpu]
        if is_done(args.work_root, kernel, target):
            log(args.work_root, "%s exhausted its budget (%ds); withdrawing from gpu%d"
                % (kernel, read_credit(args.work_root, kernel), gpu))
            backend.stop(kernel, gpu)
            del assign[gpu]
            continue
        if is_blocked(args.work_root, kernel):
            log(args.work_root, "%s was judged blocked by the runner (unmet precondition); "
                "no further retries" % kernel)
            backend.stop(kernel, gpu)
            del assign[gpu]
            continue

        hb = read_hb(args.work_root, kernel)
        age = now - hb if hb else None
        running = backend.alive(kernel, gpu)
        # Startup grace: a freshly dispatched worker has not written its first heartbeat
        # yet and must not be treated as dead.
        grace = now - args.started_at.get((gpu, kernel), now) < args.startup_grace
        if not running and not grace:
            log(args.work_root, "%s on gpu%d is gone (no container/process); reclaiming"
                % (kernel, gpu))
            backend.stop(kernel, gpu)
            del assign[gpu]
        elif age is not None and age > args.stale_sec and not grace:
            log(args.work_root, "%s on gpu%d has not sent a heartbeat for %ds; declared "
                "dead and reclaimed" % (kernel, gpu, age))
            backend.stop(kernel, gpu)
            del assign[gpu]

    # 2) Dispatch. Tasks that are unfinished, not blocked, and not already running on
    # another GPU, in ranked order.
    # This busy set is the entire deduplication mechanism: one kernel is never dispatched
    # to two GPUs editing the same kernel_src/ at once — their KEEP/revert decisions
    # would trample each other and every measurement would be untrustworthy.
    busy = set(assign.values())
    pending = [t for t in tasks
               if t not in busy
               and not is_done(args.work_root, t, target)
               and not is_blocked(args.work_root, t)]

    for gpu in args.gpus:
        if gpu in assign or bad.get(gpu, 0) >= args.bad_node_limit:
            continue
        if not pending:
            break
        kernel = pending.pop(0)
        env = dict(args.worker_env)
        env["TASK_DIR"] = os.path.join(args.tasks_dir, kernel)
        ok = backend.start(kernel, gpu, env)
        if ok:
            assign[gpu] = kernel
            args.started_at[(gpu, kernel)] = now
            bad[gpu] = 0
            log(args.work_root, "dispatch gpu%d <- %s (credited %ds / %ds)"
                % (gpu, kernel, read_credit(args.work_root, kernel), target))
        else:
            bad[gpu] = bad.get(gpu, 0) + 1
            pending.insert(0, kernel)
            if bad[gpu] >= args.bad_node_limit:
                log(args.work_root, "gpu%d failed to start %d times in a row; blacklisted"
                    % (gpu, bad[gpu]))

    remaining = [t for t in tasks if not is_done(args.work_root, t, target)
                 and not is_blocked(args.work_root, t)]
    return assign, bad, remaining


def cmd_status(work_root, tasks_dir, target):
    sdir = os.path.join(work_root, "state")
    rows = []
    for t in sorted(os.listdir(sdir) if os.path.isdir(sdir) else []):
        if not t.endswith(".status.json"):
            continue
        k = t[: -len(".status.json")]
        try:
            d = json.load(open(os.path.join(work_root, "state", t)))
        except (OSError, ValueError):
            continue
        rows.append((k, d.get("status"), d.get("credited_seconds", 0),
                     d.get("best_speedup"), d.get("attempts", 0), d.get("slot")))
    if not rows:
        print("(no worker status yet)")
        return
    print("%-34s %-9s %9s %9s %8s %s" % ("kernel", "status", "credited", "best", "attempts", "slot"))
    for r in rows:
        print("%-34s %-9s %8ds %9s %8d %s"
              % (r[0], r[1], r[2], ("%.4f" % r[3]) if r[3] else "-", r[4], r[5] or "-"))
    tot = sum(r[2] for r in rows)
    print("\ntotal credited: %.1f GPU-h / target %.1f GPU-h"
          % (tot / 3600.0, len(rows) * target / 3600.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="run", choices=["run", "status", "stop"])
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--tasks-dir")
    ap.add_argument("--skill-root")
    ap.add_argument("--runner")
    ap.add_argument("--order-file", help="roofline.json, ranked by expected e2e gain")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--target", type=int, default=172800)
    ap.add_argument("--attempt-cap", type=int, default=1500)
    # In the worker, ATTEMPT_MIN is the **floor** on the cap (so attempts are not sliced
    # too thin once the remaining budget is small). Without exposing it, a short-cap
    # smoke test gets silently raised back to 600 s.
    ap.add_argument("--attempt-min", type=int, default=600)
    ap.add_argument("--keep-threshold", type=float, default=1.045)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--min-improve", type=float, default=0.05)
    ap.add_argument("--max-tasks", type=int)
    ap.add_argument("--backend", choices=["local", "docker"], default="local")
    ap.add_argument("--image")
    ap.add_argument("--stale-sec", type=int, default=900)
    ap.add_argument("--startup-grace", type=int, default=300)
    ap.add_argument("--bad-node-limit", type=int, default=2)
    ap.add_argument("--tick-sec", type=int, default=30)
    ap.add_argument("--max-wall-sec", type=int, default=0,
                    help="wall-clock ceiling for the scheduler itself; 0 = unlimited")
    args = ap.parse_args()

    args.work_root = os.path.abspath(args.work_root)
    if args.command == "status":
        cmd_status(args.work_root, args.tasks_dir, args.target)
        return 0

    for need in ("tasks_dir", "skill_root", "runner"):
        if not getattr(args, need):
            ap.error("run requires --%s" % need.replace("_", "-"))
    args.tasks_dir = os.path.abspath(args.tasks_dir)
    args.skill_root = os.path.abspath(args.skill_root)
    args.runner = os.path.abspath(args.runner)

    if not os.path.exists(os.path.join(args.skill_root, "prompt.sh")):
        print("the selected optimize skill has no prompt.sh: %s" % args.skill_root,
              file=sys.stderr)
        return 2

    args.gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""] or [0]
    for d in ("state", "logs", "runs"):
        os.makedirs(os.path.join(args.work_root, d), exist_ok=True)

    tasks = discover_tasks(args.tasks_dir, args.order_file, args.max_tasks)
    if not tasks:
        print("no task directory containing kernel_src/ was found under %s" % args.tasks_dir,
              file=sys.stderr)
        return 2

    backend = (DockerBackend(args.work_root, args.image) if args.backend == "docker"
               else LocalBackend(args.work_root))
    if args.backend == "docker" and not args.image:
        print("--backend docker requires --image", file=sys.stderr)
        return 2
    if args.backend == "local" and len(args.gpus) > 1:
        log(args.work_root, "note: the local backend can only pin GPUs via env, which the "
                            "agent can override. For multi-GPU parallelism use "
                            "--backend docker (one container per GPU).")

    # Environment handed to the worker. No credentials here — the runner reads those
    # itself from WORKLOAD_AGENT_ENV_FILE.
    args.worker_env = {
        "WORK_ROOT": args.work_root,
        "SKILL_ROOT": args.skill_root,
        "RUNNER": args.runner,
        "TARGET": str(args.target),
        "ATTEMPT_CAP": str(args.attempt_cap),
        "ATTEMPT_MIN": str(min(args.attempt_min, args.attempt_cap)),
        "KEEP_THRESHOLD": str(args.keep_threshold),
        "BUDGET": str(args.budget),
        "MIN_IMPROVE": str(args.min_improve),
    }
    for k in ("GEAK_ROOT", "WORKLOAD_AGENT_ENV_FILE", "WORKLOAD_AGENT_STATE_BASE",
              "WORKLOAD_AGENT_MODEL", "EXTRA_STEER"):
        if os.environ.get(k):
            args.worker_env[k] = os.environ[k]

    args.started_at = {}
    assign, bad = {}, {}

    if args.command == "stop":
        for gpu in args.gpus:
            for t in tasks:
                backend.stop(t, gpu)
        log(args.work_root, "all workers stopped")
        return 0

    log(args.work_root, "starting: %d tasks / %d GPUs / %ds per task / backend=%s / "
        "skill=%s / runner=%s"
        % (len(tasks), len(args.gpus), args.target, args.backend,
           os.path.basename(args.skill_root), os.path.basename(args.runner)))
    log(args.work_root, "task order: %s" % ", ".join(tasks))

    t0 = time.time()
    try:
        while True:
            assign, bad, remaining = tick(args, backend, tasks, assign, bad)
            if not remaining:
                log(args.work_root, "every task is finished or blocked — scheduling ends")
                break
            if all(bad.get(g, 0) >= args.bad_node_limit for g in args.gpus):
                log(args.work_root, "every GPU is blacklisted; stopping for a human to look")
                return 3
            if args.max_wall_sec and time.time() - t0 > args.max_wall_sec:
                log(args.work_root, "reached the scheduler wall-clock ceiling; exiting "
                    "(worker credit is on disk and can be resumed)")
                break
            time.sleep(args.tick_sec)
    except KeyboardInterrupt:
        log(args.work_root, "interrupted; stopping all workers (credit is on disk, a rerun "
            "resumes)")
        for gpu, kernel in list(assign.items()):
            backend.stop(kernel, gpu)
        return 130

    cmd_status(args.work_root, args.tasks_dir, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
