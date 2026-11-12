#!/usr/bin/env python3
"""The optimize stage's scheduler: spread N kernel tasks across M GPUs until the time
budget is exhausted.

It knows nothing about *which optimization method* is used or *which agent* runs it — the
method comes from the skill's prompt.sh and the agent from the runner. All this file
decides is who runs on which GPU, what happens when they die, and how time is accounted.

Every design constraint below was learned the hard way in a previous 200 GPU-hour
campaign:

* **One container per slot.** A slot is as many consecutive GPUs as the task declared in
  its `meta.json` (`required_gpus`, defaulting to `--gpus-per-task`); mount exactly that
  group's /dev/dri/renderD<128+8*i> nodes into the container. Widths are per task, not
  per campaign: a single-rank elementwise kernel and a TP8/EP8 collective share one run. An env-only ROCR_VISIBLE_DEVICES does not hold — the agent resets it in its
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
import argparse, fcntl, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "worker.sh")
sys.path.insert(0, HERE)
from durable_write import read_credit_pair


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


def slot_label(gpus):
    """Stable name for a slot, used for container names, logs and state keys."""
    return ("gpu%d" % gpus[0]) if len(gpus) == 1 else ("gpus%d-%d" % (gpus[0], gpus[-1]))


def task_gpus(tasks_dir, task, default=1):
    """How many GPUs this task needs, as the task itself declares in `meta.json`.

    A campaign is not one size. A single-rank elementwise kernel needs one GPU; a
    collective that launches its own torchrun across TP8/EP8 needs all eight or it fails
    on every attempt. When the scheduler took one `--gpus-per-task` for the whole run,
    mixing the two was impossible, and a second 666-line scheduler grew beside this one
    to group tasks by requirement before calling it once per group. The requirement was
    already in `meta.json`; reading it here is the general rule, and the second scheduler
    is gone.

    `--gpus-per-task` remains the default for a task that declares nothing.
    """
    try:
        with open(os.path.join(tasks_dir, task, "meta.json")) as f:
            declared = json.load(f).get("required_gpus")
    except (OSError, ValueError, AttributeError):
        declared = None
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        return max(1, int(default))
    return declared


def allocate(free, n):
    """Take `n` consecutive GPUs out of `free`, aligned to a multiple of n.

    Consecutive and aligned because the container mounts /dev/dri/renderD<128+8*i> for
    exactly this group, and a collective's rank numbering follows the group order. A task
    is never handed fewer ranks than it asked for: that does not run slowly, it fails
    every attempt.
    """
    for i in range(0, len(free) - n + 1):
        block = free[i:i + n]
        if block[0] % n == 0 and block[-1] - block[0] == n - 1:
            return block
    return None


def mount_point(path):
    """The filesystem `path` lives on."""
    p = os.path.abspath(path)
    while p != "/" and not os.path.ismount(p):
        p = os.path.dirname(p)
    return p


def default_mounts(*paths):
    """Which host filesystems the worker container has to see, derived, not named.

    The worker reads the task tree, the skill and the run root from inside the container
    at the same absolute paths the scheduler was given, so those filesystems must be
    mounted. This used to be one literal `-v /shared_nfs:/shared_nfs`, which is the name
    of one site's NFS export written into the tool (PRINCIPLES R4): correct here, wrong
    on the next machine, and invisible until a container started and found an empty
    directory. Mounting the mount point of each path the run was actually configured with
    computes `/shared_nfs` on this node and the right answer elsewhere. `/` is never
    mounted wholesale; pass `--mount` for anything outside the configured tree.
    """
    found = []
    for path in paths:
        if not path:
            continue
        point = mount_point(path)
        if point != "/" and point not in found:
            found.append(point)
    return found


def recorded_slot(work_root, kernel):
    """The GPU group a task was last dispatched to, from its own state file.

    The node is the source of truth, not the supervisor's memory -- and with per-task
    widths there is no fixed slot table to sweep after a restart, so the task says where
    it was and the backend says whether it is still there."""
    try:
        with open(os.path.join(work_root, "state", kernel + ".status.json")) as f:
            label = json.load(f).get("slot")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(label, str):
        return None
    try:
        if label.startswith("gpus"):
            lo, hi = label[4:].split("-")
            return list(range(int(lo), int(hi) + 1))
        if label.startswith("gpu"):
            return [int(label[3:])]
    except ValueError:
        return None
    return None


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

def task_shares(tasks_dir):
    """{task: % of merged GPU time} from the profile stage's results.json (sibling of the
    tasks directory), or {} when it is absent or unreadable."""
    try:
        doc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(tasks_dir)),
                                          "results.json")))
        return {r["task"]: float(r["selection_pct_gpu_time"]) for r in doc.get("tasks", [])
                if r.get("task") and r.get("selection_pct_gpu_time") is not None}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def discover_tasks(tasks_dir, order_file=None, max_tasks=None, exclude=(), order_by="share"):
    """Task directories are the profile stage's output.

    order_by="share" (default): descending share of the profiled GPU time, read from the
    profile stage's results.json. A roofline model that cannot model a task leaves its
    expected gain null, and ranking by gain then put the largest task of a campaign
    (24% of GPU time, unmodelled) behind eight tasks worth 35% together. The share is
    measured for every task. Falls back to the roofline order when no share is known.
    order_by="roofline": descending expected end-to-end gain from --order-file.

    The order matters: with fewer GPUs than kernels, which kernels run first decides
    whether this campaign moves the end-to-end number at all.
    """
    present = sorted(d for d in os.listdir(tasks_dir)
                     if os.path.isdir(os.path.join(tasks_dir, d, "kernel_src"))
                     and d not in set(exclude))
    ordered = []
    share = task_shares(tasks_dir) if order_by == "share" else {}
    if share:
        ordered = sorted((t for t in present if t in share), key=lambda t: (-share[t], t))
    elif order_file and os.path.exists(order_file):
        try:
            data = json.load(open(order_file))
            rows = data.get("kernels") if isinstance(data, dict) else data
            score, seen = {}, []
            for r in rows or []:
                if isinstance(r, dict):
                    name = r.get("task") or r.get("task_dir") or r.get("kernel") or r.get("case_id")
                    gain = r.get("expected_e2e_gain_pct")
                else:
                    name, gain = r, None
                t = match_task(name, present)
                if not t:
                    continue
                if t not in seen:
                    seen.append(t)
                if gain is not None:
                    score[t] = score.get(t, 0.0) + float(gain)
            # Rank by summed expected e2e gain where roofline gave one; otherwise keep
            # the order the ranking file listed them in.
            ordered = sorted(seen, key=lambda t: (-score.get(t, 0.0), seen.index(t)))
        except (ValueError, OSError, AttributeError, TypeError):
            ordered = []
    ordered += [t for t in present if t not in ordered]
    if max_tasks:
        ordered = ordered[:max_tasks]
    return ordered


def match_task(name, tasks):
    """Map a ranking row's kernel name onto a task directory.

    The two namespaces never agree exactly: the profiler names a kernel by its shape
    variant (megamoe_prepare_m128, ep_combine_intranode_0) and the extraction names a
    directory after the boundary (megamoe_prepare_compact_dispatch, ep_combine_intranode).
    Leading underscore-tokens are what they share. Two must agree, so
    megamoe_prepare_m128 cannot drift onto megamoe_stage2_*, and the longest agreement
    wins.
    """
    if not name:
        return None
    base = str(name).rpartition("/")[2].split(".")[0]
    if base in tasks:
        return base
    toks = base.split("_")
    best, best_n = None, 1
    for t in tasks:
        tt = t.split("_")
        k = 0
        while k < min(len(toks), len(tt)) and toks[k] == tt[k]:
            k += 1
        if k > best_n:
            best, best_n = t, k
    return best


def state_path(work_root, kernel, suffix):
    return os.path.join(work_root, "state", "%s.%s" % (kernel, suffix))


def read_credit(work_root, kernel):
    return read_credit_pair(state_path(work_root, kernel, "credit"),
                            state_path(work_root, kernel, "status.json"))


def read_hb(work_root, kernel):
    try:
        return int(open(state_path(work_root, kernel, "hb")).read().strip() or 0)
    except (OSError, ValueError):
        return 0


def is_done(work_root, kernel, target, require_full_budget=False):
    if require_full_budget:
        try:
            with open(state_path(work_root, kernel, "status.json")) as f:
                status = json.load(f)
            return status.get("status") == "done" and status.get("credited_seconds", 0) >= target
        except (OSError, ValueError, TypeError):
            return False
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

    def __init__(self, work_root, image=None, retain_containers=False, mounts=()):
        self.work_root = work_root
        self.image = image
        self.retain_containers = retain_containers
        self.mounts = list(mounts)

    def name_of(self, kernel, gpus):
        return ("wl_opt_%s_%s" % (slot_label(gpus), kernel))[:200]

    def alive(self, kernel, gpus):
        raise NotImplementedError

    def start(self, kernel, gpus, env):
        raise NotImplementedError

    def stop(self, kernel, gpus):
        raise NotImplementedError


class LocalBackend(Backend):
    def __init__(self, work_root, image=None):
        Backend.__init__(self, work_root, image)
        self.procs = {}

    def _pidfile(self, kernel, gpus):
        return state_path(self.work_root, kernel, "pid")

    def alive(self, kernel, gpus):
        # Prefer the Popen handle. poll() also reaps the child — without reaping it
        # becomes a zombie, and a zombie is **alive** as far as `kill -0` is concerned, so
        # a worker that was -9'd would be judged still running.
        p = self.procs.get((slot_label(gpus), kernel))
        if p is not None:
            return p.poll() is None
        # No handle means the scheduler itself restarted and the worker was left behind
        # by its previous life. Fall back to /proc, again counting zombies as dead.
        try:
            pid = int(open(self._pidfile(kernel, gpus)).read().strip())
        except (OSError, ValueError):
            return False
        try:
            with open("/proc/%d/stat" % pid) as f:
                return f.read().split(") ", 1)[1].split()[0] not in ("Z", "X")
        except (OSError, IndexError):
            return False

    def start(self, kernel, gpus, env):
        e = dict(os.environ, **env)
        # The local backend can only pin via env. The agent can override it — which is
        # exactly why production uses docker.
        e["ROCR_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        e["HIP_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        e["GPU_IDS"] = ",".join(str(i) for i in range(len(gpus)))
        logf = os.path.join(self.work_root, "logs", kernel)
        os.makedirs(logf, exist_ok=True)
        out = open(os.path.join(logf, "spawn.log"), "a")
        p = subprocess.Popen(["bash", WORKER, kernel, slot_label(gpus)],
                             env=e, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        os.makedirs(os.path.dirname(self._pidfile(kernel, gpus)), exist_ok=True)
        with open(self._pidfile(kernel, gpus), "w") as f:
            f.write(str(p.pid))
        self.procs[(slot_label(gpus), kernel)] = p
        return True

    def stop(self, kernel, gpus):
        # Kill the whole process group: the worker has a ticker subshell and a runner
        # session beneath it, and killing only the worker leaves orphans that keep
        # crediting time.
        try:
            pid = int(open(self._pidfile(kernel, gpus)).read().strip())
            os.killpg(os.getpgid(pid), 15)
        except (OSError, ValueError):
            pass
        p = self.procs.pop((slot_label(gpus), kernel), None)
        if p is not None:
            try:
                p.wait(timeout=10)          # reap it; leave no zombie
            except (subprocess.TimeoutExpired, OSError):
                pass


class DockerBackend(Backend):
    def alive(self, kernel, gpus):
        rc, out = sh(["docker", "inspect", "-f", "{{.State.Running}}",
                      self.name_of(kernel, gpus)])
        return rc == 0 and out.strip() == "true"

    def start(self, kernel, gpus, env):
        name = self.name_of(kernel, gpus)
        if self.retain_containers:
            rc, observed = sh(["docker", "inspect", "-f", "{{.Id}}", name])
            if rc == 0:
                log(self.work_root, "retained container %s requires preservation and explicit cleanup before dispatch" % name)
                return False
            if not any(message in observed.lower() for message in ("no such object:", "no such container:")):
                log(self.work_root, "cannot prove retained name %s absent; refusing dispatch" % name)
                return False
        else:
            sh(["docker", "rm", "-f", name])
        cmd = ["docker", "run", "-d", "--name", name,
               "--device", "/dev/kfd",
               "--group-add", "video", "--ipc", "host",
               "--security-opt", "seccomp=unconfined",
               "--shm-size", "16g"]
        for host in self.mounts:
            cmd += ["-v", "%s:%s" % (host, host)]
        from container_runtime import runtime_options
        runtime_args, entry_prefix = runtime_options(name)
        cmd += runtime_args
        for gpu in gpus:
            render = "/dev/dri/renderD%d" % (128 + 8 * gpu)
            if not os.path.exists(render):
                log(self.work_root, "%s for GPU %d does not exist — this node does not "
                                    "number renderD as 128+8i; specify it with "
                                    "--render-base/--render-step" % (render, gpu))
                return False
            cmd += ["--device", render]
        for k, v in env.items():
            cmd += ["-e", "%s=%s" % (k, v)]
        # The container sees exactly this slot's GPUs, so the worker addresses them as
        # 0..N-1 whatever their physical indices are.
        logical_ids = ",".join(str(i) for i in range(len(gpus)))
        render_nodes = ",".join("renderD%d" % (128 + 8 * gpu) for gpu in sorted(gpus))
        cmd += ["-e", "GPU_IDS=" + logical_ids,
                "-e", "HIP_VISIBLE_DEVICES=" + logical_ids,
                "-e", "GEAK_GPU_ALLOWED=" + logical_ids,
                "-e", "GEAK_GPU_RENDER_DEVICES=" + render_nodes, self.image,
                *entry_prefix, "bash", WORKER, kernel, slot_label(gpus)]
        rc, out = sh(cmd, timeout=180)
        if rc != 0:
            log(self.work_root, "container failed to start slot=%s kernel=%s: %s"
                % (slot_label(gpus), kernel, out.strip()[:300]))
            return False
        return True

    def stop(self, kernel, gpus):
        if self.retain_containers:
            sh(["docker", "stop", "--timeout", "20", self.name_of(kernel, gpus)])
        else:
            sh(["docker", "rm", "-f", self.name_of(kernel, gpus)])


def measuring_since(work_root, kernel):
    """Seconds since the worker marked a sweep in flight, or None if it is not."""
    p = os.path.join(work_root, "state", kernel + ".measuring")
    try:
        return max(0, int(time.time() - os.path.getmtime(p)))
    except OSError:
        return None


# ---------------------------------------------------------------- main loop

def tick(args, backend, tasks, assign, bad):
    now = int(time.time())
    target = args.target
    full = getattr(args, "require_full_budget", False)

    # Recover live workers after a supervisor restart before considering new work.
    # A heartbeat alone is not a worker: confirm the container/process still exists.
    if not getattr(args, "reconciled_workers", False):
        # Slots are allocated per task now, so there is no fixed list to sweep. Each
        # task's own state file records the slot it was last dispatched to; confirm the
        # container is still up before believing it.
        for kernel in tasks:
            gpus = recorded_slot(args.work_root, kernel)
            if not gpus or kernel in assign.values():
                continue
            if backend.alive(kernel, gpus):
                label = slot_label(gpus)
                assign[label] = kernel
                args.slot_of[label] = gpus
                args.started_at[(label, kernel)] = now
        args.reconciled_workers = True

    # 1) Check reality: which slots still carry a live worker.
    for label in list(assign):
        kernel = assign[label]
        gpus = args.slot_of[label]
        if is_done(args.work_root, kernel, target, full):
            # Exhaustion is a between-attempts decision. If the worker is inside a sweep,
            # the agent's time is already spent and only the verdict is outstanding; let
            # it land. The mark ages out on its own, so a worker that died mid-sweep
            # releases the slot on a later tick instead of holding it.
            busy = measuring_since(args.work_root, kernel)
            if busy is not None and busy <= args.measure_timeout + 300:
                log(args.work_root, "%s is over budget but has been measuring for %ds; "
                    "deferring the withdrawal until the sweep returns" % (kernel, busy))
                continue
            log(args.work_root, "%s exhausted its budget (%ds); withdrawing from %s"
                % (kernel, read_credit(args.work_root, kernel), label))
            backend.stop(kernel, gpus)
            del assign[label]
            continue
        if is_blocked(args.work_root, kernel):
            log(args.work_root, "%s was judged blocked by the runner (unmet precondition); "
                "no further retries" % kernel)
            backend.stop(kernel, gpus)
            del assign[label]
            continue

        hb = read_hb(args.work_root, kernel)
        age = now - hb if hb else None
        running = backend.alive(kernel, gpus)
        # Startup grace: a freshly dispatched worker has not written its first heartbeat
        # yet and must not be treated as dead.
        grace = now - args.started_at.get((label, kernel), now) < args.startup_grace
        if not running and not grace:
            log(args.work_root, "%s on %s is gone (no container/process); reclaiming"
                % (kernel, label))
            backend.stop(kernel, gpus)
            del assign[label]
        elif age is not None and age > args.stale_sec and not grace:
            log(args.work_root, "%s on %s has not sent a heartbeat for %ds; declared "
                "dead and reclaimed" % (kernel, label, age))
            backend.stop(kernel, gpus)
            del assign[label]

    # 2) Dispatch. Tasks that are unfinished, not blocked, and not already running on
    # another GPU, in ranked order.
    # This busy set is the entire deduplication mechanism: one kernel is never dispatched
    # to two GPUs editing the same kernel_src/ at once — their KEEP/revert decisions
    # would trample each other and every measurement would be untrustworthy.
    busy = set(assign.values())
    pending = [t for t in tasks
               if t not in busy
               and not is_done(args.work_root, t, target, full)
               and not is_blocked(args.work_root, t)]

    # Each task takes the width it declared. Blacklisting is per GPU rather than per
    # slot, because a GPU that cannot start a container is bad in every grouping it
    # appears in.
    held = {g for label in assign for g in args.slot_of[label]}
    free = [g for g in args.gpus
            if g not in held and bad.get(g, 0) < args.bad_node_limit]

    for kernel in list(pending):
        if not free:
            break
        width = args.width[kernel]
        gpus = allocate(free, width)
        if gpus is None:
            # Not a failure: a wider task waits for its ranks to come free while
            # narrower ones keep the machine busy.
            continue
        label = slot_label(gpus)
        env = dict(args.worker_env)
        env["TASK_DIR"] = os.path.join(args.tasks_dir, kernel)
        if kernel in args.keep_overrides:
            env["KEEP_THRESHOLD"] = str(args.keep_overrides[kernel])
        ok = backend.start(kernel, gpus, env)
        pending.remove(kernel)
        if ok:
            assign[label] = kernel
            args.slot_of[label] = gpus
            args.started_at[(label, kernel)] = now
            for g in gpus:
                bad[g] = 0
            free = [g for g in free if g not in gpus]
            log(args.work_root, "dispatch %s <- %s (%d GPU(s), credited %ds / %ds)"
                % (label, kernel, width, read_credit(args.work_root, kernel), target))
        else:
            for g in gpus:
                bad[g] = bad.get(g, 0) + 1
                if bad[g] >= args.bad_node_limit:
                    log(args.work_root, "gpu%d failed to start %d times in a row; "
                        "blacklisted" % (g, bad[g]))
            free = [g for g in free if bad.get(g, 0) < args.bad_node_limit]

    remaining = [t for t in tasks if not is_done(args.work_root, t, target, full)
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
    ap.add_argument("--order-file", help="roofline.json, ranked by expected e2e gain "
                                         "(used with --order-by roofline, or when no "
                                         "GPU share is known)")
    ap.add_argument("--order-by", choices=("share", "roofline"), default="share",
                    help="task order: share of profiled GPU time (default, from "
                         "profile/out/results.json) or roofline's expected gain")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--gpus-per-task", type=int, default=1,
                    help="default width for a task whose meta.json does not declare "
                         "required_gpus. Each task's own declaration wins: a campaign "
                         "mixes single-rank kernels with collectives that need the whole "
                         "group, and one number for the whole run cannot express that.")
    ap.add_argument("--exclude", default="",
                    help="comma-separated task directory names to leave out of this run")
    ap.add_argument("--target", type=int, default=172800)
    ap.add_argument("--require-full-budget", action="store_true",
                    help="keep the requested per-task budget and finish the final bounded attempt")
    ap.add_argument("--attempt-cap", type=int, default=1500)
    ap.add_argument("--measure-timeout", type=int, default=1800,
                    help="ceiling on the worker's own post-attempt measurement. This is "
                         "not the attempt cap: the cap bounds the agent, this bounds the "
                         "frozen harness re-run that decides keep vs revert. A collective "
                         "sweep over 13 buckets took 1282 s on this stack.")
    # In the worker, ATTEMPT_MIN is the **floor** on the cap (so attempts are not sliced
    # too thin once the remaining budget is small). Without exposing it, a short-cap
    # smoke test gets silently raised back to 600 s.
    ap.add_argument("--attempt-min", type=int, default=600)
    ap.add_argument("--keep-threshold", type=float, default=1.045)
    ap.add_argument("--keep-threshold-override", default="", help='comma-separated task=threshold overrides. The noise band belongs to the seam: a collective kernel that barriers across 8 ranks has a far wider null spread than a single-rank one, and one global band is wrong for both.')
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--min-improve", type=float, default=0.05)
    ap.add_argument("--max-tasks", type=int)
    ap.add_argument("--backend", choices=["local", "docker"], default="local")
    ap.add_argument("--image")
    ap.add_argument("--evidence-root", help="absolute optional destination root for newly begun evidence archives")
    ap.add_argument("--mount", action="append", default=[],
                    help="host path to bind-mount into each worker container at the same "
                         "absolute path, repeatable. Without it the scheduler mounts the "
                         "filesystem each configured path (work root, tasks, skill, "
                         "evidence root, GEAK root) lives on, so no site path is written "
                         "into this tool.")
    ap.add_argument("--retain-containers", action="store_true",
                    help="stop owned containers without removal; existing retained names require explicit cleanup")
    ap.add_argument("--stale-sec", type=int, default=900)
    ap.add_argument("--startup-grace", type=int, default=300)
    ap.add_argument("--bad-node-limit", type=int, default=2)
    ap.add_argument("--tick-sec", type=int, default=30)
    ap.add_argument("--max-wall-sec", type=int, default=0,
                    help="wall-clock ceiling for the scheduler itself; 0 = unlimited")
    args = ap.parse_args()
    if args.evidence_root is not None and not os.path.isabs(args.evidence_root):
        print("--evidence-root must be resolved against the repository before dispatch", file=sys.stderr)
        return 78

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

    # A live orphaned scheduler after coordinator SIGKILL still owns its workers.
    # Keep this descriptor alive until main returns; a peer must not replace them.
    scheduler_lock = open(os.path.join(args.work_root, "scheduler.lock"), "a")
    try:
        fcntl.flock(scheduler_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        scheduler_lock.close()
        print("another scheduler owns this work root", file=sys.stderr)
        return 78

    args.slot_of = {}
    if not args.gpus:
        print("no GPUs were given", file=sys.stderr)
        return 2

    args.keep_overrides = parse_overrides(args.keep_threshold_override)
    for k, v in sorted(args.keep_overrides.items()):
        log(args.work_root, "keep threshold for %s is %.4f, not the global %.4f"
            % (k, v, args.keep_threshold))

    excluded = [t for t in args.exclude.split(",") if t.strip()]
    tasks = discover_tasks(args.tasks_dir, args.order_file, args.max_tasks, excluded,
                           args.order_by)
    if excluded:
        log(args.work_root, "excluded by configuration: %s" % ", ".join(excluded))
    if not tasks:
        print("no task directory containing kernel_src/ was found under %s" % args.tasks_dir,
              file=sys.stderr)
        return 2

    if args.retain_containers and args.backend != "docker":
        print("--retain-containers requires the docker backend", file=sys.stderr)
        return 78
    if args.backend == "docker" and not args.image:
        print("--backend docker requires --image", file=sys.stderr)
        return 2
    mounts = args.mount or default_mounts(args.work_root, args.tasks_dir, args.skill_root,
                                          args.evidence_root, os.environ.get("GEAK_ROOT"))
    backend = (DockerBackend(args.work_root, args.image, args.retain_containers, mounts)
               if args.backend == "docker" else LocalBackend(args.work_root))
    if args.backend == "docker":
        log(args.work_root, "worker containers mount: %s" % (", ".join(mounts) or "(none)"))
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
        "REQUIRE_FULL_BUDGET": "1" if args.require_full_budget else "0",
        "ATTEMPT_CAP": str(args.attempt_cap),
        "MEASURE_TIMEOUT": str(args.measure_timeout),
        "ATTEMPT_MIN": str(min(args.attempt_min, args.attempt_cap)),
        "KEEP_THRESHOLD": str(args.keep_threshold),
        "BUDGET": str(args.budget),
        "MIN_IMPROVE": str(args.min_improve),
    }
    for k in ("GEAK_ROOT", "WORKLOAD_AGENT_ENV_FILE", "WORKLOAD_AGENT_STATE_BASE",
              "WORKLOAD_AGENT_MODEL", "EXTRA_STEER"):
        if os.environ.get(k):
            args.worker_env[k] = os.environ[k]
    if args.evidence_root is not None:
        args.worker_env['WORKLOAD_EVIDENCE_ROOT'] = args.evidence_root

    args.started_at = {}
    assign, bad = {}, {}

    # Each task declares how many GPUs it needs; the default covers tasks that do not.
    args.width = {t: task_gpus(args.tasks_dir, t, args.gpus_per_task) for t in tasks}
    too_wide = sorted(t for t in tasks if args.width[t] > len(args.gpus))
    if too_wide:
        print("%s need(s) more GPUs than the %d given: %s"
              % (", ".join(too_wide), len(args.gpus),
                 ", ".join("%s=%d" % (t, args.width[t]) for t in too_wide)),
              file=sys.stderr)
        return 2

    if args.command == "stop":
        for t in tasks:
            gpus = recorded_slot(args.work_root, t)
            if gpus:
                backend.stop(t, gpus)
        log(args.work_root, "all workers stopped")
        return 0

    widths = sorted({args.width[t] for t in tasks})
    log(args.work_root, "starting: %d tasks over %d GPU(s), widths %s / %ds per task / "
        "backend=%s / skill=%s / runner=%s"
        % (len(tasks), len(args.gpus), ",".join("%dx" % w for w in widths), args.target,
           args.backend, os.path.basename(args.skill_root),
           os.path.basename(args.runner)))
    log(args.work_root, "task order: %s"
        % ", ".join("%s(%dx)" % (t, args.width[t]) for t in tasks))

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
        for label, kernel in list(assign.items()):
            backend.stop(kernel, args.slot_of[label])
        return 130

    cmd_status(args.work_root, args.tasks_dir, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
