#!/usr/bin/env python3
"""Own the probe server for the profile stage, from a declared serving.json.

    python3 probe_server.py init --out serving.json            # a template to fill in
    python3 probe_server.py run --serving serving.json --run-dir "$EVAL_DIR/probe" -- \
        python3 run_points.py --server-state "$EVAL_DIR/probe/probe_server.json" ...
    python3 probe_server.py start|stop --serving serving.json --run-dir ...   # by hand

`run` keeps the server in the foreground for the whole lifecycle the skill requires:
start it in its own process group, wait for /health (stopping at once if the process
exits), write probe_server.json and server_info.json, run the command, then stop the
group and wait for it to exit -- also on failure or a signal.

serving.json declares the deployment and nothing else is guessed:

  model, tp, gpus, host, port      the served model and the device set
  args                             the deployment's server flags, as a list
  env                              the deployment's environment
  pythonpath_prepend               e.g. the editable sglang checkout in the image
  launcher                         default ["python3", "-m", "sglang.launch_server"]
  vocab_size                       optional; default read from the model's config.json.
                                   Probe prompts are random ids below it (an id past the
                                   embedding table faults the GPU)
  compile_watch                    default true: record every Triton compile variant and the
                                   call that built it, from process start (warmup included),
                                   into <run-dir>/compile_variants.json (shared/compile_watch.py)
  compile_watch_frames             non-Triton frames whose arguments are kept per new variant
                                   (default 12; 0 = identity and stack only)
  decode_capture                   how decode kernels are made visible to the profiler:
     "profile_cuda_graph" (default)  keep the CUDA graph and add --enable-profile-cuda-graph,
                                     so decode runs as deployed and the profiler sees the
                                     replayed kernels (verified on sglang 0.5.19)
     "eager"                         --disable-cuda-graph: kernels visible, but launch gaps
                                     make every rank spin in the next all-reduce; pair with
                                     run_points.py --decode-eager
     "as_declared"                   add nothing

If the server rejects --enable-profile-cuda-graph, `run` restarts it eagerly. If the
command is run_points.py and a decode point comes back graph_hidden, `run` restarts
eagerly and runs the command again; run_points.py resumes and recaptures only what is
missing. probe_server.json records the mode that was actually used.

The first server started is the deployment's; its compile variants are also written to
<eval-dir>/evidence/compile_variants.json when the command names an --eval-dir. Variants of
an eager restart are kept beside it (compile_variants.<n>.eager.json) but are not the
deployment's: eager decode compiles kernels a graph-deployed server never launches eagerly.

GPU_ARCHS is pinned (detected once with rocminfo) unless serving.json sets it: an unset
GPU_ARCHS lets aiter's JIT enumerate every arch and storm the node with compiler processes.
Stdlib only.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(os.path.dirname(HERE), "shared")
sys.path.insert(0, SHARED)
import compile_watch  # noqa: E402

TEMPLATE = {
    "model": "/path/to/model",
    "tp": 8,
    "gpus": "0,1,2,3,4,5,6,7",
    "host": "127.0.0.1",
    "port": 31677,
    "launcher": ["python3", "-m", "sglang.launch_server"],
    "args": ["--trust-remote-code", "--enable-metrics", "--enable-cache-report",
             "--watchdog-timeout", "1800"],
    "env": {},
    "pythonpath_prepend": "",
    "decode_capture": "profile_cuda_graph",
    "compile_watch": True,
    "_note": "args/env: copy the deployment's flags and environment exactly (baseline "
             "environment.json / launch script). The probe adds only the profiler "
             "directory, --enable-profile-cuda-graph or --disable-cuda-graph.",
}
# Chains to the next sitecustomize on sys.path: only the first one is imported, and the
# deployment may ship its own (an accepted-kernel overlay in pythonpath_prepend).
WATCH_SITECUSTOMIZE = r'''# probe overlay written by probe_server.py; drop this dir from PYTHONPATH to revert.
import importlib.abc, importlib.machinery, importlib.util, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("COMPILE_WATCH_DIR"):
    try:
        import compile_watch
        compile_watch.install()
    except Exception as _e:
        sys.stderr.write("[compile_watch] install failed: %r\n" % (_e,))


class _GuardSnapshot(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """--enable-profile-cuda-graph (added by the probe, not the deployment) makes sglang
    0.5.22 dump a CUDA memory snapshot after graph capture; on ROCm torch fails to parse
    it (RuntimeError: stoi) and the server dies. The snapshot is a debug file: a failure
    to write it is logged and ignored. Bound when torch.cuda.memory is first imported, so
    nothing is imported early (an early torch import broke FlyDSL)."""
    busy = False

    def find_spec(self, name, path=None, target=None):
        if name != "torch.cuda.memory" or self.busy:
            return None
        self.busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy = False
        if spec is None or spec.loader is None:
            return None
        self.real, spec.loader = spec.loader, self
        return spec

    def create_module(self, spec):
        return self.real.create_module(spec)

    def exec_module(self, module):
        module.__spec__.loader = self.real
        self.real.exec_module(module)
        orig = getattr(module, "_dump_snapshot", None)
        if orig is not None:
            def _dump_snapshot(*a, **k):
                try:
                    return orig(*a, **k)
                except RuntimeError as e:
                    sys.stderr.write("[probe] torch.cuda.memory._dump_snapshot failed (%r); "
                                     "debug snapshot skipped\n" % (e,))
            module._dump_snapshot = _dump_snapshot
        sys.meta_path[:] = [f for f in sys.meta_path if f is not self]


if os.environ.get("PROBE_GUARD_GRAPH_SNAPSHOT"):
    sys.meta_path.insert(0, _GuardSnapshot())
_rest = [p for p in sys.path if os.path.abspath(p or ".") != _here]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
'''
REJECTED = re.compile(r"unrecognized arguments:.*--enable-profile-cuda-graph|"
                      r"no such option: --enable-profile-cuda-graph")


def log(msg):
    print("[probe_server %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def gpu_arch():
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\bgfx[0-9a-f]+\b", out)
    return m.group(0) if m else None


def watch_overlay(run_dir):
    """<run-dir>/compile_watch_overlay: compile_watch.py and the chaining sitecustomize."""
    ov = os.path.join(run_dir, "compile_watch_overlay")
    os.makedirs(ov, exist_ok=True)
    with open(os.path.join(SHARED, "compile_watch.py")) as src, \
            open(os.path.join(ov, "compile_watch.py"), "w") as dst:
        dst.write(src.read())
    with open(os.path.join(ov, "sitecustomize.py"), "w") as fh:
        fh.write(WATCH_SITECUSTOMIZE)
    return ov


def command(cfg, mode, trace_root, watch_dir=None):
    cmd = list(cfg.get("launcher") or TEMPLATE["launcher"])
    cmd += ["--model-path", cfg["model"], "--host", cfg.get("host", "127.0.0.1"),
            "--port", str(cfg["port"]), "--tp-size", str(cfg["tp"])]
    args = list(cfg.get("args") or [])
    cmd += args
    if mode == "profile_cuda_graph" and "--enable-profile-cuda-graph" not in args:
        cmd.append("--enable-profile-cuda-graph")
    if mode == "eager" and "--disable-cuda-graph" not in args:
        cmd.append("--disable-cuda-graph")
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    env["SGLANG_TORCH_PROFILER_DIR"] = trace_root
    if cfg.get("gpus"):
        env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    if not env.get("GPU_ARCHS"):
        arch = gpu_arch()
        if arch:
            env["GPU_ARCHS"] = arch
    if cfg.get("pythonpath_prepend"):
        env["PYTHONPATH"] = cfg["pythonpath_prepend"] + (":" + env["PYTHONPATH"]
                                                        if env.get("PYTHONPATH") else "")
    if mode == "profile_cuda_graph":
        env["PROBE_GUARD_GRAPH_SNAPSHOT"] = "1"
    if watch_dir or mode == "profile_cuda_graph":
        env["PYTHONPATH"] = watch_overlay(os.path.dirname(trace_root)) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if watch_dir:
        env["COMPILE_WATCH_DIR"] = watch_dir
        env["COMPILE_WATCH_FRAMES"] = str(cfg.get("compile_watch_frames", 12))
    return cmd, env


def base_url(cfg):
    return "http://%s:%s" % (cfg.get("host", "127.0.0.1"), cfg["port"])


def model_config(cfg, env):
    """The served model's config.json: a local model directory, or the HF cache snapshot
    the server resolves (HF_HUB_CACHE / HF_HOME from the deployment's env, --revision or
    refs/main). None if it cannot be found."""
    model = cfg["model"]
    if os.path.isdir(model):
        path = os.path.join(model, "config.json")
    else:
        hub = env.get("HF_HUB_CACHE") or os.path.join(
            env.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
        repo = os.path.join(hub, "models--" + model.replace("/", "--"))
        args = list(cfg.get("args") or [])
        rev = args[args.index("--revision") + 1] if "--revision" in args[:-1] else None
        try:
            rev = rev or open(os.path.join(repo, "refs", "main")).read().strip()
        except OSError:
            return None
        path = os.path.join(repo, "snapshots", rev, "config.json")
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def vocab_size(cfg, env):
    """Prompt ids the probe may send: serving.json's vocab_size, else the model config's
    (text_config's for a multimodal wrapper)."""
    if cfg.get("vocab_size"):
        return int(cfg["vocab_size"])
    mc = model_config(cfg, env) or {}
    v = mc.get("vocab_size") or (mc.get("text_config") or {}).get("vocab_size")
    return int(v) if v else None


def healthy(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as fh:
            return fh.status == 200
    except Exception:
        return False


class Server:
    def __init__(self, cfg, run_dir):
        self.cfg, self.run_dir = cfg, os.path.abspath(run_dir)
        self.trace_root = os.path.join(self.run_dir, "traces")
        self.state_path = os.path.join(self.run_dir, "probe_server.json")
        self.proc = None
        self.starts = []        # [watch_dir, mode, ready_epoch] per started server
        self.primary = None     # the first server that came up: the deployment's variants
        os.makedirs(self.trace_root, exist_ok=True)

    def start(self, mode, ready_timeout):
        watch = None
        if self.cfg.get("compile_watch", True):
            watch = os.path.join(self.run_dir, "compile_watch", "%d.%s" % (len(self.starts), mode))
            os.makedirs(watch, exist_ok=True)
        self.starts.append([watch, mode, None])
        cmd, env = command(self.cfg, mode, self.trace_root, watch)
        logp = os.path.join(self.run_dir, "server.%s.log" % mode)
        t0 = time.time()
        log("starting (%s): %s" % (mode, " ".join(cmd)))
        self.proc = subprocess.Popen(cmd, env=env, stdout=open(logp, "w"),
                                     stderr=subprocess.STDOUT, start_new_session=True)
        url = base_url(self.cfg)
        while time.time() - t0 < ready_timeout:
            if self.proc.poll() is not None:
                text = open(logp, errors="replace").read()[-20000:]
                return {"ok": False, "log": logp, "rc": self.proc.returncode,
                        "flag_rejected": bool(REJECTED.search(text))}
            if healthy(url):
                break
            time.sleep(2)
        else:
            self.stop()
            return {"ok": False, "log": logp, "rc": None, "timeout": True}
        info = None
        try:
            with urllib.request.urlopen(url + "/get_server_info", timeout=30) as fh:
                info = json.loads(fh.read().decode())
        except Exception:
            pass
        if info is not None:
            json.dump(info, open(os.path.join(self.run_dir, "server_info.json"), "w"), indent=1)
        self.starts[-1][2] = time.time()
        if self.primary is None:
            self.primary = len(self.starts) - 1
        vocab = vocab_size(self.cfg, env)
        if not vocab:
            log("vocab_size not found for %s (set \"vocab_size\" in serving.json); the probe "
                "keeps prompt ids small" % self.cfg["model"])
        state = {"base_url": url, "pid": self.proc.pid, "pgid": os.getpgid(self.proc.pid),
                 "compile_watch_dir": watch, "vocab_size": vocab,
                 "decode_capture": mode, "decode_eager": mode == "eager",
                 "trace_root": self.trace_root, "expect_files": int(self.cfg["tp"]),
                 "log": logp, "command": cmd, "startup_s": round(time.time() - t0, 1),
                 "ready_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "server_info": os.path.join(self.run_dir, "server_info.json") if info else None}
        json.dump(state, open(self.state_path, "w"), indent=1)
        log("ready in %.0fs at %s (decode_capture=%s)" % (state["startup_s"], url, mode))
        return {"ok": True, "log": logp, "state": state}

    def collect_variants(self, eval_dir=None):
        """Merge each started server's compile events; the primary's is the deployment's."""
        out = None
        for i, (watch, mode, ready) in enumerate(self.starts):
            if not watch or ready is None:
                continue
            doc = compile_watch.collect(watch, ready, mode)
            doc["deployment"] = i == self.primary
            path = os.path.join(self.run_dir, "compile_variants.json" if i == self.primary
                                else "compile_variants.%d.%s.json" % (i, mode))
            json.dump(doc, open(path, "w"), indent=1)
            log("compile variants: %d from %d process(es) -> %s"
                % (len(doc["variants"]), doc["processes"], path))
            if i == self.primary:
                out = path
                if not doc["variants"]:
                    log("WARNING: the server compiled no Triton kernel the watch saw; if it "
                        "uses Triton, another jit_cache_hook replaced ours")
                if eval_dir:
                    os.makedirs(os.path.join(eval_dir, "evidence"), exist_ok=True)
                    json.dump(doc, open(os.path.join(eval_dir, "evidence", "compile_variants.json"),
                                        "w"), indent=1)
        return out

    def stop(self, wait=120):
        if self.proc is None or self.proc.poll() is not None:
            return
        pg = os.getpgid(self.proc.pid)
        for sig, grace in ((signal.SIGTERM, wait), (signal.SIGKILL, 30)):
            try:
                os.killpg(pg, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(grace)
                break
            except subprocess.TimeoutExpired:
                continue
        # The group can outlive its leader (TP workers); wait for every member.
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                os.killpg(pg, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        log("stopped")


def start_with_fallback(srv, mode, ready_timeout):
    res = srv.start(mode, ready_timeout)
    if not res["ok"] and mode == "profile_cuda_graph" and res.get("flag_rejected"):
        log("--enable-profile-cuda-graph rejected by this server; restarting eagerly")
        res = srv.start("eager", ready_timeout)
        res["fell_back"] = "flag_rejected"
    if not res["ok"]:
        raise SystemExit("probe_server: the server did not come up (%s); see %s"
                         % ("timeout" if res.get("timeout") else "rc=%s" % res.get("rc"), res["log"]))
    return res


def graph_hidden_points(cmd):
    """Decode points a run_points.py command reported graph_hidden, if it was one."""
    if "--eval-dir" not in cmd:
        return []
    ev = cmd[cmd.index("--eval-dir") + 1]
    run = os.path.join(ev, "profile_run.json")
    try:
        d = json.load(open(run))
    except (OSError, ValueError):
        return []
    return [pid for pid, p in d.get("points", {}).items()
            if any(a.get("graph_hidden") for a in p.get("attempts", []))
            and "census" not in p]


def evidence_requested(cmd):
    """run_points.py asked for an eager evidence window and has not had it yet."""
    if "--eval-dir" not in cmd:
        return False
    ev = cmd[cmd.index("--eval-dir") + 1]
    req = os.path.join(ev, "evidence", "eager_evidence_request.json")
    if not os.path.exists(req):
        return False
    r = json.load(open(req))
    got = os.path.join(ev, "evidence", "shapes", r["point"] + "@eager.json")
    if not os.path.exists(got):
        return True
    # an earlier window may have been taken for fewer kernels (a narrower selection)
    d = json.load(open(got))
    return bool(set(r.get("kernels") or []) - set(d.get("requested") or d.get("kernels", {})))


def cmd_run(a, cfg):
    srv = Server(cfg, a.run_dir)

    def bye(signum, _frame):
        srv.stop()
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    mode = cfg.get("decode_capture") or "profile_cuda_graph"
    rc = 1
    try:
        start_with_fallback(srv, mode, a.ready_timeout)
        rc = subprocess.call(a.command)
        hidden = graph_hidden_points(a.command)
        eager = json.load(open(srv.state_path))["decode_capture"] == "eager"
        if hidden and not eager:
            log("decode points %s were graph-hidden; restarting eagerly and resuming" % hidden)
            srv.stop()
            start_with_fallback(srv, "eager", a.ready_timeout)
            rc = subprocess.call(a.command + ["--decode-eager", "--redo"] + hidden)
        elif evidence_requested(a.command) and not eager:
            # Costs stay from the graph windows; only the launch evidence (frames, ops)
            # of graph-replayed decode kernels needs an eager window.
            log("graph-replayed decode kernels need launch evidence; restarting eagerly "
                "for one evidence window")
            srv.stop()
            start_with_fallback(srv, "eager", a.ready_timeout)
            state = json.load(open(srv.state_path))
            state.update(decode_eager=False, decode_capture="profile_cuda_graph",
                         evidence_server="eager")
            json.dump(state, open(srv.state_path, "w"), indent=1)
            rc = subprocess.call(a.command + ["--evidence-capture"])
    finally:
        srv.stop()
        ev = a.command[a.command.index("--eval-dir") + 1] if "--eval-dir" in a.command else None
        try:
            srv.collect_variants(ev)
        except Exception as e:      # evidence, not the run: report and keep the exit code
            log("compile variant collection failed: %r" % (e,))
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--out", required=True)
    for name in ("run", "start", "stop"):
        p = sub.add_parser(name)
        p.add_argument("--serving", required=True)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--ready-timeout", type=float, default=2400,
                       help="seconds; a cold container JIT-compiles aiter for ~10 minutes")
        if name == "run":
            p.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.cmd == "init":
        if os.path.exists(a.out):
            raise SystemExit("probe_server: %s exists" % a.out)
        json.dump(TEMPLATE, open(a.out, "w"), indent=1)
        print("wrote %s; fill in model/tp/gpus/port/args/env from the deployment" % a.out)
        return 0
    cfg = json.load(open(a.serving))
    for k in ("model", "tp", "port"):
        if k not in cfg or cfg[k] in (None, "", "/path/to/model"):
            raise SystemExit("probe_server: serving.json needs %r" % k)
    if a.cmd == "run":
        if a.command and a.command[0] == "--":
            a.command = a.command[1:]
        if not a.command:
            raise SystemExit("probe_server run: give the command after --")
        return cmd_run(a, cfg)
    if a.cmd == "start":
        srv = Server(cfg, a.run_dir)
        res = start_with_fallback(srv, cfg.get("decode_capture") or "profile_cuda_graph",
                                  a.ready_timeout)
        print(json.dumps(res["state"]))
        return 0
    state = json.load(open(os.path.join(a.run_dir, "probe_server.json")))
    try:
        os.killpg(state["pgid"], signal.SIGTERM)
        time.sleep(20)
        os.killpg(state["pgid"], signal.SIGKILL)
    except ProcessLookupError:
        pass
    log("stopped process group %s" % state["pgid"])
    if state.get("compile_watch_dir"):
        doc = compile_watch.collect(state["compile_watch_dir"], None, state.get("decode_capture"))
        json.dump(doc, open(os.path.join(a.run_dir, "compile_variants.json"), "w"), indent=1)
        log("compile variants: %d -> %s" % (len(doc["variants"]),
                                            os.path.join(a.run_dir, "compile_variants.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
