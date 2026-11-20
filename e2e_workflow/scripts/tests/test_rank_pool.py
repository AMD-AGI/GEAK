"""CPU tests of the multi-rank UT: two rank processes on gloo stand in for two GPUs.

They cover the rank pools of ut_runner.py and the serve mode of leg_runner.py end to end (per-rank
correctness, per-sample merged timing and its receipt, rank_skew, failures, time limits, the CI's start
skew injection reaching every rank process), and the pure helpers (merge_time, the rank environment)."""
import contextlib
import datetime
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import generate_ut as gu  # noqa: E402
import ut_runner as ur  # noqa: E402
import verify_ut as vu  # noqa: E402
from ut_task_fixture import RankToyTask, have_torch  # noqa: E402


def _row(sig, samples, timer="cuda_event_graph", primed=True):
    ms = sorted(samples)[len(samples) // 2]
    return {"sig": sig, "regime": "decode", "m": 4, "ms": ms, "wall_ms": ms, "timer": timer,
            "samples": samples, "primed": primed, "host_ms": 0.01,
            "alignment": {"samples": len(samples), "late_releases": 0, "max_late_us": 0.0}}


class TestMergeTime(unittest.TestCase):
    def test_a_sample_ends_with_its_slowest_rank_then_the_median(self):
        r0 = {"cases": [_row("a", [1.0, 5.0, 1.0])]}
        r1 = {"cases": [_row("a", [4.0, 1.0, 1.0])]}
        out = ur.merge_time([r0, r1])[0]
        self.assertEqual(out["ms"], 4.0)              # max per sample: [4, 5, 1] -> median 4
        self.assertEqual(out["rank_ms"], [1.0, 1.0])  # each rank's own median
        # the per-rank medians' maximum would have hidden the skew: max(1, 1) = 1
        self.assertTrue(out["primed"])

    def test_ranks_must_agree_on_cases_timer_and_sample_count(self):
        with self.assertRaises(RuntimeError):
            ur.merge_time([{"cases": [_row("a", [1.0])]}, {"cases": [_row("b", [1.0])]}])
        with self.assertRaises(RuntimeError):
            ur.merge_time([{"cases": [_row("a", [1.0])]}, {"cases": [_row("a", [1.0], timer="cuda_event")]}])
        out = ur.merge_time([{"cases": [_row("a", [1.0, 2.0])]}, {"cases": [_row("a", [1.0])]}])[0]
        self.assertIsNone(out["ms"])                  # unequal sample counts cannot be merged per sample

    def test_a_rank_that_could_not_tell_drops_primed(self):
        out = ur.merge_time([{"cases": [_row("a", [1.0], primed=None)]}, {"cases": [_row("a", [1.0])]}])[0]
        self.assertNotIn("primed", out)


class TestRankEnvironment(unittest.TestCase):
    def test_the_deployed_device_selection_never_reaches_a_rank(self):
        meta = {"regime": {"env": {"HIP_VISIBLE_DEVICES": "0,1,2,3", "ROCR_VISIBLE_DEVICES": "0,1,2,3",
                                   "GPU_DEVICE_ORDINAL": "0", "VLLM_ROCM_USE_AITER": "1"}}}
        env, dropped = ur.deployed_env(meta)
        self.assertEqual(env, {"VLLM_ROCM_USE_AITER": "1"})
        self.assertEqual(dropped, ["GPU_DEVICE_ORDINAL", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])

    def test_a_rank_sees_the_leased_group_in_both_visibility_variables(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        with open(os.path.join(root, "meta.json"), "w") as fh:
            json.dump({"regime": {"env": {"ROCR_VISIBLE_DEVICES": "7", "FOO": "1"}}}, fh)
        pool = object.__new__(ur.RankPool)
        pool.task, pool.overlay, pool.world, pool.devices = root, "/ovl", 2, ["4", "5", "6"]
        pool.port, pool.setup_s, pool.command_s, pool.env_extra = 12345, 60, 60, {"X": "y"}
        old = {k: os.environ.get(k) for k in ("PYTHONUSERBASE", "ROCR_VISIBLE_DEVICES")}
        os.environ["PYTHONUSERBASE"] = "/tmp/ub"
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        try:
            env = pool._env()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual((env["HIP_VISIBLE_DEVICES"], env["CUDA_VISIBLE_DEVICES"]), ("4,5", "4,5"))
        self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
        self.assertEqual(env["PYTHONUSERBASE"], "/tmp/ub")      # the caller's user site reaches the ranks
        self.assertNotIn("PYTHONNOUSERSITE", env)
        self.assertEqual((env["WORLD_SIZE"], env["MASTER_PORT"], env["FOO"], env["X"]), ("2", "12345", "1", "y"))
        self.assertTrue(env["PYTHONPATH"].startswith("/ovl"))

    def test_ports_are_never_given_out_twice(self):
        ports = {ur.free_port() for _ in range(20)}
        self.assertEqual(len(ports), 20)


USERCUSTOMIZE = r'''import os
_LOG = os.environ.get("UTNEG_SKEW_LOG")
if _LOG:
    import importlib._bootstrap_external as _be
    _orig_exec = _be.SourceFileLoader.exec_module

    def _exec_module(self, module, _orig=_orig_exec):
        _orig(self, module)
        f = getattr(module, "__file__", "") or ""
        if os.path.basename(f) == "harness_lib.py" and hasattr(module, "flush_cache") \
                and not getattr(module, "_utneg_skew", False):
            orig = module.flush_cache
            rank = int(os.environ.get("RANK", "0") or 0)

            def flush_cache(torch=None, mb=None, _o=orig):
                with open(_LOG, "a") as fh:
                    fh.write(f"flush rank={rank} world={os.environ.get('WORLD_SIZE')} "
                             f"local={os.environ.get('LOCAL_RANK')} pid={os.getpid()}\n")
                return _o(torch, mb=mb)

            module.flush_cache = flush_cache
            module._utneg_skew = True
            with open(_LOG, "a") as fh:
                fh.write(f"patched rank={rank} pid={os.getpid()} file={f}\n")
    _be.SourceFileLoader.exec_module = _exec_module
'''


@unittest.skipUnless(have_torch(), "torch not installed")
class TestTwoRankRuns(unittest.TestCase):
    def setUp(self):
        self.t = RankToyTask()
        self.addCleanup(self.t.cleanup)
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        self.addCleanup(self._no_rank_left)

    def _no_rank_left(self):
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        self.assertNotIn(f"--task {self.t.task} --mode serve", out, "a rank process outlived its UT")

    def run_ut(self, *args, env_extra=None, timeout=600):
        env = self.t.env()
        env.update(env_extra or {})
        t0 = time.monotonic()
        p = subprocess.run([sys.executable, "unittest.py", *args], cwd=self.t.task, env=env,
                           capture_output=True, text=True, timeout=timeout)
        # the run names the result it wrote (A28: a partial run never writes run/ut_result.json)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("UT_RESULT: ")][-1]
        out = json.loads(line[len("UT_RESULT: "):])["result"]
        if "--out" in args:
            self.assertEqual(out, args[args.index("--out") + 1])
        with open(out) as fh:
            return p.returncode, p.stdout + p.stderr, json.load(fh), time.monotonic() - t0

    def test_two_ranks_pass_with_every_rank_timed_and_the_alignment_recorded(self):
        code, out, res, _ = self.run_ut()
        self.assertEqual(code, 0, out[-3000:])
        rc = res["receipt"]
        self.assertEqual(rc["world_size"], 2)
        for sig, c in rc["cases"].items():
            for leg in ("baseline", "current"):
                self.assertEqual(len(c[leg]["rank_ms"]), 2, sig)
        al = rc["alignment"]
        self.assertIn("gloo all_reduce(MAX) barrier", al["per_sample"])
        self.assertIn("busy-wait", al["per_sample"])
        self.assertIn("max over the ranks", al["merge"])
        self.assertIn("AITER_CUSTOM_AR_RAW_INPUT_POOL", al["harness_env"])
        self.assertEqual([r["rank"] for r in res["correctness"]["by_rank"]], [0, 1])
        self.assertTrue(all(r["launch_succeeded"] for r in res["correctness"]["by_rank"]))
        cov = res["coverage"]                                    # A27, multi-rank too
        self.assertEqual(cov["ranks"], 2)
        self.assertEqual(cov["calls"]["timed"], ["decode_m1", "decode_m4"])
        self.assertEqual(cov["calls"]["uncovered"], [])

    def test_a_wrong_candidate_is_rejected_on_every_rank(self):
        self.t.write("kernel_src/toyop.py", open(os.path.join(self.t.task, "kernel_src", "toyop.py")).read()
                     .replace("torch.tanh(x @ w)", "torch.tanh(x @ w) * 1.1"))
        code, out, res, _ = self.run_ut("--correctness-only")
        self.assertEqual(code, 1, out[-3000:])
        rows = res["correctness"]["by_rank"]
        self.assertEqual([r["rank"] for r in rows], [0, 1])
        self.assertTrue(all(r["numerical_mismatches"] > 0 and r["launch_succeeded"] for r in rows))

    def test_per_output_tolerances_hold_on_every_rank(self):
        """A42: a candidate 2% off passes tol 0.05 but not its output's own 0.005, on both ranks; the result records
        the tolerance every output was compared at and the per-output noise merged over the ranks."""
        self.t.write("kernel_src/toyop.py", open(os.path.join(self.t.task, "kernel_src", "toyop.py")).read()
                     .replace("torch.tanh(x @ w)", "torch.tanh(x @ w) * 1.02"))
        m = self.t.meta()
        m["tolerance"] = {"tol": 0.05, "per_output": {"0": 0.005}}
        self.t.write_meta(m)
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        code, out, res, _ = self.run_ut("--correctness-only")
        self.assertEqual(code, 1, out[-3000:])
        self.assertEqual(res["tolerance"]["outputs"], [{"output": "0", "tol": 0.005}])
        self.assertEqual(res["native_noise"]["per_output"]["0"]["tol"], 0.005)
        ranks = res["correctness"]["candidate"]["ranks"]
        self.assertEqual([r["rank"] for r in ranks], [0, 1])
        for r in ranks:
            rows = [x for sec in ("eager", "random") for x in r["report"]["default"][sec]
                    if x["case"] != "output_independence"]
            self.assertTrue(rows and all(not x["correct"] and (x["output"], x["tol"]) == ("0", 0.005) for x in rows))
        m["tolerance"] = {"tol": 0.05}
        self.t.write_meta(m)
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        code, out, res, _ = self.run_ut("--correctness-only")
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("tolerance", res)

    def test_rank_skew_is_rejected_on_the_last_rank_and_missed_when_ranks_share_their_data(self):
        out = os.path.join(self.t.task, "run", "skew.json")
        code, _, res, _ = self.run_ut("--control", "rank_skew", "--out", out)
        verdict, detail = vu.judge_control("rank_skew", {}, code, res, 0.01, 2)
        self.assertEqual(verdict, "PASS", detail)
        same = {"TOY_SAME_INPUTS": "1"}
        out2 = os.path.join(self.t.task, "run", "skew_same.json")
        code, _, res, _ = self.run_ut("--control", "rank_skew", "--out", out2, "--native-dir",
                                      os.path.join(self.t.task, "run", "native_same"), env_extra=same)
        verdict, detail = vu.judge_control("rank_skew", {}, code, res, 0.01, 2)
        self.assertEqual(verdict, "FAIL")          # identical data on every rank: a wrong-rank read is invisible

    def test_a_rank_that_raises_ends_its_pool_at_once(self):
        self.t.write("kernel_src/toyop.py", open(os.path.join(self.t.task, "kernel_src", "toyop.py")).read()
                     .replace("def op(x, w):\n", "def op(x, w):\n    import os\n"
                              "    if os.environ.get('RANK') == '1':\n"
                              "        raise RuntimeError('candidate broke on rank 1')\n"))
        code, out, res, took = self.run_ut("--correctness-only")
        self.assertEqual(code, 1, out[-3000:])
        self.assertIn("candidate broke on rank 1", json.dumps(res["correctness"]["candidate"]))
        self.assertLess(took, 300)

    def test_a_setup_failure_is_the_tasks_own(self):
        code, out, res, _ = self.run_ut("--correctness-only", env_extra={"TOY_FAIL_SETUP_RANK": "1"})
        self.assertEqual(code, 3, out[-3000:])
        self.assertIn("setup failed on purpose", res["reason"])

    def test_a_command_over_its_time_limit_stops_cleanly(self):
        m = self.t.meta()
        m["timeouts"] = {"setup_s": 60, "command_s": 3, "total_s": 600}
        self.t.write_meta(m)
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        code, out, res, took = self.run_ut("--correctness-only", env_extra={"TOY_SLEEP_S": "2"})
        self.assertEqual(code, 2, out[-3000:])
        self.assertIn("time limit", res["reason"])
        self.assertLess(took, 120)

    def test_the_ci_start_skew_injection_reaches_every_rank_once_per_sample(self):
        ub = tempfile.mkdtemp(prefix="userbase_")
        self.addCleanup(shutil.rmtree, ub, True)
        site = os.path.join(ub, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        os.makedirs(site)
        with open(os.path.join(site, "usercustomize.py"), "w") as fh:
            fh.write(USERCUSTOMIZE)
        log = os.path.join(ub, "inject.log")
        code, out, res, _ = self.run_ut("--timing-only", "--reps", "1",
                                        env_extra={"PYTHONUSERBASE": ub, "UTNEG_SKEW_LOG": log})
        self.assertEqual(code, 0, out[-3000:])
        lines = open(log).read().splitlines()
        patched = {l.split()[1] for l in lines if l.startswith("patched")}
        self.assertTrue({"rank=0", "rank=1"} <= patched, patched)
        flushes = [l for l in lines if l.startswith("flush")]
        for r in (0, 1):
            mine = [l for l in flushes if f"rank={r} world=2 local={r}" in l]
            # every timed sample of both legs: 2 cases x (baseline + candidate) x 50 samples
            self.assertEqual(len(mine), 2 * 2 * 50, (r, len(mine)))


# --------------------------------------------------------------------------- one rank, in this process (no torch)
# The rank side of the pool protocol (leg_runner.py serve mode) on the CPU: torch / torch.distributed are fakes
# that simulate the peers' contributions to every collective; leg_runner's clock, os._exit, stdin and stderr
# are replaced on a private copy of the module; the task's harness_lib.py / cases.py are small files in a temp
# task dir (its harness keeps the real harness_lib's rank context and input-factory call).
_SCRIPTS = os.path.dirname(HERE)
_MISSING = object()
# what a rank reads from its pool environment: cleared, then set by each test
_RANK_ENV = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GEAK_UT_DATA_RANK_SHIFT", "GEAK_UT_SETUP_S", "GEAK_UT_COMMAND_S",
             "GEAK_UT_PARENT_PID")


def _load_private(name, filename):
    """A script of the scripts dir as a module of its own (never in sys.modules); leg_runner's import-time
    sys.path scrub is undone."""
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


lr = _load_private("geak_rank_pool_leg_runner", "leg_runner.py")
hl = _load_private("geak_rank_pool_harness_lib", "harness_lib.py")


@contextlib.contextmanager
def _modules(mods):
    """sys.modules entries for the block; exactly what was there before is put back."""
    saved = {n: sys.modules.get(n, _MISSING) for n in mods}
    sys.modules.update(mods)
    try:
        yield
    finally:
        for n, mod in saved.items():
            if mod is _MISSING:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


def _restore_modules_after(test, *names):
    """Put back what sys.modules held under `names` when the test ends (the rank worker registers its task's
    harness_lib and cases there)."""
    saved = {n: sys.modules.get(n, _MISSING) for n in names}

    def restore():
        for n, mod in saved.items():
            if mod is _MISSING:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod
    test.addCleanup(restore)


def _patch(test, target, name, value):
    p = mock.patch.object(target, name, value)
    p.start()
    test.addCleanup(p.stop)
    return value


class _Exited(BaseException):
    """os._exit, stubbed: the rank process would end here."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _exit(code):
    raise _Exited(code)


class _Proxy:
    """One of leg_runner's module globals (os, sys) for a test: the overrides given, all else the real module's."""

    def __init__(self, real, **over):
        self._real = real
        self.__dict__.update(over)

    def __getattr__(self, name):
        return getattr(self._real, name)


class FakeClock:
    """leg_runner's `time`: monotonic_ns advances `step_ns` per read (every read kept), monotonic 0.25 s per read,
    sleep only recorded -- nothing ever waits."""

    def __init__(self, step_ns=1000):
        self.ns, self.step_ns, self.secs = 10 ** 12, step_ns, 100.0
        self.reads, self.sleeps = [], []

    def monotonic_ns(self):
        self.ns += self.step_ns
        self.reads.append(self.ns)
        return self.ns

    def monotonic(self):
        self.secs += 0.25
        return self.secs

    def sleep(self, seconds):
        self.sleeps.append(seconds)


class FakeTensor:
    def __init__(self, vals, dtype=None, device="cpu"):
        self.vals, self.dtype, self.device = [int(v) for v in vals], dtype, device

    def item(self):
        return self.vals[0]


class FakeDist:
    """torch.distributed of ONE rank with its peers simulated: `peers(op, group, mine)` returns the other ranks'
    contributions to an all_reduce (none by default). There is no barrier(): the control plane must not use it."""
    ReduceOp = types.SimpleNamespace(SUM="sum", MIN="min", MAX="max")

    def __init__(self, backend=None, peers=None):
        self.backend = backend                  # None: no process group yet
        self.peers = peers or (lambda op, group, mine: [])
        self.log, self.groups, self.reduces = [], [], []

    def is_initialized(self):
        return self.backend is not None

    def init_process_group(self, backend, init_method=None, world_size=None, rank=None, timeout=None):
        self.backend = backend
        self.log.append(("init", backend, init_method, world_size, rank, timeout))

    def get_backend(self):
        return self.backend

    def new_group(self, backend=None, timeout=None):
        group = types.SimpleNamespace(name=f"group{len(self.groups)}", backend=backend, timeout=timeout)
        self.groups.append(group)
        return group

    def all_reduce(self, tensor, op="sum", group=None):
        mine = tensor.vals[0]
        tensor.vals[0] = {"sum": sum, "min": min, "max": max}[op]([mine, *self.peers(op, group, mine)])
        self.reduces.append(types.SimpleNamespace(op=op, group=group, dtype=tensor.dtype, mine=mine,
                                                  out=tensor.vals[0]))

    def destroy_process_group(self):
        self.log.append(("destroy",))


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait_stream", self.name, other.name))


class _Event:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        readings = self.cuda.elapsed
        return readings.pop(0) if len(readings) > 1 else readings[0]


class FakeGraph:
    """A captured graph: the ops recorded while capturing, run again by replay()."""

    def __init__(self):
        self.ops = []

    def replay(self):
        for op in self.ops:
            op()


class FakeCuda:
    """torch.cuda: availability, events whose elapsed_time readings are scripted (the last one repeats), _sleep
    spins, streams and graph capture. What is issued is appended to `log`."""

    def __init__(self, available=True, elapsed=(2.0,), log=None):
        self.available, self.elapsed = available, list(elapsed)
        self.log = [] if log is None else log
        self.capturing, self.graphs, self.fail_next_sync = None, [], False
        self.current = _Stream("default", self.log)

    def is_available(self):
        return self.available

    def set_device(self, index):
        self.log.append(("set_device", index))

    def synchronize(self):
        self.log.append(("synchronize",))
        if self.fail_next_sync:
            self.fail_next_sync = False
            raise RuntimeError("sticky device error")

    def _sleep(self, cycles):
        self.log.append(("spin", cycles))

    def Event(self, enable_timing=False):
        return _Event(self)

    def Stream(self):
        return _Stream("side", self.log)

    def current_stream(self):
        return self.current

    @contextlib.contextmanager
    def stream(self, s):
        prev, self.current = self.current, s
        try:
            yield
        finally:
            self.current = prev

    def CUDAGraph(self):
        self.graphs.append(FakeGraph())
        return self.graphs[-1]

    @contextlib.contextmanager
    def graph(self, g, capture_error_mode=None):
        self.log.append(("capture_begin", capture_error_mode))
        self.capturing = g
        try:
            yield
        finally:
            self.capturing = None
            self.log.append(("capture_end",))


class FakeGenerator:
    def __init__(self, device="cpu"):
        self.device, self.seed = device, None

    def manual_seed(self, seed):
        self.seed = int(seed)
        return self


class FakeTorch:
    """The torch one rank sees: int64 control tensors, Generator, save/load as JSON files, cuda, distributed."""
    int64 = "int64"

    def __init__(self, cuda=None, dist=None):
        self.cuda = cuda if cuda is not None else FakeCuda(available=False)
        self.distributed = dist if dist is not None else FakeDist()
        self.loaded = []

    def tensor(self, vals, dtype=None):
        return FakeTensor(vals, dtype)

    def zeros(self, n, dtype=None):
        return FakeTensor([0] * n, dtype)

    def ones(self, n, device=None):
        return FakeTensor([1] * n, "float32", device)

    def Generator(self, device="cpu"):
        return FakeGenerator(device)

    def save(self, obj, path):
        with open(path, "w") as fh:
            json.dump(obj, fh)

    def load(self, path, map_location=None):
        self.loaded.append((path, map_location))
        with open(path) as fh:
            return json.load(fh)


_WORKER_HARNESS = '''"""harness_lib.py of the in-process rank tests: the real rank context, input-factory call and
deployment readers, and a timer that keeps time_op's multi-rank hook contract."""
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location("geak_rank_pool_real_harness", @REAL@)
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
RankContext = _real.RankContext
call_make_inputs = _real.call_make_inputs
_spin_cycles_per_us = _real._spin_cycles_per_us
deployment_graph_mode = _real.deployment_graph_mode
deployment_compile_mode = _real.deployment_compile_mode

REPEATS = 3
calls = []
installed = []
_CTX = None


def _torch():
    return sys.modules["torch"]


def set_rank_context(ctx):
    global _CTX
    _CTX = ctx
    installed.append(ctx)
    return ctx


def rank_context():
    return _CTX if _CTX is not None else RankContext()


def compiled_op(fn, regime):
    calls.append(("compiled_op", regime.get("compile")))

    def compiled(args):
        calls.append(("compiled_call", args["sig"]))
        return fn(args)
    return compiled


def check_graph_replay(*args, **kwargs):
    raise AssertionError("a rank pool must not run the single-process graph replay check")


def time_op(call, warmup=10, repeats=50, inner=1, graph=False, *, detail=False, samples=False,
            pre_sample=None, capture=None, capture_context=None, spin=None, raise_errors=False):
    """None when the call raises (with raise_errors, the exception); a graph phase captured through `capture` (eager
    timing when it returns None); pre_sample() once per sample before its launches; the per-sample times with
    samples=True; primed and host_ms only with a device timer."""
    torch = _torch()
    cuda = torch.cuda.is_available()
    calls.append(("time_op", graph, inner, capture is not None, raise_errors))
    try:
        call()
    except Exception:
        if raise_errors:
            raise
        return None
    timer = "wall"
    if cuda:
        timer = "cuda_event"
        if graph and capture is not None and capture(torch, call, inner) is not None:
            timer = "cuda_event_graph"
    kept = []
    for i in range(REPEATS):
        if pre_sample is not None:
            pre_sample()
        for _ in range(inner):
            call()
        kept.append(float(i + 1))
    if not detail:
        return 2.0
    res = {"ms": 2.0, "wall_ms": 2.5, "timer": timer, "cache_condition": {"mode": "read-evict"}}
    if samples:
        res["samples"] = kept
    if cuda:
        res.update(primed=True, host_ms=0.01)
    return res
'''.replace("@REAL@", repr(os.path.join(_SCRIPTS, "harness_lib.py")))

_WORKER_CASES = '''"""cases.py of the in-process rank tests."""
calls = []
RAISE = set()


def call(args):
    calls.append(("call", args["sig"]))
    if args["sig"] in RAISE:
        raise RuntimeError("the op raised on " + args["sig"])
    return dict(args)


def timing_cases(h, meta):
    return [{"sig": "decode_m1", "regime": "decode", "m": 1, "args": {"sig": "decode_m1"}},
            {"sig": "prefill_m8", "regime": "prefill", "m": 8, "inner": 4, "args": {"sig": "prefill_m8"}}]


def _factory(sig):
    def make_inputs(rng, ctx):
        return {"sig": sig, "rng": [rng.device, rng.seed], "draw": ctx.seed, "data_rank": ctx.data_rank,
                "rank_seed": ctx.rank_rng.seed}
    return make_inputs


def random_shapes(h, meta):
    return [{"sig": "decode_m1", "regime": "decode", "make_inputs": _factory("decode_m1")},
            {"sig": "prefill_m8", "regime": "prefill", "make_inputs": _factory("prefill_m8")}]
'''

_CASES_SETUP_GLOO = '''

def setup(h, meta, ctx):
    calls.append(("setup", ctx.as_dict()))
    h._torch().distributed.init_process_group(backend="gloo", init_method="env://", world_size=ctx.world,
                                              rank=ctx.rank)
'''

_CASES_SETUP_FAILS = '''

def setup(h, meta, ctx):
    raise RuntimeError(f"setup failed on purpose on rank {ctx.rank}")
'''

_CASES_ALIGN = '''

def align(h, meta, ctx):
    calls.append(("align", ctx.rank, meta["target_callable"]))
'''

_CASES_TEARDOWN = '''

def teardown():
    calls.append(("teardown",))
'''


def _worker_meta(**over):
    meta = {"target_callable": "json:dumps", "random_draws": 2,
            "phase_regimes": {"decode": {"cuda_graph": True},
                              "prefill": {"cuda_graph": False, "compile": "torch_compile"}},
            "timing": {"device_spin_us": 100, "release_margin_us": 50, "capture_settle_s": 0.5}}
    meta.update(over)
    return meta


class TestRankContextFromPoolEnv(unittest.TestCase):
    def test_the_rank_context_comes_from_the_pool_environment(self):
        installed = []
        h = types.SimpleNamespace(RankContext=hl.RankContext,
                                  set_rank_context=lambda ctx: installed.append(ctx) or ctx)
        table = [  # env -> (rank, world, data_rank, local_rank)
            ({}, (0, 1, 0, 0)),
            ({"WORLD_SIZE": "", "RANK": ""}, (0, 1, 0, 0)),
            # a single-rank run is rank 0 of 1 whatever RANK / LOCAL_RANK / the skew say
            ({"WORLD_SIZE": "1", "RANK": "3", "LOCAL_RANK": "2", "GEAK_UT_DATA_RANK_SHIFT": "1"}, (0, 1, 0, 0)),
            ({"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "0"}, (2, 4, 2, 0)),
            ({"WORLD_SIZE": "4", "RANK": "1"}, (1, 4, 1, 1)),
            ({"WORLD_SIZE": "4", "RANK": "1", "LOCAL_RANK": ""}, (1, 4, 1, 1)),
            # rank_skew: only the LAST rank builds another rank's data
            ({"WORLD_SIZE": "4", "RANK": "3", "GEAK_UT_DATA_RANK_SHIFT": "1"}, (3, 4, 2, 3)),
            ({"WORLD_SIZE": "4", "RANK": "2", "GEAK_UT_DATA_RANK_SHIFT": "1"}, (2, 4, 2, 2)),
            ({"WORLD_SIZE": "2", "RANK": "1", "GEAK_UT_DATA_RANK_SHIFT": "3"}, (1, 2, 0, 1)),
        ]
        for env, want in table:
            installed.clear()
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                for k in _RANK_ENV:
                    if k not in env:
                        os.environ.pop(k, None)
                ctx = lr.leg_ctx_from_env(h)
                self.assertEqual(installed, [ctx])          # installed as this process's context
                self.assertEqual((ctx.rank, ctx.world, ctx.data_rank, ctx.local_rank), want)

    def test_timing_settings_default_key_by_key(self):
        defaults = (200.0, 500.0, 0.3)          # A03 spin, A02 release margin, capture settle
        self.assertEqual(lr._timing_settings({}), defaults)
        self.assertEqual(lr._timing_settings({"timing": None}), defaults)
        self.assertEqual(lr._timing_settings({"timing": {"release_margin_us": "1000"}}), (200.0, 1000.0, 0.3))
        self.assertEqual(lr._timing_settings({"timing": {"device_spin_us": 50, "release_margin_us": 250,
                                                         "capture_settle_s": 0}}), (50.0, 250.0, 0.0))


class _FakeCtypes(types.ModuleType):
    """ctypes for _die_with_parent: CDLL(None) is this module and its prctl is only recorded."""

    def __init__(self):
        super().__init__("ctypes")
        self.calls = []

    def CDLL(self, name, use_errno=False):
        self.calls.append(("CDLL", name, use_errno))
        return self

    def prctl(self, option, arg):
        self.calls.append(("prctl", option, arg))
        return 0


class TestDieWithParent(unittest.TestCase):
    """No parent-death signal is ever installed on the test process: ctypes is a fake, os and sys are proxied."""

    def die(self, parent_pid, platform="linux", ppid=4242):
        fake, exits = _FakeCtypes(), []
        env = {} if parent_pid is None else {"GEAK_UT_PARENT_PID": parent_pid}
        with _modules({"ctypes": fake}), mock.patch.dict(os.environ, env), \
                mock.patch.object(lr, "os", _Proxy(os, getppid=lambda: ppid, _exit=exits.append)), \
                mock.patch.object(lr, "sys", _Proxy(sys, platform=platform)):
            if parent_pid is None:
                os.environ.pop("GEAK_UT_PARENT_PID", None)
            lr._die_with_parent()
        return fake.calls, exits

    def test_a_rank_dies_with_the_driver_that_started_it(self):
        calls, exits = self.die("4242", ppid=4242)
        self.assertEqual(calls, [("CDLL", None, True), ("prctl", 1, signal.SIGKILL)])     # PR_SET_PDEATHSIG
        self.assertEqual(exits, [])

    def test_a_rank_whose_driver_is_already_gone_exits_at_once(self):
        calls, exits = self.die("4242", ppid=1)      # re-parented: the driver died before the prctl took hold
        self.assertEqual(calls[-1], ("prctl", 1, signal.SIGKILL))
        self.assertEqual(exits, [1])

    def test_nothing_is_installed_outside_a_linux_pool(self):
        for parent_pid, platform in ((None, "linux"), ("", "linux"), ("4242", "darwin")):
            with self.subTest(parent_pid=parent_pid, platform=platform):
                self.assertEqual(self.die(parent_pid, platform), ([], []))


class TestWriteJson(unittest.TestCase):
    def test_an_answer_lands_whole_under_its_final_name(self):
        d = tempfile.mkdtemp(prefix="geak_rank_json_")
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "rank0.7.json")
        with open(path, "w") as fh:
            fh.write("stale")
        lr._write_json(path, {"rank": 0, "sigs": ["a"], "timeout": datetime.timedelta(seconds=3)})
        self.assertEqual(os.listdir(d), ["rank0.7.json"])          # the temp file was renamed over it
        with open(path) as fh:
            self.assertEqual(json.load(fh), {"rank": 0, "sigs": ["a"], "timeout": "0:00:03"})   # default=str


class TestRankControl(unittest.TestCase):
    def test_the_control_plane_is_a_gloo_group_of_its_own(self):
        dist = FakeDist(backend="nccl")
        ctl = lr.RankControl(FakeTorch(dist=dist), 60)
        self.assertEqual(dist.groups, [ctl.group])
        self.assertEqual((ctl.group.backend, ctl.group.timeout), ("gloo", datetime.timedelta(seconds=60)))

    def test_agree_holds_only_when_every_rank_says_yes(self):
        votes = []
        dist = FakeDist(peers=lambda op, group, mine: votes.pop(0))
        ctl = lr.RankControl(FakeTorch(dist=dist), 60)
        for mine, peers, want in ((True, [1, 1], True), (True, [1, 0], False), (False, [1, 1], False)):
            votes.append(peers)
            self.assertIs(ctl.agree(mine), want)
        self.assertEqual({(r.op, r.dtype) for r in dist.reduces}, {("min", "int64")})
        self.assertTrue(all(r.group is ctl.group for r in dist.reduces))

    def test_max_int_and_a_barrier_that_is_a_cpu_all_reduce(self):
        dist = FakeDist(peers=lambda op, group, mine: [7, 5] if op == "max" else [0])
        ctl = lr.RankControl(FakeTorch(dist=dist), 60)
        self.assertEqual(ctl.max_int(3), 7)
        self.assertEqual(ctl.max_int(12.0), 12)
        ctl.barrier()               # FakeDist has no barrier(): it must be an all-reduce every rank joins
        self.assertEqual([(r.op, r.dtype, r.mine, r.group) for r in dist.reduces],
                         [("max", "int64", 3, ctl.group), ("max", "int64", 12, ctl.group),
                          ("sum", "int64", 0, ctl.group)])


class TestSampleAligner(unittest.TestCase):
    def make(self, *, device=False, peers=None, align_call=None, elapsed=(2.0,), spin_us=100.0, margin_us=250.0):
        self.clock = _patch(self, lr, "time", FakeClock())
        self.dist = FakeDist(peers=peers)
        self.gpu = FakeCuda(available=device, elapsed=elapsed)
        torch = FakeTorch(self.gpu, self.dist)
        self.ctl = lr.RankControl(torch, 60)
        self.dev_group = types.SimpleNamespace(name="device", backend="nccl")
        hl._SPIN_RATE.clear()                       # harness_lib calibrates once per device: once per test here
        return lr.SampleAligner(torch, self.ctl, device=device, dev_group=self.dev_group, spin_us=spin_us,
                                margin_us=margin_us, spin_rate=hl._spin_cycles_per_us, align_call=align_call)

    def test_the_release_waits_for_the_last_rank_plus_the_margin(self):
        al = self.make(peers=lambda op, group, mine: [mine + 400_000])   # a peer arrived 400 us later
        al()
        rel = self.dist.reduces[-1]
        self.assertEqual((rel.op, rel.group, rel.dtype), ("max", self.ctl.group, "int64"))
        self.assertEqual(rel.mine, self.clock.reads[0] + 250_000)        # this rank's now + margin
        target = rel.out
        self.assertEqual(target, rel.mine + 400_000)                     # ... the last rank's
        self.assertGreaterEqual(self.clock.reads[-1], target)            # busy-waited up to the release
        self.assertLess(self.clock.reads[-2], target)                    # and not one read longer
        self.assertEqual(self.clock.sleeps, [])                          # never a sleeping wait
        self.assertEqual(al.receipt(), {"samples": 1, "late_releases": 0, "max_late_us": 0.0, "spin_cycles": 0,
                                        "device_sync": "none (no CUDA)"})
        self.assertEqual(self.gpu.log, [])                               # without CUDA only the release

    def test_a_release_already_past_counts_as_late_and_is_not_waited_for(self):
        delays = [1_000_000, 600_000]

        def slow_collective(op, group, mine):
            self.clock.ns += delays.pop(0)       # the all-reduce returned this long after the rank entered it
            return [mine - 1]
        al = self.make(peers=slow_collective)
        al()
        self.assertEqual(len(self.clock.reads), 3)      # entry, return, one look at the clock: no wait
        al()
        # late by 1000 us - the 250 us margin + one 1 us clock read; the 351 us of the second does not lower it
        self.assertEqual(al.receipt(), {"samples": 2, "late_releases": 2, "max_late_us": 751.0, "spin_cycles": 0,
                                        "device_sync": "none (no CUDA)"})

    def test_with_cuda_every_sample_spins_then_all_reduces_on_the_device_group(self):
        al = self.make(device=True, elapsed=(0.0, 4.0, 2.0))    # a zero reading is ignored, the last one counts
        self.assertEqual(al.spin_cycles, 50_000)                  # 1e6 cycles in 2 ms: 500/us x 100 us
        self.assertEqual([e for e in self.gpu.log if e[0] == "spin"], [("spin", 1_000_000)] * 3)
        self.assertEqual([(r.op, r.group, r.dtype) for r in self.dist.reduces],
                         [("sum", self.dev_group, "float32")])       # the device group warmed up
        self.gpu.log.clear()
        al()
        self.assertEqual(self.gpu.log, [("spin", 50_000)])
        self.assertEqual([(r.op, r.group) for r in self.dist.reduces[1:]],
                         [("max", self.ctl.group), ("sum", self.dev_group)])
        self.assertEqual(al.receipt(), {"samples": 1, "late_releases": 0, "max_late_us": 0.0, "spin_cycles": 50_000,
                                        "device_sync": "rccl_all_reduce_1"})

    def test_a_calibration_without_a_reading_queues_no_spin(self):
        al = self.make(device=True, elapsed=(0.0,))
        self.assertEqual(al.spin_cycles, 0)
        self.gpu.log.clear()
        al.device_align()
        self.assertEqual(self.gpu.log, [])
        self.assertIs(self.dist.reduces[-1].group, self.dev_group)

    def test_a_tiny_spin_is_still_one_cycle(self):
        self.assertEqual(self.make(device=True, spin_us=0.001).spin_cycles, 1)

    def test_the_tasks_align_hook_replaces_the_device_all_reduce(self):
        hits = []
        al = self.make(device=True, align_call=lambda: hits.append("align"))
        self.assertEqual(self.dist.reduces, [])          # no warm-up on any device group
        al()
        self.assertEqual(hits, ["align"])
        self.assertEqual([r.op for r in self.dist.reduces], ["max"])     # only the host release
        self.assertEqual(al.receipt()["device_sync"], "cases.align")


@contextlib.contextmanager
def _scope(log, name):
    log.append((name, "enter"))
    try:
        yield
    finally:
        log.append((name, "exit"))


class _CaptureRig(unittest.TestCase):
    """One (fake) CUDA rank with a real RankControl; `peer_votes` are the other ranks' capture votes."""

    def setUp(self):
        self.clock = _patch(self, lr, "time", FakeClock())
        self.stderr = io.StringIO()
        _patch(self, lr, "sys", _Proxy(sys, stderr=self.stderr))
        self.events = []
        self.gpu = FakeCuda(available=True, log=self.events)
        self.peer_votes = []
        self.dist = FakeDist(peers=self._peers)
        self.torch = FakeTorch(self.gpu, self.dist)
        self.ctl = lr.RankControl(self.torch, 60)

    def _peers(self, op, group, mine):
        return [self.peer_votes.pop(0)] if op == "min" else []

    def votes(self):
        return [(r.mine, r.out) for r in self.dist.reduces if r.op == "min"]

    def op(self):
        """An op recorded into the graph being captured, else run on the current stream."""
        def call():
            if self.gpu.capturing is None:
                self.events.append(("run", self.gpu.current.name))
            else:
                self.gpu.capturing.ops.append("op")
                self.events.append(("captured",))
        return call


class TestAgreedCapture(_CaptureRig):
    def test_a_graph_every_rank_captured(self):
        self.peer_votes[:] = [1]
        cases = types.SimpleNamespace(capture_context=lambda: _scope(self.events, "framework"))
        g = lr._agreed_capture(self.torch, self.ctl, cases, 0.5)(self.torch, self.op(), 2)
        self.assertIs(g[0], self.gpu.graphs[0])
        self.assertEqual((g[0].ops, g[1]), (["op", "op"], 2))
        # warm-up on a side stream, sync, settle, then a thread-local capture inside the task's capture context
        self.assertEqual(self.events, [
            ("wait_stream", "side", "default"), ("run", "side"), ("run", "side"), ("run", "side"),
            ("wait_stream", "default", "side"), ("synchronize",),
            ("framework", "enter"), ("capture_begin", "thread_local"), ("captured",), ("captured",),
            ("capture_end",), ("framework", "exit"), ("synchronize",)])
        self.assertEqual(self.clock.sleeps, [0.5])
        self.assertEqual(self.votes(), [(1, 1)])
        self.assertEqual(self.stderr.getvalue(), "")

    def test_a_rank_whose_capture_fails_makes_every_rank_go_without_a_graph(self):
        self.peer_votes[:] = [1, 1, 1]               # the peers captured every time

        def call():
            if self.gpu.capturing is not None:
                self.gpu.fail_next_sync = True       # a failed capture leaves a sticky error behind
                raise RuntimeError("operation not permitted when stream is capturing")
        self.assertIsNone(lr._agreed_capture(self.torch, self.ctl, types.SimpleNamespace(), 0.0)(self.torch, call, 1))
        # vote no, barrier, try again -- three times -- then no graph on any rank
        self.assertEqual([(r.op, r.mine, r.out) for r in self.dist.reduces], [("min", 0, 0), ("sum", 0, 0)] * 3)
        msg = "[leg_runner] rank capture failed: RuntimeError('operation not permitted when stream is capturing')\n"
        self.assertEqual(self.stderr.getvalue(), msg * 3)

    def test_a_peer_that_failed_is_retried_together(self):
        self.peer_votes[:] = [0, 1]                  # a peer could not capture the first time
        g = lr._agreed_capture(self.torch, self.ctl, types.SimpleNamespace(), 0.25)(self.torch, self.op(), 1)
        self.assertEqual(self.votes(), [(1, 0), (1, 1)])
        self.assertIs(g[0], self.gpu.graphs[1])      # the graph of the agreed retry, not this rank's first one
        self.assertEqual(self.clock.sleeps, [0.25, 0.25])
        self.assertEqual([r.op for r in self.dist.reduces], ["min", "sum", "min"])

    def test_at_least_one_try(self):
        self.peer_votes[:] = [0]
        capture = lr._agreed_capture(self.torch, self.ctl, types.SimpleNamespace(), 0.0, tries=0)
        self.assertIsNone(capture(self.torch, self.op(), 1))
        self.assertEqual(self.votes(), [(1, 0)])

    def test_the_capture_scope_is_the_tasks_capture_context_or_nothing(self):
        log = []
        with lr._capture_scope(types.SimpleNamespace(capture_context=lambda: _scope(log, "framework"))):
            log.append("inside")
        self.assertEqual(log, [("framework", "enter"), "inside", ("framework", "exit")])
        with lr._capture_scope(types.SimpleNamespace()) as got:
            self.assertIsNone(got)


class _RowErr(float):
    """An error carrying per-output row fields, as a per-output comparison returns it."""


class TestAgreedGraphReplay(_CaptureRig):
    def buffers(self):
        """fill / run / read_out of an op out = 3 * x, its captured graph replayed on each filled input."""
        state = {}

        def fill(c):
            if c["args"] is None:
                raise RuntimeError(f"no input for {c['sig']}")
            state["x"] = c["args"]

        def compute():
            state["out"] = 3 * state["x"]

        def run():
            if self.gpu.capturing is not None:
                self.gpu.capturing.ops.append(compute)
            else:
                compute()
        return fill, run, lambda: state["out"]

    def checker(self, verdicts, seen):
        def correct(out, ref, tol):
            seen.append((out, ref, tol))
            return verdicts.pop(0)
        return lr._agreed_graph_replay(types.SimpleNamespace(correct=correct), self.torch, self.ctl,
                                       types.SimpleNamespace(), 0.0)

    def test_every_case_replays_the_one_agreed_capture_and_is_judged(self):
        self.peer_votes[:] = [1, 1]
        err = _RowErr(0.5)
        err.row_fields = {"output": "0", "tol": 0.6}
        verdicts, seen = [(True, 0.000123456), (False, float("nan")), (False, float("inf")), (True, err)], []
        check = self.checker(verdicts, seen)
        fill, run, read_out = self.buffers()
        rows = [{"sig": "a", "args": 1, "ref": 3}, {"sig": "b", "args": 2, "ref": 7}, {"args": 3, "ref": 0},
                {"sig": "d", "args": 4, "ref": 12}, {"sig": "e", "args": None, "ref": 0}]
        ok, per = check(fill, run, read_out, rows, 0.01, capture_idx=1, warmup=3, strict=True)
        self.assertFalse(ok)
        self.assertEqual(per, [
            {"case": "a", "correct": True, "max_rel_err": 0.00012, "note": "graph_replay"},
            {"case": "b", "correct": False, "max_rel_err": None, "note": "graph_replay"},
            {"case": "", "correct": False, "max_rel_err": None, "note": "graph_replay"},
            {"case": "d", "correct": True, "max_rel_err": 0.5, "note": "graph_replay", "output": "0", "tol": 0.6},
            {"case": "e", "correct": False, "max_rel_err": None,
             "note": "graph_replay_raised: RuntimeError('no input for e')"}])
        # each replay computed from its own filled input (3x), not from the capture's
        self.assertEqual(seen, [(3, 3, 0.01), (6, 7, 0.01), (9, 0, 0.01), (12, 12, 0.01)])
        self.assertEqual(len(self.gpu.graphs), 1)
        verdicts.append((True, 0.0))
        self.assertEqual(check(fill, run, read_out, rows[:1], 0.01),
                         (True, [{"case": "a", "correct": True, "max_rel_err": 0.0, "note": "graph_replay"}]))

    def test_a_capture_that_failed_on_any_rank_fails_the_check(self):
        self.peer_votes[:] = [0, 0, 0]
        check = self.checker([], [])
        filled = []
        rows = [{"sig": "a", "args": 1, "ref": 3}, {"sig": "b", "args": 2, "ref": 6}]
        ok, per = check(lambda c: filled.append(c["sig"]), self.op(), lambda: None, rows, 0.01, capture_idx=1)
        self.assertFalse(ok)
        self.assertEqual(per, [{"case": "graph_replay", "correct": False, "max_rel_err": None,
                                "note": "graph_capture_failed on at least one rank"}])
        self.assertEqual(filled, ["b"])             # only the capture case: nothing replayed, nothing judged
        self.assertEqual(self.votes(), [(1, 0)] * 3)

    def test_no_cases_skip_without_a_collective(self):
        ok, per = self.checker([], [])(None, None, None, [], 0.01)
        self.assertEqual((ok, per), (True, [{"case": "graph_replay", "correct": True, "max_rel_err": None,
                                              "note": "skipped: no cases"}]))
        self.assertEqual(self.dist.reduces, [])


class _InProcessRank(unittest.TestCase):
    """Rank 1 of a two-rank pool, run in this process."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="geak_rank_worker_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.task = os.path.join(self.root, "task")
        os.makedirs(self.task)
        _restore_modules_after(self, "torch", "harness_lib", "cases")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for k in _RANK_ENV:
            os.environ.pop(k, None)
        os.environ.update({"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"})
        self.clock = _patch(self, lr, "time", FakeClock())
        self.stderr = io.StringIO()
        self.sys = _patch(self, lr, "sys", _Proxy(sys, stdin=io.StringIO(""), stderr=self.stderr))
        _patch(self, lr, "os", _Proxy(os, _exit=_exit, getpid=lambda: 4321))

    def write_task(self, extra="", meta=None):
        for name, text in (("meta.json", json.dumps(meta or _worker_meta())), ("harness_lib.py", _WORKER_HARNESS),
                           ("cases.py", _WORKER_CASES + extra)):
            with open(os.path.join(self.task, name), "w") as fh:
                fh.write(text)

    def use_torch(self, cuda=False, elapsed=(2.0,)):
        self.dist = FakeDist()
        self.gpu = FakeCuda(available=cuda, elapsed=elapsed)
        self.torch = FakeTorch(self.gpu, self.dist)
        sys.modules["torch"] = self.torch

    def worker(self, extra="", cuda=False, meta=None):
        self.write_task(extra, meta)
        self.use_torch(cuda)
        return lr.RankWorker(self.task)


class TestRankWorkerSetup(_InProcessRank):
    def test_a_cpu_rank_opens_the_pool_group_and_a_control_group_of_its_own(self):
        os.environ.update({"GEAK_UT_SETUP_S": "90", "GEAK_UT_COMMAND_S": "45"})
        w = self.worker()
        self.assertEqual(self.dist.log, [("init", "gloo", "env://", 2, 1, datetime.timedelta(seconds=90))])
        self.assertEqual(self.dist.groups, [w.ctl.group])                   # no device group without CUDA
        self.assertEqual((w.ctl.group.backend, w.ctl.group.timeout), ("gloo", datetime.timedelta(seconds=45)))
        self.assertEqual(w.ctx.as_dict(), {"rank": 1, "world": 2, "data_rank": 1, "local_rank": 1})
        self.assertEqual(w.h.installed, [w.ctx])                            # the harness's rank context
        self.assertEqual(w.profile, {"rank": 1, "setup_s": 0.25, "data_rank": 1, "backend": "gloo", "cuda": False})
        self.assertEqual((w.settle_s, w.aligner.margin_ns, w.aligner.spin_us), (0.5, 50_000, 100.0))
        self.assertEqual(w.aligner.receipt()["device_sync"], "none (no CUDA)")
        self.assertEqual(w.h.check_graph_replay.__name__, "check_graph_replay")     # left alone without CUDA
        self.assertEqual([(r.op, r.dtype, r.group) for r in self.dist.reduces],
                         [("sum", "int64", w.ctl.group)])                   # the closing control barrier
        self.assertEqual(self.gpu.log, [])

    def test_a_cuda_rank_whose_setup_built_gloo_gets_an_nccl_device_group(self):
        os.environ.update({"LOCAL_RANK": "0", "GEAK_UT_COMMAND_S": ""})
        w = self.worker(_CASES_SETUP_GLOO, cuda=True)
        self.assertEqual(w.cases.calls, [("setup", {"rank": 1, "world": 2, "data_rank": 1, "local_rank": 0})])
        self.assertEqual(self.dist.log, [("init", "gloo", "env://", 2, 1, None)])     # the task's own init only
        ctl, dev = self.dist.groups
        self.assertEqual((ctl.backend, ctl.timeout, dev.backend), ("gloo", datetime.timedelta(seconds=1800), "nccl"))
        self.assertEqual(self.gpu.log[0], ("set_device", 0))                # this rank's LOCAL device
        self.assertIs(w.aligner.dev_group, dev)
        self.assertEqual([(r.group, r.dtype) for r in self.dist.reduces], [(dev, "float32"), (ctl, "int64")])
        self.assertEqual(w.aligner.spin_cycles, 50_000)
        self.assertEqual(w.profile, {"rank": 1, "setup_s": 0.25, "data_rank": 1, "backend": "gloo", "cuda": True})
        # the harness's graph-replay check now captures by agreement
        self.assertEqual(w.h.check_graph_replay(None, None, None, [], 0.01),
                         (True, [{"case": "graph_replay", "correct": True, "max_rel_err": None,
                                  "note": "skipped: no cases"}]))

    def test_a_cuda_rank_without_setup_runs_on_the_pools_nccl_group(self):
        w = self.worker(cuda=True)
        self.assertEqual(self.dist.log, [("init", "nccl", "env://", 2, 1, datetime.timedelta(seconds=900))])
        self.assertEqual(self.dist.groups, [w.ctl.group])          # the default group is the device group
        self.assertIsNone(w.aligner.dev_group)
        self.assertEqual([r.group for r in self.dist.reduces], [None, w.ctl.group])
        self.assertEqual((w.profile["backend"], w.aligner.receipt()["device_sync"]), ("nccl", "rccl_all_reduce_1"))
        self.assertEqual(self.gpu.log[0], ("set_device", 1))

    def test_a_tasks_align_hook_replaces_the_device_group(self):
        w = self.worker(_CASES_SETUP_GLOO + _CASES_ALIGN, cuda=True)
        self.assertEqual(self.dist.groups, [w.ctl.group])          # no nccl group for it, even over gloo
        self.assertEqual([r.group for r in self.dist.reduces], [w.ctl.group])     # and no device warm-up
        w.aligner()
        self.assertEqual(w.cases.calls[-1], ("align", 1, "json:dumps"))
        self.assertEqual(w.aligner.receipt()["device_sync"], "cases.align")


class TestRankWorkerCommands(_InProcessRank):
    def test_list_names_the_timed_cases(self):
        self.assertEqual(self.worker().do("list", {}), {"rank": 1, "sigs": ["decode_m1", "prefill_m8"]})

    def test_time_on_a_cpu_rank_aligns_every_sample_and_brackets_every_case(self):
        w = self.worker()
        n0 = len(self.dist.reduces)
        out = w.do("time", {})
        self.assertEqual((out["rank"], out["identity"]["module"], out["identity"]["qualname"]), (1, "json", "dumps"))
        decode, prefill = out["cases"]
        self.assertEqual(decode, {"sig": "decode_m1", "regime": "decode", "m": 1, "ms": 2.0, "wall_ms": 2.5,
                                  "timer": "wall", "cache_condition": {"mode": "read-evict"},
                                  "samples": [1.0, 2.0, 3.0], "inner": 1,
                                  "alignment": {"samples": 3, "late_releases": 0, "max_late_us": 0.0,
                                                "spin_cycles": 0, "device_sync": "none (no CUDA)"}})
        self.assertEqual((prefill["sig"], prefill["regime"], prefill["m"], prefill["inner"]),
                         ("prefill_m8", "prefill", 8, 4))                 # the case's own inner
        self.assertEqual(prefill["alignment"]["samples"], 3)              # per case, not cumulative
        # each phase timed as deployed: decode as a graph, prefill compiled; no capture hook without CUDA
        self.assertEqual([c for c in w.h.calls if c[0] == "time_op"],
                         [("time_op", True, 1, False, True), ("time_op", False, 4, False, True)])
        self.assertIn(("compiled_call", "prefill_m8"), w.h.calls)
        # every case: a control barrier, one release per sample, a control barrier -- all on the control group
        new = self.dist.reduces[n0:]
        self.assertEqual([r.op for r in new], ["sum", "max", "max", "max", "sum"] * 2)
        self.assertTrue(all(r.group is w.ctl.group for r in new))

    def test_time_honours_the_bucket_and_an_inner_override(self):
        w = self.worker()
        out = w.do("time", {"bucket": "prefill_m8", "inner": 2})
        self.assertEqual([(r["sig"], r["inner"]) for r in out["cases"]], [("prefill_m8", 2)])
        self.assertEqual(w.do("time", {"bucket": "nope"})["cases"], [])

    def test_time_of_an_op_that_raises_raises_and_joins_no_further_collective(self):
        # A rank that went on would meet its peers' per-sample releases with its closing barrier and hang the pool
        # until command_s; raising lets serve_main write the traceback and end the rank at once.
        w = self.worker()
        w.cases.RAISE.add("decode_m1")
        n0 = len(self.dist.reduces)
        with self.assertRaisesRegex(RuntimeError, "the op raised on decode_m1"):
            w.do("time", {"bucket": "decode_m1"})
        self.assertEqual([r.op for r in self.dist.reduces[n0:]], ["sum"])      # the opening barrier only
        self.assertEqual([c for c in w.h.calls if c[0] == "time_op"], [("time_op", True, 1, False, True)])

    def test_time_on_a_cuda_rank_captures_by_agreement(self):
        w = self.worker(cuda=True)
        n0 = len(self.dist.reduces)
        self.gpu.log.clear()
        decode, prefill = w.do("time", {})["cases"]
        self.assertEqual((decode["timer"], prefill["timer"]), ("cuda_event_graph", "cuda_event"))
        self.assertEqual((decode["primed"], decode["host_ms"]), (True, 0.01))    # the timer's receipt travels
        self.assertEqual(decode["alignment"], {"samples": 3, "late_releases": 0, "max_late_us": 0.0,
                                               "spin_cycles": 50_000, "device_sync": "rccl_all_reduce_1"})
        self.assertEqual([c for c in w.h.calls if c[0] == "time_op"],
                         [("time_op", True, 1, True, True), ("time_op", False, 4, True, True)])
        # one capture (the graph-deployed decode): the meta's settle, thread-local, agreed by one MIN vote
        self.assertEqual(self.clock.sleeps, [0.5])
        self.assertEqual(self.gpu.log.count(("capture_begin", "thread_local")), 1)
        new = self.dist.reduces[n0:]
        self.assertEqual([(r.mine, r.out) for r in new if r.op == "min"], [(1, 1)])
        self.assertTrue(all(r.group is w.ctl.group for r in new if r.dtype == "int64"))
        # every sample: a spin queued, then the one-element all-reduce on the pool's (default) device group
        self.assertEqual(self.gpu.log.count(("spin", 50_000)), 6)
        self.assertEqual([r.group for r in new if r.dtype == "float32"], [None] * 6)

    def test_oracle_records_every_draw_from_the_ranks_data(self):
        os.environ["GEAK_UT_DATA_RANK_SHIFT"] = "1"         # rank_skew: the last rank builds rank 0's data
        w = self.worker()
        prefix = os.path.join(self.root, "native")
        out = w.do("oracle", {"blob": prefix, "seed": 5, "draws": 2})
        self.assertEqual(out, {"rank": 1, "out": f"{prefix}.rank1.pt", "n": 4})
        with open(out["out"]) as fh:
            blob = json.load(fh)
        self.assertEqual(sorted(blob), ["decode_m1|0", "decode_m1|1", "prefill_m8|0", "prefill_m8|1"])
        rank0 = hl.RankContext(0, 2)
        for i, seed in enumerate((5, 6)):
            self.assertEqual(blob[f"prefill_m8|{i}"], {"sig": "prefill_m8", "rng": ["cpu", seed], "draw": seed,
                                                        "data_rank": 0, "rank_seed": rank0.seed_for(seed)})
        self.assertNotEqual(rank0.seed_for(5), hl.RankContext(1, 2).seed_for(5))    # rank 1's own data differs
        self.assertIn(("compiled_call", "prefill_m8"), w.h.calls)           # prefill ran compiled, as deployed

    def test_oracle_on_a_cuda_rank_draws_on_the_device_at_least_once(self):
        w = self.worker(cuda=True, meta=_worker_meta(random_draws=0))
        out = w.do("oracle", {"blob": os.path.join(self.root, "native")})
        self.assertEqual(out["n"], 2)                     # random_draws 0: still one draw per shape
        with open(out["out"]) as fh:
            draw = json.load(fh)["decode_m1|0"]
        self.assertEqual((draw["rng"], draw["draw"], draw["data_rank"]), (["cuda", 0], 0, 1))

    def test_refs_and_correct_run_the_correctness_worker_fail_fast(self):
        w = self.worker()
        seen = []

        def correct_mode(a, task, meta, h, cases, fail_fast=False):
            seen.append((vars(a), task, meta is w.meta, h is w.h, cases is w.cases, fail_fast))
            return {"mode": a.mode, "ok": True}
        with mock.patch.object(lr, "_correct_mode", correct_mode):
            refs = w.do("refs", {"blob": "/n/native", "draws": 4})
            judged = w.do("correct", {"blob": "/n/native", "draws": None, "control": "rank_skew",
                                      "control_spec": '{"k": 1}'})
        self.assertEqual((refs, judged), ({"rank": 1, "mode": "refs", "ok": True},
                                          {"rank": 1, "mode": "correct", "ok": True}))
        self.assertEqual(seen, [
            ({"mode": "refs", "blob": "/n/native", "draws": 4, "control": "", "control_spec": ""},
             self.task, True, True, True, True),
            ({"mode": "correct", "blob": "/n/native", "draws": 0, "control": "rank_skew", "control_spec": '{"k": 1}'},
             self.task, True, True, True, True)])

    def test_correct_without_the_native_refs_is_incomplete(self):
        w = self.worker()
        prefix = os.path.join(self.root, "native")
        w.do("oracle", {"blob": prefix, "draws": 1})
        out = w.do("correct", {"blob": prefix, "draws": 1})
        self.assertEqual(out, {"rank": 1, "mode": "correct", "identity": {"json:dumps": lr._identity("json:dumps")},
                               "ok": False, "incomplete": [f"native refs missing: {prefix}.refs.rank1.pt"],
                               "errors": [], "report": {}})
        self.assertEqual(self.torch.loaded, [(f"{prefix}.rank1.pt", "cpu")])      # this rank's own blob

    def test_an_unknown_command_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown pool command 'bogus'"):
            self.worker().do("bogus", {})

    def test_a_case_executes_as_its_phase_is_deployed(self):
        w = self.worker()
        call, graph = w._case_execution({"regime": "prefill"})
        self.assertEqual((call({"sig": "x"}), graph), ({"sig": "x"}, False))
        self.assertEqual(w.h.calls[-2:], [("compiled_op", "torch_compile"), ("compiled_call", "x")])
        self.assertEqual(w._case_execution({"regime": "decode"}), (w.cases.call, True))
        with self.assertRaisesRegex(ValueError, "missing phase_regimes entry for 'verify'"):
            w._case_execution({"regime": "verify"})
        w.meta = {"regime": {"cuda_graph": False}}         # no phase_regimes: the task's one regime
        self.assertEqual(w._case_execution({"regime": "verify"}), (w.cases.call, False))
        w.meta = {}
        self.assertEqual(w._case_execution({}), (w.cases.call, True))


class TestRankWorkerClose(unittest.TestCase):
    def test_close_tears_the_task_down_then_destroys_the_group_whatever_fails(self):
        events = []

        def teardown():
            events.append("teardown")

        def bad_teardown():
            events.append("teardown")
            raise RuntimeError("teardown broke")

        def destroy():
            events.append("destroy")
            raise RuntimeError("the group is already gone")
        w = object.__new__(lr.RankWorker)
        w.torch = types.SimpleNamespace(distributed=types.SimpleNamespace(destroy_process_group=destroy))
        w.cases = types.SimpleNamespace(teardown=teardown)
        w.close()                                   # a destroy that raises is swallowed
        self.assertEqual(events, ["teardown", "destroy"])
        events.clear()
        w.cases = types.SimpleNamespace(teardown=bad_teardown)
        with self.assertRaisesRegex(RuntimeError, "teardown broke"):
            w.close()
        self.assertEqual(events, ["teardown", "destroy"])     # the group goes even so
        events.clear()
        w.cases = types.SimpleNamespace()           # no teardown hook
        w.close()
        self.assertEqual(events, ["destroy"])


class TestServeMain(_InProcessRank):
    """The rank side of ut_runner.RankPool: commands as JSON lines on stdin, answers as files in the ctl dir."""

    def serve(self, *commands):
        self.sys.stdin = io.StringIO("".join(c if isinstance(c, str) else json.dumps(c) + "\n" for c in commands))
        self.ctl_dir = os.path.join(self.root, "ctl")
        os.makedirs(self.ctl_dir)
        return lr.serve_main(self.task, self.ctl_dir)

    def answer(self, name):
        with open(os.path.join(self.ctl_dir, name)) as fh:
            return json.load(fh)

    def test_a_rank_answers_each_command_in_order_until_exit(self):
        self.write_task(_CASES_TEARDOWN)
        self.use_torch()
        prefix = os.path.join(self.root, "native")
        code = self.serve("\n", {"seq": 1, "mode": "list", "params": {}}, "  \n",
                          {"seq": 2, "mode": "oracle", "params": {"blob": prefix, "seed": 0, "draws": 1}},
                          {"seq": 3, "mode": "correct", "params": {"blob": prefix, "draws": 1}},
                          {"seq": 4, "mode": "time", "params": {"bucket": "decode_m1"}},
                          {"mode": "exit"},
                          {"seq": 5, "mode": "list", "params": {}})
        self.assertEqual(code, 0)
        self.assertEqual(sorted(os.listdir(self.ctl_dir)),       # nothing after exit, no temp file left
                         ["rank1.1.json", "rank1.2.json", "rank1.3.json", "rank1.4.json", "rank1.ready.json"])
        self.assertEqual(self.answer("rank1.ready.json"),
                         {"rank": 1, "setup_s": 0.25, "data_rank": 1, "backend": "gloo", "cuda": False, "pid": 4321})
        self.assertEqual(self.answer("rank1.1.json"), {"rank": 1, "sigs": ["decode_m1", "prefill_m8"], "seconds": 0.25})
        self.assertEqual(self.answer("rank1.2.json"), {"rank": 1, "out": f"{prefix}.rank1.pt", "n": 2, "seconds": 0.25})
        judged = self.answer("rank1.3.json")
        self.assertEqual((judged["mode"], judged["ok"], judged["incomplete"]),
                         ("correct", False, [f"native refs missing: {prefix}.refs.rank1.pt"]))
        timed = self.answer("rank1.4.json")
        self.assertEqual(([r["sig"] for r in timed["cases"]], timed["cases"][0]["alignment"]["samples"]),
                         (["decode_m1"], 3))
        # the parent reads them as answers, none of them an error
        pool = object.__new__(ur.RankPool)
        pool.dir, pool.procs, pool.deadline, pool.label = self.ctl_dir, [], None, "native"
        got = pool._wait([f"rank1.{s}.json" for s in (1, 2, 3, 4)], "time", 60)
        self.assertEqual([g["rank"] for g in got], [1, 1, 1, 1])
        # exit: the task's teardown, then the process group
        self.assertEqual(sys.modules["cases"].calls[-1], ("teardown",))
        self.assertEqual(self.dist.log[-1], ("destroy",))

    def test_a_rank_whose_stdin_closes_tears_down_and_returns(self):
        self.write_task(_CASES_TEARDOWN)
        self.use_torch()
        self.assertEqual(self.serve({"seq": 1, "mode": "list", "params": None}), 0)
        self.assertEqual(self.answer("rank1.1.json")["sigs"], ["decode_m1", "prefill_m8"])
        self.assertEqual(sys.modules["cases"].calls[-1], ("teardown",))
        self.assertEqual(self.dist.log[-1], ("destroy",))

    def test_a_command_that_raises_ends_the_rank_with_its_traceback(self):
        self.write_task()
        self.use_torch()
        with self.assertRaises(_Exited) as cm:
            self.serve({"seq": 1, "mode": "list"}, {"seq": 2, "mode": "bogus", "params": {}},
                       {"seq": 3, "mode": "list"})
        self.assertEqual(cm.exception.code, 1)
        self.assertEqual(sorted(os.listdir(self.ctl_dir)), ["rank1.1.json", "rank1.2.json", "rank1.ready.json"])
        err = self.answer("rank1.2.json")
        self.assertEqual(sorted(err), ["error", "rank"])
        self.assertEqual(err["rank"], 1)
        self.assertIn("ValueError: unknown pool command 'bogus'", err["error"])
        self.assertIn("ValueError: unknown pool command 'bogus'", self.stderr.getvalue())     # the rank's log
        pool = object.__new__(ur.RankPool)
        pool.label = "candidate"
        exc = pool._raised([err], "bogus")          # what the parent makes of it: the task's own failure, rank 1
        self.assertEqual((exc.kind, exc.rank), ("task", 1))

    def test_a_setup_failure_is_reported_for_the_pool_and_ends_the_rank(self):
        self.write_task(_CASES_SETUP_FAILS)
        self.use_torch()
        with self.assertRaises(_Exited) as cm:
            self.serve({"seq": 1, "mode": "list"})
        self.assertEqual(cm.exception.code, 1)
        self.assertEqual(os.listdir(self.ctl_dir), ["rank1.setup_error.json"])     # never ready
        rec = self.answer("rank1.setup_error.json")
        self.assertEqual(rec["rank"], 1)
        self.assertIn("RuntimeError: setup failed on purpose on rank 1", rec["error"])
        self.assertIn("setup failed on purpose on rank 1", self.stderr.getvalue())
        self.assertEqual(self.sys.stdin.tell(), 0)                    # not one command was read

    def via_main(self, *commands):
        """leg_runner.main() on the command line RankPool starts each rank with: --task T --mode serve --ctl D."""
        self.sys.stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
        self.ctl_dir = os.path.join(self.root, "ctl")
        os.makedirs(self.ctl_dir)
        with mock.patch.object(sys, "argv", ["leg_runner.py", "--task", self.task, "--mode", "serve",
                                             "--ctl", self.ctl_dir]):
            lr.main()

    def test_main_serves_the_rank_and_exits_with_its_code(self):
        self.write_task(_CASES_TEARDOWN)
        self.use_torch()
        with self.assertRaises(SystemExit) as cm:
            self.via_main({"seq": 1, "mode": "list", "params": {}}, {"mode": "exit"})
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(sorted(os.listdir(self.ctl_dir)), ["rank1.1.json", "rank1.ready.json"])
        self.assertEqual(self.answer("rank1.1.json"), {"rank": 1, "sigs": ["decode_m1", "prefill_m8"], "seconds": 0.25})
        self.assertEqual(sys.modules["cases"].calls[-1], ("teardown",))
        self.assertEqual(self.dist.log[-1], ("destroy",))

    def test_main_hands_serve_to_the_rank_before_it_reads_meta_json(self):
        """main() dispatches serve before it opens meta.json: a task without one is the rank's own setup failure,
        recorded in the ctl dir where the parent reads it -- not a FileNotFoundError out of main()."""
        self.use_torch()
        with self.assertRaises(_Exited) as cm:
            self.via_main({"seq": 1, "mode": "list", "params": {}})
        self.assertEqual(cm.exception.code, 1)
        self.assertEqual(os.listdir(self.ctl_dir), ["rank1.setup_error.json"])
        rec = self.answer("rank1.setup_error.json")
        self.assertEqual(rec["rank"], 1)
        self.assertIn("FileNotFoundError", rec["error"])
        self.assertIn(os.path.join(self.task, "meta.json"), rec["error"])
        self.assertEqual(self.sys.stdin.tell(), 0)                    # not one command was read


if __name__ == "__main__":
    unittest.main(verbosity=2)
