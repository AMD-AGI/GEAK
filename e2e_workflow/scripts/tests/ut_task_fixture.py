"""A tiny generated-UT task on disk for the CPU tests of generate_ut.py / ut_runner.py / verify_ut.py.

The "installed" op lives in a site dir outside the task (the baseline must resolve outside it); the
candidate is kernel_src/toyop.py bound as a module. Everything runs on the CPU with torch when torch is
importable; tests that need torch skip without it.
"""
import json
import os
import shutil
import tempfile

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOYOP = '''import torch


def op(x, w):
    return torch.tanh(x @ w)
'''

CASES = '''import toyop

SHAPES = {"decode_m1": 1, "decode_m4": 4}


def call(args):
    return toyop.op(*args)


def _mk(m, rng):
    import torch
    x = torch.randn(m, 16, generator=rng)
    w = torch.randn(16, 8, generator=rng)
    return (x, w)


def timing_cases(h, meta):
    import torch
    return [{"sig": sig, "regime": "decode", "m": m, "args": _mk(m, torch.Generator().manual_seed(7))}
            for sig, m in SHAPES.items()]


def random_shapes(h, meta):
    return [{"sig": sig, "regime": "decode", "make_inputs": (lambda rng, m=m: _mk(m, rng))}
            for sig, m in SHAPES.items()]


def eager_cases(h, meta):
    import torch
    return [{"sig": sig, "regime": "decode", "args": _mk(m, torch.Generator().manual_seed(11))}
            for sig, m in SHAPES.items()]
'''


def meta(site, **over):
    m = {
        "target_callable": "toyop:op",
        "candidate_bind": {"kind": "module", "module": "toyop", "file": "kernel_src/toyop.py"},
        "regime": {"cuda_graph": False},
        "tolerance": {"tol": 0.01},
        "random_draws": 2,
        "workload": {"kind": "agentx_trace_replay", "weighting": "observed",
                     "serving_weight_model": {"source": "observed_workload", "evidence_path": "w.json"},
                     "cases": [{"sig": "decode_m1", "regime": "decode", "m": 1, "effective_calls": 10},
                               {"sig": "decode_m4", "regime": "decode", "m": 4, "effective_calls": 30}]},
        "capabilities": {"stateful": False, "multi_rank": {"world_size": 1}, "per_output_compare": False},
        "case_manifest": {"dims": ["regime", "m"], "cases": [
            {"sig": "decode_m1", "regime": "decode", "m": 1, "roles": ["timing", "correctness"]},
            {"sig": "decode_m4", "regime": "decode", "m": 4, "roles": ["timing", "correctness"]}]},
        "kernel_extraction": {"source_origins": {"kernel_src/toyop.py": os.path.join(site, "toyop.py")},
                              # A35: the ranking the extraction used, outside the task dir (toy_ranking)
                              "ranking_json": os.path.join(os.path.dirname(site), "kernel_ranking.json")},
        "device_kernel": "toy",
    }
    m.update(over)
    return m


def toy_ranking(**row_over):
    """A kernel_ranking.json with the toy kernel's row: calls in DECODE only (A35)."""
    row = {"rank": 1, "name": "toy", "device_kernel": "toy", "phase": "decode", "served_regimes": ["decode"],
           "calls": {"PREFILL": 0, "DECODE": 40}, "share_of_phase_pct": {"PREFILL": 0.0, "DECODE": 5.0}}
    row.update(row_over)
    return {"schema": "geak.kernel_extract.ranking/1", "phases": {"PREFILL": {}, "DECODE": {}}, "ranking": [row]}


class ToyTask:
    """root/site/toyop.py (the installed op), root/kernel_ranking.json (the extraction's ranking) and root/task
    (the task dir)."""

    def __init__(self, **meta_over):
        self.root = tempfile.mkdtemp(prefix="geak_ut_task_")
        self.site = os.path.join(self.root, "site")
        self.task = os.path.join(self.root, "task")
        self.ranking = os.path.join(self.root, "kernel_ranking.json")
        os.makedirs(self.site)
        self.write_ranking(toy_ranking())
        os.makedirs(os.path.join(self.task, "kernel_src"))
        os.makedirs(os.path.join(self.task, "run"))
        self.write_site("toyop.py", TOYOP)
        self.write("kernel_src/toyop.py", TOYOP)
        self.write("cases.py", CASES)
        self.write_meta(meta(self.site, **meta_over))
        self.write("run/requested_kernel.json", json.dumps(
            {"short_name": "toy", "entity_kind": "gpu_kernel", "device_kernel": "toy", "required_gpu_count": 1}))

    def write(self, rel, text):
        path = os.path.join(self.task, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def write_site(self, rel, text):
        with open(os.path.join(self.site, rel), "w") as fh:
            fh.write(text)

    def write_ranking(self, ranking):
        with open(self.ranking, "w") as fh:
            json.dump(ranking, fh, indent=1)

    def write_meta(self, m):
        self.write("meta.json", json.dumps(m, indent=1))

    def meta(self):
        with open(os.path.join(self.task, "meta.json")) as fh:
            return json.load(fh)

    def read_json(self, rel):
        with open(os.path.join(self.task, rel)) as fh:
            return json.load(fh)

    def env(self):
        """Environment of a CPU-only UT run: the site dir is the installed stack."""
        env = dict(os.environ)
        env["PYTHONPATH"] = self.site + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.update(HIP_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
        return env

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- a two-rank collective toy
COLLECTIVE_OP = '''import torch
import torch.distributed as dist


def op(x, w):
    y = torch.tanh(x @ w)
    dist.all_reduce(y)                      # the collective: every rank ends with the sum
    return y
'''

RANK_CASES = '''import os
import toyop

SHAPES = {"decode_m1": 1, "decode_m4": 4}
_W = {}


def setup(h, meta, ctx):
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", init_method="env://", rank=ctx.rank, world_size=ctx.world)
    # a per-rank weight shard, built from the data rank (never from os.environ["RANK"])
    gen = torch.Generator().manual_seed(1234) if os.environ.get("TOY_SAME_INPUTS") == "1" else ctx.generator(1234)
    _W["w"] = torch.randn(16, 8, generator=gen)
    if os.environ.get("TOY_FAIL_SETUP_RANK") == str(ctx.rank):
        raise RuntimeError("setup failed on purpose")


def call(args):
    if os.environ.get("TOY_RAISE_RANK") == os.environ.get("RANK"):
        raise RuntimeError("the candidate raised on this rank")
    if os.environ.get("TOY_SLEEP_S"):
        import time
        time.sleep(float(os.environ["TOY_SLEEP_S"]))
    return toyop.op(args[0], _W["w"])


def _mk(m, gen):
    import torch
    return (torch.randn(m, 16, generator=gen),)


def _gen(h, seed):
    import torch
    if os.environ.get("TOY_SAME_INPUTS") == "1":
        return torch.Generator().manual_seed(seed)
    return h.rank_context().generator(seed)


def timing_cases(h, meta):
    return [{"sig": sig, "regime": "decode", "m": m, "args": _mk(m, _gen(h, 7))} for sig, m in SHAPES.items()]


def random_shapes(h, meta):
    return [{"sig": sig, "regime": "decode",
             "make_inputs": (lambda rng, ctx, m=m: _mk(m, RNG(rng, ctx)))} for sig, m in SHAPES.items()]


def RNG(rng, ctx):
    return rng if os.environ.get("TOY_SAME_INPUTS") == "1" else ctx.rank_rng


def eager_cases(h, meta):
    return [{"sig": sig, "regime": "decode", "args": _mk(m, _gen(h, 11))} for sig, m in SHAPES.items()]
'''


class RankToyTask(ToyTask):
    """A two-rank collective task on the CPU (gloo): per-rank inputs and weight shards from ctx."""

    def __init__(self, world=2, **meta_over):
        super().__init__(**meta_over)
        self.world = world
        self.write_site("toyop.py", COLLECTIVE_OP)
        self.write("kernel_src/toyop.py", COLLECTIVE_OP)
        self.write("cases.py", RANK_CASES)
        m = self.meta()
        m["capabilities"]["multi_rank"] = {
            "world_size": world, "comm": "collective_api", "inputs": {"x": "per_rank", "w": "per_rank"},
            "evidence": {"serving_tp": world, "source_symbols": ["dist.all_reduce"]}}
        m["timeouts"] = {"setup_s": 120, "command_s": 120, "total_s": 900}
        m["kernel_extraction"]["source_origins"] = {"kernel_src/toyop.py": os.path.join(self.site, "toyop.py")}
        m.update(meta_over)
        self.write_meta(m)
        self.write("run/requested_kernel.json", json.dumps(
            {"short_name": "toy", "entity_kind": "gpu_kernel", "device_kernel": "toy", "required_gpu_count": world}))

    def env(self):
        env = super().env()
        env["GEAK_UT_ALLOW_CPU_RANKS"] = "1"
        return env
