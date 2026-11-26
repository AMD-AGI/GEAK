#!/usr/bin/env python3
"""Ready-made ab_bench adapter for a task whose `scripts/task_runner.py` exposes the four-callable contract.

ab_bench.py loads this module itself when given `--task`; you do not write an adapter:

    python3 ab_bench.py --task <ws> --variant base=<ws>/source --variant cand=<dir>/source \
        [--case <serve case id>[,<id>...]] --control auto --permute --json ab.json

Contract read from `<task>/scripts/task_runner.py` (imported, never run as __main__):
    make_inputs(case: dict, seed: int) -> dict
    run(mod, inputs: dict, case: dict) -> tensor | tuple     # `mod` = the variant's source package
    reference(inputs: dict, case: dict) -> tensor | tuple
    tol(case: dict) -> dict(atol=, rtol=[, max_bad_frac=])
    compare(out, ref, atol, rtol, max_bad_frac) -> (ok, detail)   # task_runner or its scripts/_abh.py;
                                                                    # else harness_lib.correct
Cases come from `<task>/test_cases.json` (`serve_cases` + `edge_cases`). A runner that does not expose
this contract is refused with the list of what is missing -- then write an adapter by hand (ab_bench
--help-full has the contract).

Variants: each is a DIRECTORY holding the kernel package (an `__init__.py`; the task's `source/` or a
copy of it with one edit), loaded under its own package name so arms never alias. Shorthands:
`source` = `<task>/source`, `golden` = `<task>/source/source_golden`. With no --variant the arms are
golden (when present, the baseline) and source.

Each variant's callable launches the selected case(s) once with inputs built ONCE (shared by every
arm, so all arms read the same bytes). Several cases are launched back to back inside one sample: the
timed number is their unweighted sum -- it ranks variants, it is not the task's weighted metric.
Correctness: every arm runs every selected case once, compared against `reference()` with the task's
own `tol(case)`; a failing arm is excluded before timing. The oracle reports `tol_fail` (0 pass /
1 fail), so ab_bench's --tol does not loosen or tighten the task's tolerance.
Fingerprint: sha256 over the variant directory's *.py / *.json files (source_golden/ excluded); two
arms with identical source are refused up front (an arm with no diff measures nothing; for an A/A
noise band use ab_bench --control auto). On a pristine task golden == source, so pass a variant.

Configuration arrives through environment variables (ab_bench sets them from its flags):
    AB_TASK_DIR       task / workspace root (holds scripts/task_runner.py and test_cases.json)
    AB_TASK_VARIANTS  name=dir;name=dir   (dir relative to AB_TASK_DIR allowed; shorthands above)
    AB_TASK_CASES     comma list of case ids (default: the highest-weight serve case)
    AB_TASK_SEED      input seed (default 0)

Timing: with --task, ab_bench times captured CUDA-graph replays by default (harness_lib.time_op,
read-evict flush per sample), like most task runners; `--timer eager` times one launch per sample
instead. Use the timer your task's performance mode uses; for launch-overhead-sized cases (decode,
small M) rank with ab_bench, then confirm with the task's own performance command. GEAK verify with
the frozen commands remains the acceptance.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

REQUIRED = ("make_inputs", "run", "reference", "tol")


class TaskAdapterError(RuntimeError):
    pass


def _env(name, default=None):
    v = os.environ.get(name, default)
    return v if v not in ("", None) else default


def _import_runner(task_dir):
    runner = os.path.join(task_dir, "scripts", "task_runner.py")
    if not os.path.isfile(runner):
        raise TaskAdapterError(f"no scripts/task_runner.py under {task_dir}")
    scripts = os.path.dirname(runner)
    # Task harnesses derive TASK_DIR from sys.argv[0]; point it at the runner while importing.
    saved = sys.argv[0]
    sys.argv[0] = runner
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("_ab_task_runner", runner)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_ab_task_runner"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.argv[0] = saved
    missing = [n for n in REQUIRED if not callable(getattr(mod, n, None))]
    if missing:
        raise TaskAdapterError(
            f"{runner} does not expose {missing} (contract: make_inputs(case, seed), run(mod, inputs, "
            f"case), reference(inputs, case), tol(case)). Write an adapter by hand instead "
            f"(ab_bench.py --help-full: 'Adapter contract').")
    return mod


def _compare_fn(runner):
    for holder in (runner, sys.modules.get("_abh")):
        fn = getattr(holder, "compare", None) if holder is not None else None
        if callable(fn):
            return fn
    return None


def _load_package(name, directory):
    init = os.path.join(directory, "__init__.py")
    if not os.path.isfile(init):
        raise TaskAdapterError(f"variant dir {directory} has no __init__.py (expected a copy of the "
                               f"task's source/ package)")
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[directory])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_dir(task_dir, d):
    if d == "source":
        return os.path.join(task_dir, "source")
    if d == "golden":
        return os.path.join(task_dir, "source", "source_golden")
    return d if os.path.isabs(d) else os.path.join(task_dir, d)


def parse_variants(spec, task_dir):
    """'a=dir;b=dir' (or ',' separated) -> ordered {name: abs dir}. Empty -> golden(if any)+source."""
    out = {}
    items = [s for s in spec.replace(",", ";").split(";") if s.strip()] if spec else []
    for item in items:
        if "=" in item:
            name, d = item.split("=", 1)
        else:
            name, d = item, item
        name, d = name.strip(), d.strip()
        if not name or not d:
            raise TaskAdapterError(f"bad variant {item!r}: use NAME=DIR")
        if name in out:
            raise TaskAdapterError(f"variant name {name!r} given twice")
        out[name] = os.path.abspath(_resolve_dir(task_dir, d))
    if not out:
        g = os.path.join(task_dir, "source", "source_golden")
        if os.path.isfile(os.path.join(g, "__init__.py")):
            out["golden"] = os.path.abspath(g)
        out["source"] = os.path.abspath(os.path.join(task_dir, "source"))
    if len(out) < 2:
        raise TaskAdapterError("need at least two variants (NAME=DIR, NAME=DIR)")
    return out


def load_cases(task_dir, ids):
    with open(os.path.join(task_dir, "test_cases.json")) as f:
        spec = json.load(f)
    serve = spec.get("serve_cases", []) if isinstance(spec, dict) else []
    edge = spec.get("edge_cases", []) if isinstance(spec, dict) else []
    allc = {c["id"]: c for c in serve + edge if isinstance(c, dict) and "id" in c}
    if ids:
        unknown = [i for i in ids if i not in allc]
        if unknown:
            raise TaskAdapterError(f"unknown case id(s) {unknown}; known: {sorted(allc)}")
        return [allc[i] for i in ids]
    if not serve:
        raise TaskAdapterError("test_cases.json has no serve_cases; pass --case <id>")
    return [max(serve, key=lambda c: float(c.get("weight", 0.0) or 0.0))]


def dir_fingerprint(directory):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "build", "source_golden"))
        for fn in sorted(files):
            if fn.endswith((".py", ".json")):
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, directory).encode())
                with open(p, "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


class TaskAdapter:
    def __init__(self, task_dir, variants_spec="", case_ids=(), seed=0):
        self.task_dir = os.path.abspath(task_dir)
        self.runner = _import_runner(self.task_dir)
        self.compare = _compare_fn(self.runner)
        self.variant_dirs = parse_variants(variants_spec, self.task_dir)
        self.cases = load_cases(self.task_dir, [c for c in case_ids if c])
        self.seed = seed
        seen = {}
        for name, d in self.variant_dirs.items():
            fp = dir_fingerprint(d)
            if fp in seen:
                raise TaskAdapterError(
                    f"variants {seen[fp]!r} and {name!r} have identical source ({d}); an arm without a "
                    f"diff measures nothing. For an A/A noise band use --control auto instead.")
            seen[fp] = name
        self.mods = {}
        for i, (name, d) in enumerate(self.variant_dirs.items()):
            safe = "".join(ch if ch.isalnum() else "_" for ch in name)
            self.mods[name] = _load_package(f"geak_ab_{i}_{safe}", d)
        self.inputs = [self.runner.make_inputs(c, seed) for c in self.cases]
        self.details = {}

    def _launch_all(self, mod):
        outs = None
        for c, inp in zip(self.cases, self.inputs):
            outs = self.runner.run(mod, inp, c)
        return outs

    def variants(self):
        return {n: (lambda m=m: self._launch_all(m)) for n, m in self.mods.items()}

    def oracle(self, name):
        mod = self.mods[name]
        fails, details = 0, []
        for c, inp in zip(self.cases, self.inputs):
            out = self.runner.run(mod, inp, c)
            sync()
            ref = self.runner.reference(inp, c)
            t = dict(self.runner.tol(c))
            if self.compare is not None:
                ok, det = self.compare(out, ref, t["atol"], t["rtol"], t.get("max_bad_frac", 0.0))
            else:
                hl = _harness_lib()
                ok, err = hl.correct(out, ref, max(t["atol"], t["rtol"]))
                det = f"harness_lib.correct max_rel_err={err:.3e}"
            details.append(f"{c['id']}: {det}")
            fails += 0 if ok else 1
        self.details[name] = details
        print(f"  [ab_task_adapter] {name}: " + " | ".join(details), flush=True)
        return {"tol_fail": float(1 if fails else 0)}

    def fingerprint(self, name):
        return dir_fingerprint(self.variant_dirs[name])


def _harness_lib():
    hl = sys.modules.get("geak_harness_lib")
    if hl is None:
        raise TaskAdapterError("no compare() in the task runner and harness_lib not loaded")
    return hl


def sync():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_ADAPTER = None


def _adapter():
    global _ADAPTER
    if _ADAPTER is None:
        task = _env("AB_TASK_DIR")
        if not task:
            raise TaskAdapterError("AB_TASK_DIR is not set (use ab_bench.py --task <dir>)")
        cases = [c.strip() for c in (_env("AB_TASK_CASES", "") or "").split(",") if c.strip()]
        _ADAPTER = TaskAdapter(task, _env("AB_TASK_VARIANTS", ""), cases, int(_env("AB_TASK_SEED", "0")))
        print(f"  [ab_task_adapter] task={_ADAPTER.task_dir} cases={[c['id'] for c in _ADAPTER.cases]} "
              f"variants=" + ", ".join(f"{n}={d}" for n, d in _ADAPTER.variant_dirs.items()), flush=True)
    return _ADAPTER


# ---- the ab_bench adapter contract (module-level hooks) ------------------------------------------
def variants():
    return _adapter().variants()


def oracle(name):
    return _adapter().oracle(name)


def fingerprint(name):
    return _adapter().fingerprint(name)


# ---- offline selftest (no GPU, no torch) --------------------------------------------------------
_FAKE_RUNNER = '''
import os, sys
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
def make_inputs(case, seed):
    return {"x": case["params"]["x"] + seed}
def run(mod, inp, case):
    return mod.kernel(inp["x"])
def reference(inp, case):
    return 2.0 * inp["x"]
def tol(case):
    return dict(atol=1e-6, rtol=0.0)
def compare(out, ref, atol, rtol, max_bad_frac=0.0):
    ok = abs(out - ref) <= atol + rtol * abs(ref)
    return ok, f"err={abs(out - ref):.3g}"
if __name__ == "__main__":
    raise SystemExit("must not run as main under the adapter")
'''


def _selftest():
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "scripts"))
        with open(os.path.join(td, "scripts", "task_runner.py"), "w") as f:
            f.write(_FAKE_RUNNER)
        with open(os.path.join(td, "test_cases.json"), "w") as f:
            json.dump({"serve_cases": [{"id": "small", "weight": 0.2, "params": {"x": 1.0}},
                                       {"id": "big", "weight": 0.8, "params": {"x": 3.0}}],
                       "edge_cases": [{"id": "edge", "params": {"x": -1.0}}]}, f)
        for d, body in (("source", "def kernel(x):\n    return 2.0 * x\n"),
                        ("source/source_golden", "def kernel(x):\n    return x + x\n"),
                        ("v_bad", "def kernel(x):\n    return 2.0 * x + 1.0\n"),
                        ("v_same", "def kernel(x):\n    return 2.0 * x\n")):
            os.makedirs(os.path.join(td, d), exist_ok=True)
            with open(os.path.join(td, d, "__init__.py"), "w") as f:
                f.write(body)
        # 1. default arms, default case = highest-weight serve case; correctness via the runner's compare
        a = TaskAdapter(td)
        assert list(a.variant_dirs) == ["golden", "source"], a.variant_dirs
        assert [c["id"] for c in a.cases] == ["big"], a.cases
        v = a.variants()
        assert v["source"]() == 6.0 and v["golden"]() == 6.0
        assert a.oracle("source") == {"tol_fail": 0.0} and a.oracle("golden") == {"tol_fail": 0.0}
        assert a.fingerprint("source") != a.fingerprint("golden")
        # 2. a wrong variant fails its oracle; an identical-source variant is refused up front
        b = TaskAdapter(td, "base=source;bad=v_bad", ["small", "edge"], seed=0)
        assert b.oracle("bad") == {"tol_fail": 1.0} and b.oracle("base") == {"tol_fail": 0.0}
        assert b.variants()["base"]() == -2.0  # last case's output; both cases launched
        # 3. refusals name the fix
        for spec, cases, word in (("only=source", (), "two variants"), ("a=source;a=v_bad", (), "twice"),
                                  ("", ("nope",), "unknown case"), ("a=source;b=missing", (), "__init__"),
                                  ("base=source;same=v_same", (), "identical source")):
            try:
                TaskAdapter(td, spec, cases)
                raise AssertionError(f"accepted {spec!r} {cases!r}")
            except TaskAdapterError as e:
                assert word in str(e), (word, str(e))
        os.makedirs(os.path.join(td, "bare", "scripts"))
        with open(os.path.join(td, "bare", "scripts", "task_runner.py"), "w") as f:
            f.write("def run(mod, inp, case):\n    pass\n")
        try:
            TaskAdapter(os.path.join(td, "bare"))
            raise AssertionError("accepted a runner without the contract")
        except TaskAdapterError as e:
            assert "make_inputs" in str(e) and "by hand" in str(e), e
        # 4. end to end through ab_bench --task with a stub harness_lib (CPU)
        with open(os.path.join(td, "harness_lib.py"), "w") as f:
            f.write("import time\n"
                    "def time_op(call, warmup=10, repeats=50, inner=1, graph=False, *, detail=False):\n"
                    "    call()\n"
                    "    d = {'ms': 1.0, 'wall_ms': 1.0, 'timer': 'cuda_event', 'primed': True, "
                    "'host_ms': 0.001, 'cache_condition': {'mode': 'read-evict'}}\n"
                    "    return d if detail else 1.0\n")
        ab = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_bench.py")
        out_json = os.path.join(td, "ab.json")
        env = dict(os.environ, GEAK_HARNESS_LIB=td)
        p = subprocess.run([sys.executable, ab, "--task", td, "--variant", "base=golden",
                            "--variant", "cand=source", "--variant", "bad=v_bad", "--case", "small,big",
                            "--cells", "2", "--iters", "2", "--warmup", "1", "--json", out_json],
                           capture_output=True, text=True, check=False, env=env, timeout=120)
        out = p.stdout + p.stderr
        assert p.returncode in (0, 2), (p.returncode, out)
        res = json.load(open(out_json))
        assert res["excluded_failing_oracle"] == ["bad"], res["excluded_failing_oracle"]
        assert res["baseline"] == "base" and set(res["results"]) == {"base", "cand"}, res["results"]
        assert res["fingerprint_gate"] == "pass" and res["timer"] == "graph", res
        p = subprocess.run([sys.executable, ab, "--task", td, "--module", os.path.abspath(__file__)],
                           capture_output=True, text=True, check=False, env=env, timeout=120)
        assert p.returncode == 2 and "not both" in p.stderr, (p.returncode, p.stderr)
    print("[ab_task_adapter] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    print(__doc__)
    sys.exit(0)
