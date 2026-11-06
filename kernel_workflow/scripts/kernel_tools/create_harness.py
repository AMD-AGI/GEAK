#!/usr/bin/env python3
"""Triton-family harness generator (adapted from perf-geak; defaults to Gluon).

A GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/; the Gluon pack keeps a shim at
scripts/create_harness.py).

Generates an isolated test-harness skeleton for a Triton/Gluon kernel with four
modes (--correctness / --profile / --benchmark / --full-benchmark) plus --sweep-one.
The generated harness includes a Gluon layout-factory skeleton and TODO sections the
agent must fill (reference shapes, oracle, launch). It does NOT auto-edit the user's kernel.

Measurement and correctness are NOT implemented in the skeleton: it calls GEAK's
e2e_workflow/scripts/harness_lib.py (`time_op(detail=True)`: CUDA events, per-sample sync,
read-evict flush, median; `check_correct_multi` / `correct`). The generated file emits, per
case, `GEAK_RESULT_LATENCY_MS=<float> case_id=<id>` (kernel_workflow/roles/benchmark_engineer.md),
and per-case correctness JSON with exit 0 pass | 1 fail | 2 not runnable (unfilled skeleton:
HARNESS_UNFILLED, rc 2). Its numbers are search/screening numbers; acceptance comes from GEAK
verify (harness_lib.measure_legs).

Usage:
  python3 create_harness.py --kernel-path /path/to/kernel.py \
      --kernel-name my_kernel --output /path/to/harness.py [--kernel-type triton|gluon]
      [--harness-lib /path/to/e2e_workflow/scripts/harness_lib.py]
"""
import argparse
import re
import sys
from pathlib import Path


def detect_kernel_type(kernel_path: str) -> str:
    p = Path(kernel_path)
    if p.suffix.lower() == ".py":
        txt = p.read_text(errors="ignore")
        if "gluon.jit" in txt or "experimental.gluon" in txt:
            return "gluon"
        if "@triton.jit" in txt or "tl." in txt:
            return "triton"
    if p.suffix.lower() in (".hip", ".cu", ".cpp"):
        return "hip"
    return "unknown"


def extract_triton_kernels(kernel_path: str):
    txt = Path(kernel_path).read_text(errors="ignore")
    pat = re.compile(r"@(?:triton|gluon)\.jit\s*\n\s*def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)
    out = []
    for m in pat.finditer(txt):
        params = [p.split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
        out.append({"name": m.group(1), "params": params})
    return out


TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated tile-programming harness for `{kernel_name}` ({kernel_type}).

Fill every TODO before use. Keep imports/cache/shapes IDENTICAL across the plain
baseline, the Gluon anchor, and every candidate (benchmark hygiene).

Timing and correctness go through GEAK's e2e_workflow/scripts/harness_lib.py -- the single
owner of measurement in GEAK: CUDA-event device time, a sync per sample, a read-evict cache
flush before every sample, median. Do not hand-roll a timing loop here. Resolution order for
harness_lib: $GEAK_HARNESS_LIB (file or dir; must exist when set), a vendored harness_lib.py
beside this file, then the GEAK checkout this skeleton was generated from.

Output contract (kernel_workflow/roles/benchmark_engineer.md):
  --correctness      one JSON line per case {{"case", "correct", "max_rel_err"}}, then
                     CORRECTNESS PASS|FAIL; exit 0 pass | 1 fail | 2 not runnable
  --benchmark        one `GEAK_RESULT_LATENCY_MS=<ms> case_id=<id>` line PER CASE (30 iters,
  --full-benchmark   10 warmup / 100 iters, 10 warmup) + `GEAK_RESULT_GEOMEAN_MS=<ms>`
  --profile          one launch, minimal allocations, for a profiler attach
  --sweep-one        one config (PA_* env) -> PA_METER + one GEAK_RESULT_LATENCY_MS line
An unfilled skeleton REFUSES every mode: HARNESS_UNFILLED on stderr, exit 2.
These numbers are SEARCH/SCREENING numbers. Acceptance numbers come from GEAK verify
(harness_lib.measure_legs: fresh process per leg, interleaved pairs).
"""
import argparse
import json
import math
import os
import sys

import torch

# --- imports -------------------------------------------------------------
# Plain Triton:
#   import triton, triton.language as tl
# Gluon (supported path):
#   from triton.experimental import gluon
#   from triton.experimental.gluon import language as gl
# TODO: import the target kernel module ({kernel_path})

DEVICE = "cuda"  # ROCm exposes HIP devices through the torch cuda API
# The GEAK checkout this skeleton was generated from (last-resort harness_lib location).
_GEN_HARNESS_LIB = {harness_lib_path!r}


# --- Gluon layout factory skeleton (host-side; pass as gl.constexpr) -----
def make_blocked_layout(size_per_thread, threads_per_warp, warps_per_cta, order):
    # from triton.experimental.gluon import language as gl
    # return gl.BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)
    raise NotImplementedError("TODO: build the BlockedLayout for the load tile")


# The kernel's OWN default for every knob the sweep may pin. Filling this is what lets a sweep
# certify a one-knob winner: `plain_autotune.py`'s widest exhaustive phase is the default + one-knob
# floor, whose entries name a single knob and leave the rest at these values -- values the sweeper
# cannot see. `sweep_one` echoes the merged result as PA_EFFECTIVE_CONFIG so such a winner can be
# completed into a full grid point and walked like any other; left empty, that winner is still
# reported, just as an uncertified order-0 result.
DEFAULT_CONFIG = {{
    # TODO: the kernel's default value for each sweepable knob
    # "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2,
}}

SHAPES_QUICK = [
    # TODO: a few representative shapes
    # dict(M=4096, N=4096, K=8192, dtype=torch.float16),
]
SHAPES_FULL = [
    # TODO: the full shape stream
]
# Mixed tolerance passed to harness_lib.correct: |out-ref| <= tol*RMS(ref) + tol*|ref|.
TOL = float(os.environ.get("HARNESS_TOL", "2e-2"))   # TODO: tune per dtype


def make_inputs(shape):
    # TODO: allocate inputs (use torch.empty/rand on DEVICE; match dtype)
    raise NotImplementedError


def run_kernel(inputs):
    # TODO: launch the target kernel; return a FRESH output tensor (never a reused static buffer)
    raise NotImplementedError


def reference(inputs):
    # TODO: torch reference for the correctness oracle
    raise NotImplementedError


def case_id(shape, i):
    """Stable per-case id printed beside every number (same across processes and variants)."""
    if isinstance(shape, dict):
        return ",".join(f"{{k}}={{str(v).replace('torch.', '')}}" for k, v in shape.items())
    return f"case{{i}}:{{shape}}"


def _load_harness_lib():
    """GEAK's harness_lib, or exit 2 with the reason (never a silent fallback timer)."""
    import importlib.util
    env = os.environ.get("GEAK_HARNESS_LIB")
    if env:
        cands = [env if env.endswith(".py") else os.path.join(env, "harness_lib.py")]
        if not os.path.isfile(cands[0]):
            print(f"HARNESS_LIB_UNAVAILABLE: GEAK_HARNESS_LIB={{env}} does not name harness_lib.py",
                  file=sys.stderr)
            raise SystemExit(2)
    else:
        cands = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness_lib.py"),
                 _GEN_HARNESS_LIB]
    for path in cands:
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("harness_lib", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    print("HARNESS_LIB_UNAVAILABLE: e2e_workflow/scripts/harness_lib.py not found (set "
          "GEAK_HARNESS_LIB, or vendor harness_lib.py beside this harness). Refusing to time "
          "with a hand-rolled loop.", file=sys.stderr)
    raise SystemExit(2)


def check_correctness():
    """harness_lib.check_correct_multi (built on harness_lib.correct): every case's output is kept
    live until all are compared, which also catches an aliased/static output buffer."""
    h = _load_harness_lib()
    cases = []
    for i, shape in enumerate(SHAPES_QUICK or SHAPES_FULL):
        inp = make_inputs(shape)
        cases.append({{"args": inp, "ref": reference(inp), "sig": case_id(shape, i)}})
    ok, per_case = h.check_correct_multi(run_kernel, cases, TOL)
    for row in per_case:
        print(json.dumps(row))
    print("CORRECTNESS PASS" if ok else "CORRECTNESS FAIL")
    return ok


def _time(inp, warmup, repeats):
    """(ms, receipt) via harness_lib.time_op(detail=True). None ms = the launch raised."""
    h = _load_harness_lib()
    d = h.time_op(lambda: run_kernel(inp), warmup=warmup, repeats=repeats, detail=True)
    if d is None:
        return None, {{}}
    return d["ms"], d


def benchmark(full=False):
    shapes = SHAPES_FULL if full else SHAPES_QUICK
    warmup, repeats = (10, 100) if full else (10, 30)
    repeats = int(os.environ.get("BENCH_REPEATS", repeats))
    _load_harness_lib()   # fail fast (rc 2, HARNESS_LIB_UNAVAILABLE) before building any input
    per_case = []
    receipt = {{}}
    for i, shape in enumerate(shapes):
        ms, receipt = _time(make_inputs(shape), warmup, repeats)
        if ms is None:
            print(f"BENCH_FAILED case_id={{case_id(shape, i)}} (the launch raised)")
            raise SystemExit(1)
        per_case.append(ms)
        print(f"GEAK_RESULT_LATENCY_MS={{ms:.6f}} case_id={{case_id(shape, i)}}")
    print("BENCH_PROTOCOL " + json.dumps({{"timer": receipt.get("timer"), "warmup": warmup,
                                          "repeats": repeats,
                                          "cache_condition": receipt.get("cache_condition"),
                                          "primed": receipt.get("primed")}}))
    if per_case:
        geo = math.exp(sum(math.log(x) for x in per_case) / len(per_case))
        print(f"GEAK_RESULT_GEOMEAN_MS={{geo:.6f}}")


def profile_once():
    # single run, minimal allocations, for rocprofv3 / ATT (run under kernel_workflow gpu_lock.sh)
    run_kernel(make_inputs((SHAPES_QUICK or SHAPES_FULL)[0]))


def sweep_one():
    # Stage-Plain config sweep worker: plain_autotune.py forks ONE of these per config
    # (benchmark-hygiene: a bad config crashes the LLVM backend, so isolate it). The pinned knobs
    # arrive as PA_<KNOB> env vars (BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages/GROUP_SIZE_M/SPLIT_K).
    # Levers that change data produced OUTSIDE the kernel (split-K reduce buffers, quant scale
    # layout) arrive in PA_REBUILD_DEPENDENT (comma list) -> rebuild that dependent data per config,
    # else the sweep is INVALID (fixed dependent data is the P8 cross-boundary-coupling trap).
    h = _load_harness_lib()
    cfg = {{k[3:]: v for k, v in os.environ.items() if k.startswith("PA_") and k != "PA_REBUILD_DEPENDENT"}}
    rebuild = [s for s in os.environ.get("PA_REBUILD_DEPENDENT", "").split(",") if s]
    # What this run ACTUALLY used: the pinned knobs on top of the kernel's own defaults. See
    # DEFAULT_CONFIG above for why the sweeper needs to be told.
    eff = dict(DEFAULT_CONFIG)
    for _k, _v in cfg.items():
        try:
            eff[_k] = int(_v)
        except ValueError:
            eff[_k] = _v
    if eff:
        print("PA_EFFECTIVE_CONFIG=" + json.dumps(eff))
    # TODO: thread cfg (int-cast the knobs you use) into make_inputs / run_kernel -- a tile knob is an
    # algorithmic constexpr (check correctness), a launch knob (num_warps/num_stages) is scheduling.
    # TODO: if rebuild is non-empty, regenerate the dependent data for THIS config before timing.
    shape = (SHAPES_QUICK or SHAPES_FULL)[0]
    inp = make_inputs(shape)
    ok, err = h.correct(run_kernel(inp), reference(inp), TOL)
    if not ok:
        print(f"SWEEP_CONFIG_INCORRECT max_rel_err={{err}}")   # plain_autotune records FAILED, skips timing
        raise SystemExit(2)
    warmup, repeats = 5, int(os.environ.get("BENCH_REPEATS", "20"))
    ms, receipt = _time(inp, warmup, repeats)
    if ms is None:
        print("SWEEP_CONFIG_LAUNCH_FAILED")
        raise SystemExit(2)
    # ATTEST the protocol actually used, so the sweep's record is a measurement and not an intention.
    # harness_lib ALWAYS read-evicts before each sample (no hot mode): a sweeper that exported
    # BENCH_CACHE=hot reads "read-evict" back here and must not record the run as hot.
    if os.environ.get("BENCH_CACHE", "").lower() == "hot":
        print("BENCH_NOTE: BENCH_CACHE=hot is not offered -- harness_lib read-evicts every sample",
              file=sys.stderr)
    print("PA_METER=" + json.dumps({{"budget_ms": None, "cache": "read-evict",
                                     "timer": receipt.get("timer"), "iters": repeats,
                                     "warmup": warmup, "primed": receipt.get("primed")}}))
    print(f"GEAK_RESULT_LATENCY_MS={{ms:.6f}} case_id={{case_id(shape, 0)}}")   # the marker plain_autotune parses


def _require_filled():
    """Refuse to run while this is still the generated skeleton.

    Both shape lists ship empty and every mode loops over them, so an unfilled harness printed
    nothing and exited 0 -- indistinguishable from a clean benchmark whose marker the caller
    missed. --correctness was worse: zero shapes checked still printed CORRECTNESS PASS.
    """
    if not (SHAPES_QUICK or SHAPES_FULL):
        print("HARNESS_UNFILLED: SHAPES_QUICK and SHAPES_FULL are both empty -- this is still "
              "the generated skeleton. Fill the TODOs before measuring anything.", file=sys.stderr)
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--full-benchmark", action="store_true")
    ap.add_argument("--sweep-one", action="store_true",
                    help="bench ONE config (knobs from PA_* env); used by plain_autotune.py")
    a = ap.parse_args()
    _require_filled()
    if a.correctness:
        raise SystemExit(0 if check_correctness() else 1)
    elif a.profile:
        profile_once()
    elif a.sweep_one:
        sweep_one()
    elif a.full_benchmark:
        benchmark(full=True)
    else:
        benchmark(full=False)


if __name__ == "__main__":
    main()
'''


def default_harness_lib() -> str:
    """e2e_workflow/scripts/harness_lib.py of the GEAK checkout this generator lives in."""
    return str((Path(__file__).resolve().parents[3] / "e2e_workflow" / "scripts"
                / "harness_lib.py"))


def render(kernel_name, kernel_type, kernel_path, harness_lib_path=None) -> str:
    return TEMPLATE.format(kernel_name=kernel_name, kernel_type=kernel_type,
                           kernel_path=kernel_path,
                           harness_lib_path=harness_lib_path or default_harness_lib())


def selftest() -> int:
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness_stub_env import assert_refuses, run_skeleton

    assert Path(default_harness_lib()).is_file(), \
        f"harness_lib not at {default_harness_lib()} -- the generated harness could not time"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        kernel = d / "k.py"
        kernel.write_text("import triton\n@triton.jit\ndef my_kernel(a, b):\n    pass\n")
        harness = d / "harness.py"
        harness.write_text(render("my_kernel", "triton", str(kernel)))
        compile(harness.read_text(), str(harness), "exec")  # the format() escaping is correct

        # The measurement contract is part of every generated harness, not a habit of whoever
        # fills it in: GEAK's harness_lib times (CUDA events, per-sample sync, read-evict, median)
        # and judges correctness, one latency line per case carries its case id, and the sweep
        # worker attests the protocol it actually used.
        body = harness.read_text()
        for needle in ("h.time_op(", "detail=True", "h.check_correct_multi(", "h.correct(",
                       'GEAK_RESULT_LATENCY_MS={ms:.6f} case_id=', 'print("PA_METER="',
                       '"cache": "read-evict"', "GEAK_HARNESS_LIB", "HARNESS_LIB_UNAVAILABLE",
                       "raise SystemExit(0 if check_correctness() else 1)"):
            assert needle in body, f"the generated harness must carry the contract: {needle}"
        for banned in ("_FLUSH_BUF.zero_()", "time.perf_counter()"):
            assert banned not in body, f"hand-rolled timing/write-evict left in the template: {banned}"

        # Before the guard, --benchmark exited 0 having printed nothing and --correctness exited 0
        # printing CORRECTNESS PASS on zero shapes checked. rc 2 + HARNESS_UNFILLED, every mode.
        assert_refuses(harness, ["--correctness", "--benchmark", "--full-benchmark",
                                 "--profile", "--sweep-one"])

        # A filled harness must NOT be refused -- guards against a future check that is so eager
        # it blocks the real thing.
        filled = harness.read_text().replace("SHAPES_QUICK = [", "SHAPES_QUICK = [dict(M=1),")
        harness.write_text(filled)
        rc, out = run_skeleton(harness, ["--benchmark"])
        assert "HARNESS_UNFILLED" not in out, f"filled harness still refused\n{out}"

        # An explicit GEAK_HARNESS_LIB that does not exist is an error (rc 2), not a fallback.
        import os as _os
        import subprocess as _sp
        env = dict(_os.environ, PYTHONPATH=str(d), GEAK_HARNESS_LIB=str(d / "nope"))
        r = _sp.run([sys.executable, str(harness), "--benchmark"], capture_output=True, text=True,
                    env=env, cwd=str(d), timeout=60)
        assert r.returncode == 2 and "HARNESS_LIB_UNAVAILABLE" in r.stderr, (r.returncode, r.stderr)

        # Per-case output, end to end, with a stub harness_lib vendored beside the harness: one
        # latency line per case with its id, a geomean under a DIFFERENT marker, per-case JSON
        # correctness rows that parse_correctness --parse geak reads, and exit 1 on a failing case.
        (d / "harness_lib.py").write_text(
            "def time_op(call, warmup=10, repeats=50, inner=1, graph=False, *, detail=False):\n"
            "    call()\n"
            "    return {'ms': 0.5, 'wall_ms': 0.6, 'timer': 'stub', 'cache_condition': "
            "{'mode': 'read-evict'}, 'primed': True, 'host_ms': 0.01}\n"
            "def correct(out, ref, tol):\n"
            "    return (out == ref), (0.0 if out == ref else 1.0)\n"
            "def check_correct_multi(call, cases, tol):\n"
            "    rows = [{'case': c['sig'], 'correct': call(c['args']) == c['ref'], "
            "'max_rel_err': 0.0} for c in cases]\n"
            "    return all(r['correct'] for r in rows), rows\n")
        filled = (filled.replace("SHAPES_QUICK = [dict(M=1),", "SHAPES_QUICK = [dict(M=1), dict(M=2),")
                  .replace("def make_inputs(shape):\n    # TODO: allocate inputs (use torch.empty/rand on DEVICE; match dtype)\n    raise NotImplementedError",
                           "def make_inputs(shape):\n    return shape['M']")
                  .replace("def run_kernel(inputs):\n    # TODO: launch the target kernel; return a FRESH output tensor (never a reused static buffer)\n    raise NotImplementedError",
                           "def run_kernel(inputs):\n    return inputs * 2")
                  .replace("def reference(inputs):\n    # TODO: torch reference for the correctness oracle\n    raise NotImplementedError",
                           "def reference(inputs):\n    return inputs * (3 if inputs == 2 else 2)"))
        harness.write_text(filled)
        env = dict(_os.environ, PYTHONPATH=str(d))
        env.pop("GEAK_HARNESS_LIB", None)
        r = _sp.run([sys.executable, str(harness), "--benchmark"], capture_output=True,
                    text=True, env=env, cwd=str(d), timeout=60)
        lat = [ln for ln in r.stdout.splitlines() if ln.startswith("GEAK_RESULT_LATENCY_MS=")]
        assert r.returncode == 0 and len(lat) == 2, (r.returncode, r.stdout, r.stderr)
        assert all("case_id=M=" in ln for ln in lat), lat
        assert "GEAK_RESULT_GEOMEAN_MS=" in r.stdout and '"read-evict"' in r.stdout, r.stdout
        r = _sp.run([sys.executable, str(harness), "--correctness"], capture_output=True,
                    text=True, env=env, cwd=str(d), timeout=60)
        assert r.returncode == 1 and "CORRECTNESS FAIL" in r.stdout, (r.returncode, r.stdout, r.stderr)
        from parse_correctness import parse_geak
        verdict = parse_geak(r.stdout, r.returncode)
        assert verdict["status"] == "fail" and len(verdict["cases"]) == 2, verdict
        r = _sp.run([sys.executable, str(harness), "--sweep-one"], capture_output=True,
                    text=True, env=env, cwd=str(d), timeout=60)
        assert r.returncode == 0 and r.stdout.count("GEAK_RESULT_LATENCY_MS=") == 1, (r.stdout, r.stderr)
        assert '"cache": "read-evict"' in r.stdout, r.stdout

    print("SELFTEST PASS")
    return 0


def main():
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel-path", required=True)
    ap.add_argument("--kernel-name", default=None)
    ap.add_argument("--kernel-type", default=None, choices=["triton", "gluon", "hip", "unknown"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--harness-lib", default=None,
                    help="GEAK e2e_workflow/scripts/harness_lib.py to bake in as the last-resort "
                         "location (default: this checkout's); $GEAK_HARNESS_LIB overrides at run time")
    a = ap.parse_args()

    ktype = a.kernel_type or detect_kernel_type(a.kernel_path)
    kname = a.kernel_name
    if not kname:
        ks = extract_triton_kernels(a.kernel_path)
        kname = ks[0]["name"] if ks else "kernel"

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    hl = a.harness_lib or default_harness_lib()
    if not Path(hl).is_file():
        sys.exit(f"harness_lib not found at {hl} -- pass --harness-lib <path to "
                 f"e2e_workflow/scripts/harness_lib.py>; the generated harness times through it")
    out.write_text(render(kname, ktype, a.kernel_path, hl))
    print(f"wrote harness skeleton -> {out}  (kernel_type={ktype}, kernel={kname})")
    print("Fill every TODO before use; keep imports/cache/shapes identical across "
          "baseline and candidates.")


if __name__ == "__main__":
    main()
