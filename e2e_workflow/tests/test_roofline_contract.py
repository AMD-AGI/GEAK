"""Contract tests for the roofline profiling path.

WHY THIS EXISTS (the recurring bug class these lock out):
The per-task ``unittest.py`` is authored freehand by the Kernel Extractor agent; only the SHARED
``harness_lib.py`` (immutable, sha-checked, MANDATORY ``h.run_correctness(...)`` entrypoint) is a
stable contract. Historically the auto-generated roofline driver reached into the unittest's PRIVATE,
agent-chosen glue names -- and those drifted every template revision, silently producing all-``null``
roofline metrics in production:
    - ``_wl_cases / _case_dims / synth_case / CURRENT_CALL``   (first dead API)
    - ``_build_correctness(torch)``                             (second dead API -> the Qwen3-14B nulls)
Plus a parallel drift: the accepted-candidate was exported as ``GEAK_GEMM_CANDIDATE`` but the unittest
read a DIFFERENT env var (``META["current_callable_env"]``), so the post phase profiled the wrong callable.

The fix anchors roofline on the harness, not on agent glue: ``run_correctness`` honors a
``GEAK_ROOFLINE_SIG`` hook (candidate-only, single-case tight loop), and the thin driver just runs the
unittest's own ``main()`` with that env armed + bridges the candidate into the meta-declared env name.
These tests assert that contract WITHOUT referencing any private unittest symbol -- so they stay green
across any future unittest template, and would have failed on both historical drifts above.

Modules are imported by file path so the tests are location-agnostic (no assumption about where the
GEAK checkout lives). The harness ``sync`` is stubbed to a no-op so the tests run on a CPU-only box.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "scripts"))
_HARNESS_PATH = os.path.join(_SCRIPTS, "harness_lib.py")
_ROOFLINE_TASK_PATH = os.path.join(_SCRIPTS, "roofline_task.py")


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh_harness():
    """Load a private copy of harness_lib with sync() stubbed, so tests need no torch/CUDA."""
    hl = _load("harness_lib_under_test", _HARNESS_PATH)
    hl.sync = lambda *a, **k: None
    return hl


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)


def _clear_roofline_env():
    for key in ("GEAK_ROOFLINE_SIG", "GEAK_ROOFLINE_WARMUP", "GEAK_ROOFLINE_ITERS"):
        os.environ.pop(key, None)


# --------------------------------------------------------------------------- harness hook contract
def test_hook_runs_candidate_only_at_selected_case():
    """With GEAK_ROOFLINE_SIG set, run_correctness runs ONLY the candidate at the named case
    (warmup+iters launches), never the baseline, and exits(0) before any correctness leg."""
    hl = _fresh_harness()
    cur, base = _Spy(), _Spy()
    cases = [
        {"sig": "decode_M64", "args": {"tag": "A"}, "ref": None},
        {"sig": "prefill_M1024", "args": {"tag": "B"}, "ref": None},
    ]
    os.environ["GEAK_ROOFLINE_SIG"] = "prefill_M1024"
    os.environ["GEAK_ROOFLINE_WARMUP"] = "3"
    os.environ["GEAK_ROOFLINE_ITERS"] = "5"
    try:
        with pytest.raises(SystemExit) as ei:
            hl.run_correctness("decode", eager_cases=cases, baseline_call=base,
                               current_call=cur, random_shapes=[], tol=1e-2, draws=1)
        assert ei.value.code == 0
    finally:
        _clear_roofline_env()
    assert len(cur.calls) == 3 + 5                    # warmup + iters, nothing else
    assert all(a == {"tag": "B"} for a in cur.calls)  # the SELECTED case, not the first
    assert base.calls == []                           # candidate-only: baseline never runs


def test_hook_empty_sig_selects_first_case():
    hl = _fresh_harness()
    cur = _Spy()
    cases = [{"sig": "first", "args": {"tag": "A"}, "ref": None},
             {"sig": "second", "args": {"tag": "B"}, "ref": None}]
    os.environ["GEAK_ROOFLINE_SIG"] = ""      # armed-but-empty => first case
    os.environ["GEAK_ROOFLINE_ITERS"] = "2"
    try:
        with pytest.raises(SystemExit) as ei:
            hl.run_correctness("decode", eager_cases=cases, baseline_call=_Spy(),
                               current_call=cur, random_shapes=[], tol=1e-2, draws=1)
        assert ei.value.code == 0
    finally:
        _clear_roofline_env()
    assert all(a == {"tag": "A"} for a in cur.calls)


def test_hook_unknown_sig_exits_nonzero():
    hl = _fresh_harness()
    os.environ["GEAK_ROOFLINE_SIG"] = "no_such_case"
    try:
        with pytest.raises(SystemExit) as ei:
            hl.run_correctness("decode", eager_cases=[{"sig": "x", "args": {}, "ref": None}],
                               baseline_call=_Spy(), current_call=_Spy(),
                               random_shapes=[], tol=1e-2, draws=1)
        assert ei.value.code != 0
    finally:
        _clear_roofline_env()


def test_hook_is_noop_when_env_unset():
    """Without the env, the hook must NOT fire -- run_correctness proceeds to the real legs.
    We prove it proceeded (past the hook) via a sentinel from the first correctness check,
    without needing torch/CUDA."""
    hl = _fresh_harness()

    class _Sentinel(Exception):
        pass

    def _boom(*a, **k):
        raise _Sentinel

    hl.check_correct_multi = _boom
    _clear_roofline_env()
    with pytest.raises(_Sentinel):
        hl.run_correctness("decode", eager_cases=[{"sig": "x", "args": {}, "ref": None}],
                           baseline_call=_Spy(), current_call=_Spy(),
                           random_shapes=[], tol=1e-2, draws=1)


# --------------------------------------------------------------------------- driver end-to-end contract
def test_driver_runs_unittest_and_bridges_candidate_env(tmp_path):
    """The generated driver (DRIVER_SOURCE) must, with ZERO dependence on private unittest glue:
      (a) arm the harness hook (GEAK_ROOFLINE_SIG),
      (b) bridge GEAK_GEMM_CANDIDATE into the env var name the task DECLARES in meta.json, and
      (c) run the unittest's own main() so the harness hook fires (candidate-only).
    This is the drift guard: a fake unittest that defines NONE of _build_correctness/_current_call/
    _eager_cases still profiles correctly, because the contract is the harness + meta.json."""
    rt = _load("roofline_task_for_driver", _ROOFLINE_TASK_PATH)
    task = tmp_path / "some_task"
    task.mkdir()
    # vendored harness the fake unittest imports (same code under test)
    shutil.copy(_HARNESS_PATH, task / "harness_lib.py")
    # meta.json DECLARES a bespoke candidate env var name -> exercises generic bridging
    (task / "meta.json").write_text(json.dumps({"current_callable_env": "MY_TASK_CAND"}))
    (task / "roofline_driver.py").write_text(rt.DRIVER_SOURCE)

    # A fake unittest that uses NONE of the historical private glue names. It records the env the
    # driver set, stubs sync (no torch), and routes through the mandated harness entrypoint.
    (task / "unittest.py").write_text(
        "import importlib.util, json, os\n"
        "HERE = os.path.dirname(os.path.abspath(__file__))\n"
        "spec = importlib.util.spec_from_file_location('harness_lib', os.path.join(HERE, 'harness_lib.py'))\n"
        "h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
        "h.sync = lambda *a, **k: None\n"
        "seen = {'calls': 0}\n"
        "def _cur(args):\n"
        "    seen['calls'] += 1\n"
        "def _base(args):\n"
        "    raise AssertionError('baseline must not run under the roofline hook')\n"
        "CASES = [{'sig': 'decode_M64', 'args': {'m': 64}, 'ref': None},\n"
        "         {'sig': 'prefill_M1024', 'args': {'m': 1024}, 'ref': None}]\n"
        "def main():\n"
        "    json.dump({'GEAK_ROOFLINE_SIG': os.environ.get('GEAK_ROOFLINE_SIG'),\n"
        "               'MY_TASK_CAND': os.environ.get('MY_TASK_CAND'),\n"
        "               'CURRENT_GEMM_CALLABLE': os.environ.get('CURRENT_GEMM_CALLABLE')},\n"
        "              open(os.path.join(HERE, 'env_snapshot.json'), 'w'))\n"
        "    h.run_correctness('decode', eager_cases=CASES, baseline_call=_base,\n"
        "                      current_call=_cur, random_shapes=[], tol=1e-2, draws=1)\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    env = os.environ.copy()
    env["GEAK_GEMM_CANDIDATE"] = "mypkg:fast_gemm"   # what roofline_task exports in the post phase
    _clear_roofline_env()
    proc = subprocess.run(
        [sys.executable, str(task / "roofline_driver.py"),
         "--unittest", str(task / "unittest.py"), "--sig", "prefill_M1024",
         "--warmup", "2", "--iters", "3"],
        capture_output=True, text=True, env=env, timeout=120)

    assert proc.returncode == 0, proc.stderr
    assert "ROOFLINE_HOOK: ran case=prefill_M1024" in (proc.stdout + proc.stderr)
    snap = json.loads((task / "env_snapshot.json").read_text())
    assert snap["GEAK_ROOFLINE_SIG"] == "prefill_M1024"          # (a) hook armed
    assert snap["MY_TASK_CAND"] == "mypkg:fast_gemm"             # (b) bridged to DECLARED env name
    assert snap["CURRENT_GEMM_CALLABLE"] == "mypkg:fast_gemm"    # (b) + known-name fallback
