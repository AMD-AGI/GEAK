#!/usr/bin/env python3
"""Pure-logic coverage for tools/ep_task_runner.py — the parts that run without a
GPU: oracle dispatch, the perf-parse regexes, pass/fail gating, and the
missing-tool guard. The hipcc build + on-GPU run are verified end-to-end on the
gfx1151 box, not here.

    python3 -m pytest tools/tests/test_ep_task_runner.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ep_task_runner as mod  # noqa: E402


# -- dispatch table -------------------------------------------------------------

def test_dispatch_has_three_oracles():
    assert set(mod.DISPATCH) == {"fast", "numeric", "selfcheck"}


@pytest.mark.parametrize("oracle", ["fast", "numeric", "selfcheck"])
def test_dispatch_triples_are_callable(oracle):
    compile_fn, correct_fn, perf_fn = mod.DISPATCH[oracle]
    assert all(callable(f) for f in (compile_fn, correct_fn, perf_fn))


# -- missing-tool guard (the FileNotFoundError fix) -----------------------------

def test_missing_tool_reports_127():
    r = mod._MissingTool("no hipcc")
    assert r.returncode == 127 and r.stdout == "" and "no hipcc" in r.stderr


def test_run_returns_missing_tool_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError(2, "No such file", "hipcc")
    monkeypatch.setattr(mod.subprocess, "run", boom)
    r = mod._run(["hipcc", "-c", "x.hip"])
    assert r.returncode == 127
    assert "tool not found" in r.stderr


# -- arch detection fallback ----------------------------------------------------

def test_detect_offload_arch_falls_back_when_rocminfo_absent(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError(2, "No such file", "rocminfo")
    monkeypatch.setattr(mod.subprocess, "run", boom)
    assert mod._detect_offload_arch("gfx1151") == "gfx1151"


def test_detect_offload_arch_prefers_rocminfo(monkeypatch):
    class R:
        stdout = "  Name:  gfx942\n  something gfx900 noise\n"
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: R())
    assert mod._detect_offload_arch("gfx1151") == "gfx942"


# -- selfcheck: pass/fail gating + GPU-ms parse ---------------------------------

_SELFCHECK_OK = """--- 4096x2880 ---
Verify: max_abs_err=2.146e-06  PASS
CPU: 12.2299 ms
GPU: 0.5332 ms
Speedup: 22.9x
"""

_SELFCHECK_FAIL = """--- 4096x2880 ---
Verify: max_abs_err=9.9e-01  FAIL
CPU: 12.2 ms
GPU: 0.5 ms
"""


def test_gpu_ms_regex_extracts_first_gpu_line():
    m = mod._GPU_MS_RE.search(_SELFCHECK_OK)
    assert m and float(m.group(1)) == 0.5332


def test_selfcheck_correctness_pass(monkeypatch):
    monkeypatch.setattr(mod, "_selfcheck_run_shape",
                        lambda meta, exe, shape: (_SELFCHECK_OK, True))
    meta = {"kernel_file": "k.hip", "test_cpp": "t.cpp",
            "shapes": [{"M": 4096, "N": 2880}]}
    ok, err = mod.selfcheck_correctness(meta)
    assert ok and err is None


def test_selfcheck_correctness_fail_names_case(monkeypatch):
    monkeypatch.setattr(mod, "_selfcheck_run_shape",
                        lambda meta, exe, shape: (_SELFCHECK_FAIL, False))
    meta = {"kernel_file": "k.hip", "test_cpp": "t.cpp",
            "shapes": [{"M": 4096, "N": 2880}]}
    ok, err = mod.selfcheck_correctness(meta)
    assert not ok and "case 0" in err


def test_selfcheck_performance_one_case_per_shape(monkeypatch):
    monkeypatch.setattr(mod, "_selfcheck_run_shape",
                        lambda meta, exe, shape: (_SELFCHECK_OK, True))
    meta = {"kernel_file": "k.hip", "test_cpp": "t.cpp",
            "shapes": [{"M": 4096, "N": 2880, "iters": 50},
                       {"M": 8192, "N": 4096, "iters": 50}]}
    cases = mod.selfcheck_performance(meta)
    assert [c["test_case_id"] for c in cases] == ["4096x2880", "8192x4096"]
    assert cases[0]["execution_time_ms"] == 0.5332
    assert cases[0]["params"]["iters"] == 50


# -- fast oracle: section-tagged median parse + bit targeting -------------------

_FAST_OUT = """  --- u2 ---
  Median: 0.1234 ms, 100 GFLOPS
  --- u4 ---
  Median: 0.2000 ms, 90 GFLOPS
"""


def test_fast_performance_tags_median_by_section_and_filters_bits(monkeypatch):
    monkeypatch.setattr(mod, "_fast_run_shape",
                        lambda meta, exe, shape, root: (_FAST_OUT, True))
    meta = {"kernel_file": "k.hip", "test_cpp": "t.cpp", "gen_data_py": "g.py",
            "bits": [2], "shapes": [{"M": 128, "K": 2880, "N": 2880}]}
    cases = mod.fast_performance(meta)
    # only u2 targeted -> u4 dropped
    assert len(cases) == 1
    assert cases[0]["params"]["bits"] == "u2"
    assert cases[0]["execution_time_ms"] == 0.1234
    assert cases[0]["test_case_id"].endswith("_u2")


def test_fast_performance_no_bits_keeps_all_sections(monkeypatch):
    monkeypatch.setattr(mod, "_fast_run_shape",
                        lambda meta, exe, shape, root: (_FAST_OUT, True))
    meta = {"kernel_file": "k.hip", "test_cpp": "t.cpp", "gen_data_py": "g.py",
            "bits": [], "shapes": [{"M": 128, "K": 2880, "N": 2880}]}
    cases = mod.fast_performance(meta)
    assert {c["params"]["bits"] for c in cases} == {"u2", "u4"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
