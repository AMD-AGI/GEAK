#!/usr/bin/env python3
"""Pure-logic coverage for tools/ep_task_runner.py — the parts that run without a
GPU: oracle dispatch, the perf-parse regexes, pass/fail gating, and the
missing-tool guard. The hipcc build + on-GPU run are verified end-to-end on the
gfx1151 box, not here.

    python3 -m pytest tools/tests/test_ep_task_runner.py
"""
import json
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


# -- offload arch: task arch is authoritative, env must agree -------------------

@pytest.fixture
def no_arch_env(monkeypatch):
    for var in mod._ARCH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_resolve_arch_uses_task_arch_without_env(no_arch_env):
    assert mod._resolve_offload_arch("gfx1151") == ("gfx1151", None)


@pytest.mark.parametrize("var,val", [
    ("GPU_ARCHS", "gfx950"),
    ("PYTORCH_ROCM_ARCH", "gfx942;gfx950"),
    ("GPU_ARCHS", "gfx950:xnack-"),
])
def test_resolve_arch_accepts_agreeing_env(no_arch_env, var, val):
    no_arch_env.setenv(var, val)
    assert mod._resolve_offload_arch("gfx950") == ("gfx950", None)


@pytest.mark.parametrize("var", ["GPU_ARCHS", "PYTORCH_ROCM_ARCH"])
def test_resolve_arch_refuses_mismatch_instead_of_overriding(no_arch_env, var):
    no_arch_env.setenv(var, "gfx950")
    arch, err = mod._resolve_offload_arch("gfx1151")
    assert arch is None and var in err and "gfx1151" in err


def test_hipcc_build_fails_on_arch_mismatch_without_compiling(no_arch_env):
    no_arch_env.setenv("PYTORCH_ROCM_ARCH", "gfx950")
    no_arch_env.setattr(mod, "_run", lambda *a, **k: pytest.fail("must not compile"))
    ok, err = mod.selfcheck_compile({"arch": "gfx1151", "kernel_file": "k.hip",
                                     "test_cpp": "t.cpp"})
    assert not ok and "gfx1151" in err


def test_only_fast_kernel_tu_gets_fp_contract(no_arch_env):
    cmds = []

    class R:
        returncode = 0
    no_arch_env.setattr(mod, "_run", lambda cmd, **k: cmds.append(cmd) or R())
    meta = {"arch": "gfx950", "kernel_file": "k.hip", "test_cpp": "t.cpp"}
    assert mod.fast_compile(meta) == (True, None)
    assert [("-ffp-contract=fast" in c) for c in cmds] == [False, True, False]
    cmds.clear()
    assert mod.selfcheck_compile(meta) == (True, None)
    assert not any("-ffp-contract=fast" in c for c in cmds)


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


class _Proc:
    def __init__(self, returncode, stdout=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, ""


@pytest.mark.parametrize("rc,expected", [(0, True), (1, False), (2, False)])
def test_selfcheck_verdict_is_the_harness_exit_code(monkeypatch, rc, expected):
    # Output text is irrelevant: a "FAIL" substring with rc 0 still passes.
    monkeypatch.setattr(mod, "_run", lambda *a, **k: _Proc(rc, "FAIL-ish text\n"))
    _, passed = mod._selfcheck_run_shape({}, "exe", {"M": 1, "N": 1})
    assert passed is expected


_SELFCHECK_META = {"kernel_file": "k.hip", "test_cpp": "t.cpp",
                   "shapes": [{"M": 4096, "N": 2880, "iters": 50},
                              {"M": 8192, "N": 4096, "iters": 50}]}


def test_selfcheck_performance_one_case_per_shape(monkeypatch):
    monkeypatch.setattr(mod, "_selfcheck_run_shape",
                        lambda meta, exe, shape: (_SELFCHECK_OK, True))
    ok, cases, err = mod.selfcheck_performance(_SELFCHECK_META)
    assert ok and err is None
    assert [c["test_case_id"] for c in cases] == ["4096x2880", "8192x4096"]
    assert cases[0]["execution_time_ms"] == 0.5332
    assert cases[0]["params"]["iters"] == 50


def test_selfcheck_performance_fails_on_wrong_kernel(monkeypatch):
    # A numerically wrong kernel must not be reported as a (faster) timing.
    monkeypatch.setattr(mod, "_selfcheck_run_shape",
                        lambda meta, exe, shape: (_SELFCHECK_FAIL, False))
    ok, _, err = mod.selfcheck_performance(_SELFCHECK_META)
    assert not ok and "case 0" in err


def test_selfcheck_performance_fails_without_gpu_timing(monkeypatch):
    monkeypatch.setattr(mod, "_selfcheck_run_shape",
                        lambda meta, exe, shape: ("tool not found: x", True))
    ok, _, err = mod.selfcheck_performance(_SELFCHECK_META)
    assert not ok and "no GPU timing" in err


# -- fast oracle: section-tagged median parse + bit targeting -------------------

_FAST_OUT = """  --- u2 ---
  Median: 0.1234 ms, 100 GFLOPS
  --- u4 ---
  Median: 0.2000 ms, 90 GFLOPS
"""


def _fast_meta(bits):
    return {"kernel_file": "k.hip", "test_cpp": "t.cpp", "gen_data_py": "g.py",
            "bits": bits, "shapes": [{"M": 128, "K": 2880, "N": 2880}]}


def test_fast_performance_tags_median_by_section_and_filters_bits(monkeypatch):
    monkeypatch.setattr(mod, "_fast_run_shape",
                        lambda meta, exe, shape, root: (_FAST_OUT, True))
    ok, cases, _ = mod.fast_performance(_fast_meta([2]))
    # only u2 targeted -> u4 dropped
    assert ok and len(cases) == 1
    assert cases[0]["params"]["bits"] == "u2"
    assert cases[0]["execution_time_ms"] == 0.1234
    assert cases[0]["test_case_id"] == "128x2880x2880_gs128_z_u2"


def test_fast_performance_no_bits_keeps_all_sections(monkeypatch):
    monkeypatch.setattr(mod, "_fast_run_shape",
                        lambda meta, exe, shape, root: (_FAST_OUT, True))
    ok, cases, _ = mod.fast_performance(_fast_meta([]))
    assert ok and {c["params"]["bits"] for c in cases} == {"u2", "u4"}


def test_fast_performance_fails_when_shape_fails(monkeypatch):
    monkeypatch.setattr(mod, "_fast_run_shape",
                        lambda meta, exe, shape, root: (_FAST_OUT, False))
    ok, _, err = mod.fast_performance(_fast_meta([2]))
    assert not ok and "case 0" in err


def test_fast_performance_fails_when_no_median_parsed(monkeypatch):
    monkeypatch.setattr(mod, "_fast_run_shape",
                        lambda meta, exe, shape, root: ("ALL PASSED\n", True))
    ok, _, err = mod.fast_performance(_fast_meta([2]))
    assert not ok and "no Median" in err


# -- numeric oracle: overlay + rebuild + snapshot -------------------------------

@pytest.fixture
def numeric_task(tmp_path, monkeypatch):
    """A fake task + EP tree laid out like the HIP EP build: the EP library and
    the kernel library live in different build dirs. The 'EP build' copies the
    overlaid kernel source into the kernel library, so the snapshot's bytes
    prove which source was compiled."""
    task = tmp_path / "task"
    (task / "kernel_src").mkdir(parents=True)
    (task / "kernel_src" / "k.hip").write_text("CANDIDATE\n")
    ep = tmp_path / "ep"
    kdst = ep / "lib" / "Runtime" / "Kernels" / "hip" / "k.hip"
    kdst.parent.mkdir(parents=True)
    kdst.write_text("ORIGINAL\n")
    dll = ep / "build" / "dll" / "ep.so"
    klib = ep / "build" / "bin" / "libkernels.so"
    for lib in (dll, klib):
        lib.parent.mkdir(parents=True)
        lib.write_text("STALE\n")
        os.utime(lib, (0, 0))  # the rebuild check compares mtimes; avoid same-tick writes
    # Like an install lib/: a sibling the EP loads, an old kernel lib, non-libs.
    (dll.parent / "libdep.so").write_text("DEP\n")
    (dll.parent / "libkernels.so").write_text("OLD INSTALLED\n")
    (dll.parent / "notes.txt").write_text("not a lib\n")
    monkeypatch.setattr(mod, "TASK_DIR", str(task))
    monkeypatch.setattr(mod, "BUILD_DIR", str(task / "build"))
    meta = {"kernel_file": "k.hip", "numeric_test": "test_k.py",
            "numeric": {"ep_repo": str(ep), "kernel_dst": str(kdst), "ep_dll": str(dll),
                        "kernel_lib": str(klib),
                        "build_cmd": f"cp '{kdst}' '{klib}'"}}
    return meta, task, kdst, dll


def test_numeric_compile_requires_build_cmd(numeric_task):
    meta, *_ = numeric_task
    meta["numeric"]["build_cmd"] = ""
    ok, err = mod.numeric_compile(meta)
    assert not ok and "build_cmd" in err


def test_numeric_compile_snapshots_ep_dir_with_rebuilt_kernel_lib(numeric_task):
    meta, task, kdst, _ = numeric_task
    ok, err = mod.numeric_compile(meta)
    assert ok, err
    snap = task / "build" / "ep"
    assert sorted(os.listdir(snap)) == ["ep.so", "libdep.so", "libkernels.so"]
    assert (snap / "libdep.so").read_text() == "DEP\n"
    # the rebuilt kernel lib wins over the stale one next to the EP
    assert (snap / "libkernels.so").read_text() == "CANDIDATE\n"
    assert kdst.read_text() == "ORIGINAL\n"
    ep_repo = meta["numeric"]["ep_repo"]
    assert sorted(os.listdir(ep_repo)) == ["build", "lib"]  # no lock left in the tree


def test_numeric_compile_kernel_lib_defaults_to_ep_dll(numeric_task):
    meta, task, kdst, dll = numeric_task
    del meta["numeric"]["kernel_lib"]
    meta["numeric"]["build_cmd"] = f"cp '{kdst}' '{dll}'"
    ok, err = mod.numeric_compile(meta)
    assert ok, err
    assert (task / "build" / "ep" / "ep.so").read_text() == "CANDIDATE\n"


def test_numeric_compile_fails_if_only_ep_dll_is_rebuilt(numeric_task):
    meta, _, kdst, dll = numeric_task
    meta["numeric"]["build_cmd"] = f"cp '{kdst}' '{dll}'"
    ok, err = mod.numeric_compile(meta)
    assert not ok and "libkernels.so" in err


def test_numeric_compile_fails_on_candidate_source_probe(numeric_task):
    # benchmark_engineer.md's relocation probe: an #error in the candidate
    # must make compile fail.
    meta, task, kdst, _ = numeric_task
    (task / "kernel_src" / "k.hip").write_text("#error GEAK_CANDIDATE_SOURCE_PROBE\n")
    meta["numeric"]["build_cmd"] = f"! grep -q GEAK_CANDIDATE_SOURCE_PROBE '{kdst}'"
    ok, err = mod.numeric_compile(meta)
    assert not ok and "EP build failed" in err
    assert kdst.read_text() == "ORIGINAL\n"


def test_numeric_compile_fails_if_build_does_not_touch_dll(numeric_task):
    meta, *_ = numeric_task
    meta["numeric"]["build_cmd"] = "true"
    ok, err = mod.numeric_compile(meta)
    assert not ok and "did not rebuild" in err


def test_numeric_pytest_requires_compiled_snapshot(numeric_task):
    meta, *_ = numeric_task
    rc, out = mod._numeric_pytest(meta, [])
    assert rc != 0 and "run `compile` first" in out


def test_numeric_pytest_loads_the_snapshot_not_ep_dll(numeric_task, monkeypatch):
    meta, task, _, dll = numeric_task
    assert mod.numeric_compile(meta)[0]
    seen = []
    monkeypatch.setattr(mod, "_run", lambda cmd, **k: seen.append((cmd, k)) or _Proc(0))
    meta["numeric"]["ep_options"] = ["config_file=/x/cfg.json"]
    mod._numeric_pytest(meta, [])
    cmd, kw = seen[0]
    assert kw["cwd"] == os.path.join(meta["numeric"]["ep_repo"], "test", "numeric")
    assert f"--ep-dll={task / 'build' / 'ep' / 'ep.so'}" in cmd
    assert f"--ep-dll={dll}" not in cmd
    # every suite option is one token, so pytest's pre-parse can't mistake a
    # path value for a test path and skip the suite's conftest.py
    assert "--ep-option=config_file=/x/cfg.json" in cmd
    assert not any(a in ("--backend", "--ep-name", "--ep-dll", "--ep-option") for a in cmd)


@pytest.mark.parametrize("numeric_test,selector", [
    ("test_k.py", "test_k"),
    ("test_k and not GroupNorm", "test_k and not GroupNorm"),
])
def test_numeric_pytest_passes_numeric_test_as_one_k_expression(
        numeric_task, monkeypatch, numeric_test, selector):
    meta, *_ = numeric_task
    assert mod.numeric_compile(meta)[0]
    seen = []
    monkeypatch.setattr(mod, "_run", lambda cmd, **k: seen.append(cmd) or _Proc(0))
    meta["numeric_test"] = numeric_test
    mod._numeric_pytest(meta, [])
    cmd = seen[0]
    assert cmd[cmd.index("-k") + 1] == selector


_DURATIONS = "0.42s call     tests/test_k.py::test_k[case0]\n"


@pytest.mark.parametrize("rc,out,ok", [
    (0, _DURATIONS, True),
    (1, _DURATIONS, False),   # failing tests: timings are invalid
    (0, "1 passed\n", False),  # nothing parsed
])
def test_numeric_performance_gating(monkeypatch, rc, out, ok):
    monkeypatch.setattr(mod, "_numeric_pytest", lambda meta, extra: (rc, out))
    got_ok, cases, err = mod.numeric_performance({})
    assert got_ok is ok
    if ok:
        assert cases[0]["test_case_id"] == "test_k[case0]"
        assert cases[0]["execution_time_ms"] == pytest.approx(420.0)
    else:
        assert cases == [] and err


# -- main(): performance exit code + report -------------------------------------

def _run_main(monkeypatch, tmp_path, perf_result, mode="performance"):
    (tmp_path / "task_meta.json").write_text('{"oracle": "selfcheck"}')
    monkeypatch.setattr(mod, "TASK_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setitem(mod.DISPATCH, "selfcheck",
                        (None, None, lambda meta: perf_result))
    monkeypatch.setattr(sys, "argv", ["task_runner.py", mode])
    with pytest.raises(SystemExit) as ei:
        mod.main()
    with open(tmp_path / "build" / "performance_report.json") as f:
        return ei.value.code, json.load(f)


def test_main_performance_failure_exits_nonzero_with_fail_report(monkeypatch, tmp_path, capsys):
    code, report = _run_main(monkeypatch, tmp_path,
                             (False, [{"test_case_id": "x", "execution_time_ms": 1.0}], "boom"))
    assert code == 1
    assert report == {"status": "fail", "error": "boom", "test_cases": []}
    assert "Perf:" not in capsys.readouterr().out


def test_main_performance_success(monkeypatch, tmp_path, capsys):
    cases = [{"test_case_id": "x", "execution_time_ms": 1.5, "params": {}}]
    code, report = _run_main(monkeypatch, tmp_path, (True, cases, None))
    assert code == 0 and report["status"] == "ok" and report["test_cases"] == cases
    assert "Perf: 1.5000 ms (x)" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
