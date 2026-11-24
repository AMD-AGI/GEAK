#!/usr/bin/env python3
"""Pure-logic coverage for tools/ep_task_runner.py — the parts that run without a
GPU: oracle dispatch, the perf-parse regexes, pass/fail gating, and the
missing-tool guard. The hipcc build + on-GPU run are verified end-to-end on the
gfx1151 box, not here.

    python3 -m pytest tools/tests/test_ep_task_runner.py
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ep_task_runner as mod  # noqa: E402


@pytest.fixture(autouse=True)
def private_tempdir(tmp_path, monkeypatch):
    # The EP build lock and kernel backup live in the temp dir.
    d = tmp_path / "tmp"
    d.mkdir()
    monkeypatch.setattr(mod.tempfile, "tempdir", str(d))
    return d


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
    for var in mod._ARCH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    meta = {"op": "k", "arch": "gfx1151", "kernel_file": "k.hip", "numeric_test": "test_k.py",
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
    assert sorted(os.listdir(snap)) == ["ep.so", "libdep.so", "libkernels.so", "manifest.json"]
    assert (snap / "libdep.so").read_text() == "DEP\n"
    # the rebuilt kernel lib wins over the stale one next to the EP
    assert (snap / "libkernels.so").read_text() == "CANDIDATE\n"
    assert kdst.read_text() == "ORIGINAL\n"
    ep_repo = meta["numeric"]["ep_repo"]
    assert sorted(os.listdir(ep_repo)) == ["build", "lib"]  # no lock left in the tree
    assert not os.path.exists(mod._kernel_backup_path(str(kdst)))


def test_numeric_compile_undoes_an_overlay_a_killed_compile_left(numeric_task):
    # A compile killed mid-build (SIGKILL) leaves its candidate in the EP tree
    # and the tree's own kernel in the backup.
    meta, task, kdst, _ = numeric_task
    with open(mod._kernel_backup_path(str(kdst)), "w") as f:
        f.write("ORIGINAL\n")
    kdst.write_text("LEFT BY A KILLED COMPILE\n")
    ok, err = mod.numeric_compile(meta)
    assert ok, err
    assert (task / "build" / "ep" / "libkernels.so").read_text() == "CANDIDATE\n"
    assert kdst.read_text() == "ORIGINAL\n"
    assert not os.path.exists(mod._kernel_backup_path(str(kdst)))


def test_sigterm_during_the_build_restores_the_ep_kernel(numeric_task):
    meta, _, kdst, _ = numeric_task
    meta["numeric"]["build_cmd"] = "kill -TERM $PPID; sleep 1"
    previous = signal.getsignal(signal.SIGTERM)
    mod._exit_on_sigterm()
    try:
        with pytest.raises(SystemExit):
            mod.numeric_compile(meta)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert kdst.read_text() == "ORIGINAL\n"
    assert not os.path.exists(mod._kernel_backup_path(str(kdst)))


@pytest.mark.parametrize("ignores_sigterm", [False, True])
def test_sigterm_during_the_build_stops_the_whole_build(numeric_task, tmp_path, monkeypatch,
                                                         ignores_sigterm):
    # The background job stands in for a `docker run` client that outlives the
    # shell; a build that ignores SIGTERM is SIGKILLed after the grace period.
    meta, _, kdst, _ = numeric_task
    late = tmp_path / "late"
    trap = "trap '' TERM; " if ignores_sigterm else ""
    meta["numeric"]["build_cmd"] = f"{trap}(sleep 1; echo LATE > '{late}') & kill -TERM $PPID; wait"
    monkeypatch.setattr(mod, "_BUILD_STOP_GRACE_S", 0.3)
    previous = signal.getsignal(signal.SIGTERM)
    mod._exit_on_sigterm()
    try:
        with pytest.raises(SystemExit):
            mod.numeric_compile(meta)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert kdst.read_text() == "ORIGINAL\n"
    assert not os.path.exists(mod._kernel_backup_path(str(kdst)))
    time.sleep(1.5)
    assert not late.exists()


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


@pytest.mark.parametrize("mode", ["correctness", "performance"])
def test_numeric_modes_require_compiled_snapshot(numeric_task, monkeypatch, mode):
    meta, *_ = numeric_task
    monkeypatch.setattr(mod, "_run", lambda *a, **k: pytest.fail("must not run the EP"))
    fn = mod.numeric_correctness if mode == "correctness" else mod.numeric_performance
    err = fn(meta)[-1]
    assert "run `compile` first" in err


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
    monkeypatch.setattr(mod, "_check_snapshot", lambda meta: None)
    monkeypatch.setattr(mod, "_numeric_pytest", lambda meta, extra: (rc, out))
    got_ok, cases, err = mod.numeric_performance({"numeric": {}})
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


# -- numeric oracle: snapshot identity ------------------------------------------

def test_numeric_compile_writes_a_manifest_of_what_it_built(numeric_task):
    meta, task, *_ = numeric_task
    meta["ep_base_sha"] = "abc123"
    assert mod.numeric_compile(meta)[0]
    snap = task / "build" / "ep"
    man = json.loads((snap / "manifest.json").read_text())
    assert man["kernel_sha256"] == mod._sha256(str(task / "kernel_src" / "k.hip"))
    assert man["ep_base_sha"] == "abc123"
    assert man["libs"] == {n: mod._sha256(str(snap / n))
                           for n in ("ep.so", "libdep.so", "libkernels.so")}
    assert mod._check_snapshot(meta) is None


def test_snapshot_is_refused_after_kernel_src_changes(numeric_task, monkeypatch):
    meta, task, *_ = numeric_task
    assert mod.numeric_compile(meta)[0]
    (task / "kernel_src" / "k.hip").write_text("NEWER CANDIDATE\n")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: pytest.fail("must not run the EP"))
    ok, err = mod.numeric_correctness(meta)
    assert not ok and "changed since the last `compile`" in err
    ok, _, err = mod.numeric_performance(meta)
    assert not ok and "changed since the last `compile`" in err


def test_snapshot_is_refused_after_a_library_is_replaced(numeric_task):
    meta, task, *_ = numeric_task
    assert mod.numeric_compile(meta)[0]
    (task / "build" / "ep" / "libkernels.so").write_text("SOMETHING ELSE\n")
    assert "libkernels.so changed after `compile`" in mod._check_snapshot(meta)


def test_failed_compile_leaves_previous_snapshot_unusable(numeric_task):
    meta, task, *_ = numeric_task
    assert mod.numeric_compile(meta)[0]
    (task / "kernel_src" / "k.hip").write_text("BROKEN\n")
    meta["numeric"]["build_cmd"] = "false"
    assert not mod.numeric_compile(meta)[0]
    assert "changed since the last `compile`" in mod._check_snapshot(meta)


# -- numeric oracle: pinned runtime env -----------------------------------------

def test_ep_env_drops_inherited_hipdnn_vars_and_pins_ep_env(monkeypatch):
    monkeypatch.setenv("HIPDNN_EP_MATMUL_DP4A", "0")
    monkeypatch.setenv("HIPDNN_MATMUL_AUTOTUNE_MODE", "online")
    monkeypatch.setenv("HIPDNN_EP_TRACE_FILE", "/tmp/t.json")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/rocm/lib")
    env = mod._ep_env({"numeric": {"ep_env": {"HIPDNN_MATMUL_AUTOTUNE_MODE": "lookup"}}})
    assert env["HIPDNN_MATMUL_AUTOTUNE_MODE"] == "lookup"
    assert "HIPDNN_EP_MATMUL_DP4A" not in env and "HIPDNN_EP_TRACE_FILE" not in env
    assert env["LD_LIBRARY_PATH"] == "/opt/rocm/lib"


def test_numeric_pytest_runs_with_the_pinned_env(numeric_task, monkeypatch):
    meta, *_ = numeric_task
    meta["numeric"]["ep_env"] = {"HIPDNN_MATMUL_AUTOTUNE_MODE": "lookup"}
    monkeypatch.setenv("HIPDNN_EP_MATMUL_DP4A", "0")
    seen = []
    monkeypatch.setattr(mod, "_run", lambda cmd, **k: seen.append(k) or _Proc(0))
    mod._numeric_pytest(meta, [])
    env = seen[0]["env"]
    assert env["HIPDNN_MATMUL_AUTOTUNE_MODE"] == "lookup" and "HIPDNN_EP_MATMUL_DP4A" not in env


def test_numeric_runs_use_the_pinned_interpreter(numeric_task, monkeypatch):
    meta, *_ = numeric_task
    meta["numeric"]["perf"] = {"cases": [{"id": "a", "M": 1, "N": 32, "K": 64}]}
    seen = []
    monkeypatch.setattr(mod, "_run", lambda cmd, **k: seen.append(cmd) or _Proc(0))
    mod._numeric_pytest(meta, [])
    mod._perf_driver(meta, check_only=True)
    assert [c[0] for c in seen] == [sys.executable] * 2
    meta["numeric"]["python"] = "/venv/bin/python"
    mod._numeric_pytest(meta, [])
    mod._perf_driver(meta, check_only=True)
    assert [c[0] for c in seen[2:]] == ["/venv/bin/python"] * 2


# -- numeric oracle: frozen ABI / dispatch gate ---------------------------------

_ABI_KERNEL = """#include "hip_custom_kernels.h"
static constexpr int kTile = 16;
__global__ void k_kernel(float* y) { y[0] = 1.0f; }
static bool fastPath(int m) { return m <= kTile; }
extern "C" int hip_k(void* stream, float* y, int m) {
    if (fastPath(m)) { hipLaunchKernelGGL(k_kernel, dim3(1), dim3(64), 0, 0, y); }
    return 0;
}
"""


@pytest.fixture
def abi_task(numeric_task):
    meta, task, kdst, _ = numeric_task
    import ep_abi
    (task / "kernel_src" / "k.hip").write_text(_ABI_KERNEL)
    kdst.write_text(_ABI_KERNEL)
    meta["abi"] = ep_abi.manifest(_ABI_KERNEL, ["hip_k", "fastPath"], ["kTile"])
    return meta, task, kdst


def test_compile_accepts_a_kernel_body_edit(abi_task):
    meta, task, _ = abi_task
    (task / "kernel_src" / "k.hip").write_text(
        _ABI_KERNEL.replace("y[0] = 1.0f;", "y[threadIdx.x] = 1.0f; // tuned"))
    ok, err = mod.numeric_compile(meta)
    assert ok, err


@pytest.mark.parametrize("old,new,match", [
    ("return m <= kTile;", "return m <= 2 * kTile;", "dispatch gate fastPath()"),
    ("int m) {", "long m) {", "hip_k changed its signature"),
    ("kTile = 16;", "kTile = 8;", "frozen constant kTile"),
])
def test_compile_rejects_frozen_edits_before_building(abi_task, old, new, match):
    meta, task, kdst = abi_task
    (task / "kernel_src" / "k.hip").write_text(_ABI_KERNEL.replace(old, new))
    meta["numeric"]["build_cmd"] = "echo must-not-build; false"
    ok, err = mod.numeric_compile(meta)
    assert not ok and match in err and "frozen contract" in err
    assert kdst.read_text() == _ABI_KERNEL
    assert not (task / "build" / "ep").exists()


def test_compile_rejects_changed_exports_after_building(abi_task, monkeypatch):
    meta, task, kdst = abi_task
    import ep_abi
    meta["abi"]["exports"] = ["hip_k"]
    monkeypatch.setattr(ep_abi, "exported_c_symbols", lambda lib, nm="nm": ["hip_k", "hip_k2"])
    ok, err = mod.numeric_compile(meta)
    assert not ok and "exports new symbol hip_k2" in err
    assert kdst.read_text() == _ABI_KERNEL
    assert not (task / "build" / "ep").exists()


def test_compile_reports_missing_nm(abi_task, monkeypatch):
    meta, *_ = abi_task
    meta["abi"]["exports"] = ["hip_k"]
    monkeypatch.setenv("NM", "/no/such/nm")
    ok, err = mod.numeric_compile(meta)
    assert not ok and "cannot list the exports" in err


# -- numeric oracle: device-time perf driver ------------------------------------

_PERF = {"op": "matmul_nbits", "warmup": 2, "iters": 3, "rotate_mib": 64,
         "cases": [{"id": "a", "M": 1, "K": 64, "N": 32}, {"id": "b", "M": 4, "K": 64, "N": 32}]}


def _driver_line(case_id, status="ok", median=12.5, prefix="GEAK_EP_PERF", **extra):
    res = {"id": case_id, "status": status, "params": {"M": 1}, "shape": "m=1,n=32,k=64",
           "nodes": 4, "median_us": median, "min_us": 10.0, "p10_us": 11.0, "p90_us": 14.0,
           "mean_us": 12.6, "cv": 0.05, "samples": 9, **extra}
    return f"{prefix} {json.dumps(res)}\n"


@pytest.fixture
def perf_task(numeric_task, monkeypatch):
    meta, task, *_ = numeric_task
    meta["numeric"]["perf"] = json.loads(json.dumps(_PERF))
    meta["numeric"]["ep_env"] = {"HIPDNN_MATMUL_AUTOTUNE_MODE": "lookup"}
    assert mod.numeric_compile(meta)[0]
    calls = []

    def fake_run(stdout_for):
        def run(cmd, **kw):
            calls.append((cmd, kw))
            return _Proc(*stdout_for(cmd))
        monkeypatch.setattr(mod, "_run", run)
    return meta, task, calls, fake_run


def test_performance_reports_device_time_per_case(perf_task):
    meta, task, calls, fake_run = perf_task
    fake_run(lambda cmd: (0, "noise\n" + _driver_line("b", median=40.0) + _driver_line("a")))
    ok, cases, err = mod.numeric_performance(meta)
    assert ok, err
    assert [c["test_case_id"] for c in cases] == ["a", "b"]  # registry order
    assert cases[0]["execution_time_ms"] == pytest.approx(0.0125)
    assert cases[1]["execution_time_ms"] == pytest.approx(0.040)
    assert cases[0]["params"]["samples"] == 9 and cases[0]["params"]["shape"] == "m=1,n=32,k=64"
    cmd, kw = calls[0]
    assert cmd[1] == str(task / "harness" / "ep_perf_driver.py")
    assert f"--ep-dll={task / 'build' / 'ep' / 'ep.so'}" in cmd
    assert "--warmup=2" in cmd and "--iters=3" in cmd and "--rotate-mib=64" in cmd
    assert "--check-only" not in cmd
    cases_arg = next(a for a in cmd if a.startswith("--cases="))
    assert json.loads(cases_arg[len("--cases="):]) == _PERF["cases"]
    assert kw["env"]["HIPDNN_MATMUL_AUTOTUNE_MODE"] == "lookup"


def test_perf_output_drops_only_the_ep_teardown_summary(perf_task, monkeypatch, capsys):
    meta, *_ = perf_task
    proc = _Proc(0, _driver_line("a") + _driver_line("b"))
    proc.stderr = "[PERF SUMMARY] total_inferences=9\nhipErrorOutOfMemory in a\n"
    monkeypatch.setattr(mod, "_run", lambda cmd, **kw: proc)
    assert mod.numeric_performance(meta)[0]
    out = capsys.readouterr().out
    assert "hipErrorOutOfMemory in a" in out and "[PERF SUMMARY]" not in out


@pytest.mark.parametrize("rc,stdout,match", [
    (1, _driver_line("a") + _driver_line("b", status="fail", error="mismatch vs CPU EP"),
     "b: mismatch vs CPU EP"),
    (0, _driver_line("a"), "no result for ['b']"),
    (139, "", "exited 139"),
    (1, _driver_line("a") + _driver_line("b"), "exited 1"),
])
def test_performance_fails_on_any_bad_case(perf_task, rc, stdout, match):
    meta, _, _, fake_run = perf_task
    fake_run(lambda cmd: (rc, stdout))
    ok, cases, err = mod.numeric_performance(meta)
    assert not ok and cases == [] and match in err


def test_correctness_runs_pytest_then_the_driver_check(perf_task):
    meta, _, calls, fake_run = perf_task

    def out(cmd):
        if "-m" in cmd:
            return 0, "5 passed\n"
        return 0, _driver_line("a", prefix="GEAK_EP_CHECK") + _driver_line("b", prefix="GEAK_EP_CHECK")
    fake_run(out)
    assert mod.numeric_correctness(meta) == (True, None)
    assert "-m" in calls[0][0] and "--check-only" in calls[1][0]


def test_correctness_fails_when_the_driver_check_fails(perf_task):
    meta, _, _, fake_run = perf_task
    fake_run(lambda cmd: (0, "5 passed\n") if "-m" in cmd else
             (1, _driver_line("a", prefix="GEAK_EP_CHECK", status="fail", error="cosine 0.2")))
    ok, err = mod.numeric_correctness(meta)
    assert not ok and "cosine 0.2" in err


def test_correctness_skips_driver_when_pytest_fails(perf_task):
    meta, _, calls, fake_run = perf_task
    fake_run(lambda cmd: (1, "1 failed\n"))
    ok, err = mod.numeric_correctness(meta)
    assert not ok and "pytest" in err and len(calls) == 1


# -- export-patch -----------------------------------------------------------------

def _git(cwd, *args):
    import subprocess
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def git_task(numeric_task):
    meta, task, kdst, _ = numeric_task
    ep = meta["numeric"]["ep_repo"]
    kdst.write_text("line1\nline2\nline3\n")
    _git(ep, "init", "-q", "-b", "main")
    _git(ep, "add", "lib")
    _git(ep, "commit", "-q", "-m", "base")
    meta["ep_base_sha"] = _git(ep, "rev-parse", "HEAD")
    (task / "kernel_src" / "k.hip").write_text("line1\nline2 optimized\nline3\n")
    os.makedirs(task / "build", exist_ok=True)
    return meta, task, ep


def test_export_patch_diffs_against_the_base_without_touching_the_ep_repo(git_task):
    meta, task, ep = git_task
    meta["numeric"]["upstream_ref"] = "origin/main"
    objects_before = _git(ep, "count-objects", "-v")
    ok, report, err = mod.numeric_export_patch(meta)
    assert ok, err
    patch = (task / "build" / "hip_ep.patch").read_text()
    assert "--- a/lib/Runtime/Kernels/hip/k.hip" in patch and "+line2 optimized" in patch
    assert report["applies"] == {meta["ep_base_sha"]: "ok", "origin/main": "unresolved"}
    assert report["path"] == "lib/Runtime/Kernels/hip/k.hip" and not report["empty"]
    assert _git(ep, "status", "--porcelain", "--untracked-files=no") == ""
    assert _git(ep, "count-objects", "-v") == objects_before


def test_export_patch_round_trips_onto_a_clean_checkout(git_task, tmp_path):
    meta, task, ep = git_task
    assert mod.numeric_export_patch(meta)[0]
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", ep, str(clone))
    _git(clone, "apply", str(task / "build" / "hip_ep.patch"))
    assert (clone / "lib" / "Runtime" / "Kernels" / "hip" / "k.hip").read_text() == \
        (task / "kernel_src" / "k.hip").read_text()


def test_export_patch_reports_upstream_conflicts_without_failing(git_task):
    meta, _, ep = git_task
    _git(ep, "checkout", "-q", "-b", "upstream")
    Path(meta["numeric"]["kernel_dst"]).write_text("line1\nline2 upstream\nline3\n")
    _git(ep, "commit", "-q", "-am", "upstream change")
    _git(ep, "checkout", "-q", "main")
    meta["numeric"]["upstream_ref"] = "upstream"
    ok, report, err = mod.numeric_export_patch(meta)
    assert ok, err
    assert report["applies"][meta["ep_base_sha"]] == "ok"
    assert report["applies"]["upstream"] != "ok"


def test_export_patch_of_an_unchanged_kernel_is_empty(git_task):
    meta, task, _ = git_task
    (task / "kernel_src" / "k.hip").write_text("line1\nline2\nline3\n")
    ok, report, _ = mod.numeric_export_patch(meta)
    assert ok and report["empty"] and (task / "build" / "hip_ep.patch").read_text() == ""


def test_export_patch_needs_a_base_sha(numeric_task):
    meta, *_ = numeric_task
    ok, _, err = mod.numeric_export_patch(meta)
    assert not ok and "ep_base_sha" in err


def test_main_export_patch_refuses_non_numeric_tasks(monkeypatch, tmp_path, capsys):
    (tmp_path / "task_meta.json").write_text('{"oracle": "selfcheck"}')
    monkeypatch.setattr(mod, "TASK_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "BUILD_DIR", str(tmp_path / "build"))
    monkeypatch.setattr(sys, "argv", ["task_runner.py", "export-patch"])
    with pytest.raises(SystemExit) as ei:
        mod.main()
    assert ei.value.code == 1
    report = json.loads((tmp_path / "build" / "export_patch_report.json").read_text())
    assert report["status"] == "fail" and "numeric" in report["error"]
    assert "Export: FAIL" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
