#!/usr/bin/env python3
"""tools/hip_ep_e2e/make_task.py and leaf_runner.py with a fake compiler and fake leaves:
the generated task, the workload file, what the runner rebuilds, and the verdicts it
reads from the leaves. Building and running the real leaves needs a GPU.

    python3 -m pytest tools/hip_ep_e2e/tests/test_make_task.py
"""
import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hipep_env  # noqa: E402
import leaf_runner  # noqa: E402
import make_task  # noqa: E402

OP = "matmul_nbits_u4_prefill"

CMD = ("/opt/tr/bin/amdclang++  -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -I/s/lib/Runtime/Kernels/include "
       "-I/b/schemas -isystem /b/_deps/llvm/include -isystem /opt/tr/include -O3 -DNDEBUG -std=c++17 "
       "--offload-arch=gfx11-generic -fPIC -Wno-ignored-attributes --offload-compress -MD -MT x.o "
       "-MF x.o.d -o x.o -x hip -c /s/lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip")

SHAPES = """model,K,N,bits,block_size,has_zp,node_count
q27b,5120,34816,4,32,0,32
q27b,17408,5120,4,32,0,64
q27b-mtp,5120,34816,4,32,0,32
"""



def test_read_shapes_sums_calls_per_shape(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(SHAPES)
    s = make_task.read_shapes(str(p), [1024, 2048])
    assert [(x["M"], x["K"], x["N"], x["calls"]) for x in s] == [
        (1024, 5120, 34816, 64), (1024, 17408, 5120, 64), (2048, 5120, 34816, 64), (2048, 17408, 5120, 64)]


def _fake_tree(tmp_path):
    src, build = tmp_path / "src", tmp_path / "build"
    k = src / "lib" / "Runtime" / "Kernels"
    (k / "hip" / "autotune" / "matmul_nbits" / "lut").mkdir(parents=True)
    (k / "include").mkdir()
    (k / "unit_test" / "matmul_nbits" / "Xfp16_Wu4_Yfp16").mkdir(parents=True)
    (k / "hip" / "matmul_nbits_kernel.hip").write_text(
        " ".join(make_task.OPS[OP]["targets"][:4]) + "\n#include \"matmul_nbits_wmma_core.h\"\n")
    (k / "include" / "matmul_nbits_wmma_core.h").write_text("void GemmFp16U4Impl();\n")
    (k / "include" / "matmul_nbits_wmma_core_x.h").write_text("void GemmFp16U4ImplX();\n")
    (k / "include" / "hip_custom_kernels.h").write_text("// abi\n")
    (k / "hip" / "autotune" / "matmul_nbits" / "matmul_nbits_autotune.cpp").write_text("//\n")
    (k / "hip" / "autotune" / "matmul_nbits" / "tools").mkdir()
    (k / "hip" / "autotune" / "matmul_nbits" / "tools" / "empty_lut_data.cpp").write_text("// no table\n")
    (k / "hip" / "autotune" / "matmul_nbits" / "lut" / "gfx1151.fb").write_bytes(b"fb")
    (k / "unit_test" / "matmul_nbits" / "Xfp16_Wu4_Yfp16" / "test_matmul_nbits.cpp").write_text("//\n")
    (k / "unit_test" / "qmoe_ragged_mm" / "Xfp16_Wu4_Yfp16").mkdir(parents=True)
    (k / "unit_test" / "qmoe_ragged_mm" / "Xfp16_Wu4_Yfp16" / "test_qmoe_ragged_mm.cpp").write_text("//\n")
    (k / "hip" / "qmoe_ragged_mm_kernel.hip").write_text("#include \"matmul_nbits_wmma_core.h\"\n")
    (k / "hip" / "qmoe_kernel.hip").write_text("// routing\n")
    (build / "_deps" / "flatbuffers-src" / "include" / "flatbuffers").mkdir(parents=True)
    (build / "_deps" / "flatbuffers-src" / "include" / "flatbuffers" / "base.h").write_text("//\n")
    (build / "schemas").mkdir()
    (build / "schemas" / "matmul_nbits_autotune_generated.h").write_text("//\n")
    _commit(src)
    shapes = tmp_path / "shapes.csv"
    shapes.write_text(SHAPES)
    return src, build, shapes


def _commit(repo):
    """The hip-ep checkout is a git tree at the release tag; the tools read HEAD, not files."""
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "tag"]):
        subprocess.run(cmd, cwd=repo, check=True)


FAKE_HIPCC = """#!/bin/bash
echo "$*" >> "$(dirname "$0")/calls.log"
while [ $# -gt 0 ]; do [ "$1" = -o ] && touch "$2"; shift; done
"""


@pytest.fixture
def task(tmp_path, monkeypatch):
    """A generated task whose runner uses a fake compiler that logs its calls."""
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    monkeypatch.setenv("SERVING_GPU_LOCK", str(tmp_path / "gpu.lock"))
    out = tmp_path / "task"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr")
    cc = tmp_path / "hipcc"
    cc.write_text(FAKE_HIPCC)
    cc.chmod(0o755)
    meta["hipcc"] = str(cc)
    for name, value in (("TASK_DIR", out), ("BUILD_DIR", out / "build"), ("EXE", out / "build" / "test_direct"),
                        ("QMOE_EXE", out / "build" / "test_qmoe")):
        monkeypatch.setattr(leaf_runner, name, str(value))
    (out / "build").mkdir(exist_ok=True)
    return out, meta, tmp_path / "calls.log", src, build, shapes


def test_generate_layout(task, tmp_path):
    out, meta, _, src, build, shapes = task
    for rel in meta["editable"] + [meta["test_cpp"], meta["autotune_cpp"], meta["lut_fb"], meta["qmoe"]["test_cpp"],
                                   meta["qmoe"]["lut_stub"], *(x["file"] for x in meta["qmoe"]["sources"]),
                                   "harness/linux_compat.h", "scripts/task_runner.py", "scripts/hipep_env.py",
                                   "scripts/ep_task_runner.py", "deps/gen/matmul_nbits_autotune_generated.h",
                                   "deps/flatbuffers/include/flatbuffers/base.h"]:
        assert (out / rel).is_file(), rel
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["task_type"] == "hip_ep2hip_ep" and cfg["source_file_path"] == meta["editable"]
    assert meta["hipcc"] != "/opt/tr/bin/amdclang++" and json.loads((out / "task_meta.json").read_text())["hipcc"] == \
        "/opt/tr/bin/amdclang++"
    assert [g["M"] for g in meta["guard_shapes"]] == [1, 1]
    assert meta["lock_path"] == str(tmp_path / "gpu.lock"), "the leaves run under the serving-GPU lock"
    q = meta["qmoe"]
    assert all("--offload-compress" in x["flags"] for x in q["sources"]), "release flags for the QMoE kernels too"
    assert not {q["test_cpp"], q["lut_stub"], *(x["file"] for x in q["sources"])} & set(meta["editable"])
    assert meta["lut_fb"].endswith("/lut/gfx1151.fb")


def test_generate_takes_the_tag_and_the_overlay_not_the_working_tree(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    lut = src / "lib" / "Runtime" / "Kernels" / "hip" / "autotune" / "matmul_nbits" / "lut"
    (lut / "gfx1151.fb").write_bytes(b"left behind by the last build")
    meta = make_task.generate(OP, "gfx1201", str(src), str(build), str(shapes), str(tmp_path / "t1"), [1024], "/tr")
    assert meta["lut_fb"].endswith("/lut/gfx1201.fb"), "the task reads the device's table"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "t1"), [1024], "/tr")
    assert (tmp_path / "t1" / meta["lut_fb"]).read_bytes() == b"fb"
    ov = tmp_path / "overlay" / lut.relative_to(src)
    ov.mkdir(parents=True)
    (ov / "gfx1151.fb").write_bytes(b"refreshed")
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "t2"), [1024], "/tr",
                              overlay=str(tmp_path / "overlay"))
    assert (tmp_path / "t2" / meta["lut_fb"]).read_bytes() == b"refreshed"


def test_generate_refuses_missing_targets_and_unguarded_includers(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    k = src / "lib" / "Runtime" / "Kernels"
    (k / "README.md").write_text("GemmFp16U4Impl is the WMMA core.\n")  # not a source: ignored
    (k / "hip" / "new_moe_kernel.hip").write_text("#include \"matmul_nbits_wmma_core.h\"\n")
    _commit(src)
    with pytest.raises(SystemExit, match=r"\['lib/Runtime/Kernels/hip/new_moe_kernel.hip'\]"):
        make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "t"), [1024], "/tr")
    (k / "hip" / "new_moe_kernel.hip").unlink()
    (k / "hip" / "matmul_nbits_kernel.hip").write_text("// none\n")
    _commit(src)
    with pytest.raises(SystemExit, match="target functions"):
        make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "t"), [1024], "/tr")


def test_workload_weights(task, tmp_path):
    out = task[0]
    rep = tmp_path / "perf.json"
    rep.write_text(json.dumps({"test_cases": [
        {"test_case_id": "M1024_K5120_N34816_gs32_noz", "execution_time_ms": 9.7},
        {"test_case_id": "M1024_K17408_N5120_gs32_noz", "execution_time_ms": 4.8}]}))
    w = make_task.workload(str(out), str(rep))
    cases = w["kernels"][0]["cases"]
    assert w["schema"] == "workload-v1"
    assert cases[0]["test_case_id"] == "M1024_K5120_N34816_gs32_noz"
    assert cases[0]["weight_norm"] == pytest.approx(64 * 9.7 / (64 * 9.7 + 64 * 4.8), abs=1e-5)


# -- runner ----------------------------------------------------------------------

LINE = "  M=1024 N=34816 K=5120 gs=32 zero=0 dtype=fp16  status=0  9.7012 ms  relL2=1.234e-03  PASS"


def test_leaf_verdicts():
    assert leaf_runner.parse_leaf_line("GPU: x\n" + LINE + "\n") == {"status": 0, "ms": 9.7012, "pass": True}
    assert leaf_runner.parse_leaf_line(LINE.replace("PASS", "*** FAIL ***"))["pass"] is False
    assert leaf_runner.parse_leaf_line("HIP error at x.cpp:1 code=1") is None
    assert leaf_runner.qmoe_passed(0, "qmoe_ragged_mm coverage=3 cases=40\n  tiny PASS relL2=1e-4\nALL PASS\n")
    assert not leaf_runner.qmoe_passed(0, "  tiny PASS relL2=1e-4\n  wider *** FAIL ***\n")
    assert not leaf_runner.qmoe_passed(134, "ALL PASS\n")


def _compiled(log):
    calls = log.read_text().splitlines() if log.exists() else []
    log.write_text("")
    return sorted(os.path.basename(c.split()[c.split().index("-o") - 1]) for c in calls if " -c " in f" {c} "), \
        [c for c in calls if c.startswith("--hip-link")]


def test_the_runner_rebuilds_exactly_what_changed(task, tmp_path):
    out, meta, log, *_ = task
    assert leaf_runner.compile_task(meta) == (True, None)
    objs, links = _compiled(log)
    assert len(objs) == 7 and len(links) == 2
    qmoe_link = next(link for link in links if "test_qmoe" in link)
    assert "matmul_nbits_kernel.hip.o" in qmoe_link and "matmul_nbits_autotune.cpp.o" in qmoe_link, \
        "the QMoE test also calls the matmul_nbits kernel"
    assert leaf_runner.compile_task(meta) == (True, None) and _compiled(log) == ([], [])
    (out / meta["editable"][0]).write_text("// edited kernel\n")
    leaf_runner.compile_task(meta)
    objs, links = _compiled(log)
    assert objs == ["matmul_nbits_kernel.hip"] and len(links) == 2
    (out / meta["editable"][1]).write_text("// edited WMMA core header\n")
    leaf_runner.compile_task(meta)
    assert len(_compiled(log)[0]) == 7, "a header can reach every object"
    (out / meta["lut_fb"]).write_bytes(b"refreshed table")
    leaf_runner.compile_task(meta)
    assert len(_compiled(log)[0]) == 7
    copy = tmp_path / "workspace"
    shutil.copytree(out, copy)
    leaf_runner.TASK_DIR, leaf_runner.BUILD_DIR = str(copy), str(copy / "build")
    leaf_runner.EXE, leaf_runner.QMOE_EXE = str(copy / "build" / "test_direct"), str(copy / "build" / "test_qmoe")
    leaf_runner.compile_task(meta)
    assert _compiled(log) == ([], []), "the build does not depend on where the task sits"


def test_a_failed_link_leaves_nothing_to_run(task, tmp_path):
    out, meta, log, *_ = task
    assert leaf_runner.compile_task(meta) == (True, None)
    (out / meta["editable"][0]).write_text("// edited kernel\n")
    (tmp_path / "hipcc").write_text(FAKE_HIPCC.replace('echo "$*"', '[[ "$*" == --hip-link* ]] && exit 1; echo "$*"'))
    assert leaf_runner.compile_task(meta)[0] is False
    assert not os.path.exists(leaf_runner.EXE)


def _fake_leaf(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def test_correctness_checks_the_build_then_every_shape_the_sweep_and_the_qmoe_leaf(task, monkeypatch):
    out, meta, *_ = task
    monkeypatch.setattr(leaf_runner, "compile_task", lambda m: (False, "build step failed: kernel.o"))
    assert leaf_runner.correctness(meta) == (False, "build step failed: kernel.o")
    monkeypatch.setattr(leaf_runner, "compile_task", lambda m: (True, None))
    leaf, qmoe = out / "build" / "test_direct", out / "build" / "test_qmoe"
    _fake_leaf(qmoe, "echo '  tiny PASS relL2=1e-4'\necho 'ALL PASS'\n")
    _fake_leaf(leaf, f'echo "{LINE.replace("PASS", "*** FAIL ***")}"\n')
    ok, err = leaf_runner.correctness(meta)
    assert not ok and err.startswith("M1024_K5120_N34816_gs32_noz failed")
    _fake_leaf(leaf, f'[ "$1" = --coverage ] && {{ echo "1 failed"; echo SOME FAILED; exit 1; }}\necho "{LINE}"\n')
    ok, err = leaf_runner.correctness(meta)
    assert not ok and "full run failed" in err
    _fake_leaf(leaf, f'[ "$1" = --coverage ] && {{ echo "0 failed"; echo ALL PASSED; exit 0; }}\necho "{LINE}"\n')
    _fake_leaf(qmoe, "echo '  tiny PASS relL2=1e-4'\necho '  wider *** FAIL ***'\nexit 1\n")
    ok, err = leaf_runner.correctness(meta)
    assert not ok and "qmoe_ragged_mm" in err
    _fake_leaf(qmoe, "echo '  tiny PASS relL2=1e-4'\necho 'ALL PASS'\n")
    assert leaf_runner.correctness(meta) == (True, None)


def test_performance_builds_first_and_times_online_autotune(task, tmp_path, monkeypatch):
    out, meta, *_ = task
    checked = []
    monkeypatch.setattr(leaf_runner, "compile_task", lambda m: checked.append("built") or (True, None))
    _fake_leaf(out / "build" / "test_direct", f'echo "$HIPDNN_MATMUL_AUTOTUNE_MODE" > {tmp_path}/mode\necho "{LINE}"\n')
    ok, cases, err = leaf_runner.performance(meta)
    assert ok, err
    assert (tmp_path / "mode").read_text().strip() == "online" and checked == ["built"]
    assert [(c["test_case_id"], c["execution_time_ms"]) for c in cases][0] == ("M1024_K5120_N34816_gs32_noz", 9.7012)


def test_the_generated_runner_runs_through_ep_task_runner(task):
    out, meta, log, *_ = task
    (out / "task_meta.json").write_text(json.dumps(meta))
    p = subprocess.run([sys.executable, str(out / "scripts" / "task_runner.py"), "compile"],
                       capture_output=True, text=True, cwd=out)
    assert p.returncode == 0 and "Compilation: PASS" in p.stdout, p.stdout + p.stderr
    assert json.loads((out / "build" / "compile_report.json").read_text())["status"] == "ok"
