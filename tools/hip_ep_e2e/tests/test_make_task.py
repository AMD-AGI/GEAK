#!/usr/bin/env python3
"""Pure-logic coverage for tools/hip_ep_e2e/make_task.py and leaf_runner.py: the
shape list, the generated task layout (with the QMoE leaf as a second oracle), the
leaf output parsers, the rebuild digests, and the workload file. Building and running
the leaves needs the gfx1151 box.

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
q27b,5120,34816,4,32,0,64
q27b,17408,5120,4,32,0,64
q27b,5120,248320,4,32,0,1
q27b,5120,6144,8,32,0,4
"""


def test_read_shapes_skips_lm_head_and_other_bits(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(SHAPES)
    s = make_task.read_shapes(str(p), [1024, 2048], 4)
    assert [(x["M"], x["K"], x["N"]) for x in s] == [
        (1024, 5120, 34816), (1024, 17408, 5120), (2048, 5120, 34816), (2048, 17408, 5120)]
    assert s[0] == {"M": 1024, "K": 5120, "N": 34816, "gs": 32, "zp": False, "calls": 64}


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


def test_generate_layout(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    overlay = tmp_path / "overlay" / "lib" / "Runtime" / "Kernels" / "hip" / "autotune" / \
        "matmul_nbits" / "lut"
    overlay.mkdir(parents=True)
    (overlay / "gfx1151.fb").write_bytes(b"refreshed")
    out = tmp_path / "task"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr",
                              overlay=str(tmp_path / "overlay"))
    for rel in meta["editable"] + [meta["test_cpp"], meta["autotune_cpp"], meta["lut_fb"],
                                   "harness/linux_compat.h", "scripts/task_runner.py",
                                   "deps/gen/matmul_nbits_autotune_generated.h",
                                   "deps/flatbuffers/include/flatbuffers/base.h"]:
        assert (out / rel).is_file(), rel
    assert (out / meta["lut_fb"]).read_bytes() == b"refreshed"
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["task_type"] == "hip_ep2hip_ep"
    assert cfg["source_file_path"] == meta["editable"]
    assert meta["hipcc"] == "/opt/tr/bin/amdclang++"
    assert meta["offload_arch"] == "--offload-arch=gfx11-generic"
    assert [g["M"] for g in meta["guard_shapes"]] == [1, 1]
    assert json.loads((out / "task_meta.json").read_text())["shapes"] == meta["shapes"]


def test_generate_reads_the_tag_not_the_working_tree(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    lut = src / "lib" / "Runtime" / "Kernels" / "hip" / "autotune" / "matmul_nbits" / "lut" / "gfx1151.fb"
    lut.write_bytes(b"left behind by the last build")
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "task"), [1024], "/tr")
    assert (tmp_path / "task" / meta["lut_fb"]).read_bytes() == b"fb"


def test_the_task_reads_the_devices_table(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    meta = make_task.generate(OP, "gfx1201", str(src), str(build), str(shapes), str(tmp_path / "task"), [1024], "/tr")
    assert meta["arch"] == "gfx1201" and meta["lut_fb"].endswith("/matmul_nbits/lut/gfx1201.fb")


def test_generate_refuses_missing_targets(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    (src / "lib" / "Runtime" / "Kernels" / "hip" / "matmul_nbits_kernel.hip").write_text("// none\n")
    _commit(src)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    with pytest.raises(SystemExit, match="target functions"):
        make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "t"), [1024], "/tr")


def test_workload_weights(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    out = tmp_path / "task"
    make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr")
    rep = tmp_path / "perf.json"
    rep.write_text(json.dumps({"test_cases": [
        {"test_case_id": "M1024_K5120_N34816_gs32_noz", "execution_time_ms": 9.7},
        {"test_case_id": "M1024_K17408_N5120_gs32_noz", "execution_time_ms": 4.8}]}))
    w = make_task.workload(str(out), str(rep))
    cases = w["kernels"][0]["cases"]
    assert w["schema"] == "workload-v1"
    assert cases[0]["test_case_id"] == "M1024_K5120_N34816_gs32_noz"
    assert sum(c["weight_norm"] for c in cases) == pytest.approx(1.0)
    assert cases[0]["weight_norm"] == pytest.approx(64 * 9.7 / (64 * 9.7 + 64 * 4.8), abs=1e-5)


# -- runner ----------------------------------------------------------------------

LINE = "  M=1024 N=34816 K=5120 gs=32 zero=0 dtype=fp16  status=0  9.7012 ms  relL2=1.234e-03  PASS"


def test_parse_leaf_line():
    c = leaf_runner.parse_leaf_line("GPU: x\n" + LINE + "\n")
    assert c == {"M": 1024, "N": 34816, "K": 5120, "gs": 32, "zero": False, "dtype": "fp16",
                 "status": 0, "ms": 9.7012, "relL2": 1.234e-3, "pass": True}


def test_parse_leaf_line_fail_and_absent():
    bad = LINE.replace("PASS", "*** FAIL ***")
    assert leaf_runner.parse_leaf_line(bad)["pass"] is False
    assert leaf_runner.parse_leaf_line("HIP error at x.cpp:1 code=1") is None


def test_case_id_matches_workload():
    assert leaf_runner.case_id({"M": 1024, "K": 5120, "N": 34816, "gs": 32, "zp": False}) == \
        "M1024_K5120_N34816_gs32_noz"


def test_frozen_digest_ignores_kernel_and_task_location(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    out = tmp_path / "task"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr")
    monkeypatch.setattr(leaf_runner, "TASK_DIR", str(out))
    d0 = leaf_runner._frozen_digest(meta)
    (out / meta["kernel_file"]).write_text("// edited kernel\n")
    assert leaf_runner._frozen_digest(meta) == d0
    (out / meta["editable"][1]).write_text("// edited WMMA core header\n")
    assert leaf_runner._frozen_digest(meta) != d0
    d1 = leaf_runner._frozen_digest(meta)
    copy = tmp_path / "workspace"
    shutil.copytree(out, copy)
    monkeypatch.setattr(leaf_runner, "TASK_DIR", str(copy))
    assert leaf_runner._frozen_digest(meta) == d1


# -- the QMoE leaf --------------------------------------------------------------------

def test_generate_adds_the_qmoe_leaf_as_a_frozen_oracle(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    out = tmp_path / "task"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr")
    q = meta["qmoe"]
    assert (out / q["test_cpp"]).is_file()
    assert [x["file"] for x in q["sources"]] == ["src/lib/Runtime/Kernels/hip/qmoe_ragged_mm_kernel.hip",
                                                 "src/lib/Runtime/Kernels/hip/qmoe_kernel.hip"]
    assert all((out / x["file"]).is_file() and "--offload-compress" in x["flags"] for x in q["sources"])
    assert (out / q["lut_stub"]).is_file()
    assert not {q["test_cpp"], q["lut_stub"], *(x["file"] for x in q["sources"])} & set(meta["editable"])


def test_leaf_lock_is_the_serving_lock(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    monkeypatch.setenv("SERVING_GPU_LOCK", str(tmp_path / "gpu.lock"))
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "task"), [1024], "/tr")
    assert meta["lock_path"] == str(tmp_path / "gpu.lock")


def test_qmoe_rebuilds_when_a_header_changes_not_the_matmul_kernel(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    out = tmp_path / "task"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr")
    monkeypatch.setattr(leaf_runner, "TASK_DIR", str(out))
    d0 = leaf_runner._qmoe_digest(meta)
    (out / meta["kernel_file"]).write_text("// edited kernel\n")
    assert leaf_runner._qmoe_digest(meta) == d0
    (out / meta["editable"][1]).write_text("// edited WMMA core header\n")
    assert leaf_runner._qmoe_digest(meta) != d0


def test_qmoe_verdict_needs_all_pass_and_a_clean_exit():
    assert leaf_runner.qmoe_passed(0, "qmoe_ragged_mm coverage=3 cases=40\n  tiny PASS relL2=1e-4\nALL PASS\n")
    assert not leaf_runner.qmoe_passed(0, "  tiny PASS relL2=1e-4\n  wider *** FAIL ***\n")
    assert not leaf_runner.qmoe_passed(134, "ALL PASS\n")


FAKE_HIPCC = """#!/bin/bash
echo "$*" >> "$(dirname "$0")/calls.log"
while [ $# -gt 0 ]; do [ "$1" = -o ] && touch "$2"; shift; done
"""


def test_qmoe_leaf_relinks_with_the_tasks_kernel_and_reuses_its_own_objects(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    out = tmp_path / "task"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr")
    cc = tmp_path / "hipcc"
    cc.write_text(FAKE_HIPCC)
    cc.chmod(0o755)
    meta["hipcc"] = str(cc)
    monkeypatch.setattr(leaf_runner, "TASK_DIR", str(out))
    monkeypatch.setattr(leaf_runner, "BUILD_DIR", str(out / "build"))
    monkeypatch.setattr(leaf_runner, "QMOE_EXE", str(out / "build" / "test_qmoe"))
    (out / "build").mkdir()
    for o in ("kernel.o", "autotune.o"):
        (out / "build" / o).write_text("")
    assert leaf_runner.build_qmoe(meta) is None
    first = (tmp_path / "calls.log").read_text().splitlines()
    assert len(first) == 5 and all(" -c " in f" {c} " for c in first[:4])
    link = first[-1].split()
    assert link[0] == "--hip-link" and str(out / "build" / "kernel.o") in link and \
        str(out / "build" / "autotune.o") in link, "the QMoE test also calls the matmul_nbits kernel"
    (tmp_path / "calls.log").write_text("")
    assert leaf_runner.build_qmoe(meta) is None
    assert [c.split()[0] for c in (tmp_path / "calls.log").read_text().splitlines()] == ["--hip-link"]


def test_correctness_ends_with_the_qmoe_leaf(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    leaf, qmoe = build / "test_direct", build / "test_qmoe"
    leaf.write_text('#!/bin/bash\n[ "$1" = --coverage ] && { echo "0 failed"; echo ALL PASSED; exit 0; }\n'
                    f'echo "{LINE}"\n')
    qmoe.write_text("#!/bin/bash\necho '  tiny PASS relL2=1e-4'\necho '  wider *** FAIL ***'\nexit 1\n")
    for exe in (leaf, qmoe):
        exe.chmod(0o755)
    monkeypatch.setattr(leaf_runner, "TASK_DIR", str(tmp_path))
    monkeypatch.setattr(leaf_runner, "BUILD_DIR", str(build))
    monkeypatch.setattr(leaf_runner, "EXE", str(leaf))
    monkeypatch.setattr(leaf_runner, "QMOE_EXE", str(qmoe))
    monkeypatch.setattr(leaf_runner, "build_qmoe", lambda meta: None)
    monkeypatch.setattr(leaf_runner, "_current", lambda meta: None)
    meta = {"shapes": [{"M": 1024, "K": 5120, "N": 34816, "gs": 32, "zp": False}], "guard_shapes": [],
            "therock": "/tr", "leaf": "Xfp16_Wu4_Yfp16", "arch": "gfx1151", "coverage": 1,
            "lock_path": str(tmp_path / "gpu.lock")}
    ok, err = leaf_runner.correctness(meta)
    assert not ok and "qmoe_ragged_mm" in err
    qmoe.write_text("#!/bin/bash\necho '  tiny PASS relL2=1e-4'\necho 'ALL PASS'\n")
    assert leaf_runner.correctness(meta) == (True, None)


# -- the runner never measures an older build ---------------------------------------

def _fake_task(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    out = tmp_path / "task"
    meta = make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(out), [1024], "/tr")
    cc = tmp_path / "hipcc"
    cc.write_text(FAKE_HIPCC)
    cc.chmod(0o755)
    meta["hipcc"] = str(cc)
    monkeypatch.setattr(leaf_runner, "TASK_DIR", str(out))
    monkeypatch.setattr(leaf_runner, "BUILD_DIR", str(out / "build"))
    monkeypatch.setattr(leaf_runner, "EXE", str(out / "build" / "test_direct"))
    monkeypatch.setattr(leaf_runner, "QMOE_EXE", str(out / "build" / "test_qmoe"))
    (out / "build").mkdir(exist_ok=True)
    return out, meta


def test_an_edit_since_the_last_compile_is_built_before_it_is_measured(tmp_path, monkeypatch):
    out, meta = _fake_task(tmp_path, monkeypatch)
    log = tmp_path / "calls.log"
    assert leaf_runner._current(meta) is None, "no build yet: it compiles"
    assert log.read_text().count("--hip-link") == 1
    log.write_text("")
    assert leaf_runner._current(meta) is None and log.read_text() == "", "unchanged: no rebuild"
    (out / meta["kernel_file"]).write_text("// edited kernel\n")
    assert leaf_runner._current(meta) is None
    assert "matmul_nbits_kernel.hip" in log.read_text(), "an edited kernel is rebuilt"


def _fake_leaf(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def test_performance_times_online_autotune_and_reports_the_median(tmp_path, monkeypatch):
    out, meta = _fake_task(tmp_path, monkeypatch)
    checked = []
    monkeypatch.setattr(leaf_runner, "_current", lambda m: checked.append("build checked"))
    n = tmp_path / "n"
    n.write_text("0")
    _fake_leaf(out / "build" / "test_direct", f"""echo "$HIPDNN_MATMUL_AUTOTUNE_MODE" > {tmp_path}/mode
i=$(( $(cat {n}) + 1 )); echo $i > {n}; ms=$(( (i % 3) + 1 ))
echo "  M=1024 N=34816 K=5120 gs=32 zero=0 dtype=fp16  status=0  $ms.0000 ms  relL2=1e-03  PASS"
""")
    meta["shapes"] = meta["shapes"][:1]
    meta["perf_reps"] = 3
    ok, cases, err = leaf_runner.performance(meta)
    assert ok, err
    assert (tmp_path / "mode").read_text().strip() == "online"
    assert cases[0]["params"]["times_ms"] == [2.0, 3.0, 1.0] and cases[0]["execution_time_ms"] == 2.0
    assert checked == ["build checked"], "performance checks the build is current first"


def test_correctness_fails_on_a_failing_shape_or_a_failing_sweep(tmp_path, monkeypatch):
    out, meta = _fake_task(tmp_path, monkeypatch)
    monkeypatch.setattr(leaf_runner, "_current", lambda m: "build step failed: kernel.o")
    assert leaf_runner.correctness(meta) == (False, "build step failed: kernel.o"), \
        "correctness checks the build is current first"
    monkeypatch.setattr(leaf_runner, "_current", lambda m: None)
    monkeypatch.setattr(leaf_runner, "build_qmoe", lambda m: None)
    _fake_leaf(out / "build" / "test_qmoe", "echo 'ALL PASS'\n")
    leaf = out / "build" / "test_direct"
    _fake_leaf(leaf, f'[ "$1" = --coverage ] && {{ echo "1 failed"; echo SOME FAILED; exit 1; }}\necho "{LINE}"\n')
    ok, err = leaf_runner.correctness(meta)
    assert not ok and "full run failed" in err
    _fake_leaf(leaf, f'echo "{LINE.replace("PASS", "*** FAIL ***")}"\n')
    ok, err = leaf_runner.correctness(meta)
    assert not ok and err.startswith("M1024_K5120_N34816_gs32_noz failed")


def test_the_frozen_objects_follow_the_lookup_table(tmp_path, monkeypatch):
    out, meta = _fake_task(tmp_path, monkeypatch)
    d0 = leaf_runner._frozen_digest(meta)
    (out / meta["lut_fb"]).write_bytes(b"refreshed table")
    assert leaf_runner._frozen_digest(meta) != d0


def test_another_includer_of_the_wmma_core_is_refused(tmp_path, monkeypatch):
    src, build, shapes = _fake_tree(tmp_path)
    (src / "lib" / "Runtime" / "Kernels" / "hip" / "new_moe_kernel.hip").write_text(
        "#include \"matmul_nbits_wmma_core.h\"\n")
    (src / "lib" / "Runtime" / "Kernels" / "README.md").write_text("GemmFp16U4Impl is the WMMA core.\n")
    _commit(src)
    monkeypatch.setattr(hipep_env, "product_command", lambda b, obj: CMD)
    with pytest.raises(SystemExit, match=r"\['lib/Runtime/Kernels/hip/new_moe_kernel.hip'\]"):
        make_task.generate(OP, "gfx1151", str(src), str(build), str(shapes), str(tmp_path / "t"), [1024], "/tr")


def test_the_runner_finds_the_shared_lock_code_in_the_task(tmp_path, monkeypatch):
    out, _ = _fake_task(tmp_path, monkeypatch)
    assert (out / "scripts" / "hipep_env.py").is_file()
