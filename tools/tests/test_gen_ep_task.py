#!/usr/bin/env python3
"""Coverage for tools/gen_ep_task.py — the CPU-testable generator paths: registry
lookup, shape presets, and a selfcheck round-trip that checks the emitted task
against the generator's own SOURCES (the selfcheck template + the runner), not a
checked-in example copy. This proves the generator faithfully reproduces its
inputs without committing a redundant example task.

    python3 -m pytest tools/tests/test_gen_ep_task.py
"""
import json
import os
import sys

import pytest
import yaml

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEAK_ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import gen_ep_task as gen  # noqa: E402

# Source-of-truth the generator copies from (both are committed framework files).
TEMPLATE = os.path.join(TOOLS, "selfcheck_templates", "layernorm")
RUNNER_SRC = os.path.join(TOOLS, "ep_task_runner.py")


# -- registry -------------------------------------------------------------------

def test_registry_loads_and_has_selfcheck_op():
    reg = gen.load_registry()
    ops = {r["op"] for r in reg["ops"]}
    assert {"matmul_nbits", "matmul_nbits_fast", "layer_norm", "layernorm_selfcheck"} <= ops


def test_find_op_returns_matching_row():
    reg = gen.load_registry()
    row = gen.find_op(reg, "layernorm_selfcheck")
    assert row["oracle"] == "selfcheck"
    assert row["template"] == "layernorm"
    assert "layernorm_launch" in row["frozen_symbols"]


def test_find_op_unknown_exits():
    reg = gen.load_registry()
    with pytest.raises(SystemExit):
        gen.find_op(reg, "does_not_exist")


def test_shape_preset_exists_for_selfcheck_row():
    reg = gen.load_registry()
    row = gen.find_op(reg, "layernorm_selfcheck")
    assert row["shapes_preset"] in gen.SHAPE_PRESETS
    assert len(gen.SHAPE_PRESETS[row["shapes_preset"]]) >= 1


# -- selfcheck generation round-trip (vs the generator's sources) ---------------

@pytest.fixture
def generated(tmp_path):
    reg = gen.load_registry()
    row = gen.find_op(reg, "layernorm_selfcheck")
    out = str(tmp_path / "gen")
    os.makedirs(out, exist_ok=True)
    meta, source_rel = gen.gen_selfcheck(row, "gfx1151", out)
    gen.write_task(out, row, meta, source_rel)
    return out, meta, source_rel


def _read(path):
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("rel", [
    "kernel_src/layernorm_kernel.hip",
    "harness/test_layernorm.cpp",
    "include/layernorm_op.h",
])
def test_vendored_files_are_copied_from_the_template(generated, rel):
    out, _, _ = generated
    assert _read(os.path.join(out, rel)) == _read(os.path.join(TEMPLATE, rel))


def test_runner_is_copied_from_the_tool_source(generated):
    out, _, _ = generated
    assert _read(os.path.join(out, "scripts", "task_runner.py")) == _read(RUNNER_SRC)


def test_generated_meta_is_well_formed(generated):
    out, meta, _ = generated
    with open(os.path.join(out, "task_meta.json")) as f:
        on_disk = json.load(f)
    assert meta == on_disk  # the returned meta is what gets written
    assert meta["op"] == "layernorm_selfcheck"
    assert meta["oracle"] == "selfcheck"
    assert meta["kernel_file"] == "layernorm_kernel.hip"
    assert meta["test_cpp"] == "test_layernorm.cpp"
    assert "layernorm_launch" in meta["frozen_symbols"]
    assert len(meta["shapes"]) >= 1


def test_generated_config_is_the_geak_contract(generated):
    out, _, source_rel = generated
    with open(os.path.join(out, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["source_file_path"] == [source_rel]
    for key in ("compile_command", "correctness_command", "performance_command"):
        assert cfg[key][0].startswith("python3 scripts/task_runner.py")
    assert cfg["task_type"] == "hip_ep2hip_ep"


def test_source_file_path_points_at_editable_kernel_only(generated):
    _, _, source_rel = generated
    assert source_rel == "kernel_src/layernorm_kernel.hip"


def test_frozen_symbol_not_in_editable_source_path(generated):
    # the frozen ABI lives in include/, never in the edited kernel path
    _, meta, source_rel = generated
    assert "layernorm_launch" in meta["frozen_symbols"]
    assert source_rel.startswith("kernel_src/")


def test_selfcheck_missing_template_exits(tmp_path):
    reg = gen.load_registry()
    row = dict(gen.find_op(reg, "layernorm_selfcheck"))
    row["template"] = "no_such_template"
    with pytest.raises(SystemExit):
        gen.gen_selfcheck(row, "gfx1151", str(tmp_path / "x"))


def test_selfcheck_readme_is_a_standalone_microbenchmark(generated):
    out, _, _ = generated
    readme = open(os.path.join(out, "README.md")).read()
    assert "standalone HIP microbenchmark" in readme
    assert "does NOT validate EP integration" in readme
    assert "EP code" not in readme
    # layernorm_launch is both a target and frozen: signature frozen, body editable
    assert "implementations in `kernel_src/` remain editable" in readme


# -- registry rows + input validation -------------------------------------------

def test_every_registry_row_is_valid():
    for row in gen.load_registry()["ops"]:
        gen.validate_row(row)


def test_validate_row_reports_missing_oracle_fields():
    row = dict(gen.find_op(gen.load_registry(), "layernorm_selfcheck"))
    del row["template"]
    with pytest.raises(SystemExit, match="template"):
        gen.validate_row(row)


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["gen_ep_task.py", *argv])
    gen.main()


def test_main_has_no_oracle_override(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--op", "layer_norm", "--oracle", "selfcheck",
              "--out", str(tmp_path / "o"))


def test_main_rejects_unsupported_arch(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="gfx90a"):
        _main(monkeypatch, "--op", "layernorm_selfcheck", "--arch", "gfx90a",
              "--out", str(tmp_path / "o"))


def test_main_rejects_missing_ep_repo(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="--ep-repo not found"):
        _main(monkeypatch, "--op", "matmul_nbits", "--ep-repo", str(tmp_path / "nope"),
              "--out", str(tmp_path / "o"))


def _fake_kernel(op):
    """A kernel file defining what the registry row freezes, like the real one."""
    row = gen.find_op(gen.load_registry(), op)
    lines = ['#include "hip_custom_kernels.h"', "#include <hip/hip_runtime.h>"]
    lines += [f"static constexpr int {c} = 16;" for c in row.get("frozen_constants", [])]
    lines += [f"__global__ void {e}(float* y) {{ y[0] = 1.0f; }}" for e in row["entry_points"]]
    externs = set(row["frozen_symbols"])
    for fn in row.get("frozen_functions", []):
        if fn not in externs:
            lines.append(f"static bool {fn}(int m) {{ return m > 1; }}")
    lines += [f'extern "C" int {s}(void* stream, int m) {{ return m; }}' for s in sorted(externs)]
    return "\n".join(lines) + "\n"


@pytest.fixture
def ep_repo(tmp_path):
    kdir = tmp_path / "ep" / "hip-ep" / "lib" / "Runtime" / "Kernels"
    (kdir / "hip").mkdir(parents=True)
    (kdir / "hip" / "layer_norm_kernel.hip").write_text(_fake_kernel("layer_norm"))
    (kdir / "hip" / "matmul_nbits_kernel.hip").write_text(_fake_kernel("matmul_nbits"))
    (kdir / "include").mkdir()
    (kdir / "include" / "hip_custom_kernels.h").write_text("// h\n")
    (kdir / "test" / "example" / "gemm_fp16u2").mkdir(parents=True)
    return tmp_path / "ep"


def test_numeric_requires_build_cmd_before_writing(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    with pytest.raises(SystemExit, match="--ep-build-cmd"):
        _main(monkeypatch, "--op", "layer_norm", "--ep-repo", str(ep_repo),
              "--ep-dll", "/x/ep.so", "--out", str(out))
    assert not out.exists()


def test_numeric_meta_carries_rebuild_config(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, "--op", "layer_norm", "--ep-repo", str(ep_repo),
          "--ep-dll", "/x/ep.so", "--ep-build-cmd", "make ep", "--out", str(out))
    nm = json.loads((out / "task_meta.json").read_text())["numeric"]
    assert nm["build_cmd"] == "make ep" and nm["ep_dll"] == "/x/ep.so"
    assert nm["kernel_lib"] == "/x/ep.so"  # defaults to the EP library
    assert nm["kernel_dst"] == str(ep_repo / "hip-ep" / "lib" / "Runtime" / "Kernels"
                                   / "hip" / "layer_norm_kernel.hip")
    readme = (out / "README.md").read_text()
    assert "--durations" in readme and "NOT kernel latency" in readme


@pytest.mark.parametrize("nested", [False, True])
def test_resolve_ep_root_accepts_flat_and_nested_layouts(tmp_path, nested):
    root = tmp_path / "repo" / ("hip-ep" if nested else "")
    (root / "lib" / "Runtime" / "Kernels").mkdir(parents=True)
    assert gen.resolve_ep_root(str(tmp_path / "repo")) == str(root).rstrip("/")


def test_resolve_ep_root_rejects_non_ep_dir(tmp_path):
    with pytest.raises(SystemExit, match="lib/Runtime/Kernels"):
        gen.resolve_ep_root(str(tmp_path))


def test_numeric_meta_records_resolved_ep_root(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, "--op", "layer_norm", "--ep-repo", str(ep_repo),
          "--ep-dll", "/x/ep.so", "--ep-build-cmd", "make k", "--out", str(out))
    nm = json.loads((out / "task_meta.json").read_text())["numeric"]
    assert nm["ep_repo"] == str(ep_repo / "hip-ep")  # the runner finds test/numeric here


def test_numeric_meta_records_separate_kernel_lib(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, "--op", "layer_norm", "--ep-repo", str(ep_repo),
          "--ep-dll", "/b/dll/ep.so", "--ep-kernel-lib", "/b/bin/libk.so",
          "--ep-build-cmd", "make k", "--out", str(out))
    nm = json.loads((out / "task_meta.json").read_text())["numeric"]
    assert nm["ep_dll"] == "/b/dll/ep.so" and nm["kernel_lib"] == "/b/bin/libk.so"


def test_fast_harness_without_test_cpp_exits_cleanly(monkeypatch, tmp_path, ep_repo):
    with pytest.raises(SystemExit, match="test_\\*.cpp"):
        _main(monkeypatch, "--op", "matmul_nbits_fast", "--ep-repo", str(ep_repo),
              "--out", str(tmp_path / "o"))


# -- matmul_nbits: numeric target with frozen gates + device-time perf ----------

def test_matmul_nbits_row_is_a_gfx1151_numeric_target():
    row = gen.find_op(gen.load_registry(), "matmul_nbits")
    assert row["oracle"] == "numeric" and row["arches"] == ["gfx1151"]
    assert "hip_matmul_nbits" in row["frozen_functions"]
    assert row["ep_env"] == {"HIPDNN_MATMUL_AUTOTUNE_MODE": "lookup"}
    ids = [c["id"] for c in row["perf"]["cases"]]
    assert len(ids) == len(set(ids)) >= 1
    for c in row["perf"]["cases"]:
        assert c["K"] % 32 == 0 and c.get("zp", "none") in ("none", "u8", "fp16")
    fast = gen.find_op(gen.load_registry(), "matmul_nbits_fast")
    assert fast["oracle"] == "fast"


def _numeric_args(ep_repo, out, *extra):
    return ("--op", "matmul_nbits", "--ep-repo", str(ep_repo), "--ep-dll", "/x/ep.so",
            "--ep-build-cmd", "make k", "--out", str(out), *extra)


def test_numeric_task_vendors_abi_checker_and_perf_driver(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out))
    assert _read(out / "scripts" / "ep_abi.py") == _read(os.path.join(TOOLS, "ep_abi.py"))
    assert _read(out / "harness" / "ep_perf_driver.py") == \
        _read(os.path.join(TOOLS, "ep_perf_driver.py"))
    meta = json.loads((out / "task_meta.json").read_text())
    row = gen.find_op(gen.load_registry(), "matmul_nbits")
    assert meta["numeric"]["perf"] == row["perf"]
    assert meta["numeric"]["ep_env"] == row["ep_env"]
    assert meta["numeric"]["upstream_ref"] == "origin/main"
    assert meta["frozen_functions"] == row["frozen_functions"]


def test_ep_env_flags_extend_the_registry_env(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out, "--ep-env", "LD_LIBRARY_PATH=/rock/lib",
                                      "--ep-env", "HIPDNN_MATMUL_AUTOTUNE_MODE=online"))
    env = json.loads((out / "task_meta.json").read_text())["numeric"]["ep_env"]
    assert env == {"HIPDNN_MATMUL_AUTOTUNE_MODE": "online", "LD_LIBRARY_PATH": "/rock/lib"}
    assert "`LD_LIBRARY_PATH=/rock/lib`" in (out / "README.md").read_text()


def test_malformed_ep_env_is_rejected(monkeypatch, tmp_path, ep_repo):
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _main(monkeypatch, *_numeric_args(ep_repo, tmp_path / "o", "--ep-env", "NOVALUE"))


def test_ep_python_is_recorded(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out, "--ep-python", sys.executable))
    assert json.loads((out / "task_meta.json").read_text())["numeric"]["python"] == \
        os.path.abspath(sys.executable)
    assert f"run under `{os.path.abspath(sys.executable)}`" in (out / "README.md").read_text()
    _main(monkeypatch, *_numeric_args(ep_repo, tmp_path / "d"))
    assert json.loads((tmp_path / "d" / "task_meta.json").read_text())["numeric"]["python"] is None
    assert "run under" not in (tmp_path / "d" / "README.md").read_text()


def test_non_executable_ep_python_is_rejected(monkeypatch, tmp_path, ep_repo):
    with pytest.raises(SystemExit, match="not an executable"):
        _main(monkeypatch, *_numeric_args(ep_repo, tmp_path / "o",
                                          "--ep-python", str(tmp_path / "nope")))


def test_numeric_meta_carries_the_abi_manifest(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out))
    abi = json.loads((out / "task_meta.json").read_text())["abi"]
    row = gen.find_op(gen.load_registry(), "matmul_nbits")
    assert set(row["frozen_symbols"]) <= set(abi["extern_c"])
    assert set(abi["functions"]) == set(row["frozen_functions"])
    assert set(abi["constants"]) == set(row["frozen_constants"])  # no gate reads another
    assert abi["required_includes"] == ["hip_custom_kernels.h"]
    assert abi["exports"] is None  # no --ep-kernel-lib given
    kernel = (out / "kernel_src" / "matmul_nbits_kernel.hip").read_text()
    import ep_abi
    assert ep_abi.check_source(kernel, abi) == []


def test_constants_a_gate_reads_are_frozen_and_listed(monkeypatch, tmp_path, ep_repo):
    kernel = ep_repo / "hip-ep" / "lib" / "Runtime" / "Kernels" / "hip" / "matmul_nbits_kernel.hip"
    kernel.write_text(kernel.read_text().replace(
        "static bool wmmaMinM(int m) { return m > 1; }",
        "static constexpr int kWmmaMinMDefault = 8;\n"
        "static bool wmmaMinM(int m) { return m > kWmmaMinMDefault; }"))
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out))
    meta = json.loads((out / "task_meta.json").read_text())
    assert meta["abi"]["constants"]["kWmmaMinMDefault"] == \
        "static constexpr int kWmmaMinMDefault=8;"
    assert meta["frozen_constants"] == ["kBlockDim", "kWmmaMinMDefault"]
    assert "  - `kWmmaMinMDefault`" in (out / "README.md").read_text()


def test_variables_a_gate_reads_are_frozen_and_listed(monkeypatch, tmp_path, ep_repo):
    kernel = ep_repo / "hip-ep" / "lib" / "Runtime" / "Kernels" / "hip" / "matmul_nbits_kernel.hip"
    kernel.write_text(kernel.read_text().replace(
        "static bool wmmaMinM(int m) { return m > 1; }",
        "static int g_min_m = -1;\n"
        "static bool wmmaMinM(int m) { if (g_min_m < 0) g_min_m = 8; return m > g_min_m; }"))
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out))
    meta = json.loads((out / "task_meta.json").read_text())
    assert meta["abi"]["variables"] == {"g_min_m": "static int g_min_m=-1;"}
    assert meta["abi"]["mutable"] == ["g_min_m"]
    assert meta["frozen_variables"] == ["g_min_m"]
    assert "  - `g_min_m`" in (out / "README.md").read_text()


def test_missing_kernel_lib_warns_that_exports_are_not_checked(monkeypatch, tmp_path, ep_repo,
                                                               capsys):
    _main(monkeypatch, *_numeric_args(ep_repo, tmp_path / "o"))
    assert "no --ep-kernel-lib given" in capsys.readouterr().err
    klib = tmp_path / "libk.so"
    _main(monkeypatch, *_numeric_args(ep_repo, tmp_path / "p", "--ep-kernel-lib", str(klib)))
    assert f"--ep-kernel-lib {klib} is not built yet" in capsys.readouterr().err


def test_duplicate_perf_case_ids_are_rejected():
    row = json.loads(json.dumps(gen.find_op(gen.load_registry(), "matmul_nbits")))
    row["perf"]["cases"].append(dict(row["perf"]["cases"][0]))
    with pytest.raises(SystemExit, match="not unique"):
        gen.validate_row(row)


def test_numeric_meta_records_kernel_lib_exports(monkeypatch, tmp_path, ep_repo):
    out, klib = tmp_path / "o", tmp_path / "libk.so"
    klib.write_text("ELF")
    monkeypatch.setattr(gen.ep_abi, "exported_c_symbols", lambda lib, nm="nm": ["hip_matmul_nbits"])
    _main(monkeypatch, *_numeric_args(ep_repo, out, "--ep-kernel-lib", str(klib)))
    meta = json.loads((out / "task_meta.json").read_text())
    assert meta["abi"]["exports"] == ["hip_matmul_nbits"]
    assert "same C symbols as the baseline" in (out / "README.md").read_text()


def test_stale_frozen_symbol_is_rejected(tmp_path, ep_repo):
    row = dict(gen.find_op(gen.load_registry(), "layer_norm"), frozen_symbols=["ZeroPointsU2"])
    with pytest.raises(SystemExit, match="ZeroPointsU2"):
        gen.gen_numeric(row, "gfx1151", str(ep_repo / "hip-ep"), str(tmp_path / "o"),
                        "/x/ep.so", "make k", [])


def test_stale_entry_point_is_rejected(tmp_path, ep_repo):
    row = dict(gen.find_op(gen.load_registry(), "matmul_nbits"),
               entry_points=["matmul_nbits_gemv_dp4a_kernel", "renamed_kernel"])
    with pytest.raises(SystemExit, match="renamed_kernel"):
        gen.gen_numeric(row, "gfx1151", str(ep_repo / "hip-ep"), str(tmp_path / "o"),
                        "/x/ep.so", "make k", [])


def test_missing_frozen_gate_is_rejected(tmp_path, ep_repo):
    row = dict(gen.find_op(gen.load_registry(), "layer_norm"), frozen_functions=["noSuchGate"])
    with pytest.raises(SystemExit, match="noSuchGate"):
        gen.gen_numeric(row, "gfx1151", str(ep_repo / "hip-ep"), str(tmp_path / "o"),
                        "/x/ep.so", "make k", [])


def _git(cwd, *args):
    import subprocess
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_numeric_meta_records_the_ep_base_commit(monkeypatch, tmp_path, ep_repo):
    _git(ep_repo, "init", "-q")
    _git(ep_repo, "add", ".")
    _git(ep_repo, "commit", "-q", "-m", "base")
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out, "--ep-upstream-ref", "upstream/main"))
    meta = json.loads((out / "task_meta.json").read_text())
    assert meta["ep_base_sha"] == _git(ep_repo, "rev-parse", "HEAD")
    assert meta["numeric"]["upstream_ref"] == "upstream/main"
    readme = (out / "README.md").read_text()
    assert "export-patch" in readme and meta["ep_base_sha"] in readme


def test_dirty_ep_kernel_is_refused(monkeypatch, tmp_path, ep_repo):
    _git(ep_repo, "init", "-q")
    _git(ep_repo, "add", ".")
    _git(ep_repo, "commit", "-q", "-m", "base")
    kernel = ep_repo / "hip-ep" / "lib" / "Runtime" / "Kernels" / "hip" / "matmul_nbits_kernel.hip"
    kernel.write_text(kernel.read_text() + "// local edit\n")
    with pytest.raises(SystemExit, match="uncommitted changes"):
        _main(monkeypatch, *_numeric_args(ep_repo, tmp_path / "o"))


def test_non_git_ep_tree_has_no_base_and_no_export_section(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out))
    assert json.loads((out / "task_meta.json").read_text())["ep_base_sha"] is None
    assert "export-patch" not in (out / "README.md").read_text()


def test_numeric_readme_describes_device_time_gates_and_env(monkeypatch, tmp_path, ep_repo):
    out = tmp_path / "o"
    _main(monkeypatch, *_numeric_args(ep_repo, out))
    readme = (out / "README.md").read_text()
    assert "median per-op GPU time" in readme and "--durations" not in readme
    assert "`HIPDNN_MATMUL_AUTOTUNE_MODE=lookup`" in readme
    assert "`preferWmmaTier`" in readme and "`kBlockDim`" in readme
    assert "m1_q_proj_u8zp" in readme
    # hip_matmul_nbits is both an ABI entry and a frozen definition
    assert "remain editable, except the definitions frozen whole below." in readme


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
