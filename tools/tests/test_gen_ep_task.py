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
    assert {"matmul_nbits", "layer_norm", "layernorm_selfcheck"} <= ops


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


@pytest.fixture
def ep_repo(tmp_path):
    kdir = tmp_path / "ep" / "hip-ep" / "lib" / "Runtime" / "Kernels"
    (kdir / "hip").mkdir(parents=True)
    (kdir / "hip" / "layer_norm_kernel.hip").write_text("// k\n")
    (kdir / "hip" / "matmul_nbits_kernel.hip").write_text("// k\n")
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
        _main(monkeypatch, "--op", "matmul_nbits", "--ep-repo", str(ep_repo),
              "--out", str(tmp_path / "o"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
