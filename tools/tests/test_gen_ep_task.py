#!/usr/bin/env python3
"""Coverage for tools/gen_ep_task.py — the CPU-testable generator paths: registry
lookup, shape presets, and a full selfcheck round-trip whose emitted kernel /
harness / header / config / runner match the checked-in example task
byte-for-byte (README + task_meta differ only in formatting, so those are
compared as parsed data, not bytes).

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

EXAMPLE = os.path.join(GEAK_ROOT, "examples", "tasks", "hip_ep_layernorm_selfcheck")


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


# -- selfcheck generation round-trip -------------------------------------------

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
    with open(path) as f:
        return f.read()


@pytest.mark.parametrize("rel", [
    "kernel_src/layernorm_kernel.hip",
    "harness/test_layernorm.cpp",
    "include/layernorm_op.h",
    "config.yaml",
    "scripts/task_runner.py",
])
def test_generated_files_match_checked_in_task(generated, rel):
    out, _, _ = generated
    assert _read(os.path.join(out, rel)) == _read(os.path.join(EXAMPLE, rel))


def test_generated_meta_matches_checked_in_semantically(generated):
    out, meta, _ = generated
    with open(os.path.join(EXAMPLE, "task_meta.json")) as f:
        checked = json.load(f)
    # parsed-equal despite formatting differences
    assert meta == checked
    with open(os.path.join(out, "task_meta.json")) as f:
        assert json.load(f) == checked


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
    out, meta, source_rel = generated
    assert "layernorm_launch" in meta["frozen_symbols"]
    assert source_rel.startswith("kernel_src/")


def test_selfcheck_missing_template_exits(tmp_path):
    reg = gen.load_registry()
    row = dict(gen.find_op(reg, "layernorm_selfcheck"))
    row["template"] = "no_such_template"
    with pytest.raises(SystemExit):
        gen.gen_selfcheck(row, "gfx1151", str(tmp_path / "x"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
