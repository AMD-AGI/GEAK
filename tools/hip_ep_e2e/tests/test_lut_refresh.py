#!/usr/bin/env python3
"""Pure-logic coverage for tools/hip_ep_e2e/lut_refresh.py: the decoder-only
model view handed to extract_shapes.py, the attention geometry and hip-ep's gqa
sweep grid, and the private source copy at the tag. The sweeps and the build need
a GPU.

    python3 -m pytest tools/hip_ep_e2e/tests/test_lut_refresh.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lut_refresh  # noqa: E402

FAKE_EXTRACT = """
import argparse, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--models"); ap.add_argument("--out"); ap.add_argument("--bits")
a = ap.parse_args()
graphs = sorted(str(p.relative_to(a.models)) for p in pathlib.Path(a.models).rglob("*.onnx"))
rows = [f"{g},5120,{n},4,32,0,1" for g in graphs for n in (34816, 248320)]
open(a.out, "w").write("\\n".join(["model,K,N,bits,block_size,has_zp,node_count"] + rows) + "\\n")
"""


def _src(tmp_path):
    src = tmp_path / "hip-ep"
    at = src / "lib" / "Runtime" / "Kernels" / "hip" / "autotune" / "matmul_nbits"
    (at / "scripts").mkdir(parents=True)
    (at / "lut").mkdir()
    (at / "scripts" / "extract_shapes.py").write_text(FAKE_EXTRACT)
    (at / "lut" / "gfx1151.json").write_text("{}")
    gqa = src / "lib" / "Runtime" / "Kernels" / "hip" / "autotune" / "gqa" / "scripts"
    gqa.mkdir(parents=True)
    (gqa / "update_lut.py").write_text("GEOMETRIES = [(16, 8, 128, 0, 'qwen3-1.7b', 'prefill_v6x'),\n"
                                       "              (24, 4, 256, 0, 'qwen3.6-27b', 'prefill_v8x')]\n")
    (src / "lib" / "Runtime" / "Kernels" / "include").mkdir()
    (src / "lib" / "Runtime" / "Kernels" / "include" / "a.h").write_text("orig")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "tag"]):
        subprocess.run(cmd, cwd=src, check=True)
    return src


def test_extract_shapes_sees_only_the_decoder(tmp_path):
    src = _src(tmp_path)
    model = tmp_path / "qwen-27b"
    model.mkdir()
    (model / "model.onnx").write_text("g")
    (model / "mtp.onnx").write_text("g")
    out = tmp_path / "out"
    out.mkdir()
    csv, rows = lut_refresh.extract_shapes(str(src), str(model), str(out), str(out / "log"))
    # only the decoder graph, and no vocab projection
    assert rows == 1
    with open(csv) as f:
        assert f.read().splitlines()[1:] == ["qwen-27b/model.onnx,5120,34816,4,32,0,1"]


def test_private_tree_starts_from_the_tag(tmp_path):
    src = _src(tmp_path)
    lut = src / "lib" / "Runtime" / "Kernels" / "hip" / "autotune" / "matmul_nbits" / "lut" / "gfx1151.json"
    lut.write_text('{"left": "by the last build"}')
    root = lut_refresh.private_tree(str(src), str(tmp_path / "out"))
    with open(os.path.join(root, lut_refresh.hipep_env.autotune("matmul_nbits"), "lut", "gfx1151.json")) as f:
        assert f.read() == "{}"


def _model(tmp_path, heads, kv, d):
    model = tmp_path / "m"
    model.mkdir()
    (model / "genai_config.json").write_text(json.dumps({"model": {"decoder": {
        "num_attention_heads": heads, "num_key_value_heads": kv, "head_size": d}}}))
    return str(model)


def test_gqa_sweeps_the_models_geometry(tmp_path):
    root = lut_refresh.private_tree(str(_src(tmp_path)), str(tmp_path / "out"))
    (measure, build), _ = lut_refresh.Gqa.plan(root, _model(tmp_path, 24, 4, 256), str(tmp_path), "log")
    assert measure[:2] == ["--geometry", "24:4:256"] and build[:3] == ["--merge-existing", "--geometry", "24:4:256"]
    assert lut_refresh.Gqa.arch("gfx1151") == "1151"


def test_gqa_skips_a_geometry_its_sweep_grid_lacks(tmp_path):
    root = lut_refresh.private_tree(str(_src(tmp_path)), str(tmp_path / "out"))
    args, reason = lut_refresh.Gqa.plan(root, _model(tmp_path, 32, 8, 128), str(tmp_path), "log")
    assert args is None and "32:8:128" in reason


def test_outputs_of_an_earlier_run_are_cleared(tmp_path):
    (tmp_path / "overlay" / "lib").mkdir(parents=True)
    (tmp_path / "overlay" / "lib" / "old.fb").write_text("old table")
    (tmp_path / "kbuild").mkdir()
    (tmp_path / "lut_refresh.json").write_text("old verdict")
    (tmp_path / "matmul_nbits_diff.md").write_text("old diff")
    (tmp_path / "lut_refresh.log").write_text("kept")
    lut_refresh.clear_outputs(str(tmp_path))
    assert not any((tmp_path / f).exists() for f in ("overlay", "kbuild", "lut_refresh.json", "matmul_nbits_diff.md"))
    assert (tmp_path / "lut_refresh.log").exists()


def test_the_overlay_holds_only_the_changed_tables(tmp_path):
    root = tmp_path / "src"
    for op in ("matmul_nbits", "gqa"):
        lut = root / lut_refresh.hipep_env.autotune(op) / "lut"
        lut.mkdir(parents=True)
        for ext in ("json", "fb"):
            (lut / f"gfx1151.{ext}").write_text(f"{op} {ext}")
    lut_refresh.write_overlay(str(root), [str(root / lut_refresh.hipep_env.autotune("gqa") / "lut" / "gfx1151")],
                              str(tmp_path / "overlay"))
    files = sorted(str(p.relative_to(tmp_path / "overlay")) for p in (tmp_path / "overlay").rglob("*") if p.is_file())
    assert files == [f"lib/Runtime/Kernels/hip/autotune/gqa/lut/gfx1151.{ext}" for ext in ("fb", "json")]


class _FakeOp:
    """An op whose `update_lut.py build` rewrites its table as `new` (None: leaves it)."""
    def __init__(self, name, new):
        self.name, self.new = name, new

    def arch(self, gfx):
        return gfx

    def plan(self, root, model, out, log):
        return (["--m"], ["--merge-existing"]), ""

    def build_sweep(self, root, build, out, therock, log):
        return "sweep"


def _main(tmp_path, monkeypatch, ops, arch="gfx1151"):
    root = tmp_path / "out" / "src"
    calls = []
    for op in ops:
        lut = root / lut_refresh.hipep_env.autotune(op.name) / "lut"
        lut.mkdir(parents=True)
        (lut / "gfx1151.json").write_text('{"rows": 1}')
        (lut / "gfx1151.fb").write_bytes(b"fb")

    def run(cmd, log, env=None, cwd=None):
        calls.append(cmd)
        op = next((o for o in ops if f"/{o.name}/scripts/" in " ".join(cmd)), None)
        if op and "build" in cmd and op.new is not None:
            (root / lut_refresh.hipep_env.autotune(op.name) / "lut" / "gfx1151.json").write_text(op.new)

    monkeypatch.setattr(lut_refresh, "OPS", {o.name: o for o in ops})
    monkeypatch.setattr(lut_refresh, "private_tree", lambda src, out: str(root))
    monkeypatch.setattr(lut_refresh, "_run", run)
    monkeypatch.setattr(lut_refresh.hipep_env, "require", lambda k: "/" + k)
    monkeypatch.setattr(lut_refresh.hipep_env.Env, "from_environ",
                        staticmethod(lambda pkg=None: lut_refresh.hipep_env.Env("/p", "/tr", "/s",
                                                                                str(tmp_path / "l"), 1)))
    (tmp_path / "out" / "lut_refresh.json").parent.mkdir(exist_ok=True)
    (tmp_path / "out" / "lut_refresh.json").write_text("an earlier verdict")
    (tmp_path / "out" / "overlay" / "lib").mkdir(parents=True)  # an earlier run's overlay
    (tmp_path / "out" / "overlay" / "lib" / "old.fb").write_text("old")
    monkeypatch.setattr(sys, "argv", ["lut_refresh.py", "--model", "m", "--out", str(tmp_path / "out"),
                                      "--arch", arch, "--ops", ",".join(o.name for o in ops)])
    lut_refresh.main()
    with open(tmp_path / "out" / "lut_refresh.json") as f:
        return json.load(f)["ops"], calls


def test_only_changed_tables_are_built(tmp_path, monkeypatch):
    res, calls = _main(tmp_path, monkeypatch, [_FakeOp("matmul_nbits", '{"rows": 2}'), _FakeOp("gqa", None)])
    assert res == {"matmul_nbits": {"changed": True, "skipped": ""}, "gqa": {"changed": False, "skipped": ""}}
    build = [c for c in calls if c[1].endswith("build_kernels.sh")]
    assert len(build) == 1
    overlay = tmp_path / "out" / "overlay"
    assert sorted(str(p.relative_to(overlay)) for p in overlay.rglob("*") if p.is_file()) == \
        [f"lib/Runtime/Kernels/hip/autotune/matmul_nbits/lut/gfx1151.{e}" for e in ("fb", "json")]


def test_no_change_builds_nothing(tmp_path, monkeypatch):
    res, calls = _main(tmp_path, monkeypatch, [_FakeOp("matmul_nbits", '{"rows": 1}')])
    assert res == {"matmul_nbits": {"changed": False, "skipped": ""}}
    assert not any(c[1].endswith("build_kernels.sh") for c in calls)
    assert not (tmp_path / "out" / "overlay").exists()


def test_an_arch_hip_ep_ships_no_table_for_is_skipped(tmp_path, monkeypatch):
    res, calls = _main(tmp_path, monkeypatch, [_FakeOp("matmul_nbits", '{"rows": 2}')], arch="gfx1036")
    assert res == {"matmul_nbits": {"changed": False, "skipped": "hip-ep ships no matmul_nbits table for gfx1036"}}
    assert calls == []
