#!/usr/bin/env python3
"""Coverage for tools/gen_author_hip_task.py — the CPU-testable generator paths:
registry lookup and validation, extern-"C" signature parsing, a full round-trip
per op asserting the emitted task has every file, a well-formed arg schema in
header order, a stub whose signature matches the frozen header, and
syntactically valid generated Python, plus the generated oracle's arch
resolution. No GPU / torch / hipcc needed.

    python3 -m pytest tools/tests/test_gen_author_hip_task.py
"""
import ast
import copy
import importlib.util
import json
import os
import re
import sys

import pytest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEAK_ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import gen_author_hip_task as gen  # noqa: E402

REG = gen.load_registry()
OPS = [r["op"] for r in REG["ops"]]


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


# -- registry + signature parsing ----------------------------------------------

def test_registry_loads_and_has_seed_ops():
    assert {"rmsnorm", "layernorm"} <= set(OPS)


def test_find_op_unknown_exits():
    with pytest.raises(SystemExit):
        gen.find_op(REG, "does_not_exist")


@pytest.mark.parametrize("op", OPS)
def test_signature_parse_matches_header(op):
    row = gen.find_op(REG, op)
    hdr = os.path.join(TOOLS, row["abi_header"])
    sig = gen.parse_signature(hdr, row["launch_symbol"])
    # parsed signature = the header's extern "C" decl minus the trailing semicolon
    header_decl = 'extern "C"' + open(hdr).read().split('extern "C"', 1)[1].split(";")[0]
    assert _norm(sig) == _norm(header_decl)
    assert not sig.rstrip().endswith(";")
    # every arg name appears in the parsed signature
    for a in row["args"]:
        assert a["name"] in sig


# -- full round-trip per op -----------------------------------------------------

@pytest.mark.parametrize("op", OPS)
def test_roundtrip(op, tmp_path):
    out = str(tmp_path / f"author_{op}_hip")
    row = gen.find_op(REG, op)
    gen.write_task(out, row, "gfx1151")

    hdr_base = os.path.basename(row["abi_header"])
    ref = row["reference"]
    expected = [
        f"include/{hdr_base}",
        f"kernel_src/{op}_kernel.hip",
        f"baseline_src/{ref}_ref.py",
        "unittest.py", "harness_lib.py", "freeze.py", "meta.json", "README.md",
    ]
    for rel in expected:
        assert os.path.exists(os.path.join(out, rel)), f"missing {rel}"

    meta = json.load(open(os.path.join(out, "meta.json")))
    # arg schema preserved in exact registry (= header) order
    assert meta["args"] == row["args"]
    assert meta["launch_symbol"] == row["launch_symbol"]
    assert meta["baseline_callable"] == f"{ref}_ref:{ref}"
    assert meta["build"] is True
    assert meta["candidate_backends"] == ["hip"]
    assert meta["cases"] and all("sig" in c for c in meta["cases"])
    # integrity anchors filled by freeze.py
    for k in ("baseline_src_sha256", "harness_lib_sha256", "unittest_sha256"):
        assert len(meta[k]) == 64

    # stub: includes the header, carries the frozen signature verbatim, zeroes outs
    stub = open(os.path.join(out, "kernel_src", f"{op}_kernel.hip")).read()
    assert f'#include "../include/{hdr_base}"' in stub
    sig = gen.parse_signature(os.path.join(TOOLS, row["abi_header"]), row["launch_symbol"])
    assert _norm(sig) in _norm(stub)
    for a in row["args"]:
        if a["kind"] == "out_tensor":
            assert f"hipMemset({a['name']}" in stub

    # generated Python is syntactically valid
    ast.parse(open(os.path.join(out, "unittest.py")).read())
    ast.parse(open(os.path.join(out, "baseline_src", f"{ref}_ref.py")).read())
    ast.parse(open(os.path.join(out, "freeze.py")).read())


# -- input validation: clear exits, not tracebacks ------------------------------

def _row(**over):
    row = copy.deepcopy(gen.find_op(REG, "rmsnorm"))
    row.update(over)
    return row


@pytest.mark.parametrize("op", OPS)
def test_registry_rows_validate(op):
    gen.validate_row(gen.find_op(REG, op))


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r.pop("launch_symbol"), "missing registry fields"),
    (lambda r: r.update(shapes_preset="nope"), "unknown shapes_preset"),
    (lambda r: r.update(abi_header="no/such.h"), "abi_header not found"),
    (lambda r: r.update(reference="no_such_ref"), "reference not found"),
    (lambda r: r["args"][0].update(kind="weird"), "unknown kind"),
    (lambda r: r["args"][0].update(dtype="int4"), "needs a shape and a dtype"),
    (lambda r: r["args"][-1].pop("value"), "needs a value"),
    (lambda r: r.update(args=[a for a in r["args"] if a["kind"] != "out_tensor"]),
     "at least one out_tensor"),
    (lambda r: r["args"].append({"name": "K", "kind": "dim"}), "appear in no in_tensor"),
    (lambda r: r["args"][0].update(shape=["M", "Q"]), "lacks dims"),
])
def test_validate_row_exits_with_reason(mutate, needle):
    row = _row()
    mutate(row)
    with pytest.raises(SystemExit) as e:
        gen.validate_row(row)
    assert needle in str(e.value)


def test_main_rejects_unsupported_arch(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["gen", "--op", "rmsnorm", "--arch", "gfx000",
                                      "--out", str(tmp_path / "t")])
    with pytest.raises(SystemExit) as e:
        gen.main()
    assert "not supported" in str(e.value)
    assert not (tmp_path / "t").exists()


# -- generated oracle: arch resolution ------------------------------------------

@pytest.fixture
def oracle(tmp_path, monkeypatch):
    out = str(tmp_path / "author_rmsnorm_hip")
    gen.write_task(out, gen.find_op(REG, "rmsnorm"), "gfx1151")
    for var in ("GEAK_HIP_ARCH", "GPU_ARCHS", "PYTORCH_ROCM_ARCH"):
        monkeypatch.delenv(var, raising=False)
    spec = importlib.util.spec_from_file_location("author_oracle", f"{out}/unittest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_oracle_arch_defaults_to_task_arch(oracle):
    assert oracle._resolve_arch() == "gfx1151"


@pytest.mark.parametrize("var,val", [("GPU_ARCHS", "gfx1151"),
                                     ("PYTORCH_ROCM_ARCH", "gfx942;gfx1151:xnack-")])
def test_oracle_arch_accepts_agreeing_env(oracle, monkeypatch, var, val):
    monkeypatch.setenv(var, val)
    assert oracle._resolve_arch() == "gfx1151"


def test_oracle_arch_refuses_disagreeing_env(oracle, monkeypatch):
    monkeypatch.setenv("GPU_ARCHS", "gfx950")
    with pytest.raises(SystemExit) as e:
        oracle._resolve_arch()
    assert "gfx950" in str(e.value) and "gfx1151" in str(e.value)


def test_oracle_and_refs_are_syntactically_valid():
    ast.parse(open(gen.ORACLE_SRC).read())
    ast.parse(open(gen.FREEZE_SRC).read())
    for op in OPS:
        ref = gen.find_op(REG, op)["reference"]
        ast.parse(open(os.path.join(gen.REFS_DIR, f"{ref}.py")).read())
