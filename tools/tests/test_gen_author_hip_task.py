#!/usr/bin/env python3
"""Coverage for tools/gen_author_hip_task.py — the CPU-testable generator paths:
registry lookup, extern-"C" signature parsing, and a full round-trip per op
asserting the emitted task has every file, a well-formed arg schema in header
order, a stub whose signature matches the frozen header, and syntactically valid
generated Python. No GPU / torch / hipcc needed.

    python3 -m pytest tools/tests/test_gen_author_hip_task.py
"""
import ast
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


def test_oracle_and_refs_are_syntactically_valid():
    ast.parse(open(gen.ORACLE_SRC).read())
    ast.parse(open(gen.FREEZE_SRC).read())
    for op in OPS:
        ref = gen.find_op(REG, op)["reference"]
        ast.parse(open(os.path.join(gen.REFS_DIR, f"{ref}.py")).read())
