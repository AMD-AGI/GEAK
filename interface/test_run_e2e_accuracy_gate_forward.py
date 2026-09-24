#!/usr/bin/env python3
"""The integrate-lane correctness bar must be selectable from a handoff.

`map_args` builds the workflow's arg object from an ALLOWLIST -- a key the handoff carries but the
allowlist does not name simply never reaches `A` in e2e_workflow.js. `accuracy_gate` was such a key,
which meant both opt-in gates existed but neither could be chosen by a handoff-driven run: it fell
back to byte-exact parity with no diagnostic, because nothing was wrong from the workflow's point of
view -- it just never saw the argument.

The test that matters is the second one: a handoff WITHOUT these keys must produce exactly the args
it produced before, or this forward is not additive.

Run: python3 -m pytest GEAK/interface/test_run_e2e_accuracy_gate_forward.py -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e_gate_forward", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()

_BASE = {
    "model_path": "/models/Qwen3-4B-Instruct-2507",
    "workload": {"isl": 1024, "osl": 128, "concurrency": 8},
    "tp": 1,
    "exp_root": "/tmp/exp",
    "inferencex_path": "/tmp/ix",
    "launch_recipe": "/tmp/exp/launch_baseline.sh",
    "framework": "vllm",
}


def _args(**extra):
    h = dict(_BASE)
    h.update(extra)
    return rx.map_args(h, timeout_s=600)


def test_gate_and_its_knobs_reach_the_workflow():
    a = _args(accuracy_gate="op_tolerance", op_tol=0.02, op_tol_floor_mult=3.0)
    assert a["accuracy_gate"] == "op_tolerance"
    assert a["op_tol"] == 0.02
    assert a["op_tol_floor_mult"] == 3.0


def test_the_other_gate_still_works_through_the_same_path():
    a = _args(accuracy_gate="gsm8k", accuracy_limit=500, accuracy_tol=0.02)
    assert a["accuracy_gate"] == "gsm8k"
    assert a["accuracy_limit"] == 500
    assert a["accuracy_tol"] == 0.02


def test_absent_keys_are_a_provable_no_op():
    """No key, no entry -- the workflow keeps its own defaults, byte-identical to before."""
    a = _args()
    for k in ("accuracy_gate", "accuracy_limit", "accuracy_tol", "op_tol", "op_tol_floor_mult"):
        assert k not in a, k


def test_types_are_coerced_so_a_json_string_still_works():
    """Handoffs are hand-edited JSON; "0.02" must not reach parseFloat as a quoted string."""
    a = _args(accuracy_gate=" op_tolerance ", op_tol="0.02", accuracy_limit="500")
    assert a["accuracy_gate"] == "op_tolerance"   # stripped
    assert isinstance(a["op_tol"], float) and a["op_tol"] == 0.02
    assert isinstance(a["accuracy_limit"], int) and a["accuracy_limit"] == 500
