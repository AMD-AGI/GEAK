"""Tests for the deterministic reference-callable harness synthesizer.

Covers generality (op-agnostic), the guard conditions (return None -> LLM
fallback), the entry-point parser (incl. generic-dispatcher rejection), and the
dtype/shape primitive. GPU-dependent paths are skipped when CUDA is absent.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from minisweagent.kernel_languages import _io_dtypes as io
from minisweagent.run.preprocess_v3 import reference_harness as rh

torch = pytest.importorskip("torch")
_HAS_CUDA = torch.cuda.is_available()
_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")


# --- entry-point parsing (pure, no GPU) ------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("aiter/ops/moe_op.py(522): ck_moe_stage1_fwd", ("aiter.ops.moe_op", "ck_moe_stage1_fwd")),
        ("torch.nn.functional:relu", ("torch.nn.functional", "relu")),
        ({"entry_point": "pkg/sub/mod.py(7): fn"}, ("pkg.sub.mod", "fn")),
        ("torch/_ops.py(1197): __call__", None),   # generic dispatcher -> reject
        ("aiter/ops/x.py(1): run", None),          # generic 'run' -> reject
        ("", None),
        (None, None),
        ({"no_entry": 1}, None),
    ],
)
def test_parse_reference_entry_point(raw, expected):
    assert rh.parse_reference_entry_point(raw) == expected


# --- dtype / shape primitive (pure) ----------------------------------------
def test_parse_shape_token():
    assert io.parse_shape_token("(1073,3072) fp8") == ((1073, 3072), "fp8")
    assert io.parse_shape_token("(24960,) int") == ((24960,), "int")
    assert io.parse_shape_token("() ScalarList") == ((), "ScalarList")
    assert io.parse_shape_token("garbage") is None


def test_dtype_map_and_int_detection():
    assert io.torch_dtype_attr("fp8") == "float8_e4m3fnuz"
    assert io.torch_dtype_attr("bf16") == "bfloat16"
    assert io.torch_dtype_attr("weird_unknown") == "bfloat16"  # safe default
    assert io.is_int_token("int") and io.is_int_token("int64")
    assert not io.is_int_token("fp8")


def test_render_tensor_expr_emits_valid_tuple_literals():
    # 1-D must be a tuple literal with trailing comma
    assert "(24960,)" in io.render_tensor_expr((24960,), "int")
    assert "randint" in io.render_tensor_expr((24960,), "int")
    # fp8 built in bf16 then cast
    e = io.render_tensor_expr((64, 3072), "fp8")
    assert ".to(torch.float8_e4m3fnuz)" in e and "(64, 3072)" in e


def test_build_inputs_exprs_skips_unparseable_and_returns_none_when_empty():
    assert io.build_inputs_exprs([]) is None
    assert io.build_inputs_exprs([{"no_shape": 1}]) is None
    exprs = io.build_inputs_exprs([{"shape": "(8,8) bf16"}, {"shape": "(8,) int"}])
    assert exprs and len(exprs) == 2


# --- synthesizer guards (return None -> caller falls back to LLM) -----------
def test_synthesize_returns_none_on_unimportable_callable(tmp_path):
    assert rh.synthesize_reference_harness(
        reference_entry_point="no.such.module:fn",
        input_shapes=[{"shape": "(8,8) bf16"}],
        output_dir=tmp_path,
    ) is None


def test_synthesize_returns_none_on_no_shapes(tmp_path):
    assert rh.synthesize_reference_harness(
        reference_entry_point="torch.nn.functional:relu",
        input_shapes=[],
        output_dir=tmp_path,
    ) is None


def test_synthesize_returns_none_on_generic_dispatcher(tmp_path):
    assert rh.synthesize_reference_harness(
        reference_entry_point="torch/_ops.py(1): __call__",
        input_shapes=[{"shape": "(8,8) bf16"}],
        output_dir=tmp_path,
    ) is None


# --- generality: same synthesizer drives different ops, no op-specific code -
@_cuda
@pytest.mark.parametrize("func,shape", [("relu", "(64,3072) bf16"), ("gelu", "(128,256) bf16")])
def test_synthesize_and_run_is_op_agnostic(tmp_path, func, shape):
    hp = rh.synthesize_reference_harness(
        reference_entry_point=f"torch.nn.functional:{func}",
        input_shapes=[{"shape": shape}],
        output_dir=tmp_path,
    )
    assert hp and Path(hp).is_file()
    assert (tmp_path / "golden.pt").is_file()
    assert (tmp_path / "harness_shapes_source.txt").read_text() == "user_task:production"
    # 4-mode contract: required flags + markers present
    text = Path(hp).read_text()
    for flag in ("--correctness", "--benchmark", "--full-benchmark", "--profile"):
        assert flag in text
    for marker in ("GEAK_RESULT_LATENCY_MS", "GEAK_RESULT_SPEEDUP"):
        assert marker in text
    # correctness (candidate==original here -> OK vs pre-snapshot golden) + benchmark run
    r = subprocess.run([sys.executable, hp, "--correctness"], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, hp, "--benchmark"], capture_output=True, text=True, timeout=180)
    assert "GEAK_RESULT_LATENCY_MS=" in r.stdout


@_cuda
def test_passes_contract_validation(tmp_path):
    from minisweagent.kernel_languages.contract import validate_harness

    hp = rh.synthesize_reference_harness(
        reference_entry_point="torch.nn.functional:relu",
        input_shapes=[{"shape": "(32,32) bf16"}],
        output_dir=tmp_path,
    )
    assert hp
    # Non-aiter op -> no source-repo leak; validator must not raise.
    validate_harness(Path(hp))
