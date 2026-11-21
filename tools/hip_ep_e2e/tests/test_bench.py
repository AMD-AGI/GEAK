#!/usr/bin/env python3
"""Pure-logic coverage for tools/hip_ep_e2e/bench.py: the model_benchmark output
parser, the A/B gate, and the ABBA run order. The runs themselves are verified on
the gfx1151 box, not here.

    python3 -m pytest tools/hip_ep_e2e/tests/test_bench.py
"""
import argparse
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bench  # noqa: E402

SAMPLE = os.path.join(HERE, "data", "mb_qwen3_0p6b.txt")


def _sample():
    with open(SAMPLE) as f:
        return f.read()


# -- parser ---------------------------------------------------------------------

def test_parse_sample():
    r = bench.parse_model_benchmark(_sample())
    assert r["ttft_us_p50"] == pytest.approx(17046.1)
    assert r["decode_us_p50"] == pytest.approx(3944.39)
    assert r["prompt_tokens"] == 128
    assert r["gen_tokens"] == 16


def test_parse_truncated_raises():
    text = _sample().split("Token generation:")[0]
    with pytest.raises(ValueError, match="Token generation"):
        bench.parse_model_benchmark(text)


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        bench.parse_model_benchmark("Failed to load library /x/libamdgpu-ep.so\n")


# -- gate -----------------------------------------------------------------------

def _runs(ttft, decode, lens=(1024, 2048)):
    """Two runs per length; `ttft` / `decode` map length -> per-run values (us)."""
    return {L: {"ttft": list(ttft[L]), "decode": list(decode[L])} for L in lens}


BASE = _runs({1024: [2000e3, 2010e3], 2048: [4000e3, 4020e3]},
             {1024: [71.0e3, 71.2e3], 2048: [72.0e3, 72.2e3]})


N = {"ttft": 0.01, "decode": 0.01}


def test_gate_accepts_clear_ttft_win():
    cand = _runs({1024: [1900e3, 1905e3], 2048: [3800e3, 3810e3]},
                 {1024: [71.0e3, 71.2e3], 2048: [72.0e3, 72.2e3]})
    g = bench.gate(BASE, cand, N)
    assert g["accepted"] and g["won"] == ["ttft"]
    assert g["per_len"][1024]["ttft_gain"] == pytest.approx(1 - 1902.5 / 2005)
    assert g["per_len"][2048]["decode_held"]


def test_gate_accepts_clear_decode_win():
    cand = _runs({1024: [2000e3, 2010e3], 2048: [4000e3, 4020e3]},
                 {1024: [68.0e3, 68.2e3], 2048: [69.0e3, 69.2e3]})
    g = bench.gate(BASE, cand, N)
    assert g["accepted"] and g["won"] == ["decode"]


def test_gate_rejects_gain_inside_noise():
    cand = _runs({1024: [1995e3, 2000e3], 2048: [3980e3, 4000e3]},
                 {1024: [71.0e3, 71.2e3], 2048: [72.0e3, 72.2e3]})
    g = bench.gate(BASE, cand, N)
    assert not g["accepted"]
    assert not g["per_len"][1024]["ttft_win"]


def test_gate_rejects_decode_regression():
    cand = _runs({1024: [1800e3, 1800e3], 2048: [3600e3, 3600e3]},
                 {1024: [72.6e3, 72.6e3], 2048: [72.0e3, 72.2e3]})
    g = bench.gate(BASE, cand, N)
    assert not g["accepted"]
    assert g["per_len"][1024]["ttft_win"]
    assert not g["per_len"][1024]["decode_held"]


def test_gate_rejects_ttft_regression_behind_a_decode_win():
    cand = _runs({1024: [2100e3, 2100e3], 2048: [4000e3, 4020e3]},
                 {1024: [68.0e3, 68.2e3], 2048: [69.0e3, 69.2e3]})
    g = bench.gate(BASE, cand, N)
    assert not g["accepted"]
    assert not g["per_len"][1024]["ttft_held"]


def test_gate_uses_each_metrics_own_band():
    cand = _runs({1024: [1900e3, 1905e3], 2048: [3800e3, 3810e3]},
                 {1024: [72.0e3, 72.2e3], 2048: [73.0e3, 73.2e3]})
    assert not bench.gate(BASE, cand, N)["accepted"]
    assert bench.gate(BASE, cand, {"ttft": 0.01, "decode": 0.02})["accepted"]
    # a 1.5% decode win is outside a 1% band but inside a 2% one
    cand = _runs({1024: [2000e3, 2010e3], 2048: [4000e3, 4020e3]},
                 {1024: [69.9e3, 70.1e3], 2048: [70.9e3, 71.1e3]})
    assert bench.gate(BASE, cand, N)["accepted"]
    assert not bench.gate(BASE, cand, {"ttft": 0.01, "decode": 0.02})["accepted"]


def test_gate_needs_every_length():
    cand = _runs({1024: [1800e3, 1800e3], 2048: [4000e3, 4020e3]},
                 {1024: [71.0e3, 71.2e3], 2048: [72.0e3, 72.2e3]})
    assert not bench.gate(BASE, cand, N)["accepted"]


def test_gate_missing_length_raises():
    cand = {1024: BASE[1024]}
    with pytest.raises(KeyError):
        bench.gate(BASE, cand, N)


def test_gate_without_lengths_raises():
    with pytest.raises(ValueError, match="at least one prompt length"):
        bench.gate(BASE, BASE, N, lens=())


# -- run order ------------------------------------------------------------------

def test_abba_order():
    assert bench.run_order(2) == [("a", "b"), ("b", "a")]
    assert bench.run_order(3) == [("a", "b"), ("b", "a"), ("a", "b")]


# -- command line -----------------------------------------------------------------

def test_lengths_and_counts_must_be_positive():
    assert bench._lens("1024,2048") == [1024, 2048]
    for bad in ("", ",", "0,1024"):
        with pytest.raises(argparse.ArgumentTypeError):
            bench._lens(bad)
    with pytest.raises(argparse.ArgumentTypeError):
        bench._positive("0")


def test_failed_ab_leaves_no_verdict(tmp_path, monkeypatch):
    out = tmp_path / "ab"
    out.mkdir()
    (out / "ab.json").write_text(json.dumps({"gate": {"accepted": True}}))

    def broken(*args, **kwargs):
        raise RuntimeError("model_benchmark exited 1")

    monkeypatch.setattr(bench.hipep_env.Env, "from_environ", staticmethod(lambda pkg=None: pkg))
    monkeypatch.setattr(bench, "collect", broken)
    monkeypatch.setattr(sys, "argv", ["bench.py", "ab", "--model", "m", "--base", "a",
                                      "--cand", "b", "--out", str(out)])
    with pytest.raises(RuntimeError):
        bench.main()
    assert not (out / "ab.json").exists(), "an earlier run's verdict survived a failed rerun"


def test_counts_are_checked_on_the_command_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench.py", "ab", "--model", "m", "--base", "a", "--cand", "b",
                                      "--out", "/tmp/x", "--reps", "0"])
    with pytest.raises(SystemExit) as e:
        bench.main()
    assert e.value.code == 2


def _fake_run(tmp_path, monkeypatch, script):
    exe = tmp_path / "model_benchmark"
    exe.write_text("#!/bin/bash\n" + script)
    exe.chmod(0o755)
    env = bench.hipep_env.Env(pkg=str(tmp_path), therock="/tr", sysroot="/s",
                              lock_path=str(tmp_path / "gpu.lock"), lock_wait=5)
    monkeypatch.setattr(bench.hipep_env, "pkg_argv", lambda e, x, args=(): [str(exe), *args])
    monkeypatch.setattr(bench, "prompt_file", lambda model, length, out: "/dev/null")
    (tmp_path / "lib").mkdir(exist_ok=True)
    return env


def test_a_failed_run_raises(tmp_path, monkeypatch):
    env = _fake_run(tmp_path, monkeypatch, "echo boom; exit 7\n")
    with pytest.raises(RuntimeError, match="exited 7"):
        bench.run_once(env, "m", 1024, 4, 1, str(tmp_path / "r.log"))


def test_a_run_of_the_wrong_length_raises(tmp_path, monkeypatch):
    env = _fake_run(tmp_path, monkeypatch, f"cat {SAMPLE}\n")
    with pytest.raises(RuntimeError, match="prompt has 128 tokens, expected about 1024"):
        bench.run_once(env, "m", 1024, 16, 1, str(tmp_path / "r.log"))
    assert bench.run_once(env, "m", 128, 16, 1, str(tmp_path / "r.log"))["ttft_us_p50"] > 0
