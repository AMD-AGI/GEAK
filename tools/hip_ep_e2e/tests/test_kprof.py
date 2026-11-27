#!/usr/bin/env python3
"""Pure-logic coverage for tools/hip_ep_e2e/kprof.py: splitting a kernel trace
into forward passes, picking the measured prefill and its decode steps, kernel
shares, and kernel -> source lookup. Recording a trace needs the GPU.

    python3 -m pytest tools/hip_ep_e2e/tests/test_kprof.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import kprof  # noqa: E402

GEMM = "void (anonymous namespace)::MatMulNBitsWMMA_NoZP<128, 128, 4, 4, 8, false, 32>(__half const*, int)"
GEMV = "void matmul_nbits_gemv_dp4a_kernel<128, 2, 1>(int const*, int)"
NORM = "skip_rms_norm_kernel(float*, int)"


EMBED = "void gather_kernel<__half>(int const*, __half*)"


def _trace(generations, fw_gap_ns=1_000, gen_gap_ns=50_000_000):
    """generations: lists of forwards [(name, dur_ns), ...]; kernels 1 us apart, forwards
    fw_gap_ns apart (the prefill is followed by a long logits copy), generations gen_gap_ns."""
    rows, t = [], 0
    for gen in generations:
        for i, fw in enumerate(gen):
            for name, dur in fw:
                rows.append({"Kernel_Name": name, "Start_Timestamp": str(t),
                             "End_Timestamp": str(t + dur)})
                t += dur + 1_000
            t += 30_000_000 if i == 0 else fw_gap_ns
        t += gen_gap_ns
    return rows


PREFILL = [(EMBED, 50_000), (GEMM, 60_000_000), (NORM, 1_000_000), (GEMM, 30_000_000)]
DECODE = [(EMBED, 50_000), (GEMV, 3_000_000), (NORM, 100_000), (GEMV, 1_000_000)]
LOAD = [[("__amd_rocclr_copyBuffer", 1_000_000)]]


def test_phases_take_the_timed_generation():
    gen = [PREFILL] + [DECODE] * 4          # -g 4: a prefill plus 4 decode forwards
    rows = _trace([LOAD, gen, gen])
    ph = kprof.phases(rows, gen=4)
    assert ph["marker"] == "gather_kernel"
    assert [n for n, _ in ph["prefill"]["kernels"]] == [EMBED, GEMM, NORM, GEMM]
    assert len(ph["decode"]) == 4
    assert all([n for n, _ in d["kernels"]] == [EMBED, GEMV, NORM, GEMV] for d in ph["decode"])
    # the warmup generation is ignored: the timed prefill starts at launch 1 + 20
    assert ph["prefill"]["start"] == int(rows[1 + 20]["Start_Timestamp"])


def test_a_marker_in_the_middle_of_a_forward_is_refused():
    # the per-forward kernel that comes first in launch order is the final norm, not the start
    fw = [(GEMM, 60_000_000), ("final_norm(float*)", 10_000)]
    dec = [(GEMV, 3_000_000), ("final_norm(float*)", 10_000)]
    with pytest.raises(ValueError, match="final_norm does not open a forward: MatMulNBitsWMMA_NoZP"):
        kprof.phases(_trace([LOAD, [fw] + [dec] * 2, [fw] + [dec] * 2]), gen=2)


def test_phases_accept_gen_forwards_per_generation():
    gen = [PREFILL] + [DECODE] * 3
    ph = kprof.phases(_trace([LOAD, gen, gen]), gen=4)
    assert len(ph["decode"]) == 3


def test_phases_refuse_unknown_structure():
    gen = [PREFILL] + [DECODE] * 4
    with pytest.raises(ValueError, match="2 x \\(9 \\+ 1\\)"):
        kprof.phases(_trace([gen, gen]), gen=9)


def test_phases_refuse_when_prefill_is_not_first():
    gen = [DECODE] + [PREFILL] + [DECODE] * 3
    with pytest.raises(ValueError, match="cannot be told apart"):
        kprof.phases(_trace([gen, gen]), gen=4)


def test_short_names():
    assert kprof.short_name(GEMM) == "MatMulNBitsWMMA_NoZP<128, 128, 4, 4, 8, false, 32>"
    assert kprof.base_name(GEMM) == "MatMulNBitsWMMA_NoZP"
    assert kprof.base_name(NORM) == "skip_rms_norm_kernel"
    assert kprof.base_name("void ns::inner::k<int>(int)") == "k"


def test_table_shares_sum_to_one():
    gen = [PREFILL] + [DECODE] * 2
    ph = kprof.phases(_trace([gen, gen]), gen=2)
    t = kprof.table([ph["prefill"]], top=10)
    assert sum(k["share"] for k in t["by_base"]) == pytest.approx(1.0)
    by_base = {b["base"]: b for b in t["by_base"]}
    assert t["by_base"][0]["base"] == "MatMulNBitsWMMA_NoZP" and by_base["MatMulNBitsWMMA_NoZP"]["calls"] == 2
    assert by_base["MatMulNBitsWMMA_NoZP"]["ms"] == pytest.approx(90.0)


def test_source_lookup(tmp_path):
    k = tmp_path / "lib" / "Runtime" / "Kernels"
    (k / "hip").mkdir(parents=True)
    (k / "include").mkdir()
    (k / "hip" / "matmul_nbits_kernel.hip").write_text(
        "template <int A>\n__global__ void __launch_bounds__(256)\nMatMulNBitsWMMA_NoZP(int x) {}\n")
    (k / "include" / "hip_custom_kernels.h").write_text("// calls MatMulNBitsWMMA_NoZP\n")
    idx = kprof.SourceIndex(str(tmp_path))
    assert idx.find("MatMulNBitsWMMA_NoZP") == "lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip"
    assert idx.find("no_such_kernel") is None


def test_record_reads_this_runs_trace_not_an_earlier_one(tmp_path, monkeypatch):
    out = tmp_path / "profile"
    (out / "old_host" / "1").mkdir(parents=True)
    (out / "old_host" / "1" / "run_kernel_trace.csv").write_text("an earlier run")

    def rocprof(cmd, **kwargs):
        (out / "host" / "2").mkdir(parents=True)
        (out / "host" / "2" / "run_kernel_trace.csv").write_text("this run")
        return kprof.hipep_env.subprocess.CompletedProcess(cmd, 0, "", "")

    env = kprof.hipep_env.Env(pkg=str(tmp_path / "pkg"), therock="/tr", sysroot="/s")
    monkeypatch.setenv("SERVING_GPU_LOCK", str(tmp_path / "lock"))
    monkeypatch.setattr(kprof.bench, "prompt_file", lambda model, length, d: "prompt.txt")
    monkeypatch.setattr(kprof.hipep_env.subprocess, "run", rocprof)
    trace = kprof.record(env, "m", str(out), 1024, 4)
    assert open(trace).read() == "this run"


def test_record_refuses_more_than_one_trace(tmp_path, monkeypatch):
    out = tmp_path / "profile"

    def rocprof(cmd, **kwargs):
        for pid in ("1", "2"):  # e.g. a child process traced too
            (out / "host" / pid).mkdir(parents=True)
            (out / "host" / pid / "run_kernel_trace.csv").write_text(pid)
        return kprof.hipep_env.subprocess.CompletedProcess(cmd, 0, "", "")

    env = kprof.hipep_env.Env(pkg=str(tmp_path / "pkg"), therock="/tr", sysroot="/s")
    monkeypatch.setenv("SERVING_GPU_LOCK", str(tmp_path / "lock"))
    monkeypatch.setattr(kprof.bench, "prompt_file", lambda model, length, d: "prompt.txt")
    monkeypatch.setattr(kprof.hipep_env.subprocess, "run", rocprof)
    with pytest.raises(RuntimeError, match="2 kernel traces"):
        kprof.record(env, "m", str(out), 1024, 4)
