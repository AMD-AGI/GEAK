"""kf_trace_digest.py on a synthetic trace with known answers (+ the muse traces when present)."""
import gzip
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kf_trace_digest as D  # noqa: E402

MUSE_POINTS = Path("/home/yixiongh/geak/muse/kf_reuse/eval/fusion_capture/sampling/points")


def synthetic(path: Path):
    """Two DECODE steps of 3 layers x [norm, gemm, add] + one EXTEND step launched eagerly."""
    ev, ts, corr = [], 1000.0, 1
    for step in range(2):
        start = ts
        for layer in range(3):
            for name, dur in (("rms_norm_kernel", 2.0), ("Cijk_gemm", 10.0), ("add_kernel", 1.0)):
                ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": 7,
                           "args": {"correlation": corr, "stream": 7}})
                ev.append({"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch", "ts": 10, "dur": 1,
                           "pid": 9, "tid": 9, "args": {"correlation": corr}})
                corr += 1
                ts += dur + 1.0
        ev.append({"ph": "X", "cat": "gpu_user_annotation", "name": "step[DECODE bs=4]", "ts": start,
                   "dur": ts - start, "pid": 1, "tid": 7})
        ts += 50
    start = ts
    ev += [{"ph": "X", "cat": "python_function", "name": "sglang/srt/layers/norm.py(10): forward",
            "ts": 5000, "dur": 100, "pid": 9, "tid": 9},
           {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "ts": 5010, "dur": 20, "pid": 9, "tid": 9},
           {"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": 5015, "dur": 2, "pid": 9, "tid": 9,
            "args": {"correlation": 999}},
           {"ph": "X", "cat": "kernel", "name": "mul_kernel", "ts": start, "dur": 4.0, "pid": 1, "tid": 7,
            "args": {"correlation": 999, "stream": 7}},
           {"ph": "X", "cat": "gpu_user_annotation", "name": "step[EXTEND bs=1]", "ts": start, "dur": 8.0,
            "pid": 1, "tid": 7}]
    with gzip.open(path, "wt") as f:
        json.dump({"traceEvents": ev}, f)


def test_synthetic_digest(tmp_path):
    pt = tmp_path / "traces" / "de_4"
    pt.mkdir(parents=True)
    synthetic(pt / "123-TP-0.trace.json.gz")
    synthetic(pt / "123-TP-1.trace.json.gz")
    assert D.main(["--trace-dir", str(tmp_path / "traces"), "--out", str(tmp_path / "out")]) == 0
    d = json.loads((tmp_path / "out" / "digest.json").read_text())["groups"]["de_4"]
    assert d["trace"].endswith("TP-0.trace.json.gz")           # rank 0 picked
    dec, ext = d["step_kinds"]["DECODE"], d["step_kinds"]["EXTEND"]
    assert dec["steps"] == 2 and dec["kernels_per_step"] == 9 and dec["busy_us_per_step"] == 39.0
    assert dec["layer_period"]["period"] == 3 and dec["layer_period"]["repeats"] == 3.0
    assert dec["graph_replayed"] is True
    assert dec["short_kernel_share"]["count_frac"] == pytest.approx(2 / 3, abs=1e-3)
    assert dec["idle_frac"] == pytest.approx(1 - 39 / 48, abs=1e-3)   # 9 one-us gaps inside the 48 us span
    top = {r["kernel"]: r for r in dec["top_kernels"]}
    assert top["Cijk_gemm"]["calls_per_step"] == 3 and top["Cijk_gemm"]["us_per_step"] == 30
    seq = (tmp_path / "out" / "seq_de_4_EXTEND.tsv").read_text().splitlines()
    assert seq[1].split("\t")[6:] == ["aten::mul", "sglang/srt/layers/norm.py(10): forward"]
    assert ext["steps"] == 1


def test_skips_truncated_and_kernel_less_traces(tmp_path):
    """A failed capture leaves a truncated .gz (and maybe a salvaged CPU-only JSON) next to the
    good re-capture; the digest must use the good one, whatever the names sort to."""
    pt = tmp_path / "traces" / "round_0"
    pt.mkdir(parents=True)
    good = pt / "200-TP-0.trace.json.gz"
    synthetic(good)
    os.utime(good, (1, 1))                                   # oldest
    whole = (pt / "200-TP-0.trace.json.gz").read_bytes()
    (pt / "100-TP-0.trace.json.gz").write_bytes(whole[: len(whole) // 2])   # truncated, newer
    (pt / "salvaged-TP-0.trace.json").write_text(json.dumps(
        {"traceEvents": [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 1, "dur": 1}]}))
    assert D.main(["--trace-dir", str(tmp_path / "traces"), "--out", str(tmp_path / "out")]) == 0
    d = json.loads((tmp_path / "out" / "digest.json").read_text())["groups"]["round_0"]
    assert d["trace"] == str(good) and d["step_kinds"]["DECODE"]["steps"] == 2
    (good).unlink()
    assert D.main(["--trace-dir", str(tmp_path / "traces"), "--out", str(tmp_path / "out2")]) == 3


def test_short_name():
    assert D.short_name("_ZN5aiter26cross_device_reduce_1stageIDF16bLi8ELb0EEEvPNS_8R") == "aiter::cross_device_reduce_1stage"
    assert D.short_name("void at::native::reduce_kernel<512, 1>(at::native::ReduceOp<float>)").startswith("at::native::reduce_kernel<512")
    assert D.short_name("_fused_norm_kernel(int, int)") == "_fused_norm_kernel"


@pytest.mark.skipif(not MUSE_POINTS.is_dir(), reason="muse traces not on this box")
def test_muse_traces_match_parse_profile(tmp_path):
    """Cross-check: digest per-step device time x steps == the trace's kernel time inside step windows,
    and the decode layer period matches the model's layer count."""
    assert D.main(["--trace-dir", str(MUSE_POINTS), "--out", str(tmp_path)]) == 0
    g = json.loads((tmp_path / "digest.json").read_text())["groups"]
    assert set(g) == {"de_1", "de_2", "pr_16384", "pr_2571"}
    ev = D.load_events(g["de_1"]["trace"])
    steps = [e for e in ev if e.get("cat") == "gpu_user_annotation" and D.STEP_RE.match(e["name"])]
    inside = sum(k["dur"] for k in ev if k.get("ph") == "X" and k.get("cat") in D.KERNEL_CATS
                 and any(s["ts"] <= k["ts"] <= s["ts"] + s["dur"] for s in steps))
    got = sum(s["busy_us_per_step"] * s["steps"] for s in g["de_1"]["step_kinds"].values())
    assert got == pytest.approx(inside, rel=1e-3)
    verify = g["de_1"]["step_kinds"]["VERIFY"]
    assert verify["graph_replayed"] and 55 <= verify["layer_period"]["repeats"] <= 70
