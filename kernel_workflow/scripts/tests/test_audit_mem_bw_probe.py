"""Regression tests for mem_bw_probe.py (tool audit F5). CPU only: plan math, layout, sizing, verdict text.

F5 on MI350X: the linear read probe read ~10% below an independent tiled stream, the 2:1 mix ~13% low,
`--runlen 256 --stride 4096` crashed (int32 element offsets past 2**31), and the <60%-of-peak warning
fired on a correct gfx950 reading (best reachable stream is ~66% of the 8 TB/s datasheet value).
"""
import contextlib
import importlib.util
import inspect
import io
from pathlib import Path

import pytest

KT = Path(__file__).resolve().parents[1] / "kernel_tools"


@pytest.fixture(scope="module")
def mbp():
    spec = importlib.util.spec_from_file_location("mem_bw_probe_audit", KT / "mem_bw_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INT32_MAX = 2 ** 31 - 1


@pytest.mark.parametrize("runlen", [64, 128, 256, 1024])
def test_short_runs_at_large_stride_stay_inside_their_buffers(mbp, runlen):
    # 1 GiB of useful bytes at a 4 KiB stride spans 16-64 GiB: element offsets pass 2**31
    p = mbp.plan(1024.0, runlen, 4096, 1, 0, None)
    last_read, _ = mbp.run_offsets(p, p["n_groups"] - 1)
    assert max(last_read) + p["run_elems"] <= p["read_span_bytes"] // 4
    assert mbp.verify_plan(p, p["read_span_bytes"] + p["write_span_bytes"]) == []
    if max(last_read) > INT32_MAX:
        # the kernel must form element offsets in 64-bit, or the address wraps (the old crash)
        assert ".to(tl.int64)" in inspect.getsource(mbp._build_kernel)


def test_pure_read_allocates_no_write_span(mbp):
    p = mbp.plan(1024.0, 64, 4096, 1, 0, None)
    assert p["write_span_bytes"] == 0
    assert p["read_span_bytes"] == p["span_bytes"] == p["n_groups"] * 4096


def test_linear_mixed_stream_is_two_contiguous_streams(mbp):
    # a 2:1 linear stream reads one contiguous buffer and writes another; interleaving the write
    # slots into the read range made both streams strided and under-issued the mix
    p = mbp.plan(64.0, 4096, 4096, 2, 1, None)
    reads, writes = [], []
    for g in range(8):
        r, w = mbp.run_offsets(p, g)
        reads += r
        writes += w
    assert reads == [i * p["run_elems"] for i in range(16)]
    assert writes == [i * p["run_elems"] for i in range(8)]
    assert p["read_span_bytes"] == p["read_bytes"] and p["write_span_bytes"] == p["write_bytes"]


@pytest.mark.parametrize("runlen", [64, 256, 640, 4096, 16384])
def test_each_tile_iteration_keeps_16_kib_in_flight(mbp, runlen):
    p = mbp.plan(2048.0, runlen, max(runlen, 4096), 1, 0, None)
    assert p["rows"] * p["run_p2"] * 4 >= 16384
    assert p["groups_per_prog"] % p["rows"] == 0
    # default parallelism: a few tile iterations per program, not a long serial loop
    assert p["groups_per_prog"] // p["rows"] <= 8
    assert p["moved_bytes"] == p["n_groups"] * runlen


def test_explicit_program_count_is_kept(mbp):
    p = mbp.plan(1024.0, 4096, 4096, 1, 0, 1980)
    assert p["nprog"] == 1980 and p["n_groups"] == p["groups_per_prog"] * 1980


def test_shaped_reference_is_the_linear_stream(mbp):
    rp = mbp.reference_plan(1024.0, 256, 2, 1, None)
    assert rp["run_bytes"] == rp["stride_bytes"] == mbp.LINEAR_REF_RUNLEN


def _emit_text(mbp, readings, runlen=4096, stride=4096):
    sku = mbp.load_sku("MI350X")
    p = mbp.plan(2048.0, runlen, stride, 1, 0, None)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mbp._emit("MI350X", sku, p, readings, 2048.0, "t", None)
    return buf.getvalue()


def test_gfx950_linear_ceiling_is_not_flagged_as_under_saturating(mbp):
    # 4.7-5.3 TB/s is what a correct linear stream reaches on MI350X (59-66% of 8.0)
    assert "WARNING" not in _emit_text(mbp, [4.70, 4.72])
    assert "WARNING" in _emit_text(mbp, [2.0, 2.1])


def test_help_states_the_short_run_cost(mbp):
    assert "128/runlen" in mbp.QUICK_HELP and "128 B" in mbp.QUICK_HELP
    assert len(mbp.QUICK_HELP.strip().splitlines()) <= 22


def _run_main(mbp, monkeypatch, tmp_path, *argv):
    """main() with measure() stubbed out: argument handling and the record, no GPU."""
    monkeypatch.delenv("GEAK_HARDWARE_CONTEXT", raising=False)
    monkeypatch.setattr(mbp, "measure", lambda p, *a, **k: [4.0, 4.1, 4.05])
    out = tmp_path / "rec.json"
    monkeypatch.setattr("sys.argv", ["mem_bw_probe.py", "--sku", "MI355X", "--json", str(out), *argv])
    with contextlib.redirect_stdout(io.StringIO()):
        mbp.main()
    import json
    return json.loads(out.read_text())


def test_record_is_labelled_with_the_ratio_measured(mbp, monkeypatch, tmp_path, capsys):
    # a group holds at most 8 runs, so 9:1 is timed as 7:1: the record must say 7:1, and warn
    rec = _run_main(mbp, monkeypatch, tmp_path, "--rw-mix", "9:1")
    assert (rec["shape"]["nr"], rec["shape"]["nw"]) == (7, 1)
    assert rec["label"] == "7:1 mix" and rec["rw_mix_measured"] == "7:1", rec["label"]
    assert rec["rw_mix_requested"] == "9:1"
    assert "run as 7:1" in capsys.readouterr().err
    # an expressible ratio is labelled with its reduced pair and draws no warning
    rec = _run_main(mbp, monkeypatch, tmp_path, "--rw-mix", "40:60")
    assert rec["label"] == "2:3 mix" and rec["rw_mix_measured"] == "2:3"
    assert "WARNING" not in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["nan:1", "1:nan", "inf:1", "1:inf"])
def test_non_finite_rw_mix_is_rejected(mbp, bad):
    with pytest.raises(ValueError):
        mbp.rw_mix_to_runs(bad)


@pytest.mark.parametrize("argv", [
    ["--rw-mix", "abc"], ["--rw-mix", "nan:1"], ["--runlen", "0"], ["--runlen", "-4"],
    ["--nprog-sweep", "a,b"], ["--nprog-sweep", "100,0"], ["--nprog", "0"],
    ["--runlen-sweep", "64,x"], ["--footprint-mb", "-1"], ["--footprint-mb", "nan"],
    ["--repeats", "0"], ["--iters", "0"], ["--warmup", "-1"], ["--stride", "6"],
    ["--runlen", "4096", "--stride", "2048"],
])
def test_bad_arguments_are_usage_errors(argv):
    # refused by argparse before any SKU/context/GPU work: exit 2, a usage message, no traceback
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "GEAK_HARDWARE_CONTEXT"}
    cp = subprocess.run([sys.executable, str(KT / "mem_bw_probe.py"), "--sku", "MI355X", *argv],
                        capture_output=True, text=True, env=env, timeout=60)
    assert cp.returncode == 2 and "error:" in cp.stderr, (argv, cp.returncode, cp.stderr)
    assert "Traceback" not in cp.stderr, cp.stderr


def test_rmw_with_stride_names_the_rmw_conflict():
    # --rmw has its own shape; the refusal names that, not the stride/runlen overlap check
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "GEAK_HARDWARE_CONTEXT"}
    cp = subprocess.run([sys.executable, str(KT / "mem_bw_probe.py"), "--sku", "MI355X", "--rmw",
                         "--stride", "2048"], capture_output=True, text=True, env=env, timeout=60)
    assert cp.returncode == 2 and "drop --stride" in cp.stderr, cp.stderr
