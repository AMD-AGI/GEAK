# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the tool-audit fixes to hw_budget.py, extract_sku.py and sku.json
(stop/continue constant, measured references, labels/exit codes, device LLC size, grid tail,
help size, MI308X verification marker). CPU only.

AUDIT_REPO=<checkout> runs the same tests against another checkout (used to show they fail
before the fixes)."""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(os.environ.get("AUDIT_REPO") or Path(__file__).resolve().parents[3])
KT = REPO / "kernel_workflow" / "scripts" / "kernel_tools"
DATA = REPO / "perf_knowledge" / "hardware" / "data"
HB = KT / "hw_budget.py"


def run(*args, env=None):
    e = dict(os.environ)
    e.pop("GEAK_HARDWARE_CONTEXT", None)
    e.update(env or {})
    return subprocess.run([sys.executable, str(HB), *map(str, args)], capture_output=True,
                          text=True, env=e, cwd="/tmp", timeout=120)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


GEMM_BF16 = ["--sku", "MI350X", "--brief", "--workload", "gemm", "--shapes", "M=8192,N=8192,K=8192",
             "--dtype", "bf16", "--measured-ms", "0.8789"]
EW = ["--sku", "MI350X", "--brief", "--tensors", "x:r:fp32:8192x8192", "y:r:fp32:8192x8192",
      "out:w:fp32:8192x8192", "--flops", "134217728", "--flops-engine", "valu", "--dtype", "fp32",
      "--grid", "65536", "--footprint-mb", "768", "--dispatches", "1"]


# F2 ------------------------------------------------------------------------------------------
def test_room_verdict_cutoffs_come_from_thresholds_json(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    t = json.loads((data / "thresholds.json").read_text())
    t["hw_budget"] = {"little_room_below": 1.9, "large_room_from": 3.0}
    (data / "thresholds.json").write_text(json.dumps(t))
    # a 1.84x multiple over a calibrated compute floor: "moderate" at 1.25/2.0, "little" at 1.9/3.0
    args = [*GEMM_BF16, "--measured-tflops", "2300"]
    base = run(*args)
    moved = run(*args, env={"GEAK_HW_DATA_DIR": str(data)})
    assert base.returncode == 0 and moved.returncode == 0, base.stderr + moved.stderr
    assert "moderate room" in base.stdout, base.stdout
    assert "little room" in moved.stdout, moved.stdout
    card = subprocess.run([sys.executable, str(HB), "--help"], capture_output=True, text=True).stdout
    th = json.loads((DATA / "thresholds.json").read_text())["hw_budget"]
    assert f"< {th['little_room_below']:g}x little room" in card, card


# F3 ------------------------------------------------------------------------------------------
def test_mi350x_measured_ceilings_carry_provenance():
    row = json.loads((DATA / "sku.json").read_text())["skus"]["MI350X"]
    mc = row["measured_ceilings"]
    metrics = {(m["metric"], m.get("dtype")) for m in mc}
    assert {("hbm_tb_s", None), ("mfma_tflops", "bf16"), ("gemm_tflops", "bf16"),
            ("gemm_tflops", "fp8")} <= metrics
    for m in mc:
        assert m["measured_on"] and m["clock"] and m["basis"] in ("measured", "derived", "device-reported")


def test_best_in_class_gemm_does_not_read_as_large_room():
    r = run(*GEMM_BF16)
    assert r.returncode == 0, r.stderr
    assert "large room" not in r.stdout
    # the vendor GEMM at its own shape is printed as a comparison, scoped to that shape
    flat = " ".join(r.stdout.split())
    assert "comparison, not a ceiling" in flat and "same shape as this kernel" in flat, r.stdout
    j = json.loads(run(*GEMM_BF16, "--json").stdout)["workload"]
    ref = j["measured_reference"]
    # the verdict follows the highest sustained-MFMA ceiling, not the vendor GEMM
    assert ref["axis"] == "compute" and ref["metric"] == "mfma_tflops" and ref["value"] == 2080.0, ref
    comp = j["comparison_references"]
    assert [c["value"] for c in comp] == [1251.0] and comp[0]["same_shape"], comp
    assert max(comp[0]["multiple"]) <= 1.05, comp


def test_kernel_at_56pct_of_datasheet_peak_is_not_little_room():
    # 2*4096^3 FLOP in 0.1067 ms = ~1288 TFLOPS = ~56% of the 2300 TF bf16 datasheet peak. Read
    # against the single-shape vendor GEMM figure (1251 TF at 8192^3) it would be "at the ceiling";
    # against the sustained-MFMA ceiling it still has moderate room.
    r = run("--sku", "MI350X", "--brief", "--workload", "gemm", "--shapes", "M=4096,N=4096,K=4096",
            "--dtype", "bf16", "--measured-ms", "0.1067")
    assert r.returncode == 0, r.stderr
    assert "little room" not in r.stdout, r.stdout
    assert "moderate room" in r.stdout, r.stdout
    flat = " ".join(r.stdout.split())
    assert "at M=8192,N=8192,K=8192 only" in flat and "does not set the verdict" in flat, r.stdout


def test_measured_ceiling_provenance_stays_inside_the_repo():
    mc = json.loads((DATA / "sku.json").read_text())["skus"]["MI350X"]["measured_ceilings"]
    for m in mc:
        for k in ("source", "tool", "measured_on"):
            v = m.get(k) or ""
            assert "val/" not in v and "tool_probe" not in v and "RESULTS.md" not in v, (k, v)


def test_shipped_data_has_no_out_of_repo_references():
    import re
    pat = re.compile(r"(^|[^A-Za-z_.])val/|gpu_verify|tool_probe|RESULTS\.md|geak_gluon_feat_ab")
    for name in ("sku.json", "hw_constants.json", "thresholds.json", "workload_models.json", "README.md"):
        text = (DATA / name).read_text()
        assert not pat.search(text), (name, pat.search(text).group(0))


# F4 ------------------------------------------------------------------------------------------
def test_calibrated_memory_bound_is_not_labelled_datasheet():
    r = run(*EW, "--measured-ms", "0.1823", "--measured-hbm-tb-s", "4.673,4.687")
    assert r.returncode == 0, r.stderr
    assert "upper bound: datasheet" not in r.stdout, r.stdout
    assert "mem_bw_probe.py ->" not in r.stdout, r.stdout


def test_valu_engine_without_peak_says_so():
    r = run(*EW, "--measured-ms", "0.1823")
    assert "no VALU peak tabulated" in r.stdout and "--measured-tflops" in r.stdout, r.stdout
    assert "compute floor UNKNOWN" not in r.stdout


def test_brief_exits_2_below_the_floor():
    r = run(*EW, "--measured-ms", "0.05")
    assert r.returncode == 2, (r.returncode, r.stdout)


def test_floor_us_from_another_timer_is_a_breach_not_little_room():
    r = run("--sku", "MI350X", "--brief", "--workload", "gemm", "--shapes", "M=8,N=5120,K=2880",
            "--dtype", "bf16", "--measured-ms", "0.0148", "--grid", "320", "--floor-us", "13")
    assert r.returncode == 2, (r.returncode, r.stdout)
    assert "--floor-us does not match the timer" in r.stdout, r.stdout
    assert "little room" not in r.stdout


# F11 -----------------------------------------------------------------------------------------
def test_residency_uses_the_device_reported_llc_size():
    hb = load("hb_audit_f11", HB)
    common = dict(tensors="x:r:fp32:8192x7680, out:w:fp32:8192x7680", flops=0, grid=65536,
                  measured_ms=0.1)
    # 240 MiB: inside the 256 MB datasheet figure, outside the 224 MiB the MI350X reports
    w = hb.budget("MI350X", footprint_mb=240, **common)["workload"]
    assert not w["roofline_preconditions_refused"], w["roofline_preconditions_refused"]
    assert w["cache_residency"]["llc_mb"] == 224
    inside = hb.budget("MI350X", footprint_mb=200, **common)["workload"]["cache_residency"]["_refused"]
    assert "may be partly cache-served" in inside and "measure warm vs read-evicted" in inside


# F13 -----------------------------------------------------------------------------------------
def test_grid_tail_reports_the_time_factor():
    es = load("es_audit_f13", KT / "extract_sku.py")
    two_and_half = es.grid_waves(640, 256)
    assert "TAIL" in two_and_half and "1.20x" in two_and_half, two_and_half
    assert "TAIL" not in es.grid_waves(512, 256)
    hb = load("hb_audit_f13", HB)
    mf = hb.budget("MI350X", tensors="A:r:fp8:1e6", flops=0, grid=640)["workload"]["machine_fill"]
    assert mf.get("tail_time_factor") == 1.2 and "_tail" in mf, mf


# slim ----------------------------------------------------------------------------------------
def test_help_is_short_and_dropped_flags_are_gone():
    full = subprocess.run([sys.executable, str(HB), "--help-full"], capture_output=True, text=True).stdout
    assert len(full.splitlines()) <= 90, len(full.splitlines())
    for gone in ("--from-census", "--grid-redundancy"):
        assert gone not in full
    for tool in (HB, KT / "extract_sku.py"):
        card = subprocess.run([sys.executable, str(tool), "--help"], capture_output=True, text=True).stdout
        assert "WHY:" not in card and "Look them up" not in card, tool


# TP006 / extract_sku --check -----------------------------------------------------------------
def test_mi308x_topology_is_marked_unverified():
    row = json.loads((DATA / "sku.json").read_text())["skus"]["MI308X"]
    v = row["verification"]
    assert v["status"] == "unverified" and {"xcds", "l2_mb", "peak_tflops"} <= set(v["fields"])
    assert "datasheet" in v["reason"] and "MI308X" in v["reason"]


def test_check_requires_provenance_on_measured_ceilings():
    es = load("es_audit_check", KT / "extract_sku.py")
    doc = es.load()
    assert es.check(doc) == []
    bad = copy.deepcopy(doc)
    bad["skus"]["MI350X"]["measured_ceilings"][0].pop("clock")
    assert any("measured_on and clock" in x for x in es.check(bad))


def test_grid_refusal_exits_3_and_help_says_so():
    r = run("--sku", "MI350X", "--brief", "--workload", "gemm", "--shapes", "M=8,N=512,K=512",
            "--dtype", "bf16", "--measured-ms", "0.01", "--grid", "4")
    assert r.returncode == 3, (r.returncode, r.stdout)
    assert "does not fill the machine" in " ".join(r.stdout.split()), r.stdout
    card = subprocess.run([sys.executable, str(HB), "--help"], capture_output=True, text=True).stdout
    assert "exit 3" in card, card


# Round 3 --------------------------------------------------------------------------------------
READ_1G = "X:r:bf16:1073741824"


def test_beating_the_measured_ceiling_exits_2_without_a_verdict():
    # 2 GiB read in 0.39 ms beats the measured read-stream ceiling but not the datasheet floor.
    r = run("--sku", "MI350X", "--brief", "--tensors", READ_1G, "--flops", "0", "--measured-ms", "0.39")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    flat = " ".join(r.stdout.split())
    assert "ceiling is inapplicable" in flat and "over-counted" in flat, r.stdout
    assert "little room" not in flat and "moderate room" not in flat and "large room" not in flat, r.stdout


@pytest.mark.parametrize("mode", ["--brief", None])
def test_beating_the_measured_ceiling_prints_no_lever(mode):
    r = run("--sku", "MI350X", *([mode] if mode else []), "--tensors", READ_1G, "--flops", "0",
            "--measured-ms", "0.39")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "ceiling is inapplicable" in " ".join(r.stdout.split()), r.stdout
    assert not any(l.strip().startswith("lever:") for l in r.stdout.splitlines()), r.stdout
    # ...while the same tensors at a time above the ceiling still get one
    ok = run("--sku", "MI350X", *([mode] if mode else []), "--tensors", READ_1G, "--flops", "0",
             "--measured-ms", "2.0")
    assert any(l.strip().startswith("lever:") for l in ok.stdout.splitlines()), ok.stdout


@pytest.mark.parametrize("tensors,access", [
    (READ_1G, "linear read"),
    ("X:r:bf16:1073741824 Y:w:bf16:536870912", "2:1 read:write mix"),
    ("X:rw:bf16:1073741824", "in-place update (rmw)"),
])
def test_measured_hbm_row_follows_the_access_mix(tensors, access):
    r = run("--sku", "MI350X", "--json", "--tensors", tensors, "--flops", "0", "--measured-ms", "2.0")
    assert r.returncode == 0, (r.stdout, r.stderr)
    ref = json.loads(r.stdout)["workload"]["measured_reference"]
    assert ref["access"] == access, ref
    assert "access mix" in ref["mix_note"], ref


@pytest.mark.parametrize("tensors,far", [
    ("X:r:bf16:1073741824 Y:w:bf16:536870912", False),   # 2:1, the row's own ratio
    ("X:r:bf16:1073741824 Y:w:bf16:1073741824", True),   # 1:1
    ("X:r:bf16:1073741824 Y:w:bf16:134217728", True),    # 8:1
])
def test_rw_row_says_when_the_kernel_ratio_differs(tensors, far):
    r = run("--sku", "MI350X", "--json", "--tensors", tensors, "--flops", "0", "--measured-ms", "2.0")
    assert r.returncode == 0, (r.stdout, r.stderr)
    ref = json.loads(r.stdout)["workload"]["measured_reference"]
    assert ref["access"] == "2:1 read:write mix", ref
    assert ("nearest row; ratio differs" in ref["mix_note"]) == far, ref
    if far:
        brief = run("--sku", "MI350X", "--brief", "--tensors", tensors, "--flops", "0", "--measured-ms", "2.0")
        assert "nearest row; ratio differs" in " ".join(brief.stdout.split()), brief.stdout


@pytest.mark.parametrize("tensors,probe_mix,capped", [
    ("X:r:bf16:1073741824 Y:w:bf16:1073741824", "1:1", False),
    ("X:r:bf16:1073741824 Y:w:bf16:805306368", "4:3", False),   # 1.33:1 after rounding
    ("X:r:bf16:1073741824 Y:w:bf16:134217728", "7:1", True),    # 8:1: mem_bw_probe caps at 7:1
])
def test_rw_ratio_note_names_a_mix_the_probe_runs(tensors, probe_mix, capped):
    r = run("--sku", "MI350X", "--json", "--tensors", tensors, "--flops", "0", "--measured-ms", "2.0")
    note = json.loads(r.stdout)["workload"]["measured_reference"]["mix_note"]
    assert f"mem_bw_probe.py --rw-mix {probe_mix}" in note, note
    assert ("caps at 7:1" in note) == capped, note


def test_in_range_ratio_note_does_not_claim_the_cap():
    # 1.25:1 is inside 7:1..1:7 but no group of <=8 runs expresses it: say that, not "caps at"
    r = run("--sku", "MI350X", "--json", "--tensors", "X:r:bf16:1073741824 Y:w:bf16:858993459",
            "--flops", "0", "--measured-ms", "2.0")
    note = json.loads(r.stdout)["workload"]["measured_reference"]["mix_note"]
    assert "caps at" not in note and "cannot express" in note, note


def test_named_workload_says_the_mix_is_unknown():
    r = run("--sku", "MI350X", "--json", "--workload", "rmsnorm", "--shapes", "M=65536,N=8192",
            "--dtype", "bf16", "--measured-ms", "1")
    ref = json.loads(r.stdout)["workload"]["measured_reference"]
    assert ref["axis"] == "memory" and ref["access"] == "linear read", ref
    assert "access mix unknown" in ref["mix_note"], ref


def test_rows_without_basis_are_listed_not_used():
    r = run("--sku", "R9700", "--json", "--tensors", READ_1G, "--flops", "0", "--measured-ms", "5")
    assert r.returncode == 0, r.stderr
    w = json.loads(r.stdout)["workload"]
    assert "measured_reference" not in w, w
    assert "no recorded basis" in w["measured_ceilings_not_used"], w
    assert "hbm_tb_s_read_plus_write" in w["measured_ceilings_not_used"], w
    assert any(l.startswith("note: ") for l in json.loads(r.stdout)["bottom_line"])


def test_schema_text_matches_the_tools_use_of_measured_ceilings():
    text = json.loads((DATA / "sku.json").read_text())["_schema"]["measured_ceilings"]
    assert "never as a default specification" not in text
    for word in ("hw_budget", "access mix", "basis"):
        assert word in text, (word, text)


def test_unknown_dtype_peak_suppresses_verdict_and_lever():
    r = run("--sku", "MI300X", "--brief", "--workload", "gemm", "--shapes", "M=64,N=4096,K=4096",
            "--dtype", "fp4", "--measured-ms", "0.1")
    assert r.returncode == 0, r.stderr
    flat = " ".join(r.stdout.split())
    assert "no fp4 entry in peak_tflops" in flat, r.stdout
    assert "no room verdict" in flat, r.stdout
    for word in ("little room", "moderate room", "large room", "lever:"):
        assert word not in flat, (word, r.stdout)
