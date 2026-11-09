# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for shared specification data and local planning tools.
Native TraceLens models and calibration are independent of this SKU table.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
KT = REPO / "kernel_workflow" / "scripts" / "kernel_tools"
HW_DATA = REPO / "perf_knowledge" / "hardware" / "data"
ROOFLINE_DIR = REPO / "e2e_workflow" / "knowledge" / "analysis_skills" / "roofline"

TOOLS = ["hw_budget", "calc_perf", "mem_bw_probe", "extract_sku"]


def run(*args, env=None, cwd=None, timeout=300):
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run([str(a) for a in args], text=True, capture_output=True, check=False,
                          env=full_env, cwd=cwd or "/tmp", timeout=timeout)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


@pytest.fixture(scope="module")
def sku():
    return json.loads((HW_DATA / "sku.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def calc_perf():
    return _load("calc_perf_under_test", KT / "calc_perf.py")


@pytest.fixture(scope="module")
def extract_sku():
    return _load("extract_sku_under_test", KT / "extract_sku.py")



# --------------------------------------------------------------------------- selftests

@pytest.mark.parametrize("tool", TOOLS)
def test_roofline_tool_selftest(tool):
    path = KT / f"{tool}.py"
    result = run(sys.executable, path, "--selftest")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert "SELFTEST PASS" in out, out[-3000:]


# --------------------------------------------------------------------------- one source, agreeing

def test_calc_perf_sku_peaks_are_read_from_sku_json(calc_perf, sku):
    assert list(calc_perf.SKU_PEAKS) == list(sku["skus"])
    for name, row in sku["skus"].items():
        assert calc_perf.SKU_PEAKS[name] == row, name
    # no hard-coded duplicate table left in the source
    src = (KT / "calc_perf.py").read_text()
    assert '"peak_tflops": {"fp16": 1307.4' not in src and "GFX950 = {" not in src



def test_hw_budget_reads_the_same_rows():
    hb = _load("hw_budget_under_test", KT / "hw_budget.py")
    for name, dtype, want in (("MI355X", "fp4", 10100.0), ("MI350X", "bf16", 2300.0),
                              ("MI325X", "fp8", 2614.9)):
        w = hb.budget(name, "gemm", {"M": 4096, "N": 4096, "K": 4096}, dtype)["workload"]
        assert w["compute_ceiling_tflops"] == [want, want], (name, dtype)
        assert w["denominator_basis"] == "datasheet" and w["numerator_basis"] == "model"
        assert w["may_gate"] is False
    w = hb.budget("MI325X", "gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16")["workload"]
    assert w["hbm_ceiling_tb_s"] == [6.0, 6.0]


def test_decided_sku_numbers(sku):
    s = sku["skus"]
    assert list(s)[:2] == ["MI355X", "MI350X"], "gfx950 rows first"
    want = {
        "MI355X": dict(fp16=2500.0, bf16=2500.0, fp8=5000.0, fp4=10100.0, fp32=157.3, fp64=78.6),
        "MI350X": dict(fp16=2300.0, bf16=2300.0, fp8=4600.0, fp4=9200.0, fp32=144.2),
        "MI300X": dict(fp16=1307.4, bf16=1307.4, fp8=2614.9, fp32=163.4),
        "MI325X": dict(fp16=1307.4, bf16=1307.4, fp8=2614.9, fp32=163.4),
    }
    for name, w in want.items():
        for dt, v in w.items():
            assert s[name]["peak_tflops"][dt] == v, (name, dt)
    assert [s[n]["peak_hbm_tb_s"] for n in ("MI355X", "MI350X", "MI300X", "MI325X")] == [8.0, 8.0, 5.3, 6.0]
    assert [s[n]["cus"] for n in ("MI355X", "MI350X", "MI300X", "MI325X")] == [256, 256, 304, 304]
    g = s["AI_MAX_395"]
    assert (g["peak_hbm_tb_s"], g["peak_hbm_basis"], g["datasheet_hbm_tb_s"]) == (0.212, "measured", 0.256)
    assert (g["l2_mb"], g["mall_mb"]) == (2, 32)
    assert g["basis"] == "derived" and "fp32" in g["basis_note"]
    for n, r in s.items():
        if r["arch"] in ("gfx1100", "gfx1200"):
            assert r["geak_support"] == "unsupported", n


def test_per_arch_dtype_ratio_is_constant(sku):
    """Same CU design within an arch: CU count and clock cancel in a dtype:fp16 ratio."""
    by_arch = {}
    for name, row in sku["skus"].items():
        by_arch.setdefault(row["arch"], []).append(row["peak_tflops"])
    checked = 0
    for arch, rows in by_arch.items():
        for dt in {d for r in rows for d in r} - {"fp16"}:
            ratios = [r[dt] / r["fp16"] for r in rows if dt in r]
            if len(ratios) >= 2:
                assert (max(ratios) - min(ratios)) / max(ratios) < 0.02, (arch, dt, ratios)
                checked += 1
        for r in rows:
            assert r.get("bf16", r["fp16"]) == r["fp16"], arch   # bf16 == fp16 everywhere
    assert checked >= 8


# --------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("name,dtype", [("MI300X", "fp4"), ("R9700", "fp4"), ("AI_MAX_395", "fp8"),
                                        ("MI308X", "fp64")])
def test_missing_dtype_is_refused_not_fp16(calc_perf, name, dtype):
    with pytest.raises(SystemExit, match="no " + dtype + " peak"):
        calc_perf.resolve_sku(name, dtype=dtype)


def test_calc_perf_refuses_without_a_target():
    r = run(sys.executable, KT / "calc_perf.py", "roofline", "--M", "64", "--N", "64", "--K", "64",
            env={"PATH": "/nonexistent"})          # no rocminfo either
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)


def test_hw_budget_refuses_without_a_sku():
    r = run(sys.executable, KT / "hw_budget.py", "--workload", "gemm", "--shapes", "M=64,N=64,K=64")
    assert r.returncode != 0 and "--sku is required" in r.stderr



# --------------------------------------------------------------------------- derived docs

def test_generated_doc_blocks_are_current(extract_sku):
    files = list(extract_sku.doc_files())
    rel = {str(Path(f).relative_to(REPO)) for f in files}
    for must in ("kernel_workflow/knowledge/amd_instinct.md",
                 "perf_knowledge/hardware/cdna4_mi350/peak_tables.md",
                 "perf_knowledge/hardware/cdna3_mi300/peak_tables.md"):
        assert must in rel, must
    assert extract_sku.check_docs(extract_sku.load()) == []


def test_sku_json_invariants_clean(extract_sku):
    assert extract_sku.check(extract_sku.load()) == []


def test_thresholds_one_bound_classification_table():
    t = json.loads((HW_DATA / "thresholds.json").read_text())
    bc = t["bound_classification"]
    assert "roof_util" not in bc
    assert "headroom_vs_target_eff" not in bc
    assert (bc["sol_pipe_pct"]["high"], bc["sol_pipe_pct"]["low"]) == (60, 40)
    assert bc["vmem_pct_of_peak"]["saturated"] == t["confirm"]["hbm_bw_saturated_pct"]["value"] == 80
    assert t["fallback"]["balanced_band_pct"]["value"] == bc["sol_pipe_pct"]["balanced_band"]


def test_official_rdna_products_keep_distinct_identity_and_reference_status(sku, calc_perf):
    rows = sku["skus"]
    for name in ("R9700", "RX9070XT", "RX9070"):
        assert rows[name]["arch"] == "gfx1201"
    assert rows["R9700"]["memory_gb"] == 32
    assert rows["RX9070XT"]["memory_gb"] == 16
    assert rows["R9700"]["clock_mhz"]["boost"] == 2920
    assert rows["RX9070XT"]["clock_mhz"]["boost"] == 2970
    for name in ("RX9070XT", "RX9070"):
        assert rows[name]["geak_support"] == "unsupported"
        assert rows[name]["identity_target"] is None
        assert rows[name]["roofline_default_for_arch"] is False
        assert rows[name]["mall_mb"] == 64
    assert calc_perf.arch_default_sku("gfx1201") is None
    assert rows["RX9070XT"]["peak_tflops"]["int4"] == 779
    assert rows["RX9070"]["peak_tflops"]["int8"] == 289


def test_r9600d_conflicting_official_cache_sizes_remain_unknown(sku, extract_sku):
    import copy

    row = sku["skus"]["R9600D"]
    assert row["mall_mb"] is None
    conflict = row["source_conflicts"]["mall_mb"]
    assert conflict["status"] == "unresolved"
    assert {entry["value"] for entry in conflict["candidates"]} == {48, 64}
    assert all(entry["url"].startswith("https://") for entry in conflict["candidates"])
    mutated = copy.deepcopy(sku)
    mutated["skus"]["R9600D"]["mall_mb"] = 48
    assert any("unresolved" in issue for issue in extract_sku.check(mutated))


@pytest.mark.parametrize("name,dtype,peak", [
    ("MI355X", "fp6", 10100), ("MI355X", "mxfp6", 10100),
    ("MI355X", "mxfp4", 10100), ("MI350X", "mxfp6", 9200),
    ("MI300X", "tf32", 653.7), ("MI325X", "tf32", 653.7),
])
def test_published_precisions_work_through_both_cli_tools(name, dtype, peak):
    result = run(sys.executable, KT / "calc_perf.py", "roofline", "--sku", name,
                 "--dtype", dtype, "--M", 4096, "--N", 4096, "--K", 4096, "--json")
    assert result.returncode == 0, result.stderr
    estimate = json.loads(result.stdout)
    bw = json.loads((HW_DATA / "sku.json").read_text())["skus"][name]["peak_hbm_tb_s"]
    assert estimate["ridge"] == pytest.approx(peak / bw)
    assert estimate["may_gate"] is False
    result = run(sys.executable, KT / "hw_budget.py", "--sku", name, "--dtype", dtype,
                 "--workload", "gemm", "--shapes", "M=4096,N=4096,K=4096", "--json")
    assert result.returncode == 0, result.stderr
    w = json.loads(result.stdout)["workload"]
    assert w["compute_ceiling_tflops"] == [peak, peak]
    assert w["may_gate"] is False


@pytest.mark.parametrize("dtype,expected_mib", [
    ("bf16", 96), ("fp8", 64), ("mxfp4", 49), ("mxfp6", 57), ("fp64", 384),
])
def test_gemm_bytes_include_mx_scales_and_output_width(dtype, expected_mib):
    result = run(sys.executable, KT / "calc_perf.py", "roofline", "--sku", "MI355X",
                 "--dtype", dtype, "--M", 4096, "--N", 4096, "--K", 4096, "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["min_bytes"] == expected_mib * 2**20
    hb = _load("hw_budget_bytes", KT / "hw_budget.py")
    w = hb.budget("MI355X", "gemm", {"M": 4096, "N": 4096, "K": 4096}, dtype)["workload"]
    assert w["hbm_bytes_model"] == expected_mib * 2**20


def test_conflicting_sku_and_arch_is_refused(calc_perf):
    with pytest.raises(SystemExit, match="MI355X is gfx950"):
        calc_perf.resolve_sku("MI355X", arch="gfx942")


@pytest.mark.parametrize("name", ["MI300X", "R9700", "RX9070XT"])
def test_fp8_rate_does_not_imply_native_mxfp8_support(name, calc_perf, sku, extract_sku):
    with pytest.raises(SystemExit, match="no mxfp8 peak"):
        calc_perf.resolve_sku(name, dtype="mxfp8")
    assert extract_sku.ridge_for(sku["skus"][name], "mxfp8") is None
    hb = _load("hw_budget_mxfp8", KT / "hw_budget.py")
    w = hb.budget(name, "gemm", {"M": 4096, "N": 4096, "K": 4096}, "mxfp8")["workload"]
    assert w["compute_ceiling_tflops"] == [None, None]


@pytest.mark.parametrize("name,extra,compute,memory", [
    ("MI355X", [], "datasheet", "datasheet"),
    ("MI355X", ["--peak-hbm-bw", "4"], "datasheet", "hand-entered"),
    ("MI308X", [], "orc-derived", "orc-derived"),
    ("AI_MAX_395", [], "derived", "datasheet"),
])
def test_reference_sources_are_preserved_per_axis(name, extra, compute, memory):
    result = run(sys.executable, KT / "calc_perf.py", "roofline", "--sku", name,
                 "--M", 4096, "--N", 4096, "--K", 4096, "--json", *extra)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    assert row["compute_denominator_basis"] == compute
    assert row["memory_denominator_basis"] == memory
    assert row["denominator_basis"] == (compute if compute == memory else "mixed")
    assert row["may_gate"] is False


def test_cdna1_does_not_inherit_cdna2_combined_register_model():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'kernel_tools/amd_occupancy.py'
    spec = importlib.util.spec_from_file_location('occupancy_cdna1_boundary', path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    assert model.model_for('gfx908') is None
    assert model.waves_by_vgpr(128, 'gfx908')[0] is None
    assert model.waves_by_vgpr(128, 'gfx942')[0] == 4
