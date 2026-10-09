# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for GEAK's single roofline data source and the roofline kernel tools.

perf_knowledge/hardware/data/sku.json is the ONE per-SKU peak table. Its consumers --
kernel_workflow/scripts/kernel_tools/{hw_budget,calc_perf,mem_bw_probe,extract_sku}.py (moved out of
the Gluon pack, which keeps same-named shims under its scripts/) and the e2e roofline analysis skill
(e2e_workflow/knowledge/analysis_skills/roofline/roofline_tools.py) -- must read it, agree with it on
every overlapping field, refuse a dtype it does not list, and keep the per-arch dtype ratios fixed.
Every markdown copy of its numbers is a generated block that must match a fresh render. No GPU, no
network, no torch.
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
PACK = REPO / "perf_knowledge" / "expert_skills" / "skills" / "gluon_authoring"
PACK_SCRIPTS = PACK / "scripts"
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


@pytest.fixture(scope="module")
def rt():
    return _load("roofline_tools_under_test", ROOFLINE_DIR / "roofline_tools.py")


# --------------------------------------------------------------------------- selftests, KT + shim

@pytest.mark.parametrize("where", ["kt", "shim"])
@pytest.mark.parametrize("tool", TOOLS)
def test_roofline_tool_selftest(tool, where):
    path = (KT if where == "kt" else PACK_SCRIPTS) / f"{tool}.py"
    result = run(sys.executable, path, "--selftest")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out[-3000:]
    assert "SELFTEST PASS" in out, out[-3000:]


@pytest.mark.parametrize("where", ["kt", "shim"])
def test_hw_budget_reading_route_page_check_actually_runs(where):
    """workload_models.json reading_route.page is pack-relative and must resolve via
    _hwdata.pack_dir(); the 13-archetype / precheck-heading assertions must RUN, not skip."""
    path = (KT if where == "kt" else PACK_SCRIPTS) / "hw_budget.py"
    out = run(sys.executable, path, "--selftest").stdout
    assert "reading-route page checked" in out and "13 archetype sections" in out, out[-2000:]
    assert "SKIPPED" not in out


def test_reading_route_page_resolves_via_pack_dir():
    sys.path.insert(0, str(KT))
    try:
        import _hwdata  # noqa: PLC0415
    finally:
        sys.path.remove(str(KT))
    page = json.loads((HW_DATA / "workload_models.json").read_text())["reading_route"]["page"]
    assert (_hwdata.pack_dir() / page).is_file(), page


def test_shims_are_thin_and_point_at_kt():
    for tool in TOOLS:
        text = (PACK_SCRIPTS / f"{tool}.py").read_text()
        assert f"kernel_workflow/scripts/kernel_tools/{tool}.py" in text, tool
        assert len(text.splitlines()) < 30, f"{tool} shim carries logic"


def test_shim_import_is_the_kt_module():
    """`import calc_perf` / `import hw_budget` from the pack's scripts/ binds the KT module."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import calc_perf, hw_budget, extract_sku, "
            "mem_bw_probe; print(calc_perf.__file__); print(hw_budget.__file__); "
            "print(len(calc_perf.SKU_PEAKS))")
    r = run(sys.executable, "-c", code, PACK_SCRIPTS)
    assert r.returncode == 0, r.stderr[-2000:]
    files = r.stdout.splitlines()
    assert Path(files[0]).resolve() == (KT / "calc_perf.py").resolve()
    assert Path(files[1]).resolve() == (KT / "hw_budget.py").resolve()
    assert int(files[2]) >= 13


# --------------------------------------------------------------------------- one source, agreeing

def test_calc_perf_sku_peaks_are_read_from_sku_json(calc_perf, sku):
    assert list(calc_perf.SKU_PEAKS) == list(sku["skus"])
    for name, row in sku["skus"].items():
        assert calc_perf.SKU_PEAKS[name] == row, name
    # no hard-coded duplicate table left in the source
    src = (KT / "calc_perf.py").read_text()
    assert '"peak_tflops": {"fp16": 1307.4' not in src and "GFX950 = {" not in src


def test_sku_json_roofline_tools_and_calc_perf_agree(calc_perf, rt, sku):
    """Overlapping fields: CU count, every dtype peak, and the memory roof -- the e2e roofline
    ranks on the datasheet pin rate, hw_budget/calc_perf use the row's stored ceiling."""
    n = 0
    for name, row in sku["skus"].items():
        if row["geak_support"] != "supported":
            continue
        for product in ([row["identity_target"]] if row.get("identity_target") else []) + (
                [None] if row.get("roofline_default_for_arch") else []):
            p = rt.load_peaks(rt.SKU_JSON, row["arch"], product=product)
            assert p["sku"] == name and p["cu"] == row["cus"] == calc_perf.SKU_PEAKS[name]["cus"]
            for dt, tf in row["peak_tflops"].items():
                assert p["flops"][dt] == pytest.approx(tf * 1e12)
                label, cfg = calc_perf.resolve_sku(name, dtype=dt)
                assert cfg["peak_tflops_fp16"] == tf, (name, dt)
            pin = row.get("datasheet_hbm_tb_s") or row["peak_hbm_tb_s"]
            assert p["hbm_bw_bytes_s"] == pytest.approx(pin * 1e12)
            assert calc_perf.resolve_sku(name)[1]["peak_hbm_tb_s"] == row["peak_hbm_tb_s"]
            n += 1
    assert n >= 8


def test_hw_budget_reads_the_same_rows():
    hb = _load("hw_budget_under_test", KT / "hw_budget.py")
    for name, dtype, want in (("MI355X", "fp4", 10000.0), ("MI350X", "bf16", 2300.0),
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
        "MI355X": dict(fp16=2500.0, bf16=2500.0, fp8=5000.0, fp4=10000.0, fp32=157.3, fp64=78.6),
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


def test_roofline_tools_missing_dtype_is_unknown_not_zero(rt):
    p = rt.resolve_peaks(rt.SKU_JSON, "gfx1201", product="r9700")
    m = rt.roofline_metrics(1e8, 1e11, 1e-3, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "fp4"), 0.9)
    assert m["compute_util"] is None and m["bound_type"] == "unknown"


# --------------------------------------------------------------------------- derived docs

def test_generated_doc_blocks_are_current(extract_sku):
    files = list(extract_sku.doc_files())
    rel = {str(Path(f).relative_to(REPO)) for f in files}
    for must in ("e2e_workflow/knowledge/analysis_skills/roofline/peaks.md",
                 "kernel_workflow/knowledge/amd_instinct.md",
                 "perf_knowledge/hardware/cdna4_mi350/peak_tables.md",
                 "perf_knowledge/hardware/cdna3_mi300/peak_tables.md",
                 "perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/amd-cdna4-skus.md",
                 "perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/amd-cdna3-skus.md",
                 "perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/atlas.md"):
        assert must in rel, must
    assert extract_sku.check_docs(extract_sku.load()) == []


def test_sku_json_invariants_clean(extract_sku):
    assert extract_sku.check(extract_sku.load()) == []


def test_thresholds_one_bound_classification_table():
    t = json.loads((HW_DATA / "thresholds.json").read_text())
    bc = t["bound_classification"]
    assert bc["roof_util"]["bound_min"] == 0.60
    assert (bc["sol_pipe_pct"]["high"], bc["sol_pipe_pct"]["low"]) == (60, 40)
    assert bc["vmem_pct_of_peak"]["saturated"] == t["confirm"]["hbm_bw_saturated_pct"]["value"] == 80
    assert t["fallback"]["balanced_band_pct"]["value"] == bc["sol_pipe_pct"]["balanced_band"]
