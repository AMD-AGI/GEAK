"""Regression tests for roofline_task.py case selection + manifest/driver generation.

These lock in the fix for the e2e-layer roofline probe silently skipping every
non-`sig`-keyed task (the extracted unittest bundles key workload cases by `name`, and
attention heads are not GEMM-shaped). Before the fix, `_select_cases` required a GEMM
`sig` key so it returned [] for both the fp8 GEMM head and the prefill-attention head,
leaving their top-level <task>/roofline/ dirs empty; the auto-generated driver also
targeted a dead unittest API (_wl_cases / _case_dims / synth_case / CURRENT_CALL).

The module is imported by file path so the test is location-agnostic (no assumption
about where the GEAK checkout lives).
"""
import importlib.util
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.normpath(os.path.join(HERE, "..", "scripts", "roofline_task.py"))
_spec = importlib.util.spec_from_file_location("roofline_task_under_test", _MODULE_PATH)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


# A GEMM head: cases keyed by `name`, GEMM-shaped (scalar m/N/K), NO `sig`, NO `weight`.
GEMM_META = {
    "short_name": "fp8_gemm",
    "op_kind": "gemm",
    "workload": {
        "name_match": "_gemm_a8w8_blockscale_kernel",
        "cases": [
            {"name": "down_proj_decode_M1", "regime": "decode", "m": 1, "N": 5120,
             "K": 17408, "dtypes": ["fp8_e4m3fnuz", "fp8_e4m3fnuz"]},
            {"name": "down_proj_prefill_M1024", "regime": "prefill", "m": 1024, "N": 5120,
             "K": 17408, "dtypes": ["fp8_e4m3fnuz", "fp8_e4m3fnuz"]},
            {"name": "down_proj_decode_M64", "regime": "decode", "m": 64, "N": 5120,
             "K": 17408, "dtypes": ["fp8_e4m3fnuz", "fp8_e4m3fnuz"]},
        ],
    },
}

# An attention head: cases keyed by `name`, NON-GEMM (seqlen, 3-D dims, no scalar N/K).
ATTN_META = {
    "short_name": "prefill_attn",
    "op_kind": "attention",
    "workload": {
        "name_match": "FmhaBatchPrefill",
        "cases": [
            {"name": "prefill_s512", "regime": "prefill", "m": 512, "seqlen_q": 512,
             "seqlen_kv": 512, "weight": 7510.9, "dtypes": ["bf16", "bf16", "bf16"],
             "dims": [[512, 40, 128], [512, 8, 128], [512, 8, 128]]},
            {"name": "prefill_s1024", "regime": "prefill", "m": 1024, "seqlen_q": 1024,
             "seqlen_kv": 1024, "weight": 3200.0, "dtypes": ["bf16", "bf16", "bf16"],
             "dims": [[1024, 40, 128], [1024, 8, 128], [1024, 8, 128]]},
        ],
    },
}


def test_name_keyed_gemm_cases_selected():
    """GEMM cases keyed by name (no `sig`) must be selected, not skipped."""
    picked = rt._select_cases(GEMM_META, max_cases=3)
    ids = [rt._case_id(c) for c in picked]
    assert ids  # not empty -- the pre-fix bug returned []
    # one representative per regime first (decode + prefill both covered)
    regimes = {c.get("regime") for c in picked}
    assert regimes == {"decode", "prefill"}
    assert "down_proj_prefill_M1024" in ids


def test_non_gemm_attention_cases_selected():
    """Attention cases (no scalar N/K, no `sig`) must be selected -- highest weight first."""
    picked = rt._select_cases(ATTN_META, max_cases=3)
    ids = [rt._case_id(c) for c in picked]
    assert ids == ["prefill_s512", "prefill_s1024"]  # weight-desc, single regime


def test_legacy_sig_still_honored():
    """Older extracted bundles that key by `sig` (no `name`) still select."""
    meta = {"workload": {"cases": [
        {"sig": "M1", "regime": "decode", "weight": 1.0},
        {"sig": "M2", "regime": "prefill", "weight": 2.0},
    ]}}
    ids = [rt._case_id(c) for c in rt._select_cases(meta, max_cases=2)]
    assert set(ids) == {"M1", "M2"}


def test_empty_when_no_identifier():
    meta = {"workload": {"cases": [{"regime": "decode", "m": 1}]}}  # no name and no sig
    assert rt._select_cases(meta, max_cases=3) == []


def test_case_shape_gemm_only():
    """_case_shape emits M/N/K for GEMM cases and None for attention cases."""
    assert rt._case_shape(GEMM_META["workload"]["cases"][0]) == {"M": 1, "N": 5120, "K": 17408}
    assert rt._case_shape(ATTN_META["workload"]["cases"][0]) is None


def test_manifest_uses_name_as_case_id_and_sig_arg():
    """Manifest case_id == name, the driver command passes --sig <name>, and shape is
    present for GEMM but omitted for attention."""
    gm = rt._build_manifest(GEMM_META, "/task/gemm", "/task/gemm/roofline/driver.py",
                            "/task/gemm/unittest.py", GEMM_META["workload"]["cases"][:1],
                            gpu_id="0", patterns=["_gemm_a8w8_blockscale_kernel"])
    case = gm["cases"][0]
    assert case["case_id"] == "down_proj_decode_M1"
    assert case["command"][-2:] == ["--sig", "down_proj_decode_M1"]
    assert case["shape"] == {"M": 1, "N": 5120, "K": 17408}
    assert case["dtypes"] == ["fp8_e4m3fnuz", "fp8_e4m3fnuz"]

    am = rt._build_manifest(ATTN_META, "/task/attn", "/task/attn/roofline/driver.py",
                            "/task/attn/unittest.py", ATTN_META["workload"]["cases"][:1],
                            gpu_id="0", patterns=["FmhaBatchPrefill"])
    acase = am["cases"][0]
    assert acase["case_id"] == "prefill_s512"
    assert "shape" not in acase  # attention: no misleading GEMM shape recorded
    assert acase["command"][-1] == "prefill_s512"


def test_manifest_carries_exclude_patterns():
    """_build_manifest threads exclude_patterns into target + every case (empty by default)."""
    gm = rt._build_manifest(GEMM_META, "/task/gemm", "/task/gemm/roofline/driver.py",
                            "/task/gemm/unittest.py", GEMM_META["workload"]["cases"][:1],
                            gpu_id="0", patterns=[], exclude_patterns=[r"_gemm_a8w8_blockscale_kernel"])
    assert gm["target"]["exclude_patterns"] == [r"_gemm_a8w8_blockscale_kernel"]
    assert gm["cases"][0]["exclude_patterns"] == [r"_gemm_a8w8_blockscale_kernel"]
    # default (omitted) => empty list, never absent (collector reads target/case uniformly)
    dm = rt._build_manifest(GEMM_META, "/t", "/d.py", "/u.py",
                            GEMM_META["workload"]["cases"][:1], gpu_id="0", patterns=[])
    assert dm["target"]["exclude_patterns"] == []
    assert dm["cases"][0]["exclude_patterns"] == []


def test_baseline_exclude_patterns_from_report(tmp_path):
    """The baseline's matched kernel symbol becomes a literal-regex exclude; rocprof's
    trailing '...' truncation is stripped; short/blank names are ignored."""
    report = tmp_path / "baseline_roofline.json"
    report.write_text(json.dumps({"cases": [
        {"case_id": "a", "matched_kernel_name": "_gemm_a8w8_blockscale_kernel_GROUP_K_128..."},
        {"case_id": "b", "matched_kernel_name": "  "},          # blank -> ignored
        {"case_id": "c", "matched_kernel_name": "tiny"},         # < 8 chars -> ignored
    ]}))
    pats = rt._baseline_exclude_patterns(str(report))
    assert pats == [re.escape("_gemm_a8w8_blockscale_kernel_GROUP_K_128")]
    # the escaped pattern matches the FULL (untruncated) device symbol as a substring
    full = "void _gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_128(...)"
    assert re.search(pats[0], full)
    # missing/unreadable report -> [] (fail-soft, never raises)
    assert rt._baseline_exclude_patterns(str(tmp_path / "nope.json")) == []


def test_generated_driver_uses_build_correctness_seam():
    """The auto-generated driver must use the live _build_correctness seam, not the dead
    _wl_cases/_case_dims/synth_case/CURRENT_CALL API, and must be valid Python."""
    import py_compile
    import tempfile
    src = rt.DRIVER_SOURCE
    assert "_build_correctness" in src
    for dead in ("_wl_cases", "_case_dims", "synth_case", "CURRENT_CALL"):
        assert dead not in src, "driver still references dead API: %s" % dead
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        py_compile.compile(path, doraise=True)
    finally:
        os.unlink(path)
