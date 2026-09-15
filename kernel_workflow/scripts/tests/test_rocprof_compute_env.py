# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gfx950 L2->HBM read-bytes preflight: detect the broken formula, absorb the upstream fix.

GPU-free: every case builds a synthetic rocprofiler-compute tree, because the property
under test is textual (which expression the analysis configs carry) and the failure mode
is silent -- a stock gfx950 install returns a plausible HBM number that is ~25% low.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import rocprof_compute_env as rce  # noqa: E402

# The stock gfx950 expression, in the two spellings upstream shipped.
BAD_A = (
    "      - metric: HBM Bandwidth\n"
    "        value: ((TCC_BUBBLE_sum * 128) + (TCC_EA0_RDREQ_32B_sum * 32)"
    " + ((TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64)) / 1e9\n"
)
BAD_B = (
    "      - metric: L2-Fabric Read BW\n"
    "        value: (128 * TCC_BUBBLE_sum + 64 * (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum"
    " - TCC_EA0_RDREQ_32B_sum) + 32 * TCC_EA0_RDREQ_32B_sum) / 1e9\n"
)
BAD_PY = (
    "def calc_ai(df, idx):\n"
    "    # Use TCC_BUBBLE_sum to calculate hbm_data\n"
    "    hbm_data = (\n"
    '                    (df["TCC_BUBBLE_sum"][idx] * 128)\n'
    '                    + (df["TCC_EA0_RDREQ_32B_sum"][idx] * 32)\n'
    '                    + ((df["TCC_EA0_RDREQ_sum"][idx]\n'
    '                        - df["TCC_BUBBLE_sum"][idx]\n'
    '                        - df["TCC_EA0_RDREQ_32B_sum"][idx]) * 64)\n'
    "    )\n"
    "    return hbm_data\n"
)


def _make_tree(root: Path, arch: str = "gfx950", buggy: bool = True, version="3.4.0",
               build="69f44ce4") -> Path:
    """A minimal install tree carrying (or not carrying) the broken formula."""
    configs = root / "rocprof_compute_soc" / "analysis_configs" / arch
    configs.mkdir(parents=True, exist_ok=True)
    (root / "utils").mkdir(parents=True, exist_ok=True)
    if buggy:
        roofline, sol, calc = BAD_A, BAD_B, BAD_PY
    else:
        roofline = BAD_A.replace(
            "(TCC_BUBBLE_sum * 128) + (TCC_EA0_RDREQ_32B_sum * 32)"
            " + ((TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) * 64)",
            rce._GOOD_YAML_A,
        )
        sol = BAD_B.replace(
            "128 * TCC_BUBBLE_sum + 64 * (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum"
            " - TCC_EA0_RDREQ_32B_sum) + 32 * TCC_EA0_RDREQ_32B_sum",
            rce._GOOD_YAML_B,
        )
        calc = BAD_PY.replace(
            "# Use TCC_BUBBLE_sum to calculate hbm_data", rce._GOOD_PY_COMMENT
        ).replace(
            '(df["TCC_BUBBLE_sum"][idx] * 128)\n'
            '                    + (df["TCC_EA0_RDREQ_32B_sum"][idx] * 32)\n'
            '                    + ((df["TCC_EA0_RDREQ_sum"][idx]\n'
            '                        - df["TCC_BUBBLE_sum"][idx]\n'
            '                        - df["TCC_EA0_RDREQ_32B_sum"][idx]) * 64)',
            rce._GOOD_PY,
        )
    (configs / "0400_roofline.yaml").write_text(roofline, encoding="utf-8")
    (configs / "0200_system_speed_of_light.yaml").write_text(sol, encoding="utf-8")
    (root / "utils" / "roofline_calc.py").write_text(calc, encoding="utf-8")
    (root / "VERSION").write_text(f"{version}\n{build}\n", encoding="utf-8")
    return root


def test_stock_gfx950_tree_is_reported_untrustworthy(tmp_path):
    """The whole point: a stock install looks fine and is not. Say so explicitly."""
    root = _make_tree(tmp_path / "rpc")
    report = rce.probe(arch="gfx950", root=root)

    assert report["hbm_byte_formula"] == "buggy"
    assert report["hbm_bytes_trustworthy"] is False
    # All three sites carry the formula, and all three are flagged.
    assert {s["status"] for s in report["sites"]} == {"buggy"}
    assert len(report["sites"]) == 3


def test_version_and_platform_are_reported(tmp_path):
    """Version comes from the VERSION file, not `--version` (see module docstring)."""
    root = _make_tree(tmp_path / "opt" / "rocm-7.2.0" / "libexec" / "rocprofiler-compute")
    report = rce.probe(arch="gfx950", root=root)

    assert report["tool_version"] == "3.4.0"
    assert report["tool_build"] == "69f44ce4"
    assert report["gpu_arch"] == "gfx950"
    # 3.4.0 predates the 3.5.0 release that carries the fix.
    assert report["version_predicts_fixed"] is False


def test_version_alone_does_not_decide(tmp_path):
    """A pre-3.5.0 tree that was patched in place must read as trustworthy."""
    root = _make_tree(tmp_path / "rpc", buggy=False, version="3.4.0")
    report = rce.probe(arch="gfx950", root=root)

    assert report["version_predicts_fixed"] is False
    assert report["hbm_byte_formula"] == "patched"
    assert report["hbm_bytes_trustworthy"] is True
    assert any("patched in place" in note for note in report["notes"])


def test_ensure_absorbs_the_upstream_fix(tmp_path):
    root = _make_tree(tmp_path / "rpc")
    report = rce.ensure(arch="gfx950", root=root, mode="auto")

    assert report["action"] == "patched"
    assert report["hbm_bytes_trustworthy"] is True
    configs = root / "rocprof_compute_soc" / "analysis_configs" / "gfx950"
    for name in ("0400_roofline.yaml", "0200_system_speed_of_light.yaml"):
        text = (configs / name).read_text(encoding="utf-8")
        assert "TCC_EA0_RDREQ_128B_sum" in text and "TCC_EA0_RDREQ_64B_sum" in text
        assert "TCC_BUBBLE_sum" not in text
        assert (configs / (name + ".orig")).exists(), "original must be recoverable"
    calc = (root / "utils" / "roofline_calc.py").read_text(encoding="utf-8")
    assert "TCC_EA0_RDREQ_128B_sum" in calc and "TCC_BUBBLE_sum" not in calc
    # The comment naming the old counter must go too, or it tells the next reader the
    # opposite of what the file now does.
    assert "TCC_BUBBLE" not in calc


def test_unrelated_tcc_bubble_uses_are_not_rewritten(tmp_path):
    """TCC_BUBBLE is a real counter with a real meaning.

    1800_l2_cache_per_channel.yaml reports it per channel as itself, not as a stand-in for
    a 128B read count -- rewriting it there would invent a bug while fixing one. Same for
    the WRITE term that sits inside the very expressions we do rewrite.
    """
    root = _make_tree(tmp_path / "rpc")
    configs = root / "rocprof_compute_soc" / "analysis_configs" / "gfx950"
    per_channel = configs / "1800_l2_cache_per_channel.yaml"
    per_channel.write_text("      - expr: (TO_INT(TCC_BUBBLE[::_1]) / $denom)\n", encoding="utf-8")
    # A write term alongside the broken read term, as upstream actually ships it.
    roofline = configs / "0400_roofline.yaml"
    roofline.write_text(
        roofline.read_text(encoding="utf-8").rstrip("\n")
        + " + ((TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32)\n",
        encoding="utf-8",
    )
    before = per_channel.read_text(encoding="utf-8")

    rce.ensure(arch="gfx950", root=root, mode="auto")

    assert per_channel.read_text(encoding="utf-8") == before
    patched = roofline.read_text(encoding="utf-8")
    assert "TCC_EA0_RDREQ_128B_sum" in patched, "the read term is corrected"
    assert "(TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32" in patched, "the write term is not"


def test_ensure_is_idempotent(tmp_path):
    root = _make_tree(tmp_path / "rpc")
    first = rce.ensure(arch="gfx950", root=root, mode="auto")
    after_first = (root / "utils" / "roofline_calc.py").read_text(encoding="utf-8")
    second = rce.ensure(arch="gfx950", root=root, mode="auto")

    assert first["action"] == "patched"
    assert second["action"] == "none"
    assert second["hbm_bytes_trustworthy"] is True
    assert (root / "utils" / "roofline_calc.py").read_text(encoding="utf-8") == after_first


def test_other_arches_are_left_alone(tmp_path):
    """gfx942's formula is correct upstream; rewriting it would invent a bug."""
    root = _make_tree(tmp_path / "rpc", arch="gfx942")
    before = (root / "rocprof_compute_soc" / "analysis_configs" / "gfx942"
              / "0400_roofline.yaml").read_text(encoding="utf-8")
    report = rce.ensure(arch="gfx942", root=root, mode="auto")

    assert report["hbm_byte_formula"] == "not_applicable"
    assert report["hbm_bytes_trustworthy"] is True
    assert report["sites"] == []
    after = (root / "rocprof_compute_soc" / "analysis_configs" / "gfx942"
             / "0400_roofline.yaml").read_text(encoding="utf-8")
    assert after == before


def test_mode_off_reports_without_touching_the_tree(tmp_path):
    root = _make_tree(tmp_path / "rpc")
    before = (root / "utils" / "roofline_calc.py").read_text(encoding="utf-8")
    report = rce.ensure(arch="gfx950", root=root, mode="off")

    assert report["action"] == "skipped"
    assert report["hbm_bytes_trustworthy"] is False
    assert (root / "utils" / "roofline_calc.py").read_text(encoding="utf-8") == before


def test_unwritable_tree_degrades_instead_of_raising(tmp_path, monkeypatch):
    """The install tree is usually root-owned and GEAK often is not root.

    The refusal is injected rather than expressed as a file mode: this suite runs both as
    root (inside the profiling images) and unprivileged (L0 runners), and root ignores the
    write bit -- a chmod-based version of this test silently passes for the wrong reason.
    """
    root = _make_tree(tmp_path / "rpc")

    def _refuse(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "write_text", _refuse)
    report = rce.ensure(arch="gfx950", root=root, mode="auto")

    assert report["action"] == "failed"
    assert report["hbm_bytes_trustworthy"] is False
    assert any("~25% low" in note for note in report["notes"])


def test_missing_install_degrades(tmp_path):
    report = rce.probe(arch="gfx950", root=tmp_path / "nope")

    assert report["hbm_byte_formula"] == "unknown"
    assert report["hbm_bytes_trustworthy"] is None


def test_report_shouts_when_numbers_are_known_wrong(tmp_path):
    root = _make_tree(tmp_path / "rpc")
    text = rce.format_report(rce.probe(arch="gfx950", root=root))

    assert "UNDER-REPORTED" in text
    assert "aa5dfb9" in text, "the report must carry its provenance"


def test_cli_exit_code_signals_untrustworthy(tmp_path, capsys):
    root = _make_tree(tmp_path / "rpc")
    out_json = tmp_path / "env.json"

    rc = rce.main(["--check", "--arch", "gfx950", "--root", str(root),
                   "--json", str(out_json)])

    assert rc == 1
    assert out_json.exists()
    assert "hbm_byte_formula: buggy" in capsys.readouterr().out
