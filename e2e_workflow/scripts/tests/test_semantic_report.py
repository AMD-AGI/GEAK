"""Phase 1's report is a VIEW of the table -- these tests pin what it must not hide.

The failure it was written for: a table where decode resolved 0 of 64 shapes reached
apply-back without anyone noticing, because Phase 1 emitted only JSON. So the report
has to be loud about a blind phase, honest when the table's own declared record has
gone stale, and it has to show the fusible surface using the SAME definition Phase 2.1
gates against -- a report that draws a different surface than the gate is worse than
no report.
"""
import json
import os
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

import semantic_report as sr  # noqa: E402
import fusion_candidate_harness as harness  # noqa: E402


def _row(row_id, stage, us, dims=None, stream=1, name=None):
    row = {"row_id": row_id, "stage": stage, "duration_us": us, "stream": stream,
           "short_name": name or ("%s_kernel" % stage), "provider": "aiter"}
    if dims:
        row["shape"] = {"input_dims": dims, "input_types": ["bf16"]}
    return row


def _table(phase="decode", rows=None, declared=None, pattern="P0", layers=3):
    item = {"phase": phase, "pattern_id": pattern,
            "pattern_display_name": "%s layer" % pattern,
            "pattern_layer_count": layers, "representative_layer_id": 2,
            "rows": rows or []}
    item["layer_total_us"] = sum(r["duration_us"] for r in item["rows"])
    table = {"tables": [item], "trace_path": "/t/trace.json",
             "trace_sha256_by_path": {"/t/trace.json": "abc123"}}
    if declared is not None:
        table["phase_coverage"] = {"shape_resolution_by_phase": declared}
    return table


class SemanticReportTest(unittest.TestCase):
    def _build(self, table, floor=5.0):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "table.json")
            with open(path, "w") as fh:
                json.dump(table, fh)
            rep = sr.build(path, helper_floor=floor)
            return rep, sr.render_markdown(rep)

    # ---- coverage measured off the rows -------------------------------------

    def test_shape_coverage_is_measured_off_the_rows(self):
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]]),
                _row("b", "quant", 10.0),
                _row("c", "gemm", 10.0, dims=[[8, 16]])]
        rep, _ = self._build(_table(rows=rows))
        self.assertEqual(rep["phase_coverage_measured"]["decode"],
                         {"rows": 3, "resolved": 2, "resolved_fraction": 0.6667,
                          "patterns": 1, "layer_us": 30.0})

    def test_a_stale_declared_record_is_reported_not_trusted(self):
        # A later shape graft can populate rows and leave phase_coverage behind.
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]])]
        rep, md = self._build(_table(rows=rows,
                                     declared={"decode": {"rows": 1, "resolved": 0}}))
        self.assertEqual(rep["phase_coverage_measured"]["decode"]["resolved"], 1)
        self.assertEqual(len(rep["stale_record_notes"]), 1)
        self.assertIn("record says 0/1", rep["stale_record_notes"][0])
        self.assertIn("⚠️", md)

    def test_a_phase_with_rows_but_no_shapes_is_called_out_in_red(self):
        rows = [_row("a", "norm", 10.0), _row("b", "quant", 10.0)]
        _, md = self._build(_table(rows=rows))
        self.assertIn("🔴", md)
        self.assertIn("decode", md)

    def test_a_fully_shaped_two_phase_table_gets_no_red_callout(self):
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]]),
                _row("b", "quant", 10.0, dims=[[8, 16]])]
        table = _table(rows=rows)
        table["tables"].append(dict(table["tables"][0], phase="prefill"))
        _, md = self._build(table)
        self.assertNotIn("🔴", md)

    def test_prefill_is_rendered_before_decode(self):
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]])]
        table = _table(phase="decode", rows=rows)
        table["tables"].append(dict(table["tables"][0], phase="prefill"))
        _, md = self._build(table)
        self.assertLess(md.index("| prefill |"), md.index("| decode |"))
        self.assertLess(md.index("### prefill /"), md.index("### decode /"))

    def test_an_absent_phase_is_named_as_absent(self):
        # The DSR1 failure: a prefill-only table with no declared record, which read
        # as a complete run because nothing mentioned decode at all.
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]])]
        rep, md = self._build(_table(phase="prefill", rows=rows))
        self.assertEqual(rep["phases_absent"], ["decode"])
        self.assertIn("🔴", md)
        self.assertIn("decode", md)

    def test_a_declared_absent_phase_is_kept_even_if_unexpected(self):
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]]),
                _row("b", "quant", 10.0, dims=[[8, 16]])]
        table = _table(rows=rows)
        table["tables"].append(dict(table["tables"][0], phase="prefill"))
        table["phase_coverage"] = {"phases_absent_from_tables": ["mtp"]}
        rep, md = self._build(table)
        self.assertEqual(rep["phases_absent"], ["mtp"])
        self.assertIn("mtp", md)

    # ---- the fusible surface ------------------------------------------------

    def test_the_region_definition_is_the_one_phase_2_1_gates_against(self):
        # Not "equivalent" -- the same function. A report drawing a different
        # surface than the gate would send the next phase looking for the wrong
        # thing while both files look self-consistent.
        self.assertIs(sr._fusible_regions, harness._fusible_regions)
        self.assertIs(sr.DONOR_STAGES, harness.DONOR_STAGES)

    def test_regions_stop_at_a_donor(self):
        rows = [_row("a", "norm", 20.0), _row("b", "quant", 20.0),
                _row("g", "gemm", 500.0),
                _row("c", "elementwise", 20.0), _row("d", "quant", 20.0)]
        rep, _ = self._build(_table(rows=rows))
        self.assertEqual(rep["fusible_region_count"], 2)
        for region in rep["fusible_regions"]:
            self.assertNotIn("g", region["row_ids"])

    def test_regions_are_ranked_by_cost(self):
        rows = [_row("a", "norm", 10.0), _row("b", "quant", 10.0),
                _row("g", "gemm", 500.0),
                _row("c", "elementwise", 90.0), _row("d", "quant", 90.0)]
        rep, _ = self._build(_table(rows=rows))
        totals = [r["total_us"] for r in rep["fusible_regions"]]
        self.assertEqual(totals, sorted(totals, reverse=True))
        self.assertEqual(rep["fusible_regions"][0]["row_ids"], ["c", "d"])

    def test_a_donor_only_layer_has_no_fusible_surface(self):
        rows = [_row("g1", "gemm", 500.0), _row("g2", "attn", 400.0)]
        rep, md = self._build(_table(rows=rows))
        self.assertEqual(rep["fusible_region_count"], 0)
        self.assertEqual(rep["fusible_region_us"], 0.0)
        self.assertTrue(md.strip())

    def test_the_helper_floor_is_honoured(self):
        rows = [_row("a", "norm", 0.4), _row("b", "quant", 0.4),
                _row("g", "gemm", 500.0),
                _row("c", "elementwise", 50.0), _row("d", "quant", 50.0)]
        rep, _ = self._build(_table(rows=rows), floor=5.0)
        self.assertEqual(rep["fusible_region_count"], 1)
        self.assertEqual(rep["fusible_regions"][0]["row_ids"], ["c", "d"])

    # ---- cost accounting ----------------------------------------------------

    def test_non_donor_share_is_reported_per_pattern(self):
        rows = [_row("a", "norm", 25.0), _row("g", "gemm", 75.0)]
        rep, _ = self._build(_table(rows=rows))
        pat = rep["patterns"][0]
        self.assertEqual(pat["donor_us"], 75.0)
        self.assertEqual(pat["non_donor_us"], 25.0)
        self.assertEqual(pat["non_donor_pct"], 25.0)

    def test_phase_forward_multiplies_the_layer_by_its_layer_count(self):
        rows = [_row("a", "norm", 10.0), _row("g", "gemm", 90.0)]
        rep, _ = self._build(_table(rows=rows, layers=58))
        self.assertEqual(rep["phase_forward_us"]["decode"], 5800.0)

    def test_the_stage_mix_accounts_for_every_row(self):
        rows = [_row("a", "norm", 10.0), _row("b", "norm", 5.0),
                _row("g", "gemm", 85.0)]
        rep, _ = self._build(_table(rows=rows))
        mix = rep["patterns"][0]["stage_mix"]
        self.assertEqual(mix["norm"], {"count": 2, "us": 15.0})
        self.assertEqual(sum(v["count"] for v in mix.values()), 3)

    # ---- provenance ---------------------------------------------------------

    def test_the_trace_and_its_sha_reach_the_report(self):
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]])]
        rep, md = self._build(_table(rows=rows))
        self.assertEqual(rep["trace_paths"], ["/t/trace.json"])
        self.assertIn("abc123", md)

    def test_unique_operator_schema_names_are_visible_beside_raw_dims(self):
        row = _row("q", "quant", 6.0, dims=[[4, 8], [4, 1]])
        row["shape"]["input_types"] = ["c10::Float8_e4m3fnuz", "float"]
        row["shape"]["kernel_shape"] = {"operands": [
            {"arg_index": 0, "schema_name": "out", "direction": "output",
             "shape": [4, 8], "dtype": "c10::Float8_e4m3fnuz"},
            {"arg_index": 1, "schema_name": "scales",
             "direction": "mutable_unresolved", "shape": [4, 1],
             "dtype": "float"},
        ]}
        rep, md = self._build(_table(rows=[row]))
        detail = rep["patterns"][0]["row_detail"][0]
        self.assertEqual(detail["named_operands"][0]["schema_name"], "out")
        self.assertIn("out=Float8_e4m3fnuz[4×8]", md)
        self.assertIn("scales{mutable_unresolved}=float[4×1]", md)

    def test_two_traces_both_appear(self):
        # The split prefill/decode capture is exactly the case that went unnoticed.
        rows = [_row("a", "norm", 10.0, dims=[[8, 16]])]
        table = _table(rows=rows)
        del table["trace_path"]
        table["trace_paths"] = ["/t/extend.json", "/t/decode.json"]
        table["trace_sha256_by_path"] = {"/t/extend.json": "aaa",
                                         "/t/decode.json": "bbb"}
        rep, md = self._build(table)
        self.assertEqual(len(rep["trace_paths"]), 2)
        self.assertIn("aaa", md)
        self.assertIn("bbb", md)


if __name__ == "__main__":
    unittest.main()
