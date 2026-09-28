"""The index must describe the disk, not the plan.

The failure it guards against: a run where a phase produced nothing, and the index
simply had no row for it -- so the reader saw four tidy phases and no sign that the
fifth was missing. Absence has to be rendered, not omitted.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import report_index  # noqa: E402


class ReportIndexTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="report_index_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _write(self, rel, text):
        path = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def _json(self, rel, obj):
        return self._write(rel, json.dumps(obj))

    def _all_five(self):
        for name, _, _, _ in report_index.REPORTS:
            self._write(name, "# %s\n" % name)

    # ---- presence / absence -------------------------------------------------

    def test_a_missing_report_is_listed_as_missing_not_dropped(self):
        self._write("01_SEMANTIC.md", "# semantic\n")
        index = report_index.run(self.dir)
        md = open(index["index_path"]).read()
        self.assertEqual(len(index["reports"]), 5)
        self.assertIn("05_FUSION_APPLYBACK.md", md)
        self.assertIn("未生成", md)
        self.assertEqual(len(index["missing"]), 4)

    def test_a_complete_run_has_no_missing_warning(self):
        self._all_five()
        index = report_index.run(self.dir)
        self.assertEqual(index["missing"], [])
        self.assertNotIn("未生成", open(index["index_path"]).read())

    def test_every_present_report_is_linked_relatively(self):
        self._all_five()
        index = report_index.run(self.dir)
        md = open(index["index_path"]).read()
        for name, _, _, _ in report_index.REPORTS:
            self.assertIn("(./%s)" % name, md)

    # ---- sidecar resolution -------------------------------------------------

    def test_a_sidecar_is_found_in_its_working_dir_not_only_at_root(self):
        self._write("03_FUSION_TOPK.md", "# topk\n")
        self._json("round1/fusion/fusion_topk.json",
                   {"candidate_total": 42, "candidates_on_board": 14,
                    "execution_list": [{"exec_id": "e01"}], "truncated_count": 2})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"] if r["report"] == "03_FUSION_TOPK.md"][0]
        self.assertTrue(row["sidecar"].endswith("round1/fusion/fusion_topk.json"))
        self.assertIn("42", row["summary"])
        self.assertIn("截断 2", row["summary"])

    def test_either_candidate_json_spelling_resolves(self):
        self._write("02_FUSION_CANDIDATES.md", "# c\n")
        self._json("wd/fusion_candidate_validation.json",
                   {"status": "pass", "candidate_count": 7})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"]
               if r["report"] == "02_FUSION_CANDIDATES.md"][0]
        self.assertTrue(row["sidecar"].endswith("fusion_candidate_validation.json"))
        self.assertIn("候选 7 条", row["summary"])

    def test_the_candidate_count_is_read_from_metrics_too(self):
        # The harness reports it under metrics, not at the top level; the index has
        # to find it there or every candidate row reads "候选 ? 条".
        self._write("02_FUSION_CANDIDATES.md", "# c\n")
        self._json("wd/fusion_candidate_result.json",
                   {"status": "fail", "metrics": {"candidate_count": 42}})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"]
               if r["report"] == "02_FUSION_CANDIDATES.md"][0]
        self.assertIn("候选 42 条", row["summary"])

    def test_a_root_sidecar_wins_over_a_nested_one(self):
        self._write("03_FUSION_TOPK.md", "# topk\n")
        self._json("round1/fusion/fusion_topk.json", {"candidate_total": 1})
        self._json("fusion_topk.json", {"candidate_total": 99})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"] if r["report"] == "03_FUSION_TOPK.md"][0]
        self.assertIn("99", row["summary"])

    def test_a_report_with_no_sidecar_still_appears(self):
        self._write("04_FUSION_UNITSIDE.md", "# u\n")
        index = report_index.run(self.dir)
        row = [r for r in index["reports"]
               if r["report"] == "04_FUSION_UNITSIDE.md"][0]
        self.assertTrue(row["present"])
        self.assertEqual(row["sidecar"], "")

    def test_unreadable_json_does_not_sink_the_index(self):
        self._write("03_FUSION_TOPK.md", "# t\n")
        self._write("fusion_topk.json", "{ not json")
        index = report_index.run(self.dir)   # must not raise
        row = [r for r in index["reports"] if r["report"] == "03_FUSION_TOPK.md"][0]
        self.assertTrue(row["present"])
        self.assertEqual(row["summary"], "")

    # ---- the summaries a reader scans --------------------------------------

    def test_a_phase_with_no_resolved_shapes_is_flagged_red(self):
        self._write("01_SEMANTIC.md", "# s\n")
        self._json("semantic_report.json", {
            "phase_coverage_measured": {"prefill": {"rows": 62, "resolved": 49},
                                        "decode": {"rows": 64, "resolved": 0}},
            "fusible_region_count": 22})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"] if r["report"] == "01_SEMANTIC.md"][0]
        self.assertIn("🔴", row["flag"])
        self.assertIn("decode", row["flag"])

    def test_a_fully_shaped_semantic_phase_is_not_flagged(self):
        self._write("01_SEMANTIC.md", "# s\n")
        self._json("semantic_report.json", {
            "phase_coverage_measured": {"prefill": {"rows": 62, "resolved": 49},
                                        "decode": {"rows": 64, "resolved": 52}},
            "fusible_region_count": 22})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"] if r["report"] == "01_SEMANTIC.md"][0]
        self.assertNotIn("🔴", row["flag"])
        self.assertIn("可融合区间 22", row["summary"])

    def test_an_unaccounted_applyback_row_reaches_the_index(self):
        self._write("05_FUSION_APPLYBACK.md", "# a\n")
        self._json("fusion_applyback.json", {
            "status": "fail",
            "coverage": {"execution_list_size": 12, "accounted": 10, "unaccounted": 2},
            "counts": {"applied": 2},
            "e2e": {"throughput_tok_s": 1234.5}})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"]
               if r["report"] == "05_FUSION_APPLYBACK.md"][0]
        self.assertIn("🔴", row["flag"])
        self.assertIn("无交代 2", row["summary"])
        self.assertIn("1234.5", row["summary"])

    def test_the_unitside_denominator_is_on_the_index_line(self):
        self._write("04_FUSION_UNITSIDE.md", "# u\n")
        self._json("fusion_unitside.json", {
            "status": "fail",
            "coverage": {"in_scope": 40, "validated": 16, "not_validated": 24,
                         "waived": 0},
            "counts": {"pass": 16, "fail": 0, "blocked": 0}})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"]
               if r["report"] == "04_FUSION_UNITSIDE.md"][0]
        self.assertIn("未验证 24", row["summary"])
        self.assertIn("🔴", row["flag"])

    def test_region_coverage_is_carried_from_the_candidate_phase(self):
        self._write("02_FUSION_CANDIDATES.md", "# c\n")
        self._json("fusion_candidate_result.json", {
            "status": "pass", "candidate_count": 42,
            "region_coverage": {"regions_total": 22, "covered": 20, "deferred": 0,
                                "uncovered": 2}})
        index = report_index.run(self.dir)
        row = [r for r in index["reports"]
               if r["report"] == "02_FUSION_CANDIDATES.md"][0]
        self.assertIn("未覆盖 2", row["summary"])

    # ---- shape of the output ------------------------------------------------

    def test_the_index_is_written_where_the_reports_are(self):
        self._all_five()
        index = report_index.run(self.dir)
        self.assertEqual(index["index_path"],
                         os.path.join(self.dir, "00_INDEX.md"))
        self.assertTrue(os.path.isfile(index["index_path"]))

    def test_rerunning_overwrites_rather_than_appends(self):
        self._all_five()
        report_index.run(self.dir)
        first = open(os.path.join(self.dir, "00_INDEX.md")).read()
        report_index.run(self.dir)
        self.assertEqual(first, open(os.path.join(self.dir, "00_INDEX.md")).read())

    def test_the_index_sorts_above_the_reports_it_indexes(self):
        names = [n for n, _, _, _ in report_index.REPORTS]
        self.assertEqual(sorted([report_index.INDEX_NAME] + names),
                         [report_index.INDEX_NAME] + names)


    def test_prior_coverage_reaches_the_index_line(self):
        # A prior nobody disposed of is invisible everywhere else on this page:
        # it produced no candidate, no row, and no symptom a reader would spot.
        self._write("02_FUSION_CANDIDATES.md", "# candidates\n")
        self._json("round1/fusion/fusion_candidate_result.json", {
            "status": "fail",
            "metrics": {"candidate_count": 42},
            "prior_coverage": {"priors_total": 2, "carried": 1,
                               "already_engaged": 0, "not_applicable": 0,
                               "undisposed": 1},
        })
        index = report_index.run(self.dir)
        md = open(index["index_path"]).read()
        self.assertIn("已知先验 2", md)
        self.assertIn("无交代 1", md)

    def test_a_run_with_no_kb_shows_no_prior_line(self):
        self._write("02_FUSION_CANDIDATES.md", "# candidates\n")
        self._json("round1/fusion/fusion_candidate_result.json", {
            "status": "pass", "metrics": {"candidate_count": 7}})
        index = report_index.run(self.dir)
        self.assertNotIn("已知先验", open(index["index_path"]).read())


if __name__ == "__main__":
    unittest.main()
