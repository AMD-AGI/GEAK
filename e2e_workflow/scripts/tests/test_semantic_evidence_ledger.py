import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_evidence_ledger as ledger


class SemanticEvidenceLedgerTest(unittest.TestCase):
    def _write(self, root, name, value):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            json.dump(value, fh)
        return path

    def _document(self, rows):
        return {"tables": [{
            "phase": "decode",
            "pattern_id": "P",
            "pattern_display_name": "pattern",
            "representative_layer_id": 1,
            "selected_bucket": {
                "phase": "decode", "batch_size": 4, "input_tokens": 0},
            "event_count": len(rows),
            "layer_total_us": sum(row["duration_us"] for row in rows),
            "rows": rows,
        }]}

    def _row(self, row_id, shape_source="unresolved"):
        return {
            "pos": int(row_id[-1]),
            "row_id": row_id,
            "raw_event_index": int(row_id[-1]),
            "device_seq_index": int(row_id[-1]),
            "raw_name": row_id,
            "short_name": row_id,
            "stage": "gemm",
            "duration_us": 1.0,
            "shape": {
                "source": shape_source,
                "input_dims": [[4, 8]] if shape_source == "kernel_exact" else [],
                "input_types": ["bf16"] if shape_source == "kernel_exact" else [],
            },
            "parent_operator": {"canonical_op": "unresolved"},
        }

    def test_accumulates_probe_runs_without_downgrading_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_rows = [
                self._row("event-0", "kernel_exact"),
                self._row("event-1"),
                self._row("event-2"),
            ]
            graph_rows = [dict(row) for row in clean_rows]
            graph_rows[0]["semantic_evidence"] = {"level": "K"}
            graph_rows[1]["semantic_evidence"] = {
                "level": "P", "probe_scope": "wrapper",
                "bucket_match": "exact",
                "schema": {"tensors": [{"io": "input", "shape": [4, 8]}]},
            }
            graph_rows[1]["shape"] = {
                "source": "runtime_probe_wrapper",
                "logger_schema": graph_rows[1]["semantic_evidence"]["schema"],
            }
            graph_rows[2]["semantic_evidence"] = {
                "level": "U", "reason_code": "kernel_not_observed_in_probe",
                "reason": "not observed"}

            eager_rows = [dict(row) for row in clean_rows]
            eager_rows[0]["semantic_evidence"] = {
                "level": "P", "probe_scope": "wrapper"}
            eager_rows[1]["semantic_evidence"] = {
                "level": "U", "reason_code": "no_source_confirmed_wrapper",
                "reason": "missing"}
            eager_rows[2]["semantic_evidence"] = {
                "level": "P", "probe_scope": "kernel",
                "bucket_match": "exact",
                "schema": {"tensors": [{"io": "input", "shape": [4, 16]}]},
            }
            eager_rows[2]["shape"] = {
                "source": "runtime_probe_kernel",
                "logger_schema": eager_rows[2]["semantic_evidence"]["schema"],
            }

            result = ledger.merge(
                self._write(tmp, "clean.json", self._document(clean_rows)),
                [
                    self._write(
                        tmp, "graph.json", self._document(graph_rows)),
                    self._write(
                        tmp, "eager.json", self._document(eager_rows)),
                ],
                os.path.join(tmp, "out"))
            self.assertEqual(result["evidence_counts"], {
                "K": 1, "P": 2, "U": 0})
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            self.assertEqual(rows[0]["semantic_evidence"]["level"], "K")
            self.assertEqual(
                rows[1]["semantic_evidence"]["probe_scope"], "wrapper")
            self.assertEqual(
                rows[2]["semantic_evidence"]["probe_scope"], "kernel")
            with open(result["coverage_manifest"]) as fh:
                coverage = json.load(fh)
            self.assertEqual(coverage["probe_scope_counts"], {
                "P(wrapper)": 1, "P(kernel)": 1})

    def test_u_is_complete_only_with_machine_readable_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean = self._document([self._row("event-0")])
            probe = self._document([{
                **self._row("event-0"),
                "semantic_evidence": {
                    "level": "U",
                    "reason_code": "kernel_not_observed_in_probe",
                    "reason": "not observed",
                },
            }])
            result = ledger.merge(
                self._write(tmp, "clean.json", clean),
                [self._write(tmp, "probe.json", probe)],
                os.path.join(tmp, "out"))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["evidence_counts"]["U"], 1)
            with open(result["coverage_manifest"]) as fh:
                unavailable = json.load(fh)["unavailable"]
            self.assertEqual(
                unavailable[0]["reason_code"],
                "kernel_not_observed_in_probe")


if __name__ == "__main__":
    unittest.main()
