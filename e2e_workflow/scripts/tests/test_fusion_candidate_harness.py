import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_candidate_harness as harness


class FusionCandidateHarnessTest(unittest.TestCase):
    def _write(self, root, name, value):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            json.dump(value, fh)
        return path

    def _table(self):
        rows = [
            {
                "row_id": "r0", "pos": 0, "device_seq_index": 10,
                "stream": 8, "duration_us": 6.0, "stage": "norm",
            },
            {
                "row_id": "r1", "pos": 1, "device_seq_index": 11,
                "stream": 8, "duration_us": 4.0, "stage": "quant",
            },
            {
                "row_id": "r2", "pos": 2, "device_seq_index": 12,
                "stream": 8, "duration_us": 90.0, "stage": "gemm",
            },
        ]
        return {
            "trace_sha256": "abc",
            "tables": [{
                "phase": "prefill",
                "pattern_id": "P_DENSE",
                "pattern_display_name": "Dense",
                "pattern_layer_count": 2,
                "representative_layer_id": 0,
                "rows": rows,
            }],
        }

    def _payload(self):
        members = [
            {
                "row_id": "r0", "pos": 0, "device_seq_index": 10,
                "stream": 8, "duration_us": 6.0,
                "stage": "norm", "evidence_level": "K",
            },
            {
                "row_id": "r1", "pos": 1, "device_seq_index": 11,
                "stream": 8, "duration_us": 4.0,
                "stage": "quant", "evidence_level": "K",
            },
        ]
        api = {
            "name": "rmsnorm_group_quant",
            "coverage": "full",
            "source_kind": "runtime_environment",
            "evidence": "installed source signature",
            "constraints": ["group=128"],
        }
        return {
            "phase": "generate_plans",
            "status": "pass",
            "stage_inventory": [
                {
                    "phase": "prefill", "pattern_id": "P_DENSE",
                    "order": 0, "stage": "norm+quant",
                    "row_ids": ["r0", "r1"],
                    "fusion_opportunity": True,
                    "candidate_ids": ["c0"],
                },
                {
                    "phase": "prefill", "pattern_id": "P_DENSE",
                    "order": 1, "stage": "gemm",
                    "row_ids": ["r2"],
                    "fusion_opportunity": False,
                    "candidate_ids": [],
                    "reason": "main donor has no adjacent helper in region",
                },
            ],
            "summary_rows": [{
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_short_name": "P0 Dense",
                "pattern_display_name": "Dense", "order": 0,
                "stage": "Norm producer",
                "source_row_ids": ["r0", "r1"],
                "current_chain_us_per_layer": 10.0,
                "plans": [{
                    "order": 1, "candidate_id": "c0",
                    "plan": "Norm + group quant",
                    "plan_detail": "Fuse quant into the norm producer.",
                    "current_chain_us_per_layer": 10.0,
                    "existing_apis": [api],
                    "exact_kernel_status": "yes",
                    "addressable_us_per_layer": 4.0,
                    "estimated_savings_us": [],
                    "savings_note": "可寻址上限 4 us/层",
                }],
            }],
            "candidates": [{
                "candidate_id": "c0",
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_layer_count": 2,
                "members": members,
                "donor_row_ids": ["r0"],
                "removable_row_ids": ["r1"],
                "current_chain_us_per_layer": 10.0,
                "addressable_us_per_layer": 4.0,
                "stack_addressable_ceiling_us": 8.0,
                "readiness": "ready_for_api_validation",
                "implementation_class": "existing_api_needs_adapter",
                "exact_kernel_status": "yes",
                "existing_apis": [api],
                "risks": [], "validation_requirements": [],
            }],
        }

    def test_validates_facts_coverage_and_renders_total_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            environment = self._write(tmp, "environment.json", {
                "image": "test/image:latest",
                "inspection_evidence": ["source signature inspected"],
            })
            payload = self._payload()
            payload["environment_api_inventory_json"] = environment
            candidates = self._write(
                tmp, "candidates.json", payload)
            md = os.path.join(tmp, "report.md")
            result_path = os.path.join(tmp, "validation.json")
            result = harness.run(table, candidates, md, result_path)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["metrics"]["source_row_coverage_pct"], 100.0)
            with open(md) as fh:
                report = fh.read()
            self.assertIn("Fusion 总表（Prefill → Decode）", report)
            self.assertIn("现成 fusion kernel / API", report)
            self.assertIn("rmsnorm_group_quant", report)
            self.assertIn("① 10.000", report)

    def test_rejects_missing_rows_and_duration_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._write(
                tmp, "environment.json", {
                    "image": "test/image:latest",
                    "inspection_evidence": ["source signature inspected"],
                })
            payload["stage_inventory"] = payload["stage_inventory"][:1]
            payload["candidates"][0]["addressable_us_per_layer"] = 99.0
            table = self._write(tmp, "table.json", self._table())
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "stage inventory misses" in error
                for error in result["errors"]))
            self.assertTrue(any(
                "addressable_us_per_layer" in error
                for error in result["errors"]))

    def test_requires_short_and_full_collective_chain_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            table_payload = self._table()
            rows = table_payload["tables"][0]["rows"]
            rows[0]["stage"] = "communication"
            rows[1]["stage"] = "norm"
            rows[2]["stage"] = "quant"
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._write(
                tmp, "environment.json", {
                    "image": "test/image:latest",
                    "inspection_evidence": ["source signature inspected"],
                })
            table = self._write(tmp, "table.json", table_payload)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "requires plan 2 'allreduce + norm'" in error
                for error in result["errors"]))
            self.assertTrue(any(
                "requires plan 3 'allreduce + norm + quant'" in error
                for error in result["errors"]))
            self.assertTrue(any(
                "requires plan 1 'norm + quant'" in error
                for error in result["errors"]))

    def test_requires_full_family_when_quant_is_non_adjacent(self):
        # MoE-style: communication -> norm -> gemm(router) -> ... -> quant.
        # The norm's fp8 consumer (the expert-input quant) is not adjacent, but
        # the full narrow-to-broad family must still be required.
        with tempfile.TemporaryDirectory() as tmp:
            table_payload = self._table()
            rows = table_payload["tables"][0]["rows"]
            rows[0]["stage"] = "communication"
            rows[1]["stage"] = "norm"
            rows[2]["stage"] = "gemm"
            rows.append({
                "row_id": "r3", "pos": 3, "device_seq_index": 13,
                "stream": 8, "duration_us": 5.0, "stage": "quant",
            })
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._write(
                tmp, "environment.json", {
                    "image": "test/image:latest",
                    "inspection_evidence": ["source signature inspected"],
                })
            table = self._write(tmp, "table.json", table_payload)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "requires plan 1 'norm + quant'" in error
                for error in result["errors"]))
            self.assertTrue(any(
                "requires plan 3 'allreduce + norm + quant'" in error
                for error in result["errors"]))
            # the non-adjacent quant row r3 must be the quant member
            self.assertTrue(any(
                "r3" in error and "allreduce + norm + quant" in error
                for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
