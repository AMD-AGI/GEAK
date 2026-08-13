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

    def _env(self, root, threshold_bytes=67108864, hidden_size=16,
             dtype_bytes=2, aiter_commit=None):
        env = {
            "image": "test/image:latest",
            "inspection_evidence": ["source signature inspected"],
            "collective_fused_ar_guard": {
                "threshold_bytes": threshold_bytes,
                "source_expr": "total_bytes < 8 * 1024 * 8192",
                "source_ref": "communicator_cuda.py::fused_allreduce_rmsnorm",
            },
            "model_dims": {
                "hidden_size": hidden_size, "dtype_bytes": dtype_bytes},
        }
        if aiter_commit:
            env["toolchain"] = {"aiter_git_commit": aiter_commit}
        return self._write(root, "environment.json", env)

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

    # ---- collective fixtures (comm member present; comm->norm made
    # non-contiguous so the ①②③ collective-coverage requirement stays out of
    # the way and the size-guard logic can be tested in isolation) ----
    def _collective_table(self, tokens=8):
        rows = [
            {"row_id": "c0", "pos": 0, "device_seq_index": 20, "stream": 8,
             "duration_us": 5.0, "stage": "communication"},
            {"row_id": "n0", "pos": 1, "device_seq_index": 25, "stream": 8,
             "duration_us": 4.0, "stage": "norm"},
            {"row_id": "q0", "pos": 2, "device_seq_index": 26, "stream": 8,
             "duration_us": 3.0, "stage": "quant"},
        ]
        return {
            "trace_sha256": "abc",
            "tables": [{
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_display_name": "Dense", "pattern_layer_count": 2,
                "representative_layer_id": 0,
                "selected_bucket": {
                    "phase": "prefill", "batch_size": 1,
                    "input_tokens": tokens},
                "rows": rows,
            }],
        }

    def _collective_payload(self, exact="no"):
        api = {
            "name": "fused_allreduce_rmsnorm", "coverage": "full",
            "source_kind": "runtime_environment",
            "evidence": "installed source signature", "constraints": []}
        members = [
            {"row_id": "c0", "pos": 0, "device_seq_index": 20, "stream": 8,
             "duration_us": 5.0, "stage": "communication", "evidence_level": "K"},
            {"row_id": "n0", "pos": 1, "device_seq_index": 25, "stream": 8,
             "duration_us": 4.0, "stage": "norm", "evidence_level": "K"}]
        plan = {
            "order": 1, "candidate_id": "col0", "plan": "allreduce + norm",
            "plan_detail": "Fuse AR with the residual norm.",
            "current_chain_us_per_layer": 9.0, "existing_apis": [api],
            "exact_kernel_status": exact, "exact_reason": (
                "" if exact == "yes" else "unwired seam"),
            "addressable_us_per_layer": 4.0, "estimated_savings_us": []}
        candidate = {
            "candidate_id": "col0", "phase": "prefill", "pattern_id": "P_DENSE",
            "pattern_layer_count": 2, "members": members,
            "donor_row_ids": ["c0"], "removable_row_ids": ["n0"],
            "current_chain_us_per_layer": 9.0, "addressable_us_per_layer": 4.0,
            "stack_addressable_ceiling_us": 8.0,
            "readiness": "ready_for_api_validation",
            "implementation_class": "existing_flag_or_env",
            "exact_kernel_status": exact,
            "exact_reason": "" if exact == "yes" else "unwired seam",
            "existing_apis": [api], "risks": [], "validation_requirements": []}
        return {
            "phase": "generate_plans", "status": "pass",
            "stage_inventory": [
                {"phase": "prefill", "pattern_id": "P_DENSE", "order": 0,
                 "stage": "collective", "row_ids": ["c0", "n0"],
                 "fusion_opportunity": True, "candidate_ids": ["col0"]},
                {"phase": "prefill", "pattern_id": "P_DENSE", "order": 1,
                 "stage": "tail quant", "row_ids": ["q0"],
                 "fusion_opportunity": False, "candidate_ids": [],
                 "reason": "isolated quant, no producer in region"}],
            "summary_rows": [{
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_short_name": "P0 Dense", "pattern_display_name": "Dense",
                "order": 0, "stage": "AR + norm", "source_row_ids": ["c0", "n0"],
                "current_chain_us_per_layer": 9.0, "plans": [plan]}],
            "candidates": [candidate]}

    def test_validates_facts_coverage_and_renders_total_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
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
            payload["environment_api_inventory_json"] = self._env(tmp)
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
            payload["environment_api_inventory_json"] = self._env(tmp)
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
            payload["environment_api_inventory_json"] = self._env(tmp)
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
            self.assertTrue(any(
                "r3" in error and "allreduce + norm + quant" in error
                for error in result["errors"]))

    # ---- boundary (cross-layer) fixtures: head norm/quant at low pos, the
    # previous-layer tail all-reduce at high pos (wrap-around members) ----
    def _boundary_table(self, tokens=8):
        rows = [
            {"row_id": "h_norm", "pos": 0, "device_seq_index": 10, "stream": 8,
             "duration_us": 4.0, "stage": "norm"},
            {"row_id": "h_quant", "pos": 1, "device_seq_index": 11, "stream": 8,
             "duration_us": 3.0, "stage": "quant"},
            {"row_id": "body", "pos": 2, "device_seq_index": 12, "stream": 8,
             "duration_us": 50.0, "stage": "gemm"},
            {"row_id": "tail_ar", "pos": 3, "device_seq_index": 13, "stream": 8,
             "duration_us": 6.0, "stage": "communication"},
        ]
        return {
            "trace_sha256": "abc",
            "tables": [{
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_display_name": "Dense", "pattern_layer_count": 58,
                "representative_layer_id": 0,
                "selected_bucket": {
                    "phase": "prefill", "batch_size": 1,
                    "input_tokens": tokens},
                "rows": rows,
            }],
        }

    def _boundary_payload(self, exact="yes", occurrences=57,
                          include_occurrences=True):
        api = {
            "name": "fused_allreduce_rmsnorm_quant_per_group", "coverage": "full",
            "source_kind": "runtime_environment",
            "evidence": "installed source signature", "constraints": []}
        # wrap-around member order: previous-layer tail AR, then this-layer head
        members = [
            {"row_id": "tail_ar", "pos": 3, "device_seq_index": 13, "stream": 8,
             "duration_us": 6.0, "stage": "communication", "evidence_level": "K"},
            {"row_id": "h_norm", "pos": 0, "device_seq_index": 10, "stream": 8,
             "duration_us": 4.0, "stage": "norm", "evidence_level": "K"},
            {"row_id": "h_quant", "pos": 1, "device_seq_index": 11, "stream": 8,
             "duration_us": 3.0, "stage": "quant", "evidence_level": "K"}]
        candidate = {
            "candidate_id": "bnd0", "phase": "prefill", "pattern_id": "P_DENSE",
            "pattern_layer_count": 58, "boundary": True,
            "members": members, "donor_row_ids": ["tail_ar"],
            "removable_row_ids": ["h_norm", "h_quant"],
            "current_chain_us_per_layer": 13.0, "addressable_us_per_layer": 7.0,
            "stack_addressable_ceiling_us": 7.0 * occurrences,
            "readiness": "needs_source_dependency_proof",
            "implementation_class": "existing_flag_or_env",
            "exact_kernel_status": exact,
            "exact_reason": "" if exact == "yes" else "boundary",
            "existing_apis": [api], "risks": [], "validation_requirements": []}
        if include_occurrences:
            candidate["boundary_occurrences"] = occurrences
        plan = {
            "order": 1, "candidate_id": "bnd0",
            "plan": "allreduce + norm + quant",
            "plan_detail": "Fuse previous-layer tail AR into head norm+quant.",
            "current_chain_us_per_layer": 13.0, "existing_apis": [api],
            "exact_kernel_status": exact,
            "exact_reason": "" if exact == "yes" else "boundary",
            "addressable_us_per_layer": 7.0, "estimated_savings_us": []}
        return {
            "phase": "generate_plans", "status": "pass",
            "stage_inventory": [
                {"phase": "prefill", "pattern_id": "P_DENSE", "order": 0,
                 "stage": "boundary collective", "row_ids": [
                     "tail_ar", "h_norm", "h_quant"],
                 "fusion_opportunity": True, "candidate_ids": ["bnd0"]},
                {"phase": "prefill", "pattern_id": "P_DENSE", "order": 1,
                 "stage": "body", "row_ids": ["body"],
                 "fusion_opportunity": False, "candidate_ids": [],
                 "reason": "main donor"}],
            "summary_rows": [{
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_short_name": "P0 Dense", "pattern_display_name": "Dense",
                "order": 0, "stage": "boundary AR + norm + quant",
                "source_row_ids": ["tail_ar", "h_norm", "h_quant"],
                "single_plan_reason": "boundary ③ only (① is the shared "
                "body-start head candidate)",
                "current_chain_us_per_layer": 13.0, "plans": [plan]}],
            "candidates": [candidate]}

    def test_boundary_candidate_passes_with_wraparound_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._boundary_table())
            payload = self._boundary_payload(exact="yes", occurrences=57)
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=100000)  # 256B fits
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "pass")

    def test_boundary_missing_occurrences_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._boundary_table())
            payload = self._boundary_payload(
                exact="yes", occurrences=57, include_occurrences=False)
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=100000)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "boundary_occurrences" in error for error in result["errors"]))

    def test_boundary_size_guard_forces_no_in_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._boundary_table())
            payload = self._boundary_payload(exact="yes", occurrences=57)
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=64)  # 256B exceeds
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "exact must be no" in error for error in result["errors"]))

    def test_collective_exact_forced_no_when_tensor_exceeds_guard(self):
        # tokens=8 * hidden=16 * dtype=2 = 256 bytes >= 64 threshold -> exceeds
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._collective_table())
            payload = self._collective_payload(exact="yes")
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=64)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "exact must be no" in error for error in result["errors"]))

    def test_collective_exact_allowed_when_tensor_fits_guard(self):
        # 256 bytes < 100000 threshold -> fits -> exact=yes allowed
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._collective_table())
            payload = self._collective_payload(exact="yes")
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=100000)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "pass")
            checks = result["metrics"]["collective_guard_checks"]
            self.assertEqual(len(checks), 1)
            self.assertEqual(checks[0]["verdict"], "fits")

    def test_missing_collective_guard_fields_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            payload = self._payload()
            env = {
                "image": "test/image:latest",
                "inspection_evidence": ["src"],
                "model_dims": {"hidden_size": 16, "dtype_bytes": 2},
            }  # missing collective_fused_ar_guard
            payload["environment_api_inventory_json"] = self._write(
                tmp, "environment.json", env)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "collective_fused_ar_guard" in error
                for error in result["errors"]))

    def test_threshold_disagreeing_with_registry_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            payload = self._payload()
            # real commit is in the registry with 67108864; declare a wrong one
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=12345,
                aiter_commit="a6bb499375849eec45d68c5ccaebc8865fd422c0")
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "disagrees with guard registry" in error
                for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
