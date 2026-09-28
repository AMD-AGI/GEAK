import json
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_candidate_harness as harness


class FusionCandidateHarnessTest(unittest.TestCase):
    def _write(self, root, name, value):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            json.dump(value, fh)
        return path

    def test_roofline_json_is_preferred_over_unstructured_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            roofline = self._write(tmp, "profile_roofline.json", {
                "hbm_bw_bytes_s": 5.3e12})
            self.assertEqual(
                harness._hbm_bw_bytes_per_us(
                    {"inspection_evidence": ["GPU prose without arch"]}, roofline),
                5.3e6)

    def test_structured_arch_does_not_scan_free_text(self):
        self.assertIsNone(harness._gfx_key({
            "inspection_evidence": ["some sentence containing gfx942"]}))
        self.assertEqual(
            harness._gfx_key({"gpu": {"gcn_arch_name": "gfx942:sramecc+"}}),
            "gfx942")

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

    def _shaped(self, table, resolved=True):
        """Give every row a captured shape (or strip them all).

        The phase gate measures shape resolution off the rows themselves rather
        than trusting the declared record, so a fixture table has to carry the
        same evidence a real one does.
        """
        for item in table.get("tables", []):
            for row in item.get("rows", []):
                if resolved:
                    row.setdefault("shape", {
                        "input_dims": [[8, 16]], "input_types": ["bf16"]})
                else:
                    row.pop("shape", None)
        return table

    def _coverage(self, phases=("prefill",), resolved=True):
        """A healthy phase_coverage record: both halves of the evidence present.

        Phase 2 now refuses to build on a phase whose shapes were never
        resolved, so a fixture table must say what it actually covered.
        """
        return {
            "phases_in_tables": list(phases),
            "phases_in_trace": list(phases),
            "phases_absent_from_tables": [],
            "shape_resolution_by_phase": {
                phase: {"rows": 3, "resolved": 3 if resolved else 0,
                        "resolved_fraction": 1.0 if resolved else 0.0}
                for phase in phases},
            "decode_sequence_covered": "decode" in phases,
            "decode_shapes_covered": "decode" in phases and resolved,
            "decode_covered": "decode" in phases and resolved,
            "decode_evidence": (
                "sequence_and_shapes" if ("decode" in phases and resolved)
                else "sequence_only_shapes_unresolved" if "decode" in phases
                else "no_decode_trace_analysed"),
            "decode_requires_graph_capture": (
                "decode" in phases and not resolved),
            "required_phases": list(phases),
            "missing_required_phases": [],
        }

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
        return self._shaped({
            "trace_sha256": "abc",
            "phase_coverage": self._coverage(),
            "tables": [{
                "phase": "prefill",
                "pattern_id": "P_DENSE",
                "pattern_display_name": "Dense",
                "pattern_layer_count": 2,
                "representative_layer_id": 0,
                "rows": rows,
            }],
        })

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
                "live_call_seam": "layernorm.py:151 rmsnorm call site",
                "existing_apis": [api],
                "risks": [], "validation_requirements": [],
            }],
        }

    def test_cited_api_gate_backfills_inventory_and_warns_on_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._write(tmp, "env.json", {
                "image": "test", "inspection_evidence": ["source"]})
            catalog_path = self._write(tmp, "catalog.json", {"kernels": [{
                "name": "pkg.fused_flatten_quant",
                "op_tags": ["layout", "quant"], "dtype_tags": [],
            }]})
            payload = {"candidates": [
                {"candidate_id": "c0", "plan_detail": "flatten then quant",
                 "existing_apis": [{"name": "pkg.fused_flatten_quant"}]},
                {"candidate_id": "c1", "plan_detail": "flatten then quant",
                 "existing_apis": [{"name": "pkg.fused_flatten_quant"}]},
            ]}
            errors, warnings = [], []
            result = harness._audit_cited_apis(
                payload, catalog_path, env_path, errors, warnings)
            self.assertEqual(errors, [])
            self.assertEqual(result["cited"], 1)
            self.assertTrue(result["backfilled"])
            self.assertTrue(any("api_spread_warning" in w for w in warnings))
            with open(env_path) as fh:
                inventory = json.load(fh)
            self.assertEqual(
                inventory["available_apis"][0]["source"],
                "cited_not_inventoried")

    def test_cited_api_absent_from_catalog_is_hard_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = self._write(tmp, "env.json", {})
            catalog_path = self._write(tmp, "catalog.json", {"kernels": [{
                "name": "pkg.real_fusion", "op_tags": [], "dtype_tags": []}]})
            errors, warnings = [], []
            harness._audit_cited_apis(
                {"candidates": [{
                    "candidate_id": "c0",
                    "existing_apis": [{"name": "pkg.imagined_fusion"}],
                }]},
                catalog_path, env_path, errors, warnings)
            self.assertTrue(any("unverified_api" in error for error in errors))

    # ---- collective fixtures (comm member present; comm->norm made
    # non-contiguous so the ①②③ collective-coverage requirement stays out of
    # the way and the size-guard logic can be tested in isolation) ----
    def _collective_table(self, tokens=8):
        rows = [
            {"row_id": "c0", "pos": 0, "device_seq_index": 20, "stream": 8,
             "duration_us": 5.0, "stage": "communication"},
            {"row_id": "n0", "pos": 1, "device_seq_index": 25, "stream": 8,
             "duration_us": 4.0, "stage": "norm"},
            # dsi 40, not 26: this fixture's own stage_inventory calls q0 an
            # "isolated quant, no producer in region", so it must NOT be
            # device-adjacent to n0 -- otherwise n0+q0 is a genuine fusible
            # region and the region rule is right to demand a candidate for it.
            {"row_id": "q0", "pos": 2, "device_seq_index": 40, "stream": 8,
             "duration_us": 3.0, "stage": "quant"},
        ]
        return self._shaped({
            "trace_sha256": "abc",
            "phase_coverage": self._coverage(),
            "tables": [{
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_display_name": "Dense", "pattern_layer_count": 2,
                "representative_layer_id": 0,
                "selected_bucket": {
                    "phase": "prefill", "batch_size": 1,
                    "input_tokens": tokens},
                "rows": rows,
            }],
        })

    def _collective_payload(self, exact="yes"):
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
            "live_call_seam": "--enable-aiter-allreduce-fusion",
            "flag_routed_signature": {
                "routed_call_ref": "layernorm.py:151",
                "fused_fn": "fused_allreduce_rmsnorm",
                "arg_signature": "(x, residual, weight, eps)",
                "covers_ops": ["allreduce", "rmsnorm"]},
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

    def test_non_adjacent_quant_is_not_claimed_by_collective_ladder(self):
        # MoE-style: communication -> norm -> gemm/router -> ... -> quant.
        # The later quant is a different data domain; only AR+norm is mandatory.
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
                "requires plan 1 'allreduce + norm'" in error
                for error in result["errors"]))
            self.assertFalse(any(
                "allreduce + norm + quant" in error or
                "requires plan 1 'norm + quant'" in error
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
        return self._shaped({
            "trace_sha256": "abc",
            "phase_coverage": self._coverage(),
            "tables": [{
                "phase": "prefill", "pattern_id": "P_DENSE",
                "pattern_display_name": "Dense", "pattern_layer_count": 58,
                "representative_layer_id": 0,
                "selected_bucket": {
                    "phase": "prefill", "batch_size": 1,
                    "input_tokens": tokens},
                "rows": rows,
            }],
        })

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
            "live_call_seam": "--enable-aiter-allreduce-fusion",
            "flag_routed_signature": {
                "routed_call_ref": "layernorm.py:151",
                "fused_fn": "fused_allreduce_rmsnorm",
                "arg_signature": "(x, residual, weight, eps)",
                "covers_ops": ["allreduce", "rmsnorm"]},
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

    def test_boundary_size_guard_records_exceeds_without_forcing_no(self):
        # 现成算子(exact) is binary "kernel exists" — a size-guard exceed does NOT
        # flip it to 无. The fused kernel still exists (exact stays 有); the guard
        # only records that the fused path falls back at this shape (Top-K drops
        # it as non-actionable via collective_guard_checks).
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._boundary_table())
            payload = self._boundary_payload(exact="yes", occurrences=57)
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=64)  # 256B exceeds
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertFalse(any(
                "exact must be no" in error for error in result["errors"]))
            self.assertTrue(any(
                c["verdict"] == "exceeds"
                for c in result["metrics"]["collective_guard_checks"]))

    def test_collective_guard_records_exceeds_keeps_exact_yes(self):
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
            self.assertEqual(result["status"], "pass")
            self.assertFalse(any(
                "exact must be no" in error for error in result["errors"]))
            checks = result["metrics"]["collective_guard_checks"]
            self.assertEqual(checks[0]["verdict"], "exceeds")

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

    def test_flag_quant_family_without_scale_arg_fails(self):
        # Gate A: a *_quant family cannot be tier A on a flag whose routed
        # signature carries no scale/quant/fp8 arg (the flag does not fuse quant).
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._collective_table())
            payload = self._collective_payload(exact="yes")
            payload["candidates"][0]["family"] = "collective_norm_quant"
            # routed signature has no scale/quant token -> quant NOT fused
            payload["candidates"][0]["flag_routed_signature"][
                "arg_signature"] = "(x, residual, weight, eps)"
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=100000)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "does not fuse quant" in e for e in result["errors"]))

    def test_flag_missing_routed_signature_fails(self):
        # Gate A: every existing_flag_or_env candidate must record the routed sig.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._collective_table())
            payload = self._collective_payload(exact="yes")
            payload["candidates"][0].pop("flag_routed_signature", None)
            payload["environment_api_inventory_json"] = self._env(
                tmp, threshold_bytes=100000)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "flag_routed_signature" in e for e in result["errors"]))

    def test_author_track_requires_absence_search(self):
        # Gate B: 现成算子=无 (author-track) must record the exhaustive absence
        # search; without it the claim "no kernel" is unbacked.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            payload = self._payload()
            c0 = payload["candidates"][0]
            c0["implementation_class"] = "new_helper_kernel"
            c0["exact_kernel_status"] = "no"
            c0["exact_reason"] = "no installed kernel fuses norm+quant here"
            c0["existing_apis"] = []
            payload["summary_rows"][0]["plans"][0][
                "exact_kernel_status"] = "no"
            payload["summary_rows"][0]["plans"][0][
                "exact_reason"] = "no installed kernel"
            payload["summary_rows"][0]["plans"][0]["existing_apis"] = []
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertTrue(any(
                "absence_search" in e for e in result["errors"]))
            # with a recorded search the absence_search gate no longer fires
            c0["absence_search"] = [{
                "query": "grep -rn 'rmsnorm.*quant' aiter/ops",
                "location": "/sgl-workspace/aiter/aiter/ops",
                "result": "no fused norm+quant kernel for this dtype"}]
            candidates = self._write(tmp, "candidates2.json", payload)
            result2 = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertFalse(any(
                "absence_search" in e for e in result2["errors"]))

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

    # ---- author-track completeness: a large non-donor helper must not be
    # dropped into a fusion_opportunity=false stage ----
    def _table_with_layout_helper(self, layout_us=10.0):
        table_payload = self._table()
        table_payload["tables"][0]["rows"].append({
            "row_id": "r_layout", "pos": 3, "device_seq_index": 13,
            "stream": 8, "duration_us": layout_us, "stage": "elementwise",
            "shape": {"input_dims": [[128, 7168]],
                      "input_types": ["c10::BFloat16"]}})
        return table_payload

    def _payload_covering_layout(self):
        payload = self._payload()
        # cover the layout row in the inventory (so coverage passes) but leave
        # it out of every candidate -> it is a silently dropped helper.
        payload["stage_inventory"][1]["row_ids"] = ["r2", "r_layout"]
        return payload

    def test_large_helper_dropped_without_candidate_or_followup_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table_with_layout_helper())
            payload = self._payload_covering_layout()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "dropped without candidate" in error
                for error in result["errors"]))
            self.assertEqual(
                result["metrics"]["dropped_helper_row_count"], 1)

    def test_dropped_helper_rescued_by_followup_row_ids_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table_with_layout_helper())
            payload = self._payload_covering_layout()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["required_followups"] = [{
                "id": "f_layout",
                "reason": "output layout copy needs producer/consumer proof",
                "recommended_next_step": "capture runtime-source dependency",
                "row_ids": ["r_layout"],
            }]
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["metrics"]["dropped_helper_row_count"], 0)

    def test_large_helper_in_followup_still_fails_escalation(self):
        # A 20us helper (>= escalate floor) deferred to a followup is NOT enough:
        # it must be an actual candidate (author-track).
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(
                tmp, "table.json", self._table_with_layout_helper(20.0))
            payload = self._payload_covering_layout()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["required_followups"] = [{
                "id": "f_layout", "reason": "deferred",
                "recommended_next_step": "capture", "row_ids": ["r_layout"]}]
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "must be CANDIDATES" in error for error in result["errors"]))
            self.assertEqual(
                result["metrics"]["escalated_missing_row_count"], 1)

    def test_merge_ceiling_rejects_all_members_removable(self):
        # norm+quant with BOTH removable and no anchor -> addressable = full
        # chain = 100% -> must be rejected by the merge ceiling.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            cand = payload["candidates"][0]
            cand["removable_row_ids"] = ["r0", "r1"]      # both -> no anchor
            cand["addressable_us_per_layer"] = 10.0        # 6+4 = full chain
            cand["stack_addressable_ceiling_us"] = 20.0    # 10 * 2 layers
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "merge ceiling" in error for error in result["errors"]))

    def test_explicit_donor_anchor_allows_more_expensive_helpers_to_be_removed(self):
        # A fused GEMM can absorb dequant helpers that are individually slower
        # than the donor. The donor remains the semantic anchor even when it is
        # not the heaviest member; the old max-duration heuristic rejected this
        # valid cross-donor shape.
        with tempfile.TemporaryDirectory() as tmp:
            table_payload = self._table()
            table_payload["tables"][0]["rows"][2]["duration_us"] = 3.0
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            cand = payload["candidates"][0]
            cand["members"].append({
                "row_id": "r2", "pos": 2, "device_seq_index": 12,
                "stream": 8, "duration_us": 3.0, "stage": "gemm",
                "evidence_level": "K",
            })
            cand["donor_row_ids"] = ["r2"]
            cand["anchor_row_id"] = "r2"
            cand["removable_row_ids"] = ["r0", "r1"]
            cand["current_chain_us_per_layer"] = 13.0
            cand["addressable_us_per_layer"] = 10.0
            cand["stack_addressable_ceiling_us"] = 20.0
            summary = payload["summary_rows"][0]
            summary["source_row_ids"] = ["r0", "r1", "r2"]
            summary["current_chain_us_per_layer"] = 13.0
            summary["single_plan_reason"] = "one donor absorbs both helpers"
            plan = summary["plans"][0]
            plan["current_chain_us_per_layer"] = 13.0
            plan["addressable_us_per_layer"] = 10.0
            candidates = self._write(tmp, "candidates.json", payload)
            table = self._write(tmp, "table.json", table_payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertFalse(any(
                "merge ceiling" in error for error in result["errors"]),
                result["errors"])

    def test_collective_row_only_in_followup_fails(self):
        # A tail all-reduce covered only by a followup (not a candidate member)
        # must fail: every collective is a fusion anchor.
        with tempfile.TemporaryDirectory() as tmp:
            table_payload = self._table()
            table_payload["tables"][0]["rows"].append({
                "row_id": "tail_ar", "pos": 3, "device_seq_index": 13,
                "stream": 8, "duration_us": 40.0, "stage": "communication"})
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["stage_inventory"][1]["row_ids"] = ["r2", "tail_ar"]
            payload["required_followups"] = [{
                "id": "f_tail", "reason": "cross-layer boundary deferred",
                "recommended_next_step": "boundary candidate",
                "row_ids": ["tail_ar"]}]
            table = self._write(tmp, "table.json", table_payload)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "collective (all-reduce) rows are not candidate members"
                in error for error in result["errors"]))
            self.assertEqual(
                result["metrics"]["collective_not_candidate_count"], 1)

    def test_aggregate_escalation_clusters_small_helpers(self):
        # Several sub-15us helpers that individually escape the per-row escalate
        # floor but sum to >= the aggregate floor in one (phase, pattern) must
        # become a cluster candidate, not scattered followup rows.
        with tempfile.TemporaryDirectory() as tmp:
            table_payload = self._table()
            for i, dur in enumerate((8.0, 8.0, 8.0)):  # 24 us/layer >= 20
                table_payload["tables"][0]["rows"].append({
                    "row_id": "small%d" % i, "pos": 3 + i,
                    "device_seq_index": 13 + i, "stream": 8,
                    "duration_us": dur, "stage": "elementwise"})
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["stage_inventory"][1]["row_ids"] = [
                "r2", "small0", "small1", "small2"]
            payload["required_followups"] = [{
                "id": "f_small", "reason": "scattered small helpers",
                "recommended_next_step": "cluster fusion",
                "row_ids": ["small0", "small1", "small2"]}]
            table = self._write(tmp, "table.json", table_payload)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "cluster to" in error for error in result["errors"]))
            self.assertEqual(
                result["metrics"]["agg_escalate_violation_count"], 1)

    def test_exact_yes_adapter_without_seam_fails(self):
        # An exact=yes needs-adapter candidate with no live_call_seam must fail:
        # if a flag/env engages it with no code it should be existing_flag_or_env
        # (A / ConfigSweep), otherwise it must point at the wiring seam.
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            del payload["candidates"][0]["live_call_seam"]  # exact=yes, adapter
            table = self._write(tmp, "table.json", self._table())
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "must record live_call_seam" in error
                for error in result["errors"]))

    def test_flag_or_env_without_seam_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["candidates"][0][
                "implementation_class"] = "existing_flag_or_env"
            del payload["candidates"][0]["live_call_seam"]
            table = self._write(tmp, "table.json", self._table())
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "existing_flag_or_env must record the enabling flag" in error
                for error in result["errors"]))

    def test_any_candidate_spanning_donor_fails(self):
        # Donor-span validation applies to every non-boundary family.  This
        # collective-named candidate deliberately has no *_cluster suffix; it
        # must still fail when its members straddle a GEMM.
        with tempfile.TemporaryDirectory() as tmp:
            table_payload = self._table()  # r0 norm10, r1 quant11, r2 gemm12
            table_payload["tables"][0]["rows"].append({
                "row_id": "r3", "pos": 3, "device_seq_index": 13,
                "stream": 8, "duration_us": 8.0, "stage": "elementwise"})
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["stage_inventory"][1]["row_ids"] = ["r2", "r3"]
            # cluster candidate members r0(seq10) + r3(seq13) span r2 gemm(seq12)
            payload["candidates"][0].update({
                "candidate_id": "c_collective",
                "family": "collective_norm_quant",
                "members": [
                    {"row_id": "r0", "pos": 0, "device_seq_index": 10,
                     "stream": 8, "duration_us": 6.0, "stage": "norm",
                     "evidence_level": "P"},
                    {"row_id": "r3", "pos": 3, "device_seq_index": 13,
                     "stream": 8, "duration_us": 8.0, "stage": "elementwise",
                     "evidence_level": "P"}],
                "donor_row_ids": ["r0"], "removable_row_ids": ["r3"],
                "current_chain_us_per_layer": 14.0,
                "addressable_us_per_layer": 8.0,
                "stack_addressable_ceiling_us": 16.0})
            payload["summary_rows"][0]["source_row_ids"] = ["r0", "r3"]
            payload["summary_rows"][0]["current_chain_us_per_layer"] = 14.0
            payload["summary_rows"][0]["plans"][0].update({
                "candidate_id": "c_collective", "addressable_us_per_layer": 8.0,
                "current_chain_us_per_layer": 14.0})
            payload["stage_inventory"][0]["candidate_ids"] = ["c_collective"]
            table = self._write(tmp, "table.json", table_payload)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "spans" in e and "main donor" in e for e in result["errors"]))

    def test_provenance_trace_sha_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["source_semantic_table"] = {
                "path": "other.json", "trace_sha256": "not-abc"}
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "does not match semantic table trace_sha256" in error
                for error in result["errors"]))

    # ---- phase-coverage entry gate (1.4) --------------------------------
    # Phase 1 has always recorded how much of each phase it resolved; nothing
    # downstream read it. On DSR1 2026-08-26 two published tables carried
    # decode_covered=false with 0/64 decode rows shape-resolved and Phase 2
    # built decode candidates on them anyway.

    def _decode_table(self, resolved=False):
        """A decode table whose shapes were (or were not) actually resolved.

        `resolved=False` is the DSR1 shape: decode rows are present -- the
        SEQUENCE was captured -- but 0 of them carry a measured shape, because
        CUDA-graph replay emits no nn.Module spans to hang shapes off.
        """
        table = self._shaped(self._table(), resolved=resolved)
        table["tables"][0]["phase"] = "decode"
        cov = self._coverage(phases=("decode",), resolved=resolved)
        cov["shape_resolution_by_phase"]["decode"] = {
            "rows": 64, "resolved": 64 if resolved else 0,
            "resolved_fraction": 1.0 if resolved else 0.0}
        table["phase_coverage"] = cov
        return table

    def _decode_payload(self):
        payload = self._payload()
        for candidate in payload["candidates"]:
            candidate["phase"] = "decode"
        for row in payload["stage_inventory"]:
            row["phase"] = "decode"
        for row in payload["summary_rows"]:
            row["phase"] = "decode"
        return payload

    def test_table_without_phase_coverage_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._table()
            raw.pop("phase_coverage")
            table = self._write(tmp, "table.json", raw)
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "phase_coverage" in e and "Regenerate" in e
                for e in result["errors"]), result["errors"])

    def test_decode_candidates_on_shape_blind_decode_are_refused(self):
        # The exact DSR1 shape: decode rows are in the table, decode shapes are
        # 0/64 resolved, and the candidates are decode candidates.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._decode_table())
            payload = self._decode_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                "decode" in e and "resolved 0" in e for e in result["errors"]),
                result["errors"])
            self.assertFalse(result["phase_coverage"]["ok"])

    def test_decode_candidates_pass_once_decode_shapes_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(
                tmp, "table.json", self._decode_table(resolved=True))
            payload = self._decode_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "pass", result["errors"])
            self.assertTrue(result["phase_coverage"]["ok"])

    def test_shape_blind_decode_can_be_waived_with_a_reason(self):
        # The escape hatch exists, but it is loud: a waiver leaves a warning and
        # a recorded reason, never a silent pass.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._decode_table())
            payload = self._decode_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"),
                allow_partial_phase_coverage="eager probe queued for round 2")
            self.assertEqual(result["status"], "pass", result["errors"])
            self.assertEqual(
                result["phase_coverage"]["waiver"],
                "eager probe queued for round 2")
            self.assertTrue(any("waived" in w for w in result["warnings"]),
                            result["warnings"])

    # ---- fusible-region enumeration (3.1) -------------------------------
    # Before this rule only the collective positions had a mandatory
    # narrow-to-broad family; every other region depended on the analyst's
    # judgement against a us floor, which is why the same trace produced a
    # different candidate set on every run.

    def _region_table(self):
        """norm -> quant -> activation, all non-donor, then a GEMM donor."""
        table = self._table()
        rows = table["tables"][0]["rows"]
        rows.insert(2, {
            "row_id": "r1b", "pos": 2, "device_seq_index": 12,
            "stream": 8, "duration_us": 5.0, "stage": "activation",
            "shape": {"input_dims": [[8, 16]], "input_types": ["bf16"]}})
        rows[3]["pos"] = 3
        rows[3]["device_seq_index"] = 13
        return table

    def _region_payload(self):
        """The `_payload` candidate, plus the extra region row in the inventory.

        The candidate still fuses only norm+quant, so the activation row is
        accounted for row-by-row but the REGION it belongs to is not covered --
        which is exactly the gap the region rule exists to catch.
        """
        payload = self._payload()
        payload["stage_inventory"].append({
            "phase": "prefill", "pattern_id": "P_DENSE",
            "order": 2, "stage": "activation",
            "row_ids": ["r1b"], "fusion_opportunity": False,
            "candidate_ids": [],
            "reason": "activation left out of the norm+quant plan"})
        return payload

    def test_uncovered_fusible_region_fails(self):
        # The candidate fuses norm+quant but leaves the adjacent activation row
        # in the same region unclaimed and undeferred.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._region_table())
            payload = self._region_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("fusible region" in e for e in result["errors"]),
                            result["errors"])
            self.assertEqual(result["region_coverage"]["uncovered"], 1)

    def test_a_failing_run_still_publishes_its_report(self):
        # Suppressing the markdown on failure made a failed phase look identical to
        # a phase that never ran -- the root index just said "未生成" and the
        # uncovered regions, which are the whole point of the failure, were
        # invisible. The report publishes either way, flagged red.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._region_table())
            payload = self._region_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            out_md = os.path.join(tmp, "report.md")
            result = harness.run(table, candidates, out_md,
                                 os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(os.path.isfile(out_md))
            with open(out_md) as fh:
                text = fh.read()
            self.assertIn("校验未通过", text)
            self.assertIn("fusible region", text)

    def test_a_passing_run_is_not_prefixed_with_a_failure_banner(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._region_table())
            payload = self._region_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["required_followups"] = [{
                "row_ids": ["r-act"], "reason": "no fused activation kernel exists "
                                                "for this dtype; round 2"}]
            candidates = self._write(tmp, "candidates.json", payload)
            out_md = os.path.join(tmp, "report.md")
            result = harness.run(table, candidates, out_md,
                                 os.path.join(tmp, "validation.json"))
            with open(out_md) as fh:
                text = fh.read()
            if result["status"] == "pass":
                self.assertNotIn("校验未通过", text)
            self.assertTrue(text.strip())

    def _kb_index(self, root):
        """A one-card knowledge/learned KB, as the analyst would read it."""
        kb = os.path.join(root, "learned")
        os.makedirs(kb, exist_ok=True)
        with open(os.path.join(kb, "fusion-known.md"), "w") as fh:
            fh.write("---\nkey: fusion \u00b7 gfx942 \u00b7 fp8\ntype: lever\n"
                     "confidence: \u2605\u2605\u2605\neffect: -9% decode\n"
                     "last_seen: 2026-08-26\n---\n# A fusion we already measured\n"
                     "- lever: try it\n- source: eval_dir\n")
        index = os.path.join(kb, "INDEX.md")
        with open(index, "w") as fh:
            fh.write("# Learned\n\n## kernel fusion\n"
                     "- [gfx942] a fusion we already measured "
                     "\u2605\u2605\u2605 \u2014 (fusion-known.md)\n")
        return index

    def test_a_known_fusion_prior_with_no_disposition_fails_the_gate(self):
        # The run-to-run instability: the same trace produced a different fusion
        # set each run because a measured prior could simply never be proposed,
        # and "never proposed" leaves no artifact saying so.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._region_table())
            payload = self._region_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"),
                priors_index=self._kb_index(tmp))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("no disposition" in e
                                for e in result["errors"]), result["errors"])
            self.assertEqual(
                result["prior_coverage"]["undisposed"], 1)

    def test_a_prior_ruled_out_with_a_reason_clears_the_gate(self):
        # ADD-only: the gate is on the SENTENCE existing, not on the answer
        # being yes. An honest "does not apply here" is a complete disposition.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._region_table())
            payload = self._region_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["prior_dispositions"] = [{
                "card": "fusion-known", "disposition": "not_applicable",
                "reason": "card is gfx942 MLA decode; this table has no MLA "
                          "rows in either phase"}]
            candidates = self._write(tmp, "candidates.json", payload)
            out_md = os.path.join(tmp, "report.md")
            result = harness.run(
                table, candidates, out_md,
                os.path.join(tmp, "validation.json"),
                priors_index=self._kb_index(tmp))
            self.assertFalse(any("disposition" in e for e in result["errors"]),
                             result["errors"])
            self.assertEqual(result["prior_coverage"]["not_applicable"], 1)
            with open(out_md) as fh:
                text = fh.read()
            self.assertIn("\u5df2\u77e5\u878d\u5408\u5148\u9a8c", text)

    def test_without_a_priors_index_nothing_is_imposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._region_table())
            payload = self._region_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(table, candidates,
                                 os.path.join(tmp, "report.md"),
                                 os.path.join(tmp, "validation.json"))
            self.assertIsNone(result["prior_coverage"])
            self.assertFalse(any("disposition" in e for e in result["errors"]))

    def test_cli_defaults_enable_repository_prior_gates(self):
        self.assertTrue(os.path.isfile(harness.DEFAULT_PRIORS_INDEX))
        self.assertTrue(os.path.isfile(harness.DEFAULT_FUSION_STRATEGIES))
        argv = [
            "fusion_candidate_harness.py",
            "--semantic-table", "table.json",
            "--candidates", "candidates.json",
            "--out-md", "out.md",
            "--result-json", "result.json",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
                harness, "run", return_value={"status": "pass"}) as run:
            self.assertEqual(harness.main(), 0)
        args = run.call_args[0]
        self.assertEqual(args[8], harness.DEFAULT_PRIORS_INDEX)
        self.assertEqual(args[10], harness.DEFAULT_FUSION_STRATEGIES)

    def test_fusible_region_deferred_in_followups_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._region_table())
            payload = self._region_payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            payload["required_followups"] = [{
                "row_ids": ["r0", "r1", "r1b"],
                "reason": "no installed kernel fuses activation into this "
                          "chain; author-track, deferred to round 2"}]
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "pass", result["errors"])
            self.assertEqual(result["region_coverage"]["deferred"], 1)
            self.assertEqual(result["region_coverage"]["uncovered"], 0)

    def test_region_covered_by_the_candidate_passes(self):
        # The unmodified two-row fixture: the candidate covers the whole region.
        with tempfile.TemporaryDirectory() as tmp:
            table = self._write(tmp, "table.json", self._table())
            payload = self._payload()
            payload["environment_api_inventory_json"] = self._env(tmp)
            candidates = self._write(tmp, "candidates.json", payload)
            result = harness.run(
                table, candidates, os.path.join(tmp, "report.md"),
                os.path.join(tmp, "validation.json"))
            self.assertEqual(result["status"], "pass", result["errors"])
            self.assertEqual(result["region_coverage"]["covered"], 1)

    def test_a_region_never_spans_a_donor(self):
        # A fused kernel cannot cross a GEMM/attn/MoE/collective body, so the
        # donor partitions the layer into independent regions -- it never
        # merges the rows on either side into one.
        rows = [
            {"row_id": "a0", "pos": 0, "device_seq_index": 10, "stream": 8,
             "duration_us": 6.0, "stage": "norm"},
            {"row_id": "a1", "pos": 1, "device_seq_index": 11, "stream": 8,
             "duration_us": 4.0, "stage": "quant"},
            {"row_id": "g0", "pos": 2, "device_seq_index": 12, "stream": 8,
             "duration_us": 90.0, "stage": "gemm"},
            {"row_id": "b0", "pos": 3, "device_seq_index": 13, "stream": 8,
             "duration_us": 6.0, "stage": "norm"},
            {"row_id": "b1", "pos": 4, "device_seq_index": 14, "stream": 8,
             "duration_us": 4.0, "stage": "quant"},
        ]
        regions = harness._fusible_regions(
            {"tables": [{"phase": "prefill", "pattern_id": "P_DENSE",
                         "rows": rows}]}, 5.0)
        self.assertEqual([sorted(r[1]) for r in regions],
                         [["a0", "a1"], ["b0", "b1"]])


if __name__ == "__main__":
    unittest.main()


class MemberShapeGraftTest(FusionCandidateHarnessTest):
    # only the four tests below belong to this class

    """The harness copies member shapes out of the semantic table itself.

    Phase 3.0 builds BOTH the microbench and the split reference from
    `members[].shape.input_dims`, and the only other place a shape can come from is
    the run config -- where multiplying `input_tokens` x `kv_lora_rank` yields a
    plausible-looking wrong shape with no batch axis and no rope columns. On DSR1
    2026-08-30 the members happened to carry `shape` (155 of them) and every gate
    worked; on 2026-08-31, following the documented schema literally, they carried
    none (0), the shape-provenance gate degraded to no gate, and two fusions worth
    +20.6% and +14.0% e2e were blocked on a degenerate-path parity fail. The
    difference between those runs was luck. Grafting removes the luck.
    """

    def test_shapes_are_grafted_onto_members_from_the_table(self):
        with tempfile.TemporaryDirectory() as root:
            table = self._write(root, "table.json", self._table())
            p = self._payload()
            p["environment_api_inventory_json"] = self._env(root)
            cands = self._write(root, "cands.json", p)
            payload, _, errors, _, metrics = harness.validate(table, cands)
            self.assertEqual(errors, [])
            graft = metrics["member_shape_graft"]
            self.assertEqual(graft["grafted"], 2)
            self.assertEqual(graft["shapeless_candidates"], [])
            members = payload["candidates"][0]["members"]
            self.assertEqual([m["shape"]["input_dims"] for m in members],
                             [[[8, 16]], [[8, 16]]])
            self.assertEqual(members[0]["shape"]["input_types"], ["bf16"])

    def test_an_author_supplied_shape_is_never_overwritten(self):
        payload = self._payload()
        payload["candidates"][0]["members"][0]["shape"] = {
            "input_dims": [[1, 2]], "input_types": ["bf16"]}
        with tempfile.TemporaryDirectory() as root:
            table = self._write(root, "table.json", self._table())
            payload["environment_api_inventory_json"] = self._env(root)
            cands = self._write(root, "cands.json", payload)
            out, _, errors, _, metrics = harness.validate(table, cands)
            self.assertEqual(errors, [])
            self.assertEqual(metrics["member_shape_graft"]["already_present"], 1)
            self.assertEqual(metrics["member_shape_graft"]["grafted"], 1)
            self.assertEqual(
                out["candidates"][0]["members"][0]["shape"]["input_dims"], [[1, 2]])

    def test_empty_shape_placeholder_is_grafted_from_the_table(self):
        payload = self._payload()
        payload["candidates"][0]["members"][0]["shape"] = {
            "input_dims": [], "input_types": []}
        with tempfile.TemporaryDirectory() as root:
            table = self._write(root, "table.json", self._table())
            payload["environment_api_inventory_json"] = self._env(root)
            cands = self._write(root, "cands.json", payload)
            out, _, errors, _, metrics = harness.validate(table, cands)
            self.assertEqual(errors, [])
            self.assertEqual(metrics["member_shape_graft"]["grafted"], 2)
            self.assertEqual(
                out["candidates"][0]["members"][0]["shape"]["input_dims"],
                [[8, 16]])

    def test_a_candidate_whose_rows_resolve_no_shape_is_an_error(self):
        # A table with NO shapes at all is caught earlier, by the phase entry gate.
        # This is the subtler one the entry gate cannot see: the table resolves
        # shapes, but not on the rows THIS candidate is built from -- so Phase 3.0
        # would still have to invent the only shape that matters.
        tbl = self._table()
        for row in tbl["tables"][0]["rows"]:
            if row["row_id"] in ("r0", "r1"):
                row.pop("shape", None)
        with tempfile.TemporaryDirectory() as root:
            table = self._write(root, "table.json", tbl)
            p = self._payload()
            p["environment_api_inventory_json"] = self._env(root)
            cands = self._write(root, "cands.json", p)
            _, _, errors, _, metrics = harness.validate(table, cands)
            self.assertIn("c0", metrics["member_shape_graft"]["shapeless_candidates"])
            self.assertTrue(any("no member resolved an input_dims" in e
                                for e in errors))

    def test_the_graft_is_persisted_so_phase_3_0_can_read_it(self):
        # in-memory is not enough: fusion_unitside_harness re-reads THIS file and
        # gates every verdict's tested_shape against these member shapes.
        with tempfile.TemporaryDirectory() as root:
            table = self._write(root, "table.json", self._table())
            p = self._payload()
            p["environment_api_inventory_json"] = self._env(root)
            cands = self._write(root, "cands.json", p)
            harness.run(table, cands, os.path.join(root, "out.md"),
                        os.path.join(root, "res.json"))
            with open(cands) as fh:
                on_disk = json.load(fh)
            self.assertEqual(
                on_disk["candidates"][0]["members"][0]["shape"]["input_dims"],
                [[8, 16]])
            with open(os.path.join(root, "res.json")) as fh:
                self.assertEqual(json.load(fh)["member_shape_graft"]["grafted"], 2)
