#!/usr/bin/env python3
"""Regression checks for the deterministic KernelFusion manifest contract."""

import os
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
WORKFLOW = os.path.dirname(SCRIPTS)


class FusionCaptureManifestContractTest(unittest.TestCase):
    def _workflow_source(self):
        with open(os.path.join(WORKFLOW, "e2e_workflow.js")) as fh:
            return fh.read()

    def test_capture_generates_and_validates_manifest(self):
        with open(os.path.join(SCRIPTS, "bench_e2e.sh")) as fh:
            source = fh.read()
        self.assertIn('--trace-dir "$PROFILE_DIR"', source)
        self.assertIn('--out "$_TRACE_MANIFEST"', source)
        self.assertIn('doc.get("status") != "pass"', source)
        self.assertIn('not doc.get("analysis_rank_trace")', source)
        self.assertIn('--auto-select-rank', source)

    def test_workflow_falls_back_to_deterministic_manifest(self):
        with open(os.path.join(WORKFLOW, "e2e_workflow.js")) as fh:
            source = fh.read()
        self.assertIn(
            'const fusionCaptureDir = `${EVAL_DIR}/${fusionRound}`;',
            source,
        )
        self.assertIn(
            'const expectedFusionManifest = `${fusionCaptureDir}/profile_trace_manifest.json`;',
            source,
        )
        self.assertIn("CAPTURE_DIR: fusionCaptureDir", source)
        self.assertIn("TRACE_MANIFEST_JSON: expectedFusionManifest", source)
        self.assertIn("CAPTURE_REPEATS: 1", source)
        self.assertIn("CAPTURE_NUM_PROMPTS: Math.max(CONC * 5, CONC)", source)
        self.assertIn('TRACE_MANIFEST_JSON: fusionTraceManifest', source)
        self.assertNotIn(
            "notes: 'fusion capture produced no raw trace manifest'",
            source,
        )

    def test_capture_role_uses_bounded_deterministic_output(self):
        with open(os.path.join(WORKFLOW, "roles", "fusion_trace_collector.md")) as fh:
            source = fh.read()
        self.assertIn("`OUT_DIR=CAPTURE_DIR`", source)
        self.assertIn("`REPEATS=CAPTURE_REPEATS`", source)
        self.assertIn("`NUM_PROMPTS=CAPTURE_NUM_PROMPTS`", source)
        self.assertIn("publish exactly `TRACE_MANIFEST_JSON`", source)
        self.assertIn("Do not return before the foreground capture command", source)

    def test_explicit_fusion_discovery_is_not_silently_skipped(self):
        with open(os.path.join(WORKFLOW, "e2e_workflow.js")) as fh:
            source = fh.read()
        self.assertIn("const FUSION_REQUIRED", source)
        self.assertIn(
            "KernelFusion was explicitly required but produced no apply-back-ready Top-K",
            source,
        )
        required = source.index(
            "KernelFusion was explicitly required but produced no apply-back-ready Top-K")
        profile = source.index("phase('Profile');", required)
        block = source[required:profile]
        self.assertNotIn("throw new Error", block)
        self.assertIn("Recording failure and continuing to formal Profile", block)
        self.assertIn("failed_stage: requiredFailureStage", block)

    def test_unexpected_fusion_exception_restores_state_and_continues_profile(self):
        source = self._workflow_source()
        start = source.index("const fusionRecoveryState = {")
        caught = source.index("} catch (e) {", start)
        profile = source.index("phase('Profile');", caught)
        block = source[caught:profile]
        self.assertIn("curOverlay = fusionRecoveryState.overlay;", block)
        self.assertIn("curFlags = fusionRecoveryState.flags;", block)
        self.assertIn("curEnv = fusionRecoveryState.env;", block)
        self.assertIn("curTput = fusionRecoveryState.throughput;", block)
        self.assertIn(
            "acceptedFusions.length = fusionRecoveryState.acceptedFusionCount;",
            block,
        )
        self.assertIn("FUSION_INPUTS = { ...fusionRecoveryState.inputs };", block)
        self.assertIn("status: 'unexpected_exception'", block)
        self.assertIn("failed_stage: fusionFailureStage", block)
        self.assertIn("continuing to formal Profile", block)
        self.assertLess(caught, profile)

    def test_external_fusion_artifacts_cannot_bypass_run_local_discovery(self):
        source = self._workflow_source()
        self.assertNotIn("suppliedFusionPriorComplete", source)
        self.assertNotIn("fusionInputsComplete", source)
        self.assertNotIn("const FU =", source)
        self.assertNotIn("(A.fusion && typeof A.fusion === 'object')", source)
        self.assertNotIn("complete fusion prior/state supplied", source)
        self.assertIn("if (!FAST_MODE && FUSION_DISCOVERY_ON && BACKEND === 'sglang')", source)
        self.assertIn("FUSION_TOPK_JSON: ''", source)
        self.assertIn("FUSION_CANDIDATES_JSON: ''", source)
        self.assertIn("FUSION_UNITSIDE_JSON: ''", source)

    def test_non_sglang_backend_skips_fusion_with_a_recorded_reason(self):
        source = self._workflow_source()
        guard = source.index("if (!FAST_MODE && FUSION_DISCOVERY_ON && BACKEND !== 'sglang')")
        body = source.index("if (!FAST_MODE && FUSION_DISCOVERY_ON && BACKEND === 'sglang')")
        self.assertLess(guard, body)
        self.assertIn("status: 'skipped_backend'", source[guard:body])
        self.assertLess(body, source.index("phase('Profile');"))

    def test_applyback_cannot_time_out_into_a_stale_profile(self):
        source = self._workflow_source()
        self.assertIn("const FUSION_APPLY_TIMEOUT_MS", source)
        self.assertIn(
            "timeoutMs: FUSION_APPLY_TIMEOUT_MS",
            source,
        )

    def test_applyback_runs_one_execution_entry_per_agent_call(self):
        source = self._workflow_source()
        self.assertIn(
            "const fusionApplyEntries = fusionExecutionList.slice",
            source,
        )
        self.assertIn(
            "for (let applyIndex = 0; applyIndex < fusionApplyEntries.length; applyIndex++)",
            source,
        )
        self.assertIn("roleAgent('fusion_integrator', 'apply_one'", source)
        self.assertIn("TARGET_EXEC_ID: applyEntry.exec_id", source)
        self.assertIn("PRIOR_APPLY_RESULT: applyState", source)
        self.assertNotIn("roleAgent('fusion_integrator', 'apply_back'", source)

    def test_applyback_commits_each_terminal_win_before_the_next_call(self):
        source = self._workflow_source()
        call = source.index("roleAgent('fusion_integrator', 'apply_one'")
        commit = source.index(
            "for (const accepted of newlyAccepted) acceptedFusions.push(accepted);",
            call,
        )
        profile = source.index("phase('Profile');", commit)
        self.assertLess(call, commit)
        self.assertLess(commit, profile)
        self.assertIn(
            "fusionRecoveryState.acceptedFusionCount = acceptedFusions.length;",
            source[commit:profile],
        )
        failure = source.index("if (!step) {", call)
        stop = source.index("break;", failure)
        self.assertLess(failure, stop)
        self.assertLess(stop, profile)

    def test_applyback_role_is_scoped_to_one_entry(self):
        with open(os.path.join(WORKFLOW, "roles", "fusion_integrator.md")) as fh:
            source = fh.read()
        self.assertIn("PHASE=apply_one", source)
        self.assertIn("Process exactly", source)
        self.assertIn("`TARGET_EXEC_ID`", source)
        self.assertIn("`PRIOR_APPLY_RESULT`", source)
        self.assertIn("--allow-partial-coverage", source)

    def test_completed_shape_table_is_the_source_for_candidates_and_topk(self):
        """The eager-merged table, not the pre-probe table, must flow forward."""
        source = self._workflow_source()
        complete = source.index("if (completed) semantics = completed;")
        generate = source.index(
            "label: 'fusion-analyst:generate'", complete)
        rank = source.index("label: 'fusion-analyst:rank'", generate)
        self.assertLess(complete, generate)
        self.assertLess(generate, rank)
        generate_block = source[complete:generate]
        rank_block = source[generate:rank]
        self.assertIn(
            "SEMANTIC_TABLE_JSON: semantics.semantic_table_json",
            generate_block,
        )
        self.assertIn(
            "SEMANTIC_TABLE_JSON: semantics.semantic_table_json",
            rank_block,
        )
        self.assertIn(
            "FUSION_CANDIDATES_JSON: discover.fusion_candidates_json",
            rank_block,
        )
        self.assertIn(
            "FUSION_VALIDATION_JSON: discover.validation_json",
            rank_block,
        )
        self.assertIn("WORKLOAD_ISL: ISL", rank_block)
        self.assertIn("WORKLOAD_OSL: OSL", rank_block)
        self.assertIn("WORKLOAD_CONC: CONC", rank_block)

    def test_partial_semantics_cannot_enter_fusion_discovery(self):
        source = self._workflow_source()
        self.assertIn(
            "if (semantics && semantics.status === 'pass' && semantics.semantic_table_json)",
            source,
        )

    def test_topk_execution_list_drives_exact_unitside_candidates(self):
        """Every unit-side call must be named by Top-K exec/candidate ids."""
        source = self._workflow_source()
        start = source.index("const execById = new Map();")
        aggregate = source.index(
            "label: 'fusion-unit:aggregate'", start)
        block = source[start:aggregate]
        self.assertIn("ranked.execution_list || []", block)
        self.assertIn("unit_representative_candidate_id", block)
        self.assertIn("unit_equivalent_candidate_ids", block)
        self.assertIn("equivalentCovered", block)
        self.assertIn("EXEC_ID: item.exec_id", block)
        self.assertIn("CANDIDATE_ID: item.candidate_id", block)
        self.assertIn(
            "FUSION_CANDIDATES_JSON: discover.fusion_candidates_json",
            block,
        )
        self.assertIn("FUSION_TOPK_JSON: ranked.fusion_topk_json", block)

    def test_unitside_aggregate_is_the_only_applyback_gate(self):
        """Apply-back must consume the aggregate produced for this Top-K."""
        source = self._workflow_source()
        aggregate = source.index("label: 'fusion-unit:aggregate'")
        preserve = source.index(
            "FUSION_INPUTS.FUSION_UNITSIDE_JSON = aggregate.fusion_unitside_json;",
            aggregate,
        )
        applyback = source.index("roleAgent('fusion_integrator', 'apply_one'", preserve)
        self.assertLess(aggregate, preserve)
        self.assertLess(preserve, applyback)
        profile = source.index("phase('Profile');", applyback)
        apply_block = source[preserve:profile]
        self.assertIn(
            "FUSION_INPUTS.FUSION_TOPK_JSON && FUSION_INPUTS.FUSION_UNITSIDE_JSON",
            apply_block,
        )
        self.assertIn(
            "FUSION_UNITSIDE_JSON: FUSION_INPUTS.FUSION_UNITSIDE_JSON",
            apply_block,
        )

    def test_applyback_failure_preserves_prefusion_state_and_continues_profile(self):
        source = self._workflow_source()
        fallback = source.index(
            "KernelFusion apply-back failed or returned no terminal result")
        profile = source.index("phase('Profile');", fallback)
        block = source[fallback:profile]
        self.assertLess(fallback, profile)
        self.assertIn("pre-Fusion overlay/flags/env/throughput", block)
        self.assertIn("applied: [], blocked: [], deferred: []", block)
        self.assertIn("fusionStatus = appliedN > 0 ? 'applyback_partial_failure' : 'applyback_failed';", block)
        self.assertNotIn(
            "KernelFusion apply-back did not reach a terminal state", source)


if __name__ == "__main__":
    unittest.main()
