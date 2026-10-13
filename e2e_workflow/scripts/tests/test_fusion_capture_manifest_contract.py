#!/usr/bin/env python3
"""Regression checks for the deterministic KernelFusion manifest contract."""

import os
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
WORKFLOW = os.path.dirname(SCRIPTS)
ROLES = os.path.join(WORKFLOW, "roles")


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
        profile = source.index("phase('Strategize');", required)
        block = source[required:profile]
        self.assertNotIn("throw new Error", block)
        self.assertIn("Recording failure and continuing to Strategize", block)
        self.assertIn("failed_stage: requiredFailureStage", block)

    def test_unexpected_fusion_exception_restores_state_and_continues_profile(self):
        source = self._workflow_source()
        start = source.index("const fusionRecoveryState = {")
        caught = source.index("} catch (e) {", start)
        profile = source.index("phase('Strategize');", caught)
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
        self.assertIn("continuing to Strategize", block)
        self.assertLess(caught, profile)

    def test_external_fusion_artifacts_cannot_bypass_run_local_discovery(self):
        source = self._workflow_source()
        self.assertNotIn("suppliedFusionPriorComplete", source)
        self.assertNotIn("fusionInputsComplete", source)
        self.assertNotIn("const FU =", source)
        self.assertNotIn("(A.fusion && typeof A.fusion === 'object')", source)
        self.assertNotIn("complete fusion prior/state supplied", source)
        self.assertIn("if (!FAST_MODE && FUSION_DISCOVERY_ON && FUSION_BACKENDS.has(BACKEND))", source)
        self.assertIn("FUSION_TOPK_JSON: ''", source)
        self.assertIn("FUSION_CANDIDATES_JSON: ''", source)
        self.assertIn("FUSION_UNITSIDE_JSON: ''", source)

    def test_unsupported_backend_skips_fusion_with_a_recorded_reason(self):
        source = self._workflow_source()
        # vllm has its own capture + semantics path; dropping it from this set
        # silently skips every vllm KernelFusion run.
        self.assertIn("const FUSION_BACKENDS = new Set(['sglang', 'vllm']);", source)
        guard = source.index("if (!FAST_MODE && FUSION_DISCOVERY_ON && !FUSION_BACKENDS.has(BACKEND))")
        body = source.index("if (!FAST_MODE && FUSION_DISCOVERY_ON && FUSION_BACKENDS.has(BACKEND))")
        self.assertLess(guard, body)
        self.assertIn("status: 'skipped_backend'", source[guard:body])
        # KernelFusion now runs after the formal Profile and before Strategize.
        self.assertLess(source.index("label: 'profiler:baseline'"), guard)
        self.assertLess(body, source.index("phase('Strategize');"))

    def test_fusion_reuses_the_profile_trace_and_reprofiles_after_a_win(self):
        source = self._workflow_source()
        baseline = source.index("label: 'profiler:baseline'")
        fusion = source.index("phase('KernelFusion');")
        reprofile = source.index("label: 'profiler:post-fusion'")
        strategize = source.index("phase('Strategize');")
        self.assertLess(baseline, fusion)
        self.assertLess(fusion, reprofile)
        self.assertLess(reprofile, strategize)
        capture = source[fusion:source.index("roleAgent('semantics_mapper', 'build_table'", fusion)]
        self.assertIn("PROFILE_TRACE_DIR: profileTraceDir", capture)
        gate = source[source.index("const fusionAcceptedAtEntry"):reprofile]
        self.assertIn("if (acceptedFusions.length > fusionAcceptedAtEntry)", gate)
        with open(os.path.join(ROLES, "fusion_trace_collector.md")) as fh:
            role = fh.read()
        self.assertLess(role.index("PROFILE_TRACE_DIR"), role.index("TRACELENS_TRACE_FILE` only when"))
        self.assertIn("--auto-select-rank", role)

    def test_a_reused_or_captured_trace_must_hold_both_phases(self):
        # The Profile may skip the prefill ramp to reach steady decode; reusing that
        # trace made every Kimi-K2.5 fusion table decode-only, reported as a pass.
        with open(os.path.join(ROLES, "fusion_trace_collector.md")) as fh:
            collector = fh.read()
        reuse = collector[collector.index("Reuse the formal Profile trace first"):
                          collector.index("TRACELENS_TRACE_FILE` only when")]
        self.assertIn("--require-phases prefill,decode", reuse)
        validate = collector[collector.index("5. Require the completed capture"):]
        self.assertIn("--require-phases prefill,decode", validate)
        with open(os.path.join(ROLES, "semantics_mapper.md")) as fh:
            mapper = fh.read()
        build = mapper[mapper.index("## PHASE=build_table"):mapper.index("## PHASE=complete_table")]
        self.assertIn("--require-phases prefill,decode", build)
        self.assertIn("--layer-boundary-map <map> --require-phases prefill,decode", mapper)

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
            "const fusionApplyEntries = fusionApplyBudgetEntries(\n"
            "      fusionExecutionList, fusionUnitEligibleIds, FUSION_BUDGET);",
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
        # The commit lives in commitStep(), shared with combined mode; the serial loop
        # calls it after every apply_one call, before the next one.
        helper = source.index("const commitStep = (step, accuracyKey, label) => {")
        helper_body = source[helper:source.index("\n    };\n", helper)]
        self.assertIn("for (const accepted of newlyAccepted) acceptedFusions.push(accepted);",
                      helper_body)
        self.assertIn("fusionRecoveryState.acceptedFusionCount = acceptedFusions.length;",
                      helper_body)
        call = source.index("roleAgent('fusion_integrator', 'apply_one'")
        commit = source.index("commitStep(step, accuracyKey, applyEntry.exec_id);", call)
        profile = source.index("phase('Strategize');", commit)
        self.assertLess(helper, call)
        self.assertLess(call, commit)
        self.assertLess(commit, profile)
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
        """The eager-merged table, not the pre-probe table, must flow into the screen."""
        source = self._workflow_source()
        complete = source.index("if (completed) semantics = completed;")
        screen = source.index("label: 'fusion-screen'", complete)
        screen_block = source[complete:screen]
        self.assertIn("roleAgent('kernel_fusion_analyst', 'screen'", screen_block)
        self.assertIn("SEMANTIC_TABLE_JSON: semantics.semantic_table_json", screen_block)
        self.assertIn("WORKLOAD_ISL: ISL", screen_block)
        self.assertIn("WORKLOAD_OSL: OSL", screen_block)
        self.assertIn("WORKLOAD_CONC: CONC", screen_block)
        self.assertIn("TOP_K: FUSION_TOP_K", screen_block)
        self.assertIn("PASSES: FUSION_SCREEN_PASSES", screen_block)

    def test_partial_semantics_screens_only_resolved_phases(self):
        # A partial table (e.g. decode boundaries missing) still screens the phases
        # it resolved and logs the blind ones; a failed or table-less result does not.
        source = self._workflow_source()
        self.assertIn(
            "const semanticsUsable = semantics && semantics.semantic_table_json &&\n"
            "      (semantics.status === 'pass' || semantics.status === 'partial');",
            source,
        )
        self.assertIn("if (semanticsUsable) {", source)
        self.assertIn("semantic table is partial; screening the phases it resolved", source)

    def test_screen_execution_list_drives_unitside(self):
        """Phase 3.0 runs over exactly the execution_list fusion_screen returned."""
        source = self._workflow_source()
        screen = source.index("label: 'fusion-screen'")
        aggregate = source.index("roleAgent('fusion_unit_validator', 'aggregate'", screen)
        block = source[screen:aggregate]
        self.assertIn("FUSION_INPUTS.FUSION_TOPK_JSON = screened.fusion_topk_json;", block)
        self.assertIn("fusionExecutionList = Array.isArray(screened.execution_list)", block)
        self.assertIn("for (const entry of fusionExecutionList) {", block)
        self.assertIn("roleAgent('fusion_unit_validator', 'validate_one'", block)
        self.assertIn("EXEC_ID: job.entry.exec_id, CANDIDATE_ID: job.cid", block)
        self.assertIn("FUSION_TOPK_JSON: FUSION_INPUTS.FUSION_TOPK_JSON", block)

    def test_unitside_aggregate_is_the_only_applyback_gate(self):
        """Apply-back must consume the aggregate produced for this Top-K."""
        source = self._workflow_source()
        aggregate = source.index("roleAgent('fusion_unit_validator', 'aggregate'")
        preserve = source.index(
            "FUSION_INPUTS.FUSION_UNITSIDE_JSON = (unitAgg && unitAgg.fusion_unitside_json) || '';",
            aggregate,
        )
        eligible = source.index("fusionUnitEligibleIds = ", preserve)
        applyback = source.index("roleAgent('fusion_integrator', 'apply_one'", eligible)
        self.assertLess(aggregate, preserve)
        profile = source.index("phase('Strategize');", applyback)
        apply_block = source[eligible:profile]
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
        profile = source.index("phase('Strategize');", fallback)
        block = source[fallback:profile]
        self.assertLess(fallback, profile)
        self.assertIn("pre-Fusion overlay/flags/env/throughput", block)
        self.assertIn("applied: [], blocked: [], deferred: []", block)
        self.assertIn("fusionStatus = appliedN > 0 ? 'applyback_partial_failure' : 'applyback_failed';", block)
        self.assertNotIn(
            "KernelFusion apply-back did not reach a terminal state", source)


if __name__ == "__main__":
    unittest.main()
