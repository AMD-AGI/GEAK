"""Static contracts for GEAK-v4 source-level AVO integration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AVO_JS = ROOT / "avo-workflow" / "avo_workflow.js"
DIRECTOR = ROOT / "avo-workflow" / "roles" / "director.md"
VARIATION = ROOT / "avo-workflow" / "roles" / "variation_engineer.md"
HEADLESS_RUNNER = ROOT / "avo-workflow" / "run_avo.mjs"
GEAK_RUNNER = ROOT / "avo-workflow" / "run_geak_avo.py"
AVO_TEST_RUNNER = ROOT / "avo-workflow" / "run_tests.sh"
KERNEL_DISPATCHER = ROOT / "kernel_workflow" / "kernel_workflow.js"
CLAUDE_GUIDANCE = ROOT / "CLAUDE.md"


def test_workflow_metadata_is_first_statement():
    text = AVO_JS.read_text()
    assert text.lstrip().startswith("export const meta = {")
    assert "await import('./lib/commit_gate.mjs')" in text
    assert "\nimport {" not in text


def test_source_only_mode_is_explicit_and_low_level_modes_fail_closed():
    text = AVO_JS.read_text()
    assert "const ASM_MODE = false" in text
    assert "const IR_MODE = false" in text
    assert "const LANGUAGE_MODE = 'source'" in text
    assert "const IR_FEEDBACK = false" in text
    assert "STANDALONE_LOW_LEVEL_TARGET" in text
    assert "supports source-level optimization only" in text
    assert "/kernel_workflow" in text
    assert "/perf_knowledge" in text

    role = VARIATION.read_text()
    assert "operator=single" in role
    assert "operator=team" in role


def test_team_and_diverse_multi_start_are_defaults():
    text = AVO_JS.read_text()
    assert "A.operator != null ? A.operator : 'team'" in text
    assert "trim() || 'team'" in text
    assert "A.start_policy != null ? A.start_policy : 'diverse'" in text
    assert "trim() || 'diverse'" in text
    assert "A.start_k != null ? A.start_k : OPERATOR_DIRECTIONS" in text


def test_source_director_and_geak_kernel_roles_are_available():
    role = DIRECTOR.read_text()
    assert "PHASE=setup_native" in role
    assert "PHASE=validate" in role
    assert (ROOT / "kernel_workflow" / "roles" / "benchmark_engineer.md").is_file()
    assert (ROOT / "kernel_workflow" / "roles" / "profile_engineer.md").is_file()
    assert (ROOT / "kernel_workflow" / "scripts" / "gpu_lock.sh").is_file()
    assert (ROOT / "kernel_workflow" / "scripts" / "profile_kernel.sh").is_file()


def test_headless_defaults_are_checkout_relative_and_all_js_tests_run():
    shim = HEADLESS_RUNNER.read_text()
    assert "const DEFAULT_ROOT = path.resolve(path.dirname(CFG.scriptPath), '..')" in shim
    assert "CFG.root_dir || CFG.args.root_dir || DEFAULT_ROOT" in shim

    runner = AVO_TEST_RUNNER.read_text()
    assert '"$HERE"/tests/*.test.mjs' in runner


def test_geak_runner_maps_to_source_only_headless_avo():
    text = GEAK_RUNNER.read_text()
    assert '"avo_language_level": "source"' in text
    assert '"asm_mode": "false"' in text
    assert '"ir_mode": "false"' in text
    assert '"ir_feedback": "false"' in text
    assert 'ROOT / "kernel_workflow"' in text
    assert 'ROOT / "perf_knowledge"' in text
    assert "run_avo.mjs" in text


def test_native_dispatcher_points_avo_users_to_headless_runner():
    text = KERNEL_DISPATCHER.read_text()
    assert "MODE === 'avo'" in text
    assert "run_geak_avo.py <handoff.json> <result.json>" in text
    assert "does not provide module/filesystem primitives required by AVO" in text


def test_claude_natural_language_routes_avo_to_headless_runner():
    text = CLAUDE_GUIDANCE.read_text()
    assert "Long-horizon source evolution" in text
    assert "avo-workflow/run_geak_avo.py" in text
    assert "Do not invoke `avo_workflow.js` with the native Workflow tool" in text
    assert '`operator: "team"`' in text
    assert '`start_policy: "diverse"`' in text
    assert "--dry-run" in text
