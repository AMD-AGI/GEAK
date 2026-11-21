"""Static contracts for source-level AVO living inside a GEAK checkout."""

from pathlib import Path


AVO_DIR = Path(__file__).resolve().parents[1]
# <GEAK>/perf_knowledge/expert_skills/avo/avo-source/avo-source-impl -> <GEAK>
GEAK_ROOT = AVO_DIR.parents[4]
KERNEL_BASE = AVO_DIR / "kernel_base"
AVO_JS = AVO_DIR / "avo_workflow.js"
DIRECTOR = AVO_DIR / "roles" / "director.md"
VARIATION = AVO_DIR / "roles" / "variation_engineer.md"
HEADLESS_RUNNER = AVO_DIR / "run_avo.mjs"
GEAK_RUNNER = AVO_DIR / "run_geak_avo.py"
AVO_TEST_RUNNER = AVO_DIR / "run_tests.sh"
PROFILE_WRAPPER = KERNEL_BASE / "scripts" / "profile_kernel.sh"
GPU_LOCK_FORWARDER = KERNEL_BASE / "scripts" / "gpu_lock.sh"
OWN_KNOWLEDGE = {"amd_instinct.md"}


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
    assert "/kernel_base" in text
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


def test_source_director_and_carried_kernel_roles_are_available():
    role = DIRECTOR.read_text()
    assert "PHASE=setup_native" in role
    assert "PHASE=validate" in role
    for name in ("benchmark_engineer", "profile_engineer", "tech_lead", "integrator"):
        path = KERNEL_BASE / "roles" / f"{name}.md"
        assert path.is_file() and not path.is_symlink(), path
    assert PROFILE_WRAPPER.is_file() and not PROFILE_WRAPPER.is_symlink()


def test_gpu_lock_is_reused_from_geak_through_a_forwarder():
    geak_lock = GEAK_ROOT / "kernel_workflow" / "scripts" / "gpu_lock.sh"
    assert geak_lock.is_file()
    text = GPU_LOCK_FORWARDER.read_text()
    assert "kernel_workflow/scripts" in text and 'gpu_lock.sh" "$@"' in text
    # The forwarder must resolve to GEAK's script from where it sits.
    target = (GPU_LOCK_FORWARDER.parent / ("../" * 7) / "kernel_workflow" / "scripts").resolve()
    assert target == geak_lock.parent


def test_kernel_knowledge_is_reused_file_by_file_without_the_card_store():
    knowledge = KERNEL_BASE / "knowledge"
    geak_knowledge = GEAK_ROOT / "kernel_workflow" / "knowledge"
    assert knowledge.is_dir() and not knowledge.is_symlink()
    entries = sorted(path.name for path in knowledge.iterdir())
    assert "learned" not in entries
    assert all(name.endswith(".md") for name in entries)
    for name in entries:
        path = knowledge / name
        if name in OWN_KNOWLEDGE:
            assert path.is_file() and not path.is_symlink(), path
        else:
            assert path.is_symlink(), path
            assert path.resolve() == (geak_knowledge / name).resolve(), path
            assert path.read_text().strip(), path
    # Every knowledge file the carried roles name must be readable here.
    for name in ("optimization_strategies.md", "geomean_levers.md", "hip_optimization.md",
                 "triton_optimization.md", "wrapper_optimization.md", "amd_instinct.md",
                 "profiling_guide.md"):
        assert (knowledge / name).exists(), name


def test_profiler_wrapper_keeps_the_runner_script_contract():
    text = PROFILE_WRAPPER.read_text()
    # rocprof-compute gets a slash-free name and a separate output path.
    assert '-n "$workload_name" -p "$workload"' in text
    # The profiler receives an executable runner, not a flattened `bash -c` command.
    assert '-- "$PROFILE_RUNNER"' in text
    # The workload keeps its workspace cwd while profiler scratch stays under output_dir.
    assert "printf 'cd %q" in text and 'cd "$OUTPUT_DIR"' in text
    # A failed analysis cannot pass as successful profiling.
    assert 'if [ "$analyze_rc" -ne 0 ]' in text


def test_headless_defaults_point_at_the_geak_checkout_and_all_js_tests_run():
    shim = HEADLESS_RUNNER.read_text()
    assert "path.resolve(path.dirname(CFG.scriptPath), '../../../../..')" in shim
    assert "CFG.root_dir || CFG.args.root_dir || DEFAULT_ROOT" in shim
    assert "`${DEFAULT_ROOT}-runs/avo-source`" in shim

    runner = AVO_TEST_RUNNER.read_text()
    assert '"$HERE"/tests/*.test.mjs' in runner


def test_geak_runner_maps_to_source_only_headless_avo():
    text = GEAK_RUNNER.read_text()
    assert '"avo_language_level": "source"' in text
    assert '"asm_mode": "false"' in text
    assert '"ir_mode": "false"' in text
    assert '"ir_feedback": "false"' in text
    assert '"kernel_workflow_dir": str(KERNEL_BASE)' in text
    assert 'ROOT / "perf_knowledge"' in text
    assert "run_avo.mjs" in text


def test_geak_entry_points_do_not_know_about_avo():
    # GEAK must run exactly as before; nothing outside expert_skills refers to AVO.
    for rel in ("kernel_workflow/kernel_workflow.js", "e2e_workflow/e2e_workflow.js"):
        text = (GEAK_ROOT / rel).read_text()
        assert "run_geak_avo" not in text and "avo_workflow" not in text, rel
    assert not (GEAK_ROOT / "CLAUDE.md").exists()
