"""Static contract tests for the scheduled DRA learned-KB refresh.

These checks are intentionally GPU/network-free. The real performance
comparison remains an L1 experiment; L0 proves only that the expensive cadence
is opt-in and that its flags and learned artifacts cross every CI boundary.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/dra-kb-refresh.yml").read_text()
RUN_LOCAL = (ROOT / "ci/node/run_local.sh").read_text()
HELD_SUBMIT = (ROOT / "ci/dispatch/held_submit.sh").read_text()
RUN_E2E = (ROOT / "interface/run_e2e.py").read_text()
E2E = (ROOT / "e2e_workflow/e2e_workflow.js").read_text()


def test_refresh_is_scheduled_manual_and_serialized_with_l1():
    assert "schedule:" in WORKFLOW
    assert 'cron: "0 6 1,15 * *"' in WORKFLOW
    assert "workflow_dispatch:" in WORKFLOW
    assert "group: l1-spur-${{ github.repository }}" in WORKFLOW
    assert "cancel-in-progress: false" in WORKFLOW


def test_refresh_enables_dra_and_uses_existing_campaign_driver():
    assert 'GEAK_DRA_ENABLED: "true"' in WORKFLOW
    assert 'GEAK_USE_LEARNED_KB: "true"' in WORKFLOW
    assert "ci/dispatch/run_matrix.sh" in WORKFLOW
    assert "DRA_KB_REFRESH_MODEL" in WORKFLOW
    assert "DRA_KB_REFRESH_BUDGET_S" in WORKFLOW


def test_refresh_drains_validates_and_opens_only_a_learned_tree_pr():
    for command in (
        "drain --apply",
        "lint --cards",
        "index --check",
        "doctor",
        "stats",
    ):
        assert command in WORKFLOW
    assert "if: steps.kb.outputs.changed == 'true'" in WORKFLOW
    assert 'git status --porcelain -- "$KB_DIR"' in WORKFLOW
    assert 'git add -- "$KB_DIR"' in WORKFLOW
    assert "git add -A" not in WORKFLOW
    assert "gh pr create" in WORKFLOW
    assert "GEAK_KB_PR_TOKEN || github.token" in WORKFLOW
    assert "Empirical speedup comparison is not performed" in WORKFLOW


def test_cadence_flags_cross_held_node_and_container_boundaries():
    for name in ("GEAK_DRA_ENABLED", "GEAK_USE_LEARNED_KB"):
        assert name in HELD_SUBMIT
        assert f"-e {name}" in RUN_LOCAL
        assert name in RUN_E2E


def test_e2e_injects_dra_once_for_every_nested_kernel_lane():
    assert E2E.count("const laneArgs = (wfArgs) =>") == 1
    assert "dra_enabled: LANE_DRA_ENABLED" in E2E
    assert "use_learned_kb: LANE_USE_LEARNED_KB" in E2E
    assert "A.dra_enabled != null ? A.dra_enabled : 'false'" in E2E
