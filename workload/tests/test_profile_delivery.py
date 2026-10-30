"""The profile slot's verdict, pinned against the two deliveries that disagreed.

The same extraction skill, run through this pipeline and run by hand, produced different
deliveries, and the slot called the narrower one `ok`. These tests are the shapes of both:
the pipeline run that must now be rejected, and the hand-driven run that must now pass.
Everything here is CPU-only JSON (PRINCIPLES R11) -- what is being tested is the
judgement, not the extraction.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


WORKLOAD = Path(__file__).resolve().parents[1]
FINALIZE = WORKLOAD / "stages/profile/finalize.py"
CONFIGURED = {"selection_top_k": 10, "selection_overall_pct_gt": 0.5,
              "selection_phases": ["PREFILL", "DECODE"]}
POLICY = {"top_k_per_phase": 10, "overall_pct_gt": 0.5, "phases": ["PREFILL", "DECODE"]}
SELECTOR_ARGS = ["--top-k", "10", "--overall-pct-gt", "0.5",
                 "--phases", "PREFILL", "DECODE"]
# The ten tasks the hand-driven delivery produced, with the identities each one owns and
# the share of whole-run GPU time those identities hold. The gap between what the tasks
# own and what their captured operands actually launched is the point of the fixture:
# moe_fused owns 18.88% and timed 0.10%.
DELIVERED = [
    ("moe_fused", ["k001", "k002"], 18.88104, 0.10290),
    ("cp_allgather_vec", ["k006", "k008"], 5.25909, 5.25909),
    ("mhc_fused_post_pre", ["k007", "k009"], 5.15420, 5.15420),
    ("allreduce_2stage_tp8", ["k004"], 5.13564, 5.13564),
    ("write_c128_prefill", ["k010", "k057"], 1.59040, 1.26710),
    ("cp_reduce_scatter_first_dim", ["k011"], 1.58780, 1.58780),
    ("quant_g128_fp32", ["k012"], 1.55610, 1.55610),
    ("topk_transform_paged_v2", ["k018"], 1.35100, 0.06800),
    ("mla_v4_prefill_opus", ["k020"], 1.24840, 1.24840),
    ("quant_mx_g32_e8m0", ["k022"], 1.15610, 1.15610),
]
BLOCKED = [("k003", "a Memcpy HtoD, not a kernel"),
           ("k005", "the image ships only a precompiled .co"),
           ("k014", "DECODE-only, and all 64 capture records are PREFILL"),
           ("k015", "precompiled RCCL")]
DECLARED = round(sum(row[2] for row in DELIVERED), 5)
ACHIEVED = round(sum(row[3] for row in DELIVERED), 5)


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))


def standalone(root, policy=POLICY, selector_args=None):
    """A skeleton of the hand-driven delivery: every file the slot reads, no captures."""
    out = root / "profile_out"
    targets = [{"id": ident, "overall_pct": share / len(idents)}
               for _, idents, share, _ in DELIVERED for ident in idents]
    targets += [{"id": ident, "overall_pct": 3.46732} for ident, _ in BLOCKED]
    write(out / "selection_targets.json", {"policy": policy, "targets": targets})
    write(out / "kernel_ranking.json", {
        "ranking": [{"id": "k%03d" % n} for n in range(1, 323)],
        "denominators": {"total_device_ms": 1831204.7},
        "selection_scope": {"selector": {"args": selector_args or
                                         list(SELECTOR_ARGS)}},
        "candidate_inventory": [{"id": "k%03d" % n} for n in range(1, 61)],
        "coverage_summary": {"selected_pct_gpu_time": 61.08186}})
    identities = [{"id": ident, "task": task, "status": "delivered",
                   "declared_pct_gpu_time": share / len(idents),
                   "achieved_pct_gpu_time": achieved / len(idents)}
                  for task, idents, share, achieved in DELIVERED for ident in idents]
    identities += [{"id": ident, "status": "blocked", "why": why} for ident, why in BLOCKED]
    write(out / "results.json", {
        "identities": identities,
        "tasks": [{"task": task, "gates_passed": 7} for task, _, _, _ in DELIVERED],
        "coverage": {"delivered_declared_pct_gpu_time": DECLARED,
                     "delivered_achieved_pct_gpu_time": ACHIEVED,
                     "blocked_pct_gpu_time": 13.86927}})
    # The plan is a mid-run snapshot and is deliberately stale here, exactly as in the
    # real delivery: it still names a task that results.json later split in two.
    write(out / "task_plan.json", {"tasks": [
        {"task": "indexer_logits_topk" if task == "topk_transform_paged_v2" else task,
         "selected_device_kernels": [{"device_kernel": i} for i in idents]}
        for task, idents, _, _ in DELIVERED]})
    for task, _, _, _ in DELIVERED:
        td = out / "tasks" / task
        (td / "kernel_src").mkdir(parents=True)
        for name in ("unittest.py", "cases.py", "meta.json"):
            (td / name).write_text("{}")
        write(td / "run/ut_result.json", {"ok": True, "failed_gates": [],
                                          "gates": {"correctness": True}})
        write(td / "run/RETURN.json", {"ok": True, "short_name": task})
    return out


def finalize(root, out, params=None, task_limit=8):
    config = root / "config.json"
    write(config, {"stages": {"profile": {"params": dict(CONFIGURED)},
                              "optimize": {"params": {"task_limit": task_limit}}}})
    params_file = root / "params.json"
    write(params_file, dict(CONFIGURED) if params is None else params)
    env = {**os.environ, "OUT": str(out), "CONFIG": str(config),
           "PARAMS": str(params_file), "STAGE": "profile",
           "STAGE_SKILL": "kernel-extract"}
    proc = subprocess.run([sys.executable, str(FINALIZE)], env=env,
                          capture_output=True, text=True)
    result = json.loads((out / "result.json").read_text())
    return result, proc


def test_the_hand_driven_delivery_passes_unchanged(tmp_path):
    """Ten tasks, an eight-GPU campaign, and no complaint about either."""
    out = standalone(tmp_path)
    result, proc = finalize(tmp_path, out)
    assert result["status"] == "ok", result["notes"] + proc.stderr
    m = result["metrics"]
    assert m["tasks_built"] == 10
    assert m["tasks_capacity_deferred"] == 2 and m["tasks_within_capacity"] == 8
    owned = sum(len(idents) for _, idents, _, _ in DELIVERED)
    assert m["identities_owned_by_tasks"] == owned == 14
    assert m["identities_blocked"] == len(BLOCKED) == 4
    assert m["targets_selected"] == owned + len(BLOCKED)
    assert m["tasks_ut_passed_all_gates"] == 10 and m["tasks_delivery_gate_passed"] == 10
    assert m["declared_pct_gpu_time"] == DECLARED == 42.91977
    assert m["achieved_pct_gpu_time"] == ACHIEVED == 22.53533
    assert m["census_denominator_source"] == "kernel_ranking.json:denominators.total_device_ms"
    assert all(m["acceptance_" + claim] for claim in
               ("inventory_completion", "task_completion", "gpu_ut_status",
                "delivery_consistency"))


def test_capacity_is_a_note_not_a_defect(tmp_path):
    out = standalone(tmp_path)
    result, _ = finalize(tmp_path, out)
    assert "task_limit=8" in result["notes"] and "later wave" in result["notes"]
    assert len(list((out / "tasks").iterdir())) == 10


def test_a_census_in_a_separate_file_is_equally_acceptable(tmp_path):
    """A skill that writes its own census and one that reuses a prior one both pass."""
    out = standalone(tmp_path)
    ranking = json.loads((out / "kernel_ranking.json").read_text())
    ranking.pop("denominators")
    write(out / "kernel_ranking.json", ranking)
    write(out / "kernel_workload.json", {"total_gpu_time_ms": 1831204.7})
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "ok", result["notes"]
    assert result["metrics"]["census_denominator_source"] == \
        "kernel_workload.json:total_gpu_time_ms"


def test_a_delivery_whose_shares_have_no_denominator_is_blocked(tmp_path):
    out = standalone(tmp_path)
    ranking = json.loads((out / "kernel_ranking.json").read_text())
    ranking.pop("denominators")
    write(out / "kernel_ranking.json", ranking)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked"
    assert "denominator" in result["notes"]


def test_the_run_that_silently_used_the_skill_default_is_now_rejected(tmp_path):
    """The regression this file exists for: configured Top 10 / >0.5%, ran Top 5 / >3%.

    That run was finalized `ok`. Both pairs of numbers must appear in the notes, because
    the next person has to see which one was asked for and which one happened."""
    out = standalone(tmp_path, policy={"top_k_per_phase": 5, "overall_pct_gt": 3,
                                       "phases": ["PREFILL", "DECODE"]},
                     selector_args=["--top-k", "5", "--overall-pct-gt", "3"])
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "fail"
    notes = result["notes"]
    assert "top 5" in notes and "> 3" in notes
    assert "top 10" in notes and "> 0.5" in notes


def test_a_target_list_with_no_policy_cannot_be_attributed(tmp_path):
    out = standalone(tmp_path)
    selection = json.loads((out / "selection_targets.json").read_text())
    selection.pop("policy")
    write(out / "selection_targets.json", selection)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked" and "policy" in result["notes"]


def test_the_ranking_may_not_disagree_with_the_target_list_about_scope(tmp_path):
    out = standalone(tmp_path, selector_args=["--top-k", "5", "--overall-pct-gt", "3",
                                            "--phases", "PREFILL", "DECODE"])
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "fail" and "kernel_ranking.json" in result["notes"]


def set_scope_witness(out, witness, *, phases=None, threshold=None, omit_phases=False):
    if witness == "policy":
        path = out / "selection_targets.json"
        document = json.loads(path.read_text())
        if phases is not None:
            document["policy"]["phases"] = phases
        if threshold is not None:
            document["policy"]["overall_pct_gt"] = threshold
        if omit_phases:
            document["policy"].pop("phases")
    else:
        arguments = ["--top-k", "10", "--overall-pct-gt",
                     str(0.5 if threshold is None else threshold)]
        if not omit_phases:
            arguments += ["--phases", *(POLICY["phases"] if phases is None else phases)]
        path = out / ("kernel_ranking.json" if witness == "ranking"
                      else "selection_targets.json")
        document = json.loads(path.read_text())
        if witness == "ranking":
            document["selection_scope"]["selector"]["args"] = arguments
        else:
            document["selection_invocation"] = {"arguments": arguments}
    write(path, document)


@pytest.mark.parametrize("witness", ["policy", "ranking", "invocation"])
def test_every_scope_witness_must_match_configured_phases(tmp_path, witness):
    out = standalone(tmp_path)
    set_scope_witness(out, witness, phases=["PREFILL", "DRAFT"])
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "fail"
    assert "DRAFT" in result["notes"] and "DECODE" in result["notes"]


@pytest.mark.parametrize("witness", ["policy", "ranking", "invocation"])
def test_missing_phase_receipt_cannot_claim_configured_scope(tmp_path, witness):
    out = standalone(tmp_path)
    set_scope_witness(out, witness, omit_phases=True)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == ("blocked" if witness == "policy" else "fail")
    assert "phases" in result["notes"]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("witness", ["policy", "ranking", "invocation"])
def test_nonfinite_scope_receipts_never_match_finite_configuration(tmp_path, witness, threshold):
    out = standalone(tmp_path)
    set_scope_witness(out, witness, threshold=threshold)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "fail" and "finite" in result["notes"]


def test_recorded_phase_arguments_end_at_the_next_option(tmp_path):
    out = standalone(tmp_path, selector_args=["--phases", "PREFILL", "DECODE",
                                            "--output", "targets.json",
                                            "--top-k", "10", "--overall-pct-gt", "0.5"])
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "ok", result["notes"]


@pytest.mark.parametrize("arguments", [None, [], "--phases PREFILL DECODE",
                                      SELECTOR_ARGS + ["--top-k", "10"]])
def test_present_invocation_must_be_a_complete_unambiguous_receipt(tmp_path, arguments):
    out = standalone(tmp_path)
    path = out / "selection_targets.json"
    document = json.loads(path.read_text())
    document["selection_invocation"] = {"arguments": arguments}
    write(path, document)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "fail" and "selection_invocation" in result["notes"]


def test_descriptive_selector_keeps_policy_authoritative_without_an_argv_receipt(tmp_path):
    out = standalone(tmp_path)
    path = out / "kernel_ranking.json"
    document = json.loads(path.read_text())
    document["selection_scope"]["selector"] = "alternative implementation; scope in policy"
    write(path, document)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "ok", result["notes"]


def test_a_configured_scope_the_selector_cannot_take_fails_before_anything_else(tmp_path):
    out = standalone(tmp_path)
    result, _ = finalize(tmp_path, out, params={"selection_top_k": 0})
    assert result["status"] == "fail"
    assert "selection_top_k" in result["notes"]


@pytest.mark.parametrize("damage, expected", [
    ("missing", "no run/ut_result.json"),
    ("failed", "failed_gates"),
])
def test_a_task_whose_gates_did_not_pass_is_blocked_by_name(tmp_path, damage, expected):
    out = standalone(tmp_path)
    receipt = out / "tasks/quant_g128_fp32/run/ut_result.json"
    if damage == "missing":
        receipt.unlink()
    else:
        write(receipt, {"ok": False, "failed_gates": ["correctness"]})
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked"
    assert "quant_g128_fp32" in result["notes"] and expected in result["notes"]
    assert result["metrics"]["tasks_ut_passed_all_gates"] == 9
    assert not result["metrics"]["acceptance_gpu_ut_status"]
    assert result["metrics"]["acceptance_delivery_consistency"]


def test_the_delivery_gate_receipt_is_its_own_claim(tmp_path):
    """`RETURN.json` exists only when the gate reported ok, so absence is a failed gate."""
    out = standalone(tmp_path)
    (out / "tasks/mla_v4_prefill_opus/run/RETURN.json").unlink()
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked"
    assert "mla_v4_prefill_opus" in result["notes"]
    assert result["metrics"]["tasks_delivery_gate_passed"] == 9
    assert result["metrics"]["acceptance_gpu_ut_status"]
    assert not result["metrics"]["acceptance_delivery_consistency"]


def test_another_tasks_delivery_record_does_not_count(tmp_path):
    out = standalone(tmp_path)
    write(out / "tasks/mla_v4_prefill_opus/run/RETURN.json",
          {"ok": True, "short_name": "moe_fused"})
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked" and "moe_fused" in result["notes"]


def test_a_delivery_that_never_timed_what_it_owns_must_say_so(tmp_path):
    out = standalone(tmp_path)
    results = json.loads((out / "results.json").read_text())
    results["coverage"].pop("delivered_achieved_pct_gpu_time")
    for row in results["identities"]:
        row.pop("achieved_pct_gpu_time", None)
    write(out / "results.json", results)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked" and "achieved" in result["notes"]
    assert result["metrics"]["achieved_pct_gpu_time"] is None


def test_achieved_coverage_cannot_exceed_what_the_tasks_own(tmp_path):
    out = standalone(tmp_path)
    results = json.loads((out / "results.json").read_text())
    results["coverage"]["delivered_achieved_pct_gpu_time"] = DECLARED + 1
    write(out / "results.json", results)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked"
    assert "exceeds declared" in result["notes"]


def test_a_blocked_identity_stays_on_the_books(tmp_path):
    """Dropping a blocker is not the same as covering it, and must not read as coverage."""
    out = standalone(tmp_path)
    results = json.loads((out / "results.json").read_text())
    results["identities"] = [r for r in results["identities"] if r["id"] != "k014"]
    write(out / "results.json", results)
    plan = json.loads((out / "task_plan.json").read_text())
    write(out / "task_plan.json", plan)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked" and "k014" in result["notes"]


def test_a_task_the_record_names_but_never_built_is_not_a_delivery(tmp_path):
    out = standalone(tmp_path)
    results = json.loads((out / "results.json").read_text())
    results["tasks"].append({"task": "never_built"})
    write(out / "results.json", results)
    result, _ = finalize(tmp_path, out)
    assert result["status"] == "blocked" and "never_built" in result["notes"]
