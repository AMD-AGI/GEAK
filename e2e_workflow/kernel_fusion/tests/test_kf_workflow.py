"""kf_workflow.js control flow under scripted agents (needs node: PATH or GEAK_NODE_BIN)."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
E2E_DIR = HERE.parents[1]
NODE = os.environ.get("GEAK_NODE_BIN") or shutil.which("node")
pytestmark = pytest.mark.skipif(not NODE, reason="node not available (set GEAK_NODE_BIN)")
KF = "/tmp/kf_mock/kf"


def run(scenario):
    p = subprocess.run([NODE, str(HERE / "kf_workflow_mockrun.mjs"), scenario, str(E2E_DIR)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def labels(out):
    return [c["label"] for c in out["calls"]]


def rejected(r):
    return [(x["id"], x["stage"]) for x in r["rejected"]]


def test_accept_path_reuses_geak_roles_and_confirms_the_stack():
    out = run("accept")
    r = out["result"]
    # c1 is recommended at difficulty 3, so its smaller exclusive alternative c1s is authored too;
    # c3 is never touched (not recommended, no seam).
    assert labels(out) == ["kf:director:setup", "kf:profiler", "kf:analyst",
                           "kf:extract c1", "kf:extract c1s", "kf:extract c2",
                           "kf:integrate c1", "kf:director:validate"]
    for c in out["calls"]:
        if c["label"] != "kf:analyst":   # every other agent is a stock GEAK role file that exists
            m = re.search(r"Read (\S+/roles/(\w+)\.md)", c["prompt"])
            assert m and Path(m.group(1)).is_file(), c["label"]
    analyst = out["calls"][2]["prompt"]
    assert "Read every one of these files in full: /prompts/p.txt, /prompts/deploy.txt" in analyst and "kf_trace_digest.py" in analyst and "fallback (id" in analyst
    assert f"TRACE_DIR: {KF}/profile" in analyst
    assert all(w["scriptPath"].endswith("kernel_workflow/kernel_lane.js") and w["mode"] == "author"
               for w in out["workflows"]) and len(out["workflows"]) == 3
    integ = out["calls"][6]["prompt"]
    assert '"winner_kind":"authored"' in integ and '"code_patch":"diff c1"' in integ
    validate = out["calls"][7]["prompt"]
    assert f"FINAL_OVERLAY: {KF}/overlay/cand_accepted" in validate and "MEASUREMENT_MODE: isolated_server" in validate
    assert "REPLICAS: 2" in validate
    assert r["status"] == "accepted" and r["final_overlay"] == f"{KF}/overlay/cand_accepted"
    # synthetic acceptance stays on for throughput, off for every correctness check
    for p in (integ, validate):    # e2e_workflow.js's correctness rule, fed by run_e2e's correctness_unset_env
        assert "CORRECTNESS PROBES RUN WITHOUT SIMULATED ACCEPTANCE" in p
        assert "CORRECTNESS_UNSET_ENV: SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD" in p
        assert "env -u SGLANG_SIMULATE_ACC_LEN -u SGLANG_SIMULATE_ACC_METHOD" in p
    assert "CURRENT_ENV: SGLANG_USE_AITER=1 SGLANG_SIMULATE_ACC_LEN=4.05" in integ
    # operator constraints reach every agent
    assert all("Do not read /answers." in c["prompt"] for c in out["calls"])
    assert r["throughput_speedup"] == pytest.approx(1.025)          # the director's combined number
    assert [a["id"] for a in r["accepted"]] == ["c1"]
    assert rejected(r) == [("c2", "author")]


def test_fallback_is_applied_when_the_primary_has_no_isolated_win():
    out = run("fallback")
    assert "kf:integrate c1" not in labels(out) and "kf:integrate c1s" in labels(out)
    r = out["result"]
    assert r["status"] == "accepted" and [(a["id"], a["slot"]) for a in r["accepted"]] == [("c1s", "c1")]


def test_fallback_is_applied_when_the_primary_fails_the_ab():
    out = run("reject")
    assert labels(out)[-3:] == ["kf:integrate c1", "kf:integrate c1s", "kf:director:validate"]
    r = out["result"]
    assert [a["id"] for a in r["accepted"]] == ["c1s"] and ("c1", "apply") in rejected(r)


def test_incomplete_ab_is_finished_before_deciding():
    out = run("retry")
    assert labels(out)[-3:] == ["kf:integrate c1", "kf:integrate c1 (finish 1)", "kf:director:validate"]
    assert out["result"]["status"] == "accepted"


def test_stack_is_carried_but_needs_the_combined_confirmation():
    r = run("stack")["result"]
    assert [a["gate"] for a in r["accepted"]] == ["stack"]
    assert r["validation"]["validation_status"] == "validated_no_win"
    assert r["status"] == "no_gain" and r["final_overlay"] == "" and r["throughput_speedup"] == 1


def test_extraction_without_frozen_baseline_is_not_authored():
    out = run("unfrozen")
    assert sorted(w["kernel_path"] for w in out["workflows"]) == [f"{KF}/kernels/c1s_task", f"{KF}/kernels/c2_task"]
    r = out["result"]
    assert ("c1", "extract") in rejected(r) and [a["id"] for a in r["accepted"]] == ["c1s"]


def test_analysis_only_with_given_traces_needs_no_server():
    out = run("analysis_only")
    assert labels(out) == ["kf:analyst"]
    assert "TRACE_DIR: /traces/x" in out["calls"][0]["prompt"]
    assert out["result"]["status"] == "analysis_only" and out["workflows"] == []
    files = out["result"]["analysis_files"]
    assert files == {f"{KF}/analysis/REPORT.md": "# report", f"{KF}/analysis/MODEL_NOTES.md": "# notes"}
    assert "report_md" in out["calls"][0]["prompt"]


def test_no_trace_is_retried_then_stops_the_stage():
    out = run("no_trace")
    assert labels(out)[:2] == ["kf:director:setup", "kf:profiler"] and "kf:analyst" not in labels(out)
    assert labels(out).count("kf:profiler (retry)") == 3          # safeAgent's tries on a null result
    r = out["result"]
    assert r["status"] == "error" and "no Top-N" in r["reason"] and r["final_overlay"] == ""


def test_no_trace_once_then_recaptured():
    out = run("trace_retry")
    assert labels(out)[:4] == ["kf:director:setup", "kf:profiler", "kf:profiler (retry)", "kf:analyst"]
    assert out["result"]["status"] == "accepted"


def test_agents_are_told_not_to_end_their_turn_on_a_running_job():
    out = run("accept")
    assert all("LONG MEASUREMENTS OUTLIVE YOUR TOOL CALL" in c["prompt"] and "setsid nohup" in c["prompt"]
               and "ENDING YOUR TURN IS YOUR FINAL ANSWER" in c["prompt"]
               and "THE GEAK REPO IS READ-ONLY" in c["prompt"] for c in out["calls"])


def test_agentx_uses_the_kernel_extract_skill_for_trace_and_uts():
    out = run("agentx")
    ls = labels(out)
    assert ls[:3] == ["kf:director:setup", "kf:kernel-extract trace", "kf:analyst"]
    assert ls[-2:] == ["kf:integrate c1", "kf:director:validate"]
    # one prepare agent per region (parallel), a failed region retried once with its blocker
    assert sorted(ls[3:-2]) == sorted(["kf:kernel-extract prepare c1", "kf:kernel-extract prepare c1s",
                                       "kf:kernel-extract prepare c1s (retry)", "kf:kernel-extract prepare c2",
                                       "kf:kernel-extract prepare c2 (retry)"])
    by = {c["label"]: c["prompt"] for c in out["calls"]}
    tr, an = by["kf:kernel-extract trace"], by["kf:analyst"]
    for p in (tr, by["kf:kernel-extract prepare c1"]):     # main's standalone-extraction instruction, not the legacy role
        assert "kernel_extraction_skills/kernel-extract/SKILL.md" in p and "Do not load roles/kernel_extractor.md" in p
        assert "SKILL_ROOT: " in p and "BASELINE_LOG: /tmp/kf_mock/kf/baseline/server.log" in p
    assert "serving_probe.py" in tr and "Do NOT select targets" in tr
    assert f"TRACE_DIR: {KF}/kernel_extraction/round_kf/capture/traces" in an
    assert f"WORKLOAD_SUMMARY: {KF}/kernel_workload.json" in an                  # the skill's weights
    for t in ("c1", "c1s", "c2"):
        p = by[f"kf:kernel-extract prepare {t}"]
        assert f'"task":"{t}"' in p and "joint_ut.phases" in p and "do not defer or stop early for budget" in p
        assert f"EVIDENCE_DIR: {KF}/kernel_extraction/round_kf_{t}" in p and f"PREVIOUS_WORKLOAD_JSON: {KF}/kernel_workload.json" in p
    assert '"task":"c1s"' not in by["kf:kernel-extract prepare c1"]
    assert "operands not reconstructed" in by["kf:kernel-extract prepare c2 (retry)"]
    # c2's UT did not pass -> never authored; c1 / c1s authored on the prepared task dirs
    assert sorted(w["kernel_path"] for w in out["workflows"]) == [f"{KF}/tasks/c1", f"{KF}/tasks/c1s"]
    r = out["result"]
    assert r["status"] == "accepted" and ("c2", "extract") in rejected(r)


def test_agentx_without_a_verified_trace_stops_before_topk():
    out = run("agentx_no_trace")
    assert labels(out)[:3] == ["kf:director:setup", "kf:kernel-extract trace", "kf:kernel-extract trace (retry)"]
    assert "kf:analyst" not in labels(out) and out["result"]["status"] == "error"


def test_a_baseline_left_running_detached_is_waited_for():
    out = run("detached_baseline")
    assert labels(out)[:4] == ["kf:director:setup", "kf:baseline wait 1", "kf:baseline wait 2", "kf:profiler"]
    wait = out["calls"][1]["prompt"]
    assert "start nothing" in wait and f"{KF}/baseline/bench_summary.json" in wait
    assert out["result"]["baseline_throughput_tok_s"] == 286.9


def test_setup_failure_returns_cleanly():
    r = run("setup_fail")["result"]
    assert r["status"] == "error" and r["final_overlay"] == ""


def test_script_uses_no_sandbox_forbidden_construct():
    # the native Workflow runtime rejects these (interface/runtime/engine/conformance.mjs A-forbidden)
    src = (E2E_DIR / "kernel_fusion" / "kf_workflow.js").read_text()
    assert not re.search(r"\bDate\.now\b|\bMath\.random\b|\bnew\s+Date\b|\bprocess\.|\brequire\s*\(|\bimport\s*\(", src)
