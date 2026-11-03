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
                       capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def labels(out, authors=False):
    # author agents run in parallel with each other: compare them separately
    return [c["label"] for c in out["calls"] if authors or not c["label"].startswith("kf:author ")]


def rejected(r):
    return [(x["id"], x["stage"]) for x in r["rejected"]]


def test_all_winners_go_in_together_on_one_ab():
    out = run("accept")
    r = out["result"]
    # the triage agent picks c1, c1s, c2 and c3; c3 has no seam so it is never touched. Every kernel is
    # written before anything is applied, the select agent picks from the UT results, then ONE integrator
    # A/B of the whole set -- no per-kernel screen while the stack passes, no separate director validate.
    assert labels(out) == ["kf:director:setup", "kf:profiler", "kf:analyst", "kf:triage",
                           "kf:extract c1", "kf:extract c1s", "kf:extract c2",
                           "kf:select", "kf:integrate kf_stack"]
    setup = out["calls"][0]["prompt"]
    assert "MEASUREMENT_MODE: isolated_server" in setup and "REPLICAS: 1" in setup     # no warmup round
    by = {c["label"]: c["prompt"] for c in out["calls"]}
    assert "no limit on how many" in by["kf:triage"] and "no minimum gain" in by["kf:triage"]
    assert "no minimum speedup" in by["kf:select"] and '"isolated_speedup":1.6' in by["kf:select"]
    for c in out["calls"]:
        if c["label"] not in ("kf:analyst", "kf:triage", "kf:select") and not c["label"].startswith("kf:author "):   # stock GEAK role files
            m = re.search(r"Read (\S+/roles/(\w+)\.md)", c["prompt"])
            assert m and Path(m.group(1)).is_file(), c["label"]
    analyst = out["calls"][2]["prompt"]
    assert "links: /prompts/p.txt, /prompts/deploy.txt" in analyst and "kf_trace_digest.py" in analyst and "fallback (id" in analyst
    assert f"TRACE_DIR: {KF}/profile" in analyst
    assert "Live model code" in analyst and "this server never loads" in analyst      # seams on the live implementation
    # the fused kernels are written by plain author agents, not the GEAK kernel layer
    assert len(out["workflows"]) == 3 and all("kernel_lane" not in w["task"] for w in out["workflows"])
    au = next(c["prompt"] for c in out["calls"] if c["label"] == "kf:author c1")
    assert f"TASK_DIR={KF}/kernels/c1_task" in au and f"WORK_DIR={KF}/kernels/c1/work" in au
    assert "never edit TASK_DIR" in au and f"{KF}/kernels/c1/final_patch.diff" in au
    # deliverables: fused kernel module, graph-replay unit test, lazy overlay with an ENGAGED banner
    assert f"{KF}/kernels/c1/overlay/sitecustomize.py" in au and "[kf-c1] ENGAGED" in au and "graph replay" in au
    assert "shapes the server actually runs" in au and "weighted by served share" in au
    assert "whether or not it beats stock" in au
    assert "import only the standard library" in au
    assert sorted(l for l in labels(out, authors=True) if l.startswith("kf:author ")) == ["kf:author c1", "kf:author c1s", "kf:author c2"]
    integ = by["kf:integrate kf_stack"]
    assert "Apply ALL 2 fused kernels" in integ and "ONE e2e A/B" in integ and f"{KF}/overlay/cand_kf_stack" in integ
    for k in ("c1", "c2"):
        assert f'"overlay_dir":"{KF}/kernels/{k}/overlay"' in integ and f"[kf-{k}] ENGAGED" in integ
    assert f'"overlay_dir":"{KF}/kernels/c1s/overlay"' not in integ        # not selected
    assert "MEASUREMENT_PURPOSE: validation" in integ and "MEASUREMENT_MODE: isolated_server" in integ and "REPLICAS: 1" in integ
    # fused kernels are held to task accuracy (the integrator's ACCURACY_GATE), not byte-exact greedy parity
    assert "ACCURACY_GATE: gsm8k" in integ and "GSM8K_EVAL_SCRIPT: " in integ and "ACCURACY_TOL: 0.01" in integ
    assert 'starts with "correctness:"' in integ
    # synthetic acceptance stays on for throughput, off for every correctness check
    assert "CORRECTNESS PROBES RUN WITHOUT SIMULATED ACCEPTANCE" in integ
    assert "CORRECTNESS_UNSET_ENV: SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD" in integ
    assert "env -u SGLANG_SIMULATE_ACC_LEN -u SGLANG_SIMULATE_ACC_METHOD" in integ
    assert "CURRENT_ENV: SGLANG_USE_AITER=1 SGLANG_SIMULATE_ACC_LEN=4.05" in integ
    # operator constraints reach every agent
    assert all("Do not read /answers." in c["prompt"] for c in out["calls"])
    assert r["status"] == "accepted" and r["final_overlay"] == f"{KF}/overlay/cand_kf_stack"
    assert r["throughput_speedup"] == pytest.approx(1.025)          # cand_med / ref_med of the one A/B
    assert sorted(a["id"] for a in r["accepted"]) == ["c1", "c2"]
    assert ("c1s", "select") in rejected(r)


def test_fallback_goes_in_when_the_primary_has_no_isolated_win():
    out = run("fallback")
    integ = next(c["prompt"] for c in out["calls"] if c["label"] == "kf:integrate kf_stack")
    assert f'"overlay_dir":"{KF}/kernels/c1s/overlay"' in integ and f'"overlay_dir":"{KF}/kernels/c1/overlay"' not in integ
    r = out["result"]
    assert r["status"] == "accepted" and [a["id"] for a in r["accepted"]] == ["c1s"]


def test_a_slower_combined_stack_is_not_handed_on():
    r = run("reject")["result"]
    assert r["status"] == "no_gain" and r["final_overlay"] == "" and not r["accepted"]
    assert ("c1", "apply") in rejected(r)


def test_incomplete_ab_is_finished_before_deciding():
    out = run("retry")
    assert labels(out)[-2:] == ["kf:integrate kf_stack", "kf:integrate kf_stack (finish 1)"]
    assert out["result"]["status"] == "accepted"


def test_a_stack_gate_that_is_not_faster_is_not_handed_on():
    r = run("stack")["result"]
    assert r["validation"]["gate"] == "stack" and r["status"] == "no_gain" and r["final_overlay"] == ""


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
    assert ls[:4] == ["kf:director:setup", "kf:kernel-extract trace", "kf:analyst", "kf:triage"]
    assert ls[-2:] == ["kf:select", "kf:integrate kf_stack"]
    # one prepare agent per region (parallel), a failed region retried once with its blocker
    assert sorted(ls[4:-2]) == sorted(["kf:kernel-extract prepare c1", "kf:kernel-extract prepare c1s",
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


def test_a_measured_baseline_is_reused_not_remeasured():
    out = run("reuse_baseline")
    assert labels(out)[:2] == ["kf:director:setup", "kf:analyst"]            # given traces: no capture either
    setup = out["calls"][0]["prompt"]
    assert "already measured in /old/kf/baseline" in setup and "launch no server and no bench" in setup
    assert "already measured" not in run("accept")["calls"][0]["prompt"]


def test_a_finished_topk_is_reused_and_uts_follow_the_fusion_contract():
    out = run("reuse_topk")
    # the given analysis arrives in args (kf_stage reads the file): no agent re-types it
    assert "kf:analyst" not in labels(out) and labels(out)[1:3] == ["kf:triage", "kf:extract c1"]
    assert out["result"]["topk_json"] == "/old/kf/analysis/topk.json"
    ext = [c["prompt"] for c in out["calls"] if c["label"].startswith("kf:extract")]
    assert ext and all("FUSION UT CONTRACT" in p and "BASELINE leg only" in p for p in ext)
    assert all("replaces the region (fewer launches" in w["task"] for w in out["workflows"])
    assert out["result"]["status"] == "accepted"


def test_agentx_prepare_carries_the_fusion_ut_contract():
    out = run("agentx")
    preps = [c["prompt"] for c in out["calls"] if c["label"].startswith("kf:kernel-extract prepare")]
    assert preps and all("FUSION UT CONTRACT" in p and "never the member kernels directly" in p for p in preps)


def test_a_stack_that_fails_accuracy_is_screened_kernel_by_kernel_then_re_run():
    out = run("stack_fail")
    assert labels(out)[-3:] == ["kf:integrate kf_stack", "kf:integrate screen", "kf:integrate kf_stack (screened)"]
    by = {c["label"]: c["prompt"] for c in out["calls"]}
    sc = by["kf:integrate screen"]
    assert "CORRECTNESS SCREEN ONLY" in sc and "no throughput legs" in sc and f"{KF}/overlay/screen_" in sc
    assert "CORRECTNESS PROBES RUN WITHOUT SIMULATED ACCEPTANCE" in sc and "ACCURACY_GATE: gsm8k" in sc
    integ = by["kf:integrate kf_stack (screened)"]
    assert "Apply ALL 1 fused kernels" in integ and f'"overlay_dir":"{KF}/kernels/c2/overlay"' not in integ
    r = out["result"]
    assert [a["id"] for a in r["accepted"]] == ["c1"] and ("c2", "screen") in rejected(r)


def test_the_select_agent_has_no_speedup_floor_but_overlapping_kernels_never_go_in_together():
    integ = next(c["prompt"] for c in run("select_slow")["calls"] if c["label"] == "kf:integrate kf_stack")
    assert f'"overlay_dir":"{KF}/kernels/c2/overlay"' in integ            # 0.95x in its UT, picked anyway
    out = run("select_conflict")
    integ = next(c["prompt"] for c in out["calls"] if c["label"] == "kf:integrate kf_stack")
    assert f'"overlay_dir":"{KF}/kernels/c1/overlay"' in integ and f'"overlay_dir":"{KF}/kernels/c1s/overlay"' not in integ
    assert ("c1s", "select") in rejected(out["result"])


def test_without_a_triage_answer_the_recommended_patterns_are_authored():
    out = run("triage_none")
    assert sorted(l for l in labels(out, authors=True) if l.startswith("kf:author ")) == ["kf:author c1", "kf:author c2"]


def test_finished_author_results_of_an_earlier_run_are_reused():
    out = run("reuse")
    assert "kf:author c1" not in labels(out, authors=True) and "kf:author c2" in labels(out, authors=True)
    integ = next(c["prompt"] for c in out["calls"] if c["label"] == "kf:integrate kf_stack")
    assert '"overlay_dir":"/old/c1/overlay"' in integ and '"code_patch":"old c1"' in integ


def test_a_correctness_reject_is_final_not_retried():
    out = run("parity_reject")
    assert "kf:integrate kf_stack (finish 1)" not in labels(out)
    # one kernel in the stack: the stack verdict is already its verdict -- no screen, no second A/B
    assert "kf:integrate screen" not in labels(out) and "kf:integrate kf_stack (screened)" not in labels(out)
    assert out["result"]["status"] == "no_gain"


def test_apply_steps_outlast_the_default_agent_timeout():
    # the combined A/B owns hours of serving legs: a 2 h guard would start a second agent on the same job
    src = (E2E_DIR / "kernel_fusion" / "kf_workflow.js").read_text()
    for label in ("label, schema: SCREEN_SCHEMA", "label, schema: INTEGRATE_SCHEMA"):
        assert re.search(re.escape(label) + r", timeoutMs: APPLY_TIMEOUT_MS", src), label
    assert re.search(r"APPLY_TIMEOUT_MS = .*6 \* 3600 \* 1000", src)


def test_a_seam_given_as_a_string_still_reaches_extraction_and_authoring():
    # c32 2026-11-02: the analyst returned seam as "pkg.mod:fn"; seamOf() only read seam.target_callable, so every
    # pattern the triage picked was dropped and the stage ended no_gain without writing a single kernel.
    out = run("string_seam")
    assert labels(out)[3:7] == ["kf:triage", "kf:extract c1", "kf:extract c1s", "kf:extract c2"]
    ext = {c["label"]: c["prompt"] for c in out["calls"] if c["label"].startswith("kf:extract")}
    assert "pkg.moe:MoE.forward" in ext["kf:extract c1"] and "pkg.linear:apply" in ext["kf:extract c2"]
    assert sorted(l for l in labels(out, authors=True) if l.startswith("kf:author ")) == \
        ["kf:author c1", "kf:author c1s", "kf:author c2"]
    assert out["result"]["status"] == "accepted"


def test_agentx_a_seam_given_as_a_string_still_reaches_the_ut_prepare():
    out = run("agentx_string_seam")
    preps = sorted(l for l in labels(out) if l.startswith("kf:kernel-extract prepare"))
    assert preps[:3] == ["kf:kernel-extract prepare c1", "kf:kernel-extract prepare c1s", "kf:kernel-extract prepare c1s (retry)"]
    assert any(l.startswith("kf:author ") for l in labels(out, authors=True))


def test_triage_picks_without_a_usable_seam_end_the_stage_as_an_error_not_no_gain():
    out = run("no_seam")
    r = out["result"]
    assert r["status"] == "error" and "seam" in r["reason"]
    assert not [l for l in labels(out, authors=True) if l.startswith(("kf:extract", "kf:author ", "kf:integrate"))]


def test_the_workflow_owns_the_output_format_and_the_schema_types_the_seam():
    out = run("accept")
    an = next(c for c in out["calls"] if c["label"] == "kf:analyst")
    p = an["prompt"]
    # the strategy is authoritative on method, not on output files / formats
    assert "NOT authoritative on output files or formats" in p and "an OBJECT, never a bare string" in p
    assert "a list of candidate ids (not a per-candidate flag)" in p
    cand = an["schema"]["properties"]["candidates"]["items"]
    assert cand["properties"]["seam"]["type"] == "object" and cand["properties"]["id"]["type"] == "string"
    assert cand["required"] == ["id"] and "seam" not in cand["required"]          # L3/L4 may have no seam
    skill = E2E_DIR / "knowledge" / "kernel_fusion_skills" / "fusion-analysis"
    text = "".join(p.read_text() for p in [skill / "SKILL.md", *sorted((skill / "references").glob("*.md"))])
    assert "out/" not in text and "fusion_candidates" not in text and "recommended=true" not in text
    assert "IMPLEMENTABLE_LEVELS: L1,L2" in p                                     # capability injected, not prose
    assert "knowledge/kernel_fusion_skills/fusion-analysis/SKILL.md" in (E2E_DIR / "kernel_fusion" / "kf_workflow.js").read_text()
    assert "reference files a SKILL.md among them links" in p


def test_a_combined_ab_starts_only_while_the_hard_limit_leaves_room_for_it():
    src = (E2E_DIR / "kernel_fusion" / "kf_workflow.js").read_text()
    assert re.search(r"hardMs = TIME_BUDGET_MS \* 1\.1 \+ 900000", src)           # kf_stage's hard limit
    assert "|| DEADLINE_HIT || AB_LATE)" in src and "!DEADLINE_HIT && !RERUN_LATE)" in src
    assert "stage deadline: less than ${AB_NEEDS_S}s left for the combined A/B" in src


def test_a_gain_inside_the_noise_band_is_not_handed_on():
    # c24 r2: one replica per side, +0.22% P90 -- indistinguishable from noise
    r = run("small_gain")["result"]
    assert r["status"] == "no_gain" and r["final_overlay"] == "" and not r["accepted"]
    assert r["validation"]["noise_band_pct"] == 0.5 and "within the noise band 0.5%" in r["validation"]["reason"]
    assert run("accept")["result"]["status"] == "accepted"                          # +2.5% clears it


def test_a_hand_assembled_prepare_delivery_still_counts():
    # c28 E10: node was not on PATH, the skill could not run its packer and put the passing UT on the entry itself
    out = run("agentx_flat")
    assert "kf:kernel-extract prepare c1 (retry)" not in labels(out)
    assert f"{KF}/tasks/c1" in [w["kernel_path"] for w in out["workflows"]]
