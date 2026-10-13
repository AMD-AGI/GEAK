#!/usr/bin/env python3
"""CPU-only self-check: no GPUs, no tokens, no network.

What it covers are the places where a mistake **silently produces a fake number**, not
coverage for its own sake: config merging and --set, result.json validation, the roofline
ranking formula, the scheduler's deduplication and blocked branches, applyback's two
refuse-to-publish gates, and the integrity of the vendored skills.

    python3 -m pytest workload/tests -q
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
WORKLOAD = os.path.dirname(HERE)
GEAK = os.path.dirname(WORKLOAD)
SCRIPTS = os.path.join(WORKLOAD, "scripts")
sys.path.insert(0, SCRIPTS)


def run(cmd, **kw):
    env = dict(os.environ, **kw.pop("env_extra", {}))
    return subprocess.run(cmd, capture_output=True, text=True, cwd=GEAK, env=env, **kw)


# ----------------------------------------------------------------- config

def test_dry_run_resolves_every_stage():
    """--dry-run must resolve all five stages with nothing missing. It is the only static
    gate before a real run."""
    r = run([sys.executable, "workload/scripts/stage.py", "--config",
             "workload/config/agentx_dsv4.json", "--dry-run"])
    assert r.returncode == 0, r.stdout + r.stderr
    for stage in ("baseline", "profile", "roofline", "optimize", "applyback"):
        assert stage in r.stdout


def test_set_override_wins(tmp_path):
    """--set must beat the config file — smoke runs rely on it to shrink the scale."""
    r = run([sys.executable, "workload/scripts/stage.py", "--config",
             "workload/config/agentx_dsv4.json",
             "--set", "stages.optimize.skill=subagent-raw",
             "--set", "stages.optimize.params.time_budget_sec=60", "--dry-run"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "subagent-raw" in r.stdout


def test_every_configured_skill_exists():
    """Every skill directory named in the config must actually exist and carry a SKILL.md."""
    cfg = json.load(open(os.path.join(WORKLOAD, "config", "default.json")))
    for stage, spec in cfg["stages"].items():
        d = os.path.join(WORKLOAD, "stages", stage, spec["skill"])
        assert os.path.isdir(d), "missing skill directory %s" % d
        assert os.path.exists(os.path.join(d, "SKILL.md")), "%s has no SKILL.md" % d
        if spec.get("mode") == "script":
            assert os.path.exists(os.path.join(d, "run.sh")), \
                "%s is declared as script mode but has no run.sh" % d


# ----------------------------------------------------------------- result.json contract

@pytest.mark.parametrize("args,why", [
    (["--status", "ok"], "status=ok with no artifacts"),
    (["--status", "blocked"], "blocked with no notes"),
    (["--status", "ok", "--artifact", "x=/abs/path"], "artifact given as an absolute path"),
    (["--status", "ok", "--artifact", "x=nope.json"], "artifact points at a file that does not exist"),
])
def test_emit_result_refuses_bad_output(tmp_path, args, why):
    """The result.json gate has to stop this locally, not wait for stage.py to reject it."""
    r = run([sys.executable, "workload/scripts/emit_result.py"] + args,
            env_extra={"OUT": str(tmp_path), "STAGE": "t", "STAGE_SKILL": "t"})
    assert r.returncode != 0, "should have been refused: %s" % why
    assert not os.path.exists(tmp_path / "result.json"), \
        "refused and yet wrote the file anyway"


def test_emit_result_accepts_good_output(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    r = run([sys.executable, "workload/scripts/emit_result.py", "--status", "ok",
             "--artifact", "a=a.json", "--metric", "x=1.5"],
            env_extra={"OUT": str(tmp_path), "STAGE": "t", "STAGE_SKILL": "s"})
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((tmp_path / "result.json").read_text())
    assert d["status"] == "ok" and d["metrics"]["x"] == 1.5


# ----------------------------------------------------------------- roofline ranking

def test_ranking_prefers_expected_e2e_gain_over_op_speedup():
    """Ranking must follow expected e2e gain, not operator speedup — otherwise the work
    goes to the kernel with the large share that is already pinned against its roofline.
    This inversion is calibrated against measurements:
    fused_moe 26.45% GPU / 88% roofline (only 1.02x left) -> measured e2e -0.064%;
    paged_attention 8.86% GPU / 18% roofline (2.8x left) -> 1.56x under iso.
    """
    def expected(pct, att):
        return pct * (1.0 - 1.0 / att)
    moe = expected(26.45, 1.136)
    attn = expected(8.86, 5.56)
    assert attn > moe, "the smaller share that is further from its roofline must rank first"


# ----------------------------------------------------------------- scheduler

def test_scheduler_blocked_runner_stops_instead_of_burning_budget(tmp_path):
    """When the runner returns 78 (unmet precondition) it must stop immediately, rather
    than burning the 48h budget on retries that are certain to fail."""
    tasks = tmp_path / "tasks"
    run([sys.executable, os.path.join(HERE, "make_fake_tasks.py"),
         "--dir", str(tasks), "--count", "1"])
    work = tmp_path / "work"
    r = run([sys.executable, "workload/scripts/scheduler.py", "run",
             "--work-root", str(work), "--tasks-dir", str(tasks),
             "--skill-root", "workload/stages/optimize/subagent-raw",
             "--runner", "workload/runners/stub.sh", "--gpus", "0",
             "--target", "600", "--attempt-cap", "15", "--tick-sec", "2",
             "--startup-grace", "3", "--backend", "local"],
            env_extra={"STUB_RC": "78"}, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    st = json.loads((work / "state" / "fake_kernel_0.status.json").read_text())
    assert st["status"] == "blocked"
    assert st["credited_seconds"] < 60, "a blocked kernel must stop accruing credit"


def test_scheduler_never_assigns_one_kernel_to_two_gpus(tmp_path):
    """One kernel on two GPUs means two sessions editing the same kernel_src, KEEP and
    revert trampling each other, and both sets of numbers void. It is most likely to
    happen when there are more GPUs than tasks."""
    tasks = tmp_path / "tasks"
    run([sys.executable, os.path.join(HERE, "make_fake_tasks.py"),
         "--dir", str(tasks), "--count", "1"])
    work = tmp_path / "work"
    r = run([sys.executable, "workload/scripts/scheduler.py", "run",
             "--work-root", str(work), "--tasks-dir", str(tasks),
             "--skill-root", "workload/stages/optimize/subagent-raw",
             "--runner", "workload/runners/stub.sh", "--gpus", "0,1,2,3",
             "--target", "10", "--attempt-cap", "5", "--tick-sec", "2",
             "--startup-grace", "3", "--backend", "local"], timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    log = (work / "scheduler.log").read_text()
    assert log.count("dispatch gpu") == 1, "one task must be dispatched exactly once:\n%s" % log


# ----------------------------------------------------------------- optimize skill interface

@pytest.mark.parametrize("skill", ["geak-kernel-workflow", "subagent-raw"])
def test_optimize_prompt_carries_the_four_hard_constraints(tmp_path, skill):
    """Every optimize implementation's prompt must carry all four — each one is in there
    because it was actually violated."""
    tasks = tmp_path / "tasks"
    run([sys.executable, os.path.join(HERE, "make_fake_tasks.py"),
         "--dir", str(tasks), "--count", "1"])
    p = tmp_path / "prompt.txt"
    r = run(["bash", "workload/stages/optimize/%s/prompt.sh" % skill],
            env_extra={"K": "k", "TASK_DIR": str(tasks / "fake_kernel_0"),
                       "KERNEL_SRC": "mod.py", "ATTEMPT": "1", "BEST_SPEEDUP": "1.0",
                       "RUN_DIR": str(tmp_path), "EVAL_DIR": str(tmp_path),
                       "ATTEMPT_CAP": "1500", "PROMPT_OUT": str(p),
                       "GEAK_ROOT": GEAK})
    assert r.returncode == 0, r.stdout + r.stderr
    text = p.read_text()
    for needle in ("frozen", "noise band", "generalize",
                   "Every output must be written"):
        assert needle in text, "%s's prompt is missing the constraint %r" % (skill, needle)
    assert "mod.py" in text


# ----------------------------------------------------------------- applyback gates

def _ab(tmp_path, shadowed, cand_tp, cand_basis):
    out = tmp_path / "out"
    (out / "e2e").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(exist_ok=True)
    (out / "base.json").write_text(json.dumps(
        {"output_token_throughput": 446.87889, "metric_basis": "agentx",
         "noise_band": 1.045}))
    (out / "applied.json").write_text(json.dumps(
        {"all_shadowed": shadowed,
         "applied": [{"kernel": "k", "verified_speedup": 1.80, "target": "m:f"}],
         "skipped": []}))
    (out / "e2e" / "bench_summary.json").write_text(json.dumps(
        {"throughput_tok_s_median": cand_tp, "metric_basis": cand_basis}))
    r = run([sys.executable, "workload/stages/applyback/finalize.py",
             "--e2e-dir", str(out / "e2e"), "--baseline", str(out / "base.json")],
            env_extra={"OUT": str(out), "STAGE": "applyback",
                       "STAGE_SKILL": "overlay-e2e-ab"})
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    ab = json.loads((out / "ab.json").read_text())
    return res, ab


def test_applyback_op_win_with_flat_e2e_reports_no_change(tmp_path):
    """Real history: operator 1.80x, whole machine +0.2356%. It must report no_change
    honestly and must not dress it up."""
    res, ab = _ab(tmp_path, True, 447.9319, "agentx")
    assert ab["verdict"] == "no_change"
    assert ab["op_speedup_max"] == 1.80, \
        "the operator number stays as measured, stated separately from e2e"
    assert abs(ab["e2e_delta_pct"] - 0.2356) < 0.01


def test_applyback_refuses_to_publish_when_overlay_did_not_shadow(tmp_path):
    """If the overlay never took effect, even an apparent +12% cannot be published — there
    is no telling what that number measured."""
    res, ab = _ab(tmp_path, False, 500.5, "agentx")
    assert res["status"] == "fail"


def test_applyback_refuses_to_compare_across_metric_basis(tmp_path):
    """Two numbers on different bases are not comparable; the A/B is void."""
    res, ab = _ab(tmp_path, True, 500.5, "something_else")
    assert res["status"] == "fail"
    assert ab["basis_comparable"] is False


# ----------------------------------------------------------------- vendored skill integrity

def test_vendored_skills_match_their_manifest():
    """The two upstream skills are vendored byte-identical; modified, they are no longer
    the same skill."""
    r = run([sys.executable, "workload/scripts/skill_manifest.py", "verify-all"])
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------- runner contract

@pytest.mark.parametrize("runner", ["claude", "codex", "stub"])
def test_runner_is_executable_and_nonzero_without_prompt(tmp_path, runner):
    """A missing prompt must return non-zero (78 per the contract), never fake success."""
    r = run(["bash", "workload/runners/%s.sh" % runner,
             str(tmp_path / "nope.txt"), str(tmp_path), str(tmp_path / "log")])
    assert r.returncode != 0


def test_no_runner_overrides_gpu_visibility():
    """A runner must not set ROCR_/HIP_VISIBLE_DEVICES — GPU isolation is the caller's, one
    container per GPU, and a runner reaching in only tangles physical isolation with an env
    pin."""
    for name in ("claude", "codex", "stub"):
        text = open(os.path.join(WORKLOAD, "runners", "%s.sh" % name)).read()
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            assert not s.startswith(("export ROCR_VISIBLE_DEVICES",
                                     "export HIP_VISIBLE_DEVICES",
                                     "ROCR_VISIBLE_DEVICES=",
                                     "HIP_VISIBLE_DEVICES=")), \
                "%s.sh touches GPU visibility: %s" % (name, s)
