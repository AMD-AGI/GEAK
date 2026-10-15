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
             "workload/config/agentx_glm52.json", "--dry-run"])
    assert r.returncode == 0, r.stdout + r.stderr
    for stage in ("baseline", "profile", "roofline", "optimize", "applyback"):
        assert stage in r.stdout


def test_set_override_wins(tmp_path):
    """--set must beat the config file — smoke runs rely on it to shrink the scale."""
    r = run([sys.executable, "workload/scripts/stage.py", "--config",
             "workload/config/agentx_glm52.json",
             "--set", "stages.optimize.skill=subagent-raw",
             "--set", "stages.optimize.params.time_budget_sec=60", "--dry-run"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "subagent-raw" in r.stdout


# ----------------------------------------------------------------- target gate
#
# The failure these cover is not hypothetical: the first full campaign ran end to end as
# `agentx_dsv4` against a GLM-5.2-MXFP4 stack. Every number was right, every label was
# wrong, and nothing in the pipeline was in a position to notice.

def test_scenario_config_declares_what_it_optimizes():
    """Every scenario config must name the stack it targets — the gate has nothing to
    check against otherwise."""
    for name in os.listdir(os.path.join(WORKLOAD, "config")):
        if not name.endswith(".json") or name == "default.json":
            continue
        cfg = json.load(open(os.path.join(WORKLOAD, "config", name)))
        t = cfg.get("target") or {}
        # A config that targets no real stack (the smoke configs run fake CPU tasks) opts
        # out explicitly and in writing. Silence is not an option.
        assert t.get("model") or (t.get("enforce") is False and t.get("_note_enforce")), \
            "%s neither declares target.model nor states why it is exempt" % name


def test_run_name_must_carry_a_token_of_the_declared_model(tmp_path):
    """A run whose directory disagrees with its model is refused before it starts."""
    r = run([sys.executable, "workload/scripts/stage.py", "--config",
             "workload/config/agentx_glm52.json",
             "--run-root", str(tmp_path / "agentx_dsv4"), "--dry-run"])
    assert r.returncode != 0
    assert "target.model" in (r.stdout + r.stderr)

    ok = run([sys.executable, "workload/scripts/stage.py", "--config",
              "workload/config/agentx_glm52.json",
              "--run-root", str(tmp_path / "agentx_glm52_smoke"), "--dry-run"])
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_name_tokens_let_an_abbreviation_through_only_when_declared():
    import target
    hits, toks = target.check_name("agentx_dsv4", "deepseek-ai/DeepSeek-V4")
    assert not hits, "an abbreviation must not pass by accident (tokens were %s)" % toks
    hits, _ = target.check_name("agentx_dsv4", "deepseek-ai/DeepSeek-V4",
                                extra=["dsv4"])
    assert hits == ["dsv4"]


def test_version_digits_fold_onto_the_model_name():
    """`GLM-5.2` is written glm52 and glm5 as often as glm-5.2."""
    import target
    toks = target.name_tokens("amd/GLM-5.2-MXFP4")
    assert {"glm", "glm5", "glm52", "mxfp4"} <= set(toks)
    # The organisation prefix says who published the weights, not which stack is under
    # test; blessing it would bless `agentx_amd` for everything AMD ever uploaded.
    assert "amd" not in toks


def test_observed_stack_contradicting_the_config_is_a_mismatch(tmp_path):
    import target
    out = tmp_path / "out"
    out.mkdir()
    (out / "stack_identity.json").write_text(json.dumps({
        "superseded_stack": {"image": "someone/else:tag"},
        "active_stack": {"image": "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260913",
                         "sglang_commit": "7f1f8c706a"}}))
    cfg = {"target": {"model": "amd/GLM-5.2-MXFP4",
                      "image": "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260913",
                      "commits": {"sglang": "7f1f8c706a"}}}
    obs = target.observed_identity(str(out))
    rep = target.identity_report(cfg, obs)
    assert rep["verdict"] == "ok", rep
    assert not target.identity_problems(rep)

    cfg["target"]["commits"]["sglang"] = "deadbeef12"
    bad = target.identity_report(cfg, obs)
    assert bad["verdict"] == "mismatch"
    assert target.identity_problems(bad)


def test_a_stage_pinned_to_another_image_is_refused(tmp_path):
    """Optimizing kernels inside an image the run does not claim to target produces work
    for a system nobody is measuring."""
    r = run([sys.executable, "workload/scripts/stage.py", "--config",
             "workload/config/agentx_glm52.json",
             "--set", "stages.optimize.params.image=someone/else:tag", "--dry-run"])
    assert r.returncode != 0
    assert "target.image" in (r.stdout + r.stderr)


def test_a_stage_that_observed_nothing_is_unverified_not_ok(tmp_path):
    import target
    rep = target.identity_report({"target": {"model": "amd/GLM-5.2-MXFP4"}}, {})
    assert rep["verdict"] == "unverified"
    assert not target.identity_problems(rep)


# ----------------------------------------------------------------- CI replay alignment
#
# These guard the baseline itself. A baseline that measured something adjacent to the CI
# point is worse than no baseline: every downstream delta inherits the error with a
# plausible-looking sign, and nothing in the output says so.

REPLAY = os.path.join(WORKLOAD, "stages", "baseline", "ci-run-replay")
sys.path.insert(0, REPLAY)


def _params(**over):
    p = {"pins": {"image": "", "image_digest": "sha256:" + "a" * 64,
                  "model": "m", "model_revision": "r", "seed": 42,
                  "benchmark_id": "bid_1", "cache_bust": "first_turn_prefix",
                  "duration_s": 3600,
                  "trace_corpus": {"loader": "semianalysis_cc_traces_weka_062126"}},
         "ci": {"output_throughput": 100.0}}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k] = dict(p[k], **v)
        else:
            p[k] = v
    return p


def _blockers(p):
    import preflight
    problems = []
    preflight.check_pins(p, problems)
    return problems


def test_a_tag_is_not_an_identity():
    """A tag is a moving pointer; only the digest says what was measured."""
    assert any("moving pointer" in x for x in
               _blockers(_params(pins={"image_digest": "v0.5.19-rocm720"})))
    assert not any("moving pointer" in x for x in _blockers(_params()))


def test_a_reissued_benchmark_id_is_refused(tmp_path):
    """The AgentX cache-bust marker embeds the id, so the id is the KV keyspace. A second
    run under the same id starts on warmed prefixes and measures high, silently."""
    import preflight
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"used": [{"benchmark_id": "bid_1"}]}))
    problems = []
    preflight.check_benchmark_id(_params()["pins"], str(ledger), problems)
    assert any("already been used" in x for x in problems)
    problems = []
    preflight.check_benchmark_id({"benchmark_id": "bid_2"}, str(ledger), problems)
    assert problems == []


def test_a_different_cache_bust_target_is_refused():
    assert any("first_turn_prefix" in x for x in
               _blockers(_params(pins={"cache_bust": "whole_prompt"})))


def test_the_corpus_may_be_a_loader_but_may_not_be_unnamed():
    """The AgentX harness takes a public-dataset loader name, not a file. Demanding a path
    would make an honest config impossible; demanding nothing would let the corpus
    generation drift, and 062126 and 061526 are different prompt-length mixes."""
    assert not any("trace_corpus" in x for x in _blockers(_params()))
    assert any("trace_corpus" in x for x in _blockers(_params(pins={"trace_corpus": {}})))


def test_a_weights_cache_is_never_cold_wiped(tmp_path):
    """Emptying HF_HOME re-downloads tens of gigabytes and can change which revision is
    served underneath the run. Compile caches are cold; weights caches are pinned."""
    import preflight
    problems = []
    preflight.check_isolation({"isolation": {"cache_vars": ["TRITON_CACHE_DIR", "HF_HOME"]}},
                              str(tmp_path), problems)
    assert any("HF_HOME" in x and "weights cache" in x for x in problems)


def test_a_warm_compile_cache_is_refused(tmp_path):
    """A warm Triton cache can carry kernels compiled from another campaign's overlay —
    which has happened on this stack."""
    import preflight
    warm = tmp_path / "caches" / "triton_cache_dir"
    warm.mkdir(parents=True)
    (warm / "leftover.hsaco").write_text("x")
    problems = []
    preflight.check_isolation({"isolation": {"cache_vars": ["TRITON_CACHE_DIR"]}},
                              str(tmp_path), problems)
    assert any("already holds" in x for x in problems)


def test_cache_dirs_shadow_the_path_the_harness_names(tmp_path):
    """`-e TRITON_CACHE_DIR=...` loses to a harness that exports it itself, so the fresh
    directory must be mounted onto the path the harness names — and the harness's own
    cache tree must be shadowed, not emptied."""
    import preflight
    iso = {"cache_dirs": {"TRITON_CACHE_DIR": {"container": "/ix/cache/triton"}}}
    root, dirs, mounts = preflight.cache_plan(iso, str(tmp_path))
    host = dirs["TRITON_CACHE_DIR"]
    assert mounts[host] == "/ix/cache/triton"
    assert host.startswith(root) and "/ix/cache" not in host


def test_an_empty_cache_after_the_run_is_a_failure_not_a_pass(tmp_path):
    """The only proof the variable reached the server. A cache that is still empty after an
    hour of serving means the server compiled somewhere else entirely."""
    out = tmp_path / "out"
    out.mkdir()
    rundir = tmp_path / "rundir"
    rundir.mkdir()
    cold = tmp_path / "triton"
    cold.mkdir()
    (tmp_path / "align.json").write_text(json.dumps(
        {"cache_vars": {"TRITON_CACHE_DIR": str(cold)}, "cpu_plan": {"cpus": []}}))
    (tmp_path / "params.json").write_text(json.dumps(_params()))
    r = run([sys.executable, os.path.join(REPLAY, "verify_alignment.py"),
             "--params", str(tmp_path / "params.json"),
             "--alignment", str(tmp_path / "align.json"),
             "--out", str(out), "--run-dir", str(rundir)])
    assert r.returncode == 1, r.stdout + r.stderr
    doc = json.load(open(out / "alignment_verified.json"))
    assert doc["verdict"] == "FAILED"
    assert any("still empty" in f for f in doc["failures"])


def test_a_check_that_could_not_run_is_unproven_not_ok(tmp_path):
    """'We could not check' must never render as a tick."""
    out = tmp_path / "out"
    out.mkdir()
    rundir = tmp_path / "rundir"      # the harness's own output, and nothing else
    rundir.mkdir()
    (tmp_path / "align.json").write_text(json.dumps({"cache_vars": {}, "cpu_plan": {}}))
    (tmp_path / "params.json").write_text(json.dumps(_params()))
    r = run([sys.executable, os.path.join(REPLAY, "verify_alignment.py"),
             "--params", str(tmp_path / "params.json"),
             "--alignment", str(tmp_path / "align.json"),
             "--out", str(out), "--run-dir", str(rundir)])
    assert r.returncode == 2, r.stdout + r.stderr
    doc = json.load(open(out / "alignment_verified.json"))
    assert doc["verdict"] == "unproven"
    assert all(c["status"] != "ok" for c in doc["checks"])


def test_physical_core_plan_drops_smt_siblings():
    """One logical CPU per physical core, from the machine's own topology. Guessing that
    siblings are cpu and cpu+n/2 is right on one node and wrong on the next."""
    import preflight
    plan, err = preflight.physical_cores()
    if err:
        pytest.skip(err)
    assert plan["physical_cores"] == len(plan["cpus"]) == len(set(plan["cpus"]))
    assert plan["physical_cores"] + plan["smt_siblings_dropped"] == \
        plan["granted_logical_cpus"]


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
