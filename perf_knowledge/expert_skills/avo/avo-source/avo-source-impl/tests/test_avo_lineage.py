"""Rigorous unit tests for scripts/avo_lineage.py (the deterministic lineage writer).

Run:  python -m pytest avo-workflow/tests/test_avo_lineage.py -q
These need no GPU / LLM — they pin the exactness-critical behavior (geomean,
commit-gate refusal, lineage.json structure, message format, side-effect files).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

# Load scripts/avo_lineage.py as a module (it lives outside any package).
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "avo_lineage.py"
_spec = importlib.util.spec_from_file_location("avo_lineage", _SCRIPT)
avo_lineage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avo_lineage)


def _pc(*pairs):
    return json.dumps([{"name": n, "speedup": s} for n, s in pairs])


def _pc_lat(*triples):
    # (name, speedup, baseline_ms, optimized_ms)
    return json.dumps([{"name": n, "speedup": s, "baseline_ms": b, "optimized_ms": o}
                       for n, s, b, o in triples])


# --- geomean ---------------------------------------------------------------

def test_geomean_basic():
    assert math.isclose(avo_lineage._geomean([{"speedup": 1.6}, {"speedup": 1.4}]),
                        math.sqrt(1.6 * 1.4), rel_tol=1e-9)


def test_geomean_ignores_nonpositive():
    assert math.isclose(avo_lineage._geomean(
        [{"speedup": 2.0}, {"speedup": 0}, {"speedup": -1}, {"speedup": 2.0}]), 2.0, rel_tol=1e-9)


def test_geomean_empty_is_zero():
    assert avo_lineage._geomean([]) == 0.0


def test_geomean_skips_excluded_cases():
    assert math.isclose(avo_lineage._geomean([
        {"speedup": 2.0},
        {"speedup": 4.0, "included": False},
    ]), 2.0, rel_tol=1e-9)


def test_geomean_weighted_ratio_of_sums():
    # Unweighted geomean would be sqrt(2*1)=1.414; weighted by calls is 3/2=1.5.
    got = avo_lineage._geomean([
        {"name": "hot", "speedup": 2.0, "calls": 1024,
         "baseline_ms": 2.0, "optimized_ms": 1.0},
        {"name": "cold", "speedup": 1.0, "calls": 1,
         "baseline_ms": 1.0, "optimized_ms": 1.0},
    ])
    assert math.isclose(got, (1024 * 2.0 + 1 * 1.0) / (1024 * 1.0 + 1 * 1.0), rel_tol=1e-9)


# --- seed ------------------------------------------------------------------

def test_seed_writes_v0(tmp_path: Path):
    state = tmp_path / "avo_state"
    rc = avo_lineage.main(["seed", "--state", str(state),
                           "--per-case", _pc(("s1", 1.0), ("s2", 1.0)),
                           "--baseline-geomean-ms", "4.5"])
    assert rc == 0
    data = json.loads((state / "lineage.json").read_text())
    assert data["best_id"] == "v0"
    assert len(data["committed"]) == 1
    v0 = data["committed"][0]
    assert v0["git_ref"] == "avo-v0" and v0["score"]["speedup"] == 1.0
    assert v0["score"]["latency_ms"] == 4.5


# --- commit ----------------------------------------------------------------

def _seed(tmp_path: Path) -> Path:
    state = tmp_path / "avo_state"
    avo_lineage.main(["seed", "--state", str(state), "--per-case", _pc(("s1", 1.0), ("s2", 1.0))])
    return state


def test_commit_appends_node_and_advances_best(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    capsys.readouterr()  # discard the seed's stdout so we read only the commit message
    rc = avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                           "--per-case", _pc(("s1", 1.6), ("s2", 1.4)),
                           "--strategy", "shared-mem tiling",
                           "--rationale", "memory-bound; staged tiles",
                           "--parent", "avo-v0", "--step", "3"])
    assert rc == 0
    msg = capsys.readouterr().out
    # subject + body assertions (greppable + auditable)
    assert "avo: avo-v1 | 1.497x | shared-mem tiling" in msg
    assert "speedup:  1.4967x vs baseline (avo-v0)" in msg
    assert "parent:   avo-v0" in msg
    assert "step:     3" in msg
    assert "s1=1.600x" in msg and "s2=1.400x" in msg
    assert "memory-bound; staged tiles" in msg

    data = json.loads((state / "lineage.json").read_text())
    assert data["best_id"] == "v1" and data["active_best_id"] == "v1"
    v1 = data["committed"][1]
    assert v1["git_ref"] == "avo-v1" and v1["parent_id"] == "v0"
    assert math.isclose(v1["score"]["speedup"], math.sqrt(1.6 * 1.4), rel_tol=1e-6)
    assert v1["score"]["per_shape"] == {"s1": 1.6, "s2": 1.4}


def test_commit_writes_side_effect_files(tmp_path: Path):
    state = _seed(tmp_path)
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc(("s1", 1.6), ("s2", 1.4)), "--strategy", "x", "--step", "2"])
    attempts = (state / "attempts.jsonl").read_text().strip().splitlines()
    assert len(attempts) == 1 and json.loads(attempts[0])["committed"] is True
    assert (state / "evolution_log" / "step_0002.json").exists()


def test_off_branch_champion_preserves_tree_parent_and_resumes_as_best(
        tmp_path: Path, capsys):
    state = _seed(tmp_path)
    capsys.readouterr()
    assert avo_lineage.main([
        "commit", "--state", str(state), "--version-n", "1",
        "--per-case", _pc(("s1", 1.5), ("s2", 1.5)),
        "--strategy", "tiling", "--parent", "avo-v0", "--step", "1",
    ]) == 0
    capsys.readouterr()
    assert avo_lineage.main([
        "commit", "--state", str(state), "--version-n", "2",
        "--per-case", _pc(("s1", 1.7), ("s2", 1.6)),
        "--strategy", "vectorize", "--parent", "avo-v0", "--step", "2",
    ]) == 0
    capsys.readouterr()

    data = json.loads((state / "lineage.json").read_text())
    assert data["best_id"] == "v2"
    assert data["committed"][2]["parent_id"] == "v0"

    assert avo_lineage.main(["resume", "--state", str(state)]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["best_version"] == "avo-v2"
    assert resumed["cumulative"] > 1.6


def test_commit_refuses_below_best(tmp_path: Path):
    """Defense in depth: a candidate that does not beat the best is refused."""
    state = _seed(tmp_path)
    # establish best = 1.5
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc(("s1", 1.5), ("s2", 1.5)), "--strategy", "a", "--step", "1"])
    # candidate 1.2 < best 1.5 -> non-zero exit, lineage unchanged
    rc = avo_lineage.main(["commit", "--state", str(state), "--version-n", "2",
                           "--per-case", _pc(("s1", 1.2), ("s2", 1.2)), "--strategy", "b", "--step", "2"])
    assert rc == 3
    data = json.loads((state / "lineage.json").read_text())
    assert data["best_id"] == "v1" and len(data["committed"]) == 2  # v0 + v1 only


def test_commit_refuses_no_positive_speedup(tmp_path: Path):
    state = _seed(tmp_path)
    rc = avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                           "--per-case", _pc(("s1", 0.0)), "--strategy", "a", "--step", "1"])
    assert rc == 2


def test_context_lists_prior_versions_excluding_best(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    # patches dir + two committed versions
    (state / "patches").mkdir(parents=True, exist_ok=True)
    (state / "patches" / "v1.patch").write_text("--- a\n+++ b\n@@ v1 diff @@\n")
    (state / "patches" / "v2.patch").write_text("--- a\n+++ b\n@@ v2 diff @@\n")
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc(("s1", 1.3)), "--strategy", "tiling", "--step", "1"])
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "2",
                      "--per-case", _pc(("s1", 1.8)), "--strategy", "pipelining", "--step", "2"])
    capsys.readouterr()  # discard
    rc = avo_lineage.main(["context", "--state", str(state), "--k", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    # best is v2 -> excluded; v0 excluded; only v1 shown
    assert "avo-v1" in out and "tiling" in out and "v1 diff" in out
    assert "avo-v2" not in out and "avo-v0" not in out


def test_commit_includes_latency_and_expected(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    capsys.readouterr()
    rc = avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                           "--per-case", _pc_lat(("s1", 2.0, 4.0, 2.0), ("s2", 2.0, 8.0, 4.0)),
                           "--strategy", "vectorize", "--expected", "~2x by widening loads",
                           "--step", "1"])
    assert rc == 0
    msg = capsys.readouterr().out
    # latency = geomean(optimized_ms)=geomean(2,4)=2.8284; baseline=geomean(4,8)=5.6569
    assert "2.8284ms" in msg                       # in subject + body
    assert "latency:  2.8284ms (baseline 5.6569ms)" in msg
    assert "expected: ~2x by widening loads" in msg
    # latency_ms stored in lineage.json for the report's latency / vs_base columns
    data = json.loads((state / "lineage.json").read_text())
    assert abs(data["committed"][1]["score"]["latency_ms"] - 2.828427) < 1e-4


def test_context_excludes_v0_and_best(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    (state / "patches").mkdir(parents=True, exist_ok=True)
    (state / "patches" / "v1.patch").write_text("--- a/k\n+++ b/k\n@@ -1 +1 @@\n-o\n+v1\n")
    (state / "patches" / "v2.patch").write_text("--- a/k\n+++ b/k\n@@ -1 +1 @@\n-o\n+v2\n")
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc(("s1", 1.3)), "--strategy", "tiling", "--step", "1"])
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "2",
                      "--per-case", _pc(("s1", 1.7)), "--strategy", "vectorize", "--step", "2"])
    capsys.readouterr()
    avo_lineage.main(["context", "--state", str(state), "--k", "3"])
    out = capsys.readouterr().out
    assert "avo-v1" in out and "tiling" in out and "+v1" in out   # the non-best prior version
    assert "avo-v2" not in out                                     # current best is excluded
    assert "avo-v0" not in out                                     # baseline is excluded


def test_seed_persists_baseline_for_resume(tmp_path: Path):
    state = tmp_path / "avo_state"
    avo_lineage.main(["seed", "--state", str(state),
                      "--per-case", _pc(("s1", 1.0), ("s2", 1.0)), "--baseline-geomean-ms", "4.5"])
    base = json.loads((state / "baseline.json").read_text())
    assert base["geomean_ms"] == 4.5 and len(base["per_case"]) == 2


def test_resume_reconstructs_state(tmp_path: Path, capsys):
    state = _seed(tmp_path)  # writes baseline.json too
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc_lat(("s1", 1.5, 4.0, 2.6667)), "--strategy", "tiling", "--step", "1"])
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "2",
                      "--per-case", _pc_lat(("s1", 1.9, 4.0, 2.1053)), "--strategy", "vectorize", "--step", "2"])
    # a supervisor intervention marking a dead direction
    (state / "supervisor_log.jsonl").write_text(json.dumps({
        "ts": 1, "directive": {"diagnosis": "tiling exhausted", "mark_failed": ["tiling"],
                               "new_directions": [{"name": "warp-spec", "expected": "overlap", "prompt": "..."}]}}) + "\n")
    capsys.readouterr()
    rc = avo_lineage.main(["resume", "--state", str(state)])
    assert rc == 0
    r = json.loads(capsys.readouterr().out)
    assert r["resume"] is True
    assert r["best_version"] == "avo-v2" and r["commits"] == 2
    assert abs(r["cumulative"] - 1.9) < 1e-9
    assert len(r["committed"]) == 3                       # v0, v1, v2
    assert r["committed"][2]["parent_id"] == "v1"
    assert r["committed"][2]["per_case"][0]["name"] == "s1"
    assert r["baseline_per_case"]                          # restored from baseline.json (the per-case AVO needs)
    assert "tiling" in r["failed"]                        # from supervisor_log
    assert r["direction"]["name"] == "warp-spec"
    assert r["prior_steps"] == 0   # no variation_* / escalate_* dirs under eval root
    assert isinstance(r.get("best_ir"), dict)
    assert r["best_ir"].get("ok") is False


def test_commit_persists_score_ir_when_ir_file(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    ir_path = tmp_path / "irvec.json"
    ir_path.write_text(json.dumps({
        "lang": "triton", "ok": True, "tier": "ttgir",
        "metrics": {"convert_layout": 2, "mma_used": True},
    }), encoding="utf-8")
    capsys.readouterr()
    rc = avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                          "--per-case", _pc_lat(("s1", 1.5, 4.0, 2.6667)), "--strategy", "tiling",
                          "--step", "1", "--ir-file", str(ir_path)])
    assert rc == 0
    data = json.loads((state / "lineage.json").read_text())
    v1 = data["committed"][1]
    ir = (v1.get("score") or {}).get("ir")
    assert ir and ir.get("ok") is True and ir["metrics"]["convert_layout"] == 2


def test_resume_emits_best_ir_from_best_node(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    ir_path = tmp_path / "ir.json"
    ir_path.write_text(json.dumps({
        "lang": "triton", "ok": True, "tier": "ttgir", "metrics": {"mma_used": True},
    }), encoding="utf-8")
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc_lat(("s1", 1.5, 4.0, 2.6667)), "--strategy", "a", "--step", "1"])
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "2",
                      "--per-case", _pc_lat(("s1", 1.9, 4.0, 2.1053)), "--strategy", "b", "--step", "2",
                      "--ir-file", str(ir_path)])
    capsys.readouterr()
    assert avo_lineage.main(["resume", "--state", str(state)]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["resume"] is True
    assert r["best_ir"].get("ok") is True and r["best_ir"].get("lang") == "triton"


def test_max_completed_evolve_step_uses_max_index_not_dir_count(tmp_path: Path):
    """Team mode uses multiple dirs per step; prior_steps must be max NNNN, not wc -l."""
    evald = tmp_path
    for name in (
        "variation_0059_0",
        "variation_0060_0",
        "variation_0060_1",
        "variation_0060_2",
        "variation_0060_integrate",
        "escalate_0012_0",
    ):
        (evald / name).mkdir(parents=True)
    assert avo_lineage._max_completed_evolve_step(evald) == 60


def test_resume_prior_steps_reflects_variation_dirs(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc_lat(("s1", 1.5, 4.0, 2.6667)), "--strategy", "tiling", "--step", "1"])
    evald = tmp_path
    (evald / "variation_0042_0").mkdir()
    (evald / "variation_0042_1").mkdir()
    (evald / "variation_0042_2").mkdir()
    capsys.readouterr()
    assert avo_lineage.main(["resume", "--state", str(state)]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["resume"] is True
    assert r["prior_steps"] == 42


def test_resume_empty_lineage(tmp_path: Path, capsys):
    rc = avo_lineage.main(["resume", "--state", str(tmp_path / "nope")])
    assert rc == 0 and json.loads(capsys.readouterr().out) == {"resume": False}


# --- attempt (full search trajectory: every attempt incl. failures) --------

def test_attempt_appends_noncommitted_line(tmp_path: Path):
    state = tmp_path / "avo_state"
    rc = avo_lineage.main(["attempt", "--state", str(state), "--step", "3",
                           "--strategy", "occupancy tune", "--status", "failed",
                           "--self-speedup", "0.97", "--patch-hash", "abc123",
                           "--rationale", "regressed P3", "--worker", "1"])
    assert rc == 0
    lines = (state / "attempts.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["committed"] is False           # default; the gate decides commits
    assert rec["step_index"] == 3 and rec["worker"] == 1
    assert rec["strategy"] == "occupancy tune" and rec["status"] == "failed"
    assert abs(rec["self_speedup"] - 0.97) < 1e-9


def test_attempt_committed_flag_parses(tmp_path: Path):
    state = tmp_path / "avo_state"
    avo_lineage.main(["attempt", "--state", str(state), "--step", "1",
                      "--strategy", "x", "--committed", "true"])
    rec = json.loads((state / "attempts.jsonl").read_text().strip())
    assert rec["committed"] is True


def test_attempt_is_appendonly(tmp_path: Path):
    state = tmp_path / "avo_state"
    for i in range(3):
        avo_lineage.main(["attempt", "--state", str(state), "--step", str(i), "--strategy", f"s{i}"])
    assert len((state / "attempts.jsonl").read_text().strip().splitlines()) == 3


def test_attempt_signal_flags_recurring_failures(tmp_path: Path):
    state = tmp_path / "avo_state"
    # 'occupancy' tried 3x (never committed) -> dead; 'fusion' once -> below threshold;
    # 'graph' tried twice but later committed -> NOT flagged.
    for _ in range(3):
        avo_lineage.main(["attempt", "--state", str(state), "--step", "1", "--strategy", "occupancy"])
    avo_lineage.main(["attempt", "--state", str(state), "--step", "2", "--strategy", "fusion"])
    for _ in range(2):
        avo_lineage.main(["attempt", "--state", str(state), "--step", "3", "--strategy", "graph"])
    avo_lineage.main(["attempt", "--state", str(state), "--step", "3", "--strategy", "graph", "--committed", "true"])
    sig = avo_lineage._attempt_signal(state)
    joined = "\n".join(sig)
    assert "occupancy" in joined and "3x" in joined
    assert "fusion" not in joined          # only tried once
    assert "graph" not in joined           # eventually committed


def test_resume_merges_attempt_failure_signal(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc(("s1", 1.5)), "--strategy", "tiling", "--step", "1"])
    # a strategy that keeps failing, with NO supervisor intervention at all
    for _ in range(2):
        avo_lineage.main(["attempt", "--state", str(state), "--step", "2", "--strategy", "occupancy tune"])
    capsys.readouterr()
    avo_lineage.main(["resume", "--state", str(state)])
    r = json.loads(capsys.readouterr().out)
    assert r["resume"] is True
    assert any("occupancy tune" in s for s in r["insights"])   # dead strategy survives resume
    assert r["failed"] == []                                    # none marked failed by a supervisor


def test_heartbeat_written_on_seed_and_updated(tmp_path: Path):
    state = tmp_path / "avo_state"
    avo_lineage.main(["seed", "--state", str(state), "--per-case", _pc(("s1", 1.0))])
    hb = json.loads((state / "heartbeat.json").read_text())
    assert hb["phase"] == "seed" and hb["best_version"] == "avo-v0" and "ts" in hb
    # the explicit heartbeat subcommand merges/updates fields
    avo_lineage.main(["heartbeat", "--state", str(state), "--phase", "variation",
                      "--step", "5", "--best", "avo-v2", "--speedup", "1.8"])
    hb2 = json.loads((state / "heartbeat.json").read_text())
    assert hb2["phase"] == "variation" and hb2["step"] == 5 and hb2["best_version"] == "avo-v2"
    assert abs(hb2["best_speedup"] - 1.8) < 1e-9


def test_context_empty_when_no_priors(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    capsys.readouterr()
    rc = avo_lineage.main(["context", "--state", str(state), "--k", "3"])
    assert rc == 0 and capsys.readouterr().out.strip() == ""       # only v0 -> nothing to compare


# --- roofline --------------------------------------------------------------

def _pc_rl(name, speedup, baseline_ms, optimized_ms, flops, bytes_, dtype="fp16"):
    return {"name": name, "speedup": speedup, "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms, "flops": flops, "bytes": bytes_, "dtype": dtype}


def test_roofline_case_memory_bound():
    # AI = flops/bytes = 1e10/1e9 = 10 (< fp16 ridge ~245) -> memory region.
    bw = avo_lineage._PEAKS["gfx942"]["hbm_bw"]
    latency_ms = (1e9 / (0.5 * bw)) * 1000  # 50% of peak BW
    a = avo_lineage._roofline_case(
        {"name": "m", "optimized_ms": latency_ms, "flops": 1e10, "bytes": 1e9, "dtype": "fp16"},
        latency_key="optimized_ms", hw="gfx942")
    assert a["regime"] == "memory-bound"
    assert math.isclose(a["fraction_of_roofline"], 0.5, abs_tol=1e-3)
    assert math.isclose(a["potential_speedup"], 2.0, abs_tol=1e-2)


def test_roofline_case_compute_bound():
    pc = avo_lineage._PEAKS["gfx942"]["compute"]["fp16"]
    latency_ms = (1e12 / (0.5 * pc)) * 1000   # AI=1000, 50% of FLOP ceiling
    a = avo_lineage._roofline_case(
        {"name": "c", "optimized_ms": latency_ms, "flops": 1e12, "bytes": 1e9, "dtype": "fp16"},
        latency_key="optimized_ms", hw="gfx942")
    assert a["regime"] == "compute-bound"
    assert math.isclose(a["fraction_of_roofline"], 0.5, abs_tol=1e-3)


def test_roofline_case_none_without_telemetry():
    assert avo_lineage._roofline_case({"name": "x", "optimized_ms": 1.0},
                                      latency_key="optimized_ms", hw="gfx942") is None


def test_roofline_summary_picks_worst_case():
    bw = avo_lineage._PEAKS["gfx942"]["hbm_bw"]
    pc = avo_lineage._PEAKS["gfx942"]["compute"]["fp16"]
    per_case = [
        {"name": "mem", "optimized_ms": (1e9 / (0.5 * bw)) * 1000, "flops": 1e10, "bytes": 1e9},
        {"name": "cmp", "optimized_ms": (1e12 / (0.9 * pc)) * 1000, "flops": 1e12, "bytes": 1e9},
    ]
    s = avo_lineage._roofline_summary(per_case, latency_key="optimized_ms")
    assert s["ok"] is True and s["n_valid"] == 2
    assert s["worst_case"]["name"] == "mem"        # 2x potential beats ~1.1x
    assert s["dominant_regime"] == "memory-bound"


def test_roofline_summary_relative_only_when_no_telemetry():
    s = avo_lineage._roofline_summary([{"name": "a", "optimized_ms": 1.0}], latency_key="optimized_ms")
    assert s["ok"] is False and s["geomean_fraction"] is None


def test_roofline_case_invalid_when_fraction_over_one():
    # tiny latency vs flops => achieved >> peak => fraction >> 1 => untrustworthy telemetry
    a = avo_lineage._roofline_case(
        {"name": "bad", "optimized_ms": 0.001, "flops": 1e12, "bytes": 1e9, "dtype": "fp16"},
        latency_key="optimized_ms", hw="gfx942")
    assert a is not None and a["telemetry_invalid"] is True
    assert a["fraction_of_roofline"] is None and a["regime"] == "unknown"
    assert a["potential_speedup"] is None


def test_roofline_case_keeps_within_tolerance_overshoot():
    pc = avo_lineage._PEAKS["gfx942"]["compute"]["fp16"]
    latency_ms = (1e12 / (1.03 * pc)) * 1000   # AI=1000, 1.03x of ceiling (<= 1.05 tol)
    a = avo_lineage._roofline_case(
        {"name": "noise", "optimized_ms": latency_ms, "flops": 1e12, "bytes": 1e9},
        latency_key="optimized_ms", hw="gfx942")
    assert a["telemetry_invalid"] is False and a["fraction_of_roofline"] is not None


def test_roofline_summary_excludes_invalid_and_flags_all_invalid():
    bw = avo_lineage._PEAKS["gfx942"]["hbm_bw"]
    bad = {"name": "bad", "optimized_ms": 0.001, "flops": 1e12, "bytes": 1e9}
    s0 = avo_lineage._roofline_summary([bad], latency_key="optimized_ms")
    assert s0["ok"] is False and s0["n_valid"] == 0 and s0["n_invalid"] == 1
    good = {"name": "mem", "optimized_ms": (1e9 / (0.5 * bw)) * 1000, "flops": 1e10, "bytes": 1e9}
    s1 = avo_lineage._roofline_summary([good, bad], latency_key="optimized_ms")
    assert s1["ok"] is True and s1["n_valid"] == 1 and s1["n_invalid"] == 1
    assert s1["worst_case"]["name"] == "mem"


def test_commit_stores_roofline_in_lineage(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    bw = avo_lineage._PEAKS["gfx942"]["hbm_bw"]
    lat = (1e9 / (0.5 * bw)) * 1000
    per_case = json.dumps([_pc_rl("s1", 2.0, 2 * lat, lat, 1e10, 1e9)])
    capsys.readouterr()
    rc = avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                           "--per-case", per_case, "--strategy", "tiling", "--step", "1"])
    assert rc == 0
    data = json.loads((state / "lineage.json").read_text())
    rl = data["committed"][1]["score"]["roofline"]
    assert rl["regime"] == "memory-bound"
    assert math.isclose(rl["geomean_fraction"], 0.5, abs_tol=1e-3)


def test_roofline_subcommand(tmp_path: Path, capsys):
    bw = avo_lineage._PEAKS["gfx942"]["hbm_bw"]
    lat = (1e9 / (0.5 * bw)) * 1000
    per_case = json.dumps([{"name": "m", "optimized_ms": lat, "flops": 1e10, "bytes": 1e9}])
    rc = avo_lineage.main(["roofline", "--per-case", per_case])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["dominant_regime"] == "memory-bound"


def test_subject_truncated_when_long(tmp_path: Path, capsys):
    state = _seed(tmp_path)
    capsys.readouterr()  # discard the seed's stdout
    avo_lineage.main(["commit", "--state", str(state), "--version-n", "1",
                      "--per-case", _pc(("s1", 2.0)), "--strategy", "x" * 200, "--step", "1"])
    subject = capsys.readouterr().out.splitlines()[0]
    assert len(subject) <= 100 and subject.endswith("...")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
