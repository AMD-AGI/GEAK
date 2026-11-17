"""In-process tests of ut_runner.main: the single-rank and multi-rank flows of a generated toy task, end to end.

The task is a real generated toy (generate_ut.py all) in a temp dir and main() loads its real vendored harness_lib;
only the subprocess boundary is replaced: the task harness's `_run_leg` and `build_candidate_overlay` answer as
leg_runner.py's modes do (what each mode prints and the result files it writes), and `ut_runner.RankPool` answers
per rank as leg_runner.py's serve mode does. So Legs, run_correctness, assert_legs_differ, measure_legs, the
rank-pool orchestration and every verdict, exit code and record in run/ut_result.json are the real code's
decisions. CPU only (no torch), no subprocess, no real clock."""
import argparse
import json
import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import generate_ut as gu
import ut_runner as ur
from ut_task_fixture import RankToyTask, ToyTask

SIGS = ("decode_m1", "decode_m4")
M = {"decode_m1": 1, "decode_m4": 4}
POLICY = {"mode": "read-evict", "bytes": 512 << 20, "preparation": "float32-sum", "outside_timing": True,
          "deployment_calibrated": False}
SPIN = {"queued": "gpu_spin_before_start", "reps": 1, "us": 200.0, "samples": 50, "late": 0, "retried": False,
        "includes_dispatch": False}
GRAPH_CAPTURE = {"sizes": [1, 4], "pad": "up"}
# a graph-deployed toy: cases.py needs replay (the check reads it by AST; the fake legs never import it)
REPLAY_HOOK = "\n\ndef replay(h, meta):\n    return None\n"
AITER_NOTE = "1 (test-harness setting for co-resident pools; not deployment configuration)"
_MISSING = object()


def _graph_meta(m):
    m["regime"] = {"cuda_graph": True, "graph_capture": dict(GRAPH_CAPTURE)}


def _touch(path):
    with open(path, "wb") as fh:
        fh.write(b"torch.save blob")


def _report(ok=True, err=0.0, draws=2, graph=False):
    """correctness_suite's report of the toy: one phase ("default"), h.run_correctness's eager and random rows
    (+ the replay records of a graph deployment)."""
    eager = [{"case": s, "correct": ok, "max_rel_err": err} for s in SIGS]
    eager.append({"case": "output_independence", "correct": True, "max_rel_err": None, "note": "independent"})
    rnd = [{"case": f"random[{i}]:{s}", "correct": ok, "max_rel_err": err, "speedup": None,
            "note": "value-parity vs live baseline (correctness gates; speedup reports)"}
           for s in SIGS for i in range(int(draws or 2))]
    rep = {"eager": eager, "random": rnd}
    if graph:
        rep["graph_replay"] = [{"case": s, "correct": ok, "max_rel_err": err, "group": 0} for s in SIGS]
        rep["replay_groups"] = [{"group": 0, "graphs": [{"m": 1}, {"m": 4}], "carry_state": False, "steps": 2,
                                 "capture_sig": "decode_m4", "rows": 2, "ok": ok}]
        rep["graph_coverage"] = {"capture": GRAPH_CAPTURE, "dims": [], "deployment_graphs": [{"m": 1}, {"m": 4}],
                                 "declared": [{"m": 1}, {"m": 4}], "missing": []}
    return {"default": rep}


def _failing_report():
    return _report(ok=False, err=0.5)


class Fakes:
    """The processes of a UT run. Single rank: harness_lib._run_leg answers every leg_runner.py mode (resolve /
    oracle / refs / correct / list / time) as the worker prints it and writes the files it writes. Multi rank:
    `pool_class()` stands in for ut_runner.RankPool, every rank answering as RankWorker.do + serve_main do."""

    def __init__(self, t):
        self.task = os.path.realpath(t.task)
        self.base = os.path.join(self.task, "baseline_overlay")
        self.cand = os.path.join(self.task, "_cand_overlay")
        self.ids = {"baseline": {"module": "toyop", "file": os.path.join(os.path.realpath(t.site), "toyop.py"),
                                 "qualname": "op"},
                    "candidate": {"module": "toyop", "file": os.path.join(self.cand, "_patched", "toyop.py"),
                                  "qualname": "op"}}
        self.calls = []                     # single-rank legs: (mode, leg, bucket / --control, --inner)
        self.opts = {}                      # mode -> the options of its last leg
        self.raise_on = {}                  # mode or (mode, leg) -> exception the leg raises
        self.overlay_error = None
        self.sigs = list(SIGS)
        self.ms = {"baseline": {"decode_m1": 1.0, "decode_m4": 2.0}, "candidate": {"decode_m1": 0.5, "decode_m4": 1.0}}
        self.row = None                     # (leg, sig, inner) -> time row (default: the CPU wall timer)
        self.refs, self.judged = {}, {}     # fields over every rank's refs / correct result
        self.refs_by_rank, self.judged_by_rank = {}, {}
        self.graph = False
        # multi rank
        self.pools, self.events, self.fail = [], [], {}
        self.device_sync = "none (no CUDA)"
        self.rank_row = None                # (leg, rank, sig, inner) -> time row

    # ------------------------------------------------------------------ harness_lib boundary
    def install(self, monkeypatch):
        real = ur._load

        def load(name, path):
            mod = real(name, path)          # the task's real vendored harness_lib
            mod._run_leg = self.run_leg
            mod.build_candidate_overlay = self.build_overlay
            return mod
        monkeypatch.setattr(ur, "_load", load)
        monkeypatch.setattr(ur, "RankPool", self.pool_class())

    def build_overlay(self, task_dir, meta):
        self.calls.append(("build", None, None, None))
        if self.overlay_error is not None:
            raise self.overlay_error
        return self.base, self.cand

    def leg(self, overlay):
        return "baseline" if overlay == self.base else "candidate"

    def result(self, mode, leg, draws, rank=0):
        res = {"rank": rank, "mode": mode, "identity": {"toyop:op": dict(self.ids[leg])}, "ok": True,
               "incomplete": [], "errors": [], "report": _report(draws=draws, graph=self.graph)}
        if mode == "refs":
            res["native_noise"] = {"max_rel_err": 0.0, "cases": len(SIGS) * int(draws or 2), "noise_rel": 0.0}
            res.update(self.refs)
            res.update(self.refs_by_rank.get(rank) or {})
        else:
            res.update(self.judged)
            res.update(self.judged_by_rank.get(rank) or {})
        return res

    def run_leg(self, task_dir, overlay, mode, *, bucket="", out="", seed=0, draws=0, timeout=3600,
                target_callable="", extra=(), env_extra=None):
        assert os.path.realpath(task_dir) == self.task and timeout > 0
        leg = self.leg(overlay)
        args = [str(x) for x in extra]
        opts = dict(zip(args[::2], args[1::2]))
        self.calls.append((mode, leg, bucket or opts.get("--control"), opts.get("--inner")))
        self.opts[mode] = opts
        exc = self.raise_on.get((mode, leg)) or self.raise_on.get(mode)
        if exc is not None:
            raise exc
        if mode == "resolve":
            return dict(self.ids[leg])
        if mode == "oracle":
            _touch(out)
            return {"out": out, "n": len(SIGS) * draws}
        if mode in ("refs", "correct"):
            res = self.result(mode, leg, int(opts["--draws"]))
            with open(opts["--out"], "w") as fh:
                json.dump(res, fh)
            if mode == "refs":
                _touch(opts["--blob"] + ".refs.pt")
            return {"out": opts["--out"], "ok": res.get("ok")}
        if mode == "list":
            return {"sigs": list(self.sigs)}
        if mode == "time":
            inner = int(opts.get("--inner") or 0)
            if self.row is not None:
                row = self.row(leg, bucket, inner)
            else:
                ms = self.ms[leg][bucket]
                row = {"sig": bucket, "regime": "decode", "m": M[bucket], "ms": ms, "wall_ms": ms, "timer": "wall",
                       "cache_condition": dict(POLICY)}
            if inner:
                row["inner"] = inner
            return {"cases": [row], "identity": dict(self.ids[leg])}
        raise AssertionError(f"unexpected leg mode {mode}")

    def leg_modes(self):
        return [c[:2] for c in self.calls]

    # ------------------------------------------------------------------ rank pools
    def pool_class(self):
        farm = self

        class FakePool:
            def __init__(self, task, overlay, *, label, world, devices, setup_s, command_s, deadline,
                         env_extra=None):
                self.task, self.overlay, self.label, self.world = task, overlay, label, world
                self.devices, self.setup_s, self.command_s, self.deadline = devices, setup_s, command_s, deadline
                self.env_extra = dict(env_extra or {})
                self.closed = False
                farm.pools.append(self)
                farm.events.append(("open", label))

            def _check(self, what):
                err = farm.fail.get((self.label, what))
                if err is not None:
                    self.close(kill=True)        # a failing pool ends itself (RankPool.wait_ready / run)
                    raise err

            def wait_ready(self):
                self._check("setup")
                return [{"rank": r, "setup_s": 0.1, "data_rank": r, "backend": "gloo", "cuda": False}
                        for r in range(self.world)]

            def run(self, mode, **params):
                farm.events.append(("run", self.label, mode))
                self._check(mode)
                return [dict(farm.answer(self, r, mode, params), seconds=0.01) for r in range(self.world)]

            def close(self, kill=False):
                if not self.closed:
                    self.closed = True
                    farm.events.append(("close", self.label))
        return FakePool

    def answer(self, pool, rank, mode, params):
        leg = self.leg(pool.overlay)
        if mode == "list":
            return {"rank": rank, "sigs": list(self.sigs)}
        if mode == "oracle":
            path = f"{params['blob']}.rank{rank}.pt"
            _touch(path)
            return {"rank": rank, "out": path, "n": len(SIGS) * int(params["draws"])}
        if mode in ("refs", "correct"):
            if mode == "refs":
                _touch(f"{params['blob']}.refs.rank{rank}.pt")
            return self.result(mode, leg, params["draws"], rank)
        if mode == "time":
            sig, inner = params["bucket"], max(1, int(params.get("inner") or 1))
            if self.rank_row is not None:
                row = self.rank_row(leg, rank, sig, inner)
            else:
                ms = self.ms[leg][sig]
                row = {"sig": sig, "regime": "decode", "m": M[sig], "ms": ms, "wall_ms": ms, "timer": "wall",
                       "cache_condition": dict(POLICY)}
            row.update(samples=[row["ms"]] * 5, inner=inner,
                       alignment={"samples": 5, "late_releases": 0, "max_late_us": 0.0, "spin_cycles": 0,
                                  "device_sync": self.device_sync})
            return {"rank": rank, "cases": [row], "identity": dict(self.ids[leg])}
        raise AssertionError(f"unexpected pool command {mode}")

    def opened(self):
        return [e[1] for e in self.events if e[0] == "open"]

    def pool(self, label):
        return [p for p in self.pools if p.label == label]


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No ambient GPU lease / pool settings, a clock that never moves, sys.path restored (generate_ut.vendor
    prepends its directory for every fresh task) and sys.modules["harness_lib"] restored (main() loads the task's
    copy under that name)."""
    for k in ur.VISIBILITY_VARS + ("GEAK_UT_ALLOW_CPU_RANKS", "GEAK_UT_POOL_PARALLEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(ur, "time", types.SimpleNamespace(monotonic=lambda: 1000.0))
    saved = sys.modules.get("harness_lib", _MISSING)
    yield
    if saved is _MISSING:
        sys.modules.pop("harness_lib", None)
    else:
        sys.modules["harness_lib"] = saved


@pytest.fixture
def new_task(capsys):
    made = []

    def make(cls=ToyTask, meta_edit=None, cases_extra="", check=True):
        t = cls()
        made.append(t)
        if meta_edit is not None:
            m = t.meta()
            meta_edit(m)
            t.write_meta(m)
        if cases_extra:
            with open(os.path.join(t.task, "cases.py")) as fh:
                t.write("cases.py", fh.read() + cases_extra)
        if check:
            assert gu.main(["all", t.task]) == 0, capsys.readouterr().out
        else:                                   # bypass the generator's own check on purpose
            gu.vendor(t.task)
            gu.freeze(t.task)
        capsys.readouterr()
        return t
    yield make
    for t in made:
        t.cleanup()


@pytest.fixture
def single(monkeypatch, new_task):
    def make(meta_edit=None, cases_extra="", check=True):
        t = new_task(ToyTask, meta_edit, cases_extra, check)
        f = Fakes(t)
        f.install(monkeypatch)
        return t, f
    return make


@pytest.fixture
def multi(monkeypatch, new_task):
    def make(meta_edit=None, cases_extra="", cpu=True):
        t = new_task(RankToyTask, meta_edit, cases_extra)
        f = Fakes(t)
        f.install(monkeypatch)
        if cpu:
            monkeypatch.setenv("GEAK_UT_ALLOW_CPU_RANKS", "1")
        return t, f
    return make


def run_ut(capsys, t, *args):
    """ur.main in-process: (exit code, stdout, the result it wrote, its path); the UT_RESULT line names it."""
    capsys.readouterr()
    argv = ["--task", t.task, *args]
    code = ur.main(argv)
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith("UT_RESULT: ")][-1]
    rec = json.loads(line[len("UT_RESULT: "):])
    assert rec == {"ok": code == 0, "exit_code": code, "result": rec["result"]}
    with open(rec["result"]) as fh:
        res = json.load(fh)
    assert (res["exit_code"], res["ok"], res["smoke"]) == (code, code == 0, "pass" if code == 0 else "fail")
    assert res["command"].endswith(" ".join(argv)) and res["generated_by"] == ur.VERSION
    assert res["wall_s"] == 0.0
    return code, out, res, rec["result"]


def _path(t, *rel):
    return os.path.join(os.path.realpath(t.task), *rel)


# --------------------------------------------------------------------------- where a run writes (A28)
def _ns(**kw):
    base = {"mutation": "", "control": "", "aa": False, "correctness_only": False, "timing_only": False}
    return argparse.Namespace(**dict(base, **kw))


@pytest.mark.parametrize("kw, rel", [
    ({}, "run/ut_result.json"),
    ({"mutation": "leaf_1", "correctness_only": True}, "run/mutations/leaf_1/ut_result.json"),
    ({"correctness_only": True}, "run/ut_partial/correctness-only/ut_result.json"),
    ({"timing_only": True}, "run/ut_partial/timing-only/ut_result.json"),
    ({"aa": True}, "run/ut_partial/aa/ut_result.json"),
    ({"aa": True, "correctness_only": True}, "run/ut_partial/aa-correctness-only/ut_result.json"),
    ({"aa": True, "timing_only": True}, "run/ut_partial/aa-timing-only/ut_result.json"),
    ({"control": "rank_skew", "aa": True}, "run/ut_partial/control-rank_skew/ut_result.json"),
    ({"control": "task:my ctl/v2"}, "run/ut_partial/control-task_my_ctl_v2/ut_result.json"),
])
def test_default_out_only_a_full_run_writes_the_delivered_result(kw, rel):
    assert ur.default_out("/t", _ns(**kw)) == os.path.join("/t", *rel.split("/"))


# --------------------------------------------------------------------------- single rank: passing runs
def test_a_full_single_rank_run_passes_with_its_receipt_coverage_and_native_stamp(single, capsys):
    t, f = single()
    code, out, res, path = run_ut(capsys, t)
    assert code == 0, out
    assert path == _path(t, "run", "ut_result.json")
    assert "reason" not in res and "full_result" not in res
    assert res["gates"] == {"contract": "pass", "legs_differ": "pass", "native_self_check": "pass",
                            "correctness": "pass", "timing": "pass", "receipt": "pass"}
    assert res["contract_sha256"] == t.read_json("run/ut_contract.json")["contract_sha256"]
    assert res["snapshot_sha256"] == ur.source_snapshot(t.task)["snapshot_sha256"]
    assert (res["target_callable"], res["world_size"], res["required_gpu_count"]) == ("toyop:op", 1, 1)
    assert res["timeouts"] == {"setup_s": 900, "command_s": 3600, "total_s": 0}
    assert res["identity"] == {"baseline": f.ids["baseline"], "candidate": f.ids["candidate"]}
    assert res["native_noise"] == {"max_rel_err": 0.0, "cases": 4, "noise_rel": 0.0} and "tolerance" not in res
    assert res["correctness"]["native"]["ok"] is True
    assert res["correctness"]["by_rank"] == [{"rank": 0, "launch_succeeded": True, "numerical_mismatches": 0,
                                              "raised_cases": 0, "failing_cases": []}]
    tm = res["timing"]
    assert [(c["sig"], c["reps"], c["speedup"]) for c in tm["per_case"]] == [("decode_m1", 1, 2.0), ("decode_m4", 1, 2.0)]
    assert tm["weighted"] == tm["primary"] == 2.0 and tm["geomean"] == pytest.approx(2.0)
    assert tm["weighting"] == {"included": 2, "dropped_unserved": [], "suspect_identity": [], "reason": "",
                               "mode": "observed"}
    assert tm["by_phase"] == {"decode": {"speedup": 2.0, "calls": 40.0}}
    rc = res["receipt"]
    assert (rc["version"], rc["world_size"], rc["timer"], rc["alignment"], rc["flags"]) == \
        (ur.RECEIPT_VERSION, 1, "wall", None, [])
    assert rc["cases"]["decode_m4"]["baseline"] == {"ms": 2.0, "timer": "wall"}
    assert rc["cases"]["decode_m1"]["deployed_graph"] is False and rc["cache_mode"] == "read-evict"
    assert "weighting" not in rc and "weighting" not in res
    cov = res["coverage"]
    assert cov["weighting"] == "observed" and cov["graph"] == {} and "ranks" not in cov
    assert cov["calls"] == {"with_calls": list(SIGS), "timed": list(SIGS), "uncovered": [], "declared_uncovered": {}}
    assert all(b["correctness_checked"] == b["correctness"] for b in cov["buckets"])
    # the printed record
    assert "[ut] contract " + res["contract_sha256"] in out and "[ut] snapshot " + res["snapshot_sha256"] in out
    assert "CORRECTNESS: PASS" in out and "GEAK_WEIGHTED_SPEEDUP=2.0" in out and "GEOMEAN_SPEEDUP" not in out
    assert "case decode_m1: baseline_ms=1 optimized_ms=0.5 speedup=2 reps=1 timer=wall" in out
    printed = [ln for ln in out.splitlines() if ln.startswith("GEAK_TIMING_RECEIPT: ")]
    assert json.loads(printed[0][len("GEAK_TIMING_RECEIPT: "):]) == rc
    # the legs, in order: identity on both overlays, oracle + refs on the baseline, the judged candidate, then timing
    assert f.leg_modes()[:8] == [("build", None), ("resolve", "baseline"), ("resolve", "candidate"),
                                 ("oracle", "baseline"), ("refs", "baseline"), ("correct", "candidate"),
                                 ("list", "baseline"), ("list", "baseline")]
    assert sorted(c for c in f.calls if c[0] == "time") == sorted(
        [("time", leg, s, None) for s in SIGS for leg in ("baseline", "candidate")])
    assert f.opts["correct"]["--out"] == _path(t, "run", "ut_native", "judged_candidate.json")
    assert "--control" not in f.opts["correct"] and f.opts["refs"]["--draws"] == "2"
    stamp = t.read_json("run/ut_native/native.json")
    assert stamp["contract_sha256"] == res["contract_sha256"] and stamp["refs"]["ok"] is True


def test_without_workload_weights_the_primary_metric_is_the_unweighted_geomean(single, capsys):
    t, _ = single(meta_edit=lambda m: m.pop("workload"))
    code, out, res, _ = run_ut(capsys, t)
    assert code == 0, out
    assert "GEOMEAN_SPEEDUP=" in out and "GEAK_WEIGHTED_SPEEDUP" not in out
    tm = res["timing"]
    assert tm["weighted"] is None and tm["primary"] == tm["geomean"] == pytest.approx(2.0)
    assert tm["weighting"]["mode"] == "unweighted"
    assert tm["by_phase"] == {"decode": {"speedup": pytest.approx(2.0), "calls": 2, "basis": "unweighted geomean"}}
    assert res["receipt"]["weighting"] == res["weighting"] == res["coverage"]["weighting"] == "unweighted"


def test_an_eager_gpu_reading_on_the_queued_spin_passes_and_carries_its_dispatch_flags(single, capsys):
    """A16: an eager phase timed on the GPU spin is scored on GPU time; a dispatch slower than the kernel is a
    receipt flag, never a failure."""
    t, f = single()

    def row(leg, sig, inner):
        ms = f.ms[leg][sig]
        return {"sig": sig, "regime": "decode", "m": M[sig], "ms": ms, "wall_ms": ms * 2, "timer": "cuda_event",
                "cache_condition": dict(POLICY), "primed": False, "host_ms": ms * 1.5, "spin": dict(SPIN),
                "latency_ms": ms * 1.6}
    f.row = row
    code, out, res, _ = run_ut(capsys, t)
    assert code == 0, out
    rc = res["receipt"]
    assert rc["timer"] == "cuda_event" and rc["all_primed"] is False
    assert rc["cases"]["decode_m1"]["baseline"]["spin"]["queued"] == "gpu_spin_before_start"
    assert rc["cases"]["decode_m1"]["current"]["latency_ms"] == pytest.approx(0.8)
    assert [(x["sig"], x["leg"], x["kind"]) for x in rc["flags"]] == [
        (s, leg, "dispatch_slower_than_kernel") for s in SIGS for leg in ("baseline", "current")]
    assert "primed=False/False host_ms=1.5/0.75" in out


# --------------------------------------------------------------------------- single rank: partial runs (A28)
def test_partial_runs_leave_the_delivered_result_alone_and_say_whether_it_is_current(single, capsys):
    t, _ = single()
    code, out, _, delivered = run_ut(capsys, t)
    assert code == 0, out
    with open(delivered, "rb") as fh:
        before = fh.read()
    code, out, res, path = run_ut(capsys, t, "--correctness-only")
    assert code == 0, out
    assert path == _path(t, "run", "ut_partial", "correctness-only", "ut_result.json")
    assert res["full_result"] == {"path": "run/ut_result.json", "current": True,
                                  "why": "exit 0, same contract and snapshot"}
    assert "[ut] full UT result run/ut_result.json: current (exit 0, same contract and snapshot)" in out
    assert "timing" not in res and "receipt" not in res and "timing" not in res["gates"]
    assert res["coverage"]["calls"]["timed"] == list(SIGS)
    t.write("kernel_src/toyop.py", "import torch\n\n\ndef op(x, w):\n    return torch.tanh(x @ w) + 0\n")
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 0, out
    assert res["full_result"] == {"path": "run/ut_result.json", "current": False, "why": "another source snapshot"}
    assert "[ut] full UT result run/ut_result.json: stale (another source snapshot)" in out
    with open(delivered, "rb") as fh:
        assert fh.read() == before


def test_a_mutation_run_reuses_the_native_references_and_writes_its_own_record(single, capsys):
    t, f = single()
    code, out, _, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 0, out
    f.calls.clear()
    f.judged = {"ok": False, "report": _failing_report()}
    code, out, res, path = run_ut(capsys, t, "--mutation", "toy_leaf")
    assert code == 1, out
    assert path == _path(t, "run", "mutations", "toy_leaf", "ut_result.json")
    assert not os.path.exists(_path(t, "run", "ut_result.json"))
    # correctness only, the native references of the same contract reused: no oracle / refs leg ran
    assert [m for m, _ in f.leg_modes()] == ["build", "resolve", "resolve", "correct", "list"]
    assert res["gates"]["correctness"] == "fail" and "timing" not in res
    assert res["correctness"]["by_rank"][0]["numerical_mismatches"] == 6
    assert res["correctness"]["by_rank"][0]["failing_cases"][:2] == ["decode_m1", "decode_m4"]
    assert res["reason"].startswith("correctness failed: ") and '"max_rel_err": 0.5' in res["reason"]
    assert "CORRECTNESS: FAIL" in out and "FAIL: correctness failed: " in out
    assert res["full_result"] == {"path": "run/ut_result.json", "current": False, "why": "no full UT result"}
    assert res["native_noise"] == {"max_rel_err": 0.0, "cases": 4, "noise_rel": 0.0}     # from the stamp


@pytest.mark.parametrize("leaf", ["../x", "..", "a/b"])
def test_a_mutation_leaf_must_be_a_plain_name(leaf, tmp_path, capsys):
    with pytest.raises(SystemExit) as ei:
        ur.main(["--task", str(tmp_path), "--mutation", leaf])
    assert ei.value.code == 2
    assert "--mutation takes a leaf name of letters, digits" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "run")


def test_an_aa_run_times_the_baseline_against_itself_on_every_pair(single, capsys):
    t, f = single()
    code, out, res, path = run_ut(capsys, t, "--aa", "--no-early-stop", "--reps", "2")
    assert code == 0, out
    assert path == _path(t, "run", "ut_partial", "aa", "ut_result.json")
    assert "identity" not in res and "legs_differ" not in res["gates"]
    assert "resolve" not in [m for m, _ in f.leg_modes()]
    assert {leg for m, leg in f.leg_modes() if m in ("correct", "time")} == {"baseline"}    # both legs: baseline
    assert os.path.isfile(_path(t, "run", "ut_native", "judged_aa.json"))
    assert [c["reps"] for c in res["timing"]["per_case"]] == [2, 2]                     # no early stop
    assert res["timing"]["primary"] == 1.0
    assert res["full_result"]["current"] is False


def test_a_control_run_judges_the_baseline_under_the_control_and_neither_times_nor_records_coverage(single, capsys):
    t, f = single()
    f.judged = {"ok": False, "report": _failing_report()}            # the control is caught
    spec = {"name": "scale_out", "factor": 4}
    code, out, res, path = run_ut(capsys, t, "--control", "scale_out", "--control-spec", json.dumps(spec))
    assert code == 1, out
    assert path == _path(t, "run", "ut_partial", "control-scale_out", "ut_result.json")
    assert f.opts["correct"]["--control"] == "scale_out"
    assert json.loads(f.opts["correct"]["--control-spec"]) == spec
    assert f.opts["correct"]["--out"] == _path(t, "run", "ut_native", "judged_scale_out.json")
    assert [m for m, _ in f.leg_modes()] == ["build", "oracle", "refs", "correct"]     # no identity, list, timing
    assert "coverage" not in res and "timing" not in res and "identity" not in res
    assert res["gates"]["correctness"] == "fail"
    assert res["full_result"]["why"] == "no full UT result"


# --------------------------------------------------------------------------- single rank: legs_differ
@pytest.mark.parametrize("who, ident, code, gate, reason", [
    ("candidate", {"error": "ModuleNotFoundError: No module named 'toyop'"}, 1, "fail",
     "the candidate does not import: candidate leg cannot import toyop:op: ModuleNotFoundError"),
    ("baseline", {"error": "RuntimeError: HIP out of memory"}, 2, "env",
     "the baseline does not import: baseline leg cannot import toyop:op: RuntimeError: HIP out of memory"),
    ("baseline", {"error": "ImportError: cannot import name 'op'"}, 3, "incomplete",
     "the baseline does not import: baseline leg cannot import toyop:op: ImportError"),
    ("candidate", "same", 3, "incomplete", "both legs resolve toyop:op to the SAME code"),
])
def test_the_legs_must_import_different_code(single, capsys, who, ident, code, gate, reason):
    t, f = single()
    f.ids[who] = dict(f.ids["baseline"]) if ident == "same" else ident
    got, out, res, _ = run_ut(capsys, t)
    assert got == code, out
    assert res["gates"]["legs_differ"] == gate
    assert res["reason"].startswith(reason)
    assert {"FAIL": 1, "ENV": 2, "UT_HARNESS_INCOMPLETE": 3}[out.splitlines()[-2].split(":")[0]] == code
    assert [m for m, _ in f.leg_modes()] == ["build", "resolve", "resolve"]
    assert "correctness" not in res and "identity" not in res


def test_an_identity_probe_over_its_time_limit_is_the_environments(single, capsys):
    t, f = single()
    f.raise_on["resolve"] = subprocess.TimeoutExpired(["leg_runner.py"], 600)
    code, out, res, _ = run_ut(capsys, t)
    assert code == 2, out
    assert (res["reason"], res["gates"]["legs_differ"]) == ("identity resolution timed out", "env")
    assert "ENV: identity resolution timed out" in out


# --------------------------------------------------------------------------- single rank: correctness verdicts
def test_a_native_refs_leg_that_crashes_makes_the_ut_incomplete(single, capsys):
    t, f = single()
    f.raise_on["refs"] = RuntimeError("leg(refs) under baseline_overlay exited 1: ZeroDivisionError in cases.py")
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 3, out
    assert res["gates"]["native_self_check"] == "incomplete"
    assert res["reason"].startswith("the NATIVE refs leg crashed running this task's cases: leg(refs)")
    assert "native_noise" not in res and "correctness" not in res


def test_a_native_noisier_than_the_tolerance_stops_the_ut_and_still_records_its_noise(single, capsys):
    """A37 / A42: the native self-check stops the UT (exit 3); the result still has native_noise and the
    per-output tolerance, and the reason names noise_rel and 2 x noise_rel."""
    t, f = single()
    tol = {"tol": 0.01, "per_output": {}, "outputs": [{"output": "0", "tol": 0.01}]}
    f.refs = {"ok": False, "report": _failing_report(), "tolerance": tol,
              "native_noise": {"max_rel_err": 0.5, "cases": 4, "noise_rel": 0.0102}}
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 3, out
    assert res["gates"]["native_self_check"] == "incomplete"
    assert res["native_noise"]["noise_rel"] == 0.0102 and res["tolerance"] == tol
    assert "the NATIVE code fails this task's own checks (baseline refs leg)" in res["reason"]
    assert "noise_rel=0.0102" in res["reason"] and "2 x noise_rel = 0.0204" in res["reason"]
    assert "correct" not in [m for m, _ in f.leg_modes()]


def test_a_native_tolerance_layout_problem_stops_the_ut(single, capsys):
    t, f = single()
    why = "default: per_output names the positions ['2'], but the compared output has 1 tensor(s)"
    f.refs = {"ok": False, "incomplete": [why]}
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert (code, res["reason"], res["gates"]["native_self_check"]) == (3, why, "incomplete"), out


@pytest.mark.parametrize("text, code, gate", [
    ("leg(correct) under _cand_overlay exited 1: Traceback ... IndexError: index 9 is out of bounds", 1, "correctness"),
    ("leg(correct) under _cand_overlay exited 1: torch.OutOfMemoryError: HIP out of memory", 2, "resources"),
])
def test_a_candidate_correctness_process_that_dies(single, capsys, text, code, gate):
    """A crash of the candidate's own code fails it; one that died of its environment is no verdict (exit 2)."""
    t, f = single()
    f.raise_on[("correct", "candidate")] = RuntimeError(text)
    got, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert got == code, out
    assert res["gates"][gate] == {1: "fail", 2: "env"}[code]
    if code == 1:
        assert res["correctness"]["candidate"]["crashed"] is True
        assert res["correctness"]["by_rank"][0]["launch_succeeded"] is False
        assert "candidate correctness process crashed: leg(correct)" in res["reason"]
    else:
        assert res["reason"].startswith("candidate leg (correct) died of its environment (resources)")
        assert "correctness" not in res


def test_a_correctness_worker_that_resolved_another_callable_is_incomplete(single, capsys):
    t, f = single()
    f.judged = {"identity": {"toyop:op": f.ids["baseline"]}}
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 3, out
    assert res["reason"] == (f"the correctness worker resolved {f.ids['baseline']}, not the candidate "
                             f"{f.ids['candidate']}")
    assert res["gates"]["correctness"] == "incomplete"


def test_an_incomplete_judged_leg_stops_the_ut(single, capsys):
    t, f = single()
    f.judged = {"ok": False, "incomplete": ["default: graph coverage: deployment graphs no replay group records"]}
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 3, out
    assert res["reason"] == "default: graph coverage: deployment graphs no replay group records"
    assert res["gates"]["correctness"] == "incomplete"


def test_a_run_that_times_other_cases_than_the_manifest_is_incomplete(single, capsys):
    """ISSUE-006: the timed cases must be the manifest's timing cases."""
    t, f = single()
    f.sigs = ["decode_m1"]
    code, out, res, _ = run_ut(capsys, t)
    assert code == 3, out
    assert res["gates"]["coverage"] == "incomplete"
    assert res["reason"] == ("coverage (ISSUE-006): timed cases ['decode_m1'] differ from the manifest's timing "
                             "cases ['decode_m1', 'decode_m4']")
    assert res["coverage"]["calls"]["timed"] == ["decode_m1"] and res["coverage"]["calls"]["uncovered"] == ["decode_m4"]
    assert "time" not in [m for m, _ in f.leg_modes()]


# --------------------------------------------------------------------------- single rank: timing verdicts
@pytest.mark.parametrize("exc, args, code, gate, reason", [
    (subprocess.TimeoutExpired(["leg_runner.py"], 60), (), 2, "timeout", "a timing leg exceeded its time limit"),
    (RuntimeError("leg(time) under _cand_overlay exited 1: KeyError: 'w'"), (), 1, "timing",
     "timing leg failed: leg(time) under _cand_overlay exited 1: KeyError: 'w'"),
    (RuntimeError("leg(time) under _cand_overlay exited 1: hipErrorNoDevice"), (), 2, "timing",
     "timing leg failed (device): leg(time)"),
    (RuntimeError("leg(time) under baseline_overlay exited 1: KeyError: 'w'"), ("--aa",), 3, "timing",
     "timing leg failed: leg(time) under baseline_overlay"),
])
def test_a_timing_leg_that_fails(single, capsys, exc, args, code, gate, reason):
    t, f = single()
    f.raise_on["time"] = exc
    got, out, res, _ = run_ut(capsys, t, "--timing-only", *args)
    assert got == code, out
    assert res["gates"][gate] == {1: "fail", 2: "env", 3: "incomplete"}[code]
    assert res["reason"].startswith(reason)
    assert "timing" not in res and "correctness" not in res
    assert res["full_result"]["why"] == "no full UT result"


def _gpu_row(f, timer, spin=True, primed=True):
    def row(leg, sig, inner):
        ms = f.ms[leg][sig]
        r = {"sig": sig, "regime": "decode", "m": M[sig], "ms": ms, "wall_ms": ms, "timer": timer,
             "cache_condition": dict(POLICY)}
        if primed is not None:
            r.update(primed=primed, host_ms=0.01)
        if spin and timer == "cuda_event":
            r.update(spin=dict(SPIN), latency_ms=ms)
        return r
    return row


def test_legs_timed_with_different_timers_fail_the_receipt(single, capsys):
    t, f = single()
    gpu = _gpu_row(f, "cuda_event")
    f.row = lambda leg, sig, inner: gpu(leg, sig, inner) if leg == "baseline" else {
        "sig": sig, "regime": "decode", "m": M[sig], "ms": f.ms[leg][sig], "wall_ms": 1.0, "timer": "wall",
        "cache_condition": dict(POLICY)}
    code, out, res, _ = run_ut(capsys, t, "--timing-only")
    assert code == 1, out
    assert res["gates"]["receipt"] == "fail" and "timing" not in res["gates"]
    assert res["reason"].startswith("decode_m1: the legs used different timers (baseline cuda_event, candidate wall)")
    assert res["receipt"]["timer"] == "mixed" and res["timing"]["primary"] == 2.0     # recorded before the verdict


def test_an_eager_gpu_reading_without_the_queued_spin_is_incomplete(single, capsys):
    t, f = single()
    f.row = _gpu_row(f, "cuda_event", spin=False)
    code, out, res, _ = run_ut(capsys, t, "--timing-only")
    assert code == 3, out
    assert res["gates"]["receipt"] == "incomplete"
    assert res["reason"].startswith("decode_m1: the phase is not deployed under a CUDA graph, but the baseline leg "
                                    "was timed without the queued GPU spin (A16)")


def test_a_graph_reading_that_cannot_tell_device_time_is_not_measurable(single, capsys):
    t, f = single(meta_edit=_graph_meta, cases_extra=REPLAY_HOOK)
    f.row = _gpu_row(f, "cuda_event_graph", primed=None)
    code, out, res, _ = run_ut(capsys, t, "--timing-only")
    assert code == 2, out
    assert res["gates"]["receipt"] == "env"
    assert res["reason"].startswith("timing receipt not measurable: decode_m1: baseline leg could not tell whether "
                                    "its reading is device time (no CUDA timeline)")
    assert res["receipt"]["cases"]["decode_m1"]["deployed_graph"] is True


def test_identical_unweighted_readings_leave_no_speedup_to_report(single, capsys):
    """The pseudo-identity guard drops every bucket: the geomean is unavailable (exit 3) with its reason."""
    t, f = single(meta_edit=lambda m: m.pop("workload"))
    f.ms["candidate"] = dict(f.ms["baseline"])
    code, out, res, _ = run_ut(capsys, t, "--timing-only")
    assert code == 3, out
    assert res["gates"]["timing"] == "incomplete" and "receipt" not in res["gates"]
    assert res["reason"].startswith("geomean speedup unavailable: no measurable non-identity bucket survived")
    assert res["timing"]["weighting"]["suspect_identity"] == list(SIGS) and res["timing"]["primary"] is None
    assert "GEOMEAN_SPEEDUP=None" in out


def test_a_positive_mass_case_without_a_candidate_reading_is_a_harness_incomplete_error(single, capsys):
    """The candidate's time_op returned nothing (its call raised): serving_weighted_speedup raises the harness's
    HarnessIncompleteError, which main() reports as UT incomplete without a traceback."""
    t, f = single()
    f.row = lambda leg, sig, inner: {"sig": sig, "regime": "decode", "m": M[sig], "ms": None, "wall_ms": None,
                                     "timer": None, "cache_condition": None} if leg == "candidate" else {
        "sig": sig, "regime": "decode", "m": M[sig], "ms": 1.0, "wall_ms": 1.0, "timer": "wall",
        "cache_condition": dict(POLICY)}
    code, out, res, _ = run_ut(capsys, t, "--timing-only")
    assert code == 3, out
    assert res["reason"] == ("HarnessIncompleteError: Observed timed case decode_m1: positive call mass requires "
                             "both measured latencies")
    assert "timing" not in res and "Traceback" not in res["reason"]
    assert res["coverage"]["calls"]["timed"] is None          # timing-only: the list leg of correctness never ran


def test_a_host_dominated_graph_case_is_raised_and_its_launch_floor_probed(single, capsys):
    """A20 through main: the unprimed case is timed again at inner 2 (early stop), then probed at 16 x for the
    per-replay floor with every pair timed; the receipt records both."""
    t, f = single(meta_edit=_graph_meta, cases_extra=REPLAY_HOOK)
    f.graph = True

    def row(leg, sig, inner):
        ms = {"decode_m1": 0.01, "decode_m4": 1.0}[sig] * (1.0 if leg == "baseline" else 0.5)
        ms *= 0.95 if inner >= 32 else 1.0                  # a small per-replay floor: 5% at inner 2
        primed = not (sig == "decode_m1" and inner <= 1 and leg == "baseline")
        return {"sig": sig, "regime": "decode", "m": M[sig], "ms": ms, "wall_ms": ms, "timer": "cuda_event_graph",
                "cache_condition": dict(POLICY), "primed": primed, "host_ms": 0.001 if primed else 0.02}
    f.row = row
    code, out, res, _ = run_ut(capsys, t)
    assert code == 0, out
    c1 = res["timing"]["per_case"][0]
    assert (c1["sig"], c1["inner"]) == ("decode_m1", 2)
    assert c1["inner_raised"] == {"declared": 1, "final": 2, "tried": [1, 2], "cap": 64, "primed": True,
                                  "reason": "host-dominated at the declared inner=1 (baseline: host_ms=0.02, "
                                            "ms=0.01): raised until both legs read device-dominated"}
    assert (c1["inner_floor"]["probe_inner"], c1["inner_floor"]["dominated"], c1["inner_floor"]["kept"]) == (32, False, 2)
    inners = [c[3] for c in f.calls if c[0] == "time" and c[1] == "baseline"]
    assert inners.count("2") == 1 and inners.count("32") == 3           # the floor probe times every pair
    rc = res["receipt"]["cases"]["decode_m1"]
    assert rc["inner"] == c1["inner_raised"] and rc["inner_floor"] == c1["inner_floor"] and rc["deployed_graph"]
    assert "inner=1->2" in out and "timing both legs again with inner=2" in out
    assert res["coverage"]["graph"]["default"]["deployment_graphs"] == [{"m": 1}, {"m": 4}]
    assert res["coverage"]["graph"]["default"]["missing"] == []


# --------------------------------------------------------------------------- main: the run's own failures
def test_a_changed_test_is_refused_before_any_leg_runs(single, capsys):
    t, f = single()
    with open(os.path.join(t.task, "cases.py")) as fh:
        t.write("cases.py", fh.read() + "\n# edited\n")
    code, out, res, path = run_ut(capsys, t)
    assert code == 3, out
    assert path == _path(t, "run", "ut_result.json")
    assert res["reason"] == "test contract: cases.py: changed after freeze"
    assert res["gates"] == {"contract": "incomplete"}
    assert "snapshot_sha256" not in res and "coverage" not in res and f.calls == []
    assert "UT_HARNESS_INCOMPLETE: test contract: cases.py: changed after freeze" in out


def test_a_partial_run_without_a_frozen_contract_names_no_full_result(single, capsys):
    t, _ = single()
    os.remove(_path(t, "run", "ut_contract.json"))
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 3, out
    assert res["reason"] == "test contract: run/ut_contract.json missing: freeze the task with generate_ut.py freeze"
    assert res["contract_sha256"] is None and "full_result" not in res


def test_a_world_size_that_is_not_an_integer_stops_the_run_without_a_gate(single, capsys):
    t, _ = single()
    m = t.meta()
    m["capabilities"]["multi_rank"]["world_size"] = "two"
    t.write_meta(m)
    code, out, res, _ = run_ut(capsys, t)
    assert code == 3, out
    assert res["reason"] == "capabilities.multi_rank.world_size is not an integer: 'two'"
    assert res["gates"] == {} and res["target_callable"] == "toyop:op" and "world_size" not in res


def test_a_task_without_meta_json_is_incomplete_with_the_error_named(tmp_path, capsys):
    code = ur.main(["--task", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 3
    with open(tmp_path / "run" / "ut_result.json") as fh:
        res = json.load(fh)
    assert res["reason"].startswith("FileNotFoundError: ") and "Traceback" in res["reason"]
    assert "target_callable" not in res and "coverage" not in res and res["gates"] == {}
    assert "UT_HARNESS_INCOMPLETE: FileNotFoundError" in out


def test_an_emptied_kernel_src_is_incomplete(single, capsys):
    t, _ = single()
    os.remove(_path(t, "kernel_src", "toyop.py"))
    code, out, res, _ = run_ut(capsys, t)
    assert code == 3, out
    assert res["reason"].startswith("ValueError: kernel_src contains no source files")
    assert res["gates"] == {"contract": "pass"} and "snapshot_sha256" not in res


def test_an_unexpected_error_with_an_environment_signature_is_the_environments(single, capsys):
    t, f = single()
    f.overlay_error = RuntimeError("candidate overlay build failed: hipErrorNoDevice")
    code, out, res, _ = run_ut(capsys, t)
    assert code == 2, out
    assert res["reason"].startswith("RuntimeError: candidate overlay build failed: hipErrorNoDevice")
    assert "ENV: RuntimeError" in out
    # the harness was loaded: the coverage record is still written (nothing ran yet)
    assert res["coverage"]["calls"]["timed"] is None and "correctness_checked" not in res["coverage"]["buckets"][0]


def test_a_coverage_record_that_cannot_be_built_never_costs_the_verdict(single, capsys):
    t, _ = single(meta_edit=lambda m: m["case_manifest"].update(dims=5), check=False)
    code, out, res, _ = run_ut(capsys, t, "--timing-only")
    assert code == 0, out
    assert res["coverage"] == {"unavailable": "TypeError: 'int' object is not iterable"}


# --------------------------------------------------------------------------- multi rank
def test_a_two_rank_run_passes_with_its_pools_opened_one_at_a_time_and_all_closed(multi, capsys):
    t, f = multi()
    code, out, res, path = run_ut(capsys, t)
    assert code == 0, out
    assert path == _path(t, "run", "ut_result.json")
    assert res["gates"] == {"contract": "pass", "legs_differ": "pass", "native_self_check": "pass",
                            "correctness": "pass", "timing": "pass", "receipt": "pass"}
    assert res["multi_rank"] == {"world_size": 2, "comm": "collective_api", "devices": [], "pool_parallel": 2,
                                 "cpu_ranks": True, "harness_env": {ur.AITER_RAW_POOL: AITER_NOTE}}
    assert (res["world_size"], res["required_gpu_count"]) == (2, 2)
    assert res["timeouts"] == {"setup_s": 120.0, "command_s": 120.0, "total_s": 900.0}
    assert res["identity"] == {"baseline": f.ids["baseline"], "candidate": f.ids["candidate"]}
    # pools: the native pool stays up for the timing (pool_parallel 2), the refs pool lives only for its command
    assert f.opened() == ["native", "native_refs", "candidate"]
    i = f.events.index
    assert i(("close", "native_refs")) < i(("open", "candidate"))
    assert all(p.closed for p in f.pools) and f.events[-2:] == [("close", "native"), ("close", "candidate")]
    assert [(p.label, p.overlay, p.world, p.devices, p.env_extra) for p in f.pools] == [
        (lbl, ov, 2, [], {ur.AITER_RAW_POOL: "1"}) for lbl, ov in
        (("native", f.base), ("native_refs", f.base), ("candidate", f.cand))]
    assert (f.pools[0].setup_s, f.pools[0].command_s, f.pools[0].deadline) == (120.0, 120.0, 1000.0 + 900 - 30)
    assert ("run", "candidate", "list") in f.events
    # per rank
    assert [(r["rank"], r["launch_succeeded"]) for r in res["correctness"]["by_rank"]] == [(0, True), (1, True)]
    assert [r["rank"] for r in res["correctness"]["native"]["ranks"]] == [0, 1]
    assert res["native_noise"] == {"max_rel_err": 0.0, "noise_rel": 0.0}
    rc = res["receipt"]
    assert rc["world_size"] == 2 and rc["cases"]["decode_m4"]["baseline"]["rank_ms"] == [2.0, 2.0]
    al = rc["alignment"]
    assert al["per_sample"].endswith("(no CUDA: no device step) -> start event -> kernel -> end event")
    assert (al["samples"], al["late_releases"], al["device_spin_us"]) == (20, 0, None)
    assert al["harness_env"] == {ur.AITER_RAW_POOL: AITER_NOTE}
    assert res["timing"]["primary"] == 2.0 and res["coverage"]["ranks"] == 2
    assert "CORRECTNESS: PASS" in out and "rank_ms=" in out
    stamp = t.read_json("run/ut_native/native.json")
    assert (stamp["contract_sha256"], stamp["world"]) == (res["contract_sha256"], 2)
    for r in (0, 1):
        assert os.path.isfile(_path(t, "run", "ut_native", f"native.rank{r}.pt"))
        assert os.path.isfile(_path(t, "run", "ut_native", f"native.refs.rank{r}.pt"))


@pytest.mark.parametrize("lease, shown", [(None, "None"), ("3", "'3'")])
def test_ranks_without_enough_leased_gpus_are_an_environment_stop(multi, capsys, monkeypatch, lease, shown):
    t, f = multi(cpu=False)
    if lease:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", lease)
    code, out, res, _ = run_ut(capsys, t)
    assert code == 2, out
    assert res["gates"] == {"contract": "pass", "multi_rank": "env"}
    assert res["reason"] == (f"2 ranks need 2 visible GPUs but HIP_VISIBLE_DEVICES={shown}: run under one group "
                             "lease of the whole set (gpu_lock.sh group:<ids>)")
    assert f.calls == [] and f.pools == []


def test_with_pool_parallel_1_the_correctness_pools_close_before_the_next_one_starts(multi, capsys, monkeypatch):
    """On leased GPUs; GEAK_UT_POOL_PARALLEL=1: the oracle pool closes before the refs pool, the candidate pool
    before the timing pools; the device receipt names the RCCL all-reduce."""
    t, f = multi(cpu=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6")
    monkeypatch.setenv("GEAK_UT_POOL_PARALLEL", "1")
    f.device_sync = "rccl_all_reduce_1"

    def row(leg, rank, sig, inner):
        ms = f.ms[leg][sig]
        return {"sig": sig, "regime": "decode", "m": M[sig], "ms": ms, "wall_ms": ms, "timer": "cuda_event",
                "cache_condition": dict(POLICY), "primed": True, "host_ms": 0.01}
    f.rank_row = row
    code, out, res, _ = run_ut(capsys, t)
    assert code == 0, out
    mr = res["multi_rank"]
    assert (mr["devices"], mr["pool_parallel"], mr["cpu_ranks"]) == (["4", "5"], 1, False)
    assert all(p.devices == ["4", "5", "6"] for p in f.pools)
    # A04 with one pool at a time: every pool closes before the next starts, until the timing needs two
    assert f.events[:12] == [("open", "native"), ("run", "native", "oracle"), ("close", "native"),
                             ("open", "native_refs"), ("run", "native_refs", "refs"), ("close", "native_refs"),
                             ("open", "candidate"), ("run", "candidate", "correct"), ("run", "candidate", "list"),
                             ("close", "candidate"), ("open", "native"), ("open", "candidate")]
    assert f.opened() == ["native", "native_refs", "candidate", "native", "candidate"]
    assert all(p.closed for p in f.pools)
    al = res["receipt"]["alignment"]
    assert "GPU spin 200 us on the rank's stream -> one-element RCCL all_reduce on the pool's device group" \
        in al["per_sample"]
    assert al["device_spin_us"] == 200.0 and al["device_sync"] == "rccl_all_reduce_1"
    assert res["receipt"]["all_primed"] is True


def test_native_references_of_the_same_contract_are_reused_and_a_stale_stamp_is_not(multi, capsys):
    t, f = multi()
    code, out, _, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 0, out
    f.events.clear()
    f.pools.clear()
    code, out, res, _ = run_ut(capsys, t, "--reuse-native")
    assert code == 0, out
    assert f.opened() == ["candidate", "native"]            # the native pool only for the timing
    assert res["correctness"]["native"]["ok"] is True and res["timing"]["primary"] == 2.0
    stamp = _path(t, "run", "ut_native", "native.json")
    rec = t.read_json("run/ut_native/native.json")
    rec["contract_sha256"] = "0" * 64
    with open(stamp, "w") as fh:
        json.dump(rec, fh)
    f.events.clear()
    code, out, res, _ = run_ut(capsys, t, "--reuse-native", "--correctness-only")
    assert code == 0, out
    assert f.opened() == ["native", "native_refs", "candidate"]
    assert t.read_json("run/ut_native/native.json")["contract_sha256"] == res["contract_sha256"]


def test_a_timing_only_run_opens_both_pools_for_the_timing_and_drops_the_deployments_device_selection(
        multi, capsys):
    env = {"HIP_VISIBLE_DEVICES": "0,1,2,3", ur.AITER_RAW_POOL: "1", "VLLM_ROCM_USE_AITER": "1"}
    t, f = multi(meta_edit=lambda m: m["regime"].update(env=env))
    code, out, res, path = run_ut(capsys, t, "--timing-only")
    assert code == 0, out
    assert path == _path(t, "run", "ut_partial", "timing-only", "ut_result.json")
    assert res["multi_rank"]["harness_env"] == {"dropped_from_deployed_env": ["HIP_VISIBLE_DEVICES"]}
    assert all(p.env_extra == {} for p in f.pools)         # the deployment sets the raw pool itself
    assert f.opened() == ["native", "candidate"] and "correctness" not in res
    assert res["receipt"]["alignment"]["harness_env"] == res["multi_rank"]["harness_env"]
    assert res["coverage"]["calls"]["timed"] is None and res["full_result"]["why"] == "no full UT result"


def test_a_rank_skew_control_shifts_the_last_ranks_data_and_is_caught_there(multi, capsys, tmp_path):
    t, f = multi()
    f.judged_by_rank = {1: {"ok": False, "report": _failing_report()}}
    out_path = str(tmp_path / "skew.json")
    native_dir = str(tmp_path / "native_skew")
    code, out, res, path = run_ut(capsys, t, "--control", "rank_skew", "--control-spec", '{"shift": 1}',
                                  "--out", out_path, "--native-dir", native_dir)
    assert code == 1, out
    assert path == out_path
    cand = f.pool("candidate")[0]
    assert cand.env_extra == {"GEAK_UT_DATA_RANK_SHIFT": "1"} and cand.overlay == f.base
    assert all(ur.AITER_RAW_POOL not in p.env_extra for p in f.pools) and res["multi_rank"]["harness_env"] == {}
    assert ("run", "candidate", "list") not in f.events and "coverage" not in res and "identity" not in res
    assert [(r["rank"], r["numerical_mismatches"]) for r in res["correctness"]["by_rank"]] == [(0, 0), (1, 6)]
    assert res["reason"].startswith("correctness failed on some rank: ")
    assert os.path.isfile(os.path.join(native_dir, "native.json")) and res["full_result"]["current"] is False


def test_a_control_that_makes_the_tasks_code_raise_is_recorded_as_raised(multi, capsys):
    t, f = multi()
    tail = "Traceback (most recent call last):\nRuntimeError: index out of range under the control"
    f.fail[("candidate", "correct")] = ur.PoolError("task", "pool candidate: correct: rank 1 raised", 1, tail)
    code, out, res, _ = run_ut(capsys, t, "--control", "scale_out")
    assert code == 1, out
    judged = res["correctness"]["candidate"]
    assert judged["crashed"] is True and judged["ranks"] == [
        {"rank": 1, "ok": False, "errors": [tail], "report": {"raised": {"error": tail}}}]
    assert res["correctness"]["by_rank"] == [{"rank": 1, "launch_succeeded": False, "numerical_mismatches": 0,
                                              "raised_cases": 0, "failing_cases": []}]
    assert "CORRECTNESS: FAIL" in out and res["gates"]["correctness"] == "fail"


def test_a_candidate_rank_that_raises_fails_the_candidate(multi, capsys):
    t, f = multi()
    tail = "Traceback (most recent call last):\nRuntimeError: candidate broke on rank 1"
    f.fail[("candidate", "correct")] = ur.PoolError("task", "pool candidate: correct: rank 1 raised", 1, tail)
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 1, out
    assert res["gates"]["correctness"] == "fail" and "CORRECTNESS: FAIL" in out
    assert res["reason"].startswith('the candidate crashed on some rank: ["pool candidate: correct: rank 1 raised: ')
    assert res["correctness"]["by_rank"][0]["launch_succeeded"] is False
    assert all(p.closed for p in f.pools)


@pytest.mark.parametrize("where, exc, args, code, gate, reason", [
    (("native", "setup"), ur.PoolError("env", "pool native: setup: rank 1 stopped by its environment (resources)",
                                       1, "HIP out of memory"), (), 2, "multi_rank",
     "pool native: setup: rank 1 stopped by its environment (resources) (native): HIP out of memory"),
    (("native_refs", "refs"), ur.PoolError("task", "pool native_refs: refs: rank 0 raised", 0, "ZeroDivisionError"),
     (), 3, "multi_rank",
     ("the native pool's own code failed on this task's cases: pool native_refs: refs: rank 0 raised: "
      "ZeroDivisionError")),
    (("candidate", "correct"), ur.PoolError("timeout", "pool candidate: correct exceeded its time limit "
                                                       "(command_s / total_s)"), (), 2, "multi_rank",
     "pool candidate: correct exceeded its time limit (command_s / total_s) (candidate)"),
    (("candidate", "setup"), ur.PoolError("port", "pool candidate: rendezvous port 30123 taken", 0, ""), (), 2,
     "multi_rank", "pool candidate: rendezvous port 30123 taken (candidate)"),
    (("candidate", "correct"), ur.PoolError("task", "pool candidate: correct: rank 0 raised", 0, "KeyError"),
     ("--aa",), 3, "multi_rank",
     "the aa pool's own code failed on this task's cases: pool candidate: correct: rank 0 raised: KeyError"),
    (("candidate", "time"), ur.PoolError("task", "pool candidate: time: rank 1 raised", 1, "IndexError"), (), 1,
     "correctness", "the candidate failed: pool candidate: time: rank 1 raised: IndexError"),
    # the baseline (native) pool's timing raised: its own code, not the candidate's
    (("native", "time"), ur.PoolError("task", "pool native: time: rank 1 raised", 1, "IndexError", pool="native"),
     (), 3, "multi_rank",
     ("the baseline pool's own code failed on this task's cases: pool native: time: rank 1 raised: IndexError")),
    (("candidate", "time"), ur.PoolError("task", "pool candidate: time: rank 1 raised", 1, "IndexError"),
     ("--aa", "--timing-only"), 3, "multi_rank",
     "the aa pool's own code failed on this task's cases: pool candidate: time: rank 1 raised: IndexError"),
])
def test_a_pool_failure_is_judged_by_whose_pool_failed_and_how(multi, capsys, where, exc, args, code, gate, reason):
    t, f = multi()
    f.fail[where] = exc
    got, out, res, _ = run_ut(capsys, t, *args)
    assert got == code, out
    assert res["reason"] == reason
    assert res["gates"][gate] == {1: "fail", 2: "env", 3: "incomplete"}[code]
    assert f.pools and all(p.closed for p in f.pools)
    assert "timing" not in res


def test_ranks_that_disagree_on_the_timed_cases_make_the_timing_incomplete(multi, capsys):
    t, f = multi()

    def row(leg, rank, sig, inner):
        s = "decode_m4" if rank == 1 else sig
        return {"sig": s, "regime": "decode", "m": M[s], "ms": 1.0, "wall_ms": 1.0, "timer": "wall",
                "cache_condition": dict(POLICY)}
    f.rank_row = row
    code, out, res, _ = run_ut(capsys, t, "--timing-only")
    assert code == 3, out
    assert res["reason"] == "multi-rank timing: the ranks disagree on the timed case order at decode_m1"
    assert res["gates"]["timing"] == "incomplete" and all(p.closed for p in f.pools)


def test_a_native_that_fails_its_own_checks_on_one_rank_stops_the_ut(multi, capsys):
    t, f = multi()
    tol = {"tol": 0.01, "per_output": {}, "outputs": [{"output": "0", "tol": 0.01}]}
    f.refs = {"tolerance": tol}
    f.refs_by_rank = {1: {"ok": False, "report": _failing_report(),
                          "native_noise": {"max_rel_err": 0.5, "cases": 4, "noise_rel": 0.0102}}}
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert code == 3, out
    assert res["gates"]["native_self_check"] == "incomplete"
    assert res["reason"].startswith("the NATIVE code fails this task's own checks on some rank: cases.py or the "
                                    "tolerance is wrong, not the candidate; its self-noise is noise_rel=0.0102")
    assert res["native_noise"]["noise_rel"] == 0.0102 and res["tolerance"] == tol
    assert f.opened() == ["native", "native_refs"]


def test_a_native_layout_problem_on_one_rank_stops_the_ut(multi, capsys):
    t, f = multi()
    why = "default: per_output names the positions ['2'], but the compared output has 1 tensor(s)"
    f.refs_by_rank = {1: {"ok": False, "incomplete": [why]}}
    code, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert (code, res["reason"], res["gates"]["native_self_check"]) == (3, why, "incomplete"), out


@pytest.mark.parametrize("judged, sigs, code, gate, reason", [
    ("baseline_identity", None, 3, "correctness", "the correctness ranks resolved"),
    ({"ok": False, "incomplete": ["default: graph coverage: missing"]}, None, 3, "correctness",
     "default: graph coverage: missing"),
    (None, ["decode_m1"], 3, "coverage", "coverage (ISSUE-006): timed cases ['decode_m1'] differ"),
    ({"ok": False, "report": "failing"}, None, 1, "correctness", "correctness failed on some rank: "),
])
def test_the_judged_ranks_verdicts(multi, capsys, judged, sigs, code, gate, reason):
    t, f = multi()
    if judged == "baseline_identity":
        f.judged = {"identity": {"toyop:op": f.ids["baseline"]}}
    elif judged is not None:
        f.judged_by_rank = {1: dict(judged, report=_failing_report()) if judged.get("report") else judged}
    if sigs:
        f.sigs = sigs
    got, out, res, _ = run_ut(capsys, t, "--correctness-only")
    assert got == code, out
    assert res["reason"].startswith(reason)
    assert res["gates"][gate] == {1: "fail", 3: "incomplete"}[code]
    if code == 1:
        assert [r["numerical_mismatches"] for r in res["correctness"]["by_rank"]] == [0, 6]
        assert "CORRECTNESS: FAIL" in out


@pytest.mark.parametrize("who, ident, code, reason", [
    ("candidate", {"error": "ModuleNotFoundError: No module named 'toyop'"}, 1,
     "the candidate does not import: candidate leg cannot import toyop:op"),
    ("baseline", {"error": "RuntimeError: HIP out of memory"}, 2, "baseline leg cannot import toyop:op"),
    ("candidate", "same", 3, "both legs resolve toyop:op to the SAME code"),
])
def test_the_rank_legs_must_import_different_code(multi, capsys, who, ident, code, reason):
    t, f = multi()
    f.ids[who] = dict(f.ids["baseline"]) if ident == "same" else ident
    got, out, res, _ = run_ut(capsys, t)
    assert got == code, out
    assert res["reason"].startswith(reason)
    assert res["gates"]["legs_differ"] == {1: "fail", 2: "env", 3: "incomplete"}[code]
    assert f.pools == []


def test_a_rank_identity_probe_over_its_time_limit_is_the_environments(multi, capsys):
    """As on one rank: a probe that runs out of time says nothing about the task -- exit 2 with the legs_differ
    gate `env`, not an incomplete UT (exit 3) the agent would be told to fix -- and no pool is opened."""
    t, f = multi()
    f.raise_on["resolve"] = subprocess.TimeoutExpired(["leg_runner.py"], 600)
    code, out, res, _ = run_ut(capsys, t)
    assert code == 2, out
    assert (res["reason"], res["gates"]["legs_differ"]) == ("identity resolution timed out", "env")
    assert "ENV: identity resolution timed out" in out
    assert f.pools == []


def test_a_host_dominated_rank_graph_case_is_raised_over_the_pools(multi, capsys):
    """A20 over the rank pools: primed is the AND over the ranks; the raise and the floor probe re-time both
    pools, and the alignment receipt counts every sample they took."""
    t, f = multi(meta_edit=_graph_meta, cases_extra=REPLAY_HOOK)
    f.graph = True

    def row(leg, rank, sig, inner):
        ms = {"decode_m1": 0.01, "decode_m4": 1.0}[sig] * (1.0 if leg == "baseline" else 0.5)
        ms *= 0.95 if inner >= 32 else 1.0
        primed = not (sig == "decode_m1" and inner <= 1 and leg == "baseline" and rank == 1)
        return {"sig": sig, "regime": "decode", "m": M[sig], "ms": ms, "wall_ms": ms, "timer": "cuda_event_graph",
                "cache_condition": dict(POLICY), "primed": primed, "host_ms": 0.001 if primed else 0.02}
    f.rank_row = row
    code, out, res, _ = run_ut(capsys, t)
    assert code == 0, out
    c1 = res["timing"]["per_case"][0]
    assert (c1["inner"], c1["inner_raised"]["tried"], c1["inner_raised"]["primed"]) == (2, [1, 2], True)
    assert c1["inner_floor"]["probe_inner"] == 32 and c1["inner_floor"]["dominated"] is False
    probes = [e for e in f.events if e == ("run", "native", "time")]
    assert len(probes) == 2 + 1 + 3                       # declared pass, the raise, the 3-pair floor probe
    assert res["receipt"]["alignment"]["samples"] == 5 * 2 * len(probes)
    assert res["receipt"]["cases"]["decode_m1"]["deployed_graph"] is True
