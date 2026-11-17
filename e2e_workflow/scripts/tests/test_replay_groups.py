"""CPU tests of the A18 / A20 additions of the generated UT.

A18 (leg_runner.correctness_suite, harness_lib deployment-graph helpers, generate_ut check, verify_ut):
  - cases.replay may return one group (unchanged) or a list; every group is recorded, replayed and compared
    (the first through run_correctness, the others through check_graph_replay), rows carry their group;
  - every deployment graph of a timing case with call mass must be declared by some group (exit 3 names
    the missing ones); a group capturing on a timing case declares that case's graph;
  - a stateful graph task needs a group carrying state between consecutive replays, whose fill leaves the
    declared state alone after the first step;
  - a stateful task's sequence steps cover its declared stateful tiers;
  - native references, negative controls (incl. skip_state_write on the carry group) apply per group.
A20 (ut_runner.raise_inner): a host-dominated graph case is re-timed on both legs with a doubled inner
until both are device-dominated or the cap is reached; the receipt records it.
A42 (leg_runner per-output tolerances): with meta.tolerance.per_output every comparison -- eager, random draws,
graph replay, sequence, on both legs -- is made output by output at each output's own tolerance and its row names
the output and tolerance; the controls move each output by its own tolerance; the native leg records noise_rel per
output and names a layout per_output does not fit; without per_output nothing changes; a plugin is refused.

The graph replay itself needs a GPU; in the torch classes `check_graph_replay` is replaced by a CPU stand-in that
keeps its call order (fill the capture case, warm-up launches, capture, then fill / replay / read per case). The
CPU-only classes at the end run the real `check_graph_replay` on a fake device (FakeCuda) whose CUDA graphs record
the fake tensors' launches and replay them.
"""
import contextlib
import io
import json
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)
import generate_ut as gu  # noqa: E402
import harness_lib as h  # noqa: E402
import ut_runner as ur  # noqa: E402
import verify_ut as vu  # noqa: E402
from ut_task_fixture import ToyTask, have_torch  # noqa: E402
import leg_runner as lr  # noqa: E402  (last: importing it drops its own directory from sys.path)
sys.path.insert(0, SCRIPTS)

TOL = 1e-4
CAPTURE = {"sizes": [1, 2, 4, 8], "pad": "up"}


def graph_meta(**over):
    m = {"target_callable": "toy:op", "tolerance": {"tol": TOL},
         "regime": {"cuda_graph": True, "graph_capture": dict(CAPTURE)},
         "capabilities": {"stateful": False},
         "case_manifest": {"dims": ["regime", "m"], "cases": [
             {"sig": "d1", "regime": "decode", "m": 1, "roles": ["timing", "correctness"]},
             {"sig": "d3", "regime": "decode", "m": 3, "roles": ["timing", "correctness"]},
             {"sig": "d8", "regime": "decode", "m": 8, "roles": ["timing", "correctness"]}]}}
    m.update(over)
    return m


def cpu_graph_replay(log):
    """harness_lib.check_graph_replay on the CPU: same call order, the 'graph' launches run() eagerly."""
    def check(fill, run, read_out, cases, tol, capture_idx=0, warmup=3, *, strict=False, capture_context=None):
        log.append(("capture", cases[capture_idx].get("sig")))
        fill(cases[capture_idx])
        for _ in range(max(1, warmup)):
            run()
        per, all_ok = [], True
        for c in cases:
            fill(c)
            run()
            ok, err = h.correct(read_out(), c["ref"], tol)       # through h.correct: the RefBook swaps it
            per.append({"case": c.get("sig", ""), "correct": ok, "max_rel_err": round(err, 5),
                        "note": "graph_replay", **h._row_extra(err)})        # A42: as check_graph_replay
            all_ok = all_ok and ok
        return all_ok, per
    return check


class Toy:
    """A graph-replayed toy kernel on the CPU. Stateless: out = x * scale. Stateful: acc += x; out = acc * 2,
    so a step whose state write is dropped shows in the next step."""

    def __init__(self, torch, m, scale=1.0, stateful=False, steps=3, reset_every_step=False):
        self.torch, self.stateful, self.scale = torch, stateful, scale
        self.x = torch.zeros(m, 4)
        self.out = torch.zeros(m, 4)
        self.acc = torch.zeros(m, 4)
        self.reset_every_step = reset_every_step
        g = torch.Generator().manual_seed(m)
        self.cases = [{"sig": f"g{m}_s{k}", "reset": k == 0, "args": torch.randn(m, 4, generator=g)}
                      for k in range(steps)]

    def fill(self, case):
        self.x.copy_(case["args"])
        if self.stateful and (case.get("reset") or self.reset_every_step):
            self.acc.zero_()

    def run(self):
        if self.stateful:
            self.acc.add_(self.x)
            self.out.copy_(self.acc * 2)
        else:
            self.out.copy_(self.x * self.scale)

    def read_out(self):
        return self.out.clone()

    def group(self, graphs, **extra):
        g = {"fill": self.fill, "run": self.run, "read_out": self.read_out, "cases": self.cases,
             "capture_idx": 0, "graphs": graphs}
        if self.stateful:
            g.update(carry_state=True, state=[self.acc])
        g.update(extra)
        return g


def cases_module(make_groups, sequence=None):
    mod = types.ModuleType("toy_cases")
    mod.call = lambda args: args
    mod.eager_cases = lambda hh, meta: []
    mod.random_shapes = lambda hh, meta: []
    mod.replay = lambda hh, meta: make_groups()
    if sequence is not None:
        mod.sequence_cases = lambda hh, meta: sequence()
    return mod


@unittest.skipUnless(have_torch(), "torch not installed")
class TestReplayGroups(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.log = []
        self._saved = h.check_graph_replay
        h.check_graph_replay = cpu_graph_replay(self.log)
        self.addCleanup(setattr, h, "check_graph_replay", self._saved)

    def suite(self, meta, cases, book=None, control=None):
        book = book or lr.RefBook(h)
        res = lr.correctness_suite(h, cases, meta, blob={}, draws=1, book=book, ctx=None, strict=True,
                                   control=control)
        return res, book

    def toys(self, *ms, **kw):
        return [Toy(self.torch, m, **kw) for m in ms]

    def test_the_single_group_form_is_unchanged(self):
        t = Toy(self.torch, 8)
        res, _ = self.suite(graph_meta(), cases_module(lambda: t.group([{"m": 1}, {"m": 4}, {"m": 8}])))
        self.assertTrue(res["ok"], res)
        rep = res["report"]["default"]
        self.assertEqual({r["group"] for r in rep["graph_replay"]}, {0})
        self.assertEqual(len(rep["replay_groups"]), 1)
        self.assertEqual(rep["graph_coverage"]["missing"], [])

    def test_every_group_of_a_list_is_recorded_replayed_and_tagged(self):
        a, b, c = self.toys(1, 4, 8)
        groups = lambda: [a.group([{"m": 1}]), b.group([{"m": 4}]), c.group([{"m": 8}])]   # noqa: E731
        res, _ = self.suite(graph_meta(), cases_module(groups))
        self.assertTrue(res["ok"], res)
        self.assertEqual([x for x in self.log if x[0] == "capture"], [("capture", "g1_s0"), ("capture", "g4_s0"),
                                                                    ("capture", "g8_s0")])
        rep = res["report"]["default"]
        self.assertEqual(sorted({r["group"] for r in rep["graph_replay"]}), [0, 1, 2])
        self.assertEqual([s["graphs"] for s in rep["replay_groups"]], [[{"m": 1}], [{"m": 4}], [{"m": 8}]])
        self.assertTrue(all(s["ok"] and s["steps"] == 3 for s in rep["replay_groups"]))

    def test_a_deployment_graph_no_group_records_is_incomplete_and_named(self):
        a, c = self.toys(1, 8)
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}]), c.group([{"m": 8}])]))
        self.assertFalse(res["ok"])
        msg = " ".join(res["incomplete"])
        self.assertIn("deployment graphs no replay group records: [{'m': 4}]", msg)   # d3 replays in graph 4
        self.assertEqual(res["report"]["default"]["graph_coverage"]["missing"], [{"m": 4}])

    def test_every_group_needs_two_cases_and_graphs_of_the_captured_sizes(self):
        a, b = self.toys(1, 4)
        one = b.group([{"m": 4}], cases=b.cases[:1])
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}, {"m": 8}]), one]))
        self.assertIn("replay group 1 has only 1 replay case(s)", " ".join(res["incomplete"]))
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 3}]), b.group([{"m": 4}])]))
        self.assertIn("m=3, not a captured size", " ".join(res["incomplete"]))
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group(None)]))
        self.assertIn("`graphs` must list the deployment graph", " ".join(res["incomplete"]))

    def test_one_capture_issuing_several_sites_lists_each_of_them(self):
        """A group whose one capture issues every site of the step lists each (m, site) it reproduces."""
        meta = graph_meta(case_manifest={"dims": ["regime", "site"], "cases": [
            {"sig": f"{s}{m}", "regime": "decode", "m": m, "site": s, "roles": ["timing", "correctness"]}
            for s in ("shared", "dense") for m in (1, 2)]})
        a, b = self.toys(1, 2)
        only_m = lambda: [a.group([{"m": 1}]), b.group([{"m": 2}])]        # noqa: E731
        res, _ = self.suite(meta, cases_module(only_m))
        self.assertIn("must have exactly the keys m + the case_manifest dims besides regime and m ['site']",
                      " ".join(res["incomplete"]))
        per_site = lambda: [a.group([{"m": 1, "site": "shared"}, {"m": 1, "site": "dense"}]),   # noqa: E731
                            b.group([{"m": 2, "site": "shared"}])]
        res, _ = self.suite(meta, cases_module(per_site))
        self.assertIn("[{'m': 2, 'site': 'dense'}]", " ".join(res["incomplete"]))
        full = lambda: [a.group([{"m": 1, "site": "shared"}, {"m": 1, "site": "dense"}]),       # noqa: E731
                        b.group([{"m": 2, "site": "shared"}, {"m": 2, "site": "dense"}])]
        res, _ = self.suite(meta, cases_module(full))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["report"]["default"]["graph_coverage"]["dims"], ["site"])

    def test_a_group_capturing_on_a_timing_case_declares_its_graph(self):
        t = Toy(self.torch, 8)
        t.cases[0] = dict(t.cases[0], sig="d3")             # captures on timing case d3 -> deployment graph m=4
        res, _ = self.suite(graph_meta(), cases_module(lambda: t.group([{"m": 1}, {"m": 8}])))
        self.assertIn("captures on timing case d3, whose deployment graph {'m': 4}", " ".join(res["incomplete"]))

    def test_native_references_and_the_candidate_are_compared_group_by_group(self):
        a, b = self.toys(1, 4)
        native = cases_module(lambda: [a.group([{"m": 1}, {"m": 8}]), b.group([{"m": 4}])])
        res, book = self.suite(graph_meta(), native)
        self.assertTrue(res["ok"], res)
        self.assertEqual(len(book.dump()["graph_replay"]), 6)          # 3 steps x 2 groups recorded
        a2, b2 = Toy(self.torch, 1), Toy(self.torch, 4, scale=1.5)     # the candidate is wrong in group 1 only
        cand = cases_module(lambda: [a2.group([{"m": 1}, {"m": 8}]), b2.group([{"m": 4}])])
        res, _ = self.suite(graph_meta(), cand, book=lr.RefBook(h, book.dump()))
        rows = res["report"]["default"]["graph_replay"]
        self.assertFalse(res["ok"])
        self.assertEqual({r["group"] for r in rows if not r["correct"]}, {1})
        self.assertTrue(all(r["correct"] for r in rows if r["group"] == 0))

    def test_negative_controls_act_on_every_group(self):
        a, b = self.toys(1, 4)
        groups = lambda: [a.group([{"m": 1}, {"m": 8}]), b.group([{"m": 4}])]   # noqa: E731
        _, book = self.suite(graph_meta(), cases_module(groups))
        ctl = lr.Control("scale_out", TOL)
        res, _ = self.suite(graph_meta(), cases_module(groups), book=lr.RefBook(h, book.dump()), control=ctl)
        failed = {r["group"] for r in res["report"]["default"]["graph_replay"] if r["correct"] is False}
        self.assertEqual(failed, {0, 1})


def stateful_meta(**over):
    caps = {"stateful": True, "stateful_tiers": [{"name": "fresh", "transition": "first step"},
                                                  {"name": "carry", "transition": "later steps read it"}]}
    return graph_meta(capabilities=caps, **over)


def seq(torch, tiers=("fresh", "carry", "carry")):
    def make():
        out = []
        for i, t in enumerate(tiers):
            step = {"sig": f"s{i}", "args": torch.ones(1, 4) * i, "ref": torch.ones(1, 4) * i}
            if t is not None:
                step["tier" if isinstance(t, str) else "tiers"] = t
            out.append(step)
        return out
    return make


@unittest.skipUnless(have_torch(), "torch not installed")
class TestStateCarryAndTiers(TestReplayGroups.__base__):
    setUp = TestReplayGroups.setUp
    suite = TestReplayGroups.suite

    def all_graphs(self, stateful=True, **kw):
        return Toy(self.torch, 8, stateful=stateful, **kw)

    def test_a_stateful_graph_task_needs_a_group_carrying_state(self):
        t = self.all_graphs(stateful=False)
        res, _ = self.suite(stateful_meta(), cases_module(lambda: t.group([{"m": 1}, {"m": 4}, {"m": 8}]),
                                                          seq(self.torch)))
        self.assertIn("no replay group carries state", " ".join(res["incomplete"]))
        t = self.all_graphs()
        res, _ = self.suite(stateful_meta(), cases_module(lambda: t.group([{"m": 1}, {"m": 4}, {"m": 8}]),
                                                          seq(self.torch)))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["report"]["default"]["replay_groups"][0]["carry_state"], True)

    def test_a_carry_group_without_state_or_whose_fill_rewrites_it_is_incomplete(self):
        t = self.all_graphs()
        res, _ = self.suite(stateful_meta(), cases_module(
            lambda: t.group([{"m": 1}, {"m": 4}, {"m": 8}], state=None), seq(self.torch)))
        self.assertIn("declares carry_state but no `state`", " ".join(res["incomplete"]))
        t = self.all_graphs(reset_every_step=True)
        res, _ = self.suite(stateful_meta(), cases_module(lambda: t.group([{"m": 1}, {"m": 4}, {"m": 8}]),
                                                          seq(self.torch)))
        self.assertIn("fill rewrote the declared state at step(s) [1, 2]", " ".join(res["incomplete"]))

    def test_skip_state_write_is_caught_in_the_carry_group(self):
        t = self.all_graphs()
        mk = lambda: cases_module(lambda: t.group([{"m": 1}, {"m": 4}, {"m": 8}]), seq(self.torch))   # noqa: E731
        res, book = self.suite(stateful_meta(), mk())
        self.assertTrue(res["ok"], res)
        res, _ = self.suite(stateful_meta(), mk(), book=lr.RefBook(h, book.dump()),
                            control=lr.Control("skip_state_write", TOL))
        rep = res["report"]
        rows = rep["default"]["graph_replay"]
        self.assertTrue(rows[0]["correct"])                       # the first step reads the reset state
        self.assertFalse(all(r["correct"] for r in rows[1:]))     # later steps miss the dropped writes
        self.assertEqual(vu.carry_groups(rep), {"default": {0}})
        self.assertTrue(vu.carry_rows_failed(rep, TOL))
        # verify_ut: rejected in the sequence AND in the carry group -> PASS; in the sequence only -> FAIL
        seq_fail = [{"case": "seq[1]:s1", "correct": False, "max_rel_err": 0.5}]
        judged = lambda r: {"correctness": {"candidate": {"report": r}}}   # noqa: E731
        verdict, detail = vu.judge_control("skip_state_write", {}, 1, judged(dict(rep, sequence=seq_fail)), TOL, 1)
        self.assertEqual(verdict, "PASS", detail)
        self.assertTrue(detail["carry_state_rows"])
        clean = json.loads(json.dumps(rep))
        for r in clean["default"]["graph_replay"]:
            r.update(correct=True, max_rel_err=0.0)
        verdict, detail = vu.judge_control("skip_state_write", {}, 1, judged(dict(clean, sequence=seq_fail)), TOL, 1)
        self.assertEqual(verdict, "FAIL", detail)
        self.assertIn("carry_state group", detail["why"])

    def test_stateful_tiers_must_be_covered_by_named_sequence_steps(self):
        t = self.all_graphs()
        groups = lambda: t.group([{"m": 1}, {"m": 4}, {"m": 8}])   # noqa: E731
        res, _ = self.suite(stateful_meta(), cases_module(groups, seq(self.torch)))
        cov = res["report"]["sequence_coverage"]
        self.assertEqual(cov["uncovered"], [])
        self.assertEqual(cov["covered"]["carry"]["steps"], 2)
        for tiers, needle in ((("fresh", "fresh"), "no sequence step covers: ['carry']"),
                              (("fresh", "carry", None), "1 sequence step(s) name no tier"),
                              (("fresh", "carry", "wrap"), "undeclared tiers: ['wrap']"),
                              (("fresh", ["carry", "fresh"]), None)):
            res, _ = self.suite(stateful_meta(), cases_module(groups, seq(self.torch, tiers)))
            msg = " ".join(res["incomplete"])
            if needle:
                self.assertIn(needle, msg)
            else:
                self.assertNotIn("stateful tiers", msg)
        res, _ = self.suite(graph_meta(capabilities={"stateful": True}), cases_module(groups, seq(self.torch)))
        self.assertIn("capabilities.stateful without capabilities.stateful_tiers", " ".join(res["incomplete"]))


class TestDeploymentGraphs(unittest.TestCase):
    def test_observed_cases_with_call_mass_pad_up_to_a_captured_size(self):
        m = graph_meta(workload={"kind": "agentx_trace_replay", "serving_weight_model": {
            "source": "observed_workload", "evidence_path": "w.json"}, "cases": [
            {"sig": "d1", "regime": "decode", "m": 1, "effective_calls": 5},
            {"sig": "d3", "regime": "decode", "m": 3, "effective_calls": 0},
            {"sig": "d8", "regime": "decode", "m": 8, "effective_calls": 7}]})
        info = h.deployment_graphs(m)[""]
        self.assertEqual(info["graphs"], [{"m": 1}, {"m": 8}])        # d3 has no call mass
        self.assertEqual(info["problems"], [])
        unweighted = h.deployment_graphs(graph_meta())[""]
        self.assertEqual(unweighted["graphs"], [{"m": 1}, {"m": 4}, {"m": 8}])

    def test_every_other_manifest_dimension_identifies_the_graph(self):
        """A18.2: the graph a case replays in is M padded plus every other dimension of the case."""
        m = graph_meta(case_manifest={"dims": ["regime", "site", "m"], "cases": [
            {"sig": "a", "regime": "decode", "m": 1, "site": "shared", "roles": ["timing"]},
            {"sig": "b", "regime": "decode", "m": 1, "site": "dense", "roles": ["timing"]},
            {"sig": "c", "regime": "decode", "m": 3, "site": "dense", "roles": ["timing"]}]})
        info = h.deployment_graphs(m)[""]
        self.assertEqual(info["dims"], ["site"])                      # graph_capture.dims left out
        self.assertEqual(info["graphs"], [{"m": 1, "site": "dense"}, {"m": 1, "site": "shared"},
                                          {"m": 4, "site": "dense"}])
        m["regime"]["graph_capture"]["dims"] = []                     # a narrower declaration is refused
        self.assertIn("must be the case_manifest dims besides regime and m, ['site']",
                      " ".join(h.deployment_graphs(m)[""]["problems"]))

    def test_dims_exact_padding_and_cases_above_the_captured_sizes(self):
        m = graph_meta(case_manifest={"dims": ["regime", "d"], "cases": [
            {"sig": "a", "regime": "decode", "m": 1, "d": 384, "roles": ["timing"]},
            {"sig": "b", "regime": "decode", "m": 2, "d": 2560, "roles": ["timing"]}]})
        m["regime"]["graph_capture"]["dims"] = ["d"]
        self.assertEqual(h.deployment_graphs(m)[""]["graphs"], [{"m": 1, "d": 384}, {"m": 2, "d": 2560}])
        m["regime"]["graph_capture"].update(pad="exact", sizes=[1, 4])
        self.assertIn("m=2 has no captured graph", " ".join(h.deployment_graphs(m)[""]["problems"]))
        m = graph_meta()
        m["case_manifest"]["cases"].append({"sig": "d16", "regime": "decode", "m": 16, "roles": ["timing"]})
        self.assertIn("declare it in an eager phase", " ".join(h.deployment_graphs(m)[""]["problems"]))

    def test_phase_regimes_eager_phases_and_missing_declarations(self):
        m = graph_meta(phase_regimes={"prefill": {"cuda_graph": False},
                                      "decode": {"cuda_graph": True}})
        m.pop("regime")
        out = h.deployment_graphs(m)
        self.assertEqual(list(out), ["decode"])
        self.assertIn("graph_capture of phase 'decode' is required", out["decode"]["problems"][0])


class TestGeneratorChecks(unittest.TestCase):
    def setUp(self):
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)

    def problems(self, **meta_over):
        m = self.t.meta()
        m.update(meta_over)
        self.t.write_meta(m)
        return gu.check(self.t.task)

    def test_a_graph_phase_must_declare_its_capture_and_every_timed_m(self):
        res = self.problems(regime={"cuda_graph": True})
        self.assertIn("graph_capture is required", " ".join(res["problems"]))
        res = self.problems(regime={"cuda_graph": True, "graph_capture": {"sizes": [1, 2, 4], "pad": "up"}})
        self.assertFalse([p for p in res["problems"] if "graph_capture" in p or "captured" in p], res)
        res = self.problems(regime={"cuda_graph": True, "graph_capture": {"sizes": [1, 2], "pad": "up"}})
        self.assertIn("timing case decode_m4", " ".join(res["problems"]))     # m=4 above the captured sizes
        res = self.problems(regime={"cuda_graph": True, "graph_capture": {"sizes": [1, 2], "pad": "sideways",
                                                                          "dims": ["m"]}})
        msg = " ".join(res["problems"])
        self.assertIn("pad must be one of", msg)
        self.assertIn("dims must be a list of distinct dimension names other than m", msg)

    def test_a_stateful_task_declares_its_tiers(self):
        caps = dict(self.t.meta()["capabilities"], stateful=True)
        self.assertIn("capabilities.stateful_tiers", " ".join(self.problems(capabilities=caps)["problems"]))
        caps["stateful_tiers"] = [{"name": "wrap", "transition": "ring wraps"}]
        self.assertFalse([p for p in self.problems(capabilities=caps)["problems"] if "stateful_tiers" in p])

    def test_the_workload_is_optional_and_the_inner_cap_is_checked(self):
        m = self.t.meta()
        m.pop("workload")
        self.t.write_meta(m)
        res = gu.check(self.t.task)
        self.assertTrue(res["ok"], res)
        self.assertIn("meta.workload absent", " ".join(res["warnings"]))
        self.assertIn("graph_inner_max_factor must be a positive integer",
                      " ".join(self.problems(timing={"graph_inner_max_factor": 0.5})["problems"]))
        self.assertTrue(self.problems(timing={"graph_inner_max_factor": 16})["ok"])

    def test_the_gpu_count_rule_of_generated_tasks(self):
        """A17.5: a count written in the request equals world_size; world_size > 1 may leave it to RETURN.json."""
        self.t.write("run/requested_kernel.json", json.dumps({"op_kind": "collective"}))
        self.assertTrue(gu.check(self.t.task)["ok"])          # world 1: may be absent (no collective rule)
        self.t.write("run/requested_kernel.json", json.dumps({"required_gpu_count": 2}))
        self.assertIn("required_gpu_count=2 differs", " ".join(gu.check(self.t.task)["problems"]))


class TestRaiseInner(unittest.TestCase):
    """A20 with a fake timer: a leg reads device-dominated once inner reaches `need`."""

    META = {"regime": {"cuda_graph": True}, "phase_regimes": {"decode": {"cuda_graph": True},
                                                             "prefill": {"cuda_graph": False}}}

    def row(self, sig, inner, need, regime="decode", host_ms=0.02, ms=0.01):
        primed = inner >= need
        leg = {"timer": "cuda_event_graph", "primed": primed, "host_ms": host_ms, "cache_condition": {}}
        return {"sig": sig, "regime": regime, "baseline_ms": ms, "optimized_ms": ms, "speedup": 1.0,
                "inner": inner, "legs": {"baseline": dict(leg), "current": dict(leg)}}

    def test_inner_doubles_on_both_legs_until_both_are_primed_and_keeps_the_smallest(self):
        calls = []

        def measure(sig, inner):
            calls.append((sig, inner))
            return [self.row(sig, inner, need=8, ms=0.2)]
        # 0.2 ms windows: too long for the per-replay floor to matter, so only the primed rule acts here
        out = ur.raise_inner(h, self.META, [self.row("a", 1, need=8, ms=0.2), self.row("b", 1, need=1, ms=0.2)],
                             measure, log=lambda *_: None)
        self.assertEqual(calls, [("a", 2), ("a", 4), ("a", 8)])
        a = out[0]
        self.assertEqual(a["inner"], 8)
        self.assertEqual(a["inner_raised"]["tried"], [1, 2, 4, 8])
        self.assertEqual((a["inner_raised"]["declared"], a["inner_raised"]["final"]), (1, 8))
        self.assertTrue(a["inner_raised"]["primed"])
        self.assertNotIn("inner_raised", out[1])                       # already primed: untouched
        rc = ur.build_receipt(out, 1, phases=ur.graph_phases(h, self.META))
        self.assertEqual(rc["cases"]["a"]["inner"]["final"], 8)
        self.assertNotIn("inner", rc["cases"]["b"])
        fail, incomplete, _ = ur.receipt_problems(h, self.META, out, rc)
        self.assertEqual(incomplete, [])

    def test_the_cap_ends_the_doubling_and_the_case_is_named_with_host_ms_and_ms(self):
        meta = dict(self.META, timing={"graph_inner_max_factor": 4})
        out = ur.raise_inner(h, meta, [self.row("a", 2, need=1000)],
                             lambda sig, inner: [self.row(sig, inner, need=1000)], log=lambda *_: None)
        self.assertEqual(out[0]["inner_raised"]["tried"], [2, 4, 8])   # cap = 2 x 4
        self.assertFalse(out[0]["inner_raised"]["primed"])
        rc = ur.build_receipt(out, 1, phases=ur.graph_phases(h, meta))
        _, incomplete, _ = ur.receipt_problems(h, meta, out, rc)
        self.assertIn("a (baseline: host_ms=0.02, ms=0.01, inner 8 = the cap", incomplete[0])

    def test_eager_phases_and_cases_without_call_mass_are_left_alone(self):
        meta = dict(self.META, workload={"kind": "agentx_trace_replay", "serving_weight_model": {
            "source": "observed_workload", "evidence_path": "w"}, "cases": [
            {"sig": "a", "regime": "decode", "effective_calls": 0},
            {"sig": "p", "regime": "prefill", "effective_calls": 5}]})
        rows = [self.row("a", 1, need=8), self.row("p", 1, need=8, regime="prefill")]
        out = ur.raise_inner(h, meta, rows, lambda *a: self.fail("re-timed"), log=lambda *_: None)
        self.assertEqual(out, rows)

    def test_measure_legs_passes_the_inner_to_both_legs_and_old_calls_are_unchanged(self):
        seen = []
        orig = h._run_leg

        def fake(task, overlay, mode, **kw):
            seen.append((overlay, mode, kw.get("bucket"), kw.get("extra")))
            if mode == "list":
                return {"sigs": ["a", "b"]}
            return {"cases": [{"sig": kw["bucket"], "ms": 1.0, "timer": "t", "inner": 4 if kw.get("extra") else None}]}
        h._run_leg = fake
        try:
            rows = h.measure_legs("/t", {}, base="B", cand="C", check_differ=False, early_stop=True,
                                  sigs=["a"], inner={"a": 4})
            self.assertEqual([r["sig"] for r in rows], ["a"])
            self.assertEqual(rows[0]["inner"], 4)
            self.assertIn(("B", "time", "a", ("--inner", "4")), seen)
            self.assertIn(("C", "time", "a", ("--inner", "4")), seen)
            seen.clear()
            rows = h.measure_legs("/t", {}, base="B", cand="C", check_differ=False, max_reps=1)
            self.assertEqual([r["sig"] for r in rows], ["a", "b"])
            self.assertTrue(all(x[3] is None for x in seen))           # no extra argument at all
            self.assertNotIn("inner", rows[0])
        finally:
            h._run_leg = orig

    def test_a_failed_measurement_keeps_the_inner_of_the_reading_kept(self):
        seq = iter([[self.row("a", 2, need=8, ms=0.2)], []])
        out = ur.raise_inner(h, self.META, [self.row("a", 1, need=8, ms=0.2)], lambda sig, inner: next(seq),
                             log=lambda *_: None)
        self.assertEqual((out[0]["inner"], out[0]["inner_raised"]["final"], out[0]["inner_raised"]["tried"]),
                         (2, 2, [1, 2]))


class TestGraphLaunchFloor(unittest.TestCase):
    """A20 for the per-replay launch floor, with a fake graph timer: each leg reads K + E / inner per call (E the
    floor of one replay, K the kernel); the first pass is at the declared inner. Numbers in ms, taken from CI
    round 2 (gptoss add_rmsnorm: ~20 us at inner=1, ~2.9 us at inner=16; gptoss wvSplitK GEMM: 24.9 / 11.1 us)."""

    META = TestRaiseInner.META
    NORM = {"k_b": 0.0017, "e": 0.0189}
    GEMM = {"k_b": 0.0102, "e": 0.0147}

    def row(self, sig, inner, k_b, k_c=None, e=0.0189, regime="decode", timer="cuda_event_graph", need=0):
        k_c = k_b if k_c is None else k_c
        b, c = k_b + e / inner, k_c + e / inner
        leg = {"timer": timer, "primed": inner >= need, "host_ms": 0.005, "cache_condition": {}}
        return {"sig": sig, "regime": regime, "baseline_ms": b, "optimized_ms": c, "speedup": b / c,
                "inner": inner, "legs": {"baseline": dict(leg), "current": dict(leg)}}

    def run_raise(self, rows, model, meta=None):
        calls = []

        def measure(sig, inner):
            calls.append((sig, inner))
            return [self.row(sig, inner, **model[sig])]
        return ur.raise_inner(h, meta or self.META, rows, measure, log=lambda *_: None), calls

    def problems(self, out, meta=None):
        meta = meta or self.META
        return ur.receipt_problems(h, meta, out, ur.build_receipt(out, 1, phases=ur.graph_phases(h, meta)))

    def test_a_reading_that_is_mostly_the_floor_is_raised_until_the_floor_is_a_tenth_or_the_cap(self):
        out, calls = self.run_raise([self.row("a", 1, **self.NORM)], {"a": self.NORM})
        self.assertEqual(calls, [("a", 16), ("a", 64)])         # the probe, then 9 E/K = 100 -> the cap 64
        a = out[0]
        self.assertEqual((a["inner"], a["inner_raised"]["final"], a["inner_raised"]["tried"]), (64, 64, [1, 16, 64]))
        fl = a["inner_raised"]["floor"]
        self.assertEqual((fl["probe_inner"], fl["floor_us"]["baseline"], fl["kernel_us"]["baseline"]), (16, 18.9, 1.7))
        self.assertAlmostEqual(fl["share"]["1"]["baseline"], 0.0189 / 0.0206, places=3)
        self.assertAlmostEqual(fl["share"]["64"]["current"], (0.0189 / 64) / (0.0017 + 0.0189 / 64), places=3)
        self.assertFalse(fl["dominated"])
        self.assertIn("raised to inner=64 (the cap 64)", a["inner_raised"]["reason"])
        self.assertAlmostEqual(a["baseline_ms"], 0.0017 + 0.0189 / 64)
        rc = ur.build_receipt(out, 1, phases=ur.graph_phases(h, self.META))
        self.assertEqual(rc["cases"]["a"]["inner"]["final"], 64)
        self.assertEqual(self.problems(out), ([], [], []))

    def test_the_probe_is_kept_when_it_is_the_inner_needed(self):
        out, calls = self.run_raise([self.row("g", 1, **self.GEMM)], {"g": self.GEMM})
        self.assertEqual(calls, [("g", 16)])                    # 9 E/K = 12.97: the probe (16) holds <= 10%
        g = out[0]
        self.assertEqual((g["inner"], g["inner_raised"]["tried"]), (16, [1, 16]))
        self.assertLessEqual(g["inner_raised"]["floor"]["share"]["16"]["baseline"], ur.FLOOR_TARGET)
        self.assertNotIn("the cap", g["inner_raised"]["reason"])

    def test_a_short_window_whose_floor_is_over_a_tenth_takes_the_probe(self):
        """A20 supplement 2: qwen3 pa_decode bs1, mostly kernel time but 47% floor, is raised to the probe's inner
        (9 E/K = 7.9) with the probe's reading: one timing, no inner_floor."""
        attn = {"k_b": 0.0205, "e": 0.018}                     # 38.5 us, the floor 47% of it
        out, calls = self.run_raise([self.row("p", 1, **attn)], {"p": attn})
        self.assertEqual(calls, [("p", 16)])
        p = out[0]
        self.assertNotIn("inner_floor", p)
        rec = p["inner_raised"]
        self.assertEqual((p["inner"], rec["final"], rec["tried"], rec["floor"]["probe_inner"]), (16, 16, [1, 16], 16))
        self.assertAlmostEqual(p["baseline_ms"], 0.0205 + 0.018 / 16)          # the probe's reading
        self.assertAlmostEqual(rec["floor"]["share"]["1"]["baseline"], 0.018 / 0.0385, places=3)
        self.assertLessEqual(max(rec["floor"]["share"]["16"].values()), ur.FLOOR_TARGET)
        self.assertEqual(self.problems(out), ([], [], []))

    def test_a_short_window_whose_floor_is_at_most_a_tenth_keeps_its_inner(self):
        slow = {"k_b": 0.090, "e": 0.008}                       # 98 us, the floor 8% of it
        first = self.row("p", 1, **slow)
        out, calls = self.run_raise([first], {"p": slow})
        self.assertEqual(calls, [("p", 16)])
        p = out[0]
        self.assertNotIn("inner_raised", p)
        self.assertEqual((p["inner"], p["baseline_ms"]), (1, first["baseline_ms"]))
        self.assertEqual((p["inner_floor"]["kept"], p["inner_floor"]["dominated"]), (1, False))
        self.assertAlmostEqual(p["inner_floor"]["share"]["1"]["baseline"], 0.008 / 0.098, places=3)
        rc = ur.build_receipt(out, 1, phases=ur.graph_phases(h, self.META))
        self.assertEqual(rc["cases"]["p"]["inner_floor"]["probe_inner"], 16)
        self.assertNotIn("inner", rc["cases"]["p"])

    def test_a_floor_over_a_tenth_on_either_leg_raises_and_at_most_a_tenth_keeps(self):
        """A20 supplement 2, the line: a floor of 12% of the reading is raised to the probe (whose floor is under 1%),
        with one timing; one of 8% keeps its inner and reading and is recorded in inner_floor; 8% on the baseline and
        12% on the candidate raises both legs."""
        over = {"k_b": 0.066, "e": 0.009}                       # 75 us, the floor 12% of it
        out, calls = self.run_raise([self.row("o", 1, **over)], {"o": over})
        self.assertEqual(calls, [("o", 16)])
        o = out[0]
        self.assertNotIn("inner_floor", o)
        self.assertEqual((o["inner"], o["inner_raised"]["final"], o["inner_raised"]["tried"]), (16, 16, [1, 16]))
        self.assertAlmostEqual(o["inner_raised"]["floor"]["share"]["1"]["baseline"], 0.12, places=4)
        self.assertAlmostEqual(o["baseline_ms"], 0.066 + 0.009 / 16)
        under = {"k_b": 0.0657, "e": 0.0057}                    # 71.4 us, the floor 8% of it
        first = self.row("u", 1, **under)
        out, calls = self.run_raise([first], {"u": under})
        self.assertEqual(calls, [("u", 16)])
        u = out[0]
        self.assertNotIn("inner_raised", u)
        self.assertEqual((u["inner"], u["baseline_ms"], u["inner_floor"]["kept"]), (1, first["baseline_ms"], 1))
        self.assertAlmostEqual(u["inner_floor"]["share"]["1"]["baseline"], 0.0057 / 0.0714, places=4)
        mixed = {"k_b": 0.1035, "k_c": 0.066, "e": 0.009}       # baseline 112.5 us (8%), candidate 75 us (12%)
        out, calls = self.run_raise([self.row("m", 1, **mixed)], {"m": mixed})
        self.assertEqual(calls, [("m", 16)])
        m = out[0]
        self.assertEqual((m["inner"], m["inner_raised"]["final"]), (16, 16))
        self.assertAlmostEqual(m["inner_raised"]["floor"]["share"]["1"]["baseline"], 0.08, places=4)
        self.assertAlmostEqual(m["inner_raised"]["floor"]["share"]["1"]["current"], 0.12, places=4)
        self.assertAlmostEqual(m["speedup"], (0.1035 + 0.009 / 16) / (0.066 + 0.009 / 16))

    def test_the_inner_does_not_move_with_the_host_floor(self):
        """A20 supplement 2: with a per-replay floor anywhere from 12 to 25 us (the range seen on this host) a case
        whose floor is over a tenth of its reading takes the probe every time, with one timing: gptoss kvwrite
        (declared inner 16, 2.3 us a launch) and glm47 qk-norm (16, 1.6 us) take 256, qwen3 pa_decode bs1 (1,
        21 us) takes 16. Their floor share at the declared inner is 24-54%, across the old line of one half."""
        kinds = {"kvwrite": (16, 0.0023), "glm47_norm": (16, 0.0016), "pa_decode_bs1": (1, 0.021)}
        for kind, (n, k) in kinds.items():
            for e_us in (12, 15, 18, 21, 25):
                model = {"k_b": k, "e": e_us / 1000.0}
                with self.subTest(kind=kind, e_us=e_us):
                    out, calls = self.run_raise([self.row("c", n, **model)], {"c": model})
                    self.assertEqual(calls, [("c", 16 * n)])
                    c = out[0]
                    self.assertNotIn("inner_floor", c)
                    self.assertEqual((c["inner"], c["inner_raised"]["final"]), (16 * n, 16 * n))
                    self.assertLessEqual(max(c["inner_raised"]["floor"]["share"][str(16 * n)].values()),
                                         ur.FLOOR_TARGET)
                    self.assertAlmostEqual(c["baseline_ms"], k + model["e"] / (16 * n))

    def test_a_case_the_probe_cannot_hold_takes_the_smallest_integer_inner(self):
        """A20 supplement 2, the target: gptoss wvSplitK decode_m4_o (8.83 us a launch) needs 9 E/K = 15.8 at a
        15.5 us floor and takes the probe (16, one timing); at 16.0 us it needs 16.3 and is raised to 17, not 32,
        timed once more. The two readings differ by under 1%."""
        low = {"k_b": 0.00883, "e": 0.0155}
        out, calls = self.run_raise([self.row("o", 1, **low)], {"o": low})
        self.assertEqual(calls, [("o", 16)])
        self.assertEqual(out[0]["inner"], 16)
        high = {"k_b": 0.00883, "e": 0.016}
        out2, calls2 = self.run_raise([self.row("o", 1, **high)], {"o": high})
        self.assertEqual(calls2, [("o", 16), ("o", 17)])
        rec = out2[0]["inner_raised"]
        self.assertEqual((out2[0]["inner"], rec["final"], rec["tried"]), (17, 17, [1, 16, 17]))
        self.assertLessEqual(max(rec["floor"]["share"]["17"].values()), ur.FLOOR_TARGET)
        self.assertGreater(0.016 / 16 / (0.00883 + 0.016 / 16), ur.FLOOR_TARGET)  # the probe would not hold it
        self.assertAlmostEqual(out2[0]["baseline_ms"], 0.00883 + 0.016 / 17)
        self.assertLess(abs(out2[0]["baseline_ms"] / out[0]["baseline_ms"] - 1), 0.01)

    def test_floor_inner_is_the_smallest_integer_inner_holding_a_tenth(self):
        self.assertEqual(ur.floor_inner(0.016, 0.00883, 64), 17)            # 9 E/K = 16.31
        self.assertEqual(ur.floor_inner(0.0155, 0.00883, 64), 16)           # 15.80
        self.assertEqual(ur.floor_inner(0.0189, 0.0017, 64), 64)            # 100.06: the cap
        self.assertEqual(ur.floor_inner(0.0189, 0.0017, 1024), 101)
        self.assertEqual(ur.floor_inner(0.0003, 0.0003, 64), 9)             # exactly 9 (9.000000000000002 in floats)
        self.assertEqual(ur.floor_inner(0.0013, 0.0013, 64), 9)
        self.assertEqual(ur.floor_inner(0.02, 0.0, 64), 64)                 # K <= 0: the cap
        self.assertEqual(ur.floor_inner(0.02, -0.001, 64), 64)
        self.assertEqual(ur.floor_inner(0.0, 0.001, 64), 1)
        for e, k in ((0.016, 0.00883), (0.0003, 0.0003), (0.0189, 0.0017), (0.0147, 0.0051)):
            i = ur.floor_inner(e, k, 10 ** 6)
            self.assertLessEqual(e / (i * k + e), ur.FLOOR_TARGET + 1e-12)
            self.assertGreater(e / ((i - 1) * k + e), ur.FLOOR_TARGET)

    def test_only_a_floor_still_over_half_at_the_cap_stops_the_ut(self):
        """Exit 3 is unchanged by A20 supplement 2: gptoss add_rmsnorm decode (1.7 us a launch) reaches the cap 64
        with its floor still 11-16% of the reading (E 14-21.4 us) and passes; a 0.2 us launch whose floor is still
        over half at the cap stops the UT."""
        for e_us in (14.0, 18.9, 21.4):
            model = {"k_b": 0.0017, "e": e_us / 1000.0}
            with self.subTest(e_us=e_us):
                out, calls = self.run_raise([self.row("n", 1, **model)], {"n": model})
                self.assertEqual(calls, [("n", 16), ("n", 64)])
                fl = out[0]["inner_raised"]["floor"]
                self.assertFalse(fl["dominated"])
                self.assertTrue(0.11 <= max(fl["share"]["64"].values()) <= 0.17, fl["share"])
                self.assertEqual(self.problems(out), ([], [], []))
        tiny = {"k_b": 0.0002, "e": 0.02}
        out, calls = self.run_raise([self.row("t", 1, **tiny)], {"t": tiny})
        self.assertEqual(calls, [("t", 16), ("t", 64)])
        self.assertTrue(out[0]["inner_raised"]["floor"]["dominated"])
        _, incomplete, _ = self.problems(out)
        self.assertEqual(len(incomplete), 1)
        self.assertIn("t (inner 64, floor baseline 20.0 us, current 20.0 us per replay", incomplete[0])

    def test_the_floor_still_over_half_at_the_cap_makes_the_ut_incomplete(self):
        tiny = {"k_b": 0.0001, "e": 0.02}
        out, calls = self.run_raise([self.row("t", 1, **tiny)], {"t": tiny})
        self.assertEqual(calls, [("t", 16), ("t", 64)])
        self.assertTrue(out[0]["inner_raised"]["floor"]["dominated"])
        self.assertIn("at inner=64 (the cap 64) it is still", out[0]["inner_raised"]["reason"])
        _, incomplete, _ = self.problems(out)
        self.assertEqual(len(incomplete), 1)
        self.assertIn("per-replay graph launch floor", incomplete[0])
        self.assertIn("t (inner 64, floor baseline 20.0 us, current 20.0 us per replay", incomplete[0])
        # a larger cap lets the same case through
        meta = dict(self.META, timing={"graph_inner_max_factor": 2048})
        out, calls = self.run_raise([self.row("t", 1, **tiny)], {"t": tiny}, meta=meta)
        self.assertEqual(calls, [("t", 16), ("t", 1800)])      # 9 E/K = 1800, the smallest integer inner
        self.assertEqual(self.problems(out, meta), ([], [], []))

    def test_both_legs_get_the_inner_the_faster_one_needs(self):
        fast = dict(self.GEMM, k_c=0.0051)                      # a 2x candidate
        out, calls = self.run_raise([self.row("g", 1, **fast)], {"g": fast})
        self.assertEqual(calls, [("g", 16), ("g", 26)])        # baseline needs 13 (the probe), candidate 9 E/K = 25.9
        self.assertEqual(out[0]["inner"], 26)
        # 1.26 at inner=1: the floor hid the 2x; 1.9002 at inner=26
        self.assertAlmostEqual(out[0]["speedup"], (0.0102 + 0.0147 / 26) / (0.0051 + 0.0147 / 26))
        self.assertGreater(out[0]["speedup"], 1.9)

    def test_long_windows_eager_phases_failed_captures_and_massless_cases_are_not_probed(self):
        meta = dict(self.META, workload={"kind": "agentx_trace_replay", "serving_weight_model": {
            "source": "observed_workload", "evidence_path": "w"}, "cases": [
            {"sig": s, "regime": r, "effective_calls": n} for s, r, n in
            (("long", "decode", 5), ("eager", "prefill", 5), ("eventonly", "decode", 5), ("nomass", "decode", 0))]})
        rows = [self.row("long", 1, k_b=0.2), self.row("eager", 1, regime="prefill", **self.NORM),
                self.row("eventonly", 1, timer="cuda_event", **self.NORM), self.row("nomass", 1, **self.NORM)]
        out = ur.raise_inner(h, meta, rows, lambda *a: self.fail(f"probed {a}"), log=lambda *_: None)
        self.assertEqual(out, rows)
        # a declared inner that already makes the window long enough (64 x 2.0 us) is not probed either
        out, calls = self.run_raise([self.row("a", 64, **self.NORM)], {"a": self.NORM})
        self.assertEqual(calls, [])
        self.assertNotIn("inner_floor", out[0])

    def test_a_primed_raise_is_followed_by_the_floor_check_and_one_record_says_both(self):
        model = dict(self.NORM, need=4)
        out, calls = self.run_raise([self.row("a", 1, **model)], {"a": model})
        self.assertEqual(calls, [("a", 2), ("a", 4), ("a", 64)])   # primed at 4, then the probe at the cap
        rec = out[0]["inner_raised"]
        self.assertEqual((rec["declared"], rec["final"], rec["tried"], rec["cap"]), (1, 64, [1, 2, 4, 64], 64))
        self.assertTrue(rec["primed"])
        self.assertEqual(rec["floor"]["probe_inner"], 64)
        self.assertIn("host-dominated at the declared inner=1", rec["reason"])
        self.assertIn("; then the per-replay graph launch floor was", rec["reason"])

    def test_the_timing_lines_show_the_raise(self):
        out, _ = self.run_raise([self.row("a", 1, **self.NORM)], {"a": self.NORM})
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ur.print_timing(out, 1.0)
        self.assertIn("inner=1->64", buf.getvalue())

    def test_the_floor_timings_run_every_pair_and_the_primed_doubling_does_not(self):
        """The probe and the raise of the floor step go through measure_all (every pair timed: one outlying pair
        of an early-stopped timing would set the floor estimate and the inner); the primed doubling keeps the
        early-stopping measure."""
        model = dict(self.NORM, need=4)
        early, full = [], []

        def measure(sig, inner):
            early.append(inner)
            return [self.row(sig, inner, **model)]

        def measure_all(sig, inner):
            full.append(inner)
            return [self.row(sig, inner, **model)]
        out = ur.raise_inner(h, self.META, [self.row("a", 1, **model)], measure, log=lambda *_: None,
                             measure_all=measure_all)
        self.assertEqual((early, full, out[0]["inner"]), ([2, 4], [64], 64))
        early, full = [], []
        out = ur.raise_inner(h, self.META, [self.row("a", 1, **self.NORM)], measure, log=lambda *_: None,
                             measure_all=measure_all)
        self.assertEqual((early, full, out[0]["inner"]), ([], [16, 64], 64))

    def test_a_probe_or_a_raise_without_a_reading_is_recorded(self):
        """review-20 3.2.4: a probe that returns nothing leaves the case as it was and says so in inner_floor; a
        raise whose timing returns nothing keeps the probe's reading and says so in the floor record."""
        first = self.row("a", 1, **self.NORM)
        out = ur.raise_inner(h, self.META, [first], lambda sig, inner: [], log=lambda *_: None)
        self.assertEqual((out[0]["inner"], out[0]["baseline_ms"]), (1, first["baseline_ms"]))
        self.assertEqual(out[0]["inner_floor"], {"probe_inner": 16, "probe_failed": True, "kept": 1})
        rc = ur.build_receipt(out, 1, phases=ur.graph_phases(h, self.META))
        self.assertTrue(rc["cases"]["a"]["inner_floor"]["probe_failed"])
        seq = iter([[self.row("a", 16, **self.NORM)], []])
        out = ur.raise_inner(h, self.META, [self.row("a", 1, **self.NORM)], lambda sig, inner: next(seq),
                             log=lambda *_: None)
        rec = out[0]["inner_raised"]
        self.assertEqual((out[0]["inner"], rec["final"], rec["tried"], rec["floor"]["target_failed"]),
                         (16, 16, [1, 16], 64))
        self.assertAlmostEqual(out[0]["baseline_ms"], 0.0017 + 0.0189 / 16)


class _RankPoolStub:
    """A timing pool of `world` ranks (ut_runner.measure_pools' view of RankPool): `list`, and `time` of one bucket
    whose per-call reading on every rank is K + E / inner (K scaled by `speed`: a faster candidate pool)."""

    def __init__(self, model, world=2, speed=1.0):
        self.model, self.world, self.calls, self.speed = model, world, [], speed

    def run(self, mode, **params):
        if mode == "list":
            return [{"rank": r, "sigs": list(self.model)} for r in range(self.world)]
        sig, inner = params["bucket"], int(params.get("inner") or 1)
        self.calls.append((sig, inner))
        ms = self.model[sig]["k"] / self.speed + self.model[sig]["e"] / inner
        row = {"sig": sig, "regime": "decode", "m": 2, "timer": "cuda_event_graph", "cache_condition": {},
               "ms": ms, "wall_ms": ms, "samples": [ms] * 5, "inner": inner, "primed": True, "host_ms": 0.004,
               "alignment": {"samples": 5, "late_releases": 0, "max_late_us": 0.0, "device_sync": "rccl_all_reduce_1"}}
        return [{"rank": r, "cases": [dict(row)]} for r in range(self.world)]


class TestGraphLaunchFloorMultiRank(unittest.TestCase):
    """review-20 3.2.3: the floor step also runs for a multi-rank task. ut_runner._multi_rank times both rank pools
    with measure_pools and hands raise_inner a measure that times one case again on both pools with the given
    inner (its `again`); this drives the same composition with two two-rank pool stubs."""

    META = TestRaiseInner.META

    def run_pools(self, model, speed=1.0, reps=1):
        pn, pc = _RankPoolStub(model), _RankPoolStub(model, speed=speed)
        per_case, aligned = ur.measure_pools(pn, pc, early_stop=True, max_reps=reps)

        def again(sig, inner, early_stop=True):    # as in ut_runner._multi_rank
            rows, more = ur.measure_pools(pn, pc, early_stop=early_stop, max_reps=reps, sigs=[sig],
                                          inner={sig: inner})
            aligned.extend(more)
            return rows
        out = ur.raise_inner(h, self.META, per_case, again, log=lambda *_: None,
                             measure_all=lambda sig, inner: again(sig, inner, early_stop=False))
        return out, pn, pc, aligned

    def test_both_pools_are_probed_and_raised_together(self):
        out, pn, pc, aligned = self.run_pools({"a": {"k": 0.0017, "e": 0.0189}})
        self.assertEqual(pn.calls, [("a", 1), ("a", 16), ("a", 64)])
        self.assertEqual(pc.calls, pn.calls)
        a = out[0]
        self.assertEqual((a["inner"], a["inner_raised"]["floor"]["probe_inner"]), (64, 16))
        self.assertEqual(a["rank_ms"]["baseline"], [0.0017 + 0.0189 / 64] * 2)    # the per-rank times of the kept reading
        self.assertEqual(len(aligned), 6)                                             # every timing's alignment kept

    def test_the_floor_timings_of_the_pools_run_every_pair(self):
        """A 2x candidate pool is decisive at the probe inner after one pair; the floor step still times every pair
        on both pools (the first pass at the declared inner stops early as before)."""
        out, pn, pc, _ = self.run_pools({"a": {"k": 0.0017, "e": 0.0189}}, speed=2.0, reps=3)
        self.assertEqual(pn.calls, [("a", 1)] * 3 + [("a", 16)] * 3 + [("a", 64)] * 3)
        self.assertEqual(pc.calls, pn.calls)
        self.assertEqual((out[0]["inner"], out[0]["reps"]), (64, 3))

    def test_an_aligned_start_hides_the_floor_and_the_case_keeps_its_inner(self):
        """A multi-rank sample starts after a 200 us GPU spin and an all-reduce queued before the start event, so
        the graph's launch latency is not in the window (E ~ 0): an 87 us case (mimo decode_m2) is probed once and
        kept."""
        out, pn, _pc, _ = self.run_pools({"decode_m2": {"k": 0.0867, "e": 0.0}})
        self.assertEqual(pn.calls, [("decode_m2", 1), ("decode_m2", 16)])
        m = out[0]
        self.assertNotIn("inner_raised", m)
        self.assertEqual((m["inner"], m["inner_floor"]["kept"], m["inner_floor"]["dominated"]), (1, 1, False))
        rc = ur.build_receipt(out, 2, phases=ur.graph_phases(h, self.META))
        self.assertEqual(rc["cases"]["decode_m2"]["inner_floor"]["floor_us"], {"baseline": 0.0, "current": 0.0})

    def test_an_aligned_start_with_a_residual_floor_under_a_tenth_keeps_its_inner(self):
        """A20 supplement 2 leaves the multi-rank case alone: a few us that an aligned start may still leave in the
        window (3 us on 87 us, 3.5%) is far under the 10% line."""
        out, pn, pc, _ = self.run_pools({"decode_m2": {"k": 0.0837, "e": 0.003}})
        self.assertEqual(pn.calls, [("decode_m2", 1), ("decode_m2", 16)])
        self.assertEqual(pc.calls, pn.calls)
        m = out[0]
        self.assertNotIn("inner_raised", m)
        self.assertEqual((m["inner"], m["inner_floor"]["kept"]), (1, 1))
        self.assertAlmostEqual(m["inner_floor"]["share"]["1"]["baseline"], 0.003 / 0.0867, places=3)


class TwoOut:
    """A two-output kernel on the CPU (A42): a bf16 hidden state and an f32 mixing weight. `drift` scales the f32
    output (a candidate that is 0.5% off there), `wobble` moves it by a relative amount that changes with every call
    (run-to-run noise of the native code)."""

    def __init__(self, torch, drift=1.0, wobble=0.0, hidden_drift=1.0):
        self.torch, self.drift, self.wobble, self.calls = torch, drift, wobble, 0
        self.hidden_drift = hidden_drift
        self.x = torch.zeros(4, 8)
        self.static = None

    def call(self, x):
        torch = self.torch
        self.calls += 1
        mix = torch.sigmoid(x).float() * self.drift * (1.0 + self.wobble * (self.calls % 3))
        return ((x * 2 * self.hidden_drift).to(torch.bfloat16), mix)

    def group(self, cases):
        def fill(case):
            self.x.copy_(case["args"])

        def run():
            self.static = self.call(self.x)

        def read_out():
            return tuple(t.clone() for t in self.static)
        return {"fill": fill, "run": run, "read_out": read_out, "cases": cases, "capture_idx": 0}


def two_out_cases(torch, kernel, random=True):
    g = torch.Generator().manual_seed(5)
    eager = [{"sig": f"e{i}", "args": torch.randn(4, 8, generator=g)} for i in range(2)]
    replay = [{"sig": f"g{i}", "args": torch.randn(4, 8, generator=g)} for i in range(2)]
    steps = [{"sig": f"s{i}", "args": torch.randn(4, 8, generator=g)} for i in range(2)]
    mod = types.ModuleType("two_out_cases")
    mod.call = kernel.call
    mod.eager_cases = lambda hh, meta: [dict(c) for c in eager]
    mod.random_shapes = lambda hh, meta: [{"sig": "r", "make_inputs": lambda rng: torch.randn(4, 8, generator=rng)}
                                          ] if random else []
    mod.replay = lambda hh, meta: kernel.group([dict(c) for c in replay])
    mod.sequence_cases = lambda hh, meta: [dict(c) for c in steps]
    return mod


def two_out_meta(**tol):
    return {"target_callable": "toy:op", "tolerance": dict({"tol": 0.02}, **tol),
            "regime": {"cuda_graph": True}, "capabilities": {"stateful": False}}


class ZeroCountOut(TwoOut):
    """TwoOut returning a dict whose first output (sorted key) is an integer count that is always 0, as
    dsv41_mono_r1_s4's cache_stray_bytes (PF-03): {"cache_stray_bytes", "hid", "mix"}."""

    def call(self, x):
        hid, mix = super().call(x)
        return {"cache_stray_bytes": self.torch.zeros(2, dtype=self.torch.int64), "hid": hid, "mix": mix}

    def group(self, cases):
        g = super().group(cases)
        g["read_out"] = lambda: {k: t.clone() for k, t in self.static.items()}
        return g


@unittest.skipUnless(have_torch(), "torch not installed")
class TestPerOutputTolerances(unittest.TestCase):
    """A42: per-output tolerances in leg_runner's correctness suite (CPU graph replay as in TestReplayGroups)."""
    SECTIONS = ("eager", "random", "graph_replay", "sequence")

    def setUp(self):
        import torch
        self.torch = torch
        self._saved = h.check_graph_replay, h.check_random_vs_baseline, h.correct
        h.check_graph_replay = cpu_graph_replay([])
        self.addCleanup(self._restore)
        native = TwoOut(torch)                      # the native oracle of the random draws, as the oracle leg makes it
        self.blob = {f"r|{i}": native.call(torch.randn(4, 8, generator=torch.Generator().manual_seed(i)))
                     for i in range(2)}

    def _restore(self):
        h.check_graph_replay, h.check_random_vs_baseline, h.correct = self._saved

    def leg(self, meta, kernel, book, native, noise=None):
        """One correctness worker, as leg_runner._correct_mode runs it (refs: native, records; correct: candidate)."""
        tol = lr.task_tolerance(meta, h, native=native)
        saved = h.check_random_vs_baseline
        if noise is not None:
            h.check_random_vs_baseline = lr._recording_noise(h, noise)
        try:
            res = lr.correctness_suite(h, two_out_cases(self.torch, kernel), meta, blob=self.blob, draws=2,
                                       book=book, ctx=None, strict=False, tol=tol)
        finally:
            h.check_random_vs_baseline = saved
        return res, tol

    def legs(self, meta, drift, **candidate):
        book = lr.RefBook(h)
        native, ntol = self.leg(meta, TwoOut(self.torch), book, True)
        self.assertTrue(native["ok"], native)
        judged, _ = self.leg(meta, TwoOut(self.torch, drift=drift, **candidate), lr.RefBook(h, book.dump()), False)
        return native, judged, ntol

    def rows(self, res):
        rep = res["report"]
        out = {s: [r for r in rep["default"].get(s, []) if r["case"] != "output_independence"]
               for s in self.SECTIONS[:3]}
        out["sequence"] = rep["sequence"]
        return out

    def test_an_f32_output_half_a_percent_off_fails_every_comparison_at_its_own_tolerance(self):
        native, judged, tol = self.legs(two_out_meta(per_output={"1": 0.001}), drift=1.005)
        self.assertIsInstance(tol, lr.TaskTol)
        self.assertFalse(judged["ok"])
        self.assertEqual(judged["incomplete"], [])
        for section, rows in self.rows(judged).items():
            self.assertTrue(rows, section)
            for r in rows:
                self.assertFalse(r["correct"], (section, r))
                self.assertEqual((r["output"], r["tol"]), ("1", 0.001), (section, r))
                self.assertGreater(r["max_rel_err"], 0.001)
        self.assertEqual(tol.record()["outputs"], [{"output": "0", "tol": 0.02}, {"output": "1", "tol": 0.001}])
        # the same candidate at one tolerance, 0.02 for every output: passes everywhere, rows as before
        _, judged, tol = self.legs(two_out_meta(), drift=1.005)
        self.assertNotIsInstance(tol, lr.TaskTol)
        self.assertTrue(judged["ok"], judged)
        for section, rows in self.rows(judged).items():
            self.assertTrue(rows and all(r["correct"] for r in rows), section)
            self.assertTrue(all("output" not in r and "tol" not in r for r in rows), section)

    def test_a_row_names_the_output_furthest_over_its_own_tolerance_not_the_largest_error(self):
        """The bf16 output 1% off is within its 0.02 and has the larger error; the f32 one 0.5% off is over its 0.001:
        the row records the f32 output's error and tolerance, so the failing row reads above its tolerance."""
        _, judged, _ = self.legs(two_out_meta(per_output={"1": 0.001}), drift=1.005, hidden_drift=1.01)
        rows = [r for rs in self.rows(judged).values() for r in rs]
        self.assertTrue(rows)
        for r in rows:
            self.assertFalse(r["correct"])
            self.assertEqual((r["output"], r["tol"]), ("1", 0.001), r)
            self.assertGreater(r["max_rel_err"], r["tol"])

    def test_the_native_leg_checks_a_cases_own_ref_at_each_outputs_tolerance(self):
        """A case's own `ref` (an independent reference) is compared on the native refs leg output by output too: one
        0.5% off in the f32 output fails the native self-check at 0.001 and passes at one tol 0.02."""
        torch = self.torch
        kernel = TwoOut(torch)
        cases = two_out_cases(torch, kernel)
        eager = cases.eager_cases(h, {})
        for c in eager:
            hid, mix = kernel.call(c["args"])
            c["ref"] = (hid.clone(), mix * 1.005)
        cases.eager_cases = lambda hh, meta: [dict(c) for c in eager]
        for meta, ok in ((two_out_meta(per_output={"1": 0.001}), False), (two_out_meta(), True)):
            res = lr.correctness_suite(h, cases, meta, blob=self.blob, draws=2, book=lr.RefBook(h), ctx=None,
                                       strict=False, tol=lr.task_tolerance(meta, h, native=True))
            self.assertEqual(res["ok"], ok, res)
            rows = [r for r in res["report"]["default"]["eager"] if r["case"] != "output_independence"]
            self.assertEqual([r["correct"] for r in rows], [ok, ok])
            if not ok:
                self.assertTrue(all((r["output"], r["tol"]) == ("1", 0.001) for r in rows), rows)

    def test_a_correct_candidate_passes_and_its_rows_name_the_output_nearest_its_tolerance(self):
        native, judged, _ = self.legs(two_out_meta(per_output={"1": 0.001}), drift=1.0)
        self.assertTrue(judged["ok"], judged)
        rows = [r for rs in self.rows(judged).values() for r in rs]
        self.assertTrue(rows and all(r["correct"] and r["max_rel_err"] == 0.0 for r in rows))
        self.assertTrue(all(r["output"] in ("0", "1") for r in rows))

    def zero_count_legs(self, meta, drifts):
        """The native refs leg of ZeroCountOut (recording noise_rel, as _correct_mode does) and a candidate leg for
        each mix `drift`: (native result, its noise_rel list, [candidate results])."""
        torch = self.torch
        self.blob = {f"r|{i}": ZeroCountOut(torch).call(torch.randn(4, 8, generator=torch.Generator().manual_seed(i)))
                     for i in range(2)}
        book, noise = lr.RefBook(h), []
        native, _ = self.leg(meta, ZeroCountOut(torch), book, True, noise=noise)
        judged = [self.leg(meta, ZeroCountOut(torch, drift=d), lr.RefBook(h, book.dump()), False)[0] for d in drifts]
        return native, noise, judged

    def test_a_failing_comparison_names_the_failing_output_not_one_passing_at_tolerance_0(self):
        """PF-03, test3-e2e-01 section 2 a (review-33 section 2.1 B; dsv41_mono_r1_s4's run/attempt1_tol0): an
        always-zero count compared at tolerance 0 passes reading 0/0 = NaN in correct(). A candidate 5% off in mix
        (0.005) fails every comparison, and every row names mix with a finite error over its tolerance, which verify_ut
        records and holds above its floor (28cb6431 named the count, with no error, in every row)."""
        meta = two_out_meta(per_output={"cache_stray_bytes": 0.0, "mix": 0.005})
        _, _, (judged,) = self.zero_count_legs(meta, (1.05,))
        self.assertFalse(judged["ok"])
        self.assertEqual(judged["incomplete"], [])
        rows = [r for rs in self.rows(judged).values() for r in rs]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual((r["correct"], r["output"], r["tol"]), (False, "mix", 0.005), r)
            self.assertIsInstance(r["max_rel_err"], float, r)
            self.assertGreater(r["max_rel_err"], 0.005, r)
        tol_max = vu.largest_tolerance(meta, 0.02)
        recs = vu.classify_cases(judged["report"], 0.02)
        self.assertEqual({r["kind"] for r in recs}, {"fail"})
        self.assertEqual(vu.record_errors(recs, 0.02, tol_max), [])
        self.assertEqual(vu.static_record_problems(recs, 0.02, tol_max), [])

    def test_an_output_passing_at_tolerance_0_reads_0_and_native_noise_stays_finite(self):
        """PF-03 (review-33 section 2.1 A): the always-zero count at tolerance 0 passes reading 0/0 = NaN in correct();
        it reads 0.0, so every row keeps a finite max_rel_err and native_noise.max_rel_err is finite (28cb6431: every
        row named the count with max_rel_err null, and native_noise.max_rel_err was null). Where every output reads 0
        -- the native leg against its oracle, an identical candidate -- the row names the first output, the count, at
        0.0; a candidate 0.1% off in mix, within 0.005, names mix."""
        meta = two_out_meta(per_output={"cache_stray_bytes": 0.0, "mix": 0.005})
        native, noise, judged = self.zero_count_legs(meta, (1.0, 1.001))
        self.assertTrue(native["ok"], native)
        nrows = native["report"]["default"]["random"]
        self.assertTrue(nrows)
        for r in nrows:
            self.assertEqual((r["correct"], r["max_rel_err"], r["output"], r["tol"]),
                             (True, 0.0, "cache_stray_bytes", 0.0), r)
        nn = lr._native_noise(native["report"], noise)
        self.assertEqual((nn["max_rel_err"], nn["cases"], nn["noise_rel"]), (0.0, len(nrows), 0.0))
        for res, named in zip(judged, ("cache_stray_bytes", "mix")):
            self.assertTrue(res["ok"], res)
            rows = [r for rs in self.rows(res).values() for r in rs]
            self.assertTrue(rows)
            for r in rows:
                self.assertEqual((r["correct"], r["output"]), (True, named), r)
                self.assertIsInstance(r["max_rel_err"], float, r)
                if named == "mix":
                    self.assertTrue(0.0 < r["max_rel_err"] < 0.005, r)
                else:
                    self.assertEqual(r["max_rel_err"], 0.0, r)

    def test_without_per_output_correct_is_called_as_before(self):
        """No per_output: the checks call harness_lib.correct with the whole output and the plain tolerance, exactly as
        before; with per_output, once per output at that output's tolerance."""
        calls = []
        real = self._saved[2]

        def spy(out, ref, tol):
            calls.append((len(h.flatten_outputs(out)), type(tol).__name__, float(tol)))
            return real(out, ref, tol)
        for meta in (two_out_meta(), two_out_meta(per_output=None), two_out_meta(per_output={})):
            self.assertEqual(type(lr.task_tolerance(meta, h)), float)
            calls.clear()
            h.correct = spy
            self.legs(meta, drift=1.0)
            self.assertTrue(calls)
            self.assertEqual({c for c in calls}, {(2, "float", 0.02)})
        calls.clear()
        self.legs(two_out_meta(per_output={"1": 0.001}), drift=1.0)
        self.assertEqual({c for c in calls}, {(1, "float", 0.02), (1, "float", 0.001)})
        h.correct = real

    def test_the_native_leg_records_noise_rel_per_output(self):
        meta = two_out_meta(per_output={"1": 0.001})
        noise = []
        res, tol = self.leg(meta, TwoOut(self.torch, wobble=1e-4), lr.RefBook(h), True, noise=noise)
        self.assertTrue(res["ok"], res)
        rec = tol.noise_record()
        self.assertEqual(set(rec), {"0", "1"})
        self.assertEqual(rec["0"], {"tol": 0.02, "noise_rel": 0.0})
        self.assertEqual(rec["1"]["tol"], 0.001)
        self.assertGreater(rec["1"]["noise_rel"], 0.0)
        self.assertLess(rec["1"]["noise_rel"], 2e-4)
        self.assertEqual(lr._native_noise(res["report"], noise)["noise_rel"], rec["1"]["noise_rel"])

    def test_a_layout_per_output_does_not_fit_is_a_problem_of_the_native_leg_only(self):
        torch = self.torch
        a, b = torch.ones(2, 2), torch.ones(2, 2)
        cases = [({"2": 0.001}, (a, b), "per_output names the positions ['2'], but the compared output has 2 tensor"),
                 ({"mix": 0.001}, (a, b), "is not a dict but 2 tensor(s): name its outputs by position"),
                 ({"0": 0.001}, {"mix": a, "hid": b}, "is a dict (keys ['hid', 'mix']): name its outputs by key"),
                 ({"gate": 0.001}, {"mix": a, "hid": b}, "per_output names ['gate'], which the compared dict")]
        for per, out, words in cases:
            with self.subTest(per=per):
                meta = two_out_meta(per_output=per)
                native = lr.task_tolerance(meta, h, native=True)
                self.assertIsNone(native.fit(out))
                self.assertEqual(len(native.problems), 1)
                self.assertIn(words, native.problems[0])
                candidate = lr.task_tolerance(meta, h)
                self.assertIsNone(candidate.fit(out))
                self.assertEqual(candidate.problems, [])           # the candidate leg never blames the UT
                self.assertEqual(lr.per_output_correct(h, self._saved[2])(out, out, candidate), (False, float("inf")))
        # a dict named by key: every tensor under it, in flatten_outputs order
        t = lr.task_tolerance(two_out_meta(per_output={"mix": 0.001}), h, native=True)
        self.assertEqual(t.fit({"mix": (a, b), "hid": a}), [("hid", 0.02), ("mix[0]", 0.001), ("mix[1]", 0.001)])
        # comparisons that compare different outputs
        t = lr.task_tolerance(two_out_meta(per_output={"0": 0.001}), h, native=True)
        self.assertIsNotNone(t.fit((a, b)))
        self.assertIsNone(t.fit((a, b, a)))
        self.assertIn("need every comparison of the task to compare the same outputs", t.problems[0])
        # in the suite: the native leg is incomplete, naming it
        res, _ = self.leg(two_out_meta(per_output={"2": 0.001}), TwoOut(torch), lr.RefBook(h), True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["incomplete"], ["meta.tolerance.per_output: per_output names the positions ['2'], but "
                                             "the compared output has 2 tensor(s)"])
        # also where the native leg only records (no random draws compare anything there)
        meta = two_out_meta(per_output={"2": 0.001})
        res = lr.correctness_suite(h, two_out_cases(torch, TwoOut(torch), random=False), meta, blob={}, draws=2,
                                   book=lr.RefBook(h), ctx=None, strict=False, tol=lr.task_tolerance(meta, h, True))
        self.assertEqual(res["incomplete"], ["meta.tolerance.per_output: per_output names the positions ['2'], but "
                                             "the compared output has 2 tensor(s)"])
        # a malformed per_output (generate_ut check refuses it too) stops both legs
        res, _ = self.leg(two_out_meta(per_output={"1": -1}), TwoOut(torch), lr.RefBook(h), False)
        self.assertEqual(res["incomplete"], ["meta.tolerance.per_output['1'] must be a finite non-negative number, "
                                             "not -1"])

    def test_the_controls_move_each_output_by_its_own_tolerance(self):
        torch = self.torch
        f, i = torch.ones(3, dtype=torch.float64), torch.arange(3)
        tol = lr.task_tolerance(two_out_meta(per_output={"1": 0.001}), h)
        w = lr.Control("within_tol", tol)._perturb(((f, f.clone()), i), None)
        self.assertEqual([float(w[0][0][0]), float(w[0][1][0])], [1.0 + 0.02 / 10, 1.0 + 0.001 / 10])
        self.assertTrue(torch.equal(w[1], i))
        s = lr.Control("scale_out", tol)._perturb(((f, f.clone()), i), None)
        self.assertEqual([float(s[0][0][0]), float(s[0][1][0])], [1.0 + 4 * 0.02, 1.0 + 4 * 0.001])
        self.assertFalse(torch.equal(s[1], i))                     # a non-float output moved off its values
        # a dict output by key, its keys kept in their order
        tol = lr.task_tolerance(two_out_meta(per_output={"mix": 0.001}), h)
        w = lr.Control("within_tol", tol)._perturb({"mix": f, "hid": f.clone()}, None)
        self.assertEqual(list(w), ["mix", "hid"])
        self.assertEqual((float(w["mix"][0]), float(w["hid"][0])), (1.0 + 0.001 / 10, 1.0 + 0.02 / 10))
        # without per_output: one tolerance for every output, as before
        w = lr.Control("within_tol", 0.02)._perturb((f, f.clone()), None)
        self.assertEqual([float(w[0][0]), float(w[1][0])], [1.0 + 0.02 / 10] * 2)
        # through the suite: within_tol passes and scale_out is rejected, each at its own tolerance
        meta = two_out_meta(per_output={"1": 0.001})
        book = lr.RefBook(h)
        self.leg(meta, TwoOut(torch), book, True)
        for name, ok in (("within_tol", True), ("scale_out", False)):
            ctl = lr.Control(name, lr.task_tolerance(meta, h))
            res = lr.correctness_suite(h, two_out_cases(torch, TwoOut(torch)), meta, blob=self.blob, draws=2,
                                       book=lr.RefBook(h, book.dump()), ctx=None, strict=False, control=ctl,
                                       tol=ctl.outputs)
            self.assertEqual(res["ok"], ok, (name, res))

    def test_a_declared_comparison_plugin_makes_the_generated_ut_incomplete(self):
        for extra in ({"compare": {"file": "cmp.py", "attr": "compare"}},
                      {"capabilities": {"stateful": False, "per_output_compare": True}}):
            meta = dict(two_out_meta(), regime={"cuda_graph": False}, **extra)
            cases = two_out_cases(self.torch, TwoOut(self.torch))
            del cases.replay
            res = lr.correctness_suite(h, cases, meta, blob=self.blob, draws=2, book=lr.RefBook(h), ctx=None,
                                       strict=True)
            self.assertFalse(res["ok"])
            self.assertEqual(len(res["incomplete"]), 1, res["incomplete"])
            self.assertIn("a comparison plugin is not supported", res["incomplete"][0])
            self.assertIn("meta.tolerance.per_output", res["incomplete"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)


# --------------------------------------------------------------------------- two ranks, one capture per group
RANK_SCRIPT = r'''
import contextlib, json, os, sys, types
sys.path.insert(0, os.environ["SCRIPTS"])
import torch
import torch.distributed as dist
import harness_lib as h
import leg_runner as lr

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world)
ctl = lr.RankControl(torch, 60)


class FakeGraph:
    def __init__(self):
        self.ops = []

    def replay(self):
        for op in self.ops:
            op()


class FakeCuda:
    capturing = None
    CUDAGraph = FakeGraph

    def Stream(self):
        return types.SimpleNamespace(wait_stream=lambda *_: None)

    def current_stream(self):
        return types.SimpleNamespace(wait_stream=lambda *_: None)

    @contextlib.contextmanager
    def stream(self, s):
        yield

    def synchronize(self):
        pass

    @contextlib.contextmanager
    def graph(self, g, capture_error_mode=None):
        FakeCuda.capturing = g
        try:
            yield
        finally:
            FakeCuda.capturing = None


fake_torch = types.SimpleNamespace(cuda=FakeCuda())


def group(gi, m):
    x, out = torch.zeros(m, 4), torch.zeros(m, 4)
    gen = torch.Generator().manual_seed(100 + m + rank)          # per-rank data
    cases = [{"sig": f"g{m}_s{k}", "args": torch.randn(m, 4, generator=gen)} for k in range(3)]

    def do():
        out.copy_(x * 3)

    def run():
        if FakeCuda.capturing is not None:
            if os.environ.get("FAIL_CAPTURE") == f"{rank}:{gi}":
                raise RuntimeError("capture fails on this rank only")
            FakeCuda.capturing.ops.append(do)
        else:
            do()
    return {"fill": lambda c: x.copy_(c["args"]), "run": run, "read_out": lambda: out.clone(),
            "cases": cases, "capture_idx": 0, "graphs": [{"m": m}]}


cases = types.ModuleType("rank_cases")
cases.call = lambda a: a
cases.eager_cases = lambda hh, meta: []
cases.random_shapes = lambda hh, meta: []
cases.replay = lambda hh, meta: [group(0, 1), group(1, 4), group(2, 8)]
meta = json.loads(os.environ["META"])
h.check_graph_replay = lr._agreed_graph_replay(h, fake_torch, ctl, cases, 0.0)   # what RankWorker installs
book = lr.RefBook(h)
res = lr.correctness_suite(h, cases, meta, blob={}, draws=1, book=book, ctx=None, strict=True, fail_fast=True)
rep = res["report"]["default"]
with open(os.environ["OUT"], "w") as fh:
    json.dump({"rank": rank, "ok": res["ok"], "incomplete": res["incomplete"],
               "groups": rep["replay_groups"], "rows": rep["graph_replay"],
               "recorded": len(book.dump()["graph_replay"])}, fh)
dist.destroy_process_group()
'''


@unittest.skipUnless(have_torch(), "torch not installed")
class TestTwoRankGroups(unittest.TestCase):
    """Two gloo ranks: every replay group is captured on every rank or on none (leg_runner._agreed_capture),
    group by group, and both ranks record and compare the same comparisons."""

    def run_ranks(self, fail=""):
        import subprocess
        import tempfile
        d = tempfile.mkdtemp(prefix="geak_rank_groups_")
        self.addCleanup(__import__("shutil").rmtree, d, True)
        script = os.path.join(d, "rank.py")
        with open(script, "w") as fh:
            fh.write(RANK_SCRIPT)
        port = ur.free_port()
        procs = []
        for r in range(2):
            env = dict(os.environ, SCRIPTS=SCRIPTS, RANK=str(r), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
                       MASTER_PORT=str(port), META=json.dumps(graph_meta()), OUT=os.path.join(d, f"r{r}.json"),
                       FAIL_CAPTURE=fail, CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1",
                       OMP_NUM_THREADS="1")
            procs.append(subprocess.Popen([sys.executable, script], env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True))
        outs = [p.communicate(timeout=300)[0] for p in procs]
        for p, o in zip(procs, outs):
            self.assertEqual(p.returncode, 0, o[-3000:])
        return [json.load(open(os.path.join(d, f"r{r}.json"))) for r in range(2)]

    def test_every_group_is_captured_and_replayed_on_both_ranks(self):
        r0, r1 = self.run_ranks()
        for r in (r0, r1):
            self.assertTrue(r["ok"], r)
            self.assertEqual([g["graphs"] for g in r["groups"]], [[{"m": 1}], [{"m": 4}], [{"m": 8}]])
            self.assertEqual(sorted({x["group"] for x in r["rows"]}), [0, 1, 2])
            self.assertEqual(r["recorded"], 9)                          # 3 groups x 3 steps on every rank

    def test_a_capture_failing_on_one_rank_fails_that_group_on_every_rank(self):
        r0, r1 = self.run_ranks(fail="1:1")                               # rank 1, group 1 only
        for r in (r0, r1):
            self.assertFalse(r["ok"])
            bad = [x for x in r["rows"] if not x["correct"]]
            self.assertEqual({x["group"] for x in bad}, {1}, r["rows"])
            self.assertIn("graph_capture_failed on at least one rank", bad[0]["note"])
            self.assertEqual([g["ok"] for g in r["groups"]], [True, False, True])


# =========================================================================== CPU only: fake tensors, no torch
# The classes above need torch and are skipped on a CPU-only box (CI). The classes below run the same leg_runner
# code -- correctness_suite and its helpers, _correct_mode, main -- with the REAL harness_lib (run_correctness,
# check_graph_replay, correct, noise_rel, check_correct_multi / _sequence, check_random_vs_baseline) on FT, a
# list-backed stand-in for a torch tensor, while fake_torch() is sys.modules["torch"]. Its device, FakeCuda, is the
# boundary faked: CUDA graphs that record FT's launches while captured and run them again on replay. Toy, seq,
# cases_module, TwoOut and two_out_cases above take the fake torch as they take torch.
class FT:
    """A CPU stand-in for a torch tensor: a flat row-major list of values and a shape (a one-element tensor
    broadcasts against any other). Only what harness_lib's comparisons and leg_runner touch.

    Every op computing values is a kernel launch on fake_torch()'s device. It runs at once, except while a CUDA graph
    is being captured (FT.capturing, set by FakeCuda.graph): then it is recorded into that graph, NOT run, and every
    replay of the graph runs it on what its operands hold at that moment; a tensor it creates holds garbage (NaN)
    until then. The host reading the device (item(), torch.equal, torch.cuda.synchronize) fails during a capture."""

    __hash__ = object.__hash__
    capturing = None            # the FakeGraph being captured, if any

    def __init__(self, data, shape=None, dtype="float32"):
        self.data = list(data)
        self.shape = tuple(shape) if shape is not None else (len(self.data),)
        self.dtype, self.device = dtype, "cpu"

    @staticmethod
    def _launch(n, shape, dtype, values):
        """A launch creating a tensor of `n` elements, values() computed from its operands when it runs."""
        if FT.capturing is None:
            return FT(values(), shape, dtype)
        return FT([math.nan] * n, shape, dtype)._write(values)

    def _write(self, values):
        """A launch writing values() into this tensor's storage."""
        def launch():
            self.data[:] = values()
        if FT.capturing is None:
            launch()
        else:
            FT.capturing.launches.append(launch)
        return self

    @staticmethod
    def _sync():
        """The host waits for the device: not allowed while a graph is being captured."""
        if FT.capturing is not None:
            raise RuntimeError("operation not permitted when stream is capturing")

    def _zip(self, other, f, dtype):
        a_n, b_n = len(self.data), len(other.data) if isinstance(other, FT) else 1

        def values():
            a, b = self.data, other.data if isinstance(other, FT) else [other]
            if len(a) == 1 and len(b) != 1:
                a = a * len(b)
            elif len(b) == 1:
                b = b * len(a)
            if len(a) != len(b):
                raise RuntimeError(f"shapes {self.shape} and {other.shape} do not broadcast")
            return [f(x, y) for x, y in zip(a, b)]
        shape = other.shape if a_n == 1 and b_n != 1 else self.shape
        return self._launch(b_n if a_n == 1 else a_n, shape, dtype, values)

    def _num(self, other):
        floats = self.is_floating_point() or isinstance(other, float) or (
            isinstance(other, FT) and other.is_floating_point())
        return "float32" if floats else self.dtype

    def __add__(self, o):
        return self._zip(o, lambda x, y: x + y, self._num(o))
    __radd__ = __add__

    def __sub__(self, o):
        return self._zip(o, lambda x, y: x - y, self._num(o))

    def __mul__(self, o):
        return self._zip(o, lambda x, y: x * y, self._num(o))
    __rmul__ = __mul__

    def div(self, o):
        def q(x, y):
            if y == 0:
                return math.nan if x == 0 or math.isnan(x) else math.copysign(math.inf, x)
            return x / y
        return self._zip(o, q, "float32")
    __truediv__ = div

    def __le__(self, o):
        return self._zip(o, lambda x, y: x <= y, "bool")

    def __eq__(self, o):
        return self._zip(o, lambda x, y: x == y, "bool")

    def __ne__(self, o):
        return self._zip(o, lambda x, y: x != y, "bool")
    ne = __ne__

    def __or__(self, o):
        return self._zip(o, lambda x, y: bool(x) or bool(y), "bool")

    def __and__(self, o):
        return self._zip(o, lambda x, y: bool(x) and bool(y), "bool")

    def __invert__(self):
        return self._map(lambda x: not x, "bool")

    def _map(self, f, dtype=None):
        return self._launch(len(self.data), self.shape, dtype or self.dtype, lambda: [f(x) for x in self.data])

    def abs(self):
        return self._map(abs)

    def pow(self, p):
        return self._map(lambda x: x ** p)

    def sqrt(self):
        return self._map(lambda x: math.sqrt(x) if x >= 0 else math.nan, "float32")

    def clamp_min(self, v):
        return self._map(lambda x: x if math.isnan(x) or x >= v else v)

    def mean(self):
        return self._launch(1, (), "float32", lambda: [sum(self.data) / len(self.data) if self.data else math.nan])

    def sum(self):
        return self._launch(1, (), self.dtype, lambda: [sum(self.data)])

    def max(self):
        if not self.data:
            raise RuntimeError("max(): an empty tensor")
        return self._launch(1, (), self.dtype,
                            lambda: [math.nan if any(math.isnan(x) for x in self.data) else max(self.data)])

    def all(self):
        return self._launch(1, (), "bool", lambda: [all(bool(x) for x in self.data)])

    def any(self):
        return self._launch(1, (), "bool", lambda: [any(bool(x) for x in self.data)])

    def take(self, index):
        """torch.take: the values at the flat positions `index` holds. A position past the end is an illegal memory
        access of the launch -- under a graph, of the replay running it."""
        def values():
            if not all(0 <= int(i) < len(self.data) for i in index.data):
                raise RuntimeError("an illegal memory access was encountered")
            return [self.data[int(i)] for i in index.data]
        return self._launch(len(index.data), index.shape, self.dtype, values)

    def item(self):
        self._sync()
        if len(self.data) != 1:
            raise RuntimeError("item() of a tensor with more than one element")
        return self.data[0]

    def __bool__(self):
        return bool(self.item())

    def __float__(self):
        return float(self.item())

    def float(self):
        return self._map(float, "float32")

    def numel(self):
        return len(self.data)

    def is_floating_point(self):
        return self.dtype.startswith(("float", "bfloat"))

    def data_ptr(self):
        return id(self.data)

    def detach(self):
        return self

    def clone(self):
        return self._launch(len(self.data), self.shape, self.dtype, lambda: self.data)

    def cpu(self):
        return self

    def to(self, target=None, **_):
        if isinstance(target, str) and target not in ("cpu", "cuda"):     # a dtype (no rounding: bf16 stays exact)
            return self._launch(len(self.data), self.shape, target, lambda: self.data)
        return self                                                        # a device: FT lives on the one device

    def copy_(self, src):
        return self._write(lambda: src.data if isinstance(src, FT) else [src] * len(self.data))

    def zero_(self):
        return self.copy_(0.0)

    def add_(self, o):
        return self._write(lambda: (self + o).data)

    def __repr__(self):
        return f"FT({self.data}, shape={self.shape}, dtype={self.dtype})"


class FakeGraph:
    """torch.cuda.CUDAGraph of FakeCuda: the launches recorded while it was captured; replay() runs them again, in
    order, on what the device memory holds at that moment."""

    def __init__(self, gpu):
        self.gpu, self.launches, self.captured = gpu, [], False
        gpu.graphs.append(self)

    def replay(self):
        if not self.captured:
            raise RuntimeError("Called CUDAGraph::replay without a preceding successful capture.")
        self.gpu.events.append(("replay", self.gpu.graphs.index(self)))
        for launch in self.launches:
            launch()


class FakeStream:
    def wait_stream(self, stream):
        """One in-order device: what `stream` queued has already run."""


class FakeCuda:
    """torch.cuda of fake_torch(): one available device that runs every launch at once (so a stream orders nothing
    that is not in order already), except inside torch.cuda.graph(g), which captures on a side stream: there FT
    records its launches into g. `events` logs every capture and replay, graphs numbered in creation order."""

    Stream = FakeStream

    def __init__(self):
        self.graphs, self.events = [], []
        self._default = self._current = FakeStream()

    def is_available(self):
        return True

    def current_stream(self):
        return self._current

    def default_stream(self):
        return self._default

    @contextlib.contextmanager
    def stream(self, s):
        prev, self._current = self._current, s
        try:
            yield
        finally:
            self._current = prev

    def is_current_stream_capturing(self):
        return FT.capturing is not None

    def synchronize(self):
        FT._sync()

    def CUDAGraph(self):
        return FakeGraph(self)

    @contextlib.contextmanager
    def graph(self, g):
        self.events.append(("capture", self.graphs.index(g)))
        with self.stream(FakeStream()):
            FT.capturing = g
            try:
                yield
            finally:
                FT.capturing = None
        g.captured = True


def fake_torch():
    """A `torch` module for FT: what harness_lib, leg_runner and the toy kernels above call; its cuda is a FakeCuda.
    save/load pickle to the path, as torch.save / torch.load do."""
    t = types.ModuleType("torch")
    t.bool, t.float32, t.bfloat16, t.int64 = "bool", "float32", "bfloat16", "int64"
    t.cuda = FakeCuda()

    class Generator:
        def __init__(self, device="cpu"):
            self.device, self.seed, self.rng = device, None, None

        def manual_seed(self, seed):
            self.seed = int(seed)
            self.rng = random.Random(self.seed)
            return self
    t.Generator = Generator

    def dims(shape):
        return tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)

    def full(value):
        def make(*shape, dtype="float32", device=None):
            n = math.prod(dims(shape))
            return FT._launch(n, dims(shape), dtype, lambda: [value] * n)
        return make
    t.zeros, t.ones = full(0.0), full(1.0)
    t.randn = lambda *shape, generator: FT([generator.rng.uniform(-1.0, 1.0) for _ in range(math.prod(dims(shape)))],
                                           dims(shape))
    t.sigmoid = lambda x: x._map(lambda v: 1.0 / (1.0 + math.exp(-v)))

    def equal(a, b):
        FT._sync()
        return a.shape == b.shape and a.data == b.data
    t.equal = equal
    t.isnan = lambda a: a._map(math.isnan, "bool")
    t.isfinite = lambda a: a._map(math.isfinite, "bool")

    def where(cond, a, b):
        def values():
            other = b.data * len(a.data) if len(b.data) == 1 else b.data
            return [x if k else y for k, x, y in zip(cond.data, a.data, other)]
        return FT._launch(len(a.data), a.shape, a.dtype, values)
    t.where = where

    def save(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    def load(path, map_location=None):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t.save, t.load = save, load
    return t


_ABSENT = object()


class _FakeTorchCase(unittest.TestCase):
    """fake_torch() as sys.modules["torch"] (self.gpu its device) and rank 0 of 1 as the rank context while a test
    runs; all of it restored afterwards. h.check_graph_replay is harness_lib's own: the graph replays run on self.gpu."""

    def setUp(self):
        self.torch = fake_torch()
        self.gpu = self.torch.cuda
        prev = sys.modules.get("torch", _ABSENT)
        sys.modules["torch"] = self.torch
        self.addCleanup(self._put_back, "torch", prev)
        self.addCleanup(setattr, FT, "capturing", None)
        for name in ("check_graph_replay", "_RANK_CONTEXT"):          # a test may swap either: both are put back
            self.addCleanup(setattr, h, name, getattr(h, name))
        h._RANK_CONTEXT = None
        self.assertEqual((h.check_graph_replay.__module__, h.check_graph_replay.__qualname__),
                         ("harness_lib", "check_graph_replay"))
        self.saved_checks = (h.check_correct_multi, h.check_graph_replay, h.check_correct_sequence, h.correct)

    @staticmethod
    def _put_back(name, prev):
        if prev is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev

    def suite(self, meta, cases, book=None, control=None, strict=True, blob=None, draws=1, **kw):
        """One correctness worker (correctness_suite); its stdout is kept in self.stdout."""
        book = book or lr.RefBook(h)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                res = lr.correctness_suite(h, cases, meta, blob=blob if blob is not None else {}, draws=draws,
                                           book=book, ctx=None, strict=strict, control=control, **kw)
        finally:
            self.stdout = out.getvalue()
            # whatever happened, the suite put back every harness function it swapped and left no capture open
            self.assertEqual((h.check_correct_multi, h.check_graph_replay, h.check_correct_sequence, h.correct),
                             self.saved_checks)
            self.assertIsNone(FT.capturing)
        return res, book

    def toys(self, *ms, **kw):
        return [GpuToy(self.torch, m, **kw) for m in ms]


class GpuToy(Toy):
    """Toy noting on the fake device's event log the case each fill copied in and where each run() issued its
    launches: on the 'default' or a 'side' stream, or into the graph being captured ('capture')."""

    def fill(self, case):
        super().fill(case)
        self.torch.cuda.events.append(("fill", case.get("sig")))

    def run(self):
        cuda = self.torch.cuda
        cuda.events.append(("run", "capture" if cuda.is_current_stream_capturing() else
                            "default" if cuda.current_stream() is cuda.default_stream() else "side"))
        super().run()


def check_events(gi, sigs):
    """The device events of one GpuToy group (capture case first) through harness_lib.check_graph_replay: the capture
    case filled, three warm-up runs on a side stream, one run captured into graph `gi` (recorded, not executed), then
    for every case its fill and one replay of the graph."""
    return [("fill", sigs[0]), *[("run", "side")] * 3, ("capture", gi), ("run", "capture"),
            *[ev for s in sigs for ev in (("fill", s), ("replay", gi))]]


def captured_on(events):
    """The case filled last before each capture of a device event log."""
    sigs, filled = [], None
    for kind, what in events:
        if kind == "fill":
            filled = what
        elif kind == "capture":
            sigs.append(filled)
    return sigs


def paged_group(pages, graphs):
    """A replay group gathering a static pool by the page indices fill copies in: out = pool.take(idx). A page past the
    pool is read out of bounds, which faults when a replay runs the gather (eagerly only the capture case runs)."""
    pool, idx, out = FT([10.0, 20.0, 30.0, 40.0]), FT([0, 0], dtype="int64"), FT([0.0, 0.0])
    return {"fill": lambda c: idx.copy_(c["args"]), "run": lambda: out.copy_(pool.take(idx)), "read_out": out.clone,
            "cases": [{"sig": f"p{i}", "args": FT(p, dtype="int64")} for i, p in enumerate(pages)],
            "capture_idx": 0, "graphs": graphs}


ALL_M = ({"m": 1}, {"m": 4}, {"m": 8})


def _nop(*_args):
    return None


class TestSuiteHelpersCPU(_FakeTorchCase):
    """The helpers of leg_runner's correctness suite, one by one."""

    def test_phase_plan_lists_each_phase_once_in_case_order_with_its_regime(self):
        self.assertEqual(lr._phase_plan([{"regime": "decode"}], {"regime": {"cuda_graph": True}}),
                         [("", {"cuda_graph": True})])
        self.assertEqual(lr._phase_plan([], {}), [("", {})])
        pr = {"decode": {"cuda_graph": True}, "prefill": {"cuda_graph": False}}
        rows = [{"regime": "prefill"}, {"regime": "decode"}, {"regime": "prefill"}]
        self.assertEqual(lr._phase_plan(rows, {"phase_regimes": pr}),
                         [("prefill", pr["prefill"]), ("decode", pr["decode"])])
        with self.assertRaisesRegex(ValueError, "must map phase labels to regimes"):
            lr._phase_plan(rows, {"phase_regimes": [pr]})
        with self.assertRaisesRegex(ValueError, "missing phase_regimes entry for 'mixed'"):
            lr._phase_plan([*rows, {"regime": "mixed"}], {"phase_regimes": pr})
        with self.assertRaisesRegex(ValueError, "missing phase_regimes entry for ''"):
            lr._phase_plan([{"sig": "untagged"}], {"phase_regimes": pr})

    def test_with_ref_gives_every_case_an_explicit_ref_without_changing_it(self):
        own, bare = {"sig": "a", "ref": "R"}, {"sig": "b", "args": 2}
        out = lr._with_ref([own, bare, "not a case"])
        self.assertIs(out[0], own)
        self.assertEqual(out[1], {"sig": "b", "args": 2, "ref": None})
        self.assertNotIn("ref", bare)
        self.assertEqual(out[2], "not a case")

    def test_replay_groups_take_one_group_a_list_or_nothing_and_the_phase_when_asked(self):
        cases = types.ModuleType("replay_forms")
        self.assertIsNone(lr._replay_groups(cases, h, {}, "decode"))           # no cases.replay
        g, seen = {"fill": _nop}, []
        cases.replay = lambda hh, meta: seen.append(("h, meta", hh is h)) or g
        self.assertEqual(lr._replay_groups(cases, h, {}, "decode"), [g])
        cases.replay = lambda hh, meta, phase: seen.append(("phase", phase)) or (g, g)
        self.assertEqual(lr._replay_groups(cases, h, {}, "decode"), [g, g])
        cases.replay = lambda *a: seen.append(("*args", a[2])) or [g]
        self.assertEqual(lr._replay_groups(cases, h, {}, "prefill"), [g])
        self.assertEqual(seen, [("h, meta", True), ("phase", "decode"), ("*args", "prefill")])
        cases.replay = lambda hh, meta: "fill"
        with self.assertRaisesRegex(TypeError, r"a replay group \(dict\) or a list of groups, not str"):
            lr._replay_groups(cases, h, {}, "")

    def test_group_problem_names_why_a_group_cannot_be_recorded(self):
        full = {"fill": _nop, "run": _nop, "read_out": _nop, "cases": [{}, {}]}
        self.assertIsNone(lr._group_problem(full))
        self.assertEqual(lr._group_problem(["fill"]), "is a list, not a replay group")
        self.assertEqual(lr._group_problem(dict(full, run=None, read_out="x")), "lacks the closures ['run', 'read_out']")
        self.assertEqual(lr._group_problem(dict(full, cases=[{}])), "has only 1 replay case(s); every group needs >=2")
        self.assertEqual(lr._group_problem(dict(full, cases=None)), "has only 0 replay case(s); every group needs >=2")

    def test_state_tensors_reads_a_tensor_a_list_or_a_callable_returning_either(self):
        a, b = FT([1.0]), FT([2.0])
        for state, want in ((None, []), (a, [a]), ([a, b], [a, b]), ((b,), [b]), (lambda: (a, b), [a, b]),
                            (lambda: b, [b]), (lambda: None, [])):
            self.assertEqual([id(t) for t in lr.state_tensors({"state": state})], [id(t) for t in want])
        self.assertEqual(lr.state_tensors({}), [])

    def test_same_values_reads_nan_positions_alike_and_compares_integers_exactly(self):
        nan = math.nan
        self.assertTrue(lr._same_values(FT([1.0, nan]), FT([1.0, nan])))
        self.assertFalse(lr._same_values(FT([1.0, nan]), FT([1.0, 2.0])))
        self.assertFalse(lr._same_values(FT([1.0, 2.0]), FT([1.0, 2.5])))
        self.assertTrue(lr._same_values(FT([1, 2], dtype="int64"), FT([1, 2], dtype="int64")))
        self.assertFalse(lr._same_values(FT([1, 2], dtype="int64"), FT([1, 3], dtype="int64")))

    def test_carry_checked_flags_only_a_later_step_whose_fill_rewrites_the_state(self):
        state, rewritten = FT([0.0, 0.0]), []

        def fill(case):
            if case.get("reset"):
                state.zero_()
            if case.get("clobber"):
                state.copy_(FT([9.0, 9.0]))
            return case["sig"]
        g = lr._carry_checked({"fill": fill, "state": lambda: [state], "carry_state": True, "graphs": "kept"},
                              rewritten)
        self.assertEqual(g["graphs"], "kept")
        steps = [{"sig": "capture", "clobber": True}, {"sig": "s0", "reset": True}, {"sig": "s1"},
                 {"sig": "s2", "clobber": True}, {"sig": "s3", "clobber": True}, {"sig": "s4", "reset": True}]
        self.assertEqual([g["fill"](c) for c in steps], ["capture", "s0", "s1", "s2", "s3", "s4"])
        # fill call 0 (the capture) and call 1 (the first step) may write the state; step 2 writes it, step 3 writes
        # the values it already holds, step 4 resets it
        self.assertEqual(rewritten, [2, 4])

    def test_group_summary_counts_the_steps_names_the_capture_case_and_reads_the_rows(self):
        g = {"graphs": [{"m": 4}], "carry_state": 1, "cases": [{"sig": "a"}, {"sig": "b"}], "capture_idx": 1}
        rows = [{"correct": True}, {"correct": True}]
        self.assertEqual(lr._group_summary(g, 2, rows), {"group": 2, "graphs": [{"m": 4}], "carry_state": True,
                                                         "steps": 2, "capture_sig": "b", "rows": 2, "ok": True})
        self.assertFalse(lr._group_summary(g, 2, [*rows, {"correct": None}])["ok"])     # a row that did not pass
        self.assertIs(lr._group_summary(g, 2, rows, ok=0)["ok"], False)                 # the check's own verdict
        s = lr._group_summary(dict(g, capture_idx=5), 0, None)
        self.assertEqual((s["capture_sig"], s["rows"], s["ok"]), (None, 0, True))
        self.assertEqual(lr._group_summary("junk", 3, ["x"]), {"group": 3, "graphs": None, "carry_state": False,
                                                              "steps": 0, "capture_sig": None, "rows": 1, "ok": False})

    def test_graph_coverage_names_undeclared_graphs_bad_declarations_and_capture_cases(self):
        info = h.deployment_graphs(graph_meta())[""]          # d1 -> m 1, d3 -> m 4 (pad up), d8 -> m 8

        def grp(graphs, cap="g"):
            return {"graphs": graphs, "cases": [{"sig": cap}, {"sig": "x"}], "capture_idx": 0}
        rec, probs = lr._graph_coverage(h, "decode", info, [grp([{"m": 8}, {"m": 1}]), "junk",
                                                            grp([{"m": 4}], cap="d3")])
        self.assertEqual(probs, [])
        self.assertEqual(rec, {"capture": CAPTURE, "dims": [], "deployment_graphs": list(ALL_M),
                               "declared": list(ALL_M), "missing": []})
        rec, probs = lr._graph_coverage(h, "decode", info, [grp([{"m": 1}], cap="d3"), grp([{"m": 3}]), grp(None)])
        self.assertEqual(rec["missing"], [{"m": 4}, {"m": 8}])
        self.assertEqual(rec["declared"], [{"m": 1}])
        self.assertEqual(len(probs), 4, probs)
        self.assertTrue(all(p.startswith("decode: graph coverage: ") for p in probs), probs)
        self.assertIn("replay group 0 captures on timing case d3, whose deployment graph {'m': 4} is not among its "
                      "declared graphs [{'m': 1}]", probs[0])
        self.assertIn("replay group 1: graph {'m': 3} has m=3, not a captured size [1, 2, 4, 8]", probs[1])
        self.assertIn("replay group 2: `graphs` must list the deployment graph(s)", probs[2])
        self.assertIn("deployment graphs no replay group records: [{'m': 4}, {'m': 8}]", probs[3])
        # a malformed graph_capture: its problems, and no graph is reported missing on top of them
        bad = h.deployment_graphs(graph_meta(regime={"cuda_graph": True, "graph_capture": {"sizes": []}}))[""]
        rec, probs = lr._graph_coverage(h, "default", bad, [grp([{"m": 1}])])
        self.assertEqual(rec["missing"], [])
        self.assertIn("default: graph coverage: graph_capture.sizes must be a non-empty list", probs[0])
        self.assertIn("graph {'m': 1} has m=1, not a captured size []", probs[1])

    def test_tier_coverage_names_uncovered_untagged_and_undeclared_tiers(self):
        steps = [{"sig": "a", "tier": "fresh"}, {"sig": "b", "tiers": ["carry", "wrap"]}, {"sig": "c"}, "raw",
                 {"sig": "d", "tier": "carry"}]
        rec, probs = lr._tier_coverage(h, steps, ["fresh", "carry", "evict"])
        self.assertEqual(rec, {"declared": ["fresh", "carry", "evict"], "steps": 5,
                               "covered": {"fresh": {"steps": 1, "first": ["seq[0]:a"]},
                                           "carry": {"steps": 2, "first": ["seq[1]:b", "seq[4]:d"]},
                                           "evict": {"steps": 0, "first": []}},
                               "uncovered": ["evict"], "untagged": 2, "undeclared": ["wrap"]})
        self.assertEqual(probs, ["stateful tiers: declared stateful tiers no sequence step covers: ['evict']",
                                 ("stateful tiers: 2 sequence step(s) name no tier (`tier` or `tiers`): "
                                  "['seq[2]:c', 'seq[3]']"),
                                 "stateful tiers: sequence steps name undeclared tiers: ['wrap']"])
        _, probs = lr._tier_coverage(h, [{"tier": "x"}], [])
        self.assertEqual(probs, ["stateful tiers: capabilities.stateful without capabilities.stateful_tiers",
                                 "stateful tiers: sequence steps name undeclared tiers: ['x']"])
        self.assertEqual(lr._tier_coverage(h, [{"tiers": ("fresh",)}], ["fresh"])[1], [])


class TestReplayGroupsCPU(_FakeTorchCase):
    """A18 replay groups through correctness_suite and the real harness_lib (TestReplayGroups on fake tensors)."""

    def test_every_group_is_recorded_replayed_tagged_and_summarised(self):
        a, b, c = self.toys(1, 4, 8)
        res, book = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}]), b.group([{"m": 4}]),
                                                                   c.group([{"m": 8}])]))
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["incomplete"], res["errors"]), ([], []))
        # harness_lib.check_graph_replay on the device, group by group: each group is captured once, on its capture
        # case (filled, then warmed up on a side stream), and every case is filled into the same static buffers and
        # replayed -- run() is never called again
        self.assertEqual(captured_on(self.gpu.events), ["g1_s0", "g4_s0", "g8_s0"])
        self.assertEqual(self.gpu.events, [ev for gi, t in enumerate((a, b, c))
                                           for ev in check_events(gi, [x["sig"] for x in t.cases])])
        rep = res["report"]["default"]
        self.assertEqual([r["group"] for r in rep["graph_replay"]], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(rep["replay_groups"], [{"group": i, "graphs": [{"m": m}], "carry_state": False, "steps": 3,
                                                 "capture_sig": f"g{m}_s0", "rows": 3, "ok": True}
                                                for i, m in enumerate((1, 4, 8))])
        self.assertEqual((rep["graph_coverage"]["declared"], rep["graph_coverage"]["missing"]), (list(ALL_M), []))
        self.assertEqual((rep["eager"], rep["random"]), ([], []))
        self.assertEqual({k: len(v) for k, v in book.dump().items()}, {"eager": 0, "graph_replay": 9, "sequence": 0})
        # what was recorded is each case's own output (out = x * 1.0): every replay ran the captured launches on the
        # inputs its fill had just copied in
        self.assertEqual([t.data for t in book.dump()["graph_replay"]], [x["args"].data for t in (a, b, c)
                                                                          for x in t.cases])
        # the original single-group form: one dict
        t = Toy(self.torch, 8)
        res, _ = self.suite(graph_meta(), cases_module(lambda: t.group(list(ALL_M))))
        self.assertTrue(res["ok"], res)
        self.assertEqual(len(res["report"]["default"]["replay_groups"]), 1)
        self.assertEqual({r["group"] for r in res["report"]["default"]["graph_replay"]}, {0})

    def test_unrecordable_groups_are_announced_and_make_the_ut_incomplete(self):
        a, b, c = self.toys(1, 4, 8)
        one = b.group([{"m": 4}], cases=b.cases[:1])
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}]), one, "junk", c.group([{"m": 8}])]))
        self.assertFalse(res["ok"])
        self.assertEqual(res["incomplete"], ["default: replay group 1 has only 1 replay case(s); every group needs >=2",
                                             "default: replay group 2 is a str, not a replay group"])
        self.assertIn("UT_HARNESS_INCOMPLETE: default: replay group 1 has only 1 replay case(s)", self.stdout)
        self.assertIn("UT_HARNESS_INCOMPLETE: default: replay group 2 is a str", self.stdout)
        groups = res["report"]["default"]["replay_groups"]
        self.assertEqual(groups[1:3], [{"group": 1, "ok": False, "problem": "has only 1 replay case(s); every group "
                                                                             "needs >=2"},
                                       {"group": 2, "ok": False, "problem": "is a str, not a replay group"}])
        self.assertEqual([(g["group"], g["ok"]) for g in groups], [(0, True), (1, False), (2, False), (3, True)])
        self.assertEqual(sorted({r["group"] for r in res["report"]["default"]["graph_replay"]}), [0, 3])

    def test_a_deployment_graph_no_group_records_and_an_undeclared_capture_graph_are_incomplete(self):
        a, c = self.toys(1, 8)
        a.cases[0] = dict(a.cases[0], sig="d3")             # captures on timing case d3 -> deployment graph m=4
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}]), c.group([{"m": 8}])]))
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["incomplete"]), 2, res["incomplete"])
        self.assertIn("default: graph coverage: replay group 0 captures on timing case d3, whose deployment graph "
                      "{'m': 4} is not among its declared graphs [{'m': 1}]", res["incomplete"][0])
        self.assertIn("default: graph coverage: deployment graphs no replay group records: [{'m': 4}]",
                      res["incomplete"][1])
        rep = res["report"]["default"]
        self.assertEqual(rep["graph_coverage"]["missing"], [{"m": 4}])
        self.assertTrue(all(r["correct"] for r in rep["graph_replay"]))      # a gap of the UT, not of the kernel
        self.assertEqual(res["errors"], [])
        # not a generated UT: no coverage rule
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}]), c.group([{"m": 8}])]),
                            strict=False)
        self.assertTrue(res["ok"], res)
        self.assertNotIn("graph_coverage", res["report"]["default"])

    def test_the_candidate_is_judged_group_by_group_against_the_native_record(self):
        a, b = self.toys(1, 4)
        res, book = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}, {"m": 8}]), b.group([{"m": 4}])]))
        self.assertTrue(res["ok"], res)
        recorded = book.dump()
        self.assertEqual(len(recorded["graph_replay"]), 6)
        a2, b2 = Toy(self.torch, 1), Toy(self.torch, 4, scale=1.5)     # the candidate is wrong in group 1 only
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a2.group([{"m": 1}, {"m": 8}]), b2.group([{"m": 4}])]),
                            book=lr.RefBook(h, recorded))
        self.assertFalse(res["ok"])
        self.assertEqual((res["incomplete"], res["errors"]), ([], []))
        rep = res["report"]["default"]
        self.assertEqual([(r["group"], r["correct"]) for r in rep["graph_replay"]],
                         [(0, True)] * 3 + [(1, False)] * 3)
        self.assertTrue(all(r["max_rel_err"] > 0.3 for r in rep["graph_replay"][3:]))
        self.assertEqual([s["ok"] for s in rep["replay_groups"]], [True, False])
        self.assertNotIn("native_refs", res["report"])
        # a candidate making fewer graph comparisons than the native leg recorded fails, naming the counts
        res, _ = self.suite(graph_meta(), cases_module(lambda: a2.group(list(ALL_M))), book=lr.RefBook(h, recorded))
        self.assertFalse(res["ok"])
        self.assertTrue(all(r["correct"] for r in res["report"]["default"]["graph_replay"]))
        self.assertEqual(res["report"]["native_refs"],
                         ["graph_replay: candidate made 3 comparisons, the native leg recorded 6"])

    def test_negative_controls_act_on_every_group_and_on_the_call(self):
        a, b = self.toys(1, 4)

        def groups():
            return [a.group([{"m": 1}, {"m": 8}]), b.group([{"m": 4}])]
        _, book = self.suite(graph_meta(), cases_module(groups, seq(self.torch)))
        for name, failed in (("noop", set()), ("within_tol", set()), ("scale_out", {0, 1}),
                             ("stale_output", {0, 1})):
            res, _ = self.suite(graph_meta(), cases_module(groups, seq(self.torch)), book=lr.RefBook(h, book.dump()),
                                control=lr.Control(name, TOL))
            rep = res["report"]
            self.assertEqual({r["group"] for r in rep["default"]["graph_replay"] if r["correct"] is False}, failed,
                             name)
            if name == "stale_output":      # group 1 reads group 0's first (1, 4) output: no finite error, None
                self.assertEqual([r["max_rel_err"] for r in rep["default"]["graph_replay"] if r["group"] == 1],
                                 [None] * 3)
            self.assertEqual(res["ok"], not failed, (name, res))
            # the sequence goes through the wrapped call: scale_out moves every step but the all-zero first one,
            # stale_output hands the later steps (the same input shape) the first step's output
            want = [True, False, False] if failed else [True, True, True]
            self.assertEqual([r["correct"] for r in rep["sequence"]], want, name)


class TestStateCarryAndTiersCPU(_FakeTorchCase):
    """A18.3 / A18.4: a stateful graph task's carry group and its sequence tiers (TestStateCarryAndTiers on fake
    tensors)."""

    def carry(self, **kw):
        return Toy(self.torch, 8, stateful=True, **kw)

    def test_a_stateful_graph_task_needs_a_group_carrying_state(self):
        t = Toy(self.torch, 8)
        res, _ = self.suite(stateful_meta(), cases_module(lambda: t.group(list(ALL_M)), seq(self.torch)))
        self.assertFalse(res["ok"])
        self.assertEqual(res["incomplete"], [("capabilities.stateful with a graph-deployed phase: no replay group "
                                              "carries state between consecutive replays (carry_state: true, a "
                                              "declared `state`, >= 2 cases)")])
        t, found = self.carry(), []

        def carry_group():
            g = t.group(list(ALL_M))
            fill = g["fill"]

            def noting_fill(case):                    # the state each fill finds
                found.append(list(t.acc.data))
                fill(case)
            return dict(g, fill=noting_fill)
        res, _ = self.suite(stateful_meta(), cases_module(carry_group, seq(self.torch)))
        self.assertTrue(res["ok"], res)
        self.assertIs(res["report"]["default"]["replay_groups"][0]["carry_state"], True)
        self.assertEqual(res["report"]["sequence_coverage"]["uncovered"], [])
        # the first step's fill finds what the capture case's three warm-up launches (acc += x) left; capturing the
        # graph ran none of its launches
        self.assertEqual(found[1], [((0.0 + v) + v) + v for v in t.cases[0]["args"].data])

    def test_a_carry_group_without_readable_state_or_whose_fill_rewrites_it_is_incomplete(self):
        t = self.carry()
        res, _ = self.suite(stateful_meta(), cases_module(lambda: t.group(list(ALL_M), state=None), seq(self.torch)))
        self.assertEqual(res["incomplete"], [("default: replay group 0 declares carry_state but no `state` (the "
                                              "tensors consecutive replays carry)")])

        def unreadable():
            raise RuntimeError("the KV cache is not allocated")
        res, _ = self.suite(stateful_meta(), cases_module(lambda: t.group(list(ALL_M), state=unreadable),
                                                          seq(self.torch)))
        self.assertEqual(res["incomplete"], [
            "default: replay group 0: its `state` could not be read: RuntimeError('the KV cache is not allocated')",
            "default: replay group 0 declares carry_state but no `state` (the tensors consecutive replays carry)"])
        self.assertFalse(res["ok"])
        t = self.carry(reset_every_step=True)
        res, _ = self.suite(stateful_meta(), cases_module(lambda: t.group(list(ALL_M)), seq(self.torch)))
        self.assertEqual(res["incomplete"], [("default: replay group 0 (carry_state): fill rewrote the declared "
                                              "state at step(s) [1, 2]; a carry_state group's fill refreshes only "
                                              "the per-step inputs")])
        self.assertTrue(all(r["correct"] for r in res["report"]["default"]["graph_replay"]))

    def test_skip_state_write_is_caught_in_the_carry_group(self):
        t = self.carry()

        def mk():
            return cases_module(lambda: t.group(list(ALL_M)), seq(self.torch))
        res, book = self.suite(stateful_meta(), mk())
        self.assertTrue(res["ok"], res)
        acc, carried = [0.0] * 32, []
        for case in t.cases:        # the graph's acc += x carried from replay to replay: out = 2 * (x_0 + ... + x_k)
            acc = [p + q for p, q in zip(acc, case["args"].data)]
            carried.append([v * 2 for v in acc])
        self.assertEqual([r.data for r in book.dump()["graph_replay"]], carried)
        res, _ = self.suite(stateful_meta(), mk(), book=lr.RefBook(h, book.dump()),
                            control=lr.Control("skip_state_write", TOL))
        self.assertFalse(res["ok"])
        # the first step reads the reset state; the later steps miss the writes the control dropped
        self.assertEqual([r["correct"] for r in res["report"]["default"]["graph_replay"]], [True, False, False])
        self.assertTrue(all(r["correct"] for r in res["report"]["sequence"]))   # the identity call writes nothing

    def test_sequence_steps_must_cover_the_declared_tiers(self):
        t = self.carry()

        def groups():
            return t.group(list(ALL_M))
        res, _ = self.suite(stateful_meta(), cases_module(groups, seq(self.torch)))
        cov = res["report"]["sequence_coverage"]
        self.assertEqual((cov["uncovered"], cov["covered"]["carry"]["steps"], cov["steps"]), ([], 2, 3))
        self.assertEqual([r["case"] for r in res["report"]["sequence"]], ["seq[0]:s0", "seq[1]:s1", "seq[2]:s2"])
        res, _ = self.suite(stateful_meta(), cases_module(groups, seq(self.torch, ("fresh", "fresh"))))
        self.assertEqual(res["incomplete"], [("stateful tiers: declared stateful tiers no sequence step covers: "
                                              "['carry']")])
        res, _ = self.suite(stateful_meta(), cases_module(groups))          # no sequence_cases at all
        self.assertFalse(res["ok"])
        self.assertEqual(res["incomplete"], [("capabilities.stateful is declared but cases.py has no sequence_cases: "
                                              "a stateful kernel needs an ordered multi-step check (UT incomplete)")])
        self.assertNotIn("sequence", res["report"])
        # a hand-written task (not strict): none of the generated-UT rules
        res, _ = self.suite(stateful_meta(), cases_module(groups), strict=False)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["incomplete"], [])


class TestSuiteErrorsAndPhasesCPU(_FakeTorchCase):
    """correctness_suite's error paths (recorded, or raised with fail_fast), phases, and what a strict run passes."""

    def test_a_replay_that_cannot_be_built_is_an_error_or_with_fail_fast_raised(self):
        cases = cases_module(_nop)

        def broken(hh, meta):
            raise RuntimeError("replay cannot be built")
        cases.replay = broken
        res, _ = self.suite(graph_meta(), cases)
        self.assertFalse(res["ok"])
        self.assertEqual((len(res["errors"]), res["incomplete"]), (1, []))
        self.assertTrue(res["errors"][0].startswith("default: replay: Traceback"), res["errors"])
        self.assertIn("RuntimeError: replay cannot be built", res["errors"][0])
        self.assertIn("RuntimeError: replay cannot be built", res["report"]["default"]["error"])
        with self.assertRaisesRegex(RuntimeError, "replay cannot be built"):
            self.suite(graph_meta(), cases, fail_fast=True)

    def test_a_graph_phase_without_a_replay_bundle_is_incomplete(self):
        cases = cases_module(_nop)
        del cases.replay
        res, _ = self.suite(graph_meta(), cases)
        self.assertFalse(res["ok"])
        self.assertEqual(res["errors"], [])
        self.assertEqual(len(res["incomplete"]), 1)
        self.assertTrue(res["incomplete"][0].startswith("default: deploys under CUDA graph (deployment_graph_mode=True) "
                                                        "but no replay bundle"), res["incomplete"])
        self.assertEqual(res["report"]["default"], {"incomplete": res["incomplete"][0][len("default: "):]})
        self.assertIn("UT_HARNESS_INCOMPLETE: deploys under CUDA graph", self.stdout)

    def test_a_check_that_raises_is_an_error_or_with_fail_fast_raised(self):
        t = Toy(self.torch, 8)
        cases = cases_module(lambda: t.group(list(ALL_M)))
        cases.eager_cases = lambda hh, meta: [{"sig": "e0", "args": FT([1.0])}]

        def launch_fails(args):
            raise ValueError("kernel launch failed")
        cases.call = launch_fails
        res, _ = self.suite(graph_meta(), cases)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["errors"]), 1)
        self.assertTrue(res["errors"][0].startswith("default: Traceback"), res["errors"])
        self.assertIn("ValueError: kernel launch failed", res["errors"][0])
        self.assertIn("ValueError: kernel launch failed", res["report"]["default"]["error"])
        with self.assertRaisesRegex(ValueError, "kernel launch failed"):
            self.suite(graph_meta(), cases, fail_fast=True)

    def test_a_group_whose_check_raises_is_an_error_or_with_fail_fast_raised(self):
        """harness_lib.check_graph_replay turns what fails in a capture or a replay into rows (the next two tests);
        it raises for a group it cannot run at all -- here one listing a case by its sig instead of a case dict,
        whose failed fill it cannot even label. The worker records the error for that group, or with fail_fast
        raises it."""
        a, b = self.toys(1, 4)

        def groups():
            return [a.group([{"m": 1}, {"m": 8}]), b.group([{"m": 4}], cases=[*b.cases, "g4_s3"])]
        res, _ = self.suite(graph_meta(), cases_module(groups))
        self.assertFalse(res["ok"])
        self.assertEqual((len(res["errors"]), res["incomplete"]), (1, []))
        self.assertTrue(res["errors"][0].startswith("default: replay group 1: Traceback"), res["errors"])
        self.assertIn("TypeError: string indices must be integers", res["errors"][0])          # fill("g4_s3")
        self.assertIn("AttributeError: 'str' object has no attribute 'get'", res["errors"][0])  # then its row
        rep = res["report"]["default"]
        self.assertEqual(rep["replay_groups"][1], {"group": 1, "graphs": [{"m": 4}], "carry_state": False,
                                                   "steps": 4, "capture_sig": "g4_s0", "rows": 0, "ok": False})
        self.assertEqual({r["group"] for r in rep["graph_replay"]}, {0})
        self.assertTrue(all(r["correct"] for r in rep["graph_replay"]))
        with self.assertRaisesRegex(AttributeError, "'str' object has no attribute 'get'"):
            self.suite(graph_meta(), cases_module(groups), fail_fast=True)

    def test_a_capture_failing_in_one_group_fails_that_group_when_strict_and_is_skipped_otherwise(self):
        """The real check returns a failed capture as ONE row: graph_capture_failed when strict (a generated UT: the
        op deploys under the graph) -- nothing is raised, also with fail_fast -- and a skipped PASS otherwise (a
        hand-written UT on an image without graphs). The other groups are captured and replayed as usual. The causes:
        run() reads a value on the host (fine in the warm-up, refused while capturing), in a later group or in the
        first one (whose check runs inside run_correctness); fill raises on the capture case."""
        a, b, c = self.toys(1, 4, 8)

        def host_guarded():
            if b.x.abs().max().item() > 1e6:          # the host reads the input before the launch: not graph-safe
                raise ValueError("input out of range")
            b.run()

        def broken_fill(case):
            raise RuntimeError("fill wrote past the static buffer")

        def sigs(t):
            return [x["sig"] for x in t.cases]
        # a later group's capture refused: it was warmed up, and is never replayed; the other groups are untouched
        res, _ = self.suite(graph_meta(), cases_module(lambda: [a.group([{"m": 1}]), b.group([{"m": 4}], run=host_guarded),
                                                                c.group([{"m": 8}])]))
        self.assertEqual(self.gpu.events, [*check_events(0, sigs(a)), ("fill", "g4_s0"), *[("run", "side")] * 3,
                                           ("capture", 1), *check_events(2, sigs(c))])
        unsafe = "RuntimeError('operation not permitted when stream is capturing')"
        for order, over, err in (((a, b, c), {"run": host_guarded}, unsafe), ((b, a, c), {"run": host_guarded}, unsafe),
                                 ((a, b, c), {"fill": broken_fill}, "RuntimeError('fill wrote past the static buffer')")):
            bad = order.index(b)

            def groups(order=order, over=over):
                return [t.group([{"m": t.x.shape[0]}], **(over if t is b else {})) for t in order]
            for strict, fail_fast in ((True, False), (True, True), (False, False)):
                with self.subTest(bad=bad, cause=err, strict=strict, fail_fast=fail_fast):
                    res, _ = self.suite(graph_meta(), cases_module(groups), strict=strict, fail_fast=fail_fast)
                    self.assertEqual((res["ok"], res["errors"], res["incomplete"]), (not strict, [], []))
                    rep = res["report"]["default"]
                    note = (f"graph_capture_failed (graph deployment, strict): {err}" if strict
                            else f"skipped: capture unavailable ({err})")
                    self.assertEqual([r for r in rep["graph_replay"] if r["group"] == bad],
                                     [{"case": "graph_replay", "correct": not strict, "max_rel_err": None,
                                       "note": note, "group": bad}])
                    self.assertEqual([(r["note"], r["correct"]) for r in rep["graph_replay"] if r["group"] != bad],
                                     [("graph_replay", True)] * 6)
                    self.assertEqual([(s["group"], s["rows"], s["ok"]) for s in rep["replay_groups"]],
                                     [(i, 1 if i == bad else 3, not strict or i != bad) for i in range(3)])

    def test_a_group_whose_replay_faults_fails_only_that_group_with_a_raised_row(self):
        """The real check catches a replay that faults -- here a gather reading past its pool, which only a replay
        runs (the capture case's pages are in range): that case's row is graph_replay_raised, the group's later cases
        are still replayed and compared, the other groups are untouched, and nothing is raised, also with fail_fast."""
        a, c = self.toys(1, 8)

        def groups():
            return [a.group([{"m": 1}]), paged_group([[0, 1], [2, 9], [3, 2]], [{"m": 4}]), c.group([{"m": 8}])]
        fault = "RuntimeError('an illegal memory access was encountered')"
        for fail_fast in (False, True):
            del self.gpu.events[:], self.gpu.graphs[:]
            res, book = self.suite(graph_meta(), cases_module(groups), fail_fast=fail_fast)
            self.assertEqual((res["ok"], res["errors"], res["incomplete"]), (False, [], []))
            rep = res["report"]["default"]
            self.assertEqual([r for r in rep["graph_replay"] if r["group"] == 1], [
                {"case": "p0", "correct": True, "max_rel_err": 0.0, "note": "graph_replay", "group": 1},
                {"case": "p1", "correct": False, "max_rel_err": None, "group": 1,
                 "note": f"graph_replay_raised (OOB/stale under replay): {fault}"},
                {"case": "p2", "correct": True, "max_rel_err": 0.0, "note": "graph_replay", "group": 1}])
            self.assertEqual([(r["note"], r["correct"]) for r in rep["graph_replay"] if r["group"] != 1],
                             [("graph_replay", True)] * 6)
            self.assertEqual([(s["group"], s["rows"], s["ok"]) for s in rep["replay_groups"]],
                             [(0, 3, True), (1, 3, False), (2, 3, True)])
            self.assertEqual([ev for ev in self.gpu.events if ev in (("capture", 1), ("replay", 1))],
                             [("capture", 1)] + [("replay", 1)] * 3)
            # recorded: the pages each in-range case gathered under replay; the faulted case compared nothing
            recorded = book.dump()["graph_replay"]
            self.assertEqual((len(recorded), [t.data for t in recorded[3:5]]), (8, [[10.0, 20.0], [40.0, 30.0]]))

    def test_a_sequence_that_raises_is_an_error_or_with_fail_fast_raised(self):
        t = Toy(self.torch, 8)

        def no_steps():
            raise KeyError("steps")
        res, _ = self.suite(graph_meta(), cases_module(lambda: t.group(list(ALL_M)), no_steps))
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["errors"]), 1)
        self.assertTrue(res["errors"][0].startswith("sequence: Traceback"), res["errors"])
        self.assertIn("KeyError: 'steps'", res["report"]["sequence"]["error"])
        self.assertTrue(all(r["correct"] for r in res["report"]["default"]["graph_replay"]))
        with self.assertRaisesRegex(KeyError, "steps"):
            self.suite(graph_meta(), cases_module(lambda: t.group(list(ALL_M)), no_steps), fail_fast=True)

    def test_each_phase_runs_its_own_cases_and_only_a_graph_phase_replays(self):
        meta = {"target_callable": "toy:op", "tolerance": {"tol": TOL}, "capabilities": {"stateful": False},
                "phase_regimes": {"prefill": {"cuda_graph": False},
                                  "decode": {"cuda_graph": True, "graph_capture": dict(CAPTURE)}},
                "case_manifest": {"dims": ["regime", "m"], "cases": [
                    {"sig": "d1", "regime": "decode", "m": 1, "roles": ["timing"]},
                    {"sig": "d2", "regime": "decode", "m": 2, "roles": ["timing"]}]}}
        toy, asked = Toy(self.torch, 2), []
        cases = types.ModuleType("phase_cases")
        cases.call = lambda args: args * 2.0
        cases.eager_cases = lambda hh, m: [{"sig": "p0", "regime": "prefill", "args": FT([1.0, 2.0])},
                                           {"sig": "p1", "regime": "prefill", "args": FT([3.0, 4.0]),
                                            "ref": FT([6.0, 8.0])},
                                           {"sig": "d0", "regime": "decode", "args": FT([5.0])}]
        cases.random_shapes = lambda hh, m: [
            {"sig": "rp", "regime": "prefill", "make_inputs": lambda rng: FT([rng.seed + 1.0, 2.0])},
            {"sig": "rd", "regime": "decode", "make_inputs": lambda rng: FT([rng.seed + 3.0])}]
        cases.replay = lambda hh, m, phase: asked.append(phase) or toy.group([{"m": 1}, {"m": 2}])
        blob = {f"rp|{i}": FT([2.0 * (i + 1.0), 4.0]) for i in range(2)}
        blob.update({f"rd|{i}": FT([2.0 * (i + 3.0)]) for i in range(2)})
        res, book = self.suite(meta, cases, blob=blob, draws=2)
        self.assertTrue(res["ok"], res)
        self.assertEqual(asked, ["decode"])                         # an eager phase has no replay
        rep = res["report"]
        self.assertEqual(list(rep), ["prefill", "decode"])
        self.assertEqual([r["case"] for r in rep["prefill"]["eager"]], ["p0", "p1", "output_independence"])
        self.assertEqual([r["case"] for r in rep["decode"]["eager"]], ["d0"])
        self.assertEqual([r["case"] for r in rep["prefill"]["random"]], ["random[0]:rp", "random[1]:rp"])
        self.assertEqual([r["case"] for r in rep["decode"]["random"]], ["random[0]:rd", "random[1]:rd"])
        self.assertTrue(all(r["correct"] for p in rep.values() for k in ("eager", "random") for r in p[k]))
        self.assertNotIn("graph_replay", rep["prefill"])
        self.assertEqual([r["case"] for r in rep["decode"]["graph_replay"]], ["g2_s0", "g2_s1", "g2_s2"])
        self.assertEqual(rep["decode"]["graph_coverage"]["missing"], [])
        self.assertEqual(len(book.dump()["eager"]), 3)
        # the native leg checks a case's own ref, and the oracle's draws: a wrong one fails there
        cases.eager_cases = lambda hh, m: [{"sig": "p1", "regime": "prefill", "args": FT([3.0]), "ref": FT([7.0])}]
        blob["rd|1"] = FT([0.0])
        res, _ = self.suite(meta, cases, blob=blob, draws=2)
        self.assertFalse(res["ok"])
        self.assertEqual([r["correct"] for r in res["report"]["prefill"]["eager"]], [False])
        self.assertEqual([r["correct"] for r in res["report"]["decode"]["random"]], [True, False])

    def test_a_strict_run_hands_the_capture_context_to_every_group(self):
        seen, replay = [], h.check_graph_replay                  # harness_lib's own (see _FakeTorchCase.setUp)

        def spy(fill, run, read_out, cases, tol, capture_idx=0, **kw):
            seen.append(kw)
            return replay(fill, run, read_out, cases, tol, capture_idx=capture_idx, **kw)
        h.check_graph_replay = spy                                # put back by _FakeTorchCase's cleanup
        self.saved_checks = (h.check_correct_multi, spy, h.check_correct_sequence, h.correct)
        a, b = self.toys(1, 4)
        cases = cases_module(lambda: [a.group([{"m": 1}, {"m": 8}]), b.group([{"m": 4}])])

        @contextlib.contextmanager
        def capture_context():
            self.gpu.events.append(("enter", "capture_context"))
            yield
            self.gpu.events.append(("exit", "capture_context"))
        cases.capture_context = capture_context
        res, _ = self.suite(graph_meta(), cases)
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen, [{"strict": True, "capture_context": capture_context}] * 2)
        # the real check entered it around each group's capture, and nowhere else (not the warm-up, not a replay)
        around = [ev for ev in self.gpu.events if ev[0] in ("enter", "exit", "capture") or ev == ("run", "capture")]
        self.assertEqual(around, [ev for gi in (0, 1) for ev in (("enter", "capture_context"), ("capture", gi),
                                                                 ("run", "capture"), ("exit", "capture_context"))])
        seen.clear()
        del self.gpu.events[:]
        res, _ = self.suite(graph_meta(), cases, strict=False)
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen, [{}, {}])
        self.assertNotIn(("enter", "capture_context"), self.gpu.events)


class TestPerOutputTolerancesCPU(_FakeTorchCase):
    """A42 through correctness_suite (TestPerOutputTolerances on fake tensors)."""

    def setUp(self):
        super().setUp()
        torch = self.torch
        native = TwoOut(torch)                      # the native oracle of the random draws, as the oracle leg makes it
        self.blob = {f"r|{i}": native.call(torch.randn(4, 8, generator=torch.Generator().manual_seed(i)))
                     for i in range(2)}

    def leg(self, meta, kernel, book, native):
        tol = lr.task_tolerance(meta, h, native=native)
        res, _ = self.suite(meta, two_out_cases(self.torch, kernel), book=book, strict=False, blob=self.blob, draws=2,
                            tol=tol)
        return res, tol

    def legs(self, meta, drift):
        book = lr.RefBook(h)
        native, tol = self.leg(meta, TwoOut(self.torch), book, True)
        self.assertTrue(native["ok"], native)
        judged, _ = self.leg(meta, TwoOut(self.torch, drift=drift), lr.RefBook(h, book.dump()), False)
        return judged, tol

    @staticmethod
    def rows(res):
        rep = res["report"]
        out = {s: [r for r in rep["default"][s] if r["case"] != "output_independence"]
               for s in ("eager", "random", "graph_replay")}
        out["sequence"] = rep["sequence"]
        return out

    def test_every_comparison_is_made_output_by_output_at_its_own_tolerance(self):
        judged, tol = self.legs(two_out_meta(per_output={"1": 0.001}), drift=1.005)
        self.assertIsInstance(tol, lr.TaskTol)
        self.assertFalse(judged["ok"])
        self.assertEqual((judged["incomplete"], judged["errors"]), ([], []))
        for section, rows in self.rows(judged).items():
            self.assertTrue(rows, section)
            for r in rows:
                self.assertEqual((r["correct"], r["output"], r["tol"]), (False, "1", 0.001), (section, r))
                self.assertGreater(r["max_rel_err"], 0.001)
        self.assertEqual(tol.record()["outputs"], [{"output": "0", "tol": 0.02}, {"output": "1", "tol": 0.001}])
        # one tolerance for every output: the same candidate passes, and the rows name no output
        judged, tol = self.legs(two_out_meta(), drift=1.005)
        self.assertNotIsInstance(tol, lr.TaskTol)
        self.assertTrue(judged["ok"], judged)
        for section, rows in self.rows(judged).items():
            self.assertTrue(rows and all(r["correct"] and "output" not in r for r in rows), section)

    def test_a_layout_per_output_does_not_fit_or_a_malformed_one_is_incomplete(self):
        res, _ = self.leg(two_out_meta(per_output={"2": 0.001}), TwoOut(self.torch), lr.RefBook(h), True)
        self.assertFalse(res["ok"])
        self.assertEqual(res["incomplete"], [("meta.tolerance.per_output: per_output names the positions ['2'], but "
                                              "the compared output has 2 tensor(s)")])
        res, _ = self.leg(two_out_meta(per_output={"1": -1}), TwoOut(self.torch), lr.RefBook(h), False)
        self.assertFalse(res["ok"])
        self.assertEqual(res["incomplete"], [("meta.tolerance.per_output['1'] must be a finite non-negative number, "
                                              "not -1")])

    def test_a_declared_comparison_plugin_makes_the_generated_ut_incomplete(self):
        for extra in ({"compare": {"file": "cmp.py", "attr": "compare"}},
                      {"capabilities": {"stateful": False, "per_output_compare": True}}):
            meta = dict(two_out_meta(), regime={"cuda_graph": False}, **extra)
            cases = two_out_cases(self.torch, TwoOut(self.torch))
            del cases.replay
            res, _ = self.suite(meta, cases, blob=self.blob, draws=2)
            self.assertFalse(res["ok"])
            self.assertEqual(len(res["incomplete"]), 1, res["incomplete"])
            self.assertIn("a comparison plugin is not supported", res["incomplete"][0])
            self.assertTrue(all(r["correct"] for r in res["report"]["default"]["random"]))


class TestCorrectModeCPU(_FakeTorchCase):
    """leg_runner._correct_mode: the native `refs` worker records, the candidate `correct` worker is judged against
    those records; files per rank; controls and per-output records."""

    LIVE = "geak_cpu_live_stack"

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="geak_correct_mode_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.task = os.path.join(self.tmp, "task")
        os.makedirs(self.task)
        self.prefix = os.path.join(self.tmp, "native")
        live = types.ModuleType(self.LIVE)
        live.__file__ = os.path.join(self.tmp, "site", f"{self.LIVE}.py")

        def op(x):
            return x
        op.__module__, op.__qualname__ = self.LIVE, "op"
        live.op = op
        self.assertNotIn(self.LIVE, sys.modules)
        sys.modules[self.LIVE] = live
        self.addCleanup(sys.modules.pop, self.LIVE, None)
        native, torch = TwoOut(self.torch), self.torch
        self.blob = {f"r|{i}": native.call(torch.randn(4, 8, generator=torch.Generator().manual_seed(i)))
                     for i in range(3)}

    def meta(self, **tol):
        m = two_out_meta(**tol)
        m.update(target_callable=f"{self.LIVE}:op",
                 phase_target_callables={"decode": f"{self.LIVE}:op", "prefill": "geak_cpu_no_such_module:op"})
        return m

    def run_mode(self, mode, meta, kernel, **args):
        a = types.SimpleNamespace(blob=self.prefix, mode=mode, control="", control_spec="", draws=0)
        vars(a).update(args)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return lr._correct_mode(a, self.task, meta, h, two_out_cases(self.torch, kernel))

    def test_the_native_leg_records_and_the_candidate_leg_is_judged_against_the_records(self):
        self.torch.save(self.blob, f"{self.prefix}.pt")
        meta = self.meta()
        res = self.run_mode("refs", meta, TwoOut(self.torch))
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["rank"], res["mode"], res["incomplete"], res["errors"]), (0, "refs", [], []))
        self.assertEqual(res["identity"], {
            f"{self.LIVE}:op": {"module": self.LIVE, "qualname": "op",
                                "file": os.path.realpath(os.path.join(self.tmp, "site", f"{self.LIVE}.py"))},
            "geak_cpu_no_such_module:op": {"error": "ModuleNotFoundError: No module named 'geak_cpu_no_such_module'"}})
        self.assertEqual(len(res["report"]["default"]["random"]), 3)          # meta's default: 3 draws
        self.assertEqual(res["native_noise"], {"max_rel_err": 0.0, "cases": 3, "noise_rel": 0.0})
        self.assertNotIn("tolerance", res)
        refs = self.torch.load(f"{self.prefix}.refs.pt")
        self.assertEqual({k: len(v) for k, v in refs.items()}, {"eager": 2, "graph_replay": 2, "sequence": 2})
        # the candidate leg, judged against those records; --draws wins over the meta's default
        res = self.run_mode("correct", meta, TwoOut(self.torch, drift=1.005), draws=2)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["mode"], "correct")
        self.assertEqual(len(res["report"]["default"]["random"]), 2)
        self.assertNotIn("native_noise", res)
        res = self.run_mode("correct", meta, TwoOut(self.torch, drift=1.05))
        self.assertFalse(res["ok"])
        self.assertFalse(any(r["correct"] for r in res["report"]["default"]["graph_replay"]))
        self.assertFalse(any(r["correct"] for r in res["report"]["sequence"]))
        # verify_ut's controls come from the arguments
        res = self.run_mode("correct", meta, TwoOut(self.torch), control="scale_out")
        self.assertFalse(res["ok"])
        res = self.run_mode("correct", meta, TwoOut(self.torch), control="within_tol", control_spec="{}")
        self.assertTrue(res["ok"], res)

    def test_a_rank_of_a_pool_uses_its_own_files_and_needs_the_native_records(self):
        h._RANK_CONTEXT = h.RankContext(1, 2)                     # put back by _FakeTorchCase's cleanup
        self.torch.save(self.blob, f"{self.prefix}.rank1.pt")
        meta, refs = self.meta(), f"{self.prefix}.refs.rank1.pt"
        res = self.run_mode("correct", meta, TwoOut(self.torch))
        self.assertEqual({k: v for k, v in res.items() if k != "identity"},
                         {"rank": 1, "mode": "correct", "ok": False, "incomplete": [f"native refs missing: {refs}"],
                          "errors": [], "report": {}})
        res = self.run_mode("refs", meta, TwoOut(self.torch))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["rank"], 1)
        self.assertTrue(os.path.isfile(refs))
        self.assertFalse(os.path.exists(f"{self.prefix}.refs.pt"))
        res = self.run_mode("correct", meta, TwoOut(self.torch))
        self.assertTrue(res["ok"], res)

    def test_with_per_output_tolerances_the_native_leg_records_noise_per_output(self):
        self.torch.save(self.blob, f"{self.prefix}.pt")
        meta = self.meta(per_output={"1": 0.001})
        res = self.run_mode("refs", meta, TwoOut(self.torch, wobble=1e-4))
        self.assertTrue(res["ok"], res)
        per = res["native_noise"]["per_output"]
        self.assertEqual(per["0"], {"tol": 0.02, "noise_rel": 0.0})
        self.assertEqual(per["1"]["tol"], 0.001)
        self.assertTrue(0.0 < per["1"]["noise_rel"] < 2e-4, per)
        self.assertEqual(res["native_noise"]["noise_rel"], per["1"]["noise_rel"])
        outputs = [{"output": "0", "tol": 0.02}, {"output": "1", "tol": 0.001}]
        self.assertEqual(res["tolerance"], {"tol": 0.02, "per_output": {"1": 0.001}, "outputs": outputs})
        res = self.run_mode("correct", meta, TwoOut(self.torch, drift=1.005))
        self.assertFalse(res["ok"])
        self.assertEqual(res["tolerance"]["outputs"], outputs)
        self.assertNotIn("native_noise", res)

    def test_a_generated_task_is_held_to_the_generated_ut_rules(self):
        with open(os.path.join(self.task, "unittest.py"), "w", encoding="utf-8") as fh:
            fh.write(lr.GENERATED_MARKER + "\n")
        self.torch.save(self.blob, f"{self.prefix}.pt")
        res = self.run_mode("refs", self.meta(), TwoOut(self.torch))
        self.assertFalse(res["ok"])
        self.assertTrue(res["incomplete"][0].startswith("default: graph coverage: graph_capture is required for a "
                                                        "graph-deployed phase"), res["incomplete"])
        self.assertTrue(all(r["correct"] for r in res["report"]["default"]["random"]))

    # The task's cases.py for main(): the cases two_out_cases builds around the live stack's kernel (each leg hands
    # both to the live module first, as the installed package of that leg would carry its own kernel).
    CASES_PY = '''import torch

import {live} as live

_cases = live.two_out_cases(torch, live.kernel)
call, eager_cases, random_shapes = _cases.call, _cases.eager_cases, _cases.random_shapes
replay, sequence_cases = _cases.replay, _cases.sequence_cases
'''

    def main_leg(self, mode, label, kernel):
        """leg_runner.main() --mode refs / correct on the task directory, `kernel` the live stack's: the --out path,
        the lines main printed and the result file it wrote there."""
        live = sys.modules[self.LIVE]
        live.two_out_cases, live.kernel = two_out_cases, kernel
        dest = os.path.join(self.tmp, f"{label}.json")
        sys.argv = ["leg_runner.py", "--task", self.task, "--mode", mode, "--blob", self.prefix, "--out", dest]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lr.main()
        with open(dest, encoding="utf-8") as fh:
            return dest, out.getvalue().splitlines(), json.load(fh)

    def test_main_runs_the_refs_leg_and_judges_correct_legs_against_its_records(self):
        """leg_runner.main --mode refs, then --mode correct, on a task directory holding its own harness_lib.py (a
        copy of the real one), cases.py and meta.json, all loaded by path: main writes _correct_mode's result to
        --out and prints its verdict; a correct leg is judged against the native outputs the refs leg recorded."""
        shutil.copy(os.path.join(SCRIPTS, "harness_lib.py"), self.task)
        with open(os.path.join(self.task, "cases.py"), "w", encoding="utf-8") as fh:
            fh.write(self.CASES_PY.format(live=self.LIVE))
        with open(os.path.join(self.task, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self.meta(per_output={"1": 0.001}), fh)
        self.torch.save(self.blob, f"{self.prefix}.pt")
        # main() puts the task's harness_lib and cases into sys.modules and takes its rank from the environment
        for name in ("harness_lib", "cases"):
            self.addCleanup(self._put_back, name, sys.modules.get(name, _ABSENT))
        self.addCleanup(setattr, sys, "argv", sys.argv)
        for key in [k for k in os.environ if k in ("RANK", "WORLD_SIZE", "LOCAL_RANK") or k.startswith("GEAK_UT_")]:
            self.addCleanup(os.environ.__setitem__, key, os.environ.pop(key))
        outputs = [{"output": "0", "tol": 0.02}, {"output": "1", "tol": 0.001}]

        dest, printed, res = self.main_leg("refs", "refs", TwoOut(self.torch))
        self.assertEqual(printed, [json.dumps({"out": dest, "ok": True})])
        self.assertEqual({k: res[k] for k in ("rank", "mode", "ok", "incomplete", "errors")},
                         {"rank": 0, "mode": "refs", "ok": True, "incomplete": [], "errors": []})
        self.assertEqual(res["identity"], {
            f"{self.LIVE}:op": {"module": self.LIVE, "qualname": "op",
                                "file": os.path.realpath(os.path.join(self.tmp, "site", f"{self.LIVE}.py"))},
            "geak_cpu_no_such_module:op": {"error": "ModuleNotFoundError: No module named 'geak_cpu_no_such_module'"}})
        rep = res["report"]
        self.assertEqual([[r["case"] for r in rep["default"][s]] for s in ("eager", "random", "graph_replay")],
                         [["e0", "e1", "output_independence"], ["random[0]:r", "random[1]:r", "random[2]:r"],
                          ["g0", "g1"]])
        self.assertEqual([r["case"] for r in rep["sequence"]], ["seq[0]:s0", "seq[1]:s1"])
        self.assertTrue(all(r["correct"] for s in ("eager", "random", "graph_replay") for r in rep["default"][s]))
        self.assertEqual(res["native_noise"], {"max_rel_err": 0.0, "cases": 3, "noise_rel": 0.0, "per_output": {
            "0": {"tol": 0.02, "noise_rel": 0.0}, "1": {"tol": 0.001, "noise_rel": 0.0}}})
        self.assertEqual(res["tolerance"], {"tol": 0.02, "per_output": {"1": 0.001}, "outputs": outputs})
        refs = self.torch.load(f"{self.prefix}.refs.pt")
        self.assertEqual({k: len(v) for k, v in refs.items()}, {"eager": 2, "graph_replay": 2, "sequence": 2})
        # the task's own files ran: its harness copy got this process's rank context, the imported harness_lib none
        task_h = sys.modules["harness_lib"]
        self.assertEqual((task_h.__file__, sys.modules["cases"].__file__),
                         (os.path.join(self.task, "harness_lib.py"), os.path.join(self.task, "cases.py")))
        self.assertEqual(task_h._RANK_CONTEXT.as_dict(), {"rank": 0, "world": 1, "data_rank": 0, "local_rank": 0})
        self.assertIsNone(h._RANK_CONTEXT)

        def judged(res):
            """The rows of the comparisons the refs leg recorded: these cases carry no `ref` of their own."""
            rep = res["report"]
            return ([r for r in rep["default"]["eager"] if r["case"] != "output_independence"]
                    + rep["default"]["graph_replay"] + rep["sequence"])
        # 0.05% off on output 1: within its own tolerance (0.001), comparison by comparison
        dest, printed, res = self.main_leg("correct", "within", TwoOut(self.torch, drift=1.0005))
        self.assertEqual(printed, [json.dumps({"out": dest, "ok": True})])
        self.assertEqual({k: res[k] for k in ("rank", "mode", "ok", "incomplete", "errors")},
                         {"rank": 0, "mode": "correct", "ok": True, "incomplete": [], "errors": []})
        self.assertNotIn("native_noise", res)
        self.assertEqual(res["tolerance"]["outputs"], outputs)
        self.assertEqual([(r["correct"], r["output"], r["tol"]) for r in judged(res)], [(True, "1", 0.001)] * 6)
        self.assertTrue(all(0 < r["max_rel_err"] < 0.001 for r in judged(res)), judged(res))
        # 0.5% off: within the task's 0.02 but not output 1's 0.001 -- every recorded comparison fails
        dest, printed, res = self.main_leg("correct", "off", TwoOut(self.torch, drift=1.005))
        self.assertEqual(printed, [json.dumps({"out": dest, "ok": False})])
        self.assertEqual((res["mode"], res["ok"], res["incomplete"], res["errors"]), ("correct", False, [], []))
        self.assertEqual([(r["correct"], r["output"], r["tol"]) for r in judged(res)], [(False, "1", 0.001)] * 6)
        self.assertTrue(all(r["max_rel_err"] > 0.001 for r in judged(res)), judged(res))
        # without the refs leg's records the correct leg cannot be judged
        os.remove(f"{self.prefix}.refs.pt")
        dest, printed, res = self.main_leg("correct", "unjudged", TwoOut(self.torch))
        self.assertEqual(printed, [json.dumps({"out": dest, "ok": False})])
        self.assertEqual((res["ok"], res["incomplete"], res["errors"], res["report"]),
                         (False, [f"native refs missing: {self.prefix}.refs.pt"], [], {}))


class TestNativeNoiseCPU(unittest.TestCase):
    """A37: what the native refs leg reports of its run-to-run noise."""

    def test_recording_noise_measures_each_comparison_of_the_check_when_the_harness_can(self):
        def check(out, ref, tol):
            return [fake.correct(out, ref, tol)]

        def correct(out, ref, tol):
            return out == ref, abs(out - ref)
        fake = types.SimpleNamespace(check_random_vs_baseline=check, correct=correct)
        noise = []
        self.assertEqual(lr._recording_noise(fake, noise)(1.0, 1.5, 0.1), [(False, 0.5)])
        self.assertEqual(noise, [])                               # a harness without noise_rel: the check as it is
        fake.noise_rel = lambda out, ref: abs(out - ref) / 2
        recording = lr._recording_noise(fake, noise)
        self.assertEqual(recording(1.0, 1.5, 0.1), [(False, 0.5)])
        self.assertEqual(recording(2.0, 2.0, 0.1), [(True, 0.0)])
        self.assertEqual(noise, [0.25, 0.0])
        self.assertIs(fake.correct, correct)                      # put back after every check

    def test_native_noise_reads_the_worst_random_draw_and_counts_non_finite_noise(self):
        report = {"prefill": {"random": [{"max_rel_err": 0.5}, {"max_rel_err": None}, "row"],
                              "eager": [{"max_rel_err": 9.0}]},
                  "decode": {"random": [{"max_rel_err": 2}], "nested": {"random": [{"max_rel_err": 0.25}]}},
                  "sequence": [{"max_rel_err": 7.0}]}
        self.assertEqual(lr._native_noise(report), {"max_rel_err": 2.0, "cases": 3})
        self.assertEqual(lr._native_noise(report, [0.1, math.inf, math.nan, "x", 0.3]),
                         {"max_rel_err": 2.0, "cases": 3, "noise_rel": 0.3, "noise_rel_nonfinite": 3})
        self.assertEqual(lr._native_noise({}, []), {"max_rel_err": None, "cases": 0, "noise_rel": None})
