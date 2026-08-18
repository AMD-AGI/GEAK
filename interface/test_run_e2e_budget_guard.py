#!/usr/bin/env python3
"""Tests for the wall-clock budget guard that protects Finalize/Report/Validate.

CONTRACT under test: when the orchestrator grants a wall-clock budget, GEAK must
reserve a share of it for the closing Finalize -> Report -> Validate phases and
must NOT let optional optimization work (a nested kernel workflow, a finish-A/B
drain, an idle wait for the box) consume that reserve. Concretely:

  1. the budget partition is derived proportionally from the granted budget and
     forwarded to the workflow, so the workflow's "abandon optional work" deadline
     and this runner's stand-down sentinel are the SAME instant,
  2. the sentinel file appears at that instant (and never when no budget was
     granted), because it is the only mechanism that can make an ALREADY-RUNNING
     optional bench release the single serving slot,
  3. a first pass that died early enough to leave the reserve unspent triggers a
     scoped ``phases=final`` re-entry rather than a disk-synthesized result,
  4. a ``no_gain`` recovered from disk says WHICH of its two causes applies, so
     "nothing won" is distinguishable from "cut off before the A/Bs were judged".

Run: python3 -m pytest GEAK/interface/test_run_e2e_budget_guard.py -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SENTINEL = object()


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


class _Case(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        self._attrs: list[tuple[str, object]] = []
        self._sigterm = signal.getsignal(signal.SIGTERM)
        self.tmp = Path(tempfile.mkdtemp(prefix="run_e2e_budget_"))
        self.addCleanup(self._restore)

    def _restore(self):
        for name, prev in reversed(self._attrs):
            if prev is _SENTINEL:
                delattr(rx, name)
            else:
                setattr(rx, name, prev)
        os.environ.clear()
        os.environ.update(self._env)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, self._sigterm)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def patch_rx(self, name, value):
        self._attrs.append((name, getattr(rx, name, _SENTINEL)))
        setattr(rx, name, value)


# =========================================================================== #
# 1. budget partition
# =========================================================================== #
class TestBudgetPartition(_Case):
    """The partition must be PROPORTIONAL to the granted budget: the same formula
    has to behave sensibly from a 1h budget to a 24h one with no per-run tuning."""

    def _parts(self, timeout_s: int, h: dict | None = None) -> tuple[int, int, int]:
        p = rx.resolve_budget_partition(timeout_s, h or {})
        fin = p["time_finalize_deadline_ms"] // 1000
        cap = p["nested_workflow_cap_ms"] // 1000
        return fin, cap, timeout_s - fin

    def test_reserve_is_carved_for_every_budget_size(self):
        for budget in (3600, 21600, 32400, 43200, 86400):
            fin, cap, reserve = self._parts(budget)
            with self.subTest(budget=budget):
                # The closing window must be non-empty and must not swallow the run.
                self.assertGreater(reserve, 0, "no reserve carved")
                self.assertLess(fin, budget, "finalize deadline at/after the kill")
                self.assertGreater(fin, 0)
                # A single nested workflow may never span the whole budget.
                self.assertLess(cap, budget)

    def test_finalize_deadline_is_after_the_dispatch_deadline(self):
        """D2 must never precede D1: stopping NEW work has to come first, else the
        dispatch guard is dead code and every task is abandoned on arrival."""
        for budget in (3600, 32400, 86400):
            effective = max(60, budget - max(1200, int(budget * 0.08)))
            dispatch = max(int(effective * 0.6), effective - 10800)
            fin, _cap, _reserve = self._parts(budget)
            with self.subTest(budget=budget):
                self.assertGreaterEqual(fin, dispatch)
                self.assertLessEqual(fin, effective)

    def test_deliverable_share_is_tunable_and_clamped(self):
        """The share is a dimensionless proportion (never a tuned second count),
        and out-of-range values are clamped rather than producing a zero window."""
        base, _, _ = self._parts(32400)
        bigger_share, _, _ = self._parts(32400, {"deliverable_tail_share": 0.9})
        self.assertLess(bigger_share, base, "a larger share must reserve more time")
        for bad in (0.0, -5, 12, "junk", None):
            fin, _cap, reserve = self._parts(32400, {"deliverable_tail_share": bad})
            with self.subTest(share=bad):
                self.assertGreater(reserve, 0)
                self.assertLess(fin, 32400)

    def test_finalize_deadline_s_matches_the_forwarded_arg(self):
        for budget in (3600, 32400):
            p = rx.resolve_budget_partition(budget, {})
            self.assertEqual(
                rx.finalize_deadline_s(budget, {}),
                p["time_finalize_deadline_ms"] // 1000,
            )

    def test_map_args_forwards_the_partition_with_the_budget(self):
        h = {
            "model_path": "/models/fake",
            "exp_root": str(self.tmp / "exp"),
            "workload": {"isl": 1024, "osl": 1024, "conc": 64},
            "tp": 1,
        }
        args = rx.map_args(dict(h), 32400)
        self.assertEqual(args["time_budget_s"], 32400)
        expect = rx.resolve_budget_partition(32400, h)
        self.assertEqual(args["time_finalize_deadline_ms"],
                         expect["time_finalize_deadline_ms"])
        self.assertEqual(args["nested_workflow_cap_ms"],
                         expect["nested_workflow_cap_ms"])

    def test_no_budget_means_no_partition_args(self):
        """Budget-unaware invocation must stay byte-identical: absent budget =>
        none of the new args appear, so every guard in the workflow is inert."""
        h = {
            "model_path": "/models/fake",
            "exp_root": str(self.tmp / "exp"),
            "workload": {"isl": 1024, "osl": 1024, "conc": 64},
            "tp": 1,
        }
        args = rx.map_args(dict(h), None)
        for k in ("time_budget_s", "time_finalize_deadline_ms",
                  "nested_workflow_cap_ms"):
            self.assertNotIn(k, args)


# =========================================================================== #
# 2. stand-down sentinel
# =========================================================================== #
class TestFinalizeSentinel(_Case):
    def test_sentinel_appears_after_the_delay(self):
        eval_dir = self.tmp / "e2e"
        eval_dir.mkdir()
        path = Path(rx.arm_finalize_sentinel(str(eval_dir), 0))
        # The timer thread is a daemon; join by polling its product.
        for _ in range(200):
            if path.exists():
                break
            import time as _t
            _t.sleep(0.01)
        self.assertTrue(path.exists(), "sentinel never created")
        self.assertEqual(path.name, rx.FINALIZE_SENTINEL_NAME)
        self.assertIn("stand down", path.read_text(encoding="utf-8"))

    def test_sentinel_is_not_created_before_the_delay(self):
        eval_dir = self.tmp / "e2e"
        eval_dir.mkdir()
        path = Path(rx.arm_finalize_sentinel(str(eval_dir), 3600))
        self.assertFalse(path.exists())


class TestMainArmsTheSentinel(_Case):
    """main() must publish the sentinel PATH unconditionally (so bench_e2e.sh
    resolves the same file either way) but only ARM the timer under a budget."""

    def _run(self, *, budget: str | None) -> tuple[int, dict]:
        exp_root = self.tmp / "exp" / "geak"
        exp_root.mkdir(parents=True)
        eval_dir = exp_root / "e2e_main"
        handoff = self.tmp / "handoff.json"
        handoff.write_text(json.dumps({
            "schema_version": 2, "model_path": "/models/fake", "framework": "vllm",
            "tp": 1, "workload": {"isl": 1024, "osl": 1024, "conc": 64},
            "exp_root": str(exp_root), "eval_dir": str(eval_dir),
        }), encoding="utf-8")
        result_path = self.tmp / "result.json"
        if budget is None:
            os.environ.pop("GEAK_E2E_TIMEOUT_S", None)
        else:
            os.environ["GEAK_E2E_TIMEOUT_S"] = budget
        self.patch_rx("_git_short_sha", lambda root: "abc1234")
        seen: dict = {}

        def fake_invoke(prompt, timeout_s, eval_dir_arg):
            seen["armed"] = os.environ.get("GEAK_FINALIZE_NOW_FILE", "")
            seen["sentinel_exists"] = bool(
                seen["armed"] and Path(seen["armed"]).exists())
            (Path(eval_dir_arg)).mkdir(parents=True, exist_ok=True)
            return {"eval_dir": eval_dir_arg, "baseline_throughput_tok_s": 100.0,
                    "final_throughput_tok_s": 110.0, "throughput_speedup": 1.1,
                    "output_parity": "pass"}

        self.patch_rx("invoke_workflow", fake_invoke)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = rx.main([str(handoff), str(result_path)])
        return rc, seen

    def test_path_published_and_file_absent_at_launch(self):
        rc, seen = self._run(budget="43200")
        self.assertEqual(rc, 0)
        self.assertTrue(seen["armed"].endswith(rx.FINALIZE_SENTINEL_NAME))
        self.assertFalse(seen["sentinel_exists"],
                         "no leg may be stood down at run start")

    def test_path_published_even_without_a_budget(self):
        rc, seen = self._run(budget=None)
        self.assertEqual(rc, 0)
        self.assertTrue(seen["armed"].endswith(rx.FINALIZE_SENTINEL_NAME))
        self.assertFalse(seen["sentinel_exists"])


# =========================================================================== #
# 3. leftover-budget phases=final self-rescue
# =========================================================================== #
class TestFinalizeRescue(_Case):
    def _args(self) -> dict:
        return {"eval_dir": str(self.tmp / "e2e"), "model_path": "/models/fake"}

    def _call(self, **kw):
        defaults = dict(
            wf=None, err=None, err_class=None, h={},
            ps_args=self._args(), exp_root=self.tmp / "exp",
            eval_dir_hint=str(self.tmp / "e2e"),
            timeout_s=32400, elapsed_s=10.0,
        )
        defaults.update(kw)
        os.environ.setdefault("GEAK_FINALIZE_NOW_FILE",
                              str(self.tmp / "e2e" / rx.FINALIZE_SENTINEL_NAME))
        with contextlib.redirect_stderr(io.StringIO()):
            return rx._maybe_rescue_finalize(
                defaults["wf"], defaults["err"], defaults["err_class"],
                defaults["h"], defaults["ps_args"], defaults["exp_root"],
                defaults["eval_dir_hint"], defaults["timeout_s"],
                defaults["elapsed_s"],
            )

    def test_rescue_fires_when_the_pass_died_early_with_budget_left(self):
        calls: list[dict] = []

        def fake_invoke(prompt, timeout_s, eval_dir):
            calls.append({"timeout_s": timeout_s, "prompt": prompt})
            return {"eval_dir": eval_dir, "throughput_speedup": 1.2,
                    "validation_status": "validated_win"}

        self.patch_rx("invoke_workflow", fake_invoke)
        self.patch_rx("build_prompt", lambda a: json.dumps(a))
        wf, err, err_class = self._call(err=RuntimeError("scrape failed"))
        self.assertEqual(len(calls), 1, "rescue did not run")
        self.assertTrue(wf["rescued_finalize_pass"])
        self.assertIsNone(err)
        self.assertIsNone(err_class)
        # It must be a SCOPED re-entry: only the closing phases, cheap A/B.
        sent = json.loads(calls[0]["prompt"])
        self.assertEqual(sent["phases"], "final")
        self.assertEqual(sent["e2e_repeats"], 1)
        # Funded by the LEFTOVER clock, never the full budget again.
        self.assertLess(calls[0]["timeout_s"], 32400)

    def test_no_rescue_when_the_budget_is_already_spent(self):
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("must not re-enter"))
        wf, err, _ = self._call(err=TimeoutError("killed"), elapsed_s=32400.0)
        self.assertIsNone(wf)
        self.assertIsInstance(err, TimeoutError)

    def test_no_rescue_after_a_live_workflow_return(self):
        """A healthy first pass must never pay for a second one."""
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("must not re-enter"))
        live = {"eval_dir": str(self.tmp / "e2e"), "throughput_speedup": 1.3}
        wf, _err, _c = self._call(wf=live)
        self.assertIs(wf, live)

    def test_no_rescue_when_a_terminal_marker_is_on_disk(self):
        eval_dir = self.tmp / "e2e"
        eval_dir.mkdir(parents=True)
        (eval_dir / "director_e2e_validation.json").write_text("{}", encoding="utf-8")
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("must not re-enter"))
        wf, _err, _c = self._call(err=RuntimeError("boom"))
        self.assertIsNone(wf)

    def test_no_rescue_without_a_budget(self):
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("must not re-enter"))
        wf, _err, _c = self._call(err=RuntimeError("boom"), timeout_s=0)
        self.assertIsNone(wf)

    def test_a_worse_rescue_never_demotes_the_first_result(self):
        self.patch_rx("invoke_workflow", lambda *a, **k: {
            "eval_dir": str(self.tmp / "e2e"), "throughput_speedup": 1.0})
        self.patch_rx("build_prompt", lambda a: "prompt")
        prior = {"eval_dir": str(self.tmp / "e2e"), "throughput_speedup": 1.4,
                 "recovered_intermediate": True}
        wf, _err, _c = self._call(wf=prior)
        self.assertIs(wf, prior)

    def test_rescue_min_is_overridable(self):
        os.environ["GEAK_RESCUE_MIN_S"] = "999999"
        self.patch_rx("invoke_workflow",
                      lambda *a, **k: self.fail("must not re-enter"))
        wf, _err, _c = self._call(err=RuntimeError("boom"))
        self.assertIsNone(wf)


# =========================================================================== #
# 4. no_gain cause attribution
# =========================================================================== #
class TestIncompleteAbScan(_Case):
    def _eval_dir(self) -> Path:
        d = self.tmp / "e2e"
        (d / "baseline").mkdir(parents=True)
        (d / "baseline" / "bench_summary.json").write_text(
            json.dumps({"output_throughput_tok_s_median": 500.0}), encoding="utf-8")
        return d

    def _cand(self, eval_dir: Path, name: str, *, result: dict | None,
              cand_runs: bool) -> Path:
        c = eval_dir / "overlay" / f"cand_{name}"
        c.mkdir(parents=True)
        if result is not None:
            (c / "integrate_result.json").write_text(
                json.dumps(result), encoding="utf-8")
        if cand_runs:
            (c / "cand").mkdir()
            (c / "cand" / "bench_runs.jsonl").write_text("{}\n", encoding="utf-8")
        return c

    def test_missing_result_file_counts_as_incomplete(self):
        d = self._eval_dir()
        self._cand(d, "op_a", result=None, cand_runs=False)
        self.assertEqual(rx._scan_incomplete_ab(d), ["op_a"])

    def test_ref_only_ab_counts_as_incomplete(self):
        d = self._eval_dir()
        self._cand(d, "op_a", result={"gate": "incomplete", "ab_complete": False},
                   cand_runs=False)
        self.assertEqual(rx._scan_incomplete_ab(d), ["op_a"])

    def test_terminal_gate_with_both_legs_is_complete(self):
        d = self._eval_dir()
        self._cand(d, "op_a", result={"gate": "rejected"}, cand_runs=True)
        self._cand(d, "op_b", result={"ab_complete": True}, cand_runs=False)
        self.assertEqual(rx._scan_incomplete_ab(d), [])

    def test_scan_covers_the_final_overlay_too(self):
        d = self._eval_dir()
        c = d / "final" / "overlay" / "cand_op_late"
        c.mkdir(parents=True)
        self.assertEqual(rx._scan_incomplete_ab(d), ["op_late"])

    def test_no_gain_distinguishes_cut_off_from_nothing_won(self):
        d = self._eval_dir()
        self._cand(d, "op_a", result={"gate": "rejected"}, cand_runs=True)
        clean = rx._recover_completed_no_gain(d)
        self.assertEqual(clean["validation_status"], "recovered_no_gain")
        self.assertEqual(clean["incomplete_ab"], [])
        self.assertEqual(clean["resume_hint"], "")

        self._cand(d, "op_b", result=None, cand_runs=False)
        cut = rx._recover_completed_no_gain(d)
        self.assertEqual(cut["validation_status"], "recovered_no_gain_incomplete_ab")
        self.assertEqual(cut["incomplete_ab"], ["op_b"])
        self.assertEqual(cut["resume_hint"], "phases=final")
        # Do-no-harm is preserved in BOTH cases: never a fabricated speedup.
        self.assertEqual(cut["throughput_speedup"], 1.0)

    def test_result_json_surfaces_the_cause_and_the_telemetry(self):
        d = self._eval_dir()
        self._cand(d, "op_b", result=None, cand_runs=False)
        wf = rx._recover_completed_no_gain(d)
        wf["budget_telemetry"] = {"finalize_deadline_fired": True,
                                  "preempted": ["workflow:op_b"]}
        out = rx.normalize_result({"workload": {"isl": 1, "osl": 1, "conc": 1}}, wf)
        self.assertEqual(out["status"], "no_gain")
        self.assertEqual(out["incomplete_ab"], ["op_b"])
        self.assertEqual(out["resume_hint"], "phases=final")
        self.assertTrue(out["budget_telemetry"]["finalize_deadline_fired"])
        self.assertFalse(out["rescued_finalize_pass"])


# =========================================================================== #
# 5. the two implementations of the partition must not drift apart
# =========================================================================== #
class TestPartitionAgreesWithTheWorkflow(_Case):
    """The workflow derives the SAME partition itself when the args are absent (a
    direct, non-interface invocation). Two implementations of one formula drift, and
    a drift here is silent: the sentinel would fire at a different instant than the
    workflow's own deadline, so an optional leg could be stood down while the
    workflow still awaits it, or vice versa. Pin every shared number.
    """

    def setUp(self):
        super().setUp()
        js = _HERE.parent / "e2e_workflow" / "e2e_workflow.js"
        if not js.is_file():
            self.skipTest(f"workflow script not found at {js}")
        self.src = js.read_text(encoding="utf-8")

    def _line_with(self, needle: str) -> str:
        for line in self.src.splitlines():
            if needle in line and not line.strip().startswith("//"):
                return line
        self.fail(f"no code line contains {needle!r}")

    def test_effective_budget_carve_matches(self):
        line = self._line_with("TIME_BUDGET_MS - Math.max(")
        self.assertIn(f"{rx.BUDGET_TAIL_FRACTION}", line)
        self.assertIn(f"{rx.BUDGET_TAIL_FLOOR_S * 1000}", line)

    def test_dispatch_deadline_fraction_matches(self):
        line = self._line_with("TIME_BUDGET_EFFECTIVE_MS * ")
        self.assertIn(f"{rx.BUDGET_DISPATCH_FRACTION}", line)

    def test_tail_cap_default_matches(self):
        line = self._line_with("const TIME_TAIL_CAP_MS")
        self.assertIn(f"{rx.BUDGET_TAIL_CAP_S}", line)

    def test_deliverable_share_default_matches(self):
        line = self._line_with("A.deliverable_tail_share")
        self.assertIn(f"{rx.DELIVERABLE_TAIL_SHARE}", line)

    def test_nested_cap_floor_matches(self):
        line = self._line_with("const FAST_HEAD_WF_MS")
        self.assertIn(f"{rx.NESTED_WF_CAP_FLOOR_S * 1000}", line)

    def test_workflow_reads_the_forwarded_args(self):
        """Forwarding is pointless if the workflow ignores it."""
        for arg in ("time_finalize_deadline_ms", "nested_workflow_cap_ms",
                    "deliverable_tail_share"):
            self.assertIn(f"A.{arg}", self.src, f"workflow never reads args.{arg}")

    def test_deliverable_phases_are_never_preempted(self):
        """The guard must not be able to abandon the very phases it protects."""
        self.assertIn("NON_PREEMPTIBLE_PHASES", self.src)
        line = self._line_with("const NON_PREEMPTIBLE_PHASES")
        for phase in ("Finalize", "Report", "Validate"):
            self.assertIn(phase, line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
