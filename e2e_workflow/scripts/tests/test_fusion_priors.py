"""The known-fusion recall gate.

The failure this pins: on DSR1 the same model, the same trace and the same roles
produced a DIFFERENT fusion set on different runs. Nothing errored -- the
analyst simply rediscovered the opportunity space from scratch each time, and a
fusion that had already been measured at +11.8% e2e was silently never proposed
again. "Never proposed" leaves no artifact, so the report of the weaker run read
exactly as clean as the report of the stronger one.

The gate is on the DISPOSITION EXISTING, never on the answer being yes: per
knowledge/learned/README.md a card may only ADD candidates, so `not_applicable`
with an honest reason passes. Only silence fails.
"""

import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_priors as priors


CARD = """---
key: {key}
type: {type}
confidence: {confidence}
effect: measured
last_seen: 2026-08-26
---
# {title}
- lever: try it
- source: eval_dir
"""


class FusionPriorsTest(unittest.TestCase):
    def _kb(self, cards, group="kernel fusion", extra_groups=""):
        root = tempfile.mkdtemp()
        lines = ["# Learned", "", "## dense GEMM",
                 "- [gfx942] something else ★★ — (other-card.md)",
                 "", "## " + group, ""]
        with open(os.path.join(root, "other-card.md"), "w") as fh:
            fh.write(CARD.format(key="gemm", type="routing", confidence="**",
                                 title="Other"))
        for card in cards:
            name = card["slug"] + ".md"
            with open(os.path.join(root, name), "w") as fh:
                fh.write(CARD.format(
                    key=card.get("key", "k"), type=card["type"],
                    confidence=card.get("confidence", "★★"),
                    title=card.get("title", card["slug"])))
            lines.append("- [%s] blurb %s — (%s)"
                         % (card.get("key", "k"),
                            card.get("confidence", "★★"), name))
        lines.append("")
        lines.append(extra_groups)
        index = os.path.join(root, "INDEX.md")
        with open(index, "w") as fh:
            fh.write("\n".join(lines))
        return index

    # ---- loading -------------------------------------------------------

    def test_only_the_fusion_group_is_loaded(self):
        index = self._kb([{"slug": "fusion-a", "type": "lever"}])
        loaded = priors.load_priors(index)
        self.assertEqual([p["slug"] for p in loaded], ["fusion-a"])

    def test_a_method_card_is_advisory_not_disposable(self):
        index = self._kb([{"slug": "method-x", "type": "method"},
                          {"slug": "fusion-a", "type": "lever"}])
        loaded = {p["slug"]: p for p in priors.load_priors(index)}
        self.assertFalse(loaded["method-x"]["disposable"])
        self.assertTrue(loaded["fusion-a"]["disposable"])

    def test_a_card_with_no_frontmatter_still_demands_a_disposition(self):
        """Fail CLOSED: a malformed card must not drop out of the denominator."""
        index = self._kb([{"slug": "fusion-a", "type": "lever"}])
        with open(os.path.join(os.path.dirname(index), "fusion-a.md"), "w") as fh:
            fh.write("just a note, no frontmatter\n")
        loaded = priors.load_priors(index)
        self.assertTrue(loaded[0]["disposable"])

    def test_a_missing_index_is_not_a_crash(self):
        self.assertEqual(priors.load_priors("/nonexistent/INDEX.md"), [])

    def test_the_card_title_and_confidence_are_carried(self):
        index = self._kb([{"slug": "fusion-a", "type": "lever",
                           "title": "V-absorb dequant fold",
                           "confidence": "★★★"}])
        loaded = priors.load_priors(index)[0]
        self.assertEqual(loaded["title"], "V-absorb dequant fold")
        self.assertEqual(loaded["confidence"], "★★★")

    # ---- the gate ------------------------------------------------------

    def _priors(self, *slugs):
        index = self._kb([{"slug": s, "type": "lever"} for s in slugs])
        return priors.load_priors(index)

    def test_a_prior_with_no_disposition_fails(self):
        errors, rows = priors.check_dispositions(
            self._priors("fusion-a"), {"prior_dispositions": []})
        self.assertTrue(any("no disposition" in e for e in errors), errors)
        self.assertIsNone(rows[0]["disposition"])

    def test_an_absent_prior_dispositions_field_fails(self):
        errors, _ = priors.check_dispositions(self._priors("fusion-a"), {})
        self.assertTrue(any("prior_dispositions missing" in e for e in errors),
                        errors)

    def test_not_applicable_with_a_real_reason_passes(self):
        errors, _ = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [{
                "card": "fusion-a", "disposition": "not_applicable",
                "reason": "card is gfx942 MLA; this run is gfx950 dense"}]})
        self.assertEqual(errors, [])

    def test_not_applicable_with_an_empty_reason_fails(self):
        for reason in ("", "skipped", "n/a", "  "):
            errors, _ = priors.check_dispositions(
                self._priors("fusion-a"),
                {"prior_dispositions": [{
                    "card": "fusion-a", "disposition": "not_applicable",
                    "reason": reason}]})
            self.assertTrue(any("needs a real reason" in e for e in errors),
                            (reason, errors))

    def test_carrying_a_prior_must_name_the_candidate_it_became(self):
        errors, _ = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [{
                "card": "fusion-a", "disposition": "candidate"}]},
            {"d1_x"})
        self.assertTrue(any("needs the candidate_id" in e for e in errors),
                        errors)

    def test_a_candidate_id_that_is_not_in_this_run_fails(self):
        errors, _ = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [{
                "card": "fusion-a", "disposition": "candidate",
                "candidate_id": "d9_ghost"}]},
            {"d1_x"})
        self.assertTrue(any("not in this run's candidates" in e for e in errors),
                        errors)

    def test_a_carried_prior_that_matches_a_real_candidate_passes(self):
        errors, rows = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [{
                "card": "fusion-a", "disposition": "candidate",
                "candidate_id": "d1_x"}]},
            {"d1_x"})
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["candidate_id"], "d1_x")

    def test_the_card_name_may_carry_its_md_suffix(self):
        errors, _ = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [{
                "card": "fusion-a.md", "disposition": "already_engaged",
                "reason": "aiter fused AR is the live default at this shape",
                "engagement_evidence": {
                    "ownership_scope": "semantic_region",
                    "kernel": "fused_allreduce_rmsnorm",
                    "removed_kernel_count": 0}}]})
        self.assertEqual(errors, [])

    def test_already_engaged_rejects_whole_layer_name_only_evidence(self):
        errors, _ = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [{
                "card": "fusion-a", "disposition": "already_engaged",
                "reason": "a similarly named kernel appears in the layer",
                "engagement_evidence": {
                    "ownership_scope": "layer",
                    "kernel": "batched_gemm",
                    "removed_kernel_count": 0}}]})
        self.assertTrue(any("ownership_scope" in e for e in errors), errors)

    def test_an_unknown_disposition_word_fails(self):
        errors, _ = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [{
                "card": "fusion-a", "disposition": "maybe_later",
                "reason": "some reason"}]})
        self.assertTrue(any("must be one of" in e for e in errors), errors)

    def test_citing_a_card_that_is_not_in_the_index_fails(self):
        errors, _ = priors.check_dispositions(
            self._priors("fusion-a"),
            {"prior_dispositions": [
                {"card": "fusion-a", "disposition": "not_applicable",
                 "reason": "different gfx"},
                {"card": "fusion-ghost", "disposition": "candidate",
                 "candidate_id": "d1_x"}]},
            {"d1_x"})
        self.assertTrue(any("unknown card" in e for e in errors), errors)

    def test_a_method_card_needs_no_disposition(self):
        index = self._kb([{"slug": "method-x", "type": "method"}])
        errors, rows = priors.check_dispositions(
            priors.load_priors(index), {"prior_dispositions": []})
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)

    def test_an_empty_kb_imposes_nothing(self):
        errors, rows = priors.check_dispositions([], {})
        self.assertEqual((errors, rows), ([], []))

    # ---- reporting -----------------------------------------------------

    def test_the_summary_counts_each_disposition(self):
        given = self._priors("a", "b", "c")
        _, rows = priors.check_dispositions(
            given,
            {"prior_dispositions": [
                {"card": "a", "disposition": "candidate", "candidate_id": "x"},
                {"card": "b", "disposition": "already_engaged",
                 "reason": "live default", "engagement_evidence": {
                     "ownership_scope": "kernel_exact",
                     "kernel": "fused_b", "removed_kernel_count": 0}}]},
            {"x"})
        summary = priors.summarise(rows)
        self.assertEqual(summary["priors_total"], 3)
        self.assertEqual(summary["carried"], 1)
        self.assertEqual(summary["already_engaged"], 1)
        self.assertEqual(summary["undisposed"], 1)

    def test_an_undisposed_prior_is_red_in_the_report(self):
        given = self._priors("fusion-a")
        _, rows = priors.check_dispositions(given, {"prior_dispositions": []})
        markdown = priors.render_markdown(rows)
        self.assertIn("\U0001f534", markdown)
        self.assertIn("fusion-a", markdown)

    def test_a_fully_disposed_set_is_not_red(self):
        given = self._priors("fusion-a")
        _, rows = priors.check_dispositions(
            given,
            {"prior_dispositions": [{
                "card": "fusion-a", "disposition": "not_applicable",
                "reason": "op absent from this model's trace"}]})
        markdown = priors.render_markdown(rows)
        self.assertNotIn("\U0001f534", markdown)
        self.assertIn("op absent", markdown)

    def test_the_report_says_priors_only_add(self):
        given = self._priors("fusion-a")
        _, rows = priors.check_dispositions(given, {"prior_dispositions": []})
        self.assertIn("ADD-only", priors.render_markdown(rows))


if __name__ == "__main__":
    unittest.main()


class MeasuredPriorTest(unittest.TestCase):
    def test_gain_family_and_gfx_parsing(self):
        self.assertEqual(priors.parse_gain_pct("TPOT -5% ... +6.046% e2e output"), 6.046)
        self.assertEqual(priors.parse_gain_pct("kernel -9.13%", "+1.583% output tok/s"), 1.583)
        self.assertIsNone(priors.parse_gain_pct("-9.13% decode kernel time"))
        self.assertEqual(priors.model_family("Qwen3.5-397B-A17B-FP8"), "qwen35")
        self.assertEqual(priors.model_family("Qwen3_5LinearDecoderLayer"), "qwen35")
        self.assertEqual(priors.model_family("Qwen3-14B-FP8"), "qwen3")
        self.assertEqual(priors.gfx_targets("[gfx942 · sglang]"), ["gfx942"])

    def test_cross_branch_priors_file(self):
        loaded = {p["prior_id"]: p for p in priors.load_measured_priors()}
        gains = {pid: p["measured_e2e_gain_pct"] for pid, p in loaded.items()}
        self.assertEqual(gains, {
            "xb-qwen35-router-topk": 6.08,
            "xb-qwen35-ar-norm-quant": 0.66,
            "xb-qwen35-qk-norm-rope-cache": 0.96,
            "xb-qwen35-post-attn-ar-norm-dual-quant": 1.08,
            "xb-qwen35-silu-quant": 1.26,
            "xb-qwen35-qkvz-ba-dual-gemm": 17.72,
        })
        for prior in loaded.values():
            self.assertEqual(prior["source"], "user-reported, other GEAK branch, reported 2026-10-11")
            self.assertEqual(prior["model_family"], "qwen35")
            self.assertEqual(prior["gfx"], ["gfx942"])
            self.assertTrue(prior["seam"]["members"])

    def test_learned_cards_with_a_measured_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "INDEX.md"), "w") as fh:
                fh.write("## kernel fusion\n"
                         "- [gfx942 · sglang Qwen3.5 MoE decode] router fold ★★★ (router.md)\n"
                         "- [gfx942] how to A/B (method.md)\n")
            with open(os.path.join(tmp, "router.md"), "w") as fh:
                fh.write("---\nkey: kernel fusion (router GEMM + topk softmax) · gfx942 · "
                         "sglang Qwen3.5 MoE decode\ntype: lever\neffect: +6.046% e2e output tok/s\n"
                         "---\n# router\n")
            with open(os.path.join(tmp, "method.md"), "w") as fh:
                fh.write("---\nkey: method\ntype: method\neffect: +3% e2e\n---\n# m\n")
            found = priors.learned_measured_priors(os.path.join(tmp, "INDEX.md"))
        self.assertEqual(len(found), 1)
        prior = found[0]
        self.assertEqual(prior["measured_e2e_gain_pct"], 6.046)
        self.assertEqual(prior["model_family"], "qwen35")
        self.assertEqual(prior["phases"], ["decode"])
        self.assertEqual(prior["seam"]["tags"], ["gemm", "topk"])


class PriorStoreV2Test(unittest.TestCase):
    """measured_priors.json schema 2: measurements, polarity, append-only API."""

    SEAM = {"prior_id": "t-silu-quant", "title": "SiLU+Quant", "model_family": "Qwen3.5",
            "model": "Qwen3.5-397B-A17B-FP8", "gfx": "gfx942", "framework": "sglang",
            "seam": {"kinds": ["serial"], "members": [{"cls": "activation"}, {"cls": "quant"}]}}

    @staticmethod
    def _m(gain, date, source="run"):
        return {"gain_pct": gain, "date": date, "protocol": {"isl": 8192, "osl": 1024, "conc": 4},
                "source": source, "evidence": None}

    def test_repository_store_is_schema_2(self):
        with open(priors.DEFAULT_MEASURED_PRIORS) as fh:
            data = json.load(fh)
        self.assertEqual(data["schema_version"], 2)
        for entry in data["priors"]:
            for key in ("prior_id", "title", "model_family", "model", "gfx", "framework",
                        "polarity", "seam", "measurements"):
                self.assertIn(key, entry)
            self.assertIn(entry["polarity"], ("positive", "negative"))
            for measurement in entry["measurements"]:
                self.assertEqual(set(measurement) >= {"gain_pct", "date", "protocol", "source",
                                                      "evidence"}, True)

    def test_effective_gain_is_the_median_of_the_last_three_by_date(self):
        prior = {"measurements": [self._m(9.0, "2026-01-01"), self._m(1.0, "2026-03-01"),
                                  self._m(2.0, "2026-02-01"), self._m(4.0, "2026-04-01")]}
        # Last three by date: 1.0 (03), 2.0 (02), 4.0 (04) -> median 2.0.
        self.assertEqual(priors.effective_gain(prior), 2.0)
        self.assertEqual(priors.effective_gain({"measurements": [self._m(1.0, "2026-01-01"),
                                                                 self._m(3.0, "2026-01-02")]}), 2.0)
        self.assertIsNone(priors.effective_gain({"measurements": []}))
        self.assertEqual(priors.effective_gain({"measured_e2e_gain_pct": 1.5}), 1.5)

    def test_append_creates_appends_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "measured_priors.json")
            created = priors.append_measurement(path, self.SEAM, self._m(1.2, "2026-10-11"), "positive")
            self.assertEqual(len(created["measurements"]), 1)
            priors.append_measurement(path, "t-silu-quant", self._m(1.4, "2026-10-12"), "positive")
            priors.append_measurement(path, "t-silu-quant", self._m(1.4, "2026-10-12"), "positive")
            updated = priors.append_measurement(path, "t-silu-quant",
                                                self._m(-0.8, "2026-10-13", "stack"), "negative")
            self.assertEqual([m["gain_pct"] for m in updated["measurements"]], [1.2, 1.4, -0.8])
            self.assertEqual(updated["polarity"], "negative")
            loaded = {p["prior_id"]: p for p in priors.load_measured_priors(path)}
            self.assertEqual(loaded["t-silu-quant"]["measured_e2e_gain_pct"], 1.2)
            self.assertEqual(loaded["t-silu-quant"]["polarity"], "negative")
            self.assertEqual(loaded["t-silu-quant"]["model_family"], "qwen35")
            with open(path) as fh:
                self.assertEqual(json.load(fh)["schema_version"], 2)
            self.assertEqual([n for n in os.listdir(tmp) if n.endswith(".tmp")], [])
            with self.assertRaises(KeyError):
                priors.append_measurement(path, "nope", self._m(1.0, "2026-10-11"), "positive")
            with self.assertRaises(ValueError):
                priors.append_measurement(path, "t-silu-quant", self._m(1.0, "11/10/2026"), "positive")
            with self.assertRaises(ValueError):
                priors.append_measurement(path, "t-silu-quant", self._m(1.0, "2026-10-11"), "maybe")

    def test_append_migrates_a_schema_1_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "measured_priors.json")
            with open(path, "w") as fh:
                json.dump({"schema_version": 1, "priors": [dict(
                    self.SEAM, measured_e2e_gain_pct=1.26,
                    source="reported 2026-10-11", workload={"isl": 1})]}, fh)
            self.assertEqual(priors.load_measured_priors(path)[0]["measured_e2e_gain_pct"], 1.26)
            updated = priors.append_measurement(path, "t-silu-quant", self._m(2.0, "2026-10-12"),
                                                "positive")
            self.assertEqual([(m["gain_pct"], m["date"]) for m in updated["measurements"]],
                             [(1.26, "2026-10-11"), (2.0, "2026-10-12")])
            self.assertEqual(updated["measurements"][0]["protocol"], {"isl": 1})
