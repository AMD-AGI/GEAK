"""The vendored tuning skillset is registered in the ONE selector, on the same terms as every other
expert skill — without a byte of GEAK metadata landing inside the hash-pinned tree.

Those two requirements pull against each other, and the whole design is the resolution: the selector
metadata lives in ``tuning_index.yaml``, next to ``index.yaml`` rather than inside ``tuning/``, and
``scaffold.py --reindex`` merges it in. That makes three things checkable, and none of them is checked
anywhere else:

  1. The merge actually happens, and it is idempotent — ``index.yaml`` is committed, so a reindex that
     is not a fixed point turns every unrelated PR into a diff on this file.
  2. Every registered ``file:`` resolves inside the vendored tree. A rename upstream that the descriptor
     does not follow would otherwise show up as a skill the workflow is told to Read and cannot.
  3. ``validation_status`` is DERIVED from ``validate/claims.py``, not typed. The interesting half is
     the negative: an ``N/A`` claim must not be promoted to ``validated``. N/A means the image could not
     answer, and auto-applying an unanswered skill is precisely the failure the validator exists for.

Stdlib + PyYAML only. No GPU, no network, no container.
"""
import importlib.util
import json
import os
import unittest

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # .../expert_skills
INDEX = os.path.join(ROOT, "index.yaml")
TUNING_INDEX = os.path.join(ROOT, "tuning_index.yaml")


def _load_scaffold():
    spec = importlib.util.spec_from_file_location("scaffold", os.path.join(HERE, "scaffold.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scaffold = _load_scaffold()


def _read(path):
    with open(path) as f:
        return yaml.safe_load(f)


class TestTuningSkillsAreInTheSelector(unittest.TestCase):
    def setUp(self):
        self.index = _read(INDEX)
        self.desc = _read(TUNING_INDEX)
        self.tuning = [s for s in self.index["skills"] if s.get("scope") == "tuning"]

    def test_every_descriptor_entry_reached_the_index(self):
        self.assertEqual([s["id"] for s in self.desc["skills"]],
                         [s["id"] for s in self.tuning])
        self.assertTrue(self.tuning, "the tuning skillset registered nothing")

    def test_they_share_the_schema_the_index_declares(self):
        # The point of the merge is that a consumer filtering index.yaml needs no special case.
        for s in self.tuning:
            self.assertEqual(set(s) - {"expects"}, {"id", "file", "scope", "match", "validation_status"})
            self.assertIn("gens", s["match"])
            self.assertIn("dtypes", s["match"])
            self.assertIn("operator", s["match"])

    def test_each_registered_file_exists_in_the_vendored_tree(self):
        for s in self.tuning:
            path = os.path.join(ROOT, s["file"])
            self.assertTrue(os.path.exists(path), f"{s['id']} points at a missing {s['file']}")
            self.assertTrue(s["file"].startswith(self.desc["tree"] + "/"),
                            f"{s['id']} points outside the vendored tree")

    def test_no_geak_metadata_leaked_into_the_pinned_tree(self):
        # If someone "fixes" this by adding frontmatter upstream-side, the manifest breaks on the next
        # re-sync and the vendoring quietly becomes a fork. Catch it here instead, where it is legible.
        for s in self.tuning:
            with open(os.path.join(ROOT, s["file"])) as f:
                head = f.read(400)
            self.assertNotIn("validation_status", head)
            self.assertNotIn("scope:", head)

    def test_reindex_is_a_fixed_point(self):
        with open(INDEX) as f:
            before = f.read()
        try:
            scaffold.reindex()
            with open(INDEX) as f:
                self.assertEqual(before, f.read(),
                                 "index.yaml is committed; a reindex that changes it dirties every PR")
        finally:
            with open(INDEX, "w") as f:
                f.write(before)


class TestStatusComesFromTheValidator(unittest.TestCase):
    def _status(self, tmpdir, rows):
        os.makedirs(os.path.join(tmpdir, "validate"), exist_ok=True)
        with open(os.path.join(tmpdir, "validate", "report_x.json"), "w") as f:
            json.dump({"image": "test", "rows": rows}, f)
        return scaffold.claims_status(tmpdir)

    def test_shipped_reports_are_what_the_committed_status_reflects(self):
        derived = scaffold.claims_status(os.path.join(ROOT, _read(TUNING_INDEX)["tree"]))
        for s in _read(INDEX)["skills"]:
            if s.get("scope") == "tuning":
                claims_skill = next(d["claims_skill"] for d in _read(TUNING_INDEX)["skills"]
                                    if d["id"] == s["id"])
                self.assertEqual(s["validation_status"], derived.get(claims_skill, "unvalidated"))

    def test_a_fail_anywhere_demotes_the_skill(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            got = self._status(td, [{"skill": "tuning-ck", "status": "PASS"},
                                    {"skill": "tuning-ck", "status": "FAIL"}])
            self.assertEqual(got["tuning-ck"], "draft")

    def test_an_image_that_cannot_answer_has_not_validated_anything(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            got = self._status(td, [{"skill": "tuning-ck", "status": "N/A"},
                                    {"skill": "tuning-ck", "status": "N/A"}])
            self.assertEqual(got["tuning-ck"], "unvalidated")

    def test_a_skill_the_validator_never_heard_of_is_not_validated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(self._status(td, []), {})
            entries = scaffold.tuning_index_entries(
                report=os.path.join(td, "validate", "report_x.json"))
            self.assertTrue(entries)
            self.assertTrue(all(e["validation_status"] == "unvalidated" for e in entries))


if __name__ == "__main__":
    unittest.main()
