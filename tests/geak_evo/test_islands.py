"""Tests for geak_evo.archive.islands (drift risk 7b: island import is idempotent).

Synthetic CPU fixtures: two member campaigns of one job, each with its own store
and epoch. Member B publishes one reviewed role; member A imports it.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_islands.py
Parity run on the original modules: set EVO_PARITY_ORIGINAL=1 and put the
original ``evolution/src`` and ``bridge`` on PYTHONPATH instead (the original
only supports the default ``<campaign_root>/store`` layout).
"""

import json
import os
from pathlib import Path
import tempfile
import unittest

ORIGINAL = bool(os.environ.get("EVO_PARITY_ORIGINAL"))
if ORIGINAL:
    import island_group as islands
    from geak_evo.research_state import ResearchStore
else:
    from geak_evo.archive import islands
    from geak_evo.archive.store import ResearchStore

JOB = "1" * 64
NOTE = {"claim": "fixture", "scope": "synthetic job", "next_test": "inspect"}


def claim_context(view, *, binding=None):
    """Stand-in for the planner's claim projection: deterministic and plain."""
    return {"claim_head": view.get("head"), "binding": binding, "claims": []}


def tree(root):
    """Every file under root with its bytes and mode."""
    root = Path(root)
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode)
            for path in sorted(root.rglob("*")) if path.is_file()}


class Member:
    """One campaign: a store, a seed, and optionally one reviewed engineer turn."""

    def __init__(self, campaign_root, store_root, epoch, seed_text):
        self.root, self.epoch = Path(campaign_root), epoch
        self.root.mkdir(parents=True, exist_ok=True)
        self.work = self.root.parent / (self.root.name + "-work")
        self.work.mkdir()
        self.store = ResearchStore(store_root, allowed_epochs=[(JOB, epoch)],
                                   source_paths={JOB: ["kernel.py"]})
        self.seed = self.candidate(seed_text)
        self.history, self.opportunities = [], []

    def candidate(self, text, parents=()):
        (self.work / "kernel.py").write_text(text)
        source = self.store.capture_source(self.work, ["kernel.py"])
        return self.store.candidate(JOB, self.epoch, source, parents=parents)

    def reviewed_turn(self, text):
        """Direction D moves seed → child; the turn is settled and reviewed."""
        self.store.checkpoint("D", self.seed, note=NOTE, expected_revision=0)
        child = self.candidate(text, parents=[self.seed])
        source_id = self.store.get_candidate(child)["source_id"]
        transport = self.store.archive_evidence(child, {"transport.json": b'{"turn": 1}'})
        collection = self.store.archive_evidence(child, {
            "collection.json": json.dumps({"costs": {"model_usd": 1}}).encode(),
            "native_source_roles.json": json.dumps({"selected": child}).encode(),
            "native_idea_harvest.json": json.dumps({"selected": {"source_id": source_id}}).encode()})
        transport_ref = {"evidence_id": transport, "artifact": "transport.json"}
        collection_ref = {"evidence_id": collection, "artifact": "collection.json"}
        self.store.checkpoint("D", child, expected_revision=1, note=dict(
            NOTE, research_status="reported", research_review_id="review-1",
            reviewed_outcome_refs=[transport_ref, collection_ref],
            research_update_ref=collection_ref, latest_transport_ref=transport_ref))
        self.opportunities.append({"opportunity_id": "turn-1", "direction_id": "D"})
        self.history += [
            {"attempt_id": "turn-1", "settled": True, "candidate_id": child,
             "transport_ref": transport_ref, "collection_ref": collection_ref},
            {"review_kind": "research_review", "direction_revision": 2,
             "proposal": {"direction_id": "D", "expected_revision": 1, "note": {},
                          "transport_ref": transport_ref, "collection_ref": collection_ref}}]
        return child

    def commit_campaign(self):
        state = {"binding": {"job_id": JOB, "measurement_epoch_id": self.epoch},
                 "opportunities": self.opportunities, "history": self.history, "claim_head": None}
        self.store.checkpoint("campaign", self.seed, note=dict(NOTE, evolution=state), expected_revision=0)
        return {"campaign_id": "campaign", "state": state}


class IslandImportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.registry = self.base / "registry"
        self.registry.mkdir()
        if ORIGINAL:
            islands.current_claim_context = claim_context

    def group(self, member, stores=None):
        policy = {"island_group": {"group_id": "g", "registry_root": str(self.registry),
                                   "members": [str(self.base / "A"), str(self.base / "B")]}}
        if ORIGINAL:
            return islands.group_for_policy(policy, member.root, JOB)
        options = {"claim_context": claim_context}
        if stores is not None:
            options["store_root"] = lambda root: stores[Path(root)]
        return islands.group_for_policy(policy, member.root, JOB, **options)

    def members(self, store_root):
        a = Member(self.base / "A", store_root("A"), "epoch-a", "value = 'A seed'\n")
        b = Member(self.base / "B", store_root("B"), "epoch-b", "value = 'B seed'\n")
        self.child = b.reviewed_turn("value = 'B improved'\n")
        return a, b, a.commit_campaign(), b.commit_campaign()

    def check_idempotent_import(self, stores=None):
        a, b, snapshot_a, snapshot_b = self.members(
            (lambda name: self.base / name / "store") if stores is None else (lambda name: stores[name]))
        paths = None if stores is None else {self.base / name: path for name, path in stores.items()}
        group_a, group_b = self.group(a, paths), self.group(b, paths)
        self.assertEqual(group_b.sync(b.store, snapshot_b), {})  # Publishes; nothing to import yet.
        self.assertEqual(len(os.listdir(self.registry)), 1)
        self.assertEqual(group_a.lookup(a.store, "epoch-a", a.seed), {})

        first = group_a.sync(a.store, snapshot_a)
        self.assertEqual(len(first), 1)
        [(immigrant, records)] = first.items()
        local = a.store.get_candidate(immigrant)
        self.assertEqual((local["parents"], local["measurement_epoch_id"]), ([], "epoch-a"))
        self.assertEqual(local["source_id"], b.store.get_candidate(self.child)["source_id"])
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["is_latest_imported_for_origin"])
        self.assertEqual(set(records[0]["origin_evidence_map"]),
                         {row["evidence_id"] for row in (snapshot_b["state"]["history"][0]["transport_ref"],
                                                         snapshot_b["state"]["history"][0]["collection_ref"])})

        before = tree(a.store.root), tree(a.root), tree(self.registry)
        self.assertEqual(group_a.sync(a.store, snapshot_a), first)
        self.assertEqual(group_a.lookup(a.store, "epoch-a", a.seed), first)
        self.assertEqual((tree(a.store.root), tree(a.root), tree(self.registry)), before)
        self.assertFalse((a.root / "island-rejections").exists())
        return a, group_a, snapshot_a

    def test_importing_the_same_sibling_twice_changes_nothing(self):
        self.check_idempotent_import()

    @unittest.skipIf(ORIGINAL, "the original pins each member store to <campaign_root>/store")
    def test_member_stores_need_not_live_under_the_campaign_root(self):
        stores = {"A": self.base / "stores" / "a", "B": self.base / "stores" / "b"}
        (self.base / "stores").mkdir()
        self.check_idempotent_import(stores)
        self.assertFalse((self.base / "B" / "store").exists())

    def test_publication_that_differs_from_its_origin_journal_is_rejected(self):
        a, b, snapshot_a, snapshot_b = self.members(lambda name: self.base / name / "store")
        self.group(b).sync(b.store, snapshot_b)
        [name] = os.listdir(self.registry)
        doc = json.loads((self.registry / name).read_bytes())
        doc["outcomes"][0]["review_note"]["claim"] = "forged"
        raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
        forged = self.registry / islands._digest(raw)
        forged.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "rejected"):
            self.group(a).sync(a.store, snapshot_a)
        self.assertEqual(len(os.listdir(a.root / "island-rejections")), 1)
        forged.unlink()
        # The genuine publication may already be imported; the forged one never is.
        imported = self.group(a).sync(a.store, snapshot_a)
        self.assertEqual([len(rows) for rows in imported.values()], [1])
        self.assertFalse(any(b'"forged"' in data for data, _ in tree(a.store.root / "objects").values()))


if __name__ == "__main__":
    unittest.main()
