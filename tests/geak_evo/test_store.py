"""Tests for geak_evo.archive.store (ported from tests/test_research_state.py).

CPU fixtures exercise storage rules, not evaluator authenticity or GPU safety.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_store.py
Parity run on the original module: set EVO_PARITY_ORIGINAL=1 and put the
original ``evolution/src`` on PYTHONPATH instead.
The round-trip tests need GEAK_EVO_ORIGINAL_SRC=<original evolution/src>;
they are skipped without it.
"""

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

if os.environ.get("EVO_PARITY_ORIGINAL"):
    from geak_evo import research_state as rs
else:
    from geak_evo.archive import store as rs

PORT_ROOT = str(Path(__file__).resolve().parents[2])


class ResearchStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.workspace = self.base / "work"
        (self.workspace / "pkg").mkdir(parents=True)
        (self.workspace / "main.py").write_text("from pkg.helper import value\n")
        (self.workspace / "pkg/helper.py").write_text("value = 1\n")
        self.paths = ["main.py", "optional.py", "pkg/helper.py"]
        self.options = dict(allowed_epochs=[("job", "epoch-1"), ("job", "epoch-2")],
                            source_paths={"job": self.paths}, verify_evidence=self.verifier)
        self.store = rs.ResearchStore(self.base / "archive", **self.options)
        self.source_id = self.store.capture_source(self.workspace, self.paths)
        self.a = self.store.candidate("job", "epoch-1", self.source_id)
        self.note = {"claim": "Implementation incomplete", "scope": "declared fixture cases",
                     "next_test": "Compare the slow faithful alternative with the anchor"}

    @staticmethod
    def verifier(artifacts, expected):
        receipt = json.loads(artifacts["validation.json"])
        if receipt["binding"] != expected:
            raise rs.StateError("receipt binding mismatch")
        return receipt["checks"]

    def evidence(self, candidate=None, **changes):
        candidate = candidate or self.a
        bound = self.store.get_candidate(candidate)
        checks = dict(independent=True, settled=True, correct=True, eligible=True)
        checks.update(changes)
        payload = {"binding": self.store._binding(bound), "checks": checks}
        return self.store.archive_evidence(candidate, {"validation.json": json.dumps(payload).encode()})

    def checkpoint(self, candidate=None, revision=0, **changes):
        return self.store.checkpoint("direction", candidate or self.a, note=self.note,
                                     expected_revision=revision, **changes)

    def modified_candidate(self, value=2, **kwargs):
        (self.workspace / "pkg/helper.py").write_text("value = %s\n" % value)
        source = self.store.capture_source(self.workspace, self.paths)
        return self.store.candidate("job", "epoch-1", source, **kwargs)

    def test_complete_declared_closure_identity_and_round_trip(self):
        self.assertEqual(self.source_id, self.store.capture_source(self.workspace, reversed(self.paths)))
        (self.workspace / "main.py").chmod(0o751)
        executable = self.store.capture_source(self.workspace, self.paths)
        self.assertNotEqual(self.source_id, executable)
        destination = self.base / "restored"
        self.store.materialize(executable, destination)
        self.assertEqual((destination / "main.py").stat().st_mode & 0o111, 0o111)
        self.assertEqual((destination / "pkg/helper.py").read_bytes(), b"value = 1\n")
        self.assertFalse((destination / "optional.py").exists())
        (self.workspace / "optional.py").write_bytes(b"")
        present = self.store.capture_source(self.workspace, self.paths)
        self.assertNotEqual(executable, present)
        self.assertNotEqual(self.source_id, self.store.get_candidate(self.modified_candidate())["source_id"])

    def test_unsafe_paths_symlinks_and_nonregular_files(self):
        for name in ("/etc/passwd", "../main.py", "./main.py", "pkg//helper.py", "pkg/../main.py", "."):
            with self.subTest(name=name), self.assertRaises(rs.StateError):
                self.store.capture_source(self.workspace, [name])
        (self.workspace / "link").symlink_to(self.workspace / "pkg", target_is_directory=True)
        (self.workspace / "linked.py").symlink_to(self.workspace / "main.py")
        os.mkfifo(self.workspace / "pipe")
        for name in ("link/helper.py", "linked.py", "pipe"):
            with self.subTest(name=name), self.assertRaises((rs.StateError, OSError)):
                self.store.capture_source(self.workspace, [name])

    def test_capture_detects_change_across_files(self):
        original = rs._read
        reads = []
        def changing(root, name):
            if Path(root) == self.workspace:
                reads.append(name)
                if len(reads) == len(self.paths) + 1:
                    (self.workspace / "pkg/helper.py").write_text("value = 99\n")
            return original(root, name)
        with patch.object(rs, "_read", side_effect=changing), self.assertRaises(rs.StateError):
            self.store.capture_source(self.workspace, self.paths)

    def test_missing_workspace_is_not_an_absent_source_closure(self):
        with self.assertRaises(FileNotFoundError):
            self.store.capture_source(self.base / "missing", self.paths)

    def test_source_allowlist_and_unknown_epochs_fail_closed(self):
        partial = self.store.capture_source(self.workspace, ["main.py"])
        with self.assertRaises(rs.StateError):
            self.store.candidate("job", "epoch-1", partial)
        with self.assertRaises(rs.StateError):
            self.store.candidate("job", "unknown", self.source_id)

    def test_new_helper_registration_preserves_old_anchor_and_resumed_tip(self):
        self.checkpoint()
        old_evidence = self.evidence()
        self.store.promote("direction", self.a, old_evidence, expected_revision=1)
        (self.workspace / "new_helper.py").write_text("unfinished = True\n")
        (self.workspace / "main.py").write_text("import new_helper\n")
        expanded = self.paths + ["new_helper.py"]
        source = self.store.capture_source(self.workspace, expanded)
        with self.assertRaises(rs.StateError):
            self.store.candidate("job", "epoch-1", source, parents=[self.a])
        registration = self.store.register_source_paths("job", expanded)
        self.assertEqual(registration, self.store.register_source_paths("job", reversed(expanded)))
        child = self.store.candidate("job", "epoch-1", source, parents=[self.a])
        self.checkpoint(child, revision=2, status="unfinished")
        reopened = rs.ResearchStore(self.store.root, **dict(self.options, source_paths={"job": expanded}))
        state = reopened.get("direction")
        self.assertEqual(state["anchor"]["candidate_id"], self.a)
        self.assertEqual(state["tip"], child)
        self.assertEqual(reopened.get_candidate(child)["parents"], [self.a])
        reopened.promote("direction", self.a, old_evidence, expected_revision=3)
        restored_anchor, restored_tip = self.base / "old-layout", self.base / "new-layout"
        reopened.materialize(self.source_id, restored_anchor)
        reopened.materialize(source, restored_tip)
        self.assertFalse((restored_anchor / "new_helper.py").exists())
        self.assertEqual((restored_tip / "new_helper.py").read_text(), "unfinished = True\n")
        resumed_source = reopened.capture_source(restored_tip, expanded)
        self.assertEqual(resumed_source, source)
        empty_config = rs.ResearchStore(self.store.root, **dict(self.options, source_paths={}))
        self.assertEqual(empty_config.get("direction")["tip"], child)
        # Archiving an arbitrary alternative does not authorize that closure.
        subset = reopened.capture_source(restored_tip, ["main.py"])
        with self.assertRaises(rs.StateError):
            reopened.candidate("job", "epoch-1", subset, parents=[child])

    def test_registered_source_closure_tampering_fails_closed(self):
        registration = self.store.register_source_paths("job", self.paths)
        path = self.store.root / "closures" / registration
        path.chmod(0o644)
        path.write_bytes(b"tampered host registration")
        with self.assertRaises(rs.StateError):
            self.store.get_candidate(self.a)
        with self.assertRaises(rs.StateError):
            self.store.register_source_paths("job", self.paths)

    def test_legacy_closures_require_explicit_host_bootstrap(self):
        self.checkpoint()
        registration = self.store.register_source_paths("job", self.paths)
        (self.store.root / "closures" / registration).unlink()
        # Simulate history retained before the closure registry existed.
        reopened = rs.ResearchStore(self.store.root, **dict(
            self.options, source_paths={"job": self.paths + ["new_helper.py"]}))
        with self.assertRaises(rs.StateError):
            reopened.get("direction")
        reopened.register_source_paths("job", self.paths)
        self.assertEqual(reopened.get("direction")["tip"], self.a)

    def test_missing_parent_and_later_deleted_ancestor_are_detected(self):
        with self.assertRaises(rs.StateError):
            self.modified_candidate(parents=["0" * 64])
        child = self.modified_candidate(parents=[self.a])
        self.checkpoint(child)
        (self.store.root / "objects" / self.a).unlink()
        with self.assertRaises(rs.StateError):
            self.store.get("direction")

    def test_genealogy_does_not_infer_contribution(self):
        donor = self.modified_candidate()
        child = self.modified_candidate(3, parents=[self.a], donors=[donor])
        self.assertEqual(self.store.get_candidate(child)["contribution_claims"], [])
        evidence = self.evidence(donor)
        claim = {"source_candidate_id": donor, "claim": "Retains helper layout",
                 "scope": "Source inspection claim, benefit untested", "evidence": [evidence]}
        claimed = self.modified_candidate(3, parents=[self.a], donors=[donor], contribution_claims=[claim])
        self.assertNotEqual(child, claimed)
        self.assertEqual(self.store.get_candidate(claimed)["contribution_claims"], [claim])
        with self.assertRaises(rs.StateError):
            self.modified_candidate(3, contribution_claims=[claim])
        with self.assertRaises(rs.StateError):
            self.modified_candidate(3, contribution_claims=(entry for entry in [claim]))
        (self.store.root / "objects" / evidence).unlink()
        with self.assertRaises(rs.StateError):
            self.store.get_candidate(claimed)

    def test_parent_donor_dag_is_not_rewalked_exponentially(self):
        previous, current = self.a, self.modified_candidate(2)
        for _ in range(28):
            previous, current = current, self.store.candidate(
                "job", "epoch-1", self.source_id, parents=[current], donors=[previous])
        with patch.object(self.store, "_object", wraps=self.store._object) as reads:
            self.store.get_candidate(current)
            self.assertLess(reads.call_count, 100)

    def test_earlier_validation_preserves_failing_tip_and_scoped_note(self):
        self.checkpoint(resources={"model_tokens": 100})
        raw = self.evidence()
        self.store.promote("direction", self.a, raw, expected_revision=1)
        failing = self.modified_candidate(parents=[self.a])
        self.checkpoint(failing, revision=2, status="unfinished", resources={"model_tokens": 130})
        self.store.promote("direction", self.a, raw, expected_revision=3)
        reopened = rs.ResearchStore(self.store.root, **self.options).get("direction")
        self.assertEqual(reopened["anchor"]["candidate_id"], self.a)
        self.assertEqual(reopened["tip"], failing)
        self.assertEqual(reopened["status"], "unfinished")
        self.assertEqual(reopened["note"], self.note)
        self.assertEqual(reopened["resources"], {"model_tokens": 130})
        self.assertIn("updated_at", reopened)

    def test_agent_score_and_failed_checks_never_qualify_anchor(self):
        self.note["claimed_speedup"] = 999
        self.checkpoint()
        self.assertIsNone(self.store.get("direction")["anchor"])
        for key in ("independent", "settled", "correct", "eligible"):
            raw = self.evidence(**{key: False})
            with self.subTest(key=key), self.assertRaises(rs.StateError):
                self.store.promote("direction", self.a, raw, expected_revision=1)
            self.assertIsNone(self.store.get("direction")["anchor"])
            self.assertTrue(self.store.evidence(raw)[1])
        self.store.verify_evidence = None
        with self.assertRaises(rs.StateError):
            self.store.promote("direction", self.a, self.evidence(), expected_revision=1)

    def test_mutable_evidence_wrong_source_and_epoch_rejected(self):
        with self.assertRaises(rs.StateError):
            self.store.archive_evidence(self.a, {"receipt": str(self.workspace / "main.py")})
        other = self.modified_candidate()
        self.checkpoint(other)
        with self.assertRaises(rs.StateError):
            self.store.promote("direction", other, self.evidence(), expected_revision=1)
        newer = self.store.candidate("job", "epoch-2", self.source_id, parents=[self.a])
        with self.assertRaises(rs.StateError):
            self.checkpoint(newer, revision=1)
        self.store.checkpoint("new-epoch", newer, note=self.note, expected_revision=0)
        with self.assertRaises(rs.StateError):
            self.store.promote("new-epoch", self.a, self.evidence(), expected_revision=1)
        with self.assertRaises(rs.StateError):
            self.store.checkpoint("new-epoch", newer, note=self.note, expected_revision=1,
                                  evidence=[self.evidence()])
        self.assertEqual(self.store.get("new-epoch")["tip"], newer)

    def test_changed_raw_evidence_or_source_cannot_be_used(self):
        self.checkpoint()
        evidence = self.evidence()
        metadata, _ = self.store.evidence(evidence)
        raw = self.store.root / "objects" / metadata["artifacts"]["validation.json"]
        raw.chmod(0o644)
        raw.write_bytes(b"tampered")
        with self.assertRaises(rs.StateError):
            self.store.promote("direction", self.a, evidence, expected_revision=1)
        self.assertIsNone(self.store.get("direction")["anchor"])
        source = self.store.source(self.source_id)
        blob = self.store.root / "objects" / source["files"][0]["blob_id"]
        blob.chmod(0o644)
        blob.write_bytes(b"tampered")
        destination = self.base / "must-not-appear"
        with self.assertRaises(rs.StateError):
            self.store.materialize(self.source_id, destination)
        self.assertFalse(destination.exists())

    def test_stale_writer_and_resource_decrease_rejected(self):
        self.checkpoint(resources={"seconds": 5})
        second_writer = rs.ResearchStore(self.store.root, **self.options)
        with self.assertRaises(rs.StateError):
            second_writer.checkpoint("direction", self.a, note=self.note, expected_revision=0)
        for usage in ({"seconds": 4}, {}, {"seconds": float("nan")}, {"seconds": True}):
            with self.subTest(usage=usage), self.assertRaises(rs.StateError):
                self.checkpoint(revision=1, resources=usage)

    def test_interrupt_before_journal_commit_keeps_previous_state(self):
        self.checkpoint()
        original = rs._write
        def interrupted(path, data, immutable=False):
            if path.parent.name == "records":
                raise OSError("injected interruption before commit")
            return original(path, data, immutable)
        with patch.object(rs, "_write", side_effect=interrupted), self.assertRaises(OSError):
            self.checkpoint(revision=1, status="shelved")
        self.assertEqual(self.store.get("direction")["revision"], 1)

    def test_interruption_after_journal_commit_recovers_complete_next_state(self):
        self.checkpoint()
        evidence = self.evidence()
        original = rs._write
        def interrupted(path, data, immutable=False):
            if path.name == "index.json":
                raise OSError("injected interruption before cache replacement")
            return original(path, data, immutable)
        with patch.object(rs, "_write", side_effect=interrupted), self.assertRaises(OSError):
            self.store.promote("direction", self.a, evidence, expected_revision=1)
        reopened = rs.ResearchStore(self.store.root, **self.options).get("direction")
        self.assertEqual(reopened["revision"], 2)
        self.assertEqual(reopened["anchor"]["evidence_id"], evidence)
        self.assertEqual(reopened["tip"], self.a)

    def test_torn_temporary_files_ignored_corrupt_committed_records_rejected(self):
        self.checkpoint()
        (self.store.root / "records/.pending-interrupted").write_bytes(b'{"incomplete":')
        (self.store.root / "index.json").write_bytes(b"torn index")
        self.assertEqual(self.store.get("direction")["revision"], 1)
        committed = next((self.store.root / "records").glob("*.json"))
        committed.chmod(0o644)
        committed.write_bytes(committed.read_bytes() + b" ")
        with self.assertRaises(rs.StateError):
            self.store.get("direction")

    def test_index_replace_failure_does_not_discard_journal_or_leave_partial_index(self):
        self.checkpoint()
        old = (self.store.root / "index.json").read_bytes()
        with patch.object(rs.os, "replace", side_effect=OSError("interrupted replace")):
            with self.assertRaises(OSError):
                self.checkpoint(revision=1, status="revived")
        self.assertEqual((self.store.root / "index.json").read_bytes(), old)
        self.assertEqual(self.store.get("direction")["status"], "revived")

    def test_deleted_tail_is_detected_when_index_records_later_commit(self):
        self.checkpoint()
        self.checkpoint(revision=1, status="shelved")
        sorted((self.store.root / "records").glob("*.json"))[-1].unlink()
        with self.assertRaises(rs.StateError):
            self.store.get("direction")

    def test_correction_and_revival_preserve_old_claim_and_evidence(self):
        self.checkpoint(status="shelved", evidence=[self.evidence()])
        original_record = next((self.store.root / "records").glob("*.json")).read_bytes()
        corrected = dict(self.note, claim="The implementation failed; the algorithm remains untested",
                         supersedes_revision=1)
        self.store.checkpoint("direction", self.a, note=corrected, expected_revision=1, status="revived")
        self.assertEqual(self.store.get("direction")["note"], corrected)
        self.assertEqual(sorted((self.store.root / "records").glob("*.json"))[0].read_bytes(), original_record)



# -- drift test 9: one writer at a time across processes ----------------------

NOTE = {"claim": "c", "scope": "s", "next_test": "t"}


def _open_store(root, workspace):
    paths = ["main.py"]
    store = rs.ResearchStore(root, allowed_epochs=[("job", "epoch-1")], source_paths={"job": paths})
    return store, store.candidate("job", "epoch-1", store.capture_source(workspace, paths))


def _increment_worker(root, workspace, count, results):
    store, candidate = _open_store(root, workspace)
    done = stale = 0
    while done < count:
        current = store.get().get("direction")
        revision = current["revision"] if current else 0
        try:
            store.checkpoint("direction", candidate, note=NOTE, expected_revision=revision,
                             resources={"steps": revision + 1})
            done += 1
        except rs.StateError:
            stale += 1
    results.put((done, stale))


def _hold_lock_worker(root, workspace, held, release):
    store, _ = _open_store(root, workspace)
    with store._lock():
        held.set()
        release.wait(30)


def _blocked_writer_worker(root, workspace, finished):
    store, candidate = _open_store(root, workspace)
    store.checkpoint("direction", candidate, note=NOTE, expected_revision=0)
    finished.set()


class SingleWriterLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.workspace = base / "work"
        self.workspace.mkdir()
        (self.workspace / "main.py").write_text("x = 1\n")
        self.root = base / "archive"
        _open_store(self.root, self.workspace)
        self.context = multiprocessing.get_context("fork")

    def test_contending_writers_never_lose_or_fork_a_revision(self):
        results = self.context.Queue()
        workers = [self.context.Process(target=_increment_worker,
                                        args=(self.root, self.workspace, 15, results)) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(90)
            self.assertEqual(worker.exitcode, 0)
        outcomes = [results.get(timeout=5) for _ in workers]
        self.assertEqual(sum(done for done, _ in outcomes), 30)
        store, _ = _open_store(self.root, self.workspace)
        state = store.get("direction")
        self.assertEqual(state["revision"], 30)
        self.assertEqual(state["resources"], {"steps": 30})
        records = sorted((self.root / "records").glob("*.json"))
        self.assertEqual([path.name[:8] for path in records], ["%08d" % n for n in range(1, 31)])

    def test_writer_waits_while_another_process_holds_the_lock(self):
        held, release, finished = (self.context.Event() for _ in range(3))
        holder = self.context.Process(target=_hold_lock_worker, args=(self.root, self.workspace, held, release))
        holder.start()
        self.assertTrue(held.wait(30))
        writer = self.context.Process(target=_blocked_writer_worker, args=(self.root, self.workspace, finished))
        writer.start()
        try:
            self.assertFalse(finished.wait(1.0))
            self.assertEqual(list((self.root / "records").glob("*.json")), [])
        finally:
            release.set()
        self.assertTrue(finished.wait(30))
        holder.join(30)
        writer.join(30)
        self.assertEqual((holder.exitcode, writer.exitcode), (0, 0))


# -- saved-record compatibility: original <-> port -----------------------------

_WRITE_AND_READ = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    if os.environ.get("EVO_IMPL") == "original":
        from geak_evo import research_state as rs
    else:
        from geak_evo.archive import store as rs
    root, action = Path(sys.argv[1]), sys.argv[2]
    paths = ["main.py", "optional.py", "pkg/helper.py"]
    def verifier(artifacts, expected):
        return json.loads(artifacts["validation.json"])["checks"]
    options = dict(allowed_epochs=[("job", "epoch-1")], source_paths={"job": paths}, verify_evidence=verifier)
    store = rs.ResearchStore(root / "archive", **options)
    ids = {}
    if action == "write":
        work = root / "work"
        (work / "pkg").mkdir(parents=True)
        (work / "main.py").write_text("from pkg.helper import value\\n")
        (work / "main.py").chmod(0o755)
        (work / "pkg/helper.py").write_text("value = 1\\n")
        source = store.capture_source(work, paths)
        a = store.candidate("job", "epoch-1", source)
        (work / "pkg/helper.py").write_text("value = 2\\n")
        b_source = store.capture_source(work, paths)
        donor = store.candidate("job", "epoch-1", b_source, parents=[a])
        checks = dict(independent=True, settled=True, correct=True, eligible=True)
        payload = json.dumps({"checks": checks}, sort_keys=True).encode()
        evidence = store.archive_evidence(a, {"validation.json": payload, "raw/trace.txt": b"t\\n"})
        claim = {"source_candidate_id": donor, "claim": "keeps helper", "scope": "fixture", "evidence": []}
        (work / "pkg/helper.py").write_text("value = 3\\n")
        child = store.candidate("job", "epoch-1", store.capture_source(work, paths), parents=[a],
                                donors=[donor], contribution_claims=[claim])
        note = {"claim": "c", "scope": "s", "next_test": "t"}
        store.checkpoint("d1", a, note=note, expected_revision=0, resources={"tokens": 5})
        store.promote("d1", a, evidence, expected_revision=1)
        store.checkpoint("d1", child, note=dict(note, claim="c2"), expected_revision=2,
                         status="unfinished", resources={"tokens": 9.5}, evidence=[evidence])
        store.checkpoint("d2", donor, note=note, expected_revision=0)
        ids = dict(source=source, a=a, donor=donor, child=child, evidence=evidence)
        (root / "ids.json").write_text(json.dumps(ids))
    ids = json.loads((root / "ids.json").read_text())
    def strip(state):
        return {k: v for k, v in state.items() if k != "updated_at"}
    out = {
        "ids": ids,
        "directions": store.get(),
        "d1_rev1": store.get_revision("d1", 1),
        "d1_rev2": store.get_revision("d1", 2),
        "current_d2": store.read_current("d2"),
        "candidates": {k: store.get_candidate(ids[k]) for k in ("a", "donor", "child")},
        "source": store.source(ids["source"]),
        "evidence": [store.evidence(ids["evidence"])[0],
                     {k: v.decode() for k, v in store.evidence(ids["evidence"])[1].items()}],
        "artifact": store.evidence_artifact(ids["evidence"], "raw/trace.txt")[1].decode(),
        "objects": sorted(p.name for p in (root / "archive" / "objects").iterdir()),
        "closures": sorted(p.name for p in (root / "archive" / "closures").iterdir()),
        "records_state": [strip(json.loads(p.read_bytes())["state"])
                          for p in sorted((root / "archive" / "records").glob("*.json"))],
    }
    print(json.dumps(out, sort_keys=True))
''')


@unittest.skipUnless(os.environ.get("GEAK_EVO_ORIGINAL_SRC"), "GEAK_EVO_ORIGINAL_SRC not set")
class SavedRecordCompatibilityTests(unittest.TestCase):
    """A store written by the original research_state reads identically through the port, and back."""

    def run_impl(self, impl, root, action):
        env = dict(os.environ, EVO_IMPL=impl, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.environ["GEAK_EVO_ORIGINAL_SRC"] if impl == "original" else PORT_ROOT)
        env.pop("EVO_PARITY_ORIGINAL", None)
        done = subprocess.run([sys.executable, "-B", "-c", _WRITE_AND_READ, str(root), action],
                              cwd=str(root), env=env, capture_output=True, timeout=100)
        self.assertEqual(done.returncode, 0, done.stderr.decode()[-2000:])
        return json.loads(done.stdout)

    def round_trip(self, writer, reader):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            written = self.run_impl(writer, root, "write")
            read_back = self.run_impl(reader, root, "read")
            self.assertEqual(read_back, written)
            # The writer can still read what the reader re-indexed.
            self.assertEqual(self.run_impl(writer, root, "read"), written)
            return written

    def test_original_store_reads_identically_through_port(self):
        self.round_trip("original", "port")

    def test_port_store_reads_identically_through_original(self):
        self.round_trip("port", "original")

    def test_same_inputs_give_same_content_identities_and_records(self):
        original = self.round_trip("original", "original")
        port = self.round_trip("port", "port")
        for key in ("ids", "objects", "closures", "records_state", "candidates", "source", "evidence"):
            self.assertEqual(port[key], original[key], key)


if __name__ == "__main__":
    unittest.main()
