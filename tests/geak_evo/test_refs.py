"""Tests for geak_evo.archive.refs (file and source references).

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_refs.py
Parity run on the original engineer module: set EVO_PARITY_ORIGINAL=1 and put
the original ``evolution/src`` on PYTHONPATH instead.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("EVO_PARITY_ORIGINAL"):
    from geak_evo import engineer as refs
    from geak_evo import research_state as store_module
else:
    from geak_evo.archive import refs
    from geak_evo.archive import store as store_module


class FileRefTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.path = self.base / "note.txt"
        self.path.write_bytes(b"hello\n")

    def test_round_trip_and_change_detection(self):
        ref = refs.file_ref(self.path)
        self.assertEqual(ref, {"path": str(self.path), "sha256": hashlib.sha256(b"hello\n").hexdigest()})
        self.assertEqual(refs._read_ref(ref), b"hello\n")
        self.path.write_bytes(b"changed\n")
        with self.assertRaises(refs.EngineerError):
            refs._read_ref(ref)

    def test_errors_are_value_errors(self):
        self.assertTrue(issubclass(refs.EngineerError, ValueError))

    def test_rejects_non_canonical_paths_and_bad_shapes(self):
        (self.base / "link.txt").symlink_to(self.path)
        (self.base / "dir").mkdir()
        for path in ("note.txt", self.base / "link.txt", self.base / "dir", self.base / "dir/../note.txt"):
            with self.subTest(path=path), self.assertRaises(refs.EngineerError):
                refs.file_ref(path)
        with self.assertRaises(FileNotFoundError):
            refs.file_ref(self.base / "missing.txt")
        good = refs.file_ref(self.path)
        bad_refs = [None, [], {"path": good["path"]}, dict(good, extra=1),
                    dict(good, sha256=good["sha256"].upper()), dict(good, sha256="0" * 63),
                    dict(good, path="note.txt")]
        for ref in bad_refs:
            with self.subTest(ref=ref), self.assertRaises(refs.EngineerError):
                refs._read_ref(ref)


class SourceRefTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name).resolve()
        work = base / "work"
        work.mkdir()
        (work / "k.py").write_text("x = 1\n")
        self.store = store_module.ResearchStore(base / "archive", allowed_epochs=[("job", "e")],
                                                source_paths={"job": ["k.py", "absent.py"]})
        self.source_id = self.store.capture_source(work, ["k.py", "absent.py"])
        self.candidate_id = self.store.candidate("job", "e", self.source_id)

    def test_source_reference_round_trip(self):
        ref = refs.source_reference(self.store, self.candidate_id)
        self.assertEqual(ref["candidate_id"], self.candidate_id)
        self.assertEqual(ref["source_id"], self.source_id)
        self.assertEqual(ref["manifest"]["sha256"], self.source_id)
        self.assertEqual(refs._source_ref(ref), self.store.get_candidate(self.candidate_id))

    def test_source_ref_detects_tampering_and_mismatch(self):
        ref = refs.source_reference(self.store, self.candidate_id)
        for broken in (dict(ref, extra=1), dict(ref, candidate_id="x"), dict(ref, source_id=self.candidate_id)):
            with self.subTest(broken=sorted(broken)), self.assertRaises(refs.EngineerError):
                refs._source_ref(broken)
        blob = self.store.source(self.source_id)["files"][1]["blob_id"]
        path = self.store.root / "objects" / blob
        path.chmod(0o644)
        path.write_bytes(b"tampered")
        with self.assertRaises(refs.EngineerError):
            refs._source_ref(ref)

    def test_candidate_must_match_manifest(self):
        other_manifest = self.store._put(json.dumps({"kind": "source", "files": []}).encode())
        ref = refs.source_reference(self.store, self.candidate_id)
        ref = dict(ref, source_id=other_manifest,
                   manifest=refs.file_ref(self.store.root / "objects" / other_manifest))
        with self.assertRaises(refs.EngineerError):
            refs._source_ref(ref)


@unittest.skipIf(os.environ.get("EVO_PARITY_ORIGINAL"), "port-only names")
class PortNameTests(unittest.TestCase):
    def test_public_read_ref_and_old_alias(self):
        self.assertIs(refs._read_ref, refs.read_ref)


if __name__ == "__main__":
    unittest.main()
