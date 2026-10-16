"""Population seal custody and metric-source pins.

A seal is written once into a folder only its owner can write, as a single-link
0400 file. A seal verifies against the exact metric source it registered: by
default this checkout's ``archive/cells.py``; a seal pinning another version
needs that version admitted by the host (exact ref + its bytes) and otherwise
fails closed.

The real pre-port bytes test needs GEAK_EVO_ORIGINAL_SRC=<original evolution/src>
(its sibling ``bridge/native_cells.py`` is read) and is skipped without it.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_selection_population.py
"""

import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from geak_evo.selection import population  # noqa: E402
from geak_evo.selection.population import SelectorError, check_metric_source  # noqa: E402


class Scratch(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="geak-population-")).resolve()

    def tearDown(self):
        for path in [self.root, *self.root.rglob("*")]:
            if path.is_dir():
                path.chmod(0o700)
            else:
                path.chmod(0o600)
        shutil.rmtree(self.root)


class SealCustodyTest(Scratch):
    def test_seal_is_single_owner_readonly_file(self):
        ref = population._seal_once(self.root / "seal.json", {"status": "missing"})
        saved = (self.root / "seal.json").lstat()
        self.assertEqual(stat.S_IMODE(saved.st_mode), 0o400)
        self.assertEqual((saved.st_nlink, saved.st_uid), (1, os.geteuid()))
        self.assertEqual(ref["sha256"], population.digest(population.encoded({"status": "missing"})))

    def test_group_writable_folder_is_refused(self):
        shared = self.root / "shared"
        shared.mkdir()
        shared.chmod(0o770)
        with self.assertRaisesRegex(SelectorError, "seal_parent_must_be_trusted_directory"):
            population._seal_once(shared / "seal.json", {"status": "missing"})
        self.assertFalse((shared / "seal.json").exists())

    def test_existing_seal_is_never_replaced(self):
        population._seal_once(self.root / "seal.json", {"status": "missing"})
        with self.assertRaises(FileExistsError):
            population._seal_once(self.root / "seal.json", {"status": "selected"})


class MetricSourceTest(Scratch):
    def old_version(self, name, raw):
        """An old cells file now stored at ``name``, pinned at a different absolute path."""
        path = self.root / name
        path.write_bytes(raw)
        return path, {"path": f"/old/{name}/native_cells.py", "sha256": population.digest(raw)}

    def test_current_source_passes(self):
        seal = {"metric_source_ref": population.metric_source_ref()}
        self.assertEqual(check_metric_source(seal), population.metric_source_ref())
        self.assertTrue(seal["metric_source_ref"]["path"].endswith("/geak_evo/archive/cells.py"))

    def test_admitted_old_versions_pass_at_any_pinned_path(self):
        raw = b"def cell(): pass\n"
        for name in ("study-a", "study-b"):
            with self.subTest(name=name):
                path, pin = self.old_version(name, raw)
                admitted = {"ref": pin, "path": str(path)}
                self.assertEqual(check_metric_source({"metric_source_ref": pin}, registered_source=admitted), pin)

    def test_old_version_not_admitted_or_unreadable_fails_closed(self):
        path, pin = self.old_version("study-a", b"def cell(): pass\n")
        for admitted in (None, {"ref": pin, "path": str(self.root / "absent.py")}):
            with self.subTest(admitted=admitted):
                with self.assertRaisesRegex(SelectorError, "registered metric source unavailable; remeasure"):
                    check_metric_source({"metric_source_ref": pin}, registered_source=admitted)

    def test_wrong_pin_or_bytes_fail(self):
        path, pin = self.old_version("study-a", b"def cell(): pass\n")
        other = dict(pin, path="/old/study-b/native_cells.py")
        with self.assertRaisesRegex(SelectorError, "metric_source_differs"):
            check_metric_source({"metric_source_ref": other}, registered_source={"ref": pin, "path": str(path)})
        path.write_bytes(b"def cell(): return 1\n")
        with self.assertRaisesRegex(SelectorError, "metric_source_differs"):
            check_metric_source({"metric_source_ref": pin}, registered_source={"ref": pin, "path": str(path)})
        current = dict(population.metric_source_ref(), sha256="0" * 64)
        with self.assertRaisesRegex(SelectorError, "metric_source_differs"):
            check_metric_source({"metric_source_ref": current},
                                registered_source={"ref": current, "path": str(path)})

    @unittest.skipUnless(os.environ.get("GEAK_EVO_ORIGINAL_SRC"), "GEAK_EVO_ORIGINAL_SRC not set")
    def test_real_old_pin_passes_with_original_bytes(self):
        original = Path(os.environ["GEAK_EVO_ORIGINAL_SRC"]).parents[1] / "bridge" / "native_cells.py"
        pin = {"path": "/opt/geak-evo-renewed-full-attention-20261002-v8/bridge/native_cells.py",
               "sha256": "f21dab78613ee04811cfaf080563e5f44072974914a76169a8dda9dfbd29fbbd"}
        admitted = {"ref": pin, "path": str(original)}
        self.assertEqual(check_metric_source({"metric_source_ref": pin}, registered_source=admitted), pin)


if __name__ == "__main__":
    unittest.main()
