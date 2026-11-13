"""CPU archive/recovery checks: link identity is evidence, external data is not copied."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'retain_measurement', Path(__file__).resolve().parents[1] / 'retain_measurement.py')
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / 'profile/out/tasks/task'
        self.owner = self.root / 'optimize/out/work/runs/task'
        self.owner.mkdir(parents=True)
        (self.root / 'control').mkdir()
        (self.root / 'control/completed-archive-dedup.lock').touch()
        self.payloads = {'run/00_result.json': b'{"ok": true}\n',
                         'kernel_src/kernel.py': b'pass\n'}
        for name, raw in self.payloads.items():
            p = self.task / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
        external = self.root / 'installed'
        external.mkdir()
        (external / 'library.co').write_bytes(b'external payload')
        self.links = {'run/library.co': str(external / 'library.co'),
                      'run/library-dir': str(external),
                      'run/missing': str(external / 'missing'),
                      'run/relative': '00_result.json', 'run/cycle': 'cycle'}
        for name, target in self.links.items():
            (self.task / name).symlink_to(target)

    def assert_archive(self, archive):
        record = json.loads((archive / 'retention.json').read_text())
        self.assertTrue(record['complete'])
        self.assertEqual(record['copied_files'], len(self.payloads))
        self.assertEqual(record['copied_bytes'], sum(map(len, self.payloads.values())))
        self.assertEqual(record['copied_symlinks'], len(self.links))
        for name, raw in self.payloads.items():
            self.assertEqual((archive / name).read_bytes(), raw)
        for name, target in self.links.items():
            self.assertTrue((archive / name).is_symlink())
            self.assertEqual(os.readlink(archive / name), target)
        retention.reuse_helper().verify_symlinks(archive, record)
        return record

    def test_retains_literal_links_and_independent_regular_files(self):
        with patch.dict(os.environ, {'WORKLOAD_RETENTION_REUSE': '0'}):
            archive = retention.begin(self.task, self.owner / 'evidence', 'pre-search')
            retention.retain(self.task, archive)
        self.assert_archive(archive)
        (self.task / 'run/00_result.json').write_bytes(b'changed')
        (Path(self.links['run/library-dir']) / 'new-external-file').write_bytes(b'not archived')
        self.assert_archive(archive)

    def test_optional_reuse_still_validates_regular_index_with_links_present(self):
        with patch.dict(os.environ, {'WORKLOAD_RETENTION_REUSE': '1'}):
            first = retention.begin(self.task, self.owner / 'evidence', 'sweep')
            retention.retain(self.task, first)
            self.assert_archive(first)
            second = retention.begin(self.task, self.owner / 'evidence', 'sweep')
            retention.retain(self.task, second)
            receipt = self.assert_archive(second)
            self.assertEqual(receipt['reuse']['base_status'], 'validated_completed_hint')
            self.assertEqual(receipt['reuse']['base_archive'], str(first))
            self.assertEqual(receipt['reuse']['copied_files'] + receipt['reuse']['reused_files'],
                             len(self.payloads))

    def test_recovery_preserves_partial_archive_and_retains_links_without_rerun(self):
        with patch.dict(os.environ, {'WORKLOAD_RETENTION_REUSE': '0'}):
            pending = self.owner / 'pending_evidence.path'
            partial = retention.begin(self.task, self.owner / 'evidence', 'pre-search', pending)
            with patch.object(Path, 'symlink_to', side_effect=OSError('interrupted link copy')):
                with self.assertRaises(OSError):
                    retention.retain(self.task, partial, measure_exit=0, native_exit=0)
            partial_bytes = (partial / 'retention.json').read_bytes()
            self.assertFalse(json.loads(partial_bytes)['complete'])
            recovered = retention.recover(self.task, pending, None)
        self.assertNotEqual(recovered, partial)
        self.assertEqual((partial / 'retention.json').read_bytes(), partial_bytes)
        receipt = self.assert_archive(recovered)
        self.assertEqual(receipt['recovered_from'], str(partial))
        self.assertEqual(receipt['native_exit_code'], 0)
        self.assertFalse(pending.exists())
        self.assertEqual((self.owner / 'pre_search_evidence.path').read_text().strip(), str(recovered))

    def test_changed_archived_links_reject_reuse_and_complete_recovery(self):
        def change_target(archive):
            p = archive / 'run/library.co'
            p.unlink()
            p.symlink_to('different-target')

        def link_to_file(archive):
            p = archive / 'run/library.co'
            p.unlink()
            p.write_bytes(b'not a link')

        def file_to_link(archive):
            p = archive / 'run/00_result.json'
            p.unlink()
            p.symlink_to('library.co')

        for mutate in (change_target, link_to_file, file_to_link):
            with self.subTest(mutation=mutate.__name__), patch.dict(os.environ, {'WORKLOAD_RETENTION_REUSE': '1'}):
                pending = self.owner / 'pending_evidence.path'
                archive = retention.begin(self.task, self.owner / 'evidence', 'pre-search', pending)
                retention.retain(self.task, archive)
                mutate(archive)
                original_receipt = (archive / 'retention.json').read_bytes()
                with self.assertRaisesRegex(ValueError, 'symlink identity differs'):
                    retention.recover(self.task, pending, None)
                self.assertTrue(pending.exists())
                fresh = retention.begin(self.task, self.owner / 'evidence', 'sweep')
                retention.retain(self.task, fresh)
                receipt = self.assert_archive(fresh)
                self.assertIsNone(receipt['reuse']['base_archive'])
                self.assertIn('symlink identity differs', receipt['reuse']['base_status'])
                self.assertEqual(receipt['reuse']['reused_files'], 0)
                self.assertEqual((archive / 'retention.json').read_bytes(), original_receipt)

    def test_legacy_regular_archive_identity_remains_compatible(self):
        for name in self.links:
            (self.task / name).unlink()
        self.links = {}
        with patch.dict(os.environ, {'WORKLOAD_RETENTION_REUSE': '0'}):
            pending = self.owner / 'pending_evidence.path'
            archive = retention.begin(self.task, self.owner / 'evidence', 'pre-search', pending)
            retention.retain(self.task, archive)
            record = self.assert_archive(archive)
            del record['copied_symlinks'], record['symlink_sha256']
            retention.write_record(archive, record)
            self.assertEqual(retention.recover(self.task, pending, None), archive)

    def test_special_files_still_stop_retention_and_keep_partial_evidence(self):
        os.mkfifo(self.task / 'run/zz_pipe')
        archive = retention.begin(self.task, self.owner / 'evidence', 'sweep')
        with patch.dict(os.environ, {'WORKLOAD_RETENTION_REUSE': '0'}):
            with self.assertRaisesRegex(ValueError, 'non-regular evidence entry'):
                retention.retain(self.task, archive)
        self.assertFalse(json.loads((archive / 'retention.json').read_text())['complete'])
        self.assertEqual((archive / 'run/00_result.json').read_bytes(), self.payloads['run/00_result.json'])
        self.assertTrue((archive / 'run/cycle').is_symlink())


if __name__ == '__main__':
    unittest.main()
