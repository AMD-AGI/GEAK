# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


module_path = Path(__file__).resolve().parents[1] / 'query_population.py'
specification = importlib.util.spec_from_file_location('population_query_variant', module_path)
population_query = importlib.util.module_from_spec(specification)
specification.loader.exec_module(population_query)


class ReportedContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.lineage = self.root / 'peer_a'
        self.lineage.mkdir()
        self.context_path = self.lineage / 'r1.context.json'
        self.checkpoint_path = self.lineage / 'r1.json'
        self.context = {
            'authority': 'author_reported_interpretation_only',
            'published_source_binding': {
                'lineage_id': 'peer_a', 'round': 1, 'source_id': 'a' * 64,
                'source_tree': 'b' * 40,
                'patch_ref': {'path': str(self.lineage / 'r1.diff'), 'sha256': 'c' * 64},
            },
            'engineer': {'notes': 'Synthetic original agent observation, not a verified finding.'},
            'evidence_ref_scope': 'Origin paths are unavailable unless separately shared.',
        }
        self.checkpoint = {
            'lineage_id': 'peer_a', 'round': 1, 'source_id': 'a' * 64,
            'source_tree': 'b' * 40,
            'patch_ref': {'path': str(self.lineage / 'r1.diff'), 'sha256': 'c' * 64},
        }
        self.rebind()

    def rebind(self):
        raw = json.dumps(self.context).encode()
        self.context_path.write_bytes(raw)
        self.checkpoint['reported_context_ref'] = {
            'path': str(self.context_path), 'sha256': population_query.digest(raw),
        }
        checkpoint_raw = json.dumps(self.checkpoint).encode()
        self.checkpoint_path.write_bytes(checkpoint_raw)
        self.record = {
            'status': 'qualified', 'checkpoint_path': str(self.checkpoint_path),
            'checkpoint_sha256': population_query.digest(checkpoint_raw),
            'checkpoint': copy.deepcopy(self.checkpoint),
        }

    def read(self):
        return population_query.materialize_reported_context(self.record, self.root)

    def test_exact_context_without_upgrading_claims(self):
        before = copy.deepcopy(self.record)
        result = self.read()
        self.assertEqual(result['status'], 'available')
        self.assertEqual(result['context'], self.context)
        self.assertEqual(result['authority'], 'author_reported_interpretation_only')
        self.assertEqual(self.record, before)

    def test_missing_context_does_not_disqualify_source(self):
        self.context_path.unlink()
        self.assertEqual(self.read()['status'], 'unavailable')
        self.assertEqual(self.record['status'], 'qualified')

    def test_changed_context_hash_is_unavailable(self):
        self.context_path.write_text('{}')
        self.assertIn('hash differs', self.read()['reason'])

    def test_changed_checkpoint_is_unavailable(self):
        self.checkpoint_path.write_text('{}')
        self.assertIn('checkpoint identity changed', self.read()['reason'])

    def test_wrong_published_source_is_unavailable(self):
        self.context['published_source_binding']['source_id'] = 'd' * 64
        self.rebind()
        self.assertIn('source differs', self.read()['reason'])

    def test_origin_report_path_is_never_fetched(self):
        self.record['checkpoint']['reported_context_ref']['path'] = '/unavailable-origin/report.json'
        with mock.patch.object(population_query, 'read_json') as read_json:
            self.assertIn('outside its checkpoint', self.read()['reason'])
            read_json.assert_not_called()

    def test_symlink_escape_is_rejected(self):
        outside = self.root.parent / (self.root.name + '-outside-context.json')
        outside.write_text('{}')
        self.addCleanup(outside.unlink)
        self.context_path.unlink()
        self.context_path.symlink_to(outside)
        self.assertIn('outside its checkpoint', self.read()['reason'])

    def test_unqualified_record_does_not_read_context(self):
        self.record['status'] = 'invalid'
        with mock.patch.object(population_query, 'read_json') as read_json:
            self.assertIn('source is not qualified', self.read()['reason'])
            read_json.assert_not_called()

    def test_opt_in_query_preserves_selection_and_default_output(self):
        manifest = {'schema': 'geak.population/1', 'objective': {'kind': 'geomean'},
                    'lineages': [{'id': 'peer_a', 'optimizer': 'evolution', 'eval_dir': str(self.lineage)}]}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        candidate = {'lineage_id': 'peer_a', 'kind': 'checkpoint', 'round': 1,
                     'status': 'checkpoint_only', 'source_status': 'legacy_verified', 'speedup': 1.1}
        def checked(*args):
            return copy.deepcopy(self.record), copy.deepcopy(candidate)
        with mock.patch.object(population_query, 'check_checkpoint', side_effect=checked):
            default = population_query.query(self.root, checkpoints_only=True)
            expanded = population_query.query(self.root, checkpoints_only=True, reported_context=True)
        self.assertNotIn('reported_context_query', default['records'][0])
        self.assertEqual(expanded['records'][0].pop('reported_context_query')['status'], 'available')
        self.assertEqual(default, expanded)


if __name__ == '__main__':
    unittest.main()
