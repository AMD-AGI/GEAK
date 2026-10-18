"""Precise JSON feedback from retained replies; no model or native operations."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from geak_evo.planner import native


class PlannerJsonDiagnosticsTests(unittest.TestCase):
    def rejected_context(self, raw):
        with self.assertRaises(ValueError) as rejected:
            native._parse_recommendation_response(raw)
        failure = native._rejection('structured_response', rejected.exception)
        row = {'kind': native.REJECTED_DISPOSITION, 'operation_id': 'operation',
               'settled': True, 'costs': {'wall_seconds': 7, 'evaluations': 0},
               'failure': failure,
               'failure_ref': {'evidence_id': 'e' * 64, 'artifact': 'planner.json'}}
        planner = {'operation_id': row['operation_id'], 'settled': True, 'proposal': None,
                   'costs': row['costs'],
                   'host_disposition': {'kind': row['kind'], 'failure': failure}}
        artifacts = {'planner.json': native._json(planner), 'response.txt': raw}
        original = copy.deepcopy((row, artifacts))
        binding = {'job_id': 'job', 'measurement_epoch_id': 'epoch'}
        envelope = {'binding': binding, 'artifacts': {
            name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'objects').mkdir()
            for name, data in artifacts.items():
                (root / 'objects' / envelope['artifacts'][name]).write_bytes(data)
            store = SimpleNamespace(root=root, evidence=lambda identity: (envelope, artifacts))
            summaries, references = native._rejected_planner_context(
                store, {'state': {'binding': binding, 'history': [row]}})
            self.assertEqual(len(references), 2)
            self.assertEqual(Path(summaries[0]['raw_response_file_ref']['path']).read_bytes(), raw)
        self.assertEqual((row, artifacts), original)
        self.assertEqual(summaries[0]['validation_error'], failure)
        return summaries[0]

    def test_missing_nested_brace_reaches_next_planner_without_changing_rejection(self):
        prefix = b'Commentary\n'
        body = (b'{"proposal": {"note": {"claim": "x", "claim_requests": []}, '
                b'"recommendation": {"op": "keep"}}')
        raw = prefix + body
        summary = self.rejected_context(raw)
        self.assertEqual(summary['validation_error'], {
            'stage': 'structured_response', 'kind': 'ValueError',
            'message': 'malformed JSON container in response'})
        self.assertEqual(summary['secondary_diagnostics'], [{
            'stage': 'json_decode', 'kind': 'JSONDecodeError',
            'message': "Expecting ',' delimiter: line 2 column %d (char %d)" %
                       (len(body) + 1, len(raw))}])

    def test_malformed_outer_object_does_not_salvage_valid_nested_object(self):
        raw = b'{"proposal": {"kept": true} invalid}'
        summary = self.rejected_context(raw)
        self.assertEqual(summary['secondary_diagnostics'], [{
            'stage': 'json_decode', 'kind': 'JSONDecodeError',
            'message': "Expecting ',' delimiter: line 1 column 29 (char 28)"}])

    def test_decoder_locations_keep_character_and_fenced_body_coordinates(self):
        for raw, message in (
                ('é\n{"x": }'.encode(), 'Expecting value: line 2 column 7 (char 8)'),
                (b'Intro\n```json\n{"x": }\n```', 'Expecting value: line 1 column 7 (char 6)')):
            with self.subTest(raw=raw):
                summary = self.rejected_context(raw)
                self.assertEqual(summary['secondary_diagnostics'], [{
                    'stage': 'json_decode', 'kind': 'JSONDecodeError', 'message': message}])

    def test_valid_replies_and_non_json_rejections_are_unchanged(self):
        for raw, proposal, directive in (
                (b'null', None, {'op': 'keep'}),
                (b'{"direction_id":"d"}', {'direction_id': 'd'}, {'op': 'keep'}),
                (b'Intro\n{"proposal":null,"recommendation":{"op":"keep"}}',
                 None, {'op': 'keep'})):
            with self.subTest(raw=raw):
                _, actual_proposal, actual_directive = native._parse_recommendation_response(raw)
                self.assertEqual((actual_proposal, actual_directive), (proposal, directive))
                self.assertEqual(native._response_diagnostics(raw), [])
        duplicate = self.rejected_context(b'{"proposal":null,"proposal":null}')
        self.assertEqual(duplicate['validation_error']['message'], 'duplicate proposal field')
        self.assertEqual(duplicate['secondary_diagnostics'], [])


if __name__ == '__main__':
    unittest.main()
