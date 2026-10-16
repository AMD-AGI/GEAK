"""CPU callback acquisitions; fixture numbers are not kernel measurements."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ORIGINAL = os.environ.get('GEAK_EVO_EVALUATION_LAYOUT') == 'original'
if ORIGINAL:
    from geak_evo.jobs import job_digest
    from geak_evo.native_evaluator import NativeEvaluator, NativeEvaluationUnsettled, canonical, sha256
    from geak_evo.native_job import NativeJobCampaign
    from geak_evo.research_state import ResearchStore
else:
    from geak_evo.evaluate.jobs_v1 import job_digest
    from geak_evo.evaluate.native_evaluator import NativeEvaluator, NativeEvaluationUnsettled, canonical, sha256
    from geak_evo.evaluate.collection import NativeJobCollector as NativeJobCampaign
    from geak_evo.archive.store import ResearchStore
from engineer_job_fixture import fixture


# This trusted fixture module only validates identity; nothing is dispatched.
OPERATOR = b'''
import hashlib
import json
from pathlib import Path
def validate_descriptor(value):
    return value
def relative(name):
    assert not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/'))
    return name
def identity(root):
    root = Path(root)
    descriptor = root / 'operator_task.json'
    value = json.loads(descriptor.read_bytes())
    digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {'descriptor_sha256': digest('operator_task.json'),
            'frozen_files': {name: digest(name) for name in value['frozen_files']},
            'source_files': {name: digest(name) for name in value['editable_files']}}
'''


class NativeJobPairTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        job, files = fixture()
        job['timing']['aggregation'] = 'mean'
        second = copy.deepcopy(job['cases'][0])
        second['case_id'] = 'case-2'
        second['observed_frequency']['count'] = 5
        job['cases'].append(second)
        for name in ('timing', 'numerical'):
            job[name]['case_ids'].append('case-2')
        descriptor = {
            'editable_files': job['baseline']['source_paths'],
            'frozen_files': {'binding.json': job['artifacts']['binding.json']},
            'correctness_cases': [{'id': case['case_id']} for case in job['cases']],
            'performance_cases': [{'id': case['case_id']} for case in job['cases']],
            'timing': {'warmup': 1, 'iterations': 2}, 'limits': {'worker_timeout_s': 1},
        }
        files['operator_task.json'] = canonical(descriptor)
        job['artifacts']['operator_task.json'] = sha256(files['operator_task.json'])
        self.inputs = self.root / 'inputs'
        for name, raw in files.items():
            path = self.inputs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        operator = self.root / 'operator.py'
        operator.write_bytes(OPERATOR)
        self.job_id = job_digest(job)
        self.store = ResearchStore(self.root / 'store', allowed_epochs=[(self.job_id, 'fixture-epoch')],
                                   source_paths={self.job_id: job['baseline']['source_paths']})
        source = self.store.capture_source(self.inputs, job['baseline']['source_paths'])
        self.baseline = self.store.candidate(self.job_id, 'fixture-epoch', source)
        self.note = {'claim': 'CPU fixture', 'scope': 'No performance claim', 'next_test': 'Exercise host collection'}
        self.store.checkpoint('campaign', self.baseline, expected_revision=0, note=self.note)
        self.store.checkpoint('baseline', self.baseline, expected_revision=0, note=self.note)
        self.workspace = self.root / 'candidate'
        self.store.materialize(source, self.workspace)
        (self.workspace / 'source/baseline.py').write_text('def complete_call(x):\n    return x + 0\n')
        self.source = self.store.capture_source(self.workspace, job['baseline']['source_paths'])
        self.candidate = self.store.candidate(self.job_id, 'fixture-epoch', self.source, parents=[self.baseline])
        self.launched = []
        self.fail_correctness = False
        evaluator = NativeEvaluator(self.store, job=job, artifact_root=self.inputs,
            operator_tool={'path': str(operator), 'sha256': sha256(OPERATOR)},
            output_root=self.root / 'evaluations', measurement_epoch_id='fixture-epoch',
            launch=self.launch, settle=self.settle, repeats=2)
        # Exercise actual collection with CPU callbacks, bypassing the live GPU boundary.
        if ORIGINAL:
            self.campaign = NativeJobCampaign.__new__(NativeJobCampaign)
            self.campaign.__dict__.update(root=self.root, store=self.store, job=job, job_id=self.job_id,
                epoch='fixture-epoch', source_paths=job['baseline']['source_paths'], evaluator=evaluator,
                paired_timing={'order_seed': 7}, evaluations_per_turn=2, loop=None)
        else:
            self.campaign = NativeJobCampaign(self.store, evaluator=evaluator, job=job,
                job_id=self.job_id, measurement_epoch_id='fixture-epoch', paired_timing={'order_seed': 7})
        self.store.verify_evidence = self.campaign.verify_anchor_evidence

    def launch(self, request, directory):
        self.launched.append(request['operation_id'])
        return object()

    def settle(self, handle, request, directory):
        self.assertIsNotNone(handle)
        artifacts = {'stdout.bin': b'', 'stderr.bin': b''}
        if request['mode'] == 'correctness':
            artifacts['stdout.bin'] = b'GEAK_CORRECTNESS_PROGRESS: ' + canonical({
                'schema': 'geak-operator-correctness-progress-v1', 'completed': True, 'failure': None,
                'identity_before': request['operator_identity'], 'identity_after': request['operator_identity'],
                'cases': [{'case_id': case, 'status': 'pass'} for case in ('case-1', 'case-2')]}) + b'\n'
        else:
            baseline = request['binding']['candidate_id'] == self.baseline
            values = (10.0, 20.0) if baseline else (5.0, 25.0)
            offset = 0.02 * (int(request['operation_id'].rsplit('-', 1)[1]) - 1)
            artifacts['performance_report.json'] = canonical({'test_cases': [
                {'test_case_id': case, 'execution_time_ms': value + offset,
                 'params': {'timing_scope': 'complete_callable_synchronized_wall'}}
                for case, value in zip(('case-1', 'case-2'), values)]})
        receipt = {'schema': 'geak-evo-native-settlement-v1', 'operation_id': request['operation_id'],
            'request_sha256': sha256(canonical(request)), 'execution_started': True, 'settled': True,
            'evaluation_integrity_valid': True, 'measurement_valid': True, 'peer_exception_active': False,
            'artifacts_sha256': {name: sha256(raw) for name, raw in artifacts.items()}, 'errors': [],
            'exit_code': 1 if self.fail_correctness and request['mode'] == 'correctness' else 0,
            'qualification_ref': {'fixture': 'CPU callbacks only'}, 'physical_devices': ['CPU fixture'],
            'costs': {'gpu_seconds': 0}}
        return {'receipt': receipt, 'artifacts': artifacts}

    def collect(self, attempt='episode'):
        result = {'receipt': {'settlement': {'settled': True}, 'captured_tip_source_id': self.source,
            'request': {'workspace': str(self.workspace), 'direction_id': 'baseline', 'continuation': {'mode': 'start'}},
            'attempt_id': attempt, 'elapsed_seconds': 1.0}}
        return self.campaign.collect(result, self.candidate)

    def binding(self, candidate):
        return {'job_id': self.job_id, 'measurement_epoch_id': 'fixture-epoch',
                'candidate_source_id': self.store.get_candidate(candidate)['source_id']}

    def test_live_collection_retains_both_sources_frozen_reference_means_and_replay(self):
        # G's baseline direction is allowed to move; the measurement reference must not.
        self.store.checkpoint('baseline', self.candidate, expected_revision=1, note=self.note)
        collected = self.collect()
        artifacts = collected['artifacts']
        pair = json.loads(artifacts['native_pair/receipt.json'])
        self.assertTrue(pair['eligible'])
        self.assertEqual(pair['binding']['baseline_source_id'], self.binding(self.baseline)['candidate_source_id'])
        measurements = json.loads(artifacts['native_pair/measurements.json'])
        first = measurements['cases'][0]['blocks']
        self.assertEqual([row['baseline']['value'] for row in first], [10.0, 10.02])
        self.assertTrue(all(row['baseline']['kind'] == 'unprofiled_interval_mean' for row in first))
        self.assertEqual([case['observed_frequency']['count'] for case in measurements['cases']], [2, 5])
        feedback = json.loads(artifacts['paired_feedback.json'])
        self.assertEqual(feedback['measurement_role'], 'adaptive_search_feedback')
        self.assertAlmostEqual(feedback['comparison']['aggregate']['estimate']['speedup'], 120.07 / 135.07)
        self.assertTrue(all(case['roofline']['status'] == 'unknown' for case in feedback['comparison']['cases']))
        self.assertTrue(all(json.loads(raw)['status'] == 'unknown' for name, raw in artifacts.items()
                            if name.startswith('work_models/')))
        candidate_evidence = self.store.archive_evidence(self.candidate, artifacts)
        self.store.promote('baseline', self.candidate, candidate_evidence, expected_revision=2)
        _, reference = self.store.evidence(feedback['baseline_evidence_ref']['evidence_id'])
        self.assertTrue(self.campaign.verify_anchor_evidence(reference, self.binding(self.baseline))['eligible'])
        self.assertEqual(collected['costs']['evaluations'], 2)
        replay = self.collect()
        self.assertEqual(len(self.launched), 6)
        self.assertEqual(replay['costs'], collected['costs'])
        self.assertEqual(replay['artifacts'], artifacts)
        missing = dict(artifacts)
        del missing['native_pair/acquisition_complete.json']
        with self.assertRaises(ValueError):
            self.campaign.verify_anchor_evidence(missing, self.binding(self.candidate))
        # Pair admission also authenticates the companion's retained raw report.
        report = Path(first[0]['baseline']['raw_report_ref']['path'])
        report.chmod(0o600)
        report.write_bytes(b'changed companion report')
        with self.assertRaises(ValueError):
            self.campaign.verify_anchor_evidence(artifacts, self.binding(self.candidate))

    def test_independent_completion_after_interruption_cannot_become_paired(self):
        evaluator = self.campaign.evaluator
        original = evaluator._operation
        def interrupt(prepared, index, mode):
            if prepared['evaluation_id'].endswith('-candidate'):
                raise KeyboardInterrupt()
            return original(prepared, index, mode)
        with mock.patch.object(evaluator, '_operation', interrupt), self.assertRaises(KeyboardInterrupt):
            self.collect('interrupted')
        evaluator.evaluate(self.baseline, evaluation_id='pair-interrupted-baseline')
        evaluator.evaluate(self.candidate, evaluation_id='pair-interrupted-candidate')
        collected = self.collect('interrupted')
        pair = json.loads(collected['artifacts']['native_pair/receipt.json'])
        self.assertFalse(pair['continuous_acquisition'])
        self.assertFalse(pair['eligible'])
        self.assertNotIn('native_pair/acquisition_complete.json', collected['artifacts'])
        self.assertEqual(len(self.launched), 6)
        with self.assertRaises(ValueError):
            self.campaign.verify_anchor_evidence(collected['artifacts'], self.binding(self.candidate))

    def test_persistence_interruption_keeps_completed_acquisition_without_relaunch(self):
        evaluator = self.campaign.evaluator
        with mock.patch.object(evaluator, '_finish_evaluation', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.collect('persist')
        collected = self.collect('persist')
        pair = json.loads(collected['artifacts']['native_pair/receipt.json'])
        self.assertTrue(pair['eligible'])
        self.assertTrue(pair['continuous_acquisition'])
        self.assertEqual(len(self.launched), 6)
        self.assertTrue(self.campaign.verify_anchor_evidence(collected['artifacts'], self.binding(self.candidate))['eligible'])

    def test_terminal_source_summaries_cannot_recreate_missing_pair_order_marker(self):
        collected = self.collect('missing-order')
        self.assertTrue(json.loads(collected['artifacts']['native_pair/receipt.json'])['eligible'])
        marker = self.root / 'evaluations/pair-missing-order/acquisition_complete.json'
        marker.unlink()
        replay = self.collect('missing-order')
        pair = json.loads(replay['artifacts']['native_pair/receipt.json'])
        self.assertFalse(pair['continuous_acquisition'])
        self.assertFalse(pair['eligible'])
        self.assertFalse(marker.exists())
        self.assertNotIn('native_pair/acquisition_complete.json', replay['artifacts'])
        self.assertEqual(len(self.launched), 6)
        with self.assertRaises(ValueError):
            self.campaign.verify_anchor_evidence(replay['artifacts'], self.binding(self.candidate))

    def test_reference_failure_retains_feedback_without_inventing_candidate_work(self):
        self.fail_correctness = True
        collected = self.collect('reference-failure')
        self.assertEqual(collected['costs']['evaluations'], 1)
        self.assertEqual(len(self.launched), 1)
        self.assertNotIn('native_evaluation.json', collected['artifacts'])
        self.assertIn('native_not_launched.json', collected['artifacts'])
        self.assertFalse(json.loads(collected['artifacts']['paired_feedback.json'])['eligible'])
        observed = self.campaign.evaluator.verify_pair_evidence(collected['artifacts'])
        self.assertTrue(observed['settled'])
        self.assertNotIn('candidate', observed['source_evaluations'])
        with self.assertRaises(ValueError):
            self.campaign.verify_anchor_evidence(collected['artifacts'], self.binding(self.candidate))

    def test_source_change_after_harvest_prevents_native_launch(self):
        (self.workspace / 'source/baseline.py').write_bytes(b'# changed after harvest\n')
        with self.assertRaisesRegex(ValueError, 'changed after harvest') as caught:
            self.collect('changed-source')
        self.assertIs(type(caught.exception), ValueError)
        self.assertEqual(self.launched, [])

    def test_source_change_during_collection_rejects_association(self):
        original = self.campaign.evaluator.evaluate_pair
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            (self.workspace / 'source/baseline.py').write_bytes(b'# changed during collection\n')
            return result
        with mock.patch.object(self.campaign.evaluator, 'evaluate_pair', changed):
            with self.assertRaisesRegex(ValueError, 'changed during collection') as caught:
                self.collect('changed-during')
        self.assertIs(type(caught.exception), ValueError)
        self.assertEqual(len(self.launched), 6)

    def test_pair_analysis_binds_receipt_and_complete_observed_cases(self):
        original = self.campaign.evaluator.evaluate_pair
        for mutation in ('epoch', 'missing-case'):
            def changed(*args, **kwargs):
                result = original(*args, **kwargs)
                if mutation == 'epoch':
                    result['record']['measurement_epoch_id'] = 'forged-epoch'
                    for case in result['record']['cases']:
                        for block in case['blocks']:
                            block['measurement_epoch_id'] = 'forged-epoch'
                else:
                    result['record']['cases'].pop()
                return result
            with self.subTest(mutation=mutation), mock.patch.object(self.campaign.evaluator, 'evaluate_pair', changed):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    self.collect('forged-' + mutation)

    def test_unsettled_native_collection_keeps_original_pending_exception(self):
        original = self.campaign.evaluator.settle
        def unsettled(*args, **kwargs):
            result = original(*args, **kwargs)
            result['receipt']['settled'] = False
            return result
        with mock.patch.object(self.campaign.evaluator, 'settle', unsettled):
            with self.assertRaises(NativeEvaluationUnsettled) as caught:
                self.collect('pending')
        self.assertIs(type(caught.exception), NativeEvaluationUnsettled)
        self.assertFalse(caught.exception.result['receipt']['settled'])
        self.assertEqual(len(self.launched), 1)


if __name__ == '__main__':
    unittest.main()
