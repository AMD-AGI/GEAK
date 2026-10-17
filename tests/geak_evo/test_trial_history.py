"""Synthetic archive-only checks for the shared trial view."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.archive.store import ResearchStore, StateError
from geak_evo.archive.trials import trial_history
from geak_evo.archive.succession import snapshot_trial_history


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


class TrialHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ResearchStore(self.root / 'store', allowed_epochs=[('job', 'epoch'), ('job', 'other')],
                                   source_paths={'job': ['main.py']})
        work = self.root / 'work'
        work.mkdir()
        (work / 'main.py').write_text('pass\n')
        self.source = self.store.capture_source(work, ['main.py'])
        self.candidate = self.store.candidate('job', 'epoch', self.source)

    def collection(self, *, outcomes=None, feedbacks=None, observations=None, candidate=None):
        candidate = candidate or self.candidate
        feedbacks = feedbacks or [dict(status='slower', source_id=self.source, operation_id='op', turn_id='attempt')]
        links, raw, trials, rows = [], {}, [], []
        for index, feedback in enumerate(feedbacks):
            payload = encoded(feedback)
            name = 'feedback-%s.json' % index
            ref = dict(path='/unavailable/feedback-%s.json' % index, sha256=hashlib.sha256(payload).hexdigest())
            raw[name] = payload
            trials.append(dict(archive_ref=ref, error=(outcomes or [None] * len(feedbacks))[index]))
            rows.append(dict(trial_index=index, candidate_id=candidate, source_id=self.source,
                operation_id=feedback.get('operation_id'), owner_turn_id='attempt', feedback_ref=ref,
                source_retention_error=None, feedback_error=None))
        inventory = encoded(dict(attempt_id='attempt', trials=rows))
        raw.update({'native_trial_sources.json': inventory, 'archive/manifest.json': encoded(dict(trials=trials))})
        raw_id = self.store.archive_evidence(candidate, raw)
        for index, row in enumerate(rows):
            links.append(dict(original_ref=row['feedback_ref'],
                              evidence_ref=dict(evidence_id=raw_id, artifact='feedback-%s.json' % index)))
        obs = []
        for index, times in enumerate(observations or []):
            obs.append(dict(source_id=self.source, operation_id=rows[index]['operation_id'],
                            feedback_ref=rows[index]['feedback_ref'], per_case_times_ms=times))
        collection = self.store.archive_evidence(candidate, {'native_trial_sources.json': inventory,
            'native_idea_harvest.json': encoded(dict(trials=trials)), 'native_evidence_index.json': encoded(links),
            'native_idea_context.json': encoded(dict(native_observations=obs))})
        return collection, rows

    def test_middle_trials_and_same_source_repetitions_remain_present(self):
        feedbacks = [dict(status='slower', source_id=self.source, operation_id='op-%s' % i, turn_id='attempt')
                     for i in range(35)]
        identity, _ = self.collection(feedbacks=feedbacks, observations=[{'case': float(i + 1)} for i in range(35)])
        view = trial_history(self.store, [identity])
        self.assertEqual(view['coverage'], dict(inventory_count=1, trial_count=35, omitted_rows=0, outside_inventory='unknown'))
        self.assertEqual(len(view['groups']), 1)
        self.assertEqual({row['operation_id'] for row in view['trials']}, {'op-%s' % i for i in range(35)})
        self.assertEqual(len({row['trial_id'] for row in view['trials']}), 35)
        self.assertTrue(all(len(row['evidence_ids']) == 2 for row in view['trials']))
        self.assertIn('never tested', view['scope'])

    def test_failure_prose_does_not_claim_a_phase_or_reject_an_idea(self):
        identity, _ = self.collection(outcomes=['correctness failed'])
        row = trial_history(self.store, [identity])['trials'][0]
        self.assertEqual(row['outcome']['error'], 'correctness failed')
        self.assertIsNone(row['stages']['correctness'])
        self.assertIsNone(row['stages']['benchmark'])
        self.assertIsNone(row['times_ms'])
        self.assertEqual(row['stages']['reported_status'], 'slower')
        self.assertIsNone(row['timing_context']['comparator_source_id'])

    def test_structured_stages_require_source_operation_and_turn_binding(self):
        feedback = dict(source_id=self.source, operation_id='op', turn_id='attempt', admitted=False,
                        status='admission_refused', native_observations={'correctness': {'status': 'not_run'}})
        identity, _ = self.collection(feedbacks=[feedback])
        row = trial_history(self.store, [identity])['trials'][0]
        self.assertIs(row['stages']['admission'], False)
        self.assertEqual(row['stages']['correctness'], 'not_run')
        for key, wrong in [('source_id', '0' * 64), ('turn_id', 'another-attempt')]:
            broken = dict(feedback, **{key: wrong})
            identity, _ = self.collection(feedbacks=[broken], observations=[{'case': 1.0}])
            row = trial_history(self.store, [identity])['trials'][0]
            self.assertEqual(row['association']['status'], 'conflict')
            self.assertIsNone(row['times_ms'])
            self.assertIsNone(row['stages']['admission'])
            self.assertIsNone(row['stages']['correctness'])

    def test_conflicting_observations_withhold_scalar_without_dropping_trial(self):
        identity, rows = self.collection(observations=[{'case': 9.0}])
        conflict = self.store.archive_evidence(self.candidate, {'native_idea_context.json': encoded(dict(
            native_observations=[dict(source_id=self.source, operation_id='op', feedback_ref=rows[0]['feedback_ref'],
                                      per_case_times_ms={'case': 1.0})]))})
        row = trial_history(self.store, [identity, conflict])['trials'][0]
        self.assertEqual(row['association']['status'], 'conflict')
        self.assertEqual(len(row['observation_refs']), 2)
        self.assertIsNone(row['times_ms'])
        self.assertEqual(row['candidate_id'], self.candidate)

    def test_same_source_other_epoch_cannot_supply_observation(self):
        identity, rows = self.collection()
        other = self.store.candidate('job', 'other', self.source)
        foreign = self.store.archive_evidence(other, {'native_idea_context.json': encoded(dict(
            native_observations=[dict(source_id=self.source, operation_id='op', feedback_ref=rows[0]['feedback_ref'],
                                      per_case_times_ms={'case': 0.001})]))})
        row = trial_history(self.store, [identity, foreign])['trials'][0]
        self.assertIsNone(row['times_ms'])
        first, _ = self.collection(observations=[{'case': 9.0}])
        second, _ = self.collection(candidate=other, observations=[{'case': 7.0}])
        together = trial_history(self.store, [first, second])
        self.assertEqual(len(together['groups']), 2)
        self.assertEqual({row['measurement_epoch_id']: row['times_ms'] for row in together['trials']},
                         {'epoch': {'case': 9.0}, 'other': {'case': 7.0}})
        self.assertTrue(all(row['feedback_state'] == 'available' for row in together['trials']))

    def test_invalid_inventory_index_cannot_alias_a_different_trial(self):
        identity, _ = self.collection()
        _, artifacts = self.store.evidence(identity)
        inventory = json.loads(artifacts['native_trial_sources.json'])
        for invalid in (-1, True, 1):
            inventory['trials'][0]['trial_index'] = invalid
            altered = dict(artifacts, **{'native_trial_sources.json': encoded(inventory)})
            broken = self.store.archive_evidence(self.candidate, altered)
            with self.assertRaisesRegex(StateError, 'inventory position'):
                trial_history(self.store, [broken])

    def test_parent_view_uses_same_rows_and_saved_episode_handles(self):
        identity, _ = self.collection(observations=[{'case': 9.0}])
        packet = self.store.archive_evidence(self.candidate, {'packet.json': encoded(dict(proposal={'engineer_id': 'e1'}))})
        snapshot = dict(state=dict(opportunities=[dict(opportunity_id='attempt', direction_id='direction')],
            history=[dict(attempt_id='attempt', collection_ref=dict(evidence_id=identity, artifact='native_idea_harvest.json'),
                          packet_ref=dict(evidence_id=packet, artifact='packet.json'))]),
            archive=dict(directions={'direction': {'card': {'idea': {'idea_id': 'idea'}}}}))
        view = snapshot_trial_history(self.store, snapshot)
        row = view['trials'][0]
        self.assertEqual((row['direction_id'], row['idea_id'], row['engineer_id']), ('direction', 'idea', 'e1'))
        self.assertEqual(row['times_ms'], {'case': 9.0})
        self.assertEqual(view['trials'], trial_history(self.store, [identity], attempts={
            'attempt': dict(direction_id='direction', idea_id='idea', engineer_id='e1')})['trials'])


if __name__ == '__main__':
    unittest.main()
