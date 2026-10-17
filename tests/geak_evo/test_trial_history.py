"""Synthetic archive-only checks for the shared trial view."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from geak_evo.archive.store import ResearchStore, StateError
from geak_evo.archive.trials import trial_history
from geak_evo.archive.export import build
from geak_evo.archive.succession import snapshot_trial_history, retained_trial_candidates


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
        inventory = encoded(dict(schema_version=1, attempt_id='attempt', trials=rows))
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
        self.assertEqual(view['coverage'], dict(inventory_count=1, trial_count=35,
            count_label='retained trial rows (any state)', omitted_rows=0, outside_inventory='unknown'))
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

    def test_multiple_linked_copies_are_one_component_in_every_caller(self):
        identity, _ = self.collection(observations=[{'case': 9.0}])
        _, collection = self.store.evidence(identity)
        links = json.loads(collection['native_evidence_index.json'])
        first = links[0]['evidence_ref']['evidence_id']
        _, raw = self.store.evidence(first)
        third = self.store.archive_evidence(self.candidate, dict(raw, marker=b'third copy'))
        onward = dict(original_ref=links[0]['original_ref'],
                      evidence_ref=dict(evidence_id=third, artifact='feedback-0.json'))
        second = self.store.archive_evidence(self.candidate, dict(raw, marker=b'second copy',
            **{'native_evidence_index.json': encoded([onward])}))
        links.append(dict(original_ref=links[0]['original_ref'],
                          evidence_ref=dict(evidence_id=second, artifact='feedback-0.json')))
        root = self.store.archive_evidence(self.candidate, dict(collection,
            **{'native_evidence_index.json': encoded(links)}))
        snapshot = dict(state=dict(opportunities=[], history=[dict(attempt_id='attempt',
            collection_ref=dict(evidence_id=root, artifact='native_idea_harvest.json'))]))
        parent = snapshot_trial_history(self.store, snapshot)
        for seeds in ([root], [second, root, third, first], [first, third, root, second]):
            planner = snapshot_trial_history(self.store, snapshot, seeds)
            self.assertEqual(parent, planner)
            self.assertEqual(planner['coverage']['inventory_count'], 1)
            self.assertEqual(planner['coverage']['trial_count'], 1)
            self.assertEqual(len(planner['trials'][0]['inventory_refs']), 4)
        # Adding a previously unseen link can change the canonical display ID.
        # The original immutable row alias remains valid and is retained.
        prior = trial_history(self.store, [first])
        self.assertIn(prior['trials'][0]['inventory_refs'][0], parent['trials'][0]['inventory_refs'])
        self.assertIn('later links may change', parent['identity']['trial_id'])
        unlinked = self.store.archive_evidence(self.candidate, dict(raw, marker=b'unlinked episode'))
        independent = trial_history(self.store, [root, unlinked])
        self.assertEqual(independent['coverage']['trial_count'], 2)
        self.assertEqual(len({row['trial_id'] for row in independent['trials']}), 2)
        self.store.checkpoint('campaign', self.candidate, expected_revision=0, evidence=[root],
            note=dict(claim='synthetic', scope='test', next_test='none', evolution={'history': []}))
        _, _, files = build(self.store)
        exported = json.loads(files['trial-history.json'])
        self.assertEqual(exported['coverage']['trial_count'], 1)
        self.assertEqual(len(json.loads(files['trials.json'])), 1)
        self.assertEqual({row['trial_id'] for row in exported['trials']},
                         {row['trial_id'] for row in parent['trials']})

    def test_legacy_manifest_builds_without_loading_unrelated_artifact(self):
        identity, rows = self.collection(outcomes=['unfinished'])
        _, collection = self.store.evidence(identity)
        first = json.loads(collection['native_evidence_index.json'])[0]['evidence_ref']['evidence_id']
        _, raw = self.store.evidence(first)
        harvest = json.loads(raw['archive/manifest.json'])
        harvest['terminal'] = {'source_id': self.source}
        harvest['trials'][0]['source_archive'] = {'source_id': self.source}
        raw['archive/manifest.json'] = encoded(harvest)
        raw['native_source_roles.json'] = encoded({'terminal': self.candidate})
        self.assertEqual(retained_trial_candidates(self.store, raw), {self.candidate: [0]})
        legacy = self.store.archive_evidence(self.candidate, dict(raw, **{'build/kernel.so': b'unrelated build'}))
        unrelated = self.store._bound_evidence(legacy)['artifacts']['build/kernel.so']
        snapshot = dict(state=dict(opportunities=[], history=[dict(attempt_id='attempt',
            collection_ref=dict(evidence_id=legacy, artifact='archive/manifest.json'))]))
        original_read = self.store._bytes
        def read_selected(blob):
            self.assertNotEqual(blob, unrelated, 'trial view must not load an unrelated build')
            return original_read(blob)
        with mock.patch.object(self.store, '_bytes', side_effect=read_selected):
            parent = snapshot_trial_history(self.store, snapshot)
            planner = snapshot_trial_history(self.store, snapshot, [legacy])
        self.assertEqual(parent, planner)
        self.assertEqual(parent['trials'][0]['outcome']['error'], 'unfinished')
        self.assertEqual(parent['trials'][0]['raw_trial_ref']['artifact'], 'archive/manifest.json')
        self.assertEqual(parent['trials'][0]['candidate_id'], rows[0]['candidate_id'])
        self.assertIsNone(parent['trials'][0]['times_ms'])
        self.assertIsNone(parent['trials'][0]['stages']['correctness'])

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
