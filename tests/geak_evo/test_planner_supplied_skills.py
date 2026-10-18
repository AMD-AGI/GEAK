"""Supplied skill navigation uses frozen references, never inline bodies."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from geak_evo.planner import native


class SuppliedSkillPromptTests(unittest.TestCase):
    def test_no_skill_leaves_prompt_unchanged(self):
        index = [{'original_ref': {'path': '/inputs/context.md'},
                  'readable_ref': {'path': '/public/replicas/context', 'sha256': 'a' * 64}}]
        self.assertEqual(native._supplied_skill_prompt(index), '')
        self.assertEqual(native._supplied_skill_prompt([]), '')

    def test_readable_path_hash_and_relative_lookup_without_body(self):
        body = b'PRIVATE SKILL BODY: inspect hardware evidence carefully.'
        digest = hashlib.sha256(body).hexdigest()
        index = [{'original_ref': {'path': '/inputs/evidence/SKILL.md', 'sha256': digest},
                  'readable_ref': {'path': '/public/replicas/' + digest, 'sha256': digest}},
                 {'original_ref': {'path': '/inputs/evidence/references/guide.md'},
                  'readable_ref': {'path': '/public/replicas/guide', 'sha256': 'b' * 64}}]
        with mock.patch.object(native, '_read_ref', side_effect=AssertionError('body read')):
            prompt = native._supplied_skill_prompt(index)
        self.assertIn('/public/replicas/' + digest, prompt)
        self.assertIn('sha256: ' + digest, prompt)
        self.assertIn('/inputs/evidence/SKILL.md', prompt)
        self.assertIn("skill's original directory", prompt)
        self.assertIn('READ_INDEX.json original_ref paths', prompt)
        self.assertNotIn(body.decode(), prompt)

    def test_actual_freeze_and_request_preserve_skill_asset_aliases(self):
        from test_shared_host import SharedHostTests
        from geak_evo.archive.refs import file_ref

        fixture = SharedHostTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        supplied = fixture.root / 'supplied'
        supplied.mkdir()
        refs = []
        body = 'SKILL BODY MUST REMAIN FILE CONTENT ONLY\n'
        for name in ('SKILL.md', 'references/guide.md', 'references/limits.md',
                     'scripts/read.py', 'examples/example.md'):
            path = supplied / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(body if name == 'SKILL.md' else 'Supplied reference: ' + name)
            refs.append(file_ref(path))
        unrelated = fixture.root / 'unrelated.md'
        unrelated.write_text('Unrelated supplied context remains available through ordinary copies.')
        sibling = fixture.root / 'supplied-other'
        sibling.mkdir()
        (sibling / 'guide.md').write_text('A prefix match is not a skill descendant.')
        unrelated_refs = [file_ref(unrelated), file_ref(sibling / 'guide.md')]
        # Duplicate declarations must not duplicate original-path aliases.
        supplied_refs = refs + [refs[0], refs[1]] + unrelated_refs
        search = fixture.prepare(common_refs=supplied_refs)
        with self.assertRaises(native.NativePlannerPending):
            search.run(max_steps=1)
        self.assertEqual(len(fixture.transport.launches), 1)
        prompt = fixture.transport.launches[0]['prompt']
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
        index_path = next(fixture.root.glob('planner/calls/*/public/READ_INDEX.json'))
        index = json.loads(index_path.read_text())
        aliases = {row['original_ref']['path']: row['readable_ref'] for row in index}
        skill_paths = {ref['path'] for ref in refs}
        self.assertEqual(sum(row['original_ref']['path'] in skill_paths for row in index), 5)
        self.assertFalse(any(ref['path'] in aliases for ref in unrelated_refs))
        for ref in refs:
            readable = aliases[ref['path']]
            self.assertEqual(readable['sha256'], ref['sha256'])
            self.assertEqual(Path(readable['path']).read_bytes(), Path(ref['path']).read_bytes())
        self.assertIn(aliases[refs[0]['path']]['path'], prompt)
        self.assertIn('sha256: ' + refs[0]['sha256'], prompt)
        self.assertIn('READ_INDEX.json original_ref paths', prompt)
        self.assertNotIn(body.strip(), prompt)
        # Keep the same supplied-input count/bytes while removing only skill discovery.
        # The real freeze must add exactly five aliases, independent of archive blobs.
        ordinary = supplied / 'ordinary.md'
        ordinary.write_text(body)
        ordinary_ref = file_ref(ordinary)
        search.planner.read_only = [ordinary_ref if ref == refs[0] else ref for ref in supplied_refs]
        comparison = fixture.root / 'no-skill-comparison'
        comparison.mkdir()
        for name in ('intent.json', 'context.json'):
            (comparison / name).write_bytes((index_path.parent.parent / name).read_bytes())
        context = json.loads((index_path.parent.parent / 'context.json').read_text())
        _, plain_index, _ = search.planner._freeze(context, comparison)
        self.assertEqual(len(index) - len(plain_index), 5)
        objects = str(fixture.fixture.store.root / 'objects') + '/'
        self.assertEqual([row['original_ref'] for row in index if row['original_ref']['path'].startswith(objects)],
                         [row['original_ref'] for row in plain_index if row['original_ref']['path'].startswith(objects)])
