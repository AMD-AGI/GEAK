# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native terminal evidence with disposable files and no SDK or GPU."""

import hashlib
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from interface.native_cost_controls import native_journal as journal


class NativeJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='geak-journal-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'session' / 'subagents' / 'workflows' / 'wf_test'
        self.path = self.directory / 'journal.jsonl'

    def reader(self):
        return journal.NativeJournal(self.directory, session_id='session', run_id='wf_test')

    def start(self, agent='agent-1', key=None):
        return {'type': 'started', 'key': key or 'v2:' + 'a' * 64, 'agentId': agent}

    def result(self, value=None, agent='agent-1', key=None):
        return {**self.start(agent, key), 'type': 'result', 'result': {'epoch': 42} if value is None else value}

    def write(self, rows):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

    def append(self, rows):
        with self.path.open('a') as stream:
            stream.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

    def test_missing_files_remain_pending_without_creation(self):
        reader = self.reader()
        self.assertEqual(reader.result_for('agent-1')['status'], 'pending')
        self.assertFalse(self.directory.exists())
        self.directory.mkdir(parents=True)
        self.assertEqual(reader.snapshot()['status'], 'pending')
        self.assertFalse(self.path.exists())

    def test_start_requires_its_own_full_terminal_value(self):
        self.write([self.start()])
        reader = self.reader()
        self.assertEqual(reader.result_for('agent-1')['status'], 'pending')
        self.append([self.result()])
        before = self.path.read_bytes()
        result = reader.result_for('agent-1')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['value'], {'epoch': 42})
        self.assertEqual(result['key'], self.start()['key'])
        self.assertEqual(result['entry_sha256'], hashlib.sha256(before.splitlines()[1]).hexdigest())
        self.assertEqual(json.loads(result['result_json']), result['value'])
        result['value']['epoch'] = 99
        self.assertEqual(reader.result_for('agent-1')['value'], {'epoch': 42})
        self.assertEqual(reader.result_for('different-agent')['status'], 'pending')
        self.assertEqual(self.path.read_bytes(), before)

    def test_long_unicode_results_preserve_the_raw_json_value(self):
        value = {'result': {'quote': '\" and } : , result', 'text': '🙂é' * 500}, 'n': 1.25e20}
        result = self.result(value)
        self.write([self.start()])
        raw = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        with self.path.open('a') as stream:
            stream.write('  ' + raw + '  \n')
        actual = self.reader().result_for('agent-1')
        self.assertEqual(actual['value'], value)
        self.assertEqual(actual['result_json'], json.dumps(value, ensure_ascii=False, separators=(',', ':')))
        self.assertGreater(len(actual['result_json']), 400)

    def test_result_field_can_come_first_with_an_escaped_key(self):
        self.write([self.start()])
        with self.path.open('a') as stream:
            stream.write('{"res\\u0075lt" : {"ok":true}, "type":"result", "key":"v2:'
                         + 'a' * 64 + '", "agentId":"agent-1"}\n')
        self.assertEqual(self.reader().result_for('agent-1')['result_json'], '{"ok":true}')

    def test_each_agent_gets_its_own_result_even_with_one_cache_key(self):
        self.write([self.start(), self.start('agent-2'), self.result(), self.result({'epoch': 43}, 'agent-2')])
        reader = self.reader()
        self.assertEqual(reader.result_for('agent-1')['value']['epoch'], 42)
        self.assertEqual(reader.result_for('agent-2')['value']['epoch'], 43)

    def test_partial_tail_blocks_completion_until_the_record_finishes(self):
        self.write([self.start(), self.result(), self.start('agent-2')])
        reader = self.reader()
        tail = json.dumps(self.result({'ok': True}, 'agent-2'))
        with self.path.open('a') as stream:
            stream.write(tail[:30])
        self.assertEqual(reader.result_for('agent-1')['status'], 'pending')
        with self.path.open('a') as stream:
            stream.write(tail[30:] + '\n')
        self.assertEqual(reader.result_for('agent-2')['value'], {'ok': True})

    def test_empty_journal_is_pending(self):
        self.write([])
        self.assertEqual(self.reader().snapshot()['status'], 'pending')

    def test_duplicate_or_conflicting_terminal_records_latch_error(self):
        for last in (self.result(), self.result({'epoch': 99}), {**self.start(), 'type': 'error', 'error': 'failed'}):
            with self.subTest(last=last):
                self.write([self.start(), self.result(), last])
                reader = self.reader()
                self.assertEqual(reader.result_for('agent-1')['status'], 'error')
                self.write([self.start(), self.result()])
                self.assertEqual(reader.result_for('agent-1')['status'], 'error')

    def test_native_error_does_not_complete_or_hide_other_results(self):
        self.write([self.start(), {**self.start(), 'type': 'error', 'error': 'failed'},
                    self.start('agent-2'), self.result({'ok': True}, 'agent-2')])
        reader = self.reader()
        self.assertEqual(reader.result_for('agent-1')['status'], 'error')
        self.assertEqual(reader.result_for('agent-2')['status'], 'complete')

    def test_identity_order_and_field_conflicts_refuse_completion(self):
        cases = [
            [self.result()], [self.start(), self.start()],
            [self.start(), self.result(key='v2:' + 'b' * 64)],
            [{**self.start(), 'agentId': '../agent'}], [{**self.start(), 'key': 'wrong'}],
            [{**self.start(), 'extra': True}], [self.start(), {**self.result(), 'extra': True}],
            [self.start(), {**self.result(), 'type': 'unknown'}],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                self.write(rows)
                self.assertEqual(self.reader().snapshot()['status'], 'error')

    def test_malformed_and_nonfinite_json_refuse_completion(self):
        cases = [b'[]\n', b'not JSON\n', b'\xff\n', b'\n',
                 b'{"type":"started","type":"result"}\n']
        for token in ('NaN', 'Infinity', '1e400'):
            cases.append(json.dumps(self.start()).encode() + b'\n' +
                         ('{"type":"result","key":"v2:' + 'a' * 64 +
                          '","agentId":"agent-1","result":' + token + '}\n').encode())
        cases.append(json.dumps(self.start()).encode() + b'\n' +
                     ('{"type":"result","key":"v2:' + 'a' * 64 +
                      '","agentId":"agent-1","result":{"ok":true,"ok":false}}\n').encode())
        self.directory.mkdir(parents=True)
        for raw in cases:
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                self.assertEqual(self.reader().snapshot()['status'], 'error')

    def test_completed_prefix_cannot_be_changed_or_truncated(self):
        for action in ('overwrite', 'truncate', 'overwrite-and-append'):
            with self.subTest(action=action):
                self.write([self.start(), self.result()])
                reader = self.reader()
                self.assertEqual(reader.result_for('agent-1')['status'], 'complete')
                if action == 'truncate':
                    self.write([self.start()])
                else:
                    self.write([self.start(), self.result({'epoch': 99})])
                    if action == 'overwrite-and-append':
                        self.append([self.start('agent-2')])
                self.assertEqual(reader.result_for('agent-1')['status'], 'error')

    def test_replacement_and_disappearance_refuse_completion(self):
        for action in ('replace', 'remove'):
            with self.subTest(action=action):
                self.write([self.start(), self.result()])
                reader = self.reader()
                self.assertEqual(reader.snapshot()['status'], 'complete')
                if action == 'replace':
                    replacement = self.directory / 'replacement'
                    replacement.write_bytes(self.path.read_bytes())
                    replacement.replace(self.path)
                else:
                    self.path.unlink()
                self.assertEqual(reader.result_for('agent-1')['status'], 'error')

    def test_replaced_run_directory_refuses_completion(self):
        self.write([self.start(), self.result()])
        reader = self.reader()
        self.assertEqual(reader.snapshot()['status'], 'complete')
        self.directory.rename(self.directory.with_name('old'))
        self.write([self.start(), self.result()])
        self.assertEqual(reader.snapshot()['status'], 'error')

    def test_directory_identity_is_retained_before_the_journal_exists(self):
        self.directory.mkdir(parents=True)
        reader = self.reader()
        self.assertEqual(reader.snapshot()['status'], 'pending')
        self.directory.rename(self.directory.with_name('old'))
        self.write([self.start(), self.result()])
        self.assertEqual(reader.snapshot()['status'], 'error')

    def test_directory_replacement_during_read_cannot_return_old_results(self):
        self.write([self.start(), self.result()])
        reader = self.reader()
        read = os.read
        replaced = False

        def replace_during_read(descriptor, count):
            nonlocal replaced
            value = read(descriptor, count)
            if not replaced:
                replaced = True
                self.directory.rename(self.directory.with_name('old'))
                self.write([self.start(), self.result({'epoch': 99})])
            return value

        with mock.patch.object(journal.os, 'read', side_effect=replace_during_read):
            self.assertEqual(reader.result_for('agent-1')['status'], 'error')

    def test_symlink_files_and_ancestors_are_not_followed(self):
        self.directory.mkdir(parents=True)
        target = self.root / 'target'
        target.write_text(json.dumps(self.start()) + '\n')
        self.path.symlink_to(target)
        self.assertEqual(self.reader().snapshot()['status'], 'error')
        self.path.unlink()
        self.directory.rmdir()
        self.directory.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(self.reader().snapshot()['status'], 'error')
        self.assertEqual(target.read_text(), json.dumps(self.start()) + '\n')

    def test_special_files_are_rejected_without_blocking(self):
        self.directory.mkdir(parents=True)
        os.mkfifo(self.path)
        self.assertEqual(self.reader().snapshot()['status'], 'error')
        self.path.unlink()
        self.path.mkdir()
        self.assertEqual(self.reader().snapshot()['status'], 'error')

    def test_owner_and_size_limits_refuse_completion(self):
        self.write([self.start(), self.result()])
        with mock.patch.object(journal.os, 'geteuid', return_value=os.geteuid() + 1):
            self.assertEqual(self.reader().snapshot()['status'], 'error')
        with mock.patch.object(journal, 'MAX_JOURNAL_BYTES', 1):
            self.assertEqual(self.reader().snapshot()['status'], 'error')

    def test_concurrent_append_waits_for_a_stable_snapshot(self):
        self.write([self.start(), self.result()])
        reader = self.reader()
        read = os.read
        appended = False

        def append_during_read(descriptor, count):
            nonlocal appended
            value = read(descriptor, count)
            if not appended:
                appended = True
                self.append([self.start('agent-2')])
            return value

        with mock.patch.object(journal.os, 'read', side_effect=append_during_read):
            self.assertEqual(reader.snapshot()['status'], 'pending')
        self.assertEqual(reader.result_for('agent-1')['status'], 'complete')

    def test_unstable_first_read_still_commits_its_observed_prefix(self):
        self.write([self.start(), self.result()])
        reader = self.reader()
        read = os.read
        appended = False

        def append_during_read(descriptor, count):
            nonlocal appended
            value = read(descriptor, count)
            if not appended:
                appended = True
                self.append([self.start('agent-2')])
            return value

        with mock.patch.object(journal.os, 'read', side_effect=append_during_read):
            self.assertEqual(reader.snapshot()['status'], 'pending')
        self.write([self.start(), self.result({'epoch': 99}), self.start('agent-2')])
        self.assertEqual(reader.result_for('agent-1')['status'], 'error')

    def test_short_read_and_unreadable_paths_do_not_authorize_results(self):
        self.write([self.start(), self.result()])
        with mock.patch.object(journal.os, 'read', return_value=b''):
            self.assertEqual(self.reader().snapshot()['status'], 'error')
        reader = self.reader()
        with mock.patch.object(reader, '_open_directory', side_effect=PermissionError()):
            self.assertEqual(reader.snapshot()['status'], 'error')

    def test_descriptor_identity_rejects_ambiguous_paths(self):
        for directory, session, run in [
                ('relative/session/subagents/workflows/wf_test', 'session', 'wf_test'),
                (str(self.directory), 'wrong', 'wf_test'), (str(self.directory), 'session', 'wf_other'),
                (str(self.directory) + '/../wf_test', 'session', 'wf_test'),
                (str(self.directory), '../session', 'wf_test'), (str(self.directory), 'session', '../wf_test')]:
            with self.subTest(directory=directory, session=session, run=run), self.assertRaises(journal.JournalError):
                journal.NativeJournal(directory, session_id=session, run_id=run)
        with self.assertRaises(journal.JournalError):
            self.reader().result_for('../agent')

    def test_snapshot_caller_cannot_change_a_later_result(self):
        self.write([self.start(), self.result()])
        reader = self.reader()
        snapshot = reader.snapshot()
        original = deepcopy(snapshot)
        snapshot['results']['agent-1']['value']['epoch'] = 999
        self.assertEqual(reader.snapshot(), original)


if __name__ == '__main__':
    unittest.main()
