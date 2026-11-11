# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for protected inputs and complete native runner evidence."""

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from geak.native import external_runner as runner
from geak.native.external import read_external_report
from tests.native.fixtures import evaluation_request, fixture, raw_reports, sha256


class ProtectedRunnerTests(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / 'task'
        self.contract = fixture(self.task)
        for (name, raw) in {'kernel_src/reference.txt': b'protected sibling\n', 'captures/input.bin': b'\x00synthetic fixture\xff'}.items():
            path = self.task / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            self.contract['artifacts'][name] = sha256(raw)
        self.workspace = self.root / 'workspace'
        shutil.copytree(self.task, self.workspace)
        self.source_name = self.contract['source_paths'][0]
        (self.workspace / self.source_name).write_bytes(b'# selected private candidate\n')
        self.request = evaluation_request(self.contract, self.task, self.workspace, evaluation_id='protected-copy')
        self.expected = {name: (self.workspace / name).read_bytes() for name in self.contract['artifacts']}
        self.protected_names = sorted(set(self.contract['artifacts']) - set(self.contract['source_paths']))

    @staticmethod
    def files(root):
        return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob('*') if path.is_file()}

    def checked_run(self, *, run_name='output', **options):
        output = self.root / ('native' + '-' + run_name)
        raw = raw_reports(self.request)
        calls = []
        (task_before, workspace_before) = (self.files(self.task), self.files(self.workspace))

        def process(argv, **kwargs):
            calls.append(argv)
            work = kwargs['cwd']
            for (name, expected) in self.expected.items():
                shared = options.get('shared_files') or {}
                self.assertEqual((work / name).is_symlink(), name in shared, name)
                if name in shared:
                    self.assertEqual(os.readlink(work / name), str(options['protected_root'].absolute() / name))
                    parent = (work / name).parent
                    while parent != work.parent:
                        self.assertTrue(parent.is_dir(), str(parent))
                        self.assertFalse(parent.is_symlink(), str(parent))
                        parent = parent.parent
                self.assertEqual((work / name).read_bytes(), expected, name)
            prefix = f'process_{len(calls):02d}/'
            for name in ('ut_result.json', '_runs_v2.jsonl'):
                (work / name).write_bytes(raw[prefix + name])
            kwargs['stdout'].write(b'synthetic stdout\n')
            kwargs['stderr'].write(b'synthetic stderr\n')
            return subprocess.CompletedProcess(argv, 0)
        with mock.patch.dict(os.environ, {'PATH': '/usr/bin'}, clear=True), mock.patch.object(runner.subprocess, 'run', side_effect=process):
            self.assertEqual(runner.run(self.request, self.workspace, output, **options), 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.files(self.task), task_before)
        self.assertEqual(self.files(self.workspace), workspace_before)
        evidence = {name: (output / name).read_bytes() for name in raw}
        self.assertEqual(read_external_report(self.request, evidence), read_external_report(self.request, raw))
        start = json.loads(evidence['START.json'])
        self.assertEqual(start['parent_evaluation'], json.loads(raw['START.json'])['parent_evaluation'])
        self.assertEqual(start['source_files'], {name: sha256(body) for (name, body) in self.expected.items()})
        self.assertNotIn('storage', start)
        for index in range(1, 4):
            prefix = f'process_{index:02d}/'
            receipt = json.loads(evidence[prefix + 'RECEIPT.json'])
            self.assertTrue(receipt['sources_unchanged'])
            for name in ('ut_result.json', '_runs_v2.jsonl'):
                self.assertEqual(evidence[prefix + name], raw[prefix + name])
                self.assertEqual(receipt[name + '_sha256'], sha256(raw[prefix + name]))
            for name in ('stdout.log', 'stderr.log'):
                self.assertEqual(receipt[name + '_sha256'], sha256((output / prefix / name).read_bytes()))
        return (start, json.loads((output / 'OBSERVATIONS.json').read_bytes()))

    def test_default_still_copies_complete_workspace_without_changing_inputs(self):
        self.checked_run()

    def test_protected_root_supplies_omitted_files_but_candidate_stays_private(self):
        self.assertNotEqual((self.task / self.source_name).read_bytes(), self.expected[self.source_name])
        for name in self.protected_names:
            (self.workspace / name).unlink()
        self.checked_run(protected_root=self.task)

    def test_protected_root_overrides_stale_protected_workspace_copies(self):
        for name in self.protected_names:
            (self.workspace / name).write_bytes(b'stale workspace protected copy\n')
        self.checked_run(protected_root=self.task)

    def test_none_and_empty_shared_maps_keep_regular_copy_default(self):
        for (index, shared) in enumerate((None, {})):
            with self.subTest(shared=shared):
                options = {'shared_files': shared}
                if shared is not None:
                    options['protected_root'] = self.task
                self.checked_run(run_name=f'empty-{index}', **options)

    def test_exact_shared_leaves_are_links_with_unchanged_evidence(self):
        shared = {name: self.contract['artifacts'][name] for name in ('captures/input.bin', 'kernel_src/reference.txt')}
        for name in self.protected_names:
            (self.workspace / name).unlink()
        copied = self.checked_run(run_name='copied', protected_root=self.task)
        linked = self.checked_run(run_name='linked', protected_root=self.task, shared_files=shared)
        self.assertEqual(linked, copied)

    def test_shared_map_requires_root_and_exact_registered_protected_pins(self):
        name = 'captures/input.bin'
        digest = self.contract['artifacts'][name]
        cases = (('missing-root', {name: digest}, None), ('empty-missing-root', {}, None), ('wrong-type', [(name, digest)], self.task), ('unregistered', {'captures/other.bin': digest}, self.task), ('unregistered-null', {'outside': None}, self.task), ('wrong-pin', {name: '0' * 64}, self.task), ('parent-traversal', {'../captures/input.bin': digest}, self.task), ('noncanonical', {'captures//input.bin': digest}, self.task))
        for (kind, shared, protected) in cases:
            with self.subTest(case=kind):
                output = self.root / ('native' + '-' + kind)
                with mock.patch.object(runner.subprocess, 'run') as process:
                    with self.assertRaises(ValueError):
                        runner.run(self.request, self.workspace, output, protected_root=protected, shared_files=shared)
                process.assert_not_called()
                self.assertFalse((output / 'START.json').exists())

    def test_source_cannot_be_shared_even_when_bytes_and_contract_pin_match(self):
        (self.workspace / self.source_name).write_bytes((self.task / self.source_name).read_bytes())
        request = evaluation_request(self.contract, self.task, self.workspace, evaluation_id='original-source')
        shared = {self.source_name: self.contract['artifacts'][self.source_name]}
        with mock.patch.object(runner.subprocess, 'run') as process:
            with self.assertRaises(ValueError):
                runner.run(request, self.workspace, self.root / ('native' + '-shared-source'), protected_root=self.task, shared_files=shared)
            process.assert_not_called()

    def test_post_ut_shared_link_changes_reject_even_identical_replacement_bytes(self):
        name = 'captures/input.bin'
        shared = {name: self.contract['artifacts'][name]}
        raw = raw_reports(self.request)
        (task_before, workspace_before) = (self.files(self.task), self.files(self.workspace))
        for kind in ('wrong-target', 'relative-target', 'regular-file', 'parent-symlink'):
            with self.subTest(replacement=kind):
                output = self.root / ('native' + '-' + kind)

                def process(argv, **options):
                    (work, leaf) = (options['cwd'], options['cwd'] / name)
                    self.assertTrue(leaf.is_symlink())
                    self.assertEqual(os.readlink(leaf), str(self.task / name))
                    if kind == 'parent-symlink':
                        moved = output / 'moved-captures'
                        leaf.parent.rename(moved)
                        (work / 'captures').symlink_to(moved, target_is_directory=True)
                    else:
                        leaf.unlink()
                        if kind == 'regular-file':
                            leaf.write_bytes(self.expected[name])
                        elif kind == 'relative-target':
                            leaf.symlink_to(os.path.relpath(self.task / name, leaf.parent))
                        else:
                            alternate = output / 'same-bytes.bin'
                            alternate.write_bytes(self.expected[name])
                            leaf.symlink_to(alternate)
                    self.assertEqual(leaf.read_bytes(), self.expected[name])
                    for report in ('ut_result.json', '_runs_v2.jsonl'):
                        (work / report).write_bytes(raw['process_01/' + report])
                    return subprocess.CompletedProcess(argv, 0)
                with mock.patch.dict(os.environ, {'PATH': '/usr/bin'}, clear=True), mock.patch.object(runner.subprocess, 'run', side_effect=process) as launched:
                    self.assertEqual(runner.run(self.request, self.workspace, output, protected_root=self.task, shared_files=shared), 1)
                launched.assert_called_once()
                receipt = json.loads((output / 'process_01/RECEIPT.json').read_bytes())
                self.assertEqual(receipt['returncode'], 0)
                self.assertIs(receipt['sources_unchanged'], False)
                self.assertEqual(receipt.get('artifact_errors', []), [])
                self.assertFalse((output / 'process_02').exists())
                self.assertFalse((output / 'OBSERVATIONS.json').exists())
                self.assertEqual(self.files(self.task), task_before)
                self.assertEqual(self.files(self.workspace), workspace_before)

    def test_post_ut_shared_target_is_rehashed(self):
        name = 'captures/input.bin'
        shared = {name: self.contract['artifacts'][name]}
        raw = raw_reports(self.request)
        output = self.root / ('native' + '-target-tamper')

        def process(argv, **options):
            (self.task / name).write_bytes(b'x' * len(self.expected[name]))
            for report in ('ut_result.json', '_runs_v2.jsonl'):
                (options['cwd'] / report).write_bytes(raw['process_01/' + report])
            return subprocess.CompletedProcess(argv, 0)
        try:
            with mock.patch.object(runner.subprocess, 'run', side_effect=process) as launched:
                self.assertEqual(runner.run(self.request, self.workspace, output, protected_root=self.task, shared_files=shared), 1)
            launched.assert_called_once()
            receipt = json.loads((output / 'process_01/RECEIPT.json').read_bytes())
            self.assertEqual(receipt['returncode'], 0)
            self.assertIs(receipt['sources_unchanged'], False)
            self.assertEqual(receipt.get('artifact_errors', []), [])
        finally:
            (self.task / name).write_bytes(self.expected[name])

    def test_shared_link_identity_must_survive_canonical_target_read(self):
        name = 'captures/input.bin'
        target = self.task / name
        work = self.root / ('native' + '-link-race')
        leaf = work / name
        leaf.parent.mkdir(parents=True)
        leaf.symlink_to(target)
        before = leaf.lstat()
        read_target = runner._read

        def replace_after_read(path):
            self.assertEqual(path, target)
            raw = read_target(path)
            leaf.rename(leaf.with_name('previous-link'))
            leaf.symlink_to(target)
            self.assertEqual(os.readlink(leaf), str(target))
            return raw
        with mock.patch.object(runner, '_read', side_effect=replace_after_read) as read:
            with self.assertRaisesRegex(ValueError, 'shared input link changed'):
                runner.shared_input_digest(work, name, self.task)
        read.assert_called_once_with(target)
        self.assertNotEqual(leaf.lstat().st_ino, before.st_ino)
        self.assertEqual(target.read_bytes(), self.expected[name])

    def test_bad_or_missing_protected_input_cannot_fall_back_to_workspace(self):
        name = 'harness_lib.py'
        for failure in ('tampered', 'missing'):
            with self.subTest(failure=failure):
                protected = self.root / ('native' + '-' + failure)
                shutil.copytree(self.task, protected)
                if failure == 'tampered':
                    (protected / name).write_bytes(b'changed protected helper\n')
                else:
                    (protected / name).unlink()
                before = (self.files(protected), self.files(self.workspace))
                output = self.root / ('native' + '-' + failure + '-output')
                with mock.patch.object(runner.subprocess, 'run') as process:
                    with self.assertRaises((ValueError, OSError)):
                        runner.run(self.request, self.workspace, output, protected_root=protected)
                process.assert_not_called()
                self.assertFalse((output / 'START.json').exists())
                self.assertEqual((self.files(protected), self.files(self.workspace)), before)

    def test_missing_private_candidate_cannot_fall_back_to_protected_root(self):
        (self.task / self.source_name).write_bytes(self.expected[self.source_name])
        (self.workspace / self.source_name).unlink()
        with mock.patch.object(runner.subprocess, 'run') as process:
            with self.assertRaises((ValueError, OSError)):
                runner.run(self.request, self.workspace, self.root / ('native' + '-missing-source'), protected_root=self.task)
            process.assert_not_called()


class CompiledEvidenceTests(unittest.TestCase):
    def test_retains_compiler_outputs_with_hashes_and_limited_attribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, output = root / "work", root / "output"
            work.mkdir()
            output.mkdir()
            expected = {
                "kernel.amdgcn": b"synthetic assembly",
                "kernel.ttgir": b"synthetic IR",
                "kernel.json": b'{"synthetic": true}',
            }
            for name, raw in expected.items():
                (work / name).write_bytes(raw)
            (work / "unrelated.json").write_bytes(b"{}")
            (work / "link.co").symlink_to(work / "kernel.amdgcn")

            with mock.patch.dict(os.environ, {}, clear=True):
                runner.export_compiled(output, work, 0, time.monotonic() + 5)

            index = json.loads((output / "compiled/INDEX.json").read_bytes())
            self.assertEqual({row["relative"] for row in index["files"]}, set(expected))
            self.assertEqual(index["attribution"], "created_or_changed_after_runner_start")
            self.assertIs(index["execution_proven"], False)
            self.assertEqual(index["omitted_count"], 0)
            for row in index["files"]:
                raw = expected[row["relative"]]
                self.assertEqual((output / "compiled" / row["path"]).read_bytes(), raw)
                self.assertEqual(row["sha256"], sha256(raw))
                self.assertEqual(row["size"], len(raw))


if __name__ == "__main__":
    unittest.main()
