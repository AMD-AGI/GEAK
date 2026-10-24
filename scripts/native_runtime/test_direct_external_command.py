# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for direct capture, raw-evidence replay and source/patch binding.

The only runner used here writes the existing synthetic three-process fixture;
no task module, GPU operation, container or ownership service is started.
"""

import contextlib
import difflib
import fcntl
import hashlib
import importlib.util
import io
import json
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module("direct_external_under_test", Path(__file__).with_name("direct_external_command.py"))
fixtures = load_module("direct_external_cpu_fixtures", REPO / "tests/native/fixtures.py")


class DirectExternalCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task"
        self.contract = fixtures.fixture(self.task)
        (self.task / "task.json").write_text(json.dumps(self.contract))
        self.workspace = self.root / "workspace"
        shutil.copytree(self.task, self.workspace)
        self.candidate = self.workspace / "kernel_src/module.py"
        self.candidate.write_bytes(b"# synthetic changed candidate\n")
        self.state = self.root / "state"
        self.gpu_lock = self.root / "gpu_0.lock"
        self.raw_options = {}
        runner_patch = mock.patch.object(adapter, "run_native", side_effect=self.fake_native)
        self.runner = runner_patch.start()
        self.addCleanup(runner_patch.stop)

    def fake_native(self, request, workspace, output):
        self.assertEqual(workspace, self.workspace)
        output.mkdir(parents=True, exist_ok=False)
        for name, body in fixtures.raw_reports(request, **self.raw_options).items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return 0

    def argv(self, action, identity=None, *, workspace=None, extra=()):
        args = ["--task-root", str(self.task), "--state-root", str(self.state),
                "--gpu-lock-file", str(self.gpu_lock),
                action, "--workspace", str(workspace or self.workspace)]
        if identity is not None:
            args.extend(["--candidate-id", identity["candidate_id"],
                         "--evaluation-id", identity["evaluation_id"]])
        return [*args, *extra]

    def invoke(self, action, identity=None, *, workspace=None, extra=()):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = adapter.main(self.argv(action, identity, workspace=workspace, extra=extra))
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def capture(self):
        code, rows = self.invoke("measure", extra=["--capture-only"])
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema"], "geak-native-measurement-identity-v1")
        self.assertIs(rows[0]["capture_only"], True)
        return rows[0]

    def native_root(self, identity):
        return self.state / "evaluations" / identity["evaluation_id"] / "native"

    def test_capture_freezes_source_ids_without_dispatch_and_measure_replays_once(self):
        first, second = self.capture(), self.capture()
        self.runner.assert_not_called()
        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(first["source_id"], second["source_id"])
        self.assertNotEqual(first["evaluation_id"], second["evaluation_id"])
        self.assertEqual(first["source_paths"], self.contract["source_paths"])
        self.assertRegex(first["candidate_id"], r"^[0-9a-f]{64}$")
        self.assertFalse(self.native_root(first).exists())

        code, measured = self.invoke("measure", first)
        self.assertEqual(code, 0)
        self.assertEqual(len(measured), 2)
        expected = {key: first[key] for key in ("candidate_id", "evaluation_id")}
        self.assertEqual(measured[-1]["external_evaluation"], expected)
        self.assertIsNone(measured[-1]["patch_sha256"])
        self.assertAlmostEqual(measured[-1]["primary_speedup"], 200 / 70)
        code, replayed = self.invoke("measure", first)
        self.assertEqual(code, 0)
        self.assertEqual(replayed[-1], measured[-1])
        self.runner.assert_called_once()

    def test_captured_source_bytes_and_modes_must_still_match_before_dispatch(self):
        identity = self.capture()
        original = self.candidate.read_bytes()
        for change in ("bytes", "mode"):
            with self.subTest(change=change):
                if change == "bytes":
                    self.candidate.write_bytes(b"# changed after capture\n")
                else:
                    self.candidate.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "source capture"):
                    self.invoke("measure", identity)
                self.runner.assert_not_called()
                self.candidate.write_bytes(original)
                self.candidate.chmod(0o644)
        self.assertEqual(self.invoke("measure", identity)[0], 0)
        self.runner.assert_called_once()

    def test_foreign_candidate_id_cannot_use_a_captured_evaluation(self):
        identity = self.capture()
        foreign = dict(identity, candidate_id="0" * 64)
        for action in ("measure", "read"):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "identity"):
                self.invoke(action, foreign)
        self.runner.assert_not_called()

    def test_read_recomputes_raw_evidence_and_rejects_tampering_despite_cached_claim(self):
        identity = self.capture()
        self.invoke("measure", identity)
        native = self.native_root(identity)
        (native / "OBSERVATIONS.json").write_text(json.dumps({
            "primary_speedup": 999, "external_measurement_valid": True,
            "native_reproducible_gain": True,
        }))
        code, rows = self.invoke("read", identity)
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["primary_speedup"], 200 / 70)
        self.assertEqual(rows[0]["objective"], {"kind": "external_primary", "primary_regime": "extend"})
        self.assertIs(rows[0]["external_measurement_valid"], True)

        (native / "process_03/ut_result.json").write_bytes(b"{}")
        with self.assertRaises((ValueError, KeyError)):
            self.invoke("read", identity)
        self.runner.assert_called_once()

    def test_patch_must_reconstruct_measured_source_without_changing_original(self):
        identity = self.capture()
        self.invoke("measure", identity)
        original_path = self.task / "kernel_src/module.py"
        original = original_path.read_bytes()
        patch = self.root / "candidate.diff"
        patch.write_text("".join(difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            self.candidate.read_text().splitlines(keepends=True),
            fromfile="a/kernel_src/module.py", tofile="b/kernel_src/module.py")))
        code, rows = self.invoke("read", identity, workspace=self.task, extra=["--patch", str(patch)])
        self.assertEqual(code, 0)
        self.assertEqual(rows[0]["external_evaluation"]["candidate_id"], identity["candidate_id"])
        self.assertEqual(rows[0]["patch_sha256"], hashlib.sha256(patch.read_bytes()).hexdigest())
        self.assertEqual(original_path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "patch differs"):
            self.invoke("read", identity, workspace=self.task)
        patch.write_text(patch.read_text().replace("synthetic changed candidate", "a different candidate"))
        with self.assertRaisesRegex(ValueError, "patch differs"):
            self.invoke("read", identity, workspace=self.task, extra=["--patch", str(patch)])
        self.assertEqual(original_path.read_bytes(), original)
        self.runner.assert_called_once()

    def test_git_patch_keeps_both_editable_sources_and_binary_bytes_excludes_stray_edit(self):
        binary, baseline = "kernel_src/weights.bin", "baseline_src/weights.bin"
        original_binary = b"\0original binary\xff\n"
        for root in (self.task, self.workspace):
            (root / binary).write_bytes(original_binary)
            (root / baseline).write_bytes(original_binary)
            (root / "operator-note.txt").write_text("original note\n")
        self.contract["source_paths"].append(binary)
        self.contract["original_source_map"][binary] = baseline
        for name in (binary, baseline):
            self.contract["artifacts"][name] = hashlib.sha256(original_binary).hexdigest()
        (self.task / "task.json").write_text(json.dumps(self.contract))
        self.candidate.write_bytes((self.task / "kernel_src/module.py").read_bytes())

        def git(directory, *args):
            return subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.PIPE)

        git(self.workspace, "init", "-q")
        git(self.workspace, "config", "user.name", "Saptarshi Majumder")
        git(self.workspace, "config", "user.email", "32464366+iraj465@users.noreply.github.com")
        git(self.workspace, "add", "-A")
        git(self.workspace, "commit", "-qm", "CPU fixture ORIGINAL")
        anchor = git(self.workspace, "rev-parse", "HEAD").decode().strip()
        self.candidate.write_bytes(b"# changed candidate source\n")
        changed_binary = b"\0changed binary\xfe\x80\n"
        (self.workspace / binary).write_bytes(changed_binary)
        (self.workspace / "operator-note.txt").write_text("stray tracked edit\n")
        identity = self.capture()
        self.invoke("measure", identity)
        patch = self.root / "source-paths.diff"
        patch.write_bytes(git(self.workspace, "--literal-pathspecs", "diff", "--binary",
            "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/", anchor,
            "--", *identity["source_paths"]))
        self.assertIn(b"GIT binary patch", patch.read_bytes())
        self.assertNotIn(b"operator-note.txt", patch.read_bytes())
        replay = self.root / "replay"
        shutil.copytree(self.task, replay)
        git(replay, "init", "-q")
        git(replay, "apply", "--binary", "--", str(patch))
        self.assertEqual((replay / "kernel_src/module.py").read_bytes(), self.candidate.read_bytes())
        self.assertEqual((replay / binary).read_bytes(), changed_binary)
        self.assertEqual((replay / "operator-note.txt").read_text(), "original note\n")
        code, rows = self.invoke("read", identity, workspace=self.task, extra=["--patch", str(patch)])
        self.assertEqual(code, 0)
        self.assertEqual(rows[0]["patch_sha256"], hashlib.sha256(patch.read_bytes()).hexdigest())
        self.assertEqual(rows[0]["external_evaluation"]["candidate_id"], identity["candidate_id"])
        self.assertEqual((self.task / binary).read_bytes(), original_binary)

    def test_protected_harness_change_refuses_before_dispatch(self):
        (self.workspace / "ut_main.py").write_bytes(b"# changed harness\n")
        with self.assertRaisesRegex(ValueError, "protected task artifact changed"):
            self.invoke("measure")
        self.runner.assert_not_called()

    def test_incomplete_existing_operation_is_never_relaunched(self):
        identity = self.capture()

        def interrupted(request, workspace, output):
            output.mkdir()
            (output / "START.json").write_bytes(fixtures.raw_reports(request)["START.json"])
            return 1

        self.runner.side_effect = interrupted
        with self.assertRaisesRegex(ValueError, "native UT failed"):
            self.invoke("measure", identity)
        with self.assertRaises((OSError, ValueError, KeyError)):
            self.invoke("measure", identity)
        self.runner.assert_called_once()

    def test_valid_no_gain_and_secondary_regression_remain_valid_measurements(self):
        for options, regression in (({"original": True}, False), ({"draft_regression": True}, True)):
            with self.subTest(options=options):
                self.raw_options = options
                code, rows = self.invoke("measure")
                self.assertEqual(code, 0)
                value = rows[-1]
                self.assertIs(value["external_measurement_valid"], True)
                self.assertIs(value["native_reproducible_gain"], False)
                self.assertIs(value["native_reproducible_regression"], regression)
                self.assertEqual(value["validation_status"], "no_improvement")
        self.assertEqual(self.runner.call_count, 2)

    def test_shared_gpu_lock_serializes_complete_native_operations(self):
        identities = [self.capture(), self.capture()]
        first_entered, second_entered = threading.Event(), threading.Event()
        second_attempted, release_first = threading.Event(), threading.Event()
        counts_lock = threading.Lock()
        entered = 0
        results, errors = [], []

        def blocked_native(request, workspace, output):
            nonlocal entered
            with counts_lock:
                entered += 1
                index = entered
            if index == 1:
                first_entered.set()
                if not release_first.wait(5):
                    raise RuntimeError("test did not release first synthetic runner")
            else:
                second_entered.set()
            return self.fake_native(request, workspace, output)

        def worker(index):
            try:
                if index == 1:
                    second_attempted.set()
                results.append(adapter.main(self.argv("measure", identities[index])))
            except Exception as error:
                errors.append(error)

        self.runner.side_effect = blocked_native
        threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(2)]
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                threads[0].start()
                self.assertTrue(first_entered.wait(3), "first synthetic runner did not start")
                threads[1].start()
                self.assertTrue(second_attempted.wait(3))
                self.assertFalse(second_entered.wait(0.15), "native operations overlapped under one GPU lock")
            finally:
                release_first.set()
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, [0, 0])
        self.assertTrue(second_entered.is_set())
        self.assertEqual(self.runner.call_count, 2)
        self.assertFalse((self.state / "measurement.lock").exists())

    def test_lock_timeout_does_not_start_native_measurement(self):
        identity = self.capture()
        with self.gpu_lock.open("a+b") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            with mock.patch.object(adapter, "GPU_LOCK_WAIT_SECONDS", 0.05):
                with self.assertRaisesRegex(ValueError, "GPU lock unavailable after 0.05s"):
                    self.invoke("measure", identity)
        self.runner.assert_not_called()
        code, rows = self.invoke("measure", identity)
        self.assertEqual(code, 0)
        self.assertIs(rows[-1]["external_measurement_valid"], True)
        self.runner.assert_called_once()

    def test_lock_released_after_deadline_does_not_start_measurement(self):
        identity = self.capture()
        with self.gpu_lock.open("a+b") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            def release_later():
                time.sleep(0.2)
                fcntl.flock(holder, fcntl.LOCK_UN)
            release = threading.Thread(target=release_later)
            release.start()
            try:
                with mock.patch.object(adapter, "GPU_LOCK_WAIT_SECONDS", 0.05), \
                        self.assertRaisesRegex(ValueError, "GPU lock unavailable after 0.05s"):
                    self.invoke("measure", identity)
            finally:
                release.join(3)
        self.runner.assert_not_called()

    def test_flock_utility_exit_keeps_parent_lock_until_descriptor_closes(self):
        with self.gpu_lock.open("a+b") as lock:
            adapter.acquire_gpu_lock(lock)
            held = subprocess.run(["flock", "-n", str(self.gpu_lock), "true"], check=False)
            self.assertEqual(held.returncode, 1, "utility exit must not release the parent's lock")
        released = subprocess.run(["flock", "-n", str(self.gpu_lock), "true"], check=False)
        self.assertEqual(released.returncode, 0, "closing the parent's descriptor releases the lock")

    def test_external_flock_holder_blocks_native_runner_until_release(self):
        identity = self.capture()
        lock_attempted, native_started = threading.Event(), threading.Event()
        results, errors = [], []
        real_run = subprocess.run

        def observed_lock_command(argv, **kwargs):
            if argv[0] == "flock":
                lock_attempted.set()
            return real_run(argv, **kwargs)

        def observed_native(request, workspace, output):
            native_started.set()
            return self.fake_native(request, workspace, output)

        def worker():
            try:
                results.append(adapter.main(self.argv("measure", identity)))
            except Exception as error:
                errors.append(error)

        # This separate holder models stock gpu_lock.sh's flock inode. The
        # adapter itself is invoked directly, with no recursive outer wrapper.
        holder_argv = ["flock", "--exclusive", str(self.gpu_lock), sys.executable, "-u", "-c",
                       "import sys; print('locked', flush=True); sys.stdin.buffer.read(1)"]
        thread = threading.Thread(target=worker, daemon=True)
        self.runner.side_effect = observed_native
        with subprocess.Popen(holder_argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as holder:
            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(adapter.subprocess, "run", side_effect=observed_lock_command):
                try:
                    ready, _, _ = select.select([holder.stdout], [], [], 3)
                    self.assertTrue(ready, "external flock holder did not acquire its lock")
                    self.assertEqual(holder.stdout.readline(), b"locked\n")
                    thread.start()
                    self.assertTrue(lock_attempted.wait(3), "adapter did not reach flock")
                    self.assertFalse(native_started.wait(0.15),
                                     "adapter ignored an existing holder of the GPU lock")
                finally:
                    try:
                        _, stderr = holder.communicate(input=b"release", timeout=3)
                    except subprocess.TimeoutExpired:
                        holder.kill()
                        _, stderr = holder.communicate(timeout=3)
                    if thread.ident is not None:
                        thread.join(5)
        self.assertEqual(holder.returncode, 0, stderr.decode(errors="replace"))
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [0])
        self.assertTrue(native_started.is_set())
        self.runner.assert_called_once()
        self.assertFalse((self.state / "measurement.lock").exists())


if __name__ == "__main__":
    unittest.main()
