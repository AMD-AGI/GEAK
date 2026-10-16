"""CPU protocol fixtures. These never invoke a model, Codex, or a GPU tool."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# The same CPU suite runs against Oct 15 and the port. Only module paths and
# the missing snapshot prompt location are adapted.
import os
from functools import partial

if os.environ.get("EVO_ENGINEER_PARITY") == "original":
    import geak_evo.engineer as engineer_module
    from geak_evo.jobs import job_digest
    from geak_evo.research_state import ResearchStore
else:
    from geak_evo.engineer import runner as engineer_module
    from geak_evo.evaluate.jobs_v1 import job_digest
    from geak_evo.archive.store import ResearchStore

from engineer_job_fixture import fixture
from model_profile_fixture import profile

EngineerError = engineer_module.EngineerError
codex_command = engineer_module.codex_command
file_ref = engineer_module.file_ref
inspect_events = engineer_module.inspect_events
recover_turn = engineer_module.recover_turn
source_reference = engineer_module.source_reference
validate_request = engineer_module.validate_request
run_turn = partial(engineer_module.run_turn,
    prompt_path=Path(__file__).resolve().parents[2] / "geak_evo/prompts/engineer.md")


THREAD = "01990000-0000-7000-8000-000000000001"
NEXT_THREAD = "01990000-0000-7000-8000-000000000002"


class EngineerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = profile(self.root)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.attempts = self.root / "attempts"
        self.attempts.mkdir()
        job, files = fixture()
        for name, content in files.items():
            path = self.artifacts / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.job_path = self.root / "job.json"
        self.job_path.write_text(json.dumps(job))
        self.job_id = job_digest(job)
        self.epoch = "cpu-fixture-epoch-1"
        self.paths = ["implementation.py"]
        (self.workspace / self.paths[0]).write_text("answer = 1\n")
        self.store = ResearchStore(self.root / "store", allowed_epochs=[(self.job_id, self.epoch)],
                                   source_paths={self.job_id: self.paths})
        source = self.capture(str(self.workspace))
        candidate = self.store.candidate(self.job_id, self.epoch, source)
        self.note = self.root / "note.md"
        self.note.write_text("CPU fixture only; next question is whether the input was delivered.\n")
        self.request = {
            "schema_version": 1, "direction_id": "fixture-direction",
            "job": {"path": str(self.job_path), "job_id": self.job_id,
                    "artifact_root": str(self.artifacts)},
            "measurement_epoch_id": self.epoch, "workspace": str(self.workspace),
            "anchor": None, "tip": source_reference(self.store, candidate),
            "donors": [], "read_only": [file_ref(self.note)],
            "next_question": "Inspect the delivered job and record the next CPU test.",
            "budget": {"wall_seconds": 5, "max_tokens": None},
            "continuation": {"mode": "fresh", "prior_receipt": None},
        }

    def capture(self, workspace):
        return self.store.capture_source(workspace, self.paths)

    def execute(self, body, request=None, *, settled=True):
        script = self.root / "fixture.py"
        script.write_text("import json, pathlib, sys, time\n"
                          "prompt = sys.stdin.read()\n"
                          "assignment = json.loads(prompt.split('Host assignment (JSON):\\n', 1)[1])\n"
                          "assert assignment['job']['job_id'] == " + repr(self.job_id) + "\n" + body)

        def launch(command, **kwargs):
            self.assertEqual(command[0], "codex")
            self.assertFalse(kwargs["shell"])
            self.assertTrue(kwargs["start_new_session"])
            return subprocess.Popen([sys.executable, "-B", str(script)] + command[1:], **kwargs)

        def settle(process, attempt):
            # This trusted test fixture creates no descendants or GPU resources.
            process.wait(timeout=3)
            return {"settled": settled, "scope": "CPU fixture child only; no worker isolation qualification"}

        return run_turn(request or self.request, self.attempts, backend_options={"model_profile": self.profile},
                        launch_worker=launch, settle_worker=settle, capture_tip=self.capture)

    def events(self, thread=THREAD, usage=None):
        if usage is None:
            usage = {"input_tokens": 17, "cached_input_tokens": 5, "output_tokens": 9}
        events = [{"type": "thread.started", "thread_id": thread}, {"type": "turn.started"},
                  {"type": "turn.completed", "usage": usage}]
        return "\n".join("print(" + repr(json.dumps(event)) + ", flush=True)" for event in events) + "\n"

    def continue_from(self, result, mode):
        request = copy.deepcopy(self.request)
        source = result["receipt"]["captured_tip_source_id"]
        request["tip"] = source_reference(self.store, self.store.candidate(self.job_id, self.epoch, source))
        request["continuation"] = {"mode": mode, "prior_receipt": result["receipt_ref"]}
        return request

    def test_live_launch_is_refused_without_boundary(self):
        with self.assertRaisesRegex(EngineerError, "unqualified"):
            run_turn(self.request, self.attempts)
        self.assertEqual(list(self.attempts.iterdir()), [])

    def test_named_permissions_replace_legacy_sandbox_and_preserve_resume(self):
        first = self.execute(self.events())
        request = self.continue_from(first, "resume")
        command = codex_command(request, self.root / "report.md", model_profile=self.profile,
                                permission_profile="engineer-cpu", config_overrides={
                                    "permissions.engineer-cpu.extends": ":workspace",
                                    "permissions.engineer-cpu.filesystem": {":root": "deny", ":minimal": "read"},
                                    "permissions.engineer-cpu.network.enabled": False,
                                    "features.multi_agent": False,
                                    "shell_environment_policy.inherit": "none",
                                    "shell_environment_policy.set": {"PYTHONDONTWRITEBYTECODE": "1"},
                                })
        self.assertNotIn("--sandbox", command)
        self.assertIn("--strict-config", command)
        self.assertIn('default_permissions="engineer-cpu"', command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        self.assertIn('permissions.engineer-cpu.filesystem={":minimal"="read",":root"="deny"}', command)
        self.assertIn('features.multi_agent=false', command)
        self.assertEqual(command[-3:], ["resume", THREAD, "-"])
        legacy = codex_command(self.request, self.root / "legacy.md", model_profile=self.profile)
        self.assertEqual(legacy[legacy.index("--sandbox") + 1], "workspace-write")

    def test_strict_config_exception_is_explicit_and_keeps_profile_controls(self):
        command = codex_command(self.request, self.root / "report.md", model_profile=self.profile,
                                permission_profile="engineer-cpu", strict_config=False)
        self.assertNotIn("--strict-config", command)
        self.assertIn("--ignore-rules", command)
        self.assertIn('default_permissions="engineer-cpu"', command)
        self.assertIn('approval_policy="never"', command)
        with self.assertRaisesRegex(EngineerError, "compatibility exception"):
            codex_command(self.request, self.root / "report.md", model_profile=self.profile, strict_config=False)

    def test_config_overrides_cannot_replace_policy_or_carry_authentication(self):
        overrides = [
            {"sandbox_mode": "danger-full-access"},
            {"default_permissions": ":danger-full-access"},
            {"approval_policy": "on-request"},
            {"model_reasoning_effort": "none"},
            {"model_providers.amd.http_headers": {"api-key": "synthetic-secret"}},
            {"shell_environment_policy.set": {"OPENAI_API_KEY": "synthetic-secret"}},
            {"features.unreviewed_provider_logging": True},
            {"orchestrator": {"mcp": {"enabled": False}, "skills": {"enabled": False}}},
            {"permissions.other.filesystem": {":root": "write"}},
        ]
        for values in overrides:
            with self.subTest(key=next(iter(values))):
                with self.assertRaisesRegex(EngineerError, "unsupported or credential-bearing"):
                    codex_command(self.request, self.root / "report.md", model_profile=self.profile,
                                  permission_profile="engineer-cpu", config_overrides=values)
        with self.assertRaisesRegex(EngineerError, "unsupported or credential-bearing"):
            codex_command(self.request, self.root / "report.md", model_profile=self.profile,
                          config_overrides={"permissions.engineer-cpu.extends": ":workspace"})

    def test_real_source_capture_survives_missing_report(self):
        result = self.execute("pathlib.Path('implementation.py').write_text('answer = 2\\n')\n" + self.events())
        receipt = result["receipt"]
        self.assertTrue(receipt["transport_complete"])
        self.assertEqual(receipt["report_status"], "missing")
        self.assertNotEqual(receipt["captured_tip_source_id"], self.request["tip"]["source_id"])
        self.assertEqual(receipt["usage"]["input_tokens"], 17)
        self.assertIsNone(receipt["monetary_cost"])
        self.store.materialize(receipt["captured_tip_source_id"], self.root / "retained-tip")
        self.assertEqual((self.root / "retained-tip" / "implementation.py").read_text(), "answer = 2\n")
        self.assertEqual(json.loads(Path(result["receipt_ref"]["path"]).read_text()), receipt)

    def test_resume_exact_session_and_handoff_new_identity(self):
        first = self.execute(self.events())
        resume = self.continue_from(first, "resume")
        command = codex_command(resume, self.root / "out.md", model_profile=self.profile)
        self.assertEqual(command[-3:], ["resume", THREAD, "-"])
        self.assertNotIn("--last", command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        second = self.execute(self.events(), resume)
        self.assertTrue(second["receipt"]["transport_complete"])
        handoff = self.continue_from(second, "handoff")
        self.assertNotIn("resume", codex_command(handoff, self.root / "out.md", model_profile=self.profile))
        third = self.execute(self.events(NEXT_THREAD), handoff)
        self.assertTrue(third["receipt"]["transport_complete"])
        self.assertEqual(third["receipt"]["request"]["continuation"]["prior_receipt"], second["receipt_ref"])
        wrong = self.execute(self.events(THREAD), self.continue_from(third, "resume"))
        self.assertFalse(wrong["receipt"]["identity_valid"])

    def test_stale_receipt_is_rejected_even_when_source_is_unchanged(self):
        first = self.execute(self.events())
        resume = self.continue_from(first, "resume")
        self.execute(self.events(), resume)
        with self.assertRaisesRegex(EngineerError, "latest workspace/conversation"):
            self.execute(self.events(), resume)

    def test_registry_symlink_is_rejected(self):
        unrelated = self.root / "other"
        unrelated.mkdir()
        (self.workspace.parent / ".geak-engineer-owners").symlink_to(unrelated, target_is_directory=True)
        with self.assertRaisesRegex(EngineerError, "private host directory"):
            self.execute(self.events())

    def test_unsettled_owner_blocks_a_different_attempt_root(self):
        result = run_turn(self.request, self.attempts, backend_options={"model_profile": self.profile},
                          launch_worker=lambda command, **kw: subprocess.Popen([sys.executable, "-B", "-c", "pass"], **kw),
                          settle_worker=lambda process, attempt: {"settled": False, "reason": "fixture uncertainty"},
                          capture_tip=self.capture)
        self.assertIsNone(result["receipt"]["captured_tip_source_id"])
        other = self.root / "different-attempt-root"
        other.mkdir()
        with self.assertRaisesRegex(EngineerError, "unsettled predecessor"):
            run_turn(self.request, other, backend_options={"model_profile": self.profile}, launch_worker=lambda *a, **k: self.fail("must not launch"),
                     settle_worker=lambda *a: {"settled": True}, capture_tip=self.capture)

    @staticmethod
    def verified_recovery(receipt, attempt):
        # The original fixture has already waited for its only CPU child. This
        # callback stands for independent host evidence, never a live qualifier.
        return {"settled": True, "process_reaped": True, "scope": "Known CPU fixture child only"}

    def test_later_positive_settlement_appends_receipt_and_resumes_exact_tip(self):
        first = self.execute("pathlib.Path('implementation.py').write_text('answer = 8\\n')\n" + self.events(), settled=False)
        original_bytes = Path(first["receipt_ref"]["path"]).read_bytes()
        self.assertIsNone(first["receipt"]["captured_tip_source_id"])
        recovered = recover_turn(first["receipt_ref"], verify_recovery=self.verified_recovery, capture_tip=self.capture)
        self.assertEqual(Path(first["receipt_ref"]["path"]).read_bytes(), original_bytes)
        self.assertEqual(recovered["receipt"]["recovery_of"], first["receipt_ref"])
        self.assertEqual(recovered["receipt"]["attempt_id"], first["receipt"]["attempt_id"])
        self.assertEqual(recovered["receipt"]["request"], first["receipt"]["request"])
        self.assertEqual(recovered["receipt"]["usage"], first["receipt"]["usage"])
        self.assertIsNone(recovered["receipt"]["monetary_cost"])
        self.assertTrue(recovered["receipt"]["transport_complete"])
        self.assertNotEqual(recovered["receipt"]["captured_tip_source_id"], self.request["tip"]["source_id"])
        same = recover_turn(first["receipt_ref"], verify_recovery=lambda *a: self.fail("already committed"),
                            capture_tip=lambda *a: self.fail("already captured"))
        self.assertEqual(same, recovered)
        second = self.execute(self.events(), self.continue_from(recovered, "resume"))
        self.assertTrue(second["receipt"]["transport_complete"])
        with self.assertRaisesRegex(EngineerError, "stale workspace attempt"):
            recover_turn(first["receipt_ref"], verify_recovery=self.verified_recovery, capture_tip=self.capture)

    def test_recovery_requires_positive_host_settlement_before_capture(self):
        first = self.execute(self.events(), settled=False)
        owner_path = next((self.root / ".geak-engineer-owners").glob("*.owner.json"))
        owner_bytes = owner_path.read_bytes()
        for verdict in (True, {"settled": True}, {"settled": False, "process_reaped": True, "scope": "fixture"},
                        {"settled": True, "process_reaped": False, "scope": "fixture"}):
            with self.subTest(verdict=verdict), self.assertRaisesRegex(EngineerError, "positive host"):
                recover_turn(first["receipt_ref"], verify_recovery=lambda *a: verdict,
                             capture_tip=lambda *a: self.fail("must not capture uncertain source"))
        self.assertEqual(owner_path.read_bytes(), owner_bytes)
        self.assertEqual(list(Path(first["attempt_dir"]).glob("recovery-*.json")), [])

    def test_recovery_replays_append_when_owner_commit_was_interrupted(self):
        first = self.execute(self.events(), settled=False)
        module = engineer_module
        original_write = module._write
        def crash_owner(path, value):
            if path.name.endswith(".owner.json"):
                raise KeyboardInterrupt()
            return original_write(path, value)
        with mock.patch.object(module, "_write", side_effect=crash_owner), self.assertRaises(KeyboardInterrupt):
            recover_turn(first["receipt_ref"], verify_recovery=self.verified_recovery, capture_tip=self.capture)
        appended = next(Path(first["attempt_dir"]).glob("recovery-*.json"))
        appended_ref = file_ref(appended)
        recovered = recover_turn(first["receipt_ref"], verify_recovery=self.verified_recovery, capture_tip=self.capture)
        self.assertEqual(recovered["receipt_ref"], appended_ref)
        self.assertEqual(len(list(Path(first["attempt_dir"]).glob("recovery-*.json"))), 1)

    def test_leftover_owner_staging_file_does_not_block_recovery(self):
        first = self.execute(self.events(), settled=False)
        owner_path = next((self.root / ".geak-engineer-owners").glob("*.owner.json"))
        leftover = owner_path.with_name(owner_path.name + ".tmp")
        leftover.write_bytes(b"partial prior owner write")
        recovered = recover_turn(first["receipt_ref"], verify_recovery=self.verified_recovery, capture_tip=self.capture)
        self.assertEqual(json.loads(owner_path.read_bytes())["receipt_ref"], recovered["receipt_ref"])
        self.assertEqual(leftover.read_bytes(), b"partial prior owner write")

    def test_partial_original_receipt_recovers_without_inventing_completion(self):
        module = engineer_module
        original_write = module._write
        def crash_final_receipt(path, value):
            if path.name == "receipt.json" and "transport_complete" in value:
                raise KeyboardInterrupt()
            return original_write(path, value)
        with mock.patch.object(module, "_write", side_effect=crash_final_receipt), self.assertRaises(KeyboardInterrupt):
            self.execute(self.events(), settled=False)
        path = next(self.attempts.glob("*/receipt.json"))
        original_ref = file_ref(path)
        self.assertEqual(json.loads(path.read_bytes())["process_status"], "running")
        recovered = recover_turn(original_ref, verify_recovery=self.verified_recovery, capture_tip=self.capture)
        self.assertTrue(recovered["receipt"]["settlement"]["settled"])
        self.assertTrue(recovered["receipt"]["process_reaped"])
        self.assertFalse(recovered["receipt"]["transport_complete"])
        self.assertIsNone(recovered["receipt"]["exit_code"])
        self.assertIsNotNone(recovered["receipt"]["captured_tip_source_id"])
        self.assertEqual(file_ref(path), original_ref)

    def test_cancellation_stops_owned_child_and_writes_final_receipt(self):
        class InterruptOnce(subprocess.Popen):
            interrupted = False

            def wait(self, *args, **kwargs):
                if not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt()
                return super().wait(*args, **kwargs)

        children = []

        def launch(command, **kwargs):
            child = InterruptOnce([sys.executable, "-B", "-c", "import time; time.sleep(30)"], **kwargs)
            children.append(child)
            return child

        with self.assertRaises(KeyboardInterrupt):
            run_turn(self.request, self.attempts, backend_options={"model_profile": self.profile}, launch_worker=launch,
                     settle_worker=lambda process, attempt: {"settled": process.returncode is not None},
                     capture_tip=self.capture)
        self.assertIsNotNone(children[0].returncode)
        path = next(self.attempts.glob("*/receipt.json"))
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt["process_status"], "interrupted")
        self.assertTrue(receipt["settlement"]["settled"])
        self.assertIsNotNone(receipt["captured_tip_source_id"])

    def test_source_manifest_and_candidate_are_readable_and_bound(self):
        manifest = Path(self.request["tip"]["manifest"]["path"])
        source = json.loads(manifest.read_text())
        blob = manifest.parent / source["files"][0]["blob_id"]
        self.assertEqual(blob.read_text(), "answer = 1\n")
        blob.chmod(0o600)
        blob.write_text("archive corruption\n")
        with self.assertRaisesRegex(EngineerError, "immutable reference changed"):
            validate_request(self.request)

    def test_anchor_requires_current_epoch_but_donor_remains_historical(self):
        other = ResearchStore(self.store.root, allowed_epochs=[(self.job_id, "historical-epoch")],
                              source_paths={self.job_id: self.paths})
        identity = other.candidate(self.job_id, "historical-epoch", self.request["tip"]["source_id"])
        source = source_reference(other, identity)
        request = copy.deepcopy(self.request)
        request["anchor"] = source
        with self.assertRaisesRegex(EngineerError, "anchor candidate belongs to a different"):
            validate_request(request)
        request["anchor"] = None
        request["donors"] = [source]
        self.assertEqual(validate_request(request)["donors"], [source])

    def test_no_silent_handoff_or_unsettled_predecessor(self):
        first = self.execute(self.events())
        same = self.execute(self.events(), self.continue_from(first, "handoff"))
        self.assertFalse(same["receipt"]["identity_valid"])
        prior_path = Path(first["receipt_ref"]["path"])
        altered = json.loads(prior_path.read_text())
        altered["settlement"]["settled"] = False
        prior_path.write_text(json.dumps(altered))
        resume = self.continue_from(first, "resume")
        resume["continuation"]["prior_receipt"] = file_ref(prior_path)
        with self.assertRaisesRegex(EngineerError, "not settled"):
            validate_request(resume)

    def test_resume_cannot_replace_exact_harvested_tip(self):
        first = self.execute("pathlib.Path('implementation.py').write_text('answer = 3\\n')\n" + self.events())
        resume = self.continue_from(first, "resume")
        resume["tip"] = self.request["tip"]
        with self.assertRaisesRegex(EngineerError, "exact harvested tip"):
            validate_request(resume)

    def test_exit_zero_is_not_a_terminal_event(self):
        result = self.execute("print(" + repr(json.dumps({"type": "thread.started", "thread_id": THREAD})) + ")\n")
        receipt = result["receipt"]
        self.assertEqual(receipt["exit_code"], 0)
        self.assertFalse(receipt["transport_complete"])
        self.assertIsNone(receipt["usage"])
        self.assertEqual(receipt["usage_status"], "unknown")
        self.assertIsNotNone(receipt["captured_tip_source_id"])

    def test_nonzero_exit_keeps_usage_but_is_not_complete(self):
        receipt = self.execute(self.events() + "sys.exit(7)\n")["receipt"]
        self.assertEqual(receipt["exit_code"], 7)
        self.assertEqual(receipt["usage"]["output_tokens"], 9)
        self.assertFalse(receipt["transport_complete"])

    def test_missing_and_invalid_usage_are_not_zero(self):
        event_path = self.root / "events.jsonl"
        for usage in (None, {"input_tokens": True, "cached_input_tokens": 0, "output_tokens": 0},
                      {"input_tokens": 0, "cached_input_tokens": 1, "output_tokens": 0}):
            event_path.write_text(json.dumps({"type": "thread.started", "thread_id": THREAD}) + "\n" +
                                  json.dumps({"type": "turn.completed", "usage": usage}) + "\n")
            self.assertIsNone(inspect_events(event_path)["usage"])
        zero = self.execute(self.events(usage={"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}))
        self.assertEqual(zero["receipt"]["usage"]["input_tokens"], 0)

    def test_duplicate_terminals_are_not_double_counted(self):
        receipt = self.execute(self.events() + self.events())["receipt"]
        self.assertFalse(receipt["identity_valid"])
        self.assertIsNone(receipt["usage"])
        self.assertIn("terminal_event_missing_or_duplicated", receipt["protocol_errors"])

    def test_timeout_reaps_owned_process_and_preserves_attempt(self):
        request = copy.deepcopy(self.request)
        request["budget"]["wall_seconds"] = 0.05
        result = self.execute("pathlib.Path('implementation.py').write_text('unfinished = True\\n')\ntime.sleep(30)\n", request)
        receipt = result["receipt"]
        self.assertEqual(receipt["process_status"], "timed_out")
        self.assertTrue(receipt["process_reaped"])
        self.assertTrue(receipt["settlement"]["settled"])
        self.assertFalse(receipt["transport_complete"])
        self.assertIsNone(receipt["usage"])
        self.assertIsNotNone(receipt["captured_tip_source_id"])

    def test_stdout_stderr_are_drained_without_pipe_deadlock(self):
        receipt = self.execute("sys.stderr.write('x' * 200000)\n" + self.events())["receipt"]
        self.assertTrue(receipt["transport_complete"])

    def test_frozen_input_mutation_is_recorded(self):
        body = "pathlib.Path(" + repr(str(self.note)) + ").write_text('changed by CPU fixture')\n"
        receipt = self.execute(body + self.events())["receipt"]
        self.assertFalse(receipt["immutable_inputs_unchanged"])
        self.assertFalse(receipt["transport_complete"])
        self.assertIsNotNone(receipt["captured_tip_source_id"])

    def test_invalid_events_and_large_records_are_explicit(self):
        path = self.root / "bad.jsonl"
        path.write_bytes(b'not-json\n{"type":')
        self.assertIn("invalid_jsonl_event", inspect_events(path)["protocol_errors"])
        path.write_bytes(b"x" * 100)
        self.assertIn("oversized_event", inspect_events(path, max_line_bytes=20)["protocol_errors"])

    def test_wrong_workspace_and_changed_reference_refuse_before_launch(self):
        (self.workspace / "implementation.py").write_text("different = True\n")
        with self.assertRaisesRegex(EngineerError, "supplied tip"):
            self.execute(self.events())
        self.note.write_text("tampered\n")
        with self.assertRaisesRegex(EngineerError, "immutable reference changed"):
            validate_request(self.request)


if __name__ == "__main__":
    unittest.main()
