# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test external source pins and specialist selection without a native process."""

import argparse
import json
import os
import signal
import socket
import tempfile
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from interface.native_cost_controls import hyperloom_catalog_profile as profile
from interface.native_cost_controls.native_prefix_marker import sha, wire
from interface.native_cost_controls.produce_catalog import TERMINAL

DISPATCHER_SOURCE = '''
class SpecialistSubprocessConfig:
    permission_mode: str = "bypassPermissions"
    output_format: str = "stream-json"
    extra_claude_args: tuple = ()
    mcp_config_path: str | None = None
    leaf_agents_json: str | None = None

class SpecialistSubprocessDispatcher:
    def _writable_dirs(self, workspace, worktree):
        directories = []
        if worktree is not None:
            directories.append(str(worktree))
        directories.append(str(workspace))
        for root in self.config.framework_source_roots:
            if root and Path(root).is_dir() and root not in directories:
                directories.append(root)
        return directories

    def _build_claude_cmd(self, *, system_prompt_file, system_prompt, workspace,
                          worktree, disallowed_tools=frozenset()):
        system_prompt_file.write_text(system_prompt)
        cfg = self.config
        command = [cfg.claude_executable, "--print", "--output-format", cfg.output_format,
                   "--verbose", "--permission-mode", cfg.permission_mode,
                   "--system-prompt-file", str(system_prompt_file)]
        if cfg.model:
            command.extend(["--model", cfg.model])
        if disallowed_tools:
            command.extend(["--disallowedTools", ",".join(sorted(disallowed_tools))])
        from .leaf import build_leaf_agents_json
        command.extend(["--agents", cfg.leaf_agents_json or build_leaf_agents_json()])
        if cfg.mcp_config_path:
            command.extend(["--mcp-config", cfg.mcp_config_path])
        for directory in self._writable_dirs(workspace, worktree):
            command.extend(["--add-dir", directory])
        if cfg.extra_claude_args:
            command.extend(list(cfg.extra_claude_args))
        return command
'''

LEAF_SOURCE = '''
LEAF_AGENT_NAME = "hyperloom-leaf"
LEAF_AGENT_TOOLS = ("Read", "Grep")
_LEAF_AGENT_PROMPT = "Source leaf prompt. " + BASH_KILL_SAFETY_PREAMBLE
_LEAF_AGENT_DESCRIPTION = "Source leaf description."
def build_leaf_agents_json():
    return json.dumps({LEAF_AGENT_NAME: {"description": _LEAF_AGENT_DESCRIPTION,
                      "prompt": _LEAF_AGENT_PROMPT, "tools": list(LEAF_AGENT_TOOLS)}})
'''


def request(session="synthetic-session", count=23):
    return {"model": "test-model", "max_tokens": 64000,
            "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
            "metadata": {"user_id": json.dumps({"session_id": session})},
            "system": [{"type": "text", "text": "synthetic"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "synthetic"}]}],
            "tools": [{"name": "SyntheticTool" + str(index), "description": "Synthetic definition.",
                       "input_schema": {"type": "object", "properties": {}}} for index in range(count)]}


class _SourceFixture:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "checkout"
        self.raw_sources = {
            profile.DISPATCHER: DISPATCHER_SOURCE,
            profile.LEAF: LEAF_SOURCE,
            profile.RUNNER: 'SPECIALIST_TOOL_DENYLIST = frozenset({"KillShell", "SlashCommand"})\n',
            profile.PROMPT: 'BASH_KILL_SAFETY_PREAMBLE = "Source safety text."\n',
            # This source remains bound evidence. Its unrelated code cannot execute.
            profile.FACTORY: 'raise RuntimeError("Do not execute the factory.")\n',
        }
        self.options = argparse.Namespace(prefix_count=18, model="test-model", timeout=1,
            cli=Path("/synthetic/claude"), hyperloom_source=self.source,
            hyperloom_source_manifest=self.root / "sources.json", hyperloom_framework_root=[])
        self.pin()

    def pin(self):
        hashes = {}
        for name, source in self.raw_sources.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = source.encode()
            path.write_bytes(raw)
            hashes[name] = sha(raw)
        raw = wire({"schema": profile.SOURCE_SCHEMA, "files": hashes}) + b"\n"
        self.options.hyperloom_source_manifest.write_bytes(raw)
        self.options.hyperloom_source_manifest_sha256 = sha(raw)

    def command(self):
        temporary = self.root / "capture"
        temporary.mkdir()
        return profile._source_command(self.options, temporary)


class SourceProfileTests(_SourceFixture, unittest.TestCase):
    def test_bindings_pin_all_sources_without_executing_them(self):
        result = profile.bindings(self.options)
        self.assertEqual(set(result["source_sha256"]), set(profile.SOURCE_FILES))
        self.assertEqual(result["profile_module_sha256"], sha(Path(profile.__file__).read_bytes()))
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_source_mismatch_fails_before_any_native_process(self):
        (self.source / profile.LEAF).write_text(LEAF_SOURCE + "# changed\n")
        with patch.object(profile.subprocess, "Popen") as process, \
                self.assertRaisesRegex(ValueError, "differs from the pinned manifest"):
            profile.capture(self.options)
        process.assert_not_called()

    def test_manifest_hash_must_match_independent_pin(self):
        self.options.hyperloom_source_manifest_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            profile.bindings(self.options)

    def test_missing_manifest_pin_fails_before_a_native_process(self):
        del self.options.hyperloom_source_manifest_sha256
        with patch.object(profile.subprocess, "Popen") as process, \
                self.assertRaisesRegex(ValueError, "checkout, source manifest, and manifest hash"):
            profile.capture(self.options)
        process.assert_not_called()

    def test_source_manifest_schema_must_be_supported(self):
        manifest = json.loads(self.options.hyperloom_source_manifest.read_bytes())
        manifest["schema"] = "unsupported-source-schema"
        raw = wire(manifest)
        self.options.hyperloom_source_manifest.write_bytes(raw)
        self.options.hyperloom_source_manifest_sha256 = sha(raw)
        with self.assertRaisesRegex(ValueError, "schema is unsupported"):
            profile.bindings(self.options)

    def test_source_input_byte_limit_rejects_an_oversized_manifest(self):
        self.options.hyperloom_source_manifest.write_bytes(b" " * (profile.MAX_SOURCE_BYTES + 1))
        with patch.object(profile.subprocess, "Popen") as process, \
                self.assertRaisesRegex(ValueError, "exceeds the byte limit"):
            profile.capture(self.options)
        process.assert_not_called()

    def test_missing_source_dependency_is_rejected(self):
        manifest = json.loads(self.options.hyperloom_source_manifest.read_bytes())
        del manifest["files"][profile.PROMPT]
        raw = wire(manifest)
        self.options.hyperloom_source_manifest.write_bytes(raw)
        self.options.hyperloom_source_manifest_sha256 = sha(raw)
        with self.assertRaisesRegex(ValueError, "all five"):
            profile.bindings(self.options)

    def test_source_symlink_cannot_escape_checkout(self):
        source = self.source / profile.LEAF
        outside = self.root / "outside.py"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            profile.bindings(self.options)

    def test_only_18_prefix_count_is_supported(self):
        self.options.prefix_count = 22
        with self.assertRaisesRegex(ValueError, "prefix count of 18"):
            profile.bindings(self.options)

    def test_custom_effort_is_rejected_instead_of_ignored(self):
        self.options.effort = "medium"
        with self.assertRaisesRegex(ValueError, "source default effort"):
            profile.bindings(self.options)

    def test_source_command_preserves_filters_leaf_and_default_effort(self):
        framework = self.root / "framework"
        framework.mkdir()
        self.options.hyperloom_framework_root = [framework, framework]
        command, work, session, result = self.command()
        self.assertEqual(command[0], str(self.options.cli))
        self.assertEqual(command[command.index("--disallowedTools") + 1], "KillShell,SlashCommand")
        self.assertEqual(command[command.index("--permission-mode") + 1], "bypassPermissions")
        self.assertEqual(command[command.index("--model") + 1], "test-model")
        self.assertEqual(command[command.index("--session-id") + 1], session)
        agents = json.loads(command[command.index("--agents") + 1])
        self.assertEqual(agents["hyperloom-leaf"]["tools"], ["Read", "Grep"])
        self.assertIn("Source safety text.", agents["hyperloom-leaf"]["prompt"])
        self.assertEqual([command[index + 1] for index, value in enumerate(command) if value == "--add-dir"],
                         [str(work), str(framework)])
        self.assertNotIn("--effort", command)
        self.assertFalse(result["factory_executed"])
        self.assertFalse(result["actor_executed"])
        self.assertFalse(result["leaf_agent_executed"])
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_revised_source_leaf_definition_changes_selected_profile(self):
        self.raw_sources[profile.LEAF] = LEAF_SOURCE.replace('("Read", "Grep")', '("Glob",)')
        self.pin()
        command, _, _, _ = self.command()
        agents = json.loads(command[command.index("--agents") + 1])
        self.assertEqual(agents["hyperloom-leaf"]["tools"], ["Glob"])

    def test_changed_source_effort_default_is_rejected(self):
        self.raw_sources[profile.DISPATCHER] = DISPATCHER_SOURCE.replace(
            'extra_claude_args: tuple = ()', 'extra_claude_args: tuple = ("--effort", "medium")')
        self.pin()
        with self.assertRaisesRegex(ValueError, "source defaults differ"):
            self.command()

    def test_changed_source_filter_is_rejected(self):
        self.raw_sources[profile.RUNNER] = 'SPECIALIST_TOOL_DENYLIST = frozenset({"Bash"})\n'
        self.pin()
        with self.assertRaisesRegex(ValueError, "source tool filter differs"):
            self.command()

    def test_duplicate_source_definition_is_rejected(self):
        self.raw_sources[profile.LEAF] += '\nLEAF_AGENT_NAME = "another-leaf"\n'
        self.pin()
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            self.command()

    def test_unsupported_source_filter_literal_is_rejected(self):
        self.raw_sources[profile.RUNNER] = "SPECIALIST_TOOL_DENYLIST = frozenset()\n"
        self.pin()
        with self.assertRaisesRegex(ValueError, "source literal is unsupported"):
            self.command()

    def test_unexpected_source_leaf_identity_is_rejected(self):
        self.raw_sources[profile.LEAF] = LEAF_SOURCE.replace('"hyperloom-leaf"', '"another-leaf"')
        self.pin()
        with self.assertRaisesRegex(ValueError, "leaf definitions are unsupported"):
            self.command()

    def test_selected_method_cannot_import_an_unbound_module(self):
        self.raw_sources[profile.DISPATCHER] = DISPATCHER_SOURCE.replace(
            "from .leaf import build_leaf_agents_json", "import subprocess\n        from .leaf import build_leaf_agents_json")
        self.pin()
        with self.assertRaisesRegex(ValueError, "unsupported dependency"):
            self.command()

    def test_source_command_cannot_replace_cli_or_add_effective_flags(self):
        for replacement in ('command[0] = "/other/claude"',
                            'command.extend(["--model", "different-model"])',
                            'command.extend(["--permission-mode", "default"])'):
            with self.subTest(replacement=replacement):
                self.raw_sources[profile.DISPATCHER] = DISPATCHER_SOURCE.replace(
                    "return command", replacement + "\n        return command")
                self.pin()
                temporary = self.root / ("capture-" + sha(replacement.encode())[:8])
                temporary.mkdir()
                with self.assertRaisesRegex(ValueError, "supported pinned CLI profile"):
                    profile._source_command(self.options, temporary)

    def test_framework_roots_must_exist_and_be_absolute(self):
        self.options.hyperloom_framework_root = [Path("relative")]
        with self.assertRaisesRegex(ValueError, "existing absolute"):
            profile.bindings(self.options)


class SpecialistCaptureTests(unittest.TestCase):
    def test_main_session_exports_source_selected_prefix_and_preserves_request(self):
        recording = profile._specialist_capture("synthetic-session")
        recording.validate_headers({"x-claude-code-session-id": "synthetic-session"})
        body = request()
        raw = wire(body)
        result = recording.message(raw)
        self.assertIn(TERMINAL.encode(), result)
        self.assertNotIn(b'"tool_use"', result)
        self.assertEqual(recording.catalog, wire(body["tools"][:18]) + b"\n")
        self.assertEqual(recording.request["tool_count"], 23)
        self.assertEqual(recording.request["portable_policy"], {
            "applied": True, "reason": "marked", "only_marker_bytes_changed": True})
        self.assertNotIn("raw", recording.request)
        with self.assertRaisesRegex(ValueError, "more than one"):
            recording.message(raw)

    def test_leaf_request_cannot_establish_specialist_catalog(self):
        recording = profile._specialist_capture("synthetic-session")
        with self.assertRaisesRegex(ValueError, "leaf request"):
            recording.validate_headers({"x-claude-code-session-id": "synthetic-session",
                                        "x-claude-code-agent-id": "synthetic-leaf"})
        self.assertIsNone(recording.catalog)

    def test_wrong_or_missing_session_is_rejected(self):
        for headers in ({}, {"x-claude-code-session-id": "different-session"}):
            with self.subTest(headers=headers):
                recording = profile._specialist_capture("synthetic-session")
                with self.assertRaisesRegex(ValueError, "selected native session"):
                    recording.validate_headers(headers)
                self.assertIsNone(recording.catalog)

    def test_request_body_session_must_match_the_selected_header(self):
        recording = profile._specialist_capture("synthetic-session")
        recording.validate_headers({"x-claude-code-session-id": "synthetic-session"})
        with self.assertRaisesRegex(ValueError, "request body does not match"):
            recording.message(wire(request(session="different-session")))
        self.assertIsNone(recording.catalog)

    def test_wrong_total_and_structured_output_are_rejected(self):
        cases = [request(count=18), request(count=24), request()]
        cases[-1]["tools"][-1]["name"] = "StructuredOutput"
        for body in cases:
            with self.subTest(total=len(body["tools"])):
                recording = profile._specialist_capture("synthetic-session")
                with self.assertRaisesRegex(ValueError, "23 tools without StructuredOutput"):
                    recording.message(wire(body))
                self.assertIsNone(recording.catalog)

    def test_successful_result_contains_only_terminal_text(self):
        profile._native_result(wire({"type": "result", "result": TERMINAL, "is_error": False}), TERMINAL)

    def test_non_object_native_record_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported record"):
            profile._native_result(b"[]\n", TERMINAL)

    def test_tool_calls_errors_and_duplicate_results_are_rejected(self):
        result = {"type": "result", "result": TERMINAL, "is_error": False}
        for output in (wire({**result, "is_error": True}), wire({**result, "result": "other"}),
                       wire(result) + b"\n" + wire(result),
                       wire({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Write"}]}})):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                profile._native_result(output, TERMINAL)


class _LocalCLIProcess:
    """Replace the CLI process with one HTTP exchange to the real local stub."""

    def __init__(self, command, *, mode="success", after_response=None, **options):
        self.command = command
        self.options = options
        self.mode = mode
        self.after_response = after_response
        self.pid = 999999999
        self.returncode = 0
        self.communications = []
        self.exited = False
        self.response_status = None
        self.response = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.exited = True

    def communicate(self, input=None, timeout=None):
        self.communications.append((input, timeout))
        if self.mode == "timeout":
            if len(self.communications) == 1:
                raise profile.subprocess.TimeoutExpired(self.command, timeout)
            self.returncode = -signal.SIGKILL
            return b"", b""
        if self.mode == "nonzero":
            self.returncode = 7
            return b"", b"synthetic child failure"
        if self.mode != "no_request":
            endpoint = urlsplit(self.options["env"]["ANTHROPIC_BASE_URL"])
            if endpoint.hostname != "127.0.0.1" or endpoint.scheme != "http":
                raise AssertionError("The fake CLI can contact only the local synthetic endpoint.")
            session = self.command[self.command.index("--session-id") + 1]
            body = request(session=session)
            if self.mode == "wrong_model":
                body["model"] = "unexpected-model"
            elif self.mode == "policy_decline":
                body["system"] = [{"type": "compaction", "content": "synthetic compaction"}]
            headers = {"x-claude-code-session-id": session, "Content-Type": "application/json"}
            if self.mode == "wrong_session":
                headers["x-claude-code-session-id"] = "another-session"
            connection = HTTPConnection(endpoint.hostname, endpoint.port, timeout=2)
            try:
                connection.request("POST", "/v1/messages", body=wire(body), headers=headers)
                response = connection.getresponse()
                self.response_status = response.status
                self.response = response.read()
            finally:
                connection.close()
            if self.after_response is not None:
                self.after_response()
        return wire({"type": "result", "result": TERMINAL, "is_error": False}), b""


class CaptureOrchestrationTests(_SourceFixture, unittest.TestCase):
    def run_capture(self, *, mode="success", after_response=None):
        created = []

        def fake_process(command, **options):
            process = _LocalCLIProcess(command, mode=mode, after_response=after_response, **options)
            created.append(process)
            return process

        self.created = created
        with patch.object(profile.subprocess, "Popen", side_effect=fake_process):
            return profile.capture(self.options)

    def assert_capture_resources_closed(self):
        self.assertEqual(len(self.created), 1)
        process = self.created[0]
        self.assertTrue(process.exited)
        self.assertFalse(Path(process.options["cwd"]).parent.exists())
        endpoint = urlsplit(process.options["env"]["ANTHROPIC_BASE_URL"])
        with self.assertRaises(OSError), socket.create_connection((endpoint.hostname, endpoint.port), timeout=0.2):
            pass

    def test_capture_uses_local_http_and_restores_the_callers_environment(self):
        caller = {"ANTHROPIC_BASE_URL": "https://unused.invalid", "ANTHROPIC_API_KEY": "synthetic-caller-key",
                  "HYPERLOOM_SPECIALIST_EFFORT": "medium", "CLAUDE_CODE_EFFORT_LEVEL": "low",
                  "GEAK_CLAUDE_EFFORT": "medium", "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "123",
                  "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "42", "CLAUDE_CONFIG_DIR": "/synthetic/caller-config",
                  "CUDA_VISIBLE_DEVICES": "synthetic-gpu", "HIP_VISIBLE_DEVICES": "synthetic-gpu",
                  "ROCR_VISIBLE_DEVICES": "synthetic-gpu"}
        with patch.dict(os.environ, caller, clear=True):
            recording, selected = self.run_capture()
            self.assertEqual(dict(os.environ), caller)
        process = self.created[0]
        self.assertTrue(process.options["start_new_session"])
        self.assertEqual(process.communications, [
            (b"Synthetic catalog capture. Return terminal text. Do not use any tool.\n", self.options.timeout)])
        self.assertEqual(process.options["env"]["ANTHROPIC_API_KEY"], "synthetic-local-key")
        self.assertEqual(process.options["env"]["IS_SANDBOX"], "1")
        for name in ("HYPERLOOM_SPECIALIST_EFFORT", "CLAUDE_CODE_EFFORT_LEVEL", "GEAK_CLAUDE_EFFORT",
                     "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"):
            self.assertNotIn(name, process.options["env"])
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            self.assertEqual(process.options["env"][name], "")
        self.assertEqual(process.response_status, 200)
        self.assertIn(TERMINAL.encode(), process.response)
        self.assertEqual(recording.request["tool_count"], 23)
        self.assertEqual(recording.request["prefix_count"], 18)
        self.assertTrue(recording.request["portable_policy"]["only_marker_bytes_changed"])
        self.assertEqual(selected["role"], "specialist_main")
        self.assert_capture_resources_closed()

    def test_timeout_kills_the_child_group_and_drains_its_pipes(self):
        with patch.object(profile.os, "killpg") as kill_group, \
                self.assertRaisesRegex(RuntimeError, "exceeded its time limit"):
            self.run_capture(mode="timeout")
        process = self.created[0]
        kill_group.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertEqual(len(process.communications), 2)
        self.assertEqual(process.communications[1], (None, None))
        self.assert_capture_resources_closed()

    def test_nonzero_child_exit_closes_the_server_and_workspace(self):
        with self.assertRaisesRegex(RuntimeError, "CLI failed"):
            self.run_capture(mode="nonzero")
        self.assert_capture_resources_closed()

    def test_terminal_output_alone_cannot_establish_a_catalog(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete exactly once"):
            self.run_capture(mode="no_request")
        self.assert_capture_resources_closed()

    def test_failed_http_session_cannot_establish_a_catalog(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete exactly once"):
            self.run_capture(mode="wrong_session")
        self.assertEqual(self.created[0].response_status, 400)
        self.assert_capture_resources_closed()

    def test_changed_request_model_cannot_establish_a_catalog(self):
        with self.assertRaisesRegex(RuntimeError, "changed the selected model"):
            self.run_capture(mode="wrong_model")
        self.assertEqual(self.created[0].response_status, 200)
        self.assert_capture_resources_closed()

    def test_policy_decline_cannot_establish_a_catalog(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete exactly once"):
            self.run_capture(mode="policy_decline")
        self.assertEqual(self.created[0].response_status, 400)
        self.assert_capture_resources_closed()

    def test_capture_rejects_an_invalid_marker_receipt_after_http_success(self):
        recordings = []
        make_capture = profile._specialist_capture

        def collect_capture(session_id):
            recording = make_capture(session_id)
            recordings.append(recording)
            return recording

        def invalidate_receipt():
            recordings[0].request["portable_policy"]["only_marker_bytes_changed"] = False

        with patch.object(profile, "_specialist_capture", side_effect=collect_capture), \
                self.assertRaisesRegex(RuntimeError, "portable cache policy declined"):
            self.run_capture(after_response=invalidate_receipt)
        self.assertEqual(self.created[0].response_status, 200)
        self.assert_capture_resources_closed()

    def test_repinning_source_during_capture_rejects_the_catalog(self):
        def repin():
            self.raw_sources[profile.LEAF] += "\n# Source changed during the synthetic exchange.\n"
            self.pin()

        with self.assertRaisesRegex(RuntimeError, "source changed during the synthetic capture"):
            self.run_capture(after_response=repin)
        self.assert_capture_resources_closed()


if __name__ == "__main__":
    unittest.main()
