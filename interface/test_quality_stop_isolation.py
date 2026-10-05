# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test actor isolation contracts with synthetic OS records and local files.

These tests start no Docker container and expose no GPU device. Separate native
probe receipts establish the measured OS behavior. The cases here protect the
path, lifecycle, source, and refusal contracts against later code changes.
"""

import ast
import asyncio
import builtins
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interface.native_cost_controls import quality_stop_isolation as isolation
from interface.native_cost_controls.quality_stop_controller import StopRejected

SESSION = "12345678-1234-1234-1234-123456789abc"
CONTAINER = "a" * 64
IMAGE = "sha256:" + "b" * 64
INIT_PID = 2000000001
CLI_PID = 2000000002


@dataclass
class Options:
    cwd: str
    session_id: str | None = SESSION
    setting_sources: list = field(default_factory=list)
    mcp_servers: dict = field(default_factory=dict)
    tools: list = field(default_factory=list)
    allowed_tools: list = field(default_factory=list)
    disallowed_tools: list = field(default_factory=list)
    strict_mcp_config: bool = False
    cli_path: str | None = None
    env: dict = field(default_factory=dict)


class SyntheticBoundary(isolation.DockerActorBoundary):
    """Replace only external OS records. Keep the actual contract checks."""

    def __init__(self, *, fixture, view, **kwargs):
        self.fixture = fixture
        self.view = deepcopy(view)
        self.commands = []
        self.extra_fds = []
        self.processes = {
            INIT_PID: {"pid": INIT_PID, "start_ticks": "1", "exe_sha256": "1" * 64,
                       "argv": ["/usr/bin/sleep", "infinity"]},
        }
        self.top_override = None
        self.fail_unpause = False
        self._sync_pids()
        super().__init__(**kwargs)

    def _sync_pids(self):
        (self.fixture.root / "cgroup/cgroup.procs").write_text("\n".join(map(str, self.processes)) + "\n")

    def _docker(self, *arguments):
        self.commands.append(arguments)
        if arguments[0] == "inspect":
            return json.dumps([self.view]).encode()
        if arguments[0] == "top":
            return self.top_override if self.top_override is not None else (
                "PID\n" + "\n".join(map(str, self.processes)) + "\n").encode()
        if arguments[0] in {"pause", "unpause"}:
            if arguments[0] == "unpause" and self.fail_unpause:
                raise StopRejected("synthetic_unpause_failed")
            paused = arguments[0] == "pause"
            self.view["State"]["Paused"] = paused
            (self.fixture.root / "cgroup/cgroup.events").write_text(f"populated 1\nfrozen {int(paused)}\n")
            return b""
        raise AssertionError("Unexpected synthetic Docker operation")

    def _process_identity(self, pid):
        return deepcopy(self.processes[pid])

    def _cgroup_path(self, _pid):
        return self.fixture.root / "cgroup"

    def _namespace(self, pid, kind):
        return ("host" if pid == os.getpid() else "container") + ":" + kind

    def _verify_cli_environment(self, pid):
        data = b"\0".join(value.encode() for value in self.view["Config"]["Env"]) + b"\0"
        with patch.object(Path, "read_bytes", return_value=data):
            return isolation.DockerActorBoundary._verify_cli_environment(self, pid)

    def _verify_processes(self):
        original = Path.iterdir
        expected = {f"/proc/{INIT_PID}/fd", f"/proc/{CLI_PID}/fd"}
        def descriptors(path):
            return iter(self.extra_fds) if str(path) in expected else original(path)
        with patch.object(Path, "iterdir", descriptors):
            return super()._verify_processes()

    def add_cli(self):
        self.processes[CLI_PID] = {
            "pid": CLI_PID, "start_ticks": "2", "exe_sha256": "2" * 64,
            "argv": ["/sdk/claude", "--output-format", "stream-json", "--tools", ",".join(isolation.ACTOR_TOOLS),
                     "--disallowedTools", ",".join(isolation.FILE_TOOLS), "--setting-sources=",
                     "--strict-mcp-config", "--session-id=" + SESSION],
        }
        self._sync_pids()


class IsolationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="iso-", dir=Path.home())
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        short_temporary = tempfile.TemporaryDirectory(prefix="iq-")
        self.addCleanup(short_temporary.cleanup)
        self.bash_tmp = Path(short_temporary.name) / "b"
        self.native_tmp = Path(short_temporary.name) / "n"
        self.bash_tmp.mkdir(mode=0o700)
        self.native_tmp.mkdir(mode=0o700)
        for name in ("candidate", "source", "inputs/admission", "private", "native_config/projects",
                     "native_config/shell-snapshots", "cgroup"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "source/workflow.js").write_text("return {fixture:true};\n")
        (self.root / "inputs/admission/open").write_bytes(b"open\n")
        (self.root / "cgroup/cgroup.events").write_text("populated 1\nfrozen 0\n")
        self.home = os.environ["HOME"]

    def make(self, *, fresh=False, readonly_extra=(), mount_edit=None, **changes):
        workspace = self.root / ("experiment" if fresh else "candidate")
        workspace.mkdir(exist_ok=True)
        candidate = workspace / "eval/workspace" if fresh else workspace
        readonly = [str(self.root / "source"), str(self.root / "inputs"),
                    str(self.root / "native_config/shell-snapshots"), *map(str, readonly_extra)]
        writable = [str(workspace), "/tmp"]
        shell = self.root / "inputs/bash"
        if shell.exists():
            shell.chmod(0o755)
        shell.write_text(isolation.shell_source(readonly_paths=readonly, writable_paths=writable, home=self.home,
            initial_cwd=workspace, bash_tmp_root=self.bash_tmp, native_tmp_root=self.native_tmp))
        shell.chmod(0o555)
        wrapper = self.root / "private/cli"
        if wrapper.exists():
            wrapper.chmod(0o755)
        wrapper.write_text(isolation.cli_wrapper_source(container_id=CONTAINER, executable="/sdk/claude", cwd=workspace))
        wrapper.chmod(0o555)
        mounts = [isolation.BindMount(workspace, workspace, True),
            isolation.BindMount(self.root / "source", self.root / "source"),
            isolation.BindMount(self.root / "inputs", self.root / "inputs"),
            isolation.BindMount(shell, Path(isolation.SHELL_PATH)),
            isolation.BindMount(self.root / "inputs/admission", Path(isolation.ADMISSION_PATH)),
            isolation.BindMount(self.root / "native_config", self.root / "native_config", True),
            isolation.BindMount(self.bash_tmp, self.bash_tmp, True),
            isolation.BindMount(self.native_tmp, self.native_tmp, True)]
        if mount_edit:
            mounts = mount_edit(mounts)
        arguments = {"container_id": CONTAINER, "image_id": IMAGE, "mounts": mounts, "shell_file": shell,
            "shell_readonly_paths": readonly, "shell_writable_paths": writable, "home": self.home,
            "candidate_root": candidate, "workspace_mount_root": workspace, "journal_root": self.root / "native_config/projects",
            "private_roots": [self.root / "private"], "admission_dir": self.root / "inputs/admission",
            "immutable_paths": [self.root / "source/workflow.js"], "mutable_output_roots": [workspace],
            "cli_executable": "/sdk/claude", "cli_sha256": "2" * 64, "bash_tmp_root": self.bash_tmp,
            "native_tmp_root": self.native_tmp, "sdk_wrapper_path": wrapper}
        arguments.update(changes)
        view = {
            "Id": CONTAINER, "Image": IMAGE,
            "State": {"Pid": INIT_PID, "Running": True, "Paused": False, "Restarting": False, "OOMKilled": False},
            "Config": {"User": f"{os.geteuid()}:{os.getegid()}", "Entrypoint": ["/usr/bin/sleep"], "Cmd": ["infinity"],
                       "Env": ["HOME=" + self.home, "SHELL=" + isolation.SHELL_PATH,
                               "CLAUDE_CODE_SHELL=" + isolation.SHELL_PATH,
                               "CLAUDE_CONFIG_DIR=" + str(self.root / "native_config"),
                               "TMPDIR=" + str(self.native_tmp), "CLAUDE_CODE_TMPDIR=" + str(self.native_tmp)]},
            "HostConfig": {"ReadonlyRootfs": True, "Privileged": False, "CapDrop": ["ALL"], "CapAdd": [],
                           "SecurityOpt": ["no-new-privileges", "seccomp=unconfined", "apparmor=unconfined"],
                           "MaskedPaths": [], "ReadonlyPaths": [], "NetworkMode": "none", "PidMode": "",
                           "IpcMode": "private", "UTSMode": "", "CgroupnsMode": "private",
                           "Devices": [], "DeviceRequests": [], "DeviceCgroupRules": [], "GroupAdd": []},
            "Mounts": [{"Source": str(mount.source), "Destination": str(mount.target), "RW": mount.writable,
                        "Type": "bind", "Propagation": "rprivate"} for mount in mounts],
        }
        return SyntheticBoundary(fixture=self, view=view, **arguments)

    def ready(self):
        boundary = self.make()
        boundary.prepare_sdk_options(Options(cwd=str(boundary.workspace_mount_root)))
        boundary.add_cli()
        boundary.native_started()
        return boundary

    def output(self, boundary, *, session=SESSION, root=None):
        path = (root or boundary.native_output_root) / "project" / session / "tasks/task.output"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"result":{"fixture":true}}\n')
        return path

    def test_fresh_eval_directory_is_absent_until_native_setup(self):
        boundary = self.make(fresh=True)
        self.assertFalse(boundary.candidate_root.parent.exists())
        boundary.candidate_root.parent.mkdir()
        with self.assertRaisesRegex(StopRejected, "eval_directory_not_fresh"):
            self.make(fresh=True)

    def test_private_temporary_roots_must_be_separate_and_owned(self):
        with self.assertRaisesRegex(StopRejected, "temporary_roots_overlap"):
            self.make(native_tmp_root=self.bash_tmp)
        (self.native_tmp).chmod(0o755)
        with self.assertRaisesRegex(StopRejected, "temporary_directory_not_private"):
            self.make()

    def test_private_state_and_native_temp_mount_aliases_are_rejected(self):
        with self.assertRaisesRegex(StopRejected, "private_state_exposed"):
            self.make(mount_edit=lambda mounts: mounts + [isolation.BindMount(self.root / "private", Path("/private_alias"))])
        with self.assertRaisesRegex(StopRejected, "native_output_exposed"):
            self.make(readonly_extra=[self.native_tmp])

    def test_native_journal_cannot_enter_the_actor_shell(self):
        with self.assertRaisesRegex(StopRejected, "native_journal_exposed"):
            self.make(readonly_extra=[self.root / "native_config"])

    def test_writable_source_alias_is_rejected_even_when_primary_mount_is_readonly(self):
        with self.assertRaisesRegex(StopRejected, "immutable_source_writable"):
            self.make(mount_edit=lambda mounts: mounts + [isolation.BindMount(self.root / "source", Path("/alias"), True)])

    def test_source_content_or_inode_change_invalidates_a_lease(self):
        boundary = self.ready()
        path = self.root / "source/workflow.js"
        original = path.read_bytes()
        path.write_bytes(b"changed\n")
        with self.assertRaisesRegex(StopRejected, "immutable_source_changed"):
            boundary._verify_immutable_sources()
        path.rename(path.with_name("retained-original.js"))
        path.write_bytes(original)
        with self.assertRaisesRegex(StopRejected, "immutable_source_changed"):
            boundary._verify_immutable_sources()

    def test_gpu_mapping_and_privilege_changes_are_rejected(self):
        boundary = self.make()
        mutations = [
            ("Devices", [{"PathOnHost": "/dev/kfd"}]), ("DeviceRequests", [{}]), ("DeviceCgroupRules", ["c 226:* rwm"]),
            ("GroupAdd", ["109"]), ("Privileged", True), ("ReadonlyRootfs", False), ("CapAdd", ["SYS_ADMIN"]),
            ("PidMode", "host"), ("NetworkMode", "host"), ("IpcMode", "host"), ("CgroupnsMode", "host"),
        ]
        for field_name, value in mutations:
            changed = deepcopy(boundary.view)
            changed["HostConfig"][field_name] = value
            with self.subTest(field=field_name), self.assertRaises(StopRejected):
                boundary._verify_container(changed, paused=False)

    def test_container_identity_launch_and_mount_changes_are_rejected(self):
        boundary = self.make()
        changes = [lambda view: view.update(Image="sha256:" + "c" * 64),
                   lambda view: view["State"].update(Pid=INIT_PID + 99),
                   lambda view: view["State"].update(OOMKilled=True),
                   lambda view: view["Config"].update(User="0:0"),
                   lambda view: view["Config"].update(Entrypoint=["/bin/bash"]),
                   lambda view: view["Config"].update(Healthcheck={"Test": ["CMD", "true"]}),
                   lambda view: view["Mounts"][0].update(Propagation="shared"),
                   lambda view: view["Mounts"].pop(),
                   lambda view: view["HostConfig"].update(SecurityOpt=["no-new-privileges"])]
        for index, change in enumerate(changes):
            view = deepcopy(boundary.view)
            change(view)
            with self.subTest(index=index), self.assertRaises(StopRejected):
                boundary._verify_container(view, paused=False)

    def test_sdk_profile_restricts_tools_and_binds_the_session(self):
        boundary = self.make()
        supplied = Options(cwd=str(boundary.workspace_mount_root))
        prepared = boundary.prepare_sdk_options(supplied)
        self.assertEqual(prepared.tools, list(isolation.ACTOR_TOOLS))
        self.assertEqual(prepared.disallowed_tools, list(isolation.FILE_TOOLS))
        self.assertTrue(prepared.strict_mcp_config)
        self.assertEqual(boundary._native_session_id, SESSION)
        self.assertEqual(supplied.tools, [])
        self.assertEqual(prepared.env["CLAUDE_CONFIG_DIR"], str(boundary.journal_root.parent))
        self.assertEqual(supplied.env, {})
        with self.assertRaisesRegex(StopRejected, "session_not_predeclared"):
            boundary.prepare_sdk_options(Options(cwd=supplied.cwd, session_id=None))
        with self.assertRaisesRegex(StopRejected, "sdk_settings_unsupported"):
            isolation.sdk_options(Options(cwd=supplied.cwd, setting_sources=["user"]), cli_path=boundary.sdk_wrapper_path)

    def test_sdk_mirror_directory_is_per_actor_and_preserves_caller_environment(self):
        first = self.make()
        other = type(self)(self._testMethodName)
        other.setUp()
        self.addCleanup(other.doCleanups)
        second = other.make()
        for parent_directory in (None, str(self.root / "unrelated_parent_config")):
            with self.subTest(parent_directory=parent_directory), patch.dict(os.environ):
                if parent_directory is None:
                    os.environ.pop("CLAUDE_CONFIG_DIR", None)
                else:
                    os.environ["CLAUDE_CONFIG_DIR"] = parent_directory
                parent = dict(os.environ)
                environment = {"UNRELATED_OPTION": "preserved", "CLAUDE_CONFIG_DIR": str(self.root / "unrelated_option_config")}
                prepared = []
                for boundary in (first, second):
                    supplied = Options(cwd=str(boundary.workspace_mount_root), env=environment)
                    result = boundary.prepare_sdk_options(supplied)
                    self.assertEqual(result.env, {**environment, "CLAUDE_CONFIG_DIR": str(boundary.journal_root.parent)})
                    self.assertIsNot(result.env, environment)
                    self.assertEqual(supplied.env, environment)
                    prepared.append(result)
                self.assertNotEqual(prepared[0].env["CLAUDE_CONFIG_DIR"], prepared[1].env["CLAUDE_CONFIG_DIR"])
                self.assertEqual(environment["CLAUDE_CONFIG_DIR"], str(self.root / "unrelated_option_config"))
                self.assertEqual(dict(os.environ), parent)

    def test_actual_sdk_mirror_accepts_each_actor_without_parent_configuration(self):
        try:
            from claude_agent_sdk import ClaudeAgentOptions
            from claude_agent_sdk._internal.session_resume import build_mirror_batcher
            from claude_agent_sdk._internal.transcript_mirror_batcher import (
                _MirrorEntry,
            )
        except ImportError:
            self.skipTest("The native SDK mirror API is not installed.")
        first = self.make()
        other = type(self)(self._testMethodName)
        other.setUp()
        self.addCleanup(other.doCleanups)
        second = other.make()

        class Store:
            def __init__(self):
                self.calls = []

            async def append(self, key, entries):
                self.calls.append((key, entries))

        async def check(boundary):
            environment = {"UNRELATED_OPTION": "preserved", "CLAUDE_CONFIG_DIR": str(self.root / "unrelated_option_config")}
            supplied = ClaudeAgentOptions(cwd=str(boundary.workspace_mount_root), session_id=SESSION,
                                          setting_sources=[], env=environment)
            prepared = boundary.prepare_sdk_options(supplied)
            store = Store()
            errors = []

            async def on_error(key, error):
                errors.append((key, error))

            batcher = build_mirror_batcher(store, None, prepared.env, on_error, flush_mode="eager")
            self.assertEqual(batcher.projects_dir, str(boundary.journal_root))
            path = boundary.journal_root / "project" / SESSION / "subagents/workflows/wf_fixture/agent-fixture.jsonl"
            entries = [{"type": "user", "parentUuid": None, "message": {"content": "Synthetic initial task"}}]
            failures = []
            await batcher._do_flush([_MirrorEntry(str(path), entries, len(json.dumps(entries)))], failures)
            self.assertEqual(errors, [])
            self.assertEqual(failures, [])
            self.assertEqual(store.calls, [({"project_key": "project", "session_id": SESSION,
                                           "subpath": "subagents/workflows/wf_fixture/agent-fixture"}, entries)])
            self.assertEqual(prepared.env["UNRELATED_OPTION"], "preserved")
            self.assertEqual(supplied.env, environment)

        for parent_directory in (None, str(self.root / "unrelated_parent_config")):
            with self.subTest(parent_directory=parent_directory), patch.dict(os.environ):
                if parent_directory is None:
                    os.environ.pop("CLAUDE_CONFIG_DIR", None)
                else:
                    os.environ["CLAUDE_CONFIG_DIR"] = parent_directory
                parent = dict(os.environ)
                for boundary in (first, second):
                    asyncio.run(check(boundary))
                self.assertEqual(dict(os.environ), parent)

    def test_private_sdk_wrapper_cannot_change_or_have_a_second_link(self):
        boundary = self.make()
        path = boundary.sdk_wrapper_path
        path.chmod(0o755)
        path.write_text("#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(StopRejected, "sdk_wrapper_changed"):
            boundary.prepare_sdk_options(Options(cwd=str(boundary.workspace_mount_root)))
        os.link(path, self.root / "wrapper_alias")
        with self.assertRaisesRegex(StopRejected, "sdk_wrapper_not_regular"):
            boundary.prepare_sdk_options(Options(cwd=str(boundary.workspace_mount_root)))

    def test_native_cli_requires_exact_tools_and_session(self):
        boundary = self.make()
        boundary.prepare_sdk_options(Options(cwd=str(boundary.workspace_mount_root)))
        boundary.add_cli()
        original = deepcopy(boundary.processes[CLI_PID])
        for old, new in ((",".join(isolation.ACTOR_TOOLS), "Bash,Write"),
                         ("--setting-sources=", "--setting-sources=user"),
                         ("--session-id=" + SESSION, "--session-id=aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")):
            boundary.processes[CLI_PID] = deepcopy(original)
            arguments = boundary.processes[CLI_PID]["argv"]
            arguments[arguments.index(old)] = new
            with self.subTest(old=old), self.assertRaises(StopRejected):
                boundary.bind_cli()
        boundary.processes[CLI_PID] = original
        boundary.bind_cli()
        with self.assertRaisesRegex(StopRejected, "cli_already_bound"):
            boundary.bind_cli()

    def test_extra_process_or_cgroup_mismatch_blocks_the_checkpoint(self):
        boundary = self.ready()
        extra = CLI_PID + 1
        boundary.processes[extra] = {**boundary.processes[CLI_PID], "pid": extra}
        boundary._sync_pids()
        with self.assertRaisesRegex(StopRejected, "unfinished_actor_process"):
            boundary._verify_processes()
        (self.root / "cgroup/cgroup.procs").write_text(str(INIT_PID) + "\n")
        with self.assertRaisesRegex(StopRejected, "census_incomplete"):
            boundary._pids()
        boundary.top_override = b"PID\nnot-a-pid\n"
        with self.assertRaisesRegex(StopRejected, "census_invalid"):
            boundary._pids()

    def test_replaced_cli_identity_and_environment_are_rejected(self):
        boundary = self.ready()
        boundary.processes[CLI_PID]["start_ticks"] = "different"
        with self.assertRaisesRegex(StopRejected, "process_replaced"):
            boundary._verify_processes()
        boundary.processes[CLI_PID]["start_ticks"] = "2"
        boundary.view["Config"]["Env"] = [value for value in boundary.view["Config"]["Env"] if not value.startswith("SHELL=")]
        with self.assertRaisesRegex(StopRejected, "shell_override_changed"):
            boundary._verify_processes()

    def test_gpu_file_descriptor_is_rejected_without_opening_a_device(self):
        boundary = self.ready()
        descriptor = SimpleNamespace(stat=lambda: SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=os.makedev(226, 144)))
        boundary.extra_fds = [descriptor]
        with patch.object(os, "readlink", return_value="/dev/dri/renderD144"), \
                self.assertRaisesRegex(StopRejected, "native_gpu_descriptor"):
            boundary._verify_processes()

    def test_closed_descriptor_during_audit_does_not_hide_an_extra_process(self):
        boundary = self.ready()
        boundary.extra_fds = [SimpleNamespace()]
        with patch.object(os, "readlink", side_effect=FileNotFoundError):
            boundary._verify_processes()
        boundary.processes[CLI_PID + 1] = {**boundary.processes[CLI_PID], "pid": CLI_PID + 1}
        boundary._sync_pids()
        with patch.object(os, "readlink", side_effect=FileNotFoundError), \
                self.assertRaisesRegex(StopRejected, "unfinished_actor_process"):
            boundary._verify_processes()

    def test_lease_stays_frozen_and_resumes_after_a_body_error(self):
        boundary = self.ready()
        with self.assertRaisesRegex(RuntimeError, "synthetic body failure"), \
                boundary.lease(candidate_root=boundary.candidate_root, protected_paths=[self.root / "private"]):
            self.assertTrue(boundary._leased)
            self.assertTrue(boundary.view["State"]["Paused"])
            raise RuntimeError("synthetic body failure")
        self.assertFalse(boundary._leased)
        self.assertFalse(boundary.view["State"]["Paused"])
        self.assertEqual([entry[0] for entry in boundary.commands if entry[0] in {"pause", "unpause"}], ["pause", "unpause"])

    def test_failed_resume_invalidates_the_boundary(self):
        boundary = self.ready()
        boundary.fail_unpause = True
        with self.assertRaisesRegex(StopRejected, "synthetic_unpause_failed"), \
                boundary.lease(candidate_root=boundary.candidate_root, protected_paths=[self.root / "private"]):
            pass
        self.assertTrue(boundary.failed)
        self.assertFalse(boundary._leased)

    def test_native_return_seals_admission_before_resume(self):
        boundary = self.ready()
        output = boundary.candidate_root / "result.json"
        output.write_text('{"fixture":true}\n')
        protected = [self.root / "private", output]
        with boundary.lease(candidate_root=boundary.candidate_root, protected_paths=protected):
            boundary.check(stage="native_return", candidate_root=boundary.candidate_root, protected_paths=protected)
            self.assertTrue(boundary.sealed)
            self.assertEqual((boundary.admission_dir / "open").read_bytes(), b"sealed\n")
            boundary._seal_admission()
        self.assertFalse(boundary.view["State"]["Paused"])

    def test_unclassified_protected_path_is_rejected(self):
        boundary = self.ready()
        with self.assertRaisesRegex(StopRejected, "protected_path_unclassified"), \
                boundary.lease(candidate_root=boundary.candidate_root, protected_paths=[self.root / "unclassified"]):
            pass

    def test_native_output_requires_the_bound_session_and_private_root(self):
        boundary = self.make()
        path = self.output(boundary)
        with self.assertRaisesRegex(StopRejected, "output_session_changed"):
            boundary.read_native_output(path)
        boundary.prepare_sdk_options(Options(cwd=str(boundary.workspace_mount_root)))
        self.assertEqual(boundary.read_native_output(path), path.read_text())
        other = self.output(boundary, session="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        actor_file = self.output(boundary, root=self.bash_tmp)
        for candidate in (other, actor_file, path.with_suffix(".json")):
            with self.subTest(path=str(candidate)), self.assertRaises(StopRejected):
                boundary.read_native_output(candidate)

    def test_native_output_rejects_symlinks_hardlinks_and_fifos(self):
        boundary = self.ready()
        path = self.output(boundary)
        alias = path.with_name("alias.output")
        alias.symlink_to(path)
        with self.assertRaisesRegex(StopRejected, "path_not_canonical"):
            boundary.read_native_output(alias)
        alias.unlink()
        os.link(path, alias)
        with self.assertRaisesRegex(StopRejected, "output_not_regular"):
            boundary.read_native_output(path)
        alias.unlink()
        path.unlink()
        os.mkfifo(path)
        with self.assertRaisesRegex(StopRejected, "output_not_regular"):
            boundary.read_native_output(path)

    def test_native_output_replacement_during_read_is_rejected(self):
        boundary = self.ready()
        path = self.output(boundary)
        original = os.read
        changed = False
        def replace_during_read(descriptor, count):
            nonlocal changed
            if not changed:
                changed = True
                path.unlink()
                path.write_text("different output\n")
            return original(descriptor, count)
        with patch.object(os, "read", replace_during_read), self.assertRaisesRegex(StopRejected, "output_changed"):
            boundary.read_native_output(path)

    def test_native_output_size_limit_precedes_reading(self):
        boundary = self.ready()
        path = self.output(boundary)
        with path.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        with patch.object(os, "read", side_effect=AssertionError("Oversized output was read")), \
                self.assertRaisesRegex(StopRejected, "output_not_regular"):
            boundary.read_native_output(path)

    def test_native_temporary_root_replacement_is_rejected(self):
        boundary = self.ready()
        path = self.output(boundary)
        boundary.native_tmp_root.rename(self.native_tmp.with_name("old-native"))
        boundary.native_tmp_root.mkdir(mode=0o700)
        with self.assertRaisesRegex(StopRejected, "temporary_directory_changed"):
            boundary.read_native_output(path)

    def test_shell_generator_keeps_native_temp_outside_the_inner_namespace(self):
        boundary = self.make()
        source = boundary.shell_file.read_text()
        parsed = ast.parse(source)
        assignment = next(node for node in parsed.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets))
        arguments = ast.literal_eval(assignment.value.left.left)
        self.assertIn("--unshare-all", arguments)
        self.assertIn("--clearenv", arguments)
        self.assertNotIn(str(boundary.native_tmp_root), arguments)
        index = arguments.index(str(boundary.bash_tmp_root))
        self.assertEqual(arguments[index - 1:index + 2], ["--bind", str(boundary.bash_tmp_root), "/tmp"])

    def test_shell_generator_rejects_sensitive_mounts_and_environment(self):
        base = {"readonly_paths": [str(self.root / "source")], "writable_paths": [str(self.root / "candidate"), "/tmp"],
                "home": self.home, "initial_cwd": self.root / "candidate", "bash_tmp_root": self.bash_tmp, "native_tmp_root": self.native_tmp}
        for path in ("/", "/proc", "/dev", "/run", "/sys", self.home):
            with self.subTest(path=path), self.assertRaises(StopRejected):
                isolation.shell_source(**{**base, "readonly_paths": [path]})
        for key in ("BASH_ENV", "LD_PRELOAD", "PYTHONPATH", "TMPDIR", "CLAUDE_CODE_TMPDIR"):
            with self.subTest(key=key), self.assertRaisesRegex(StopRejected, "environment_unsafe"):
                isolation.shell_source(**base, environment={key: "/foreign"})

    def bridge(self, *, value=None, mode="regular", target=None, gate=b"open\n"):
        """Run the generated bridge with actual files and a synthetic shell child."""
        boundary = self.make()
        target = target or self.native_tmp / "claude-abcd-cwd"
        inner = self.bash_tmp / (".geak-cwd-" + "1" * 32)
        original_open = builtins.open
        commands = []
        def open_gate(path, *args, **kwargs):
            if str(path) == isolation.ADMISSION_PATH + "/open":
                return io.BytesIO(gate)
            return original_open(path, *args, **kwargs)
        def child(arguments, **kwargs):
            self.assertEqual(kwargs.get("env"), {})
            commands.append(arguments)
            if mode == "symlink":
                inner.symlink_to(self.root / "source/workflow.js")
            elif mode == "fifo":
                os.mkfifo(inner)
            elif mode not in {"missing", "failure"}:
                raw = value if isinstance(value, bytes) else (str(value or boundary.candidate_root) + "\n").encode()
                inner.write_bytes(raw)
                if mode == "hardlink":
                    os.link(inner, self.bash_tmp / "second-link")
            return SimpleNamespace(returncode=1 if mode == "failure" else 0)
        source = boundary.shell_file.read_text()
        arguments = ["fixture-wrapper", "-c", "true && pwd -P >| " + str(target)]
        with patch.object(builtins, "open", open_gate), patch.object(os, "closerange"), \
                patch.object(subprocess, "run", child), patch("secrets.token_hex", return_value="1" * 32), \
                patch.object(sys, "argv", arguments), patch.object(sys, "stderr", io.StringIO()), \
                self.assertRaises(SystemExit) as stopped:
            # The generator is fixed repository code. The shell child remains a stub.
            exec(compile(source, "synthetic-isolation-cwd-bridge", "exec"), {"__name__": "__main__"})  # noqa: S102
        return stopped.exception.code, target, commands

    def test_cwd_bridge_accepts_only_actor_directories(self):
        code, target, commands = self.bridge()
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(), str(self.root / "candidate") + "\n")
        self.assertEqual(len(commands), 1)
        target.unlink()
        (self.bash_tmp / "child").mkdir()
        code, target, _commands = self.bridge(value="/tmp/child")
        self.assertEqual(code, 0)
        self.assertEqual(target.read_text(), str(self.bash_tmp / "child") + "\n")

    def test_cwd_bridge_rejects_untrusted_file_shapes_and_values(self):
        cases = [("symlink", None), ("hardlink", None), ("fifo", None),
                 ("regular", str(self.native_tmp)), ("regular", "/tmp/../escape"),
                 ("regular", b"\xff"), ("regular", b"/" + b"x" * 4097)]
        for mode, value in cases:
            with self.subTest(mode=mode, value=value):
                code, target, _commands = self.bridge(mode=mode, value=value)
                self.assertEqual(code, 125)
                self.assertFalse(target.exists())
                (self.bash_tmp / "second-link").unlink(missing_ok=True)

    def test_cwd_bridge_never_overwrites_native_output_or_a_prior_cwd(self):
        target = self.native_tmp / "claude-abcd-cwd"
        target.write_text("PRIOR_CWD\n")
        code, _target, _commands = self.bridge(target=target)
        self.assertEqual(code, 125)
        self.assertEqual(target.read_text(), "PRIOR_CWD\n")
        target.unlink()
        output = self.native_tmp / "task.output"
        output.write_text("NATIVE_OUTPUT\n")
        code, _target, _commands = self.bridge(target=output)
        self.assertEqual(code, 0)
        self.assertEqual(output.read_text(), "NATIVE_OUTPUT\n")
        (self.bash_tmp / (".geak-cwd-" + "1" * 32)).unlink()
        code, _target, commands = self.bridge(gate=b"sealed\n")
        self.assertEqual(code, 125)
        self.assertEqual(commands, [])

    def test_cwd_bridge_handles_exec_without_a_suffix_and_shell_failure(self):
        code, target, _commands = self.bridge(mode="missing")
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())
        code, target, _commands = self.bridge(mode="failure")
        self.assertEqual(code, 1)
        self.assertFalse(target.exists())

    def test_bind_mount_rejects_host_devices_aliases_and_invalid_modes(self):
        link = self.root / "device_alias"
        link.symlink_to("/dev")
        for source, target, writable in ((Path("/dev"), Path("/dev"), False),
                                         (link, Path("/alias"), False),
                                         (self.root, Path("relative"), False),
                                         (self.root, Path("/valid"), 1)):
            with self.subTest(source=str(source), target=str(target)), self.assertRaises(StopRejected):
                isolation.BindMount(source, target, writable)

    def test_real_current_process_identity_needs_no_gpu_or_privileged_probe(self):
        value = isolation.DockerActorBoundary._process_identity(os.getpid())
        self.assertEqual(value["pid"], os.getpid())
        self.assertTrue(value["start_ticks"].isdigit())
        self.assertEqual(len(value["exe_sha256"]), 64)
        self.assertTrue(isolation.DockerActorBoundary._namespace(os.getpid(), "pid").startswith("pid:["))
        for pid in (True, 0, 1):
            with self.subTest(pid=pid), self.assertRaises(StopRejected):
                isolation.DockerActorBoundary._process_identity(pid)

    def test_cgroup_parser_requires_one_nonroot_v2_path(self):
        with patch.object(Path, "read_text", return_value="0::/fixture/group\n"):
            self.assertEqual(isolation.DockerActorBoundary._cgroup_path(123), Path("/sys/fs/cgroup/fixture/group"))
        for value in ("", "0::/\n", "1:cpu:/fixture\n", "0::/../escape\n", "0::/one\n0::/two\n"):
            with self.subTest(value=value), patch.object(Path, "read_text", return_value=value), self.assertRaises(StopRejected):
                isolation.DockerActorBoundary._cgroup_path(123)

    def test_docker_errors_return_fixed_rejections(self):
        boundary = self.make()
        with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b"fixture")):
            self.assertEqual(isolation.DockerActorBoundary._docker(boundary, "inspect", CONTAINER), b"fixture")
        for response in (SimpleNamespace(returncode=1, stdout=b"private error"), subprocess.TimeoutExpired("docker", 20)):
            context = patch.object(subprocess, "run", side_effect=response) if isinstance(response, BaseException) else patch.object(subprocess, "run", return_value=response)
            with context, self.assertRaises(StopRejected):
                isolation.DockerActorBoundary._docker(boundary, "inspect", CONTAINER)
        with patch.object(boundary, "_docker", return_value=b"invalid JSON"), \
                self.assertRaisesRegex(StopRejected, "inspection_invalid"):
            boundary._inspect()


if __name__ == "__main__":
    unittest.main()
