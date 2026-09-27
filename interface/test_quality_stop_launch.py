# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check actor assembly without a provider, GPU, or Docker daemon."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interface.native_cost_controls.quality_stop_controller import StopRejected
from interface.native_cost_controls.quality_stop_launch import (
    DockerCommands,
    create_native_actor,
)


class FakeDocker:
    def __init__(self):
        self.calls = []
        self.cid = "a" * 64

    def run(self, *arguments):
        self.calls.append(arguments)
        return (self.cid + "\n").encode() if arguments[0] == "create" else b""

    def stop_exact(self, cid):
        self.calls.append(("stop_exact", cid))

    def require_closed(self, cid):
        self.calls.append(("require_closed", cid))


class ActorLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.task = self.root / "source", self.root / "task"
        self.source.mkdir()
        self.task.mkdir()
        (self.source / "workflow.js").write_text("// trusted fixture source\n")
        self.cli = self.root / "claude"
        self.cli.write_bytes(b"synthetic immutable native executable")
        self.docker, self.boundaries = FakeDocker(), []
        self.options = {"trial_id": "fixture_trial", "image_id": "sha256:" + "b" * 64,
            "source_root": self.source, "task_root": self.task, "workspace_root": self.root / "work",
            "candidate_root": self.root / "work/eval/workspace", "private_root": self.root / "private",
            "launch_root": self.root / "launch", "native_config_root": self.root / "native_config",
            "native_home_root": self.root / "native_home", "bash_tmp_root": self.root / "bash_tmp",
            "native_tmp_root": self.root / "native_tmp", "cache_root": self.root / "work/cache",
            "cli_binary": self.cli, "cli_sha256": hashlib.sha256(self.cli.read_bytes()).hexdigest(),
            "cpu_set": "0-1", "shell_environment": {"PYTHONDONTWRITEBYTECODE": "1"},
            "immutable_paths": [self.source / "workflow.js"], "docker": self.docker,
            "boundary_factory": self.boundary}

    def boundary(self, **arguments):
        self.boundaries.append(arguments)
        return SimpleNamespace(**arguments)

    def test_assembly_keeps_fresh_eval_private_state_and_exact_cli(self):
        launch = create_native_actor(**self.options)
        self.assertEqual(launch.container_id, self.docker.cid)
        self.assertEqual([call[0] for call in self.docker.calls], ["create", "start"])
        self.assertFalse(self.options["candidate_root"].parent.exists())
        self.assertTrue(launch.wrapper_path.is_relative_to(self.options["private_root"]))
        self.assertIn(self.docker.cid, launch.wrapper_path.read_text())
        boundary = self.boundaries[0]
        self.assertEqual(boundary["workspace_mount_root"], self.options["workspace_root"])
        self.assertEqual(boundary["native_tmp_root"], self.options["native_tmp_root"])
        self.assertEqual(boundary["shell_writable_paths"][-1], "/tmp")
        self.assertTrue(all(mount.source != self.options["private_root"] for mount in boundary["mounts"]))
        self.assertFalse(any(mount.source == Path("/tmp") for mount in boundary["mounts"]))
        record = json.loads(launch.manifest_path.read_text())
        self.assertFalse(record["provider_credentials_in_argv"])
        self.assertEqual(record["cli_sha256"], self.options["cli_sha256"])
        self.assertTrue((self.options["launch_root"] / "ACTOR_READY.json").exists())

    def test_source_identity_and_existing_eval_fail_before_docker(self):
        for change in ({"cli_sha256": "0" * 64}, {"source_root": Path("relative")},
                       {"cpu_set": "all"}):
            with self.subTest(change=change), self.assertRaises(StopRejected):
                create_native_actor(**{**self.options, **change})
        self.options["candidate_root"].parent.mkdir(parents=True)
        with self.assertRaisesRegex(StopRejected, "eval_directory_not_fresh"):
            create_native_actor(**self.options)
        self.assertEqual(self.docker.calls, [])

    def test_actor_context_closes_the_exact_container_when_its_body_fails(self):
        with self.assertRaisesRegex(RuntimeError, "fixture_body_failed"), create_native_actor(**self.options) as actor:
            raise RuntimeError("fixture_body_failed")
        self.assertTrue(actor.closed)
        self.assertEqual(self.docker.calls[-2:], [("stop_exact", self.docker.cid), ("require_closed", self.docker.cid)])
        before = list(self.docker.calls)
        actor.close()
        self.assertEqual(self.docker.calls, before)
        with self.assertRaisesRegex(StopRejected, "actor_launch_already_closed"):
            actor.__enter__()

    def test_credentials_and_unreserved_device_flags_are_rejected(self):
        for change in ({"native_environment": {"ANTHROPIC_API_KEY": "synthetic"}},
                       {"shell_environment": {"OPENAI_API_KEY": "synthetic"}},
                       {"device_arguments": ["--privileged"]},
                       {"device_arguments": ["--device", "/dev/kfd"]}):
            with self.subTest(change=tuple(change)), self.assertRaises(StopRejected):
                create_native_actor(**{**self.options, **change})
        self.assertEqual(self.docker.calls, [])

    def test_failed_boundary_removes_only_the_returned_owned_cpu_container(self):
        def reject(**_arguments):
            raise StopRejected("fixture_boundary_rejected")
        with self.assertRaisesRegex(StopRejected, "fixture_boundary_rejected"):
            create_native_actor(**{**self.options, "boundary_factory": reject})
        self.assertEqual(self.docker.calls[-1], ("rm", "--force", self.docker.cid))

    def test_invalid_create_reply_cannot_become_a_foreign_cleanup_target(self):
        self.docker.cid = "foreign-name"
        with self.assertRaisesRegex(StopRejected, "actor_container_id_invalid"):
            create_native_actor(**self.options)
        self.assertEqual([call[0] for call in self.docker.calls], ["create"])

    def test_gpu_reservation_retains_its_cleanup_ownership(self):
        registered, blocked = [], []
        reservation = SimpleNamespace(register_actor=registered.append, block=blocked.append)
        def reject(**_arguments):
            raise StopRejected("fixture_boundary_rejected")
        with self.assertRaises(StopRejected):
            create_native_actor(**{**self.options, "reservation": reservation, "boundary_factory": reject,
                                   "device_arguments": ["--device", "/dev/kfd", "--group-add", "44"]})
        self.assertEqual(registered, [self.docker.cid])
        self.assertEqual(blocked, ["actor_startup_failed"])
        self.assertFalse(any(call[0] == "rm" for call in self.docker.calls))


class DockerLifecycleTests(unittest.TestCase):
    cid = "d" * 64

    def setUp(self):
        self.docker = DockerCommands()
        self.view = {"Id": self.cid, "State": {"Running": False, "Paused": False}}

    def test_command_and_attachment_keep_explicit_environment(self):
        with patch("interface.native_cost_controls.quality_stop_launch.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout=b"fixture")) as run:
            self.assertEqual(self.docker.run("version"), b"fixture")
            self.assertEqual(run.call_args.args[0], ["/usr/bin/docker", "version"])
            self.assertEqual(set(run.call_args.kwargs["env"]), {"PATH", "HOME"})
            run.return_value.returncode = 1
            with self.assertRaises(StopRejected):
                self.docker.run("version")
        with patch("interface.native_cost_controls.quality_stop_launch.subprocess.Popen") as start:
            self.docker.start(self.cid, stdout=1, stderr=2)
            self.assertEqual(start.call_args.args[0], ("/usr/bin/docker", "start", "--attach", self.cid))
            self.assertTrue(start.call_args.kwargs["start_new_session"])

    def test_invalid_or_changed_container_identity_fails(self):
        with patch.object(self.docker, "run", return_value=json.dumps([self.view]).encode()) as run:
            self.assertEqual(self.docker.inspect(self.cid), self.view)
            with self.assertRaises(StopRejected):
                self.docker.cancel_start("foreign-name")
            self.assertEqual(run.call_count, 1)
            run.return_value = b"[]"
            with self.assertRaisesRegex(StopRejected, "container_identity_changed"):
                self.docker.inspect(self.cid)

    def test_closed_and_absent_containers_pass_and_live_or_unknown_inventory_fail(self):
        with patch.object(self.docker, "run", return_value=(self.cid + "\n").encode()) as run, \
             patch.object(self.docker, "inspect", return_value=self.view):
            self.docker.require_closed(self.cid)
            self.view["State"]["Running"] = True
            with self.assertRaisesRegex(StopRejected, "prior_actor_still_running"):
                self.docker.require_closed(self.cid)
            run.return_value = b"foreign-name\n"
            with self.assertRaisesRegex(StopRejected, "container_inventory_unknown"):
                self.docker.require_closed(self.cid)
            run.return_value = b""
            self.docker.require_closed(self.cid)

    def test_stop_targets_only_the_owned_container_and_accepts_a_closed_kill_race(self):
        self.view["State"]["Running"] = True
        def kill(*args):
            self.assertEqual(args, ("kill", "--signal", "KILL", self.cid))
            self.view["State"]["Running"] = False
            raise StopRejected("fixture_kill_race")
        with patch.object(self.docker, "inspect", return_value=self.view), patch.object(self.docker, "run", side_effect=kill):
            self.assertEqual(self.docker.stop_exact(self.cid), self.view)

    def test_unknown_stop_and_paused_timeout_are_rejected(self):
        self.view["State"]["Running"] = True
        with patch.object(self.docker, "inspect", return_value=self.view), \
             patch.object(self.docker, "run", side_effect=StopRejected("fixture_kill_failed")), \
             self.assertRaisesRegex(StopRejected, "container_stop_unknown"):
            self.docker.stop_exact(self.cid)
        self.view["State"].update(Running=False, Paused=True)
        with patch.object(self.docker, "inspect", return_value=self.view), \
             patch("interface.native_cost_controls.quality_stop_launch.time.monotonic", side_effect=[0, 0, 16]), \
             patch("interface.native_cost_controls.quality_stop_launch.time.sleep"), \
             self.assertRaisesRegex(StopRejected, "container_stop_unknown"):
            self.docker.stop_exact(self.cid)

    def test_cancel_confirms_absence_after_exact_removal(self):
        with patch.object(self.docker, "run", return_value=b"") as run:
            self.docker.cancel_start(self.cid)
        self.assertEqual(run.call_args_list[0].args, ("rm", "--force", self.cid))
        self.assertEqual(run.call_args_list[1].args[0:3], ("container", "ls", "--all"))
