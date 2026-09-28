# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create the concrete Docker/bwrap actor used by the native stopping profile.

The default profile is CPU-only. A trusted reservation supplies device arguments
and its reviewed boundary class for a GPU run. This builder never queries a
provider or starts the scorer. The caller retains the recorder until settlement.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .quality_stop_controller import StopRejected, require
from .quality_stop_isolation import (
    BindMount,
    DockerActorBoundary,
    cli_wrapper_source,
    shell_source,
)


def _path(value):
    value = Path(value)
    require(value.is_absolute() and value.resolve() == value
            and all(character.isalnum() or character in "/_.-" for character in str(value)), "actor_launch_path_invalid")
    return value


def _write(path, value, *, executable=False):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o500 if executable else 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


class DockerCommands:
    """Run and close exact owned containers without client environment overrides."""

    synthetic = False
    failure_namespace = "actor"
    command_failure = "actor_docker_command_failed"
    command_timeout = 30
    environment: ClassVar[dict[str, str]] = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]}

    def _require(self, condition, suffix):
        require(condition, self.failure_namespace + "_" + suffix)

    def _identity(self, cid):
        self._require(isinstance(cid, str) and re.fullmatch(r"[0-9a-f]{64}", cid), "container_id_invalid")

    def run(self, *arguments):
        result = subprocess.run(["/usr/bin/docker", *arguments], stdin=subprocess.DEVNULL, capture_output=True,
                                check=False, timeout=self.command_timeout, env=self.environment)
        require(result.returncode == 0, self.command_failure)
        return result.stdout

    def inspect(self, cid):
        self._identity(cid)
        rows = json.loads(self.run("inspect", cid))
        self._require(len(rows) == 1 and rows[0]["Id"] == cid, "container_identity_changed")
        return rows[0]

    def attach_argv(self, cid):
        self._identity(cid)
        return ("/usr/bin/docker", "start", "--attach", cid)

    def start(self, cid, *, stdout, stderr):
        return subprocess.Popen(self.attach_argv(cid), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                env=self.environment, start_new_session=True)

    def stop_exact(self, cid):
        view = self.inspect(cid)
        if view["State"]["Running"]:
            try:
                self.run("kill", "--signal", "KILL", cid)
            except StopRejected:
                self._require(self.inspect(cid)["State"]["Running"] is False, "container_stop_unknown")
        limit = time.monotonic() + 15
        while time.monotonic() < limit:
            view = self.inspect(cid)
            if view["State"]["Running"] is False and view["State"]["Paused"] is False:
                return view
            time.sleep(.05)
        self._require(False, "container_stop_unknown")

    def require_closed(self, cid):
        self._identity(cid)
        ids = self.run("container", "ls", "--all", "--no-trunc", "--format", "{{.ID}}").decode().splitlines()
        self._require(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in ids), "container_inventory_unknown")
        if cid in ids:
            state = self.inspect(cid)["State"]
            self._require(state["Running"] is False and state["Paused"] is False, "prior_actor_still_running")

    def cancel_start(self, cid):
        """Remove only the owned container when its pending start is uncertain."""
        self._identity(cid)
        self.run("rm", "--force", cid)
        self.require_closed(cid)


@dataclass
class ActorLaunch:
    container_id: str
    boundary: DockerActorBoundary
    manifest_path: Path
    wrapper_path: Path
    docker: object
    closed: bool = False

    def close(self):
        """Stop and verify this actor before the caller releases its reservation."""
        if not self.closed:
            self.docker.stop_exact(self.container_id)
            self.docker.require_closed(self.container_id)
            self.closed = True

    def __enter__(self):
        require(not self.closed, "actor_launch_already_closed")
        return self

    def __exit__(self, _kind, _value, _traceback):
        self.close()


def create_native_actor(*, trial_id, image_id, source_root, task_root, workspace_root, candidate_root,
                        private_root, launch_root, native_config_root, native_home_root,
                        bash_tmp_root, native_tmp_root, cache_root, cli_binary, cli_sha256,
                        cpu_set, shell_environment, immutable_paths, docker=None,
                        boundary_factory=DockerActorBoundary, shell_factory=shell_source,
                        boundary_extra=None, readonly_extra=(), device_arguments=(), labels=None,
                        native_environment=None, reservation=None, memory_bytes=None):
    """Create one fresh actor and return its verified boundary and launch record."""
    require(isinstance(trial_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", trial_id), "actor_trial_id_invalid")
    require(isinstance(image_id, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", image_id), "actor_image_id_invalid")
    require(isinstance(cli_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", cli_sha256), "actor_cli_hash_invalid")
    source_root, task_root = _path(source_root), _path(task_root)
    workspace_root, candidate_root = _path(workspace_root), _path(candidate_root)
    private_root, launch_root = _path(private_root), _path(launch_root)
    native_config_root, native_home_root = _path(native_config_root), _path(native_home_root)
    bash_tmp_root, native_tmp_root, cache_root = _path(bash_tmp_root), _path(native_tmp_root), _path(cache_root)
    cli_binary = _path(cli_binary)
    require(source_root.is_dir() and task_root.is_dir() and cli_binary.is_file(), "actor_input_missing")
    require(hashlib.sha256(cli_binary.read_bytes()).hexdigest() == cli_sha256, "actor_cli_source_changed")
    require(workspace_root in candidate_root.parents and not candidate_root.parent.exists(), "actor_eval_directory_not_fresh")
    require(isinstance(cpu_set, str) and re.fullmatch(r"[0-9,-]+", cpu_set), "actor_cpu_set_invalid")
    require(private_root not in workspace_root.parents and workspace_root not in private_root.parents
            and private_root != workspace_root, "actor_private_workspace_overlap")
    require(isinstance(native_environment or {}, dict), "actor_native_environment_invalid")
    require(not any(key.startswith(("ANTHROPIC_API", "ANTHROPIC_AUTH", "ANTHROPIC_CUSTOM", "OPENAI_"))
                    for key in (native_environment or {})), "actor_provider_credentials_in_config")
    require(isinstance(shell_environment, dict) and not any(key.startswith(("ANTHROPIC_", "OPENAI_"))
            for key in shell_environment), "actor_shell_credentials_in_config")
    device_arguments = tuple(device_arguments)
    require(not device_arguments or (reservation is not None and len(device_arguments) % 2 == 0
            and all(device_arguments[index] in {"--device", "--group-add"}
                    and isinstance(device_arguments[index + 1], str) and not device_arguments[index + 1].startswith("-")
                    for index in range(0, len(device_arguments), 2))), "actor_device_arguments_invalid")
    launch_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for path in (workspace_root, native_config_root, native_home_root, bash_tmp_root, native_tmp_root, cache_root):
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    journal_root = native_config_root / "projects"
    journal_root.mkdir(mode=0o700)
    admission = launch_root / "admission"
    admission.mkdir(mode=0o700)
    _write(admission / "open", b"open\n")
    readonly = tuple(dict.fromkeys(["/usr", "/bin", "/lib", "/lib64", "/etc",
                                   str(source_root), str(task_root), *map(str, readonly_extra)]))
    writable = tuple(dict.fromkeys([str(workspace_root), str(cache_root), "/tmp"]))
    shell_file = launch_root / "bash"
    shell_arguments = {"readonly_paths": readonly, "writable_paths": writable, "home": os.environ["HOME"],
                       "initial_cwd": workspace_root, "bash_tmp_root": bash_tmp_root, "native_tmp_root": native_tmp_root}
    if shell_factory is shell_source:
        shell_arguments["environment"] = shell_environment
    else:
        shell_arguments.update(cache_root=cache_root, shell_environment=shell_environment)
    _write(shell_file, shell_factory(**shell_arguments).encode(), executable=True)
    mounts = [BindMount(native_home_root, Path(os.environ["HOME"]), True),
              BindMount(native_config_root, native_config_root, True), BindMount(bash_tmp_root, bash_tmp_root, True),
              BindMount(native_tmp_root, native_tmp_root, True), BindMount(workspace_root, workspace_root, True),
              BindMount(cache_root, cache_root, True), BindMount(source_root, source_root), BindMount(task_root, task_root),
              BindMount(shell_file, Path("/isolation/bash")), BindMount(admission, Path("/isolation/admission")),
              BindMount(Path("/usr/bin/bwrap"), Path("/usr/bin/bwrap")), BindMount(cli_binary, Path("/sdk/claude"))]
    environment = {"HOME": os.environ["HOME"], "SHELL": "/isolation/bash", "CLAUDE_CODE_SHELL": "/isolation/bash",
                   "CLAUDE_CONFIG_DIR": str(native_config_root), "TMPDIR": str(native_tmp_root),
                   "CLAUDE_CODE_TMPDIR": str(native_tmp_root), "DISABLE_AUTOUPDATER": "1",
                   "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    extra_environment = dict(native_environment or {})
    require(not set(extra_environment).intersection(environment), "actor_fixed_environment_changed")
    environment.update(extra_environment)
    command = ["create", "--name", "geak-quality-" + trial_id, "--network", "host", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--security-opt", "seccomp=unconfined",
               "--security-opt", "apparmor=unconfined", "--security-opt", "systempaths=unconfined",
               "--user", f"{os.geteuid()}:{os.getegid()}", "--cpuset-cpus", cpu_set,
               "--shm-size", "8g", "--entrypoint", "/usr/bin/sleep"]
    if memory_bytes is not None:
        require(type(memory_bytes) is int and memory_bytes > 0, "actor_memory_limit_invalid")
        command += ["--memory", str(memory_bytes), "--memory-swap", str(memory_bytes)]
    command += list(device_arguments)
    for key, value in sorted((labels or {}).items()):
        command += ["--label", str(key) + "=" + str(value)]
    for mount in mounts:
        command += ["--mount", f"type=bind,src={mount.source},dst={mount.target}" + ("" if mount.writable else ",readonly")]
    for key, value in sorted(environment.items()):
        command += ["--env", key + "=" + value]
    command += [image_id, "infinity"]
    docker = docker or DockerCommands()
    manifest_path = launch_root / "ACTOR_LAUNCH.json"
    record = {"schema": "geak-quality-actor-launch-v1", "trial_id": trial_id, "image_id": image_id,
              "create_argv": command, "shell_sha256": hashlib.sha256(shell_file.read_bytes()).hexdigest(),
              "cli_sha256": cli_sha256, "provider_credentials_in_argv": False, "status": "creating"}
    _write(manifest_path, (json.dumps(record, indent=2) + "\n").encode())
    cid = None
    try:
        created = docker.run(*command).decode().strip()
        require(re.fullmatch(r"[0-9a-f]{64}", created), "actor_container_id_invalid")
        cid = created
        if reservation is not None:
            reservation.register_actor(cid)
        docker.run("start", cid)
        wrapper_path = private_root / "native_cli.py"
        _write(wrapper_path, cli_wrapper_source(container_id=cid, executable="/sdk/claude", cwd=workspace_root).encode(), executable=True)
        boundary = boundary_factory(container_id=cid, image_id=image_id, mounts=mounts, shell_file=shell_file,
            shell_readonly_paths=readonly, shell_writable_paths=writable, home=os.environ["HOME"], candidate_root=candidate_root,
            journal_root=journal_root, private_roots=(private_root,), admission_dir=admission,
            immutable_paths=tuple(map(_path, immutable_paths)) + (Path("/usr/bin/bwrap"), cli_binary),
            mutable_output_roots=(workspace_root, cache_root), cli_executable="/sdk/claude", cli_sha256=cli_sha256,
            bash_tmp_root=bash_tmp_root, native_tmp_root=native_tmp_root, workspace_mount_root=workspace_root,
            expected_network="host", shell_environment=shell_environment, sdk_wrapper_path=wrapper_path,
            **(boundary_extra or {}))
        record.update(status="ready", container_id=cid, wrapper_sha256=hashlib.sha256(wrapper_path.read_bytes()).hexdigest())
        _write(launch_root / "ACTOR_READY.json", (json.dumps(record, indent=2) + "\n").encode())
        return ActorLaunch(cid, boundary, manifest_path, wrapper_path, docker)
    except BaseException:
        if cid is not None:
            # A reservation owns GPU cleanup. The standalone CPU builder owns
            # only the exact container whose create reply it just received.
            if reservation is not None:
                reservation.block("actor_startup_failed")
            else:
                docker.run("rm", "--force", cid)
        raise
