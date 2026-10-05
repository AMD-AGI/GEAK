# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze an owned Docker actor while the controller stays on the host.

The native CLI uses an immutable shell launcher. That launcher enters a second
mount, PID, user, and network namespace before it executes any command. Native
file tools are disabled. Native journals exist in the outer namespace only.

This module implements the CPU isolation profile. It deliberately rejects GPU
device mappings. A GPU profile also needs independently qualified allocation,
driver-quiescence, and evaluator controls. Freezing a process does not cancel
GPU work that the process already submitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from .quality_stop_controller import ExecutionBoundary, StopRejected, require

ACTOR_TOOLS = ("Workflow", "Bash", "StructuredOutput", "TaskOutput", "TaskList", "TaskGet")
FILE_TOOLS = ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Glob", "Grep")
SHELL_PATH = "/isolation/bash"
ADMISSION_PATH = "/isolation/admission"
PROFILE = "geak-docker-bwrap-cpu-v1"
_SECURITY = frozenset(("no-new-privileges", "seccomp=unconfined", "apparmor=unconfined"))


def _path(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve() == path, "isolation_path_not_canonical")
    return path


def _inside(path, root):
    return path == root or root in path.parents


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class BindMount:
    """Describe one exact bind mount in the trusted launch manifest."""

    source: Path
    target: Path
    writable: bool = False

    def __post_init__(self):
        object.__setattr__(self, "source", _path(self.source))
        require(self.source.is_dir() or self.source.is_file(), "isolation_mount_source_not_regular")
        require(not any(_inside(self.source, Path(root)) for root in ("/proc", "/sys", "/dev")),
                "isolation_host_device_mount_forbidden")
        target = Path(self.target)
        require(target.is_absolute() and ".." not in target.parts, "isolation_target_not_canonical")
        require(type(self.writable) is bool, "isolation_mount_mode_invalid")
        object.__setattr__(self, "target", target)


def shell_source(*, readonly_paths, writable_paths, home, initial_cwd, bash_tmp_root, native_tmp_root, environment=None):
    """Build the immutable inner shell without inheriting secrets or file FDs.

    Paths refer to the outer container. Do not include its journal directory,
    provider credential files, Docker socket, or the host controller directory.
    The launcher never interpolates the command into another shell command.
    """
    ro = tuple(str(Path(path)) for path in readonly_paths)
    rw = tuple(str(Path(path)) for path in writable_paths)
    roots = ro + rw
    bash_tmp, native_tmp = str(_path(bash_tmp_root)), str(_path(native_tmp_root))
    require("/tmp" in rw and "/tmp" not in ro and bash_tmp != native_tmp
            and not _inside(Path(bash_tmp), Path(native_tmp)) and not _inside(Path(native_tmp), Path(bash_tmp)),
            "isolation_private_temporary_roots_required")
    require(re.fullmatch(r"/[A-Za-z0-9_./-]+", native_tmp), "isolation_native_tmp_path_unsupported")
    require(roots and len(set(roots)) == len(roots), "isolation_duplicate_shell_mount")
    for value in (*roots, str(home), str(initial_cwd)):
        require(value.startswith("/") and ".." not in Path(value).parts, "isolation_shell_path_invalid")
    require(all(not _inside(Path("/proc"), Path(value)) and not _inside(Path(value), Path("/proc"))
                and not _inside(Path("/dev"), Path(value)) and not _inside(Path(value), Path("/dev"))
                and value not in {"/", "/run", "/sys"} for value in roots), "isolation_shell_sensitive_mount")
    require(str(home) not in roots, "isolation_native_home_exposed")
    values = {"HOME": str(home), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
              "TMPDIR": "/tmp", "CLAUDE_CODE_TMPDIR": "/tmp"}
    extra = dict(environment or {})
    require(not set(extra).intersection({"HOME", "BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",
                                        "TMPDIR", "CLAUDE_CODE_TMPDIR"}),
            "isolation_shell_environment_unsafe")
    require(all(isinstance(key, str) and isinstance(value, str) and "\x00" not in key + value
                for key, value in extra.items()), "isolation_shell_environment_invalid")
    values.update(extra)
    args = ["/usr/bin/bwrap", "--die-with-parent", "--unshare-all", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for path in ro:
        args += ["--ro-bind", path, path]
    for path in rw:
        args += ["--bind", bash_tmp if path == "/tmp" else path, path]
    args += ["--proc", "/proc", "--dev", "/dev", "--dir", str(home)]
    for key, value in sorted(values.items()):
        args += ["--setenv", key, value]
    # -I prevents inherited PYTHONPATH, user site packages, and Python startup
    # variables from changing the launcher before it clears the environment.
    prefix = ("#!/usr/bin/python3 -I\n"
            "import os, resource, sys, re, secrets, stat, subprocess\n"
            "from pathlib import Path\n"
            "try:\n"
            f"    allowed = open({ADMISSION_PATH + '/open'!r}, 'rb').read() == b'open\\n'\n"
            "except OSError:\n"
            "    allowed = False\n"
            "if not allowed:\n"
            "    sys.stderr.write('The host closed Bash admission.\\n')\n"
            "    sys.exit(125)\n"
            f"roots = {roots!r}\n"
            f"initial = {str(initial_cwd)!r}\n"
            f"bash_tmp = {bash_tmp!r}\n"
            f"native_tmp = {native_tmp!r}\n")
    before = r'''
def inside(path, root):
    return path == root or path.startswith(root + '/')

def directory(path):
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in Path(path).parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

cwd = os.getcwd()
if inside(cwd, bash_tmp):
    cwd = '/tmp' + cwd[len(bash_tmp):]
elif not any(inside(cwd, root) for root in roots if root != '/tmp'):
    cwd = initial
shell_arguments = sys.argv[1:]
native_cwd = inner_name = None
if '-c' in shell_arguments and shell_arguments:
    tail = re.search(r' && pwd -P >\| (' + re.escape(native_tmp) + r'/claude-[0-9a-f]{4}-cwd)$', shell_arguments[-1])
    if tail:
        native_cwd = Path(tail.group(1)).name
        inner_name = '.geak-cwd-' + secrets.token_hex(16)
        shell_arguments[-1] = shell_arguments[-1][:tail.start()] + ' && pwd -P >| /tmp/' + inner_name
limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
os.closerange(3, 1048576 if limit == resource.RLIM_INFINITY else limit)
'''
    after = r'''
result = subprocess.run(args, check=False, env={})
if native_cwd is not None and result.returncode == 0:
    source_dir = target_dir = source_fd = target_fd = None
    try:
        source_dir = directory(bash_tmp)
        try:
            source_fd = os.open(inner_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_dir)
        except FileNotFoundError:
            # A successful exec can replace Bash before the native pwd suffix.
            sys.exit(0)
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid() or info.st_size > 4096:
            raise ValueError('invalid_cwd_file')
        raw = os.read(source_fd, 4097)
        final = os.fstat(source_fd)
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (final.st_size, final.st_mtime_ns, final.st_ctime_ns):
            raise ValueError('changed_cwd_file')
        value = raw.decode('utf-8').strip()
        if not value.startswith('/') or '..' in Path(value).parts or '\x00' in value or '\n' in value:
            raise ValueError('invalid_cwd_value')
        if inside(value, '/tmp'):
            value = bash_tmp + value[len('/tmp'):]
        elif not any(inside(value, root) for root in roots if root != '/tmp'):
            raise ValueError('cwd_outside_actor_roots')
        if not Path(value).is_dir() or str(Path(value).resolve()) != value:
            raise ValueError('cwd_not_canonical_directory')
        target_dir = directory(native_tmp)
        target_fd = os.open(native_cwd, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=target_dir)
        os.write(target_fd, (value + '\n').encode('utf-8'))
        os.fsync(target_fd)
    except (OSError, ValueError, UnicodeError):
        sys.stderr.write('The shell wrapper rejected the directory result.\n')
        sys.exit(125)
    finally:
        for descriptor in (source_fd, target_fd):
            if descriptor is not None:
                os.close(descriptor)
        if source_dir is not None:
            try:
                os.unlink(inner_name, dir_fd=source_dir)
            except FileNotFoundError:
                pass
            os.close(source_dir)
        if target_dir is not None:
            os.close(target_dir)
sys.exit(result.returncode if result.returncode >= 0 else 128 - result.returncode)
'''
    return prefix + before + f"args = {args!r} + ['--chdir', cwd, '/bin/bash'] + shell_arguments\n" + after


def sdk_options(options, *, cli_path):
    """Apply the same actual tool restrictions to both experimental arms."""
    require(options.setting_sources == [] and not options.mcp_servers, "isolation_sdk_settings_unsupported")
    return replace(options, cli_path=str(_path(cli_path)), tools=list(ACTOR_TOOLS),
                   allowed_tools=list(ACTOR_TOOLS), disallowed_tools=list(FILE_TOOLS),
                   strict_mcp_config=True, mcp_servers={})


def cli_wrapper_source(*, container_id, executable, cwd):
    """Connect the host SDK streams to the one native CLI container.

    Docker copies only named environment variables. This source contains no
    provider credential. The inner Bash launcher clears the native environment.
    """
    require(re.fullmatch(r"[a-f0-9]{64}", container_id), "isolation_container_identity_invalid")
    require(str(executable).startswith("/") and str(cwd).startswith("/"), "isolation_cli_path_invalid")
    variables = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS",
                 "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION", "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    return ("#!/usr/bin/python3 -I\n"
            "import os, sys\n"
            f"args = ['/usr/bin/docker', 'exec', '-i', '--workdir', {str(cwd)!r}]\n"
            f"for key in {variables!r}:\n"
            "    if key in os.environ:\n"
            "        args += ['--env', key]\n"
            f"args += [{container_id!r}, {str(executable)!r}] + sys.argv[1:]\n"
            "os.execv(args[0], args)\n")


class DockerActorBoundary(ExecutionBoundary):
    """Inspect and freeze one already-created, owned CPU actor container.

    The trusted launcher creates the container with exact mounts. Its only
    permanent process is `/usr/bin/sleep infinity`. The host SDK starts the CLI
    through `cli_wrapper_source`. Call `bind_cli()` after the CLI starts and
    before the first checkpoint. This checks the actual executable and tool
    arguments. Any other process blocks a checkpoint, including double forks.

    `private_roots` must include the controller state directory. `journal_root`
    is writable in the outer container, but absent from the inner shell mounts.
    Export directories may be writable during search. The freezer protects
    those directories continuously throughout every authority lease. The final
    native-return check seals the immutable shell's admission gate before the
    CLI resumes. The SDK must confirm that return before it disconnects.
    """

    def __init__(self, *, container_id, image_id, mounts, shell_file, shell_readonly_paths,
                 shell_writable_paths, home, candidate_root, journal_root, private_roots, admission_dir,
                 immutable_paths, mutable_output_roots, cli_executable, cli_sha256,
                 bash_tmp_root, native_tmp_root, workspace_mount_root=None, expected_network="none",
                 shell_environment=None, sdk_wrapper_path=None):
        require(re.fullmatch(r"[a-f0-9]{64}", container_id), "isolation_container_identity_invalid")
        require(re.fullmatch(r"sha256:[a-f0-9]{64}", image_id), "isolation_image_identity_invalid")
        require(re.fullmatch(r"[a-f0-9]{64}", cli_sha256), "isolation_cli_hash_invalid")
        require(expected_network in {"none", "host"}, "isolation_network_unsupported")
        self.container_id, self.image_id = container_id, image_id
        self.mounts = tuple(mounts)
        require(self.mounts and all(isinstance(item, BindMount) for item in self.mounts), "isolation_mounts_required")
        require(len({item.target for item in self.mounts}) == len(self.mounts), "isolation_mount_collision")
        self.shell_file, self.candidate_root = _path(shell_file), _path(candidate_root)
        self.bash_tmp_root, self.native_tmp_root = _path(bash_tmp_root), _path(native_tmp_root)
        require(self.bash_tmp_root != self.native_tmp_root
                and not _inside(self.bash_tmp_root, self.native_tmp_root)
                and not _inside(self.native_tmp_root, self.bash_tmp_root), "isolation_temporary_roots_overlap")
        require(len((str(self.native_tmp_root) + "/claude-" + str(os.geteuid())).encode()) <= 44,
                "isolation_native_tmp_path_too_long")
        self.native_output_root = self.native_tmp_root / ("claude-" + str(os.geteuid()))
        self._temporary_identities = {}
        for path in (self.bash_tmp_root, self.native_tmp_root):
            info = path.stat()
            require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700,
                    "isolation_temporary_directory_not_private")
            self._temporary_identities[path] = (info.st_dev, info.st_ino, info.st_uid)
        self.workspace_mount_root = _path(workspace_mount_root or self.candidate_root)
        require(self.workspace_mount_root.is_dir() and _inside(self.candidate_root, self.workspace_mount_root),
                "isolation_candidate_outside_workspace_mount")
        if self.workspace_mount_root != self.candidate_root:
            require(not self.candidate_root.parent.exists() and not self.candidate_root.parent.is_symlink(),
                    "isolation_eval_directory_not_fresh")
        self.admission_dir = _path(admission_dir)
        self.sealed = False
        require((self.admission_dir / "open").read_bytes() == b"open\n", "isolation_admission_not_open")
        self.journal_root = _path(journal_root)
        require(self.journal_root.name == "projects", "isolation_native_journal_layout_unsupported")
        self.private_roots = tuple(_path(path) for path in private_roots)
        require(self.private_roots, "isolation_private_roots_required")
        self.immutable_paths = frozenset(_path(path) for path in immutable_paths) | {self.shell_file}
        self.mutable_output_roots = tuple(_path(path) for path in mutable_output_roots)
        self._immutable_bindings = {}
        for path in self.immutable_paths:
            info = path.stat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "isolation_immutable_source_not_regular")
            self._immutable_bindings[path] = (info.st_dev, info.st_ino, _digest(path))
        for path in self.mutable_output_roots:
            require(all(not _inside(path, root) and not _inside(root, path)
                        for root in (*self.private_roots, *self.immutable_paths)), "isolation_output_overlaps_trusted_state")
        self.shell_readonly_paths = tuple(Path(path) for path in shell_readonly_paths)
        self.shell_writable_paths = tuple(Path(path) for path in shell_writable_paths)
        self.shell_environment = dict(shell_environment or {})
        self.home = str(home)
        self.cli_executable, self.cli_sha256 = str(cli_executable), cli_sha256
        self.sdk_wrapper_path = _path(sdk_wrapper_path) if sdk_wrapper_path is not None else None
        self.expected_network = expected_network
        self.shell_sha256 = _digest(self.shell_file)
        self._cli = None
        self._native_session_id = None
        self._lock = threading.RLock()
        self._leased = False
        self.failed = False
        self.receipts = []
        view = self._inspect()
        self._init = self._process_identity(view["State"]["Pid"])
        require(self._init["argv"] == ["/usr/bin/sleep", "infinity"], "isolation_init_process_changed")
        self._cgroup = self._cgroup_path(self._init["pid"])
        self._verify_container(view, paused=False)
        self._verify_shell_mounts()

    def _docker(self, *args):
        try:
            value = subprocess.run(["/usr/bin/docker", *args], stdin=subprocess.DEVNULL, capture_output=True,
                                   check=False, timeout=20, env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]})
        except (OSError, subprocess.SubprocessError):
            raise StopRejected("isolation_docker_unavailable") from None
        require(value.returncode == 0, "isolation_docker_operation_failed")
        return value.stdout

    def _inspect(self):
        try:
            rows = json.loads(self._docker("inspect", self.container_id))
            require(len(rows) == 1 and rows[0]["Id"] == self.container_id, "isolation_container_identity_changed")
            return rows[0]
        except (ValueError, TypeError, KeyError):
            raise StopRejected("isolation_container_inspection_invalid") from None

    def _verify_container(self, view, *, paused):
        host, config, state = view["HostConfig"], view["Config"], view["State"]
        require(view["Image"] == self.image_id and state["Running"] is True and state["Paused"] is paused
                and state["Restarting"] is False and not state["OOMKilled"], "isolation_container_state_changed")
        require(state["Pid"] == self._init["pid"], "isolation_init_process_replaced")
        require(host["ReadonlyRootfs"] is True and host["Privileged"] is False
                and set(host["CapDrop"] or []) == {"ALL"} and not host["CapAdd"], "isolation_container_privilege_changed")
        security = {item.removesuffix(":true") for item in host["SecurityOpt"] or []}
        require(security == _SECURITY and host["MaskedPaths"] == [] and host["ReadonlyPaths"] == [],
                "isolation_container_security_changed")
        require(host["NetworkMode"] == self.expected_network and host["PidMode"] == ""
                and host["IpcMode"] == "private" and host["UTSMode"] == ""
                and host["CgroupnsMode"] == "private", "isolation_container_namespace_changed")
        self._verify_device_profile(view)
        require(config["User"] == f"{os.geteuid()}:{os.getegid()}", "isolation_actor_user_changed")
        require(config["Entrypoint"] == ["/usr/bin/sleep"] and config["Cmd"] == ["infinity"]
                and not config.get("Healthcheck") and not host.get("Init"), "isolation_launch_process_changed")
        environment = dict(value.split("=", 1) for value in config["Env"] if "=" in value)
        require(environment.get("HOME") == self.home and environment.get("CLAUDE_CODE_SHELL") == SHELL_PATH
                and environment.get("SHELL") == SHELL_PATH
                and environment.get("CLAUDE_CONFIG_DIR") == str(self.journal_root.parent)
                and environment.get("TMPDIR") == str(self.native_tmp_root)
                and environment.get("CLAUDE_CODE_TMPDIR") == str(self.native_tmp_root), "isolation_native_shell_changed")
        expected = {(str(item.source), str(item.target), item.writable) for item in self.mounts}
        actual = {(item["Source"], item["Destination"], item["RW"]) for item in view["Mounts"]}
        require(actual == expected and all(item["Type"] == "bind" and item["Propagation"] == "rprivate"
                                          for item in view["Mounts"]), "isolation_mounts_changed")
        require(any(item.source == self.shell_file and item.target == Path(SHELL_PATH) and not item.writable
                    for item in self.mounts), "isolation_shell_not_immutable")
        require(any(item.source == self.admission_dir and item.target == Path(ADMISSION_PATH) and not item.writable
                    for item in self.mounts), "isolation_admission_not_immutable")
        require((self.admission_dir / "open").read_bytes() == (b"sealed\n" if self.sealed else b"open\n"),
                "isolation_admission_state_changed")
        require(_digest(self.shell_file) == self.shell_sha256, "isolation_shell_source_changed")
        require(any(item.source == self.journal_root.parent and item.target == self.journal_root.parent
                    and item.writable for item in self.mounts), "isolation_journal_host_path_changed")
        for path, identity in self._temporary_identities.items():
            info = path.stat()
            require(path.resolve() == path and (info.st_dev, info.st_ino, info.st_uid) == identity
                    and stat.S_IMODE(info.st_mode) == 0o700,
                    "isolation_temporary_directory_changed")
            require(any(item.source == path and item.target == path and item.writable for item in self.mounts),
                    "isolation_temporary_mount_changed")
        for private in self.private_roots:
            require(all(not _inside(private, item.source) and not _inside(item.source, private) for item in self.mounts),
                    "isolation_private_state_exposed")
        require(self._namespace(self._init["pid"], "pid") != self._namespace(os.getpid(), "pid")
                and self._namespace(self._init["pid"], "mnt") != self._namespace(os.getpid(), "mnt"),
                "isolation_host_memory_exposed")
        self._verify_immutable_sources()

    def _verify_device_profile(self, view):
        """Reject GPU devices unless a qualified subclass supplies OS checks."""
        host = view["HostConfig"]
        require(not host["Devices"] and not host["DeviceRequests"] and not host["DeviceCgroupRules"],
                "isolation_gpu_profile_not_qualified")
        require(not host["GroupAdd"], "isolation_actor_user_changed")

    def _expected_shell_source(self):
        """Return the exact immutable shell source for the default CPU profile."""
        return shell_source(readonly_paths=self.shell_readonly_paths, writable_paths=self.shell_writable_paths,
                            home=self.home, initial_cwd=self.workspace_mount_root, environment=self.shell_environment,
                            bash_tmp_root=self.bash_tmp_root, native_tmp_root=self.native_tmp_root)

    def _verify_immutable_sources(self):
        for path, identity in self._immutable_bindings.items():
            info = path.stat()
            require(path.resolve() == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and (info.st_dev, info.st_ino, _digest(path)) == identity, "isolation_immutable_source_changed")
            for mount in self.mounts:
                if _inside(path, mount.source) or _inside(mount.source, path):
                    require(not mount.writable, "isolation_immutable_source_writable")
                    target = mount.target / path.relative_to(mount.source) if _inside(path, mount.source) else mount.target
                    require(all(not _inside(target, root) and not _inside(root, target)
                                for root in self.shell_writable_paths), "isolation_immutable_shell_alias_writable")

    @staticmethod
    def _namespace(pid, kind):
        return os.readlink(f"/proc/{pid}/ns/{kind}")

    def _verify_shell_mounts(self):
        shell_paths = self.shell_readonly_paths + self.shell_writable_paths
        for path in shell_paths:
            require(path not in {Path("/"), Path("/proc"), Path("/sys"), Path("/run"), Path(self.home)},
                    "isolation_shell_sensitive_mount")
            visible_path = self.bash_tmp_root if path == Path("/tmp") else path
            for mount in self.mounts:
                if _inside(mount.target, visible_path):
                    require(not _inside(self.journal_root, mount.source) and not _inside(mount.source, self.journal_root),
                            "isolation_native_journal_exposed")
                    require(not _inside(self.native_tmp_root, mount.source) and not _inside(mount.source, self.native_tmp_root),
                            "isolation_native_output_exposed")
                if _inside(visible_path, mount.target):
                    source = mount.source / visible_path.relative_to(mount.target)
                    require(not _inside(source, self.journal_root) and not _inside(self.journal_root, source),
                            "isolation_native_journal_exposed")
                    require(not _inside(source, self.native_tmp_root) and not _inside(self.native_tmp_root, source),
                            "isolation_native_output_exposed")
        source = self._expected_shell_source()
        require(self.shell_file.read_text() == source, "isolation_shell_policy_changed")
        require(any(item.source == self.workspace_mount_root and item.target == self.workspace_mount_root and item.writable
                    for item in self.mounts) and self.workspace_mount_root in self.shell_writable_paths,
                "isolation_candidate_mount_changed")

    @staticmethod
    def _cgroup_path(pid):
        lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        require(len(lines) == 1 and lines[0].startswith("0::/"), "isolation_cgroup_v2_required")
        relative = Path(lines[0][3:])
        require(".." not in relative.parts and relative != Path("/"), "isolation_cgroup_path_invalid")
        return Path("/sys/fs/cgroup") / str(relative).lstrip("/")

    @staticmethod
    def _process_identity(pid):
        require(type(pid) is int and pid > 1, "isolation_process_identity_invalid")
        base = Path(f"/proc/{pid}")
        raw = (base / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        command = (base / "cmdline").read_bytes()
        require(command.endswith(b"\0"), "isolation_process_arguments_invalid")
        argv = [value.decode() for value in command[:-1].split(b"\0")]
        require(fields[0] not in {"Z", "X"}, "isolation_process_not_live")
        return {"pid": pid, "start_ticks": fields[19], "exe_sha256": _digest(base / "exe"), "argv": argv}

    def _pids(self):
        rows = self._docker("top", self.container_id, "-eo", "pid").decode().splitlines()
        require(rows and rows[0].strip() == "PID", "isolation_process_census_invalid")
        try:
            pids = {int(row.strip()) for row in rows[1:]}
        except ValueError:
            raise StopRejected("isolation_process_census_invalid") from None
        require(pids and pids == {int(value) for value in (self._cgroup / "cgroup.procs").read_text().split()},
                "isolation_process_census_incomplete")
        return pids

    def bind_cli(self):
        """Pin the actual native process before any model work can certify."""
        with self._lock:
            require(self._cli is None and not self._leased and not self.failed, "isolation_cli_already_bound")
            matches = []
            for pid in self._pids() - {self._init["pid"]}:
                item = self._process_identity(pid)
                if item["exe_sha256"] == self.cli_sha256:
                    matches.append(item)
            require(len(matches) == 1, "isolation_native_cli_missing")
            item = matches[0]
            args = item["argv"]
            require(args[0] == self.cli_executable and "--output-format" in args and "stream-json" in args,
                    "isolation_native_cli_arguments_changed")
            for option, expected in (("--tools", ",".join(ACTOR_TOOLS)), ("--disallowedTools", ",".join(FILE_TOOLS))):
                require(args.count(option) == 1 and args[args.index(option) + 1] == expected,
                        "isolation_native_tools_not_restricted")
            require([value for value in args if value.startswith("--setting-sources")] == ["--setting-sources="]
                    and "--strict-mcp-config" in args, "isolation_native_settings_not_restricted")
            sessions = [value.removeprefix("--session-id=") for value in args if value.startswith("--session-id=")]
            require(len(sessions) == 1
                    and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", sessions[0])
                    and self._native_session_id in (None, sessions[0]), "isolation_native_session_changed")
            self._native_session_id = sessions[0]
            self._verify_cli_environment(item["pid"])
            self._cli = item

    def prepare_sdk_options(self, options):
        """Bind the private SDK wrapper and apply the common tool profile."""
        path = self.sdk_wrapper_path
        require(path is not None and path.is_file() and path.resolve() == path, "isolation_sdk_wrapper_required")
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.geteuid(),
                "isolation_sdk_wrapper_not_regular")
        require(all(not _inside(path, mount.source) for mount in self.mounts), "isolation_sdk_wrapper_exposed")
        expected = cli_wrapper_source(container_id=self.container_id, executable=self.cli_executable,
                                      cwd=self.workspace_mount_root)
        require(path.read_text() == expected and str(options.cwd) == str(self.workspace_mount_root),
                "isolation_sdk_wrapper_changed")
        require(isinstance(options.session_id, str)
                and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", options.session_id),
                "isolation_native_session_not_predeclared")
        self._native_session_id = options.session_id
        prepared = sdk_options(options, cli_path=path)
        environment = dict(prepared.env or {})
        environment["CLAUDE_CONFIG_DIR"] = str(self.journal_root.parent)
        return replace(prepared, env=environment)

    def native_started(self):
        """Pin the real native CLI before the SDK submits the first query."""
        self.bind_cli()

    def read_native_output(self, value):
        """Read only a stable native task output through protected descriptors."""
        path = _path(value)
        require(_inside(path, self.native_output_root) and path.parent.name == "tasks" and path.suffix == ".output",
                "isolation_native_output_path_invalid")
        require(self._native_session_id is not None and path.parts[-3] == self._native_session_id,
                "isolation_native_output_session_changed")
        info = self.native_tmp_root.stat()
        require((info.st_dev, info.st_ino, info.st_uid) == self._temporary_identities[self.native_tmp_root]
                and stat.S_IMODE(info.st_mode) == 0o700,
                "isolation_temporary_directory_changed")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory = os.open("/", flags)
        descriptor = None
        try:
            for part in path.parts[1:-1]:
                child = os.open(part, flags, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid() and before.st_nlink == 1
                    and before.st_size <= 64 * 1024 * 1024, "isolation_native_output_not_regular")
            chunks, remaining = [], before.st_size
            while remaining:
                part = os.read(descriptor, min(remaining, 1024 * 1024))
                require(bool(part), "isolation_native_output_changed")
                chunks.append(part)
                remaining -= len(part)
            after = os.fstat(descriptor)
            current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
            require(identity(before) == identity(after) == identity(current) and path.resolve() == path,
                    "isolation_native_output_changed")
            root_info = self.native_tmp_root.stat()
            require((root_info.st_dev, root_info.st_ino, root_info.st_uid) == self._temporary_identities[self.native_tmp_root],
                    "isolation_temporary_directory_changed")
            return b"".join(chunks).decode("utf-8")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory)

    def _verify_cli_environment(self, pid):
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        environment = dict(value.split(b"=", 1) for value in raw.split(b"\0") if b"=" in value)
        require(environment.get(b"HOME") == self.home.encode()
                and environment.get(b"SHELL") == SHELL_PATH.encode()
                and environment.get(b"CLAUDE_CODE_SHELL") == SHELL_PATH.encode()
                and environment.get(b"CLAUDE_CONFIG_DIR") == str(self.journal_root.parent).encode()
                and environment.get(b"TMPDIR") == str(self.native_tmp_root).encode()
                and environment.get(b"CLAUDE_CODE_TMPDIR") == str(self.native_tmp_root).encode(),
                "isolation_native_shell_override_changed")

    def _verify_processes(self):
        require(self._cli is not None, "isolation_native_cli_unbound")
        self._verify_cli_environment(self._cli["pid"])
        require(self._pids() == {self._init["pid"], self._cli["pid"]}, "isolation_unfinished_actor_process")
        for expected in (self._init, self._cli):
            require(self._process_identity(expected["pid"]) == expected, "isolation_native_process_replaced")
            for entry in Path(f"/proc/{expected['pid']}/fd").iterdir():
                try:
                    name = os.readlink(entry)
                    info = entry.stat()
                except FileNotFoundError:
                    continue
                require(not name.startswith(("/dev/kfd", "/dev/dri/"))
                        and not (stat.S_ISCHR(info.st_mode) and os.major(info.st_rdev) == 226),
                        "isolation_native_gpu_descriptor")

    def _verify_frozen(self):
        require(self._leased and not self.failed, "isolation_lease_missing")
        values = dict(line.split() for line in (self._cgroup / "cgroup.events").read_text().splitlines())
        require(values.get("frozen") == "1" and values.get("populated") == "1", "isolation_freezer_not_active")
        self._verify_container(self._inspect(), paused=True)

    def check(self, *, stage, candidate_root, protected_paths):
        require(_path(candidate_root) == self.candidate_root, "isolation_candidate_changed")
        require(self.candidate_root.is_dir(), "isolation_candidate_missing")
        self._verify_frozen()
        self._verify_processes()
        for raw in protected_paths:
            path = _path(raw)
            require(path in self.immutable_paths or any(_inside(path, root)
                    for root in (*self.private_roots, *self.mutable_output_roots)), "isolation_protected_path_unclassified")
            for item in self.mounts:
                if _inside(path, item.source) and item.writable:
                    require(any(_inside(path, root) for root in self.mutable_output_roots),
                            "isolation_unclassified_protected_mount")
        if stage == "native_return":
            self._seal_admission()
        self.receipts.append({"profile": PROFILE, "stage": str(stage), "container_id": self.container_id,
                              "image_id": self.image_id, "shell_sha256": self.shell_sha256,
                              "frozen": True, "processes": [{key: value for key, value in item.items() if key != "argv"}
                                                              for item in (self._init, self._cli)],
                              "gpu_qualified": False})

    def _seal_admission(self):
        """Close every future Bash start before the frozen actor can resume."""
        self._verify_frozen()
        if self.sealed:
            return
        target = self.admission_dir / ".closed.tmp"
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"sealed\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(target, self.admission_dir / "open")
        directory = os.open(self.admission_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.sealed = True

    @contextmanager
    def lease(self, *, candidate_root, protected_paths):
        """Hold the kernel cgroup freezer across snapshot, processes, and signing."""
        with self._lock:
            require(not self.failed and not self._leased, "isolation_lease_unavailable")
            self._verify_container(self._inspect(), paused=False)
            self._docker("pause", self.container_id)
            self._leased = True
            try:
                self.check(stage="lease_enter", candidate_root=candidate_root, protected_paths=protected_paths)
                yield
                self.check(stage="lease_exit", candidate_root=candidate_root, protected_paths=protected_paths)
            finally:
                try:
                    self._verify_frozen()
                    self._docker("unpause", self.container_id)
                    self._verify_container(self._inspect(), paused=False)
                except BaseException:
                    self.failed = True
                    raise
                finally:
                    self._leased = False
