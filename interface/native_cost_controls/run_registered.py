# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one explicit Linux command through the registered native cache policy.

The command receives its original arguments and inherited standard streams.
Only its ANTHROPIC_BASE_URL changes. Its descendants inherit that endpoint.
The wrapper records no requests, responses, command arguments, or credentials.

Use a noninteractive command that honors the supplied endpoint. The wrapper
acts as a Linux child subreaper. It terminates and reaps remaining descendants,
including detached sessions, before closing the proxy. Terminal job control
and callers with existing children or additional threads are unsupported.
"""

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .cache_proxy import SharedToolCacheProxy
from .native_prefix_marker import DECODER, Decision, sha, wire
from .registered_prefix import RegisteredNativeToolCachePolicy, RegisteredToolPrefix

_TERMINATE_GRACE_SECONDS = 2.0
_PROVIDERS = (
    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS", "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", "CLAUDE_CODE_USE_MANTLE",
)
_TRANSPORT_SETTINGS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
    "NODE_EXTRA_CA_CERTS", "NODE_TLS_REJECT_UNAUTHORIZED", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "ANTHROPIC_UNIX_SOCKET", "CLAUDE_CODE_CLIENT_CERT", "CLAUDE_CODE_CLIENT_KEY",
    "CLAUDE_CODE_CLIENT_CERT_PASSPHRASE", "SSL_CERT_FILE", "SSL_CERT_DIR",
)


class WrapperError(ValueError):
    """Report a fixed setup error without command or catalog contents."""


class RegisteredPrefixSetPolicy:
    """Select the longest exact registered prefix before applying its policy."""

    enabled = True

    def __init__(self, registrations):
        prefixes = tuple(registrations)
        if not prefixes or not all(isinstance(prefix, RegisteredToolPrefix) for prefix in prefixes):
            raise ValueError("Supply at least one verified source catalog registration.")
        # Sorting also makes selection independent of the argument order.
        self.registrations = tuple(sorted(set(prefixes),
            key=lambda prefix: (-prefix.count, prefix.prefix_sha256, prefix.catalog_sha256)))
        self._policies = tuple(RegisteredNativeToolCachePolicy(prefix, enabled=True)
            for prefix in self.registrations)

    def apply(self, raw_body, *, method="POST", path="/v1/messages"):
        if not isinstance(raw_body, bytes):
            raise TypeError("The request body must contain bytes.")
        if method != "POST" or not isinstance(path, str) or path.split("?", 1)[0] != "/v1/messages":
            return Decision(raw_body, False, "unsupported_endpoint")
        try:
            body = DECODER.decode(raw_body.decode("utf-8"))
            if not isinstance(body, dict):
                return Decision(raw_body, False, "unsupported_request")
            tools = body.get("tools")
            if not isinstance(tools, list):
                return Decision(raw_body, False, "prefix_missing")
            for policy in self._policies:
                prefix = policy.prefix
                if len(tools) >= prefix.count and sha(wire(tools[:prefix.count])) == prefix.prefix_sha256:
                    # A matching prefix owns the decision, including a decline.
                    return policy.apply(raw_body, method=method, path=path)
            return Decision(raw_body, False, "prefix_mismatch")
        except (ValueError, TypeError, UnicodeError, KeyError, IndexError, RecursionError):
            return Decision(raw_body, False, "unsupported_json")


def _direct_children():
    """Read authoritative direct child IDs without reading process arguments."""
    main = Path(f"/proc/self/task/{os.getpid()}/children")
    children = {int(pid) for pid in main.read_text().split()}
    for task in Path("/proc/self/task").iterdir():
        if task.name == str(os.getpid()):
            continue
        try:
            children.update(int(pid) for pid in (task / "children").read_text().split())
        except FileNotFoundError:
            # The proxy thread can stop while the wrapper reads the task list.
            continue
    return children


@contextmanager
def _subreaper():
    """Reserve exclusive Linux child ownership before any command starts."""
    if (not sys.platform.startswith("linux") or threading.current_thread() is not threading.main_thread()
            or threading.active_count() != 1 or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL):
        raise WrapperError("The wrapper requires an exclusive Linux process with the default SIGCHLD handler.")
    required = (hasattr(os, "pidfd_open"), hasattr(signal, "pidfd_send_signal"),
        hasattr(os, "waitid"), hasattr(os, "WNOWAIT"))
    if not all(required):
        raise WrapperError("The Linux process controls are unavailable.")
    previous = ctypes.c_int()
    try:
        if len(tuple(Path("/proc/self/task").iterdir())) != 1:
            raise WrapperError("The wrapper requires an exclusive Linux process with no additional threads.")
        if _direct_children():
            raise WrapperError("The wrapper cannot own a process that already has children.")
        probe = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(probe, 0)
        finally:
            os.close(probe)
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        if prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
            raise OSError("The child subreaper state is unavailable.")
        if prctl(36, 1, 0, 0, 0) != 0:
            raise OSError("The child subreaper could not start.")
    except (OSError, AttributeError) as error:
        raise WrapperError("The Linux process controls are unavailable.") from error
    try:
        yield
    finally:
        # Keep adoption enabled if a process-control failure prevents cleanup.
        # Never report that ownership ended while children remain.
        try:
            remaining = _direct_children()
        except OSError as error:
            raise WrapperError("The child ownership state could not be checked.") from error
        if remaining:
            raise WrapperError("The wrapper could not finish child cleanup.")
        if prctl(36, previous.value, 0, 0, 0) != 0:
            raise WrapperError("The child subreaper state could not be restored.")


class _OwnedChildren:
    """Keep pidfds for unreaped direct children and newly adopted descendants."""

    def __init__(self):
        self.handles = {}

    def refresh(self):
        # SIGCHLD stays at SIG_DFL, and this wrapper owns every child. These
        # direct child IDs cannot be reused before this wrapper reaps them.
        for pid in _direct_children():
            if pid not in self.handles:
                self.handles[pid] = os.pidfd_open(pid)

    def send(self, signum, *, only=None, refresh=True):
        if refresh:
            self.refresh()
        sent = set()
        failure = None
        for pid, descriptor in tuple(self.handles.items()):
            if only is not None and pid not in only:
                continue
            try:
                signal.pidfd_send_signal(descriptor, signum)
            except ProcessLookupError:
                pass
            except OSError as error:
                failure = error
                continue
            sent.add(pid)
        if failure is not None:
            raise failure
        return sent

    def reap(self, leader):
        for pid in tuple(self.handles):
            if pid == leader:
                continue
            reaped, _ = os.waitpid(pid, os.WNOHANG)
            if reaped:
                os.close(self.handles.pop(pid))

    def close(self):
        for descriptor in self.handles.values():
            os.close(descriptor)
        self.handles.clear()


def _stop_children(owned, leader, deadline, pending, initial_signal, signaled):
    """Terminate adopted generations before reaping the command leader."""
    while True:
        owned.refresh()
        while pending:
            initial_signal = pending.pop(0)
            signaled.update(owned.send(initial_signal))
        if time.monotonic() >= deadline:
            owned.send(signal.SIGKILL)
        else:
            selected = set(owned.handles) - signaled
            signaled.update(owned.send(initial_signal, only=selected))
        owned.reap(leader)
        signaled.intersection_update(owned.handles)
        # An exiting parent can expose another detached generation. Read the
        # authoritative list again before deciding that cleanup is complete.
        exited = os.waitid(os.P_PID, leader, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        owned.refresh()
        if exited is not None and not set(owned.handles) - {leader}:
            return
        time.sleep(0.02)


def _run_child(command, environment):
    child = None
    pending = []
    interrupted_at = None
    first_signal = None
    previous = {}
    owned = _OwnedChildren()
    signaled = set()

    def forward(signum, _frame):
        nonlocal interrupted_at, first_signal
        if interrupted_at is None:
            interrupted_at = time.monotonic()
            first_signal = signum
        # Never signal from this handler. The main loop owns descriptor use
        # and closure, so a late signal cannot act on a reused PID or pidfd.
        pending.append(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, forward)
        # Do not use preexec_fn. The proxy already owns a server thread.
        child = subprocess.Popen(command, env=environment, start_new_session=True)
        while True:
            owned.refresh()
            while pending:
                signaled.update(owned.send(pending.pop(0)))
            exited = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if exited is not None:
                break
            if interrupted_at is not None and time.monotonic() - interrupted_at >= _TERMINATE_GRACE_SECONDS:
                owned.send(signal.SIGKILL)
            owned.reap(child.pid)
            signaled.intersection_update(owned.handles)
            time.sleep(0.02)
    except OSError as error:
        message = "The child command could not start." if child is None else "The child process controls failed."
        raise WrapperError(message) from error
    finally:
        try:
            if child is not None:
                deadline = (interrupted_at if interrupted_at is not None else time.monotonic())
                try:
                    _stop_children(owned, child.pid, deadline + _TERMINATE_GRACE_SECONDS,
                        pending, first_signal or signal.SIGTERM, signaled)
                except OSError:
                    # A temporary discovery failure must not skip descriptors
                    # whose ownership the wrapper already established.
                    try:
                        try:
                            owned.send(signal.SIGKILL, refresh=False)
                        finally:
                            _stop_children(owned, child.pid, time.monotonic(), pending, signal.SIGKILL, signaled)
                    except OSError as error:
                        raise WrapperError("The wrapper could not finish child cleanup.") from error
                # Reserve the leader's PID until every descendant signal ends.
                returncode = child.wait()
        finally:
            owned.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return returncode if returncode >= 0 else 128 - returncode


def run_registered(command, registrations):
    """Verify the transport before launching the unchanged command arguments."""
    if not command or isinstance(command, (str, bytes)):
        raise WrapperError("Supply a command argument list.")
    policy = RegisteredPrefixSetPolicy(registrations)
    environment = os.environ.copy()
    if any(str(environment.get(name, "")).lower() not in ("", "0", "false") for name in _PROVIDERS):
        raise WrapperError("The provider transport is unsupported.")
    if any(environment.get(name) for name in _TRANSPORT_SETTINGS):
        raise WrapperError("The transport settings are unsupported.")
    upstream = environment.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    with _subreaper(), SharedToolCacheProxy(upstream, policy, enabled=True) as proxy:
        if not proxy.active:
            raise WrapperError("The registered cache proxy could not start.")
        environment["ANTHROPIC_BASE_URL"] = proxy.base_url
        return _run_child(command, environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
        usage="%(prog)s --catalog PATH SHA256 [--catalog PATH SHA256 ...] -- COMMAND [ARG ...]")
    parser.add_argument("--catalog", action="append", nargs=2, required=True,
        metavar=("PATH", "SHA256"), help="Register an absolute catalog path and its SHA-256 hash.")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" not in arguments:
        if "--help" in arguments or "-h" in arguments:
            parser.parse_args(arguments)
        parser.error("Separate the command from the registrations with --.")
    split = arguments.index("--")
    options = parser.parse_args(arguments[:split])
    command = arguments[split + 1:]
    if not command:
        parser.error("Supply a command after --.")
    try:
        if any(not Path(path).is_absolute() for path, _ in options.catalog):
            raise ValueError("The source catalog path must be absolute.")
        registrations = tuple(RegisteredToolPrefix.from_file(path, expected_sha256=expected)
            for path, expected in options.catalog)
    except (OSError, ValueError):
        parser.error("A source catalog could not be verified.")
    try:
        return run_registered(command, registrations)
    except WrapperError as error:
        parser.error(str(error))
    except OSError:
        parser.error("The child command could not start.")


if __name__ == "__main__":
    raise SystemExit(main())
