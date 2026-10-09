#!/usr/bin/env python3
"""Client for the GPU broker: a context manager, a CLI, and the fallback that makes it optional.

    # as a library
    from gpu_client import lease
    with lease(kind="exclusive", group="kernel_name", priority="measure") as g:
        run_the_benchmark(gpu=g.lane)          # g.lane is the granted card

    # as a CLI -- runs a command holding a lease, which is what gpu_lock.sh delegates to
    python3 gpu_client.py run --group kernel_name --priority measure -- python bench.py

    # residency: park, then wake per reading
    with lease(kind="resident", vram_mb=8000, group="kernel_name") as park:
        for cfg in configs:
            with park.wake(priority="wake"):   # short exclusive window on the SAME lane
                measure(cfg)

OFF BY DEFAULT IN GEAK. The broker is an opt-in layer over GEAK's single GPU lock
(kernel_workflow/scripts/gpu_lock.sh, lock dir /tmp/team_gpu_locks). Nothing here connects unless
GEAK_GPU_BROKER=1, and nothing spawns a daemon unless GEAK_GPU_BROKER_AUTOSTART=1 as well; with the
broker off every call degrades to the unbrokered path below.

THE FALLBACK IS THE POINT. Every entry here degrades to "no broker, carry on" rather than failing.
If the daemon is not running and cannot be started, `lease()` yields a lease with `lane=None` and
`brokered=False`, and the caller proceeds exactly as it did before this file existed. That is what
lets `gpu_lock.sh` delegate when GEAK_GPU_BROKER=1: on a box with a broker you get fleet-wide
scheduling, on a box without one you get the flock path, and no call site has to know which.

WHY A HEARTBEAT THREAD. A lease has a TTL so that a client which dies without closing its socket
cannot hold a lane forever. But a legitimate 40-minute climb would then lose its lease mid-run, so
the holder renews in the background at a third of the TTL. Three missed renewals and the broker
reclaims -- which is the intended behaviour for a wedged client and is survivable for a slow one.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from group_identity import resolve_group  # noqa: E402

RUNTIME_ROOT = os.environ.get(
    "GEAK_GPU_RUNTIME_ROOT",
    os.path.join(os.path.expanduser("~"), ".cache", "tile-runtime", "gpu-broker"),
)
CANONICAL_SOCK = os.path.join(RUNTIME_ROOT, "gpu.sock")
DEFAULT_SOCK = os.environ.get("GEAK_GPU_BROKER_SOCK", CANONICAL_SOCK)
# OPT-IN in GEAK: GEAK's gpu_lock.sh (flock on /tmp/team_gpu_locks) is the default allocator and the
# broker is a layer a campaign turns on with GEAK_GPU_BROKER=1. AUTOSTART is likewise off: a daemon
# is started deliberately (`gpu_broker.py --serve`) or with GEAK_GPU_BROKER_AUTOSTART=1.
# Both are read LIVE (see _env_flag below) so exporting them after import takes effect.
ENABLED_DEFAULT = "0"
AUTOSTART_DEFAULT = "0"

MARKER_NAME = ".geak_gpu_sock"
# ADOPTING THE CANONICAL SOCKET BY DEFAULT IS THE WRONG DEGRADE, and it is silent both ways.
#
# The canonical socket ($GEAK_GPU_RUNTIME_ROOT/gpu.sock, default
# ~/.cache/tile-runtime/gpu-broker/gpu.sock) belongs to whichever campaign on the box started first. A run that was
# handed some OTHER pool and never wrote a marker connects to it anyway: if the lanes overlap it
# queues behind a stranger's work, and if they do not every acquire comes back "no such lane(s)" and
# the client degrades to UNBROKERED -- an unarbitrated card -- while the log still reads "broker
# ready". Measured: a GPU-0 fleet adopted a broker serving lanes 6,7 and ran its whole
# sweep on the flock path with no queueing and no fair share.
#
# Measured again, separately: a campaign started NO broker of its own and wrote no marker, while
# the canonical socket on that box had belonged to an unrelated fleet for weeks. Every
# reading of that campaign was one connect() away from a foreign pool.
#
# So the canonical path is only adopted when SOMEBODY NAMED IT -- `GEAK_GPU_BROKER_SOCK`, or a
# `.geak_gpu_sock` marker at a parent of $PWD. Unnamed, this refuses and the caller degrades to the
# flock path it always took, which is correct-but-slower rather than fast-and-contaminated.
# `GEAK_GPU_REQUIRE_MARKER=0` restores the old adopt-anything behaviour for a single-tenant box.
#
# Read LIVE rather than bound at import. Binding early has already cost a debugging round (see
# residency.py's note): an agent that exports the variable after the first import changes nothing.
def _env_flag(name, default="1"):
    return os.environ.get(name, default) not in ("0", "false", "no", "")


def enabled():
    """GEAK_GPU_BROKER=1 turns the client on; unset/0 keeps every call unbrokered."""
    return _env_flag("GEAK_GPU_BROKER", ENABLED_DEFAULT)


def autostart_enabled():
    return _env_flag("GEAK_GPU_BROKER_AUTOSTART", AUTOSTART_DEFAULT)


REQUIRE_MARKER = _env_flag("GEAK_GPU_REQUIRE_MARKER")   # the documented default; re-read per call
_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg, file=sys.stderr)


def _run_state_heartbeat(phase, lease_id, next_step, last_artifact=None):
    """Best-effort fleet observability; never changes broker correctness."""
    state = os.environ.get("TILE_RUN_STATE")
    if not state:
        return
    tool = os.environ.get("TILE_RUN_STATE_TOOL")
    if not tool:
        candidate = os.path.join(os.path.dirname(HERE), "scripts", "run_state.py")
        tool = candidate if os.path.isfile(candidate) else ""
    if not tool or not os.path.isfile(tool):
        return
    owner = os.environ.get("TILE_RUN_OWNER", "gpu-client")
    argv = [sys.executable, tool, "heartbeat", "--state", state, "--phase", phase,
            "--owner", owner, "--next-step", next_step, "--lease", str(lease_id)]
    if last_artifact:
        argv += ["--last-artifact", last_artifact]
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False)


def discover_sock(start=None):
    """`(sock_path, source)` -- which broker this run belongs to, and who said so.

    The same two sources, in the same order, as `gpu_lock.sh`: an explicit `GEAK_GPU_BROKER_SOCK`,
    else the nearest `.geak_gpu_sock` marker at or above `start`. `source` is `"env"` / `"marker"` /
    `"unnamed"`, and the third is what `connect()` refuses to adopt the canonical socket on.
    """
    s = os.environ.get("GEAK_GPU_BROKER_SOCK", "").strip()
    if s:
        return s, "env"
    d = os.path.abspath(start or os.getcwd())
    while d and d != os.path.dirname(d):
        marker = os.path.join(d, MARKER_NAME)
        if os.path.isfile(marker):
            try:
                with open(marker) as fh:
                    v = fh.readline().strip()
                if v:
                    return v, "marker"
            except OSError:
                pass
        d = os.path.dirname(d)
    return "", "unnamed"


class BrokerUnavailable(Exception):
    pass


class Conn:
    """One connection. LEASES ARE SCOPED TO IT -- closing it releases everything, which is exactly
    the property that makes crash recovery free."""

    def __init__(self, sock_path=DEFAULT_SOCK, label=None, timeout=30.0):
        self.sock_path = sock_path
        self.label = label or f"pid{os.getpid()}"
        self._s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._s.settimeout(timeout)
        try:
            self._s.connect(sock_path)
        except OSError as exc:
            raise BrokerUnavailable(str(exc)) from exc
        self._f = self._s.makefile("rwb")
        self._lock = threading.Lock()
        self._labelled = False

    def call(self, req, timeout=None):
        with self._lock:
            if not self._labelled:
                req = dict(req, label=self.label)
                self._labelled = True
            if timeout is not None:
                self._s.settimeout(timeout)
            try:
                self._f.write((json.dumps(req) + "\n").encode())
                self._f.flush()
                line = self._f.readline()
            except OSError as exc:
                raise BrokerUnavailable(str(exc)) from exc
            if not line:
                raise BrokerUnavailable("broker closed the connection")
            return json.loads(line.decode())

    def close(self):
        with contextlib.suppress(Exception):
            self._f.close()
        with contextlib.suppress(Exception):
            self._s.close()


def _spawn_broker(sock_path):
    """Start a daemon if none is listening. Idempotent by construction: the daemon itself probes the
    socket and exits quietly if somebody beat it there, so a race between eight simultaneous clients
    ends with one daemon and seven silent exits. The flock is belt-and-braces to keep that cheap."""
    lockf = sock_path + ".start.lock"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(sock_path)), exist_ok=True)
        import fcntl
        fh = open(lockf, "w")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            time.sleep(1.5)          # somebody else is starting it; give them a moment
            return
        try:
            subprocess.Popen(
                [sys.executable, os.path.join(HERE, "gpu_broker.py"), "--serve",
                 "--sock", sock_path],
                stdout=subprocess.DEVNULL,
                stderr=open(os.path.join(RUNTIME_ROOT, "broker.log"), "a"),
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        finally:
            # Hold the lock across the sleep so the losers wait for a listening socket, not just for
            # the fork to return.
            for _ in range(60):
                if os.path.exists(sock_path):
                    break
                time.sleep(0.1)
            with contextlib.suppress(Exception):
                fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
    except Exception:                                            # noqa: BLE001
        pass


def _connect_once(sock_path, label):
    return Conn(sock_path, label=label)


def connect(sock_path=DEFAULT_SOCK, label=None, autostart=None, attempts=None):
    """Connect, retrying a REFUSED connection before giving up.

    A refusal is not the same as an absence, and conflating them is expensive here. When many
    clients connect at once some will find the listen backlog full and be refused -- a transient,
    entirely recoverable condition. Treating that as "no broker" makes the client fall back to
    UNBROKERED and run its benchmark on an unarbitrated card, which is the exact contamination the
    daemon exists to prevent, appearing only under load and leaving no trace in the artifact.

    Measured before this retry existed: 32 simultaneous clients, 8 silently unbrokered.

    A genuinely absent daemon still fails fast, because all the retries together are under a
    second -- so the no-broker fallback stays immediate, which the tests assert.
    """
    if not enabled():
        raise BrokerUnavailable("broker disabled (GEAK_GPU_BROKER is not 1; off by default in GEAK)")
    # WHOSE POOL IS THIS. Refusing an unnamed canonical socket is a correctness guard, not a policy:
    # see REQUIRE_MARKER above for the two measured incidents. The refusal is loud once per process,
    # because the failure it replaces was silent and the fix is one line at the campaign root.
    if _env_flag("GEAK_GPU_REQUIRE_MARKER") and sock_path == CANONICAL_SOCK:
        named, source = discover_sock()
        if source == "unnamed":
            _warn_once("unnamed-canonical", (
                f"[gpu_client] REFUSING the canonical socket {CANONICAL_SOCK}: nothing named it. That "
                f"socket belongs to whichever campaign on this box started first, and adopting it "
                f"either queues this run behind a stranger or returns 'no such lane(s)' and degrades "
                f"to an UNARBITRATED card while the log still says a broker is up.\n"
                f"  Name the pool this run belongs to, at the campaign root:\n"
                f"    printf '%s\\n' <sock-path> > <campaign-root>/{MARKER_NAME}\n"
                f"  or export GEAK_GPU_BROKER_SOCK, or set GEAK_GPU_REQUIRE_MARKER=0 on a "
                f"single-tenant box.\n"
                f"  Falling back to the flock path -- correct, and without queueing or fair share."))
            raise BrokerUnavailable(
                f"canonical socket not adopted: no {MARKER_NAME} marker and no GEAK_GPU_BROKER_SOCK")
        if named and os.path.abspath(named) != os.path.abspath(sock_path):
            # The marker names a DIFFERENT broker than the caller asked for. Honour the marker: it is
            # the campaign's own statement of which pool it is a tenant of.
            sock_path = named
    # Autostart applies to the CANONICAL socket only. A caller that names some other path is asking
    # to talk to a specific broker -- a test fixture, a second fleet, a debugging instance -- and
    # conjuring a fresh daemon there is never what it meant. Caught by the client selftest, which
    # pointed at a deliberately absent socket to prove the fallback and instead got a live broker
    # spawned on it; the fallback path was consequently never exercised at all.
    want_start = (autostart_enabled() if autostart is None else autostart) and sock_path == DEFAULT_SOCK
    n = int(attempts if attempts is not None
            else os.environ.get("GEAK_GPU_BROKER_CONNECT_ATTEMPTS", "6"))

    def dial():
        return _connect_once(sock_path, label)

    # Only retry when the socket EXISTS -- a full backlog is transient, a missing file is not, and
    # retrying the latter would turn the no-broker fallback into a multi-second stall on every call.
    if os.path.exists(sock_path):
        try:
            return with_retry(dial, attempts=n, base_s=0.05, cap_s=0.8,
                              retry_on=(BrokerUnavailable,))
        except BrokerUnavailable:
            if not want_start:
                raise
    else:
        try:
            return dial()
        except BrokerUnavailable:
            if not want_start:
                raise
    _spawn_broker(sock_path)
    return with_retry(dial, attempts=n, base_s=0.05, cap_s=0.8,
                      retry_on=(BrokerUnavailable,))


class Lease:
    """A granted lease with a background heartbeat. `lane is None` means unbrokered: no daemon was
    reachable, the caller should carry on with whatever it did before."""

    def __init__(self, conn=None, lease_id=None, lane=None, kind=None, ttl_s=900.0,
                 waited_s=0.0, group=None, sock_path=None, token=None,
                 denied_reason=None, lane_mismatch=False, sock_source=None):
        self.conn, self.lease_id, self.lane, self.kind = conn, lease_id, lane, kind
        self.ttl_s, self.waited_s, self.group = ttl_s, waited_s, group
        self.brokered = conn is not None and lease_id is not None
        # WHY this lease is unbrokered, when it is. An unbrokered lease used to be indistinguishable
        # from "no daemon on this box", and the two call for opposite responses: no daemon is fine and
        # expected, whereas a daemon that owns none of our lanes means the run adopted the wrong pool
        # and every reading after this one is unarbitrated. `lane_mismatch` is that second case.
        self.denied_reason = denied_reason
        self.lane_mismatch = bool(lane_mismatch)
        self.sock_source = sock_source
        self.sock_path = sock_path or getattr(conn, "sock_path", DEFAULT_SOCK)
        # Authorises RENEW from the heartbeat's own socket, which is a different `owner` than the
        # connection that won the lease. Renew-only: it cannot release or acquire.
        self.token = token
        # Set when the broker says this lease is gone (reclaimed, or released by someone else).
        # A holder that keeps computing after that is writing to a lane the broker has since given
        # to somebody else, so `lost` is what callers check before trusting their own grant.
        self.lost = False
        self._stop = threading.Event()
        self._hb = None
        self._hbconn = None
        if self.brokered:
            self._hb = threading.Thread(target=self._beat, daemon=True)
            self._hb.start()

    def _beat(self):
        """Renew on a DEDICATED connection, never the caller's.

        `Conn.call` holds a per-connection mutex for the whole round-trip, so sharing one socket
        means the heartbeat cannot fire while any other call is in flight. That is not a rare
        window -- it is the steady state of this system: a resident parks, then issues a BLOCKING
        `acquire` for its measurement window, which by design waits until a lane frees. The renew
        thread sits behind that mutex for the entire wait, the TTL lapses, and the resident loses
        the VRAM it is still holding. Measured: a 4 s TTL, a 14 s queue, resident gone.

        The failure is worse than a lost lease. The process does not know, keeps its tensors, and
        the broker hands its lane to someone else -- two tenants computing on one card, which is the
        exact contamination the lease exists to prevent.

        A second socket costs one fd per lease and is immune by construction.
        """
        period = max(2.0, self.ttl_s / 3.0)
        while not self._stop.wait(period):
            try:
                if self._hbconn is None:
                    self._hbconn = Conn(self.sock_path, label=f"hb:{self.lease_id}", timeout=15.0)
                r = self._hbconn.call({"op": "renew", "lease": self.lease_id,
                                       "ttl_s": self.ttl_s, "token": self.token})
                if isinstance(r, dict) and r.get("ok") is False:
                    # The broker no longer knows this lease. It was reclaimed while we were slow, so
                    # the lane may already belong to someone else. Say so loudly rather than
                    # continuing to act as a holder.
                    self.lost = True
                    return
            except Exception:                                    # noqa: BLE001
                # Broker unreachable. Drop the cached socket and retry on the next beat rather than
                # giving up for good -- a daemon restart should not silently end a 40-minute climb's
                # renewals, which would then expire it the moment the daemon came back.
                try:
                    if self._hbconn is not None:
                        self._hbconn.close()
                except Exception:                                # noqa: BLE001
                    pass
                self._hbconn = None

    @contextlib.contextmanager
    def wake(self, priority="wake", timeout_s=600.0):
        """Take a SHORT exclusive window on the lane this resident already occupies.

        This is the "park in VRAM, wake to measure" half of the design. The process stays warm --
        torch imported, CUDA context up, input tensors built -- and pays only the queueing cost of a
        brief exclusive window per reading. `want_lane` is the lane we are parked on, and the broker
        will not substitute another: our VRAM is here, so a grant elsewhere would be unusable.
        """
        if not self.brokered or self.kind != "resident":
            yield self          # unbrokered, or not a resident: nothing to coordinate
            return
        r = self.conn.call({"op": "acquire", "kind": "exclusive", "want_lane": self.lane,
                            "lanes": [self.lane], "priority": priority,
                            "group": self.group, "timeout_s": timeout_s,
                            "meta": {"command_kind": "resident_wake",
                                     "estimated_runtime_s": min(float(timeout_s), 119.0),
                                     "safe_yield_kind": "resident_wake_boundary",
                                     "resident_event": "wake",
                                     "reason": "resident_measurement"}},
                           timeout=timeout_s + 30)
        if not r.get("ok"):
            # Degrade rather than fail: a measurement taken without the window is still a
            # measurement, and the alternative is losing the whole sweep to a busy box. The caller
            # can see it happened via the returned flag.
            yield self
            return
        sub = Lease(self.conn, r["lease"], r["lane"], "exclusive", r.get("ttl_s", 900.0),
                    r.get("waited_s", 0.0), self.group, sock_path=self.sock_path,
                    token=r.get("token"))
        try:
            yield sub
        finally:
            sub.boundary("yield")
            sub.release()  # stops its heartbeat; a yielded lease has already been released server-side

    def boundary(self, action="status"):
        """Participate in the broker's advisory safe-boundary protocol.

        The broker never stops a GPU process. Callers that advertise a safe yield boundary poll
        here and explicitly choose ``continue``, ``yield``, or ``cancel`` when asked.
        """
        if not self.brokered:
            return {"ok": True, "lease": self.lease_id, "pause_requested": None,
                    "next": "continue"}
        try:
            return self.conn.call({"op": "boundary", "lease": self.lease_id, "action": action})
        except BrokerUnavailable as exc:
            return {"ok": False, "error": str(exc)}

    def release(self):
        self._stop.set()
        if self.brokered:
            with contextlib.suppress(Exception):
                self.conn.call({"op": "release", "lease": self.lease_id})
            self.brokered = False
        # The heartbeat's own socket. One fd per lease is cheap; leaking one per lease across a
        # sweep's thousands of measurement windows is not.
        if self._hbconn is not None:
            with contextlib.suppress(Exception):
                self._hbconn.close()
            self._hbconn = None


@contextlib.contextmanager
def lease(kind="exclusive", group=None, priority="measure", lanes=None, vram_mb=0,
          want_lane=None, timeout_s=1800.0, label=None, sock_path=DEFAULT_SOCK,
          own_conn=True, conn=None, require_group=False, command_kind=None,
          estimated_runtime_s=None, safe_yield_kind=None, meta=None):
    """Acquire, yield, release. Never raises for want of a broker -- see the module docstring."""
    resolved_group = group
    if not str(resolved_group or "").strip():
        resolved_group, _ = resolve_group()
    if require_group and not str(resolved_group or "").strip():
        yield Lease(denied_reason="group is required but no stable group identity was found")
        return

    request_meta = dict(meta or {})
    for key, value in (("command_kind", command_kind),
                       ("estimated_runtime_s", estimated_runtime_s),
                       ("safe_yield_kind", safe_yield_kind)):
        if value not in (None, ""):
            request_meta[key] = value
    c, mine = conn, False
    try:
        if c is None:
            c = connect(sock_path, label=label or resolved_group)
            mine = own_conn
    except BrokerUnavailable as exc:
        # unbrokered: lane=None, brokered=False. The reason travels so a caller can tell "no daemon"
        # from "we refused to adopt somebody else's".
        yield Lease(denied_reason=str(exc))
        return

    lv = None
    try:
        r = c.call({"op": "acquire", "kind": kind, "group": resolved_group, "priority": priority,
                    "lanes": list(lanes) if lanes else None, "vram_mb": int(vram_mb or 0),
                    "want_lane": want_lane, "timeout_s": timeout_s,
                    "require_group": bool(require_group), "meta": request_meta},
                   timeout=timeout_s + 30)
        if not r.get("ok"):
            why = str(r.get("error") or "acquire refused")
            # A LANE MISMATCH IS NOT A BUSY BOX. "no such lane(s)" means this broker owns none of the
            # cards we were handed, i.e. we are talking to the wrong pool -- and the old silent
            # degrade turned that into an unarbitrated measurement with a reassuring log line.
            mismatch = "no such lane" in why
            if mismatch:
                _warn_once(f"lane-mismatch:{c.sock_path}", (
                    f"[gpu_client] LANE MISMATCH on {c.sock_path}: {why}. This broker owns none of "
                    f"the lanes this run was given, so it is not this run's pool -- every reading "
                    f"from here is UNARBITRATED even though a broker is up. Point "
                    f"{MARKER_NAME}/GEAK_GPU_BROKER_SOCK at the broker that owns these lanes, or "
                    f"start one for them."))
            yield Lease(denied_reason=why, lane_mismatch=mismatch,
                        sock_source=getattr(c, "sock_path", None))
            return
        # The connection's OWN path, not the one asked for: `connect()` follows a `.geak_gpu_sock`
        # marker when the caller named the canonical socket, and recording the requested path would
        # attribute the lease to a broker it was never taken from.
        lv = Lease(c, r["lease"], r["lane"], r.get("kind", kind), r.get("ttl_s", 900.0),
                   r.get("waited_s", 0.0), resolved_group,
                   sock_path=getattr(c, "sock_path", None) or sock_path,
                   token=r.get("token"), sock_source=getattr(c, "sock_path", None))
        _run_state_heartbeat("lease-active", lv.lease_id,
                             str(request_meta.get("command_kind") or "complete current lease"))
        yield lv
    except BrokerUnavailable as exc:
        yield Lease(denied_reason=str(exc))
    finally:
        if lv is not None:
            lv.release()
            _run_state_heartbeat("lease-released", lv.lease_id,
                                 "publish the next stage artifact")
        if mine and c is not None:
            c.close()


# ---------------------------------------------------------------------------
# Retry with backoff. Used by callers whose work is idempotent (a measurement is).
# ---------------------------------------------------------------------------
def with_retry(fn, attempts=3, base_s=0.5, cap_s=20.0, jitter=0.3, retry_on=(Exception,),
               on_retry=None):
    """Exponential backoff with FULL jitter.

    The jitter is not decoration. Sixteen kernels that all fail at the same instant (a broker
    restart, a transient driver hiccup) and all back off on the same deterministic schedule will
    retry in lockstep forever -- a thundering herd that looks exactly like a hung fleet. Randomising
    the wait is what breaks the synchronisation.
    """
    last = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except retry_on as exc:                                  # noqa: PERF203
            last = exc
            if i == attempts - 1:
                break
            delay = min(cap_s, base_s * (2 ** i))
            delay = delay * (1.0 - jitter) + random.random() * delay * jitter
            if on_retry:
                on_retry(i + 1, exc, delay)
            time.sleep(delay)
    raise last


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_run(a, rest):
    """Hold a lease for the lifetime of a command. This is what gpu_lock.sh delegates to."""
    if not rest:
        print("nothing to run", file=sys.stderr)
        return 2
    t0 = time.time()
    with lease(kind=a.kind, group=a.group, priority=a.priority,
               lanes=[int(x) for x in a.lanes.split(",") if x.strip().isdigit()] if a.lanes else None,
               vram_mb=a.vram_mb, timeout_s=a.timeout_s, label=a.label,
               require_group=a.require_group, command_kind=a.command_kind,
               estimated_runtime_s=a.estimated_runtime_s,
               safe_yield_kind=a.safe_yield_kind) as g:
        env = dict(os.environ)
        if g.lane is not None:
            env["HIP_VISIBLE_DEVICES"] = str(g.lane)
            env["GEAK_GPU_LANE"] = str(g.lane)
            env["GEAK_GPU_LEASE"] = str(g.lease_id)
            env["GEAK_GPU_GROUP"] = str(g.group)
        elif a.require_group and "group is required" in str(g.denied_reason):
            print(f"ERROR: {g.denied_reason}", file=sys.stderr)
            return 4
        elif a.require_broker:
            print("ERROR: --require-broker set but no broker is reachable", file=sys.stderr)
            return 3
        if a.verbose:
            where = f"lane {g.lane}" if g.lane is not None else "UNBROKERED (no daemon)"
            print(f"[gpu_client] {where} after {g.waited_s}s wait", file=sys.stderr)
        rc = subprocess.call(rest, env=env)
        # A LEASE THAT DIED UNDER A RUNNING COMMAND IS A MEASUREMENT PROBLEM, NOT A BOOKKEEPING ONE.
        # If the heartbeat could not renew, the broker reclaimed this lane and may already have
        # granted it to somebody else -- so the command we just ran finished on a card it did not
        # own, possibly alongside another tenant. The number it produced is not a clean measurement,
        # and nothing downstream can tell.
        #
        # Reported on stderr rather than by failing: the command may have been a compile, or the
        # caller may not be timing anything, and turning a scheduling hiccup into a failed run is
        # the wrong default. It is loud, it names the lane, and it says what to distrust.
        if g.brokered and getattr(g, "lost", False):
            print(f"[gpu_client] WARNING: the lease on lane {g.lane} was RECLAIMED while the "
                  f"command was still running (the heartbeat could not renew in time). That lane "
                  f"may have been granted to another tenant, so any timing from this run is "
                  f"suspect. Re-measure before trusting it.", file=sys.stderr)
    if a.verbose:
        print(f"[gpu_client] rc={rc} total={time.time() - t0:.1f}s", file=sys.stderr)
    return rc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # `--selftest` is the same thing as the `selftest` subcommand. Both spellings exist because this
    # file uses subcommands while the repo's selftest registry (`validate.py`) invokes every tool it
    # guards as `<tool> --selftest` -- and an unregisterable selftest is the unrun selftest that
    # registry exists to prevent. `gpu_broker.py` and `residency.py` already answer the flag form.
    if argv[:1] == ["--selftest"]:
        return selftest()
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(description="GPU broker client")
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run a command holding a lease")
    r.add_argument("--kind", default="exclusive", choices=["exclusive", "resident"])
    r.add_argument("--group", default=None, help="fair-share key; normally the kernel name")
    r.add_argument("--priority", default="measure")
    r.add_argument("--lanes", default="", help="restrict to these lanes (the allocation fence)")
    r.add_argument("--vram-mb", type=int, default=0)
    r.add_argument("--timeout-s", type=float, default=1800.0)
    r.add_argument("--label", default=None)
    r.add_argument("--require-broker", action="store_true",
                   help="fail instead of running unbrokered")
    r.add_argument("--require-group", action="store_true",
                   help="fail when no stable per-kernel group can be resolved")
    r.add_argument("--command-kind", default=None,
                   help="runtime metadata, e.g. profile, compile, resident_wake")
    r.add_argument("--estimated-runtime-s", type=float, default=None,
                   help="runtime metadata: short <120, medium <=300, long >300")
    r.add_argument("--safe-yield-kind", default=None,
                   help="advisory cooperative yield boundary; never forcibly enforced")
    r.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("ping")
    st = sub.add_parser("status", help="show broker socket, lanes, leases, TTLs, groups and metadata")
    st.add_argument("--sock", default=DEFAULT_SOCK)
    st.add_argument("--json", action="store_true")
    sub.add_parser("selftest")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a, rest)
    if a.cmd == "ping":
        try:
            c = connect(label="cli")
            print(json.dumps(c.call({"op": "ping"})))
            c.close()
            return 0
        except BrokerUnavailable as exc:
            print(f"unavailable: {exc}", file=sys.stderr)
            return 1
    if a.cmd == "status":
        try:
            c = connect(a.sock, label="client-status", autostart=False)
            result = c.call({"op": "status"})
            c.close()
        except BrokerUnavailable as exc:
            print(f"unavailable: {exc}", file=sys.stderr)
            return 1
        if a.json:
            print(json.dumps(result, indent=2))
        else:
            status = result.get("status") or {}
            socket_info = status.get("socket") or {}
            print(f"socket={socket_info.get('path')} listening={socket_info.get('listening')}")
            for lane in status.get("lanes", []):
                lease = lane.get("exclusive_lease") or {}
                print(f"lane={lane.get('lane')} exclusive={lease.get('lease_id')} "
                      f"group={lease.get('group')} ttl={lease.get('ttl_left_s')} "
                      f"residents={len(lane.get('resident_leases') or [])}")
        return 0 if result.get("ok") else 1
    if a.cmd == "selftest":
        return selftest()
    ap.print_help()
    return 2


def selftest():
    """Exercises the client against a REAL broker on a temp socket. No GPU needed."""
    import tempfile
    fails = []

    def check(c, m):
        print(("  ok: " if c else "  FAIL: ") + m)
        if not c:
            fails.append(m)

    print("\n# gpu_client selftest (real broker, temp socket, no GPU)")
    # The broker is opt-in (GEAK_GPU_BROKER=1); the selftest opts in for its own temp broker and
    # first proves the default really is off.
    _saved_enabled = os.environ.pop("GEAK_GPU_BROKER", None)
    check(not enabled(), "the client is OFF unless GEAK_GPU_BROKER=1")
    os.environ["GEAK_GPU_BROKER"] = "1"
    try:
        return _selftest_body(check, fails)
    finally:
        if _saved_enabled is None:
            os.environ.pop("GEAK_GPU_BROKER", None)
        else:
            os.environ["GEAK_GPU_BROKER"] = _saved_enabled


def _selftest_body(check, fails):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sock = os.path.join(td, "b.sock")
        # `--no-probe`: SYNTHETIC lanes, which is what this fixture always meant. Without it the
        # broker probes real sysfs for lanes 0 and 1, marks them foreign-busy on a loaded box, and
        # re-applies that fence on every poll -- so seven checks of the CLIENT failed for the sole
        # reason that the machine was busy, and passed again when it was not. The `drain` call below
        # was an attempt at the same thing that could not hold, because the poll undoes it.
        proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "gpu_broker.py"), "--serve", "--sock", sock,
             "--gpus", "0,1", "--state-dir", td, "--no-probe"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                if os.path.exists(sock):
                    break
                time.sleep(0.05)
            check(os.path.exists(sock), "the broker came up on a temp socket")

            c = connect(sock, label="selftest", autostart=False)
            check(c.call({"op": "ping"}).get("ok"), "ping over a real socket")

            # These fake lanes may look foreign-busy if the real box is loaded; clear the fence so
            # the test measures the CLIENT, not this machine's current occupancy.
            c.call({"op": "drain", "lane": 0, "disabled": False})

            with lease(group="k1", priority="measure", sock_path=sock, timeout_s=10) as g:
                check(g.brokered and g.lane in (0, 1), f"lease acquired (lane={g.lane})")
                held = g.lane
            with lease(group="k1", priority="measure", sock_path=sock, timeout_s=10) as g2:
                check(g2.brokered, "a second lease after release succeeds")
                check(g2.lane in (0, 1), f"and lands on a real lane (lane={g2.lane})")
            check(held is not None, "the first lease reported its lane")

            with lease(kind="resident", vram_mb=512, group="k2", sock_path=sock,
                       timeout_s=10) as park:
                check(park.brokered and park.kind == "resident", "a resident lease is granted")
                parked_lane = park.lane
                with park.wake(timeout_s=10) as w:
                    check(w.lane == parked_lane, "wake() lands on the SAME lane it parked on")
                    check(w.kind == "exclusive", "and the window is exclusive")

            # The property the whole fallback story rests on.
            with lease(group="k3", sock_path=os.path.join(td, "nope.sock"),
                       timeout_s=2) as g3:
                check(not g3.brokered and g3.lane is None,
                      "no broker -> an UNBROKERED lease, not an exception")
                check(bool(g3.denied_reason), "and it says WHY it is unbrokered")

            # A LANE MISMATCH IS ITS OWN STATE. This broker owns lane 0; asking it for lane 7 means we
            # are talking to the wrong pool, and the old code yielded a bare unbrokered lease -- so an
            # unarbitrated card looked exactly like a box with no scheduler. Measured on one fleet.
            #
            # stderr is CAPTURED rather than allowed through: the banner is deliberately loud in
            # production (the failure it replaces was silent), and a selftest that writes to stderr is
            # read as a failing selftest by the tree validator. Capturing also lets the banner itself
            # be asserted, which is stronger than suppressing it.
            _err = io.StringIO()
            with contextlib.redirect_stderr(_err), \
                    lease(group="k4", sock_path=sock, lanes=[7], timeout_s=5) as g4:
                check(not g4.brokered, "a lane this broker does not own is not granted")
                check(g4.lane_mismatch, "and it is reported as a LANE MISMATCH, not as 'no broker'")
                check("no such lane" in (g4.denied_reason or ""), "with the broker's own reason")
            check("LANE MISMATCH" in _err.getvalue() and "UNARBITRATED" in _err.getvalue(),
                  "and it says so on stderr, where an operator will see it")
            # ...while a genuinely absent daemon is NOT a mismatch, because the responses differ
            with lease(group="k5", sock_path=os.path.join(td, "nope2.sock"), timeout_s=2) as g5:
                check(not g5.lane_mismatch, "an absent daemon is not a lane mismatch")

            # WHOSE POOL. The canonical socket is only adopted when something named it, because it
            # belongs to whichever campaign started first. Measured: a campaign with no marker
            # and no broker of its own was one connect() from another fleet's long-held pool.
            _saved = os.environ.pop("GEAK_GPU_BROKER_SOCK", None)
            _cwd = os.getcwd()
            try:
                os.chdir(td)                       # no marker at or above a fresh temp dir
                _, src = discover_sock()
                check(src == "unnamed", "an unmarked tree names no broker")
                raised = False
                _err2 = io.StringIO()
                with contextlib.redirect_stderr(_err2):
                    try:
                        connect(CANONICAL_SOCK, autostart=False)
                    except BrokerUnavailable as exc:
                        raised = "not adopted" in str(exc)
                check(raised, "an UNNAMED canonical socket is refused rather than adopted")
                check(MARKER_NAME in _err2.getvalue(),
                      "and the refusal names the one line that fixes it")
                # a marker names it, and then it is this run's pool to talk to
                with open(os.path.join(td, MARKER_NAME), "w") as fh:
                    fh.write(sock + "\n")
                named, src2 = discover_sock()
                check(src2 == "marker" and named == sock, "a marker names the pool")
                with lease(group="k6", sock_path=CANONICAL_SOCK, timeout_s=5) as g6:
                    check(g6.brokered and g6.sock_path == sock,
                          "and connect() follows the marker instead of the canonical path")
                os.unlink(os.path.join(td, MARKER_NAME))
                # The escape hatch for a single-tenant box, and the flag is read LIVE so exporting it
                # after import actually works -- binding it at import is a trap this file has already
                # paid for once (see residency.py's GEAK_GPU_BROKER note).
                os.environ["GEAK_GPU_REQUIRE_MARKER"] = "0"
                try:
                    adopted = False
                    try:
                        connect(CANONICAL_SOCK, autostart=False)
                        adopted = True
                    except BrokerUnavailable as exc:
                        adopted = "not adopted" not in str(exc)
                    check(adopted, "GEAK_GPU_REQUIRE_MARKER=0 stops refusing (adopt-anything)")
                finally:
                    os.environ.pop("GEAK_GPU_REQUIRE_MARKER", None)
            finally:
                os.chdir(_cwd)
                if _saved is not None:
                    os.environ["GEAK_GPU_BROKER_SOCK"] = _saved

            n = {"calls": 0}

            def flaky():
                n["calls"] += 1
                if n["calls"] < 3:
                    raise RuntimeError("transient")
                return "ok"

            check(with_retry(flaky, attempts=5, base_s=0.01) == "ok" and n["calls"] == 3,
                  "with_retry retries then succeeds")
            try:
                with_retry(lambda: (_ for _ in ()).throw(RuntimeError("always")),
                           attempts=2, base_s=0.01)
                check(False, "with_retry re-raises after the last attempt")
            except RuntimeError:
                check(True, "with_retry re-raises after the last attempt")
            c.close()
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    print("\nSELFTEST PASS: client works and degrades" if not fails else f"\nFAIL: {len(fails)} check(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
