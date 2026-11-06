#!/usr/bin/env python3
"""Integration tests: a REAL broker over a REAL socket, with real processes dying. No GPU needed.

`test_scheduler_core.py` proves the policy with a fake clock. This proves the parts a pure state
machine cannot: that a killed client's lane comes back, that concurrent clients never double-book a
lane, that the daemon survives garbage on the wire, and that everything degrades when it is absent.

The lanes here are ids the box may or may not have. That is deliberate and harmless -- nothing
launches a kernel, so a "lane" is just a token being allocated. What is under test is the
arbitration, and arbitration does not care whether the GPU is real.

    python3 scheduler/tests/test_broker_integration.py
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

SCHED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCHED)

# The broker is OPT-IN in GEAK (GEAK_GPU_BROKER=1; gpu_client reads it live). These tests exercise
# the broker, so they opt in for the whole process; the fallback tests below switch it off locally.
os.environ["GEAK_GPU_BROKER"] = "1"
# GEAK's gpu_lock.sh (reached through the pack's scripts/gpu_lock.sh shim) flocks
# /tmp/team_gpu_locks/gpu_<id>.lock. A test must never queue behind -- or block -- a real tenant of
# lanes 0/1, so every wrapper call below uses a private lock namespace (a test-only override).
_LOCK_NS = tempfile.mkdtemp(prefix="sched_test_locks_")

# Imported rather than spelled as literals so a re-banding of the scheduler moves the tests with it.
from core import PRIO_MEASURE, PRIO_VERIFY  # noqa: E402

# WHERE A SIBLING SCRIPT IS, given that this scheduler ships into three different layouts:
#   <pack>/scheduler/          + <pack>/scripts/<name>           a composed per-DSL skill pack
#   <pack>/scripts/scheduler/  + <pack>/scripts/<dsl>/<name>     the composed unified router pack
#   .../vendor/amd/scheduler/  + .../lang/<dsl>/scripts/<name>   skills-src, different layers
# plus `tools/`, where the tree this was transplanted from keeps them. A hardcoded sibling path
# resolved in exactly one of those and failed in the rest -- as ModuleNotFoundError for sweep_pool
# and as a bare "does not exist" for gpu_lock.sh, neither of which reads like a layout problem.
_DSLS = ("triton", "gluon", "tilelang", "flydsl", "hip", "cuda", "cutedsl")


def _sibling(name):
    up = os.path.dirname(SCHED)
    cands = [os.path.join(up, "scripts", name), os.path.join(up, name),
             os.path.join(up, "tools", name)]
    cands += [os.path.join(up, d, name) for d in _DSLS]
    cands += [os.path.join(os.path.dirname(os.path.dirname(up)), "lang", d, "scripts", name)
              for d in _DSLS]
    return next((p for p in cands if os.path.isfile(p)), None)

FAILS = []


def ok(cond, msg):
    """Print, record -- and RAISE, so pytest sees the failure too.

    This file reads as a script and `main()` does return 1 on failure, but check.sh runs the
    directory under pytest, which decides pass/fail from exceptions and nothing else. With a bare
    list every check here was invisible to CI: verified by deleting a field the assertions below
    require and watching pytest report "1 passed" while the check printed ok. A test that cannot
    fail in CI is worse than no test, because it is a green light nobody is holding."""
    print(("  ok: " if cond else "  FAIL: ") + msg)
    if not cond:
        FAILS.append(msg)
        raise AssertionError(msg)


def section(t):
    print(f"\n# {t}")


class BrokerFixture:
    """A real daemon on a private socket, torn down on exit."""

    def __init__(self, gpus="0,1", extra=()):
        self.td = tempfile.mkdtemp(prefix="geak_bt_")
        self.sock = os.path.join(self.td, "b.sock")
        # --no-probe: SYNTHETIC lanes. These tests allocate lane ids on a box whose real GPU 0 may
        # be holding 200+ GB of somebody else's model, and the foreign-tenant check would then
        # refuse every grant -- failing tests about queueing and reclaim for a reason that has
        # nothing to do with either. Draining is not a substitute: the maintenance probe re-marks
        # the lane two seconds later.
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(SCHED, "gpu_broker.py"), "--serve",
             "--sock", self.sock, "--gpus", gpus, "--state-dir", self.td, "--no-probe", *extra],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(200):
            if os.path.exists(self.sock):
                break
            time.sleep(0.05)

    def rpc(self, req, timeout=20):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(self.sock)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            c = s.recv(65536)
            if not c:
                break
            buf += c
        s.close()
        return json.loads(buf.decode() or "{}")

    def status(self):
        return self.rpc({"op": "status"})["status"]

    def journal(self):
        p = os.path.join(self.td, "journal.ndjson")
        if not os.path.exists(p):
            return []
        with open(p) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def stop(self, keep_dir=False):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        # Clean the state dir up. Each fixture makes one, and this file makes ~15 fixtures per run
        # -- left behind, a few CI runs bury /tmp under hundreds of `geak_bt_*` dirs holding
        # journals nobody will ever read. Tests that want to inspect a journal do so before exit.
        if not keep_dir:
            import shutil
            shutil.rmtree(self.td, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()
        return False


# ---------------------------------------------------------------------------
def test_daemon_comes_up():
    section("the daemon starts, answers, and reports its lanes")
    with BrokerFixture("0,1,2") as b:
        r = b.rpc({"op": "ping"})
        ok(r.get("ok"), "ping answered")
        ok(r.get("lanes") == [0, 1, 2], f"lanes reported: {r.get('lanes')}")
        ok(len(b.status()["lanes"]) == 3, "status lists all three")


def test_killed_client_releases_its_lane():
    section("a client killed with SIGKILL loses its lane -- the crash-recovery path")
    with BrokerFixture("0") as b:
        holder = subprocess.Popen(
            [sys.executable, "-c", f'''
import json, socket, sys, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect({b.sock!r})
s.sendall((json.dumps({{"op":"acquire","kind":"exclusive","label":"victim",
                        "group":"g","timeout_s":30}})+"\\n").encode())
buf=b""
while not buf.endswith(b"\\n"):
    buf += s.recv(65536)
print(buf.decode().strip(), flush=True)
time.sleep(300)
'''], stdout=subprocess.PIPE, text=True)
        line = holder.stdout.readline()
        got = json.loads(line)
        ok(got.get("ok"), f"the victim holds a lease: {got}")
        ok(b.status()["lanes"][0]["exclusive"] is not None, "the broker shows the lane held")

        holder.kill()
        holder.wait(timeout=10)
        # The socket closes with the process, so reclaim should be immediate -- no TTL wait.
        freed = False
        for _ in range(50):
            if b.status()["lanes"][0]["exclusive"] is None:
                freed = True
                break
            time.sleep(0.1)
        ok(freed, "SIGKILL on the holder frees the lane within a second (socket close, not TTL)")
        recs = [r for r in b.journal() if r.get("event") == "disconnect_reclaim"]
        ok(bool(recs), "and the reclaim is recorded in the journal")
        # A reclaim ENDS a lease exactly as a release does, so the journal has to say for how long.
        # With only a count, every crash-reclaimed lease drops out of any held-time sum, and the
        # resulting utilisation reads like a measurement while actually being a lower bound. This
        # is the field that makes the two paths add up.
        if recs:
            det = recs[-1].get("reclaimed") or []
            ok(len(det) == recs[-1].get("leases"),
               "the reclaim names every lease it dropped, not just how many")
            ok(all(isinstance(d.get("held_s"), (int, float)) and d.get("lane") is not None
                   for d in det),
               "and each carries its held_s and lane, so utilisation can include it")


def test_no_double_booking_under_concurrency():
    section("32 concurrent clients over 2 lanes never double-book")
    with BrokerFixture("0,1") as b:
        import gpu_client
        overlaps = []
        active = {}
        lk = threading.Lock()

        served = []

        def worker(i):
            with gpu_client.lease(group=f"g{i % 4}", sock_path=b.sock, timeout_s=60) as g:
                if not g.brokered:
                    return
                with lk:
                    served.append(i)
                    if g.lane in active:
                        overlaps.append((g.lane, active[g.lane], i))
                    active[g.lane] = i
                time.sleep(0.05)
                with lk:
                    if active.get(g.lane) == i:
                        del active[g.lane]

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(32)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=120)
        ok(not overlaps, f"no two clients held the same lane at once (overlaps={overlaps})")
        # Counted per CLIENT, not from stats["granted"]. The stat is a running total sampled after
        # the fact and races with the release path; what the test actually means to assert is that
        # no client was dropped, which is a property of the clients.
        ok(len(served) == 32, f"every client got a lane ({len(served)}/32)")
        ok(not b.status()["queue"], "the queue drained completely")


def test_garbage_on_the_wire():
    section("a malformed client cannot take the daemon down")
    with BrokerFixture("0") as b:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect(b.sock)
        s.sendall(b"this is not json\n")
        resp = json.loads(s.recv(65536).decode().strip())
        ok(resp.get("ok") is False and "bad json" in resp.get("error", ""),
           "garbage gets an error, not a crash")
        s.sendall((json.dumps({"op": "nope"}) + "\n").encode())
        resp2 = json.loads(s.recv(65536).decode().strip())
        ok(resp2.get("ok") is False, "an unknown op is refused")
        s.close()
        ok(b.rpc({"op": "ping"}).get("ok"), "the daemon is still serving afterwards")

        # A connection that dies mid-message must not wedge the handler thread either.
        s2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s2.connect(b.sock)
        s2.sendall(b'{"op": "acq')     # truncated, then vanish
        s2.close()
        time.sleep(0.3)
        ok(b.rpc({"op": "ping"}).get("ok"), "a truncated message + abrupt close is survived")


def test_priority_is_honoured_end_to_end():
    section("priority ordering survives the socket, not just the state machine")
    with BrokerFixture("0") as b:
        import gpu_client
        c0 = gpu_client.Conn(b.sock, label="holder")
        held = c0.call({"op": "acquire", "kind": "exclusive", "group": "h", "timeout_s": 30})
        ok(held.get("ok"), "the lane is occupied")

        order = []
        lk = threading.Lock()

        def ask(name, prio, delay):
            time.sleep(delay)
            with gpu_client.lease(group=name, priority=prio, sock_path=b.sock,
                                  timeout_s=60) as g:
                if g.brokered:
                    with lk:
                        order.append(name)
                    time.sleep(0.05)

        ts = [threading.Thread(target=ask, args=("bulk", "bulk", 0.0)),
              threading.Thread(target=ask, args=("verify", "verify", 0.2))]
        for t in ts:
            t.start()
        time.sleep(0.6)
        c0.call({"op": "release", "lease": held["lease"]})
        for t in ts:
            t.join(timeout=60)
        c0.close()
        ok(order[:1] == ["verify"],
           f"verify (higher priority) ran before bulk despite queueing later: {order}")


def test_queued_disconnect_cancels_immediately():
    section("a queued client disconnect cancels immediately instead of leaving a zombie request")
    with BrokerFixture("0") as b:
        import gpu_client
        holder = gpu_client.Conn(b.sock, label="holder")
        held = holder.call({"op": "acquire", "kind": "exclusive", "group": "holder",
                            "timeout_s": 20})
        ok(held.get("ok"), "the lane is occupied")

        queued = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        queued.settimeout(10)
        queued.connect(b.sock)
        queued.sendall((json.dumps({
            "op": "acquire", "kind": "exclusive", "label": "queued-victim",
            "group": "queued", "timeout_s": 60,
            "meta": {"command_kind": "profile", "estimated_runtime_s": 600},
        }) + "\n").encode())
        enqueued = False
        for _ in range(30):
            enqueued = bool(b.status()["queue"])
            if enqueued:
                break
            time.sleep(0.05)
        ok(enqueued, "the request entered the queue before its client disconnects")
        queued.close()

        gone = False
        for _ in range(30):
            gone = not b.status()["queue"]
            if gone:
                break
            time.sleep(0.05)
        ok(gone, "the queued request is removed within the disconnect polling interval")
        cancels = [r for r in b.journal() if r.get("event") == "cancel"]
        ok(any(r.get("reason") == "client_disconnected" and r.get("state") == "queued"
               for r in cancels), f"journal records the cancellation reason: {cancels}")
        holder.call({"op": "release", "lease": held["lease"]})
        holder.close()


def test_cooperative_pause_never_forces_a_running_process():
    section("a waiting short request asks for a boundary yield but never revokes a running lease")
    with BrokerFixture("0") as b:
        import gpu_client
        holder = gpu_client.Conn(b.sock, label="cooperative-holder")
        held = holder.call({
            "op": "acquire", "kind": "exclusive", "group": "long",
            "priority": "measure", "timeout_s": 20,
            "meta": {"command_kind": "profile", "estimated_runtime_s": 600,
                     "safe_yield_kind": "profile_pass_boundary"},
        })
        ok(held.get("ok"), "a safely-yieldable long lease is held")

        short = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        short.settimeout(10)
        short.connect(b.sock)
        short.sendall((json.dumps({
            "op": "acquire", "kind": "exclusive", "label": "short-waiter",
            "group": "short", "priority": "measure", "timeout_s": 30,
            "meta": {"command_kind": "verify", "estimated_runtime_s": 10},
        }) + "\n").encode())
        pause = None
        for _ in range(30):
            leases = b.status()["leases"]
            pause = next((x.get("pause_requested") for x in leases
                          if x.get("lease_id") == held["lease"]), None)
            if pause:
                break
            time.sleep(0.05)
        ok(pause and pause.get("reason") == "short_waiter",
           f"the broker requests a cooperative boundary: {pause}")
        ok(any(x.get("lease_id") == held["lease"] for x in b.status()["leases"]),
           "the running lease remains held; the broker did not suspend or revoke it")

        continued = holder.call({"op": "boundary", "lease": held["lease"], "action": "continue"})
        ok(continued.get("ok") and any(x.get("lease_id") == held["lease"]
                                       for x in b.status()["leases"]),
           "the holder may explicitly continue at its boundary")
        yielded = holder.call({"op": "boundary", "lease": held["lease"], "action": "yield"})
        ok(yielded.get("ok"), "the holder can explicitly yield at its boundary")
        response = json.loads(short.recv(65536).decode().strip())
        ok(response.get("ok") and response.get("lane") == 0,
           f"the queued short request receives the yielded lane: {response}")
        short.close()
        holder.close()
        events = {r.get("event") for r in b.journal()}
        ok({"pause_requested", "pause_continue", "pause_yield"} <= events,
           f"the cooperative protocol is journalled: {sorted(events)}")


def test_status_json_exposes_pool_state_and_metadata():
    section("broker and client status JSON expose socket, lanes, leases, TTLs, groups and metadata")
    with BrokerFixture("0") as b:
        import gpu_client
        holder = gpu_client.Conn(b.sock, label="status-holder")
        held = holder.call({
            "op": "acquire", "kind": "exclusive", "group": "status-kernel",
            "timeout_s": 20,
            "meta": {"command_kind": "compile", "estimated_runtime_s": 240,
                     "safe_yield_kind": "compile_unit_boundary"},
        })
        ok(held.get("ok"), "a lease exists to inspect")
        broker_status = subprocess.run(
            [sys.executable, os.path.join(SCHED, "gpu_broker.py"), "--status", "--json",
             "--sock", b.sock],
            capture_output=True, text=True, timeout=30)
        client_status = subprocess.run(
            [sys.executable, os.path.join(SCHED, "gpu_client.py"), "status", "--json",
             "--sock", b.sock],
            capture_output=True, text=True, timeout=30)
        ok(broker_status.returncode == 0 and client_status.returncode == 0,
           f"both status CLIs return JSON (broker={broker_status.stderr}, client={client_status.stderr})")
        broker_json = json.loads(broker_status.stdout)
        client_json = json.loads(client_status.stdout)
        for status in (broker_json["status"], client_json["status"]):
            lease = status["lanes"][0]["exclusive_lease"]
            ok(status["socket"]["path"] == b.sock and status["socket"]["listening"],
               f"socket state is explicit: {status['socket']}")
            ok(lease["group"] == "status-kernel" and lease["ttl_left_s"] is not None,
               f"lease group + TTL are explicit: {lease}")
            ok(lease["metadata"]["command_kind"] == "compile",
               f"runtime metadata is explicit: {lease['metadata']}")
        holder.call({"op": "release", "lease": held["lease"]})
        holder.close()


def test_resident_park_and_wake_are_journalled_as_separate_states():
    section("resident sweeps park in VRAM and take short exclusive wake windows")
    with BrokerFixture("0") as b:
        import gpu_client
        with gpu_client.lease(kind="resident", group="resident-kernel", vram_mb=64,
                              sock_path=b.sock, timeout_s=20,
                              command_kind="resident_sweep", estimated_runtime_s=900,
                              safe_yield_kind="resident_wake_boundary",
                              meta={"resident_event": "park", "reason": "resident_sweep_park"}) as park:
            ok(park.brokered and park.kind == "resident", "the warm worker is registered as resident")
            parked = b.status()["lanes"][0]["resident_leases"][0]
            ok(parked["metadata"]["command_kind"] == "resident_sweep",
               f"status identifies the parked sweep: {parked}")
            with park.wake(timeout_s=20) as wake:
                ok(wake.brokered and wake.kind == "exclusive" and wake.lane == park.lane,
                   "one measurement gets a short exclusive wake on the parked lane")
        events = b.journal()
        names = {r.get("event") for r in events}
        ok({"resident_park", "resident_wake"} <= names,
           f"park and wake transitions are journalled: {sorted(names)}")
        wake_events = [r for r in events if r.get("event") == "resident_wake"]
        ok(wake_events and wake_events[-1].get("exclusive") is True
           and wake_events[-1].get("reason") == "resident_measurement",
           f"the wake journal carries exclusivity and reason: {wake_events[-1:]}")


def test_drain_takes_a_lane_out_of_service():
    section("drain: a lane stops being granted but running work is untouched")
    with BrokerFixture("0,1") as b:
        import gpu_client
        with gpu_client.lease(group="g", sock_path=b.sock, timeout_s=30) as g1:
            ok(g1.brokered, "a lease is held")
            b.rpc({"op": "drain", "lane": 0, "disabled": True})
            b.rpc({"op": "drain", "lane": 1, "disabled": True})
            ok(all(l["disabled"] for l in b.status()["lanes"]), "both lanes drained")
            # The running lease survives -- draining is not killing.
            ok(any(l["exclusive"] for l in b.status()["lanes"]),
               "the in-flight lease is NOT revoked by a drain")
            with gpu_client.lease(group="g2", sock_path=b.sock, timeout_s=2) as g2:
                ok(not g2.brokered, "a new request on a fully drained box is not granted")
        b.rpc({"op": "drain", "lane": 0, "disabled": False})
        with gpu_client.lease(group="g3", sock_path=b.sock, timeout_s=10) as g3:
            ok(g3.brokered and g3.lane == 0, "undraining restores service")


def test_journal_records_the_scheduler_cost():
    section("the journal carries what the fleet report reads back")
    with BrokerFixture("0") as b:
        import gpu_client
        for i in range(3):
            with gpu_client.lease(group=f"k{i}", sock_path=b.sock, timeout_s=30):
                time.sleep(0.02)
        recs = b.journal()
        ev = {r["event"] for r in recs}
        ok("grant" in ev and "release" in ev, f"grants and releases are journalled: {sorted(ev)}")
        rel = [r for r in recs if r["event"] == "release"]
        ok(all("held_s" in r for r in rel), "each release records how long the lane was held")
        gr = [r for r in recs if r["event"] == "grant"]
        ok({r["group"] for r in gr} >= {"k0", "k1", "k2"},
           "and each grant is attributed to its fair-share group")


def test_fair_share_key_reaches_the_broker():
    section("the fair-share key is EXACT, not guessed from the path shape")
    lock = _sibling("gpu_lock.sh")
    if lock is None:
        ok(False, "gpu_lock.sh exists next to this scheduler")
        return
    with open(lock) as fh:
        if "broker" not in fh.read():
            # See test_gpu_lock_delegation: a pack may override this wrapper without broker support,
            # and the marker route is a property of the wrapper rather than of the scheduler.
            print("    SKIP: this pack's gpu_lock.sh has no broker delegation, so the marker route "
                  "it would carry does not exist here")
            return
    with BrokerFixture("0,1") as b:
        base = tempfile.mkdtemp()
        # The DOCUMENTED default layout: <exp_root>/team_<kernel>_<ts>/<kernel>/round_N/engineer_i/ws
        # This shape is the whole point of the test. A path-shape heuristic matches the `team_`
        # wrapper and returns `team_fused_moe_20260814`, which is per-RUN -- so two runs of one
        # kernel look like two tenants and fair share quietly stops meaning anything. The marker
        # file kernel_lane.js writes at the eval-dir root is what makes it exact.
        evald = os.path.join(base, "exp", "team_fused_moe_20260814", "fused_moe")
        ws = os.path.join(evald, "round_1", "engineer_0", "workspace")
        os.makedirs(ws, exist_ok=True)
        with open(os.path.join(evald, ".geak_gpu_group"), "w") as fh:
            fh.write("fused_moe\n")

        env = dict(os.environ, GEAK_GPU_BROKER_SOCK=b.sock, GEAK_GPU_BROKER="1",
                   GEAK_GPU_BROKER_AUTOSTART="0", GEAK_GPU_LOCK_DIR=_LOCK_NS)
        env.pop("GEAK_GPU_GROUP", None)
        r = subprocess.run(["bash", lock, "0,1", "true"],
                           capture_output=True, text=True, env=env, cwd=ws, timeout=120)
        ok(r.returncode == 0, f"the wrapper ran (rc={r.returncode}) {r.stderr[-200:]}")
        groups = {rr.get("group") for rr in b.journal() if rr.get("event") == "grant"}
        ok("fused_moe" in groups,
           f"the acquisition is attributed to the KERNEL, not the run dir: {groups}")
        ok(not any(str(g).startswith("team_") for g in groups),
           f"and never to the per-run `team_<kernel>_<ts>` wrapper: {groups}")

        # No marker (a hand-run tree): the heuristic must still not return the team_ wrapper.
        ws2 = os.path.join(base, "exp", "team_gemm_a8w8_20260101", "gemm_a8w8", "ws")
        os.makedirs(ws2, exist_ok=True)
        subprocess.run(["bash", lock, "0,1", "true"],
                       capture_output=True, text=True, env=env, cwd=ws2, timeout=120)
        groups2 = {rr.get("group") for rr in b.journal() if rr.get("event") == "grant"}
        ok("gemm_a8w8" in groups2,
           f"without a marker the fallback still finds the kernel: {groups2}")

        # The shared Python/shell fallback must discard direction leaves, not turn every `dir_N`
        # worktree into a different fair-share tenant.
        ws3 = os.path.join(base, "manual_kernel", "dir_17")
        os.makedirs(ws3, exist_ok=True)
        r3 = subprocess.run(["bash", lock, "0,1", "true"],
                            capture_output=True, text=True, env=env, cwd=ws3, timeout=120)
        ok(r3.returncode == 0, f"a direction-leaf fallback runs (rc={r3.returncode})")
        groups3 = {rr.get("group") for rr in b.journal() if rr.get("event") == "grant"}
        ok("manual_kernel" in groups3 and "dir_17" not in groups3,
           f"the fallback ignores dir_<n>: {groups3}")

        client = os.path.join(SCHED, "gpu_client.py")
        r4 = subprocess.run([sys.executable, client, "run", "--require-group",
                             "--timeout-s", "5", "--", "true"],
                            capture_output=True, text=True, env=env, cwd=ws3, timeout=30)
        ok(r4.returncode == 0,
           f"the direct client shares the same direction-leaf fallback (rc={r4.returncode})")
        groups4 = {rr.get("group") for rr in b.journal() if rr.get("event") == "grant"}
        ok("manual_kernel" in groups4,
           f"the direct client attributes the same kernel group: {groups4}")

        empty_env = dict(env)
        empty_env.pop("GEAK_GPU_GROUP", None)
        r5 = subprocess.run([sys.executable, client, "run", "--require-group",
                             "--timeout-s", "5", "--", "echo", "must-not-run"],
                            capture_output=True, text=True, env=empty_env, cwd="/", timeout=30)
        ok(r5.returncode == 4 and "must-not-run" not in r5.stdout,
           "--require-group fails closed when no stable group can be resolved")

        wrapped_env = dict(empty_env, GEAK_GPU_REQUIRE_GROUP="1")
        r6 = subprocess.run(["bash", lock, "0", "echo", "must-not-run-wrapped"],
                            capture_output=True, text=True, env=wrapped_env, cwd="/", timeout=30)
        ok(r6.returncode == 4 and "must-not-run-wrapped" not in r6.stdout,
           "a managed gpu_lock brokered run propagates require-group fail-closed")


def test_priority_marker_reaches_the_broker():
    section("the priority band rides the same marker-file route as the fair-share key")
    lock = _sibling("gpu_lock.sh")
    if lock is None:
        ok(False, "gpu_lock.sh exists next to this scheduler")
        return
    with open(lock) as fh:
        if "broker" not in fh.read():
            # See test_gpu_lock_delegation: a pack may override this wrapper without broker support,
            # and the marker route is a property of the wrapper rather than of the scheduler.
            print("    SKIP: this pack's gpu_lock.sh has no broker delegation, so the marker route "
                  "it would carry does not exist here")
            return
    with BrokerFixture("0,1") as b:
        base = tempfile.mkdtemp()
        # Why this exists: the broker has six priority bands and ages queued requests so nothing
        # starves, but in one measured fleet EVERY request arrived as the default `measure`, so
        # there was nothing to order by and it degraded to FIFO. The lane that was actually winning
        # (2.26x, 7 kept wins) got 7 grants and ~0 s of hold while two lanes still stuck in CENSUS
        # took 280 grants and 45 min of GPU. A lane that has measured something above the floor is
        # promoted to `verify` (30) so it stops queueing behind lanes that have not.
        evald = os.path.join(base, "exp", "team_gemm_20260815", "gemm")
        ws = os.path.join(evald, "ic_tech_lead", "workspace")
        os.makedirs(ws, exist_ok=True)
        with open(os.path.join(evald, ".geak_gpu_group"), "w") as fh:
            fh.write("gemm\n")
        with open(os.path.join(evald, ".geak_gpu_priority"), "w") as fh:
            fh.write("verify\n")

        env = dict(os.environ, GEAK_GPU_BROKER_SOCK=b.sock, GEAK_GPU_BROKER="1",
                   GEAK_GPU_BROKER_AUTOSTART="0", GEAK_GPU_LOCK_DIR=_LOCK_NS)
        env.pop("GEAK_GPU_GROUP", None)
        env.pop("GEAK_GPU_PRIORITY", None)
        r = subprocess.run(["bash", lock, "0,1", "true"],
                           capture_output=True, text=True, env=env, cwd=ws, timeout=120)
        ok(r.returncode == 0, f"the wrapper ran (rc={r.returncode}) {r.stderr[-200:]}")
        prios = [rr.get("priority") for rr in b.journal()
                 if rr.get("event") == "enqueue" and rr.get("group") == "gemm"]
        ok(prios and all(p == PRIO_VERIFY for p in prios),
           f"a promoted lane enqueues at `verify` ({PRIO_VERIFY}), not the default: {prios}")

        # A lane with no marker keeps the default. Promotion is opt-in on evidence, and a lane that
        # has not earned it must not be silently moved either way.
        ws2 = os.path.join(base, "exp", "team_moe_20260815", "moe", "ws")
        os.makedirs(ws2, exist_ok=True)
        subprocess.run(["bash", lock, "0,1", "true"],
                       capture_output=True, text=True, env=env, cwd=ws2, timeout=120)
        prios2 = [rr.get("priority") for rr in b.journal()
                  if rr.get("event") == "enqueue" and rr.get("group") == "moe"]
        ok(prios2 and all(p == PRIO_MEASURE for p in prios2),
           f"an unpromoted lane stays at the default `measure` ({PRIO_MEASURE}): {prios2}")

        # A typo must not refuse work. parse_priority() falls back rather than raising, because a
        # scheduler that rejects a run over a misspelt label has turned a tuning knob into an outage.
        evald3 = os.path.join(base, "exp", "team_attn_20260815", "attn")
        ws3 = os.path.join(evald3, "ws")
        os.makedirs(ws3, exist_ok=True)
        with open(os.path.join(evald3, ".geak_gpu_group"), "w") as fh:
            fh.write("attn\n")
        with open(os.path.join(evald3, ".geak_gpu_priority"), "w") as fh:
            fh.write("not-a-band\n")
        r3 = subprocess.run(["bash", lock, "0,1", "true"],
                            capture_output=True, text=True, env=env, cwd=ws3, timeout=120)
        ok(r3.returncode == 0, f"a garbage band still runs the command (rc={r3.returncode})")
        prios3 = [rr.get("priority") for rr in b.journal()
                  if rr.get("event") == "enqueue" and rr.get("group") == "attn"]
        ok(prios3 and all(p == PRIO_MEASURE for p in prios3),
           f"and lands on the default instead of refusing: {prios3}")


def test_heartbeat_survives_a_blocked_call_on_the_same_connection():
    section("a lease renews while its owner is blocked in a queued acquire")
    # THE BUG THIS PINS. `Conn.call` holds a per-connection mutex for the whole round-trip. When the
    # heartbeat shared that socket it could not fire while any other call was in flight -- and the
    # in-flight call is normally a BLOCKING acquire, because that is what residency does: park, then
    # wait for a measurement window. So the resident lost the lease covering VRAM it was still
    # holding, the broker handed its lane to somebody else, and two tenants computed on one card.
    #
    # Measured before the fix: 4 s TTL, 14 s queue, resident evicted. The renew now goes over a
    # dedicated socket, authorised by a per-lease token (a second socket is a different `owner`, and
    # owner-identity is what makes connection-close reclaim work, so it could not simply be relaxed).
    import gpu_client
    with BrokerFixture("0", extra=("--lease-ttl-s", "4")) as b:
        c = gpu_client.Conn(b.sock, label="holder")
        g = c.call({"op": "acquire", "kind": "resident", "vram_mb": 10,
                    "group": "g", "timeout_s": 10})
        ok(g.get("ok"), "the resident parked")
        ok(bool(g.get("token")), "the grant carries a renew token")
        lv = gpu_client.Lease(c, g["lease"], g["lane"], "resident", ttl_s=4.0,
                              group="g", sock_path=b.sock, token=g.get("token"))

        # Occupy the lane so the holder's own wake QUEUES, which is what blocks its socket.
        c2 = gpu_client.Conn(b.sock, label="squatter")
        c2.call({"op": "acquire", "kind": "exclusive", "group": "other", "timeout_s": 30})

        threading.Thread(
            target=lambda: c.call({"op": "acquire", "kind": "exclusive", "want_lane": 0,
                                   "lanes": [0], "group": "g", "timeout_s": 20}, timeout=25),
            daemon=True).start()

        time.sleep(13)          # >3x the TTL: only a live heartbeat can keep it alive
        res = b.status()["lanes"][0]["residents"]
        ok(bool(res), f"the resident is STILL parked after 3x its TTL (residents={res})")
        ok(lv.lost is False, "and the lease does not report itself lost")
        lv.release()
        c.close()
        c2.close()

    # The token must be narrow: renew only, and only with the right token.
    with BrokerFixture("0") as b2:
        c = gpu_client.Conn(b2.sock, label="owner")
        g = c.call({"op": "acquire", "kind": "exclusive", "group": "g", "timeout_s": 10})
        other = gpu_client.Conn(b2.sock, label="stranger")
        ok(other.call({"op": "renew", "lease": g["lease"], "token": g["token"]}).get("ok"),
           "the token authorises renew from another socket")
        ok(not other.call({"op": "renew", "lease": g["lease"]}).get("ok"),
           "without it, another socket still cannot renew")
        ok(not other.call({"op": "renew", "lease": g["lease"], "token": "wrong"}).get("ok"),
           "and a wrong token is refused")
        ok(not other.call({"op": "release", "lease": g["lease"]}).get("ok"),
           "the token confers no power to RELEASE somebody else's lease")
        c.close()
        other.close()


def test_a_reading_records_whether_it_was_arbitrated():
    section("every sweep reading says whether it held an exclusive window")
    # A reading taken without a window is not wrong, it is WEAKER: the lane was not arbitrated for
    # its duration, so another tenant may have been computing on the same card. By the time a config
    # is ranked nobody can reconstruct that, and a contaminated reading is indistinguishable from a
    # merely slow one -- so the fact has to travel on the reading itself.
    _pool = _sibling("sweep_pool.py")
    if _pool is None:
        # SKIPPED, said out loud. This case is the seam between sweep_pool and the broker, so it
        # only exists where both are present -- but a skip that prints nothing is a skip that reads
        # as a pass, which is the one thing a test must never do.
        print("    SKIP: no sweep_pool.py next to this scheduler -- the sweep/broker seam is not "
              "testable in this layout")
        return
    sys.path.insert(0, os.path.dirname(_pool))
    td = tempfile.mkdtemp()
    fake = os.path.join(td, "fw.py")
    with open(fake, "w") as fh:
        fh.write(
            "import json,sys\n"
            "print(json.dumps({'ready':True,'adapter':'f','owns_meter':True,'anchor':{}}),flush=True)\n"
            "for l in sys.stdin:\n"
            "    l=l.strip()\n"
            "    if not l: continue\n"
            "    r=json.loads(l)\n"
            "    if r.get('stop'): break\n"
            "    print(json.dumps({'id':r.get('id'),'status':'ok','ms':1.0,'err':0.0,\n"
            "                      'effective':{}}),flush=True)\n")

    with BrokerFixture("0") as b:
        env_sock = os.environ.get("GEAK_GPU_BROKER_SOCK")
        os.environ["GEAK_GPU_BROKER_SOCK"] = b.sock
        os.environ.pop("GEAK_SWEEP_NO_LEASE", None)
        try:
            import importlib
            import gpu_client
            importlib.reload(gpu_client)
            import sweep_pool
            importlib.reload(sweep_pool)
            p = sweep_pool.Pool(sweep_pool.driver_argv_builder([sys.executable, fake]),
                                [0], group="stamped")
            r = p.measure({"x": 1}, 50.0)
            ok(r.get("exclusive_window") is True,
               f"with a broker the reading is marked arbitrated: {r.get('exclusive_window')}")
            ok(any(rr.get("group") == "stamped"
                   for rr in b.journal() if rr.get("event") == "grant"),
               "and the broker really did grant a window for it")
            p.stop()

            # The stamp must be read INSIDE the window: Lease.release() clears `brokered`, so
            # stamping after the `with` exits reports every reading as unwindowed -- the opposite
            # of the truth, and as silent as the bug the stamp exists to expose.
            os.environ["GEAK_GPU_BROKER"] = "0"
            importlib.reload(gpu_client)
            importlib.reload(sweep_pool)
            p2 = sweep_pool.Pool(sweep_pool.driver_argv_builder([sys.executable, fake]),
                                 [0], group="stamped")
            r2 = p2.measure({"x": 1}, 50.0)
            ok(r2.get("exclusive_window") is False,
               "with no broker it is marked UNarbitrated rather than left blank")
            p2.stop()
        finally:
            os.environ["GEAK_GPU_BROKER"] = "1"          # back to this file's opt-in
            if env_sock is None:
                os.environ.pop("GEAK_GPU_BROKER_SOCK", None)
            else:
                os.environ["GEAK_GPU_BROKER_SOCK"] = env_sock


def test_ttl_reclaim_is_journalled():
    section("a TTL reclaim reaches the journal (the auditor counts it)")
    # A 1-second TTL and a client that connects, acquires, and then never renews. The heartbeat
    # lives in gpu_client.Lease, so a raw socket holder is exactly the wedged case.
    with BrokerFixture("0", extra=("--lease-ttl-s", "1")) as b:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(20)
        s.connect(b.sock)
        s.sendall((json.dumps({"op": "acquire", "kind": "exclusive", "label": "wedged",
                               "group": "g", "timeout_s": 20}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            buf += s.recv(65536)
        ok(json.loads(buf).get("ok"), "the wedged client holds a lease")

        # Keep the socket OPEN (so disconnect-reclaim cannot fire) and simply stop renewing.
        reclaimed = False
        for _ in range(80):
            time.sleep(0.25)
            if any(r.get("event") == "lease_expired" for r in b.journal()):
                reclaimed = True
                break
        ok(reclaimed, "the expired lease is journalled as `lease_expired`, not silently dropped")

        rec = [r for r in b.journal() if r.get("event") == "lease_expired"]
        ok(rec and rec[0].get("lane") == 0 and rec[0].get("held_s") is not None,
           f"and carries lane + held_s: {rec[:1]}")

        # The auditor must actually surface it -- the whole point of emitting it.
        sys.path.insert(0, SCHED)
        import audit_journal
        a = audit_journal.audit(b.journal())
        ok(a["lease_reclaims_ttl"] >= 1,
           f"audit_journal reports lease_reclaims_ttl={a['lease_reclaims_ttl']} (was always 0)")
        s.close()


def test_shutdown_reports_success():
    section("--shutdown succeeds AND says so")
    b = BrokerFixture("0")
    try:
        r = b.rpc({"op": "shutdown"})
        # It used to stop the daemon while replying `unknown op 'shutdown'` -- the op had no
        # handler, and the connection loop checked the name separately. Working while reporting
        # failure is worse than failing: every scripted teardown logged an error it was right to
        # ignore, which is how a real error later gets ignored too.
        ok(r.get("ok") is True, f"the reply reports success: {r}")
        ok(r.get("stopping") is True, "and says it is stopping")
        stopped = False
        for _ in range(60):
            if b.proc.poll() is not None:
                stopped = True
                break
            time.sleep(0.1)
        ok(stopped, "the daemon actually exits")
    finally:
        b.stop()


def test_auditor_scopes_to_one_daemon():
    section("the auditor does not mix two daemons' lease-id spaces")
    sys.path.insert(0, SCHED)
    import audit_journal

    # Two daemon lifetimes in one file, each numbering its leases from 1 -- which is what actually
    # happens, since the journal path is fixed and a restart appends. Replaying both together sees
    # "lease 1 granted while lease 1 is open" and reports a double-booking that never occurred.
    # Observed for real: a box with five daemon lifetimes in one file audited as 2 violations.
    recs = [
        {"seq": 1, "event": "broker_start", "pid": 100},
        {"seq": 2, "event": "grant", "kind": "exclusive", "lane": 0, "lease": 1, "group": "a"},
        # ...daemon 1 dies here WITHOUT releasing (kill -9), so its lease is never closed.
        {"seq": 3, "event": "broker_start", "pid": 200},
        {"seq": 4, "event": "grant", "kind": "exclusive", "lane": 0, "lease": 1, "group": "b"},
        {"seq": 5, "event": "release", "lane": 0, "lease": 1, "held_s": 1.0},
    ]
    a_all = audit_journal.audit(list(recs), epoch="all")
    ok(a_all["double_bookings"] == 1,
       "auditing ALL epochs sees the (phantom) overlap -- so the scoping is load-bearing")

    a_last = audit_journal.audit(list(recs))
    ok(a_last["double_bookings"] == 0,
       "auditing the LAST epoch (the default) is clean, because it is one id space")
    ok(a_last["epochs_in_file"] == 2, f"and it reports how many it found: {a_last}")
    ok(a_last["grants"] == 1, "only the current daemon's grants are counted")


def test_no_broker_degrades():
    section("with no daemon at all, callers still run (the whole fallback story)")
    import gpu_client
    with tempfile.TemporaryDirectory() as td:
        absent = os.path.join(td, "absent.sock")
        t0 = time.time()
        with gpu_client.lease(group="k", sock_path=absent, timeout_s=5) as g:
            ok(not g.brokered and g.lane is None, "an unbrokered lease is returned")
        ok(time.time() - t0 < 3, "and it returns immediately rather than waiting out the timeout")

        # The CLI path -- what gpu_lock.sh delegates to -- must run the command regardless.
        env = dict(os.environ, GEAK_GPU_BROKER_SOCK=absent, GEAK_GPU_BROKER_AUTOSTART="0", GEAK_GPU_LOCK_DIR=_LOCK_NS)
        r = subprocess.run(
            [sys.executable, os.path.join(SCHED, "gpu_client.py"), "run", "--group", "k",
             "--timeout-s", "5", "--", "echo", "ran-anyway"],
            capture_output=True, text=True, env=env, timeout=60)
        ok(r.returncode == 0 and "ran-anyway" in r.stdout,
           f"`gpu_client run` executes the command with no broker (rc={r.returncode})")

        # ...and --require-broker is the opt-in for callers that would rather fail.
        r2 = subprocess.run(
            [sys.executable, os.path.join(SCHED, "gpu_client.py"), "run", "--require-broker",
             "--timeout-s", "5", "--", "echo", "should-not-run"],
            capture_output=True, text=True, env=env, timeout=60)
        ok(r2.returncode != 0 and "should-not-run" not in r2.stdout,
           "--require-broker refuses instead of running unbrokered")



def test_lease_overrun_is_reported():
    """A hold long enough to block others is SAID SO while it is happening, not only at release.

    Measured over a real 4-GPU fleet: 0.58% of leases held 65.5% of every GPU-second, the longest
    running 1294 s, and those few holds produced every wait over a minute plus all 57 queue timeouts.
    None of it was visible in flight -- `held_s` is journalled at release, so a 20-minute hold is
    invisible for 20 minutes, which is exactly the window in which someone could have acted. TTL does
    not cover it either: the client heartbeat renews for as long as the holder lives, so `lease_expired`
    was 0 across that whole run while the 1294 s lease was in flight."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gb_ov", os.path.join(SCHED, "gpu_broker.py"))
    gb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gb)
    with tempfile.TemporaryDirectory() as td:
        def events(kind):
            return [json.loads(l) for l in open(os.path.join(td, "journal.ndjson"))
                    if json.loads(l).get("event") == kind]

        b = gb.Broker([0, 1], state_dir=td, probe_hardware=False, lease_overrun_s=1.0)
        r = b.op_acquire({"kind": "exclusive", "lanes": [0], "group": "g", "timeout_s": 5}, "own-A")
        ok(r.get("ok"), "an exclusive lease is granted")
        b.maintenance()
        ok(not events("lease_overrun"), "a fresh hold is not an overrun")

        time.sleep(1.2)
        b.maintenance()
        b.maintenance()          # the tick runs every 2 s for the life of the hold
        ev = events("lease_overrun")
        ok(len(ev) == 1, f"a hold past the floor is reported EXACTLY once, not per tick (got {len(ev)})")
        ok(ev[0].get("lane") == 0 and ev[0].get("group") == "g" and ev[0].get("held_s") >= 1.0,
           "and it names the lane, the group and how long it has been held")

        rel = b.op_release({"lease": r["lease"]}, "own-A")
        ok(rel.get("ok"), "the lease still releases normally")
        ok(events("release")[-1].get("overran") is True,
           "the release carries `overran`, so an audit that walks releases alone still sees it")

        # A RESIDENT is a parked process, not a card being held from anyone -- it must not be flagged,
        # or the signal drowns in the residency the split exists to encourage.
        b2 = gb.Broker([0], state_dir=td, probe_hardware=False, lease_overrun_s=1.0)
        before = len(events("lease_overrun"))
        b2.op_acquire({"kind": "resident", "lanes": [0], "group": "g", "vram_mb": 64,
                       "timeout_s": 5}, "own-R")
        time.sleep(1.2)
        b2.maintenance()
        ok(len(events("lease_overrun")) == before, "a RESIDENT lease is never an overrun")


def test_audit_reads_the_shape_before_advising():
    """`add cards or cut concurrency` is one of two opposite cures, so it may not be the only one.

    A queueing TOTAL cannot distinguish SPREAD (every acquisition pays: really oversubscribed) from
    CONCENTRATED (nearly all instant, a few holds block everyone). The line printed the oversubscribed
    advice for both -- on a run whose median wait was 0.00 s and where 87% of acquisitions were
    instant, it recommended shrinking a fleet whose actual problem was a handful of long holds."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("aj_shape", os.path.join(SCHED, "audit_journal.py"))
    aj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aj)

    def synth(path, waits, holds):
        recs = [{"seq": 0, "t": 0, "event": "broker_start", "pid": 1, "lanes": [0], "sock": "x"}]
        s = 1
        for i, (w, h) in enumerate(zip(waits, holds)):
            recs.append({"seq": s, "t": s, "event": "grant", "lease": i, "lane": 0,
                         "kind": "exclusive", "owner": f"o{i}", "group": "g", "waited_s": w}); s += 1
            recs.append({"seq": s, "t": s, "event": "release", "lease": i, "lane": 0,
                         "owner": f"o{i}", "held_s": h}); s += 1
        open(path, "w").write("\n".join(json.dumps(r) for r in recs))

    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "spread.ndjson")
        synth(p1, [9] * 40, [3] * 40)
        t1 = aj.render(aj.audit(aj.load(p1)))
        ok("SPREAD" in t1 and "Add cards or cut concurrency" in t1,
           "a run where the MEDIAN acquisition waits is called oversubscribed")

        p2 = os.path.join(td, "conc.ndjson")
        synth(p2, [0] * 38 + [900, 900], [0.5] * 38 + [900, 0.5])
        t2 = aj.render(aj.audit(aj.load(p2)))
        ok("CONCENTRATED" in t2, "a run where one hold blocks everyone is NOT called oversubscribed")
        ok("Add cards" not in t2.split("CONCENTRATED")[1].split("longest")[0],
           "and it does not recommend the cure for the opposite problem")
        ok("longest EXCLUSIVE holds" in t2 and "900.0s" in t2,
           "it NAMES the blocking hold, which is the actionable half")


def test_gpu_lock_delegation():
    section("gpu_lock.sh delegates to the broker and still sets up the environment")
    lock = _sibling("gpu_lock.sh")
    if lock is None:
        ok(False, "gpu_lock.sh exists next to this scheduler")
        return
    # WHAT THIS PACK'S WRAPPER ACTUALLY CLAIMS TO DO. A DSL may override `gpu_lock.sh`, and one in
    # this tree ships a copy with no broker delegation and no `GEAK_GPU_GROUP` at all -- so running
    # the vendor assertions against it reports a deliberate override as a broker defect. Read the
    # file rather than assume, and say out loud which assertions do not apply: a silently-dropped
    # assertion is exactly the failure mode this subsystem's tests exist to prevent.
    with open(lock) as fh:
        _wrapper = fh.read()
    if "broker" not in _wrapper:
        print("    SKIP: this pack overrides gpu_lock.sh with a copy that has no broker delegation "
              "(no `broker`, no GEAK_GPU_GROUP) -- it cannot participate in the pool through this "
              "wrapper, which is a property of that pack and not of the scheduler")
        return
    _sets_torch = "TORCH_EXTENSIONS_DIR" in _wrapper
    with BrokerFixture("0,1") as b:
        env = dict(os.environ, GEAK_GPU_BROKER_SOCK=b.sock, GEAK_GPU_BROKER="1",
                   GEAK_GPU_BROKER_AUTOSTART="0", GEAK_GPU_GROUP="itest",
                   GEAK_GPU_LOCK_DIR=_LOCK_NS)
        r = subprocess.run(
            ["bash", lock, "0,1", "bash", "-c",
             'echo "LANE=$GEAK_GPU_LANE HIP=$HIP_VISIBLE_DEVICES TORCH=${TORCH_EXTENSIONS_DIR:+set}"'],
            capture_output=True, text=True, env=env, timeout=120, cwd=tempfile.mkdtemp())
        out = r.stdout.strip()
        ok(r.returncode == 0, f"the wrapper succeeded (rc={r.returncode}) {r.stderr[-300:]}")
        ok("LANE=" in out and "HIP=" in out, f"a lane was granted and exported: {out}")
        if _sets_torch:
            ok("TORCH=set" in out,
               "TORCH_EXTENSIONS_DIR is STILL set -- delegation did not skip the env setup")
        else:
            print("    n/a: this pack's gpu_lock.sh does not set TORCH_EXTENSIONS_DIR, so "
                  "'delegation preserves it' is not a claim about this copy")
        ok(any(rr.get("group") == "itest" for rr in b.journal() if rr.get("event") == "grant"),
           "the acquisition is attributed to GEAK_GPU_GROUP")

        # And with the broker switched off, the legacy path must still work.
        #
        # GEAK_GPU_REQUIRE_IDLE=0 because this asserts a CODE PATH, not the state of this box: the
        # legacy path's idleness check reads real sysfs, and on a shared machine lane 0 may well be
        # holding somebody's model -- in which case it correctly refuses, and the test would fail
        # for a reason that has nothing to do with the delegation being tested. (Observed: GPU 0
        # here sits at 231 GB VRAM, so the unguarded form failed exactly this way.)
        env2 = dict(env, GEAK_GPU_BROKER="0", GEAK_GPU_REQUIRE_IDLE="0")
        r2 = subprocess.run(["bash", lock, "0", "bash", "-c", 'echo "HIP=$HIP_VISIBLE_DEVICES"'],
                            capture_output=True, text=True, env=env2, timeout=120,
                            cwd=tempfile.mkdtemp())
        ok(r2.returncode == 0 and "HIP=0" in r2.stdout,
           f"GEAK_GPU_BROKER=0 falls back to flock (rc={r2.returncode}) {r2.stderr[-200:]}")

        # OFF BY DEFAULT: with the variable UNSET, a live broker is not consulted at all.
        grants_before = sum(1 for rr in b.journal() if rr.get("event") == "grant")
        env3 = dict(env, GEAK_GPU_REQUIRE_IDLE="0", GEAK_GPU_GROUP="must_not_reach_broker")
        env3.pop("GEAK_GPU_BROKER", None)
        r3 = subprocess.run(["bash", lock, "0", "bash", "-c", 'echo "HIP=$HIP_VISIBLE_DEVICES"'],
                            capture_output=True, text=True, env=env3, timeout=120,
                            cwd=tempfile.mkdtemp())
        grants_after = sum(1 for rr in b.journal() if rr.get("event") == "grant")
        ok(r3.returncode == 0 and "HIP=0" in r3.stdout and grants_after == grants_before,
           f"GEAK_GPU_BROKER unset -> flock only, the broker is not consulted "
           f"(rc={r3.returncode}, grants {grants_before}->{grants_after})")


def main():
    print("=" * 72)
    print("scheduler — integration tests (real daemon, real sockets, no GPU)")
    print("=" * 72)
    for fn in [
        test_daemon_comes_up,
        test_killed_client_releases_its_lane,
        test_no_double_booking_under_concurrency,
        test_garbage_on_the_wire,
        test_priority_is_honoured_end_to_end,
        test_queued_disconnect_cancels_immediately,
        test_cooperative_pause_never_forces_a_running_process,
        test_status_json_exposes_pool_state_and_metadata,
        test_resident_park_and_wake_are_journalled_as_separate_states,
        test_drain_takes_a_lane_out_of_service,
        test_journal_records_the_scheduler_cost,
        test_fair_share_key_reaches_the_broker,
        test_priority_marker_reaches_the_broker,
        test_heartbeat_survives_a_blocked_call_on_the_same_connection,
        test_a_reading_records_whether_it_was_arbitrated,
        test_ttl_reclaim_is_journalled,
        test_shutdown_reports_success,
        test_auditor_scopes_to_one_daemon,
        test_no_broker_degrades,
        test_gpu_lock_delegation,
        test_lease_overrun_is_reported,
        test_audit_reads_the_shape_before_advising,
    ]:
        try:
            fn()
        except Exception as exc:                                 # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILS.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} check(s) failed")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("SELFTEST PASS: the broker arbitrates, recovers, and degrades")
    return 0


if __name__ == "__main__":
    sys.exit(main())
